#!/usr/bin/env python3
"""
Live in-game MLB trader for Polymarket.

Monitors live MLB games, runs Monte Carlo simulations at each half-inning
state change, compares model win probabilities to Polymarket prices, and
places directional bets when edges exceed thresholds.

Sizing is phase-aware: the model's edge over the market concentrates in
late innings (7th-9th), so Kelly fractions are scaled by a confidence
multiplier that increases with game progression.

Backtest evidence (2025 season, 100 games, 1751 state points):
  - Innings 1-6: model log loss worse than market → skip
  - Inning 7:    model ≈ market (ΔLL = +0.0006)  → small size
  - Inning 8:    model ≈ market (ΔLL = +0.0059)  → medium size
  - Inning 9+:   model BEATS market (ΔLL = -0.016) → full size

Requires:
  - POLYMARKET_PRIVATE_KEY env var
  - POLYMARKET_API_KEY, POLYMARKET_API_SECRET, POLYMARKET_API_PASSPHRASE
  - py-clob-client package
  - Simulation artifacts (run `make build-sim-data` first)
  - xRV data + matchup models (run full pipeline first)

Usage:
    # Dry run (no real orders)
    python src/live_trader.py --bankroll 5000 --dry-run

    # Live trading
    python src/live_trader.py --bankroll 5000

    # Custom settings
    python src/live_trader.py --bankroll 10000 --min-edge 0.05 --poll 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger("live_trader")

ET = ZoneInfo("America/New_York")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate import (
    GameState as SimGameState,
    SimConfig,
    TeamContext,
    load_sim_artifacts,
    load_simulation_context,
    monte_carlo_win_prob,
    fetch_live_game_state,
)
from backtest_vs_kalshi import load_backtest_data
from predict import fetch_lineup
from polymarket_bot import (
    PolyMarket,
    fetch_poly_mlb_markets,
    fetch_orderbook,
    fetch_clob_midpoint,
    compute_half_kelly,
    round_to_tick,
)
from scrape_polymarket import resolve_team_abbr

# ── Constants ─────────────────────────────────────────────────────────────────

MLB_API = "https://statsapi.mlb.com/api/v1"
CLOB_API = "https://clob.polymarket.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_DIR = DATA_DIR / "state"
AUDIT_DIR = DATA_DIR / "audit" / "live"

TICK_SIZE = 0.01

# Phase confidence multipliers (from backtest analysis)
# Model's edge concentrates in late innings; early innings are noise.
# Backtest: 25% Kelly, 3% min edge → +3.6% ROI, Sharpe 1.12 after fees
PHASE_CONFIDENCE = {
    1: 0.00,   # skip — model worse than market
    2: 0.00,
    3: 0.00,
    4: 0.00,
    5: 0.00,
    6: 0.15,   # model starts converging
    7: 0.40,   # model ≈ market
    8: 0.65,   # model close to market
    9: 1.00,   # model beats market — full size
}
EXTRAS_CONFIDENCE = 1.00

# Polymarket sports taker fee: C × 0.03 × p × (1-p)
# Symmetric around 50%, near-zero at extremes (where our late-game trades are)
TAKER_FEE_RATE = 0.03

MAX_RETRIES = 3
RETRY_BACKOFF = [1.0, 2.0, 4.0]


async def _retry_async(func, *args, label: str = "", retries: int = MAX_RETRIES, **kwargs):
    """Run a sync function via asyncio.to_thread with retry + exponential backoff."""
    for attempt in range(retries):
        try:
            if kwargs:
                return await asyncio.to_thread(lambda: func(*args, **kwargs))
            return await asyncio.to_thread(func, *args)
        except Exception as e:
            if attempt < retries - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                log.warning("%s attempt %d failed (%s), retrying in %.0fs",
                            label, attempt + 1, e, wait)
                await asyncio.sleep(wait)
            else:
                log.error("%s failed after %d attempts: %s", label, retries, e)
                raise


def _polymarket_fee(price: float) -> float:
    """Per-contract Polymarket taker fee at a given price level."""
    return TAKER_FEE_RATE * price * (1 - price)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TraderConfig:
    bankroll: float = 5000.0
    dry_run: bool = True
    poll_interval_s: float = 30.0
    n_sims: int = 1000
    kelly_fraction: float = 0.25     # quarter-Kelly (best risk-adjusted in backtest)
    min_edge: float = 0.03           # 3% min edge (backtest: +3.6% ROI after fees)
    max_position_pct: float = 0.10   # 10% of bankroll per game
    max_total_exposure_pct: float = 0.25  # 25% total across all games
    max_loss: float = 500.0          # kill switch
    season: int = 2025
    target_date: str = ""


# ── Game Tracking ─────────────────────────────────────────────────────────────

@dataclass
class LiveGame:
    """Tracks a single live game with model context and position."""
    game_pk: int
    home_team: str
    away_team: str
    game_time_utc: str = ""

    # Simulation context (built once at game discovery)
    home_ctx: TeamContext | None = None
    away_ctx: TeamContext | None = None
    context_ready: bool = False

    # Current game state from MLB API
    sim_state: SimGameState | None = None
    last_half_inning: str = ""        # e.g. "Top 5"
    game_status: str = "Preview"      # Preview, Live, Final

    # Model output (updated each half-inning change)
    model_home_wp: float | None = None
    model_updated_at: float = 0.0     # unix timestamp

    # Polymarket data
    poly_market: PolyMarket | None = None
    poly_home_mid: float | None = None
    poly_home_bid: float | None = None
    poly_home_ask: float | None = None
    poly_spread: float | None = None

    # Edge + sizing
    edge: float = 0.0                 # model - market
    phase_confidence: float = 0.0
    half_kelly: float = 0.0
    side: str = ""                    # "BUY_HOME" or "BUY_AWAY"
    target_size_usd: float = 0.0

    # Position tracking
    net_position: float = 0.0        # +ve = long home, -ve = long away
    cost_basis: float = 0.0
    realized_pnl: float = 0.0
    active_order_id: str | None = None
    active_order_side: str = ""
    active_order_price: float = 0.0
    active_order_size: float = 0.0
    fill_history: list = field(default_factory=list)

    # Risk flags
    halted: bool = False
    halt_reason: str = ""

    @property
    def game_key(self) -> str:
        return f"{self.away_team}@{self.home_team}"

    @property
    def inning(self) -> int:
        return self.sim_state.inning if self.sim_state else 0

    @property
    def inning_label(self) -> str:
        if not self.sim_state:
            return "Pre"
        return f"{self.sim_state.top_bottom} {self.sim_state.inning}"

    @property
    def score_str(self) -> str:
        if not self.sim_state:
            return "0-0"
        return f"{self.sim_state.away_score}-{self.sim_state.home_score}"

    @property
    def unrealized_pnl(self) -> float:
        if self.net_position == 0 or self.poly_home_mid is None:
            return 0.0
        if self.net_position > 0:
            # Long home: value = position * current_price
            return self.net_position * self.poly_home_mid - self.cost_basis
        else:
            # Long away: value = |position| * (1 - current_price)
            return abs(self.net_position) * (1 - self.poly_home_mid) - self.cost_basis

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def target_position(self) -> float:
        """Target position in contracts (signed: +ve=home, -ve=away).

        This is the Kelly-optimal position at the current state.
        The rebalancing logic trades toward this target.
        """
        if self.target_size_usd <= 0 or not self.poly_home_mid:
            return 0.0
        if self.side == "BUY_HOME":
            price = self.poly_home_ask or self.poly_home_mid
            return self.target_size_usd / price if price > 0 else 0.0
        elif self.side == "BUY_AWAY":
            price = (1 - self.poly_home_bid) if self.poly_home_bid else (1 - self.poly_home_mid)
            return -(self.target_size_usd / price) if price > 0 else 0.0
        return 0.0


# ── Main Trader ───────────────────────────────────────────────────────────────

class LiveTrader:
    def __init__(self, config: TraderConfig):
        self.config = config
        self.games: dict[str, LiveGame] = {}
        self._running = False
        self._killed = False
        self._cycle_count = 0
        self._clob_client = None

        # Simulation artifacts (loaded once)
        self._base_rates = None
        self._transition_matrix = None
        self._idx = None
        self._matchup_models = None
        # No fixed seed — each sim must be independent for live trading
        self._sim_config = SimConfig(n_sims=config.n_sims, random_seed=None)

        # Per-thread HTTP clients to avoid sharing a single client across threads
        self._http = httpx.Client(timeout=30.0)

    # ── Initialization ────────────────────────────────────────────────────

    async def initialize(self):
        """Load models, discover games, build contexts."""
        print(f"\n{'='*70}")
        print(f"  MLB Live In-Game Trader")
        print(f"  Date: {self.config.target_date}")
        print(f"  Mode: {'DRY RUN' if self.config.dry_run else 'LIVE TRADING'}")
        print(f"  Bankroll: ${self.config.bankroll:,.0f}")
        print(f"  Kelly fraction: {self.config.kelly_fraction:.0%}")
        print(f"  Min edge: {self.config.min_edge:.0%}")
        print(f"  Taker fee: {TAKER_FEE_RATE:.0%} (sports)")
        print(f"  Max per game: ${self.config.bankroll * self.config.max_position_pct:,.0f}")
        print(f"  Max total: ${self.config.bankroll * self.config.max_total_exposure_pct:,.0f}")
        print(f"  Kill switch: -${self.config.max_loss:,.0f}")
        print(f"  Sims per state: {self.config.n_sims}")
        print(f"{'='*70}")

        # Step 1: Load simulation artifacts
        print("\n[1/5] Loading simulation artifacts...")
        self._base_rates, self._transition_matrix = load_sim_artifacts()
        print("  OK")

        # Step 2: Load xRV data and matchup models
        print("[2/5] Loading xRV and matchup models...")
        shared = load_backtest_data(self.config.season)
        if shared is None:
            print("  FATAL: Cannot load xRV data")
            return False
        self._idx, self._matchup_models, self._lineups = shared
        print("  OK")

        # Step 3: Discover today's MLB games
        print(f"[3/5] Discovering MLB games for {self.config.target_date}...")
        await self._discover_games()
        if not self.games:
            print("  No live or upcoming games found")
            return False
        print(f"  Found {len(self.games)} games")

        # Step 4: Discover Polymarket markets
        print("[4/5] Discovering Polymarket markets...")
        await self._discover_markets()

        # Step 5: Initialize trading client
        print("[5/5] Initializing trading client...")
        if not self.config.dry_run:
            try:
                from py_clob_client.client import ClobClient
                from py_clob_client.clob_types import ApiCreds

                creds = ApiCreds(
                    api_key=os.environ["POLYMARKET_API_KEY"],
                    api_secret=os.environ["POLYMARKET_API_SECRET"],
                    api_passphrase=os.environ["POLYMARKET_API_PASSPHRASE"],
                )
                self._clob_client = ClobClient(
                    host="https://clob.polymarket.com",
                    key=os.environ["POLYMARKET_PRIVATE_KEY"],
                    chain_id=137,
                    creds=creds,
                )
                print("  Connected to Polymarket CLOB (live trading)")
            except Exception as e:
                print(f"  WARNING: Could not init CLOB client: {e}")
                print("  Falling back to dry-run mode")
                self.config.dry_run = True
                self._clob_client = None
        else:
            print("  Dry run mode — no orders will be placed")

        # Load saved state for crash recovery
        await self._load_state()

        return True

    async def _discover_games(self):
        """Fetch today's MLB schedule and create LiveGame entries."""
        try:
            resp = self._http.get(
                f"{MLB_API}/schedule",
                params={
                    "sportId": 1,
                    "date": self.config.target_date,
                    "hydrate": "probablePitcher,team,linescore",
                    "gameType": "R",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  MLB API error: {e}")
            return

        for d in data.get("dates", []):
            for g in d.get("games", []):
                status = g.get("status", {}).get("abstractGameState", "")
                # Include Preview (upcoming), Live (in progress), but not Final
                if status == "Final":
                    continue

                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                ht = home.get("team", {}).get("abbreviation", "")
                at = away.get("team", {}).get("abbreviation", "")
                game_pk = g.get("gamePk")

                if not ht or not at or not game_pk:
                    continue

                game = LiveGame(
                    game_pk=game_pk,
                    home_team=ht,
                    away_team=at,
                    game_time_utc=g.get("gameDate", ""),
                    game_status=status,
                )
                self.games[game.game_key] = game
                print(f"    {game.game_key} (pk={game_pk}) — {status}")

    async def _discover_markets(self):
        """Match Polymarket markets to MLB games."""
        poly_markets = await asyncio.to_thread(
            fetch_poly_mlb_markets, self.config.target_date
        )
        print(f"  Found {len(poly_markets)} Polymarket moneyline markets")

        matched = 0
        for pm in poly_markets:
            # Try both orderings (Poly may have teams in either order)
            for ht, at in [
                (pm.home_team, pm.away_team),
                (pm.away_team, pm.home_team),
            ]:
                key = f"{at}@{ht}"
                if key in self.games:
                    self.games[key].poly_market = pm
                    matched += 1
                    print(f"    Matched: {key} → {pm.question[:50]}")
                    break

        print(f"  Matched {matched}/{len(self.games)} games")

    # ── Build Simulation Context ──────────────────────────────────────────

    async def _build_context(self, game: LiveGame):
        """Build MC simulation context for a game (lineup + xRV matchups)."""
        if game.context_ready:
            return

        # Get lineup
        lu = self._lineups.get(game.game_pk)
        if lu is None:
            try:
                lu = await asyncio.to_thread(
                    fetch_lineup, self._http, game.game_pk
                )
            except Exception:
                lu = None

        if not lu or not lu.get("home") or not lu.get("away"):
            return

        # Get SP IDs from live feed
        try:
            resp = self._http.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{game.game_pk}/feed/live"
            )
            resp.raise_for_status()
            feed = resp.json()
            bd = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
            home_sp = bd.get("home", {}).get("pitchers", [0])[0]
            away_sp = bd.get("away", {}).get("pitchers", [0])[0]
        except Exception:
            home_sp = 0
            away_sp = 0

        game_info = {
            "game_pk": game.game_pk,
            "game_date": self.config.target_date,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "home_sp_id": home_sp or 0,
            "away_sp_id": away_sp or 0,
        }

        try:
            home_ctx, away_ctx = await asyncio.to_thread(
                load_simulation_context,
                game_info,
                self.config.target_date,
                lu,
                self._idx,
                self._matchup_models or {},
                self._base_rates,
                None,  # mo_models — V1 only
                self._sim_config,
            )
            game.home_ctx = home_ctx
            game.away_ctx = away_ctx
            game.context_ready = True
        except Exception as e:
            log.error("%s: context build failed: %s", game.game_key, e)

    # ── Core Loop ─────────────────────────────────────────────────────────

    async def run(self):
        """Main trading loop."""
        self._running = True

        # Handle graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        print(f"\n  Starting main loop (poll every {self.config.poll_interval_s}s)...")
        print(f"  Press Ctrl+C to stop\n")

        while self._running:
            try:
                self._cycle_count += 1
                await self._cycle()
                await asyncio.sleep(self.config.poll_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Error in cycle %d: %s", self._cycle_count, e)
                await asyncio.sleep(5)

        # Cleanup
        await self._cleanup()

    async def _cycle(self):
        """One iteration of the main loop."""

        # Kill switch check
        if self._check_kill_switch():
            return

        # Update game states from MLB API
        await self._update_game_states()

        # For each live game: run sim, check edge, trade
        for key, game in self.games.items():
            if game.game_status == "Final":
                await self._handle_game_final(game)
                continue

            if game.game_status != "Live":
                continue

            if game.halted:
                continue

            # Build context if needed (first time game goes live)
            if not game.context_ready:
                await self._build_context(game)
                if not game.context_ready:
                    continue

            # Always update market price (needed for P&L display + kill switch)
            await self._update_market_price(game)

            # Check for half-inning change — only re-simulate + trade on change
            current_hi = game.inning_label
            if current_hi == game.last_half_inning:
                continue  # no state change — wait for next half-inning

            game.last_half_inning = current_hi

            # Run MC simulation
            await self._run_simulation(game)

            # Compute edge and sizing
            self._compute_edge(game)

            # Rebalance position to Kelly-optimal target
            await self._maybe_trade(game)

        # Check fills on active orders
        await self._check_fills()

        # Display dashboard
        self._print_dashboard()

        # Save state
        self._save_state()

    # ── State Updates ─────────────────────────────────────────────────────

    async def _update_game_states(self):
        """Poll MLB API for current game states (with retry)."""
        for key, game in list(self.games.items()):
            if game.game_status == "Final":
                continue

            try:
                resp = await _retry_async(
                    self._http.get,
                    f"{MLB_API}/schedule",
                    label=f"{key} schedule",
                    params={"sportId": 1, "gamePk": game.game_pk, "hydrate": "linescore"},
                )
                resp.raise_for_status()
                data = resp.json()
                for d in data.get("dates", []):
                    for g in d.get("games", []):
                        status = g.get("status", {}).get("abstractGameState", "")
                        game.game_status = status
            except Exception as e:
                log.warning("%s: schedule poll failed: %s", key, e)

            if game.game_status == "Live":
                try:
                    state = await _retry_async(
                        fetch_live_game_state, self._http, game.game_pk,
                        label=f"{key} live state",
                    )
                    game.sim_state = state
                except Exception as e:
                    log.warning("%s: live state fetch failed: %s", key, e)

    async def _run_simulation(self, game: LiveGame):
        """Run MC simulation at current game state."""
        if not game.sim_state or not game.home_ctx or not game.away_ctx:
            return

        try:
            result = await asyncio.to_thread(
                monte_carlo_win_prob,
                game.home_ctx,
                game.away_ctx,
                game.sim_state,
                self._transition_matrix,
                self._sim_config,
            )
            game.model_home_wp = result["home_wp"]
            game.model_updated_at = time.time()

            score = game.score_str
            print(f"    {game.game_key} {game.inning_label} "
                  f"({score}): model={game.model_home_wp:.1%}")
        except Exception as e:
            log.error("%s: sim failed: %s", game.game_key, e)

    async def _update_market_price(self, game: LiveGame):
        """Fetch current Polymarket price for a game (with retry)."""
        if not game.poly_market:
            return

        try:
            book = await _retry_async(
                fetch_orderbook, self._http, game.poly_market.home_token_id,
                label=f"{game.game_key} orderbook",
            )
            game.poly_home_bid = book.get("best_bid")
            game.poly_home_ask = book.get("best_ask")
            game.poly_home_mid = book.get("midpoint")
            game.poly_spread = book.get("spread")
        except Exception as e:
            log.warning("%s: market price fetch failed: %s (stale price kept)", game.game_key, e)

    # ── Edge Computation + Phase-Aware Sizing ─────────────────────────────

    def _compute_edge(self, game: LiveGame):
        """Compute fee-adjusted edge, phase confidence, and target position size.

        The edge is reduced by the expected Polymarket taker fee before
        comparing against the min_edge threshold and computing Kelly size.
        """
        if game.model_home_wp is None or game.poly_home_mid is None:
            game.edge = 0.0
            game.phase_confidence = 0.0
            game.half_kelly = 0.0
            game.side = ""
            game.target_size_usd = 0.0
            return

        p = game.poly_home_mid
        raw_edge = game.model_home_wp - p

        # Subtract fee drag from edge (fee is symmetric around p)
        fee_drag = _polymarket_fee(p)
        game.edge = raw_edge - fee_drag if raw_edge > 0 else raw_edge + fee_drag

        # Phase confidence based on inning
        inning = game.inning
        if inning >= 10:
            game.phase_confidence = EXTRAS_CONFIDENCE
        else:
            game.phase_confidence = PHASE_CONFIDENCE.get(inning, 0.0)

        # Full Kelly fraction and side (using fee-adjusted edge)
        edge = game.edge
        if abs(edge) < 1e-6 or p <= 0 or p >= 1:
            game.half_kelly = 0.0
            game.side = ""
            game.target_size_usd = 0.0
            return

        if edge > 0:
            full_kelly = edge / (1 - p)
            game.side = "BUY_HOME"
        else:
            full_kelly = abs(edge) / p
            game.side = "BUY_AWAY"

        # Apply Kelly fraction and phase confidence
        scaled_kelly = full_kelly * self.config.kelly_fraction * game.phase_confidence
        game.half_kelly = full_kelly * self.config.kelly_fraction

        # Cap at max position per game
        max_per_game = self.config.bankroll * self.config.max_position_pct
        kelly_size = self.config.bankroll * scaled_kelly
        game.target_size_usd = min(kelly_size, max_per_game)

    # ── Trading ───────────────────────────────────────────────────────────

    async def _maybe_trade(self, game: LiveGame):
        """Rebalance position to Kelly-optimal target at each half-inning.

        This is the core of the strategy: at every half-inning boundary,
        compute the target position from Kelly sizing and trade toward it.
        This includes:
          - Opening new positions when edge appears
          - Adding to positions when target grows
          - Partially unwinding when target shrinks
          - Fully closing when edge drops below threshold
          - Reversing direction when model flips
        """
        # Gate 1: must have market
        if not game.poly_market:
            return

        # Gate 2: phase confidence must be positive and edge must exceed
        # threshold for us to HOLD a position. If not, unwind any existing.
        abs_edge = abs(game.edge)
        has_edge = (abs_edge >= self.config.min_edge and game.phase_confidence > 0)

        # Compute target position (signed contracts: +ve=home, -ve=away)
        if has_edge:
            target = game.target_position
        else:
            target = 0.0  # unwind everything

        # Current position
        current = game.net_position  # +ve=home, -ve=away

        # Delta to trade
        delta = target - current

        # If delta is tiny, skip
        if abs(delta) < 1:
            return

        # Check total exposure limit (only for increasing exposure)
        total_exposure = sum(abs(g.cost_basis) for g in self.games.values())
        max_total = self.config.bankroll * self.config.max_total_exposure_pct
        increasing_exposure = abs(target) > abs(current)

        if increasing_exposure and total_exposure >= max_total:
            # Can still unwind, just can't increase
            if delta > 0 and current >= 0:
                return
            if delta < 0 and current <= 0:
                return
            # If we're reducing position (going toward zero), allow it
            delta = -current  # just flatten
            if abs(delta) < 1:
                return

        # Cancel existing order before placing new one
        await self._cancel_order(game)

        # Determine order parameters based on delta direction
        if delta > 0:
            # Need to get more long-home (buy home or sell away)
            if current >= 0:
                # Buying more home tokens
                price = game.poly_home_ask or game.poly_home_mid
                if not price:
                    return
                order_price = round_to_tick(price)
                token_id = game.poly_market.home_token_id
                clob_side = "BUY"
                n_contracts = int(abs(delta))
                trade_side = "BUY_HOME"
            else:
                # Unwinding away position: sell away tokens
                away_price = (1 - game.poly_home_ask) if game.poly_home_ask else (1 - game.poly_home_mid)
                if not away_price or away_price <= 0:
                    return
                order_price = round_to_tick(away_price)
                token_id = game.poly_market.away_token_id
                clob_side = "SELL"
                # Don't sell more than we have
                n_contracts = int(min(abs(delta), abs(current)))
                trade_side = "SELL_AWAY"
        else:
            # Need to get more long-away (buy away or sell home)
            if current <= 0:
                # Buying more away tokens
                bid = game.poly_home_bid or game.poly_home_mid
                if not bid:
                    return
                price = 1 - bid
                order_price = round_to_tick(price)
                token_id = game.poly_market.away_token_id
                clob_side = "BUY"
                n_contracts = int(abs(delta))
                trade_side = "BUY_AWAY"
            else:
                # Unwinding home position: sell home tokens
                price = game.poly_home_bid or game.poly_home_mid
                if not price:
                    return
                order_price = round_to_tick(price)
                token_id = game.poly_market.home_token_id
                clob_side = "SELL"
                # Don't sell more than we have
                n_contracts = int(min(abs(delta), abs(current)))
                trade_side = "SELL_HOME"

        if n_contracts < 1:
            return

        # Place order
        result = await self._place_order(
            token_id=token_id,
            side=clob_side,
            price=order_price,
            size=float(n_contracts),
            game=game,
            neg_risk=game.poly_market.neg_risk,
        )

        if result:
            order_id = result.get("orderID") or result.get("id", "")
            game.active_order_id = order_id
            game.active_order_side = trade_side
            game.active_order_price = order_price
            game.active_order_size = float(n_contracts)

            self._log_trade(game, "ORDER", n_contracts, order_price)

    async def _place_order(
        self, token_id: str, side: str, price: float, size: float,
        game: LiveGame, neg_risk: bool = False,
    ) -> dict | None:
        """Place order via py-clob-client or log in dry-run."""
        label = f"{game.game_key} {game.inning_label}"

        if self.config.dry_run:
            print(f"    [DRY RUN] {label}: {side} {size:.0f} @ {price:.2f} "
                  f"({game.side}, conf={game.phase_confidence:.0%})")
            return {"orderID": f"dry_{game.game_key}_{time.time():.0f}"}

        if not self._clob_client:
            return None

        try:
            from py_clob_client.clob_types import (
                OrderArgs, PartialCreateOrderOptions,
            )

            order_args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=token_id,
            )
            options = PartialCreateOrderOptions(
                tick_size="0.01",
                neg_risk=neg_risk if neg_risk else None,
            )
            result = await asyncio.to_thread(
                self._clob_client.create_and_post_order,
                order_args,
                options,
            )
            order_id = result.get("orderID") or result.get("id", "")
            print(f"    ORDER: {label}: {side} {size:.0f} @ {price:.2f} "
                  f"({game.side}) -> {order_id}")
            return result
        except Exception as e:
            log.error("ORDER FAILED %s: %s", label, e)
            return None

    async def _cancel_order(self, game: LiveGame):
        """Cancel active order for a game."""
        if not game.active_order_id:
            return

        if (not game.active_order_id.startswith("dry_")
                and not self.config.dry_run
                and self._clob_client):
            try:
                await asyncio.to_thread(
                    self._clob_client.cancel, game.active_order_id
                )
            except Exception as e:
                log.error("%s: cancel order %s failed: %s",
                          game.game_key, game.active_order_id, e)

        game.active_order_id = None
        game.active_order_side = ""
        game.active_order_price = 0.0
        game.active_order_size = 0.0

    def _apply_fill(self, game: LiveGame, side: str, fill_size: float,
                     fill_price: float):
        """Apply a fill to position tracking with correct cost basis logic.

        Buy: increases position, adds to cost basis.
        Sell: decreases position, reduces cost basis proportionally, realizes P&L.
        """
        fee = abs(fill_size) * TAKER_FEE_RATE * fill_price * (1 - fill_price)

        if side in ("BUY_HOME", "BUY"):
            game.net_position += fill_size
            game.cost_basis += fill_size * fill_price + fee
        elif side in ("BUY_AWAY",):
            game.net_position -= fill_size
            game.cost_basis += fill_size * fill_price + fee
        elif side in ("SELL_HOME",):
            # Selling home tokens: reduces long-home position
            proceeds = fill_size * fill_price - fee
            if abs(game.net_position) > 0:
                # Proportional cost basis reduction
                avg_cost_per_unit = game.cost_basis / abs(game.net_position)
                cost_removed = fill_size * avg_cost_per_unit
                game.realized_pnl += proceeds - cost_removed
                game.cost_basis = max(0.0, game.cost_basis - cost_removed)
            else:
                game.realized_pnl += proceeds
            game.net_position -= fill_size
        elif side in ("SELL_AWAY",):
            # Selling away tokens: reduces long-away position
            proceeds = fill_size * fill_price - fee
            if abs(game.net_position) > 0:
                avg_cost_per_unit = game.cost_basis / abs(game.net_position)
                cost_removed = fill_size * avg_cost_per_unit
                game.realized_pnl += proceeds - cost_removed
                game.cost_basis = max(0.0, game.cost_basis - cost_removed)
            else:
                game.realized_pnl += proceeds
            game.net_position += fill_size

        game.fill_history.append({
            "order_id": game.active_order_id,
            "side": side,
            "size": fill_size,
            "price": fill_price,
            "fee": fee,
            "ts": time.time(),
            "inning": game.inning_label,
        })

        print(f"    FILL: {game.game_key} {side} "
              f"{fill_size:.0f} @ {fill_price:.2f} fee=${fee:.2f} "
              f"(pos={game.net_position:+.0f}, cost=${game.cost_basis:.2f}, "
              f"realized=${game.realized_pnl:+.2f})")

    async def _check_fills(self):
        """Check for fills on active orders. In dry-run, simulate instant fills."""
        for key, game in self.games.items():
            if not game.active_order_id:
                continue

            # Dry-run: simulate instant full fill
            if game.active_order_id.startswith("dry_"):
                fill_size = game.active_order_size
                if fill_size > 0:
                    self._apply_fill(
                        game, game.active_order_side,
                        fill_size, game.active_order_price,
                    )
                game.active_order_id = None
                game.active_order_side = ""
                game.active_order_price = 0.0
                game.active_order_size = 0.0
                continue

            # Live: check with CLOB API
            if not self._clob_client:
                continue

            try:
                order = await asyncio.to_thread(
                    self._clob_client.get_order, game.active_order_id
                )
            except Exception:
                continue

            size_matched = float(order.get("size_matched", 0))
            prev_filled = sum(
                f["size"] for f in game.fill_history
                if f.get("order_id") == game.active_order_id
            )
            new_fill = size_matched - prev_filled

            if new_fill > 0:
                self._apply_fill(
                    game, game.active_order_side,
                    new_fill, game.active_order_price,
                )

            # Clear fully filled or canceled orders
            status = order.get("status", "")
            if status in ("matched", "canceled"):
                game.active_order_id = None
                game.active_order_side = ""
                game.active_order_price = 0.0
                game.active_order_size = 0.0

    async def _handle_game_final(self, game: LiveGame):
        """Handle game completion — cancel orders, settle position."""
        if game.halted and game.halt_reason == "FINAL":
            return  # already handled

        await self._cancel_order(game)

        # Settle position
        if game.net_position != 0 and game.sim_state:
            home_won = game.sim_state.home_score > game.sim_state.away_score
            if game.net_position > 0:
                # Long home
                settle_value = abs(game.net_position) * (1.0 if home_won else 0.0)
            else:
                # Long away
                settle_value = abs(game.net_position) * (0.0 if home_won else 1.0)

            game.realized_pnl = settle_value - game.cost_basis
            result_str = "HOME WIN" if home_won else "AWAY WIN"
            print(f"\n  SETTLE: {game.game_key} — {result_str} — "
                  f"P&L: ${game.realized_pnl:+.2f}")

        game.halted = True
        game.halt_reason = "FINAL"

    # ── Risk Management ───────────────────────────────────────────────────

    def _check_kill_switch(self) -> bool:
        """Check if total P&L exceeds max loss — halt all trading."""
        if self._killed:
            return True

        total_pnl = sum(g.total_pnl for g in self.games.values())
        if total_pnl < -self.config.max_loss:
            print(f"\n  *** KILL SWITCH: Total P&L ${total_pnl:+.2f} < "
                  f"-${self.config.max_loss:.0f} → HALTING ALL TRADING")
            self._killed = True
            # Cancel all orders
            for game in self.games.values():
                game.halted = True
                game.halt_reason = "KILL_SWITCH"
            return True

        return False

    # ── Dashboard ─────────────────────────────────────────────────────────

    def _print_dashboard(self):
        """Print compact status dashboard."""
        now = datetime.now(timezone.utc)
        now_et = now.astimezone(ET)

        print(f"\n{'='*120}")
        print(f"  Live Trader — {self.config.target_date} — "
              f"Cycle #{self._cycle_count} — {now_et.strftime('%I:%M:%S%p')} ET — "
              f"{'DRY RUN' if self.config.dry_run else 'LIVE'}")
        print(f"  Bankroll: ${self.config.bankroll:,.0f} | "
              f"Kelly: {self.config.kelly_fraction:.0%} | "
              f"Min edge: {self.config.min_edge:.0%} | "
              f"Fee: {TAKER_FEE_RATE:.0%} | "
              f"Sims: {self.config.n_sims}")
        print(f"{'='*120}")

        print(f"\n  {'Game':<14s} {'Status':<8s} {'Inning':<8s} {'Score':<7s} "
              f"{'Model':>6s} {'Mkt':>6s} {'Edge':>6s} {'Phase':>5s} "
              f"{'Kelly':>6s} {'$Size':>7s} {'Pos':>6s} {'P&L':>8s} {'Order':>12s}")
        print(f"  {'-'*116}")

        total_pnl = 0.0
        total_exposure = 0.0
        active_orders = 0

        for key, g in sorted(self.games.items()):
            model_str = f"{g.model_home_wp:.1%}" if g.model_home_wp is not None else "  ---"
            mkt_str = f"{g.poly_home_mid:.1%}" if g.poly_home_mid is not None else "  ---"
            edge_str = f"{g.edge:+.1%}" if g.poly_home_mid is not None else "  ---"
            phase_str = f"{g.phase_confidence:.0%}" if g.phase_confidence > 0 else "  ---"
            kelly_str = f"{g.half_kelly * g.phase_confidence:.1%}" if g.half_kelly > 0 else "  ---"
            size_str = f"${g.target_size_usd:,.0f}" if g.target_size_usd > 0 else "  ---"
            pos_str = f"{g.net_position:+.0f}" if g.net_position != 0 else "  ---"
            pnl_str = f"${g.total_pnl:+.2f}" if g.cost_basis > 0 or g.realized_pnl != 0 else "  ---"

            order_str = "---"
            if g.halted:
                order_str = g.halt_reason[:12]
            elif g.active_order_id:
                side_short = g.active_order_side.replace("BUY_", "")
                order_str = f"{side_short}@{g.active_order_price:.2f}"
                active_orders += 1
                total_exposure += abs(g.cost_basis)

            total_pnl += g.total_pnl

            print(f"  {key:<14s} {g.game_status:<8s} {g.inning_label:<8s} "
                  f"{g.score_str:<7s} {model_str:>6s} {mkt_str:>6s} "
                  f"{edge_str:>6s} {phase_str:>5s} {kelly_str:>6s} "
                  f"{size_str:>7s} {pos_str:>6s} {pnl_str:>8s} {order_str:>12s}")

        print(f"\n  Active orders: {active_orders} | "
              f"Exposure: ${total_exposure:,.0f} / "
              f"${self.config.bankroll * self.config.max_total_exposure_pct:,.0f} | "
              f"P&L: ${total_pnl:+.2f}")

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_trade(self, game: LiveGame, action: str, size: float, price: float):
        """Append trade to audit log."""
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        log_path = AUDIT_DIR / f"trades_{self.config.target_date}.jsonl"

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "game_key": game.game_key,
            "game_pk": game.game_pk,
            "action": action,
            "side": game.side,
            "size": size,
            "price": price,
            "model_wp": game.model_home_wp,
            "market_mid": game.poly_home_mid,
            "edge": game.edge,
            "phase_confidence": game.phase_confidence,
            "inning": game.inning_label,
            "score": game.score_str,
            "half_kelly": game.half_kelly,
            "target_size": game.target_size_usd,
            "net_position": game.net_position,
            "cost_basis": game.cost_basis,
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── State Persistence ─────────────────────────────────────────────────

    def _save_state(self):
        """Save positions and orders for crash recovery."""

        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = STATE_DIR / f"live_trader_{self.config.target_date}.json"

        state = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "target_date": self.config.target_date,
            "cycle": self._cycle_count,
            "games": {},
        }

        for key, g in self.games.items():
            state["games"][key] = {
                "game_pk": g.game_pk,
                "net_position": g.net_position,
                "cost_basis": g.cost_basis,
                "realized_pnl": g.realized_pnl,
                "active_order_id": g.active_order_id,
                "active_order_side": g.active_order_side,
                "active_order_price": g.active_order_price,
                "active_order_size": g.active_order_size,
                "fill_history": g.fill_history,
                "halted": g.halted,
                "halt_reason": g.halt_reason,
            }

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    async def _load_state(self):
        """Load saved state for crash recovery."""
        state_path = STATE_DIR / f"live_trader_{self.config.target_date}.json"
        if not state_path.exists():
            return

        try:
            with open(state_path) as f:
                state = json.load(f)
        except Exception:
            return

        restored = 0
        for key, saved in state.get("games", {}).items():
            game = self.games.get(key)
            if not game:
                continue

            game.net_position = saved.get("net_position", 0.0)
            game.cost_basis = saved.get("cost_basis", 0.0)
            game.realized_pnl = saved.get("realized_pnl", 0.0)
            game.fill_history = saved.get("fill_history", [])
            game.active_order_id = saved.get("active_order_id")
            game.active_order_side = saved.get("active_order_side", "")
            game.active_order_price = saved.get("active_order_price", 0.0)
            game.active_order_size = saved.get("active_order_size", 0.0)

            if saved.get("halted"):
                game.halted = True
                game.halt_reason = saved.get("halt_reason", "restored")

            if game.net_position != 0:
                restored += 1

        if restored:
            print(f"  Restored positions for {restored} games from saved state")

    # ── Shutdown ──────────────────────────────────────────────────────────

    def _shutdown(self):
        """Graceful shutdown signal handler."""
        print("\n\n  Shutting down...")
        self._running = False

    async def _cleanup(self):
        """Cancel all orders and save final state."""
        print("  Cancelling all active orders...")
        for game in self.games.values():
            await self._cancel_order(game)

        self._save_state()

        # Print final P&L
        total_pnl = sum(g.total_pnl for g in self.games.values())
        total_realized = sum(g.realized_pnl for g in self.games.values())
        total_fills = sum(len(g.fill_history) for g in self.games.values())

        print(f"\n{'='*70}")
        print(f"  SESSION SUMMARY")
        print(f"  P&L: ${total_pnl:+.2f} (realized: ${total_realized:+.2f})")
        print(f"  Total fills: {total_fills}")
        print(f"  Cycles: {self._cycle_count}")
        print(f"{'='*70}")

        self._http.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MLB Live In-Game Trader (Polymarket)")
    parser.add_argument("--bankroll", type=float, required=True,
                        help="Bankroll in USD")
    parser.add_argument("--date", type=str, default=None,
                        help="Game date (YYYY-MM-DD). Default: today")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Dry run (no real orders)")
    parser.add_argument("--min-edge", type=float, default=0.03,
                        help="Minimum edge to trade (default: 0.03)")
    parser.add_argument("--kelly", type=float, default=0.25,
                        help="Kelly fraction (default: 0.25 = quarter-Kelly)")
    parser.add_argument("--poll", type=float, default=30.0,
                        help="Poll interval in seconds (default: 30)")
    parser.add_argument("--n-sims", type=int, default=5000,
                        help="MC simulations per state (default: 5000)")
    parser.add_argument("--max-position", type=float, default=0.10,
                        help="Max position per game as fraction of bankroll (default: 0.10)")
    parser.add_argument("--max-total", type=float, default=0.25,
                        help="Max total exposure as fraction of bankroll (default: 0.25)")
    parser.add_argument("--max-loss", type=float, default=500.0,
                        help="Kill switch: halt if total P&L drops below this (default: 500)")
    parser.add_argument("--season", type=int, default=2025,
                        help="Season for xRV data (default: 2025)")
    args = parser.parse_args()

    # Configure structured logging (console + file)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                AUDIT_DIR / f"live_{date.today().strftime('%Y%m%d')}.log",
            ),
        ],
    )

    target_date = args.date or date.today().strftime("%Y-%m-%d")

    config = TraderConfig(
        bankroll=args.bankroll,
        dry_run=args.dry_run,
        poll_interval_s=args.poll,
        n_sims=args.n_sims,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge,
        max_position_pct=args.max_position,
        max_total_exposure_pct=args.max_total,
        max_loss=args.max_loss,
        season=args.season,
        target_date=target_date,
    )

    trader = LiveTrader(config)

    async def _run():
        ok = await trader.initialize()
        if ok:
            await trader.run()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
