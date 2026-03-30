#!/usr/bin/env python3
"""
Automated pre-game MLB market-making bot for Polymarket.

Uses model fair values from picks.py, fetches live Polymarket orderbooks,
monitors MLB API for lineup/SP changes, and places two-sided quotes with
half-Kelly sizing.

Requires:
  - POLYMARKET_PRIVATE_KEY env var (Polygon wallet)
  - py-clob-client + polymir packages
  - Model picks JSON (run `make picks` first)

Usage:
    # Dry run (no real orders)
    python src/polymarket_bot.py --date 2026-03-30 --dry-run

    # Live trading with $5000 bankroll
    python src/polymarket_bot.py --date 2026-03-30 --bankroll 5000

    # Only trade strong-conviction games
    python src/polymarket_bot.py --date 2026-03-30 --min-edge 0.05

    # Custom poll interval
    python src/polymarket_bot.py --date 2026-03-30 --interval 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_polymarket import TEAM_NAME_TO_ABBR, resolve_team_abbr

# ── Constants ──────────────────────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
MLB_API = "https://statsapi.mlb.com/api/v1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PICKS_DIR = DATA_DIR / "picks"

# Quoting parameters
PRE_LINEUP_HALF_SPREAD = 0.03   # ±3¢ before lineups
POST_LINEUP_HALF_SPREAD = 0.015  # ±1.5¢ after lineups
MIN_EDGE_TO_QUOTE = 0.01        # don't quote if edge < 1%
TICK_SIZE = 0.01                 # Polymarket cent precision
MAX_POSITION_PER_GAME = 0.10    # max 10% of bankroll per game


# ── Data Models ────────────────────────────────────────────────────────────────

@dataclass
class PolyMarket:
    """A Polymarket MLB game market with both team token IDs."""
    home_team: str
    away_team: str
    home_token_id: str
    away_token_id: str
    game_start_time: str  # ISO datetime
    condition_id: str
    question: str
    volume: float = 0.0


@dataclass
class GameState:
    """Tracked state for a single game."""
    home_team: str
    away_team: str
    model_fair: float  # home win prob from model
    home_sp: str = ""
    away_sp: str = ""
    game_time_utc: str = ""  # ISO

    # Polymarket data
    poly_market: PolyMarket | None = None
    poly_home_bid: float | None = None  # best bid for home token
    poly_home_ask: float | None = None  # best ask for home token
    poly_home_mid: float | None = None
    poly_volume: float = 0.0

    # Lineup tracking
    home_lineup_confirmed: bool = False
    away_lineup_confirmed: bool = False
    home_sp_confirmed: bool = False
    away_sp_confirmed: bool = False
    last_home_sp: str = ""
    last_away_sp: str = ""

    # Order tracking
    bid_order_id: str | None = None
    ask_order_id: str | None = None
    bid_price: float | None = None
    ask_price: float | None = None
    bid_size: float = 0.0
    ask_size: float = 0.0
    net_position: float = 0.0  # +ve = long home, -ve = short home (long away)

    # Computed
    edge: float = 0.0
    half_kelly: float = 0.0
    side: str = ""  # "BUY_HOME" or "BUY_AWAY"
    conviction: str = "skip"
    quote_halted: bool = False  # halt on SP change until model re-runs
    halt_reason: str = ""

    @property
    def lineups_confirmed(self) -> bool:
        return self.home_lineup_confirmed and self.away_lineup_confirmed

    @property
    def half_spread(self) -> float:
        return POST_LINEUP_HALF_SPREAD if self.lineups_confirmed else PRE_LINEUP_HALF_SPREAD

    @property
    def game_time_et(self) -> str:
        """Game time in ET for display."""
        if not self.game_time_utc:
            return "TBD"
        try:
            dt = datetime.fromisoformat(self.game_time_utc.replace("Z", "+00:00"))
            et = dt - timedelta(hours=4)  # EDT offset
            return et.strftime("%-I:%M%p ET")
        except (ValueError, TypeError):
            return "TBD"


@dataclass
class BotConfig:
    """Bot runtime configuration."""
    target_date: str
    bankroll: float = 5000.0
    min_edge: float = 0.02
    poll_interval_s: float = 30.0
    dry_run: bool = True
    max_games: int = 15
    use_polymir: bool = True  # use polymir LiveClobClient for orders


# ── Polymarket Market Discovery ────────────────────────────────────────────────

def fetch_poly_mlb_markets(target_date: str) -> list[PolyMarket]:
    """Fetch active Polymarket MLB game markets for the target date."""
    markets = []
    offset = 0

    with httpx.Client(timeout=30.0) as client:
        while True:
            params = {
                "tag_slug": "mlb",
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": offset,
            }
            resp = client.get(f"{GAMMA_API}/events", params=params)
            resp.raise_for_status()
            events = resp.json()

            if not events:
                break

            for event in events:
                title = event.get("title", "")
                # Skip futures, props, etc.
                skip_words = [
                    "Series Winner", "Champion", "MVP", "Cy Young",
                    "Division", "Home Run", "sweep", "record", "props",
                ]
                if any(w.lower() in title.lower() for w in skip_words):
                    continue

                for mkt in event.get("markets", []):
                    outcomes = mkt.get("outcomes", [])
                    if isinstance(outcomes, str):
                        outcomes = json.loads(outcomes)
                    if len(outcomes) != 2:
                        continue
                    if "Yes" in outcomes or "No" in outcomes:
                        continue

                    t0 = resolve_team_abbr(outcomes[0])
                    t1 = resolve_team_abbr(outcomes[1])
                    if not t0 or not t1 or t0 == t1:
                        continue

                    # Check date
                    game_date = _parse_game_date(event, mkt)
                    if game_date != target_date:
                        continue

                    clob_ids = mkt.get("clobTokenIds", [])
                    if isinstance(clob_ids, str) and clob_ids:
                        clob_ids = json.loads(clob_ids)
                    if len(clob_ids) != 2:
                        continue

                    # t0 maps to clob_ids[0], t1 maps to clob_ids[1]
                    # We need to figure out which is home/away
                    # We'll resolve this later when matching to model picks
                    markets.append(PolyMarket(
                        home_team=t0,  # tentative — will fix in matching
                        away_team=t1,
                        home_token_id=clob_ids[0],
                        away_token_id=clob_ids[1],
                        game_start_time=mkt.get("gameStartTime", ""),
                        condition_id=mkt.get("conditionId", mkt.get("condition_id", "")),
                        question=mkt.get("question", ""),
                        volume=float(mkt.get("volumeNum") or mkt.get("volume") or 0),
                    ))

            if len(events) < 100:
                break
            offset += 100

    print(f"  Found {len(markets)} Polymarket MLB game markets for {target_date}")
    return markets


def _parse_game_date(event: dict, market: dict) -> str:
    """Extract game date from Polymarket event/market."""
    gst = market.get("gameStartTime")
    if gst:
        try:
            dt = datetime.fromisoformat(gst.replace("Z", "+00:00") if gst.endswith("Z") else gst)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    end = market.get("endDate") or event.get("endDate")
    if end:
        try:
            dt = datetime.fromisoformat(end.replace("Z", "+00:00") if end.endswith("Z") else end)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    return ""


def fetch_orderbook(client: httpx.Client, token_id: str) -> dict:
    """Fetch CLOB orderbook for a token. Returns {bids, asks, midpoint, spread}."""
    try:
        resp = client.get(f"{CLOB_API}/book", params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"bids": [], "asks": [], "midpoint": None, "spread": None, "error": str(e)}

    bids = [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])]
    asks = [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])]

    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
    spread = (best_ask - best_bid) if best_bid and best_ask else None

    return {
        "bids": bids,
        "asks": asks,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "midpoint": mid,
        "spread": spread,
        "bid_depth": sum(s for _, s in bids[:5]),
        "ask_depth": sum(s for _, s in asks[:5]),
    }


# ── MLB API Monitoring ─────────────────────────────────────────────────────────

def fetch_mlb_game_status(target_date: str) -> list[dict]:
    """Fetch current game status, SPs, and lineup availability from MLB API."""
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            f"{MLB_API}/schedule",
            params={
                "sportId": 1,
                "date": target_date,
                "hydrate": "probablePitcher,team,linescore",
                "gameType": "R",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})

            games.append({
                "game_pk": g.get("gamePk"),
                "home_team": home.get("team", {}).get("abbreviation", ""),
                "away_team": away.get("team", {}).get("abbreviation", ""),
                "home_sp": home_pitcher.get("fullName", ""),
                "away_sp": away_pitcher.get("fullName", ""),
                "status": g.get("status", {}).get("abstractGameState", ""),
                "game_time": g.get("gameDate", ""),
            })
    return games


def check_lineup_status(game_pk: int) -> dict:
    """Check if lineups are posted for a game."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{MLB_API}/game/{game_pk}/boxscore")
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return {"home_lineup": False, "away_lineup": False}

    result = {}
    for side in ("home", "away"):
        batting_order = data.get("teams", {}).get(side, {}).get("battingOrder", [])
        result[f"{side}_lineup"] = len(batting_order) >= 9
    return result


# ── Kelly Sizing ───────────────────────────────────────────────────────────────

def compute_half_kelly(model_prob: float, market_price: float) -> tuple[float, str]:
    """Compute half-Kelly fraction and side for a binary contract.

    Returns (half_kelly_fraction, side).
    Side is "BUY_HOME" if model says home is underpriced,
    "BUY_AWAY" if model says home is overpriced.

    For buying YES (home) at price c with model prob p:
        Kelly = (p - c) / (1 - c)

    For buying NO (away) — i.e., selling home YES:
        Equivalent to buying NO at (1 - c) with model prob (1 - p):
        Kelly = ((1 - p) - (1 - c)) / c = (c - p) / c
    """
    if market_price is None or market_price <= 0 or market_price >= 1:
        return 0.0, ""

    edge = model_prob - market_price

    if edge > 0:
        # Model says home is underpriced → buy YES (home token)
        kelly = edge / (1 - market_price)
        return max(kelly / 2, 0.0), "BUY_HOME"
    elif edge < 0:
        # Model says home is overpriced → buy NO (away token)
        kelly = abs(edge) / market_price
        return max(kelly / 2, 0.0), "BUY_AWAY"
    else:
        return 0.0, ""


def round_to_tick(price: float) -> float:
    """Round price to nearest cent (Polymarket tick size)."""
    return round(round(price / TICK_SIZE) * TICK_SIZE, 2)


# ── Core Bot ───────────────────────────────────────────────────────────────────

class MLBMarketMaker:
    """Pre-game market-making bot for Polymarket MLB markets."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.games: dict[str, GameState] = {}  # keyed by "HOME_AWAY"
        self._running = False
        self._cycle_count = 0
        self._clob_client = None  # polymir LiveClobClient or None in dry-run
        self._mlb_games: list[dict] = []  # raw MLB API data for lineup checks

    def _game_key(self, home: str, away: str) -> str:
        return f"{away}@{home}"

    # ── Initialization ─────────────────────────────────────────────────

    def load_model_picks(self) -> int:
        """Load model fair values from picks JSON."""
        picks_path = PICKS_DIR / f"picks_{self.config.target_date}.json"
        if not picks_path.exists():
            print(f"  ERROR: {picks_path} not found. Run `make picks` first.")
            return 0

        with open(picks_path) as f:
            data = json.load(f)

        for p in data.get("picks", []):
            key = self._game_key(p["home_team"], p["away_team"])
            self.games[key] = GameState(
                home_team=p["home_team"],
                away_team=p["away_team"],
                model_fair=p["model_fair"],
                home_sp=p.get("home_sp", ""),
                away_sp=p.get("away_sp", ""),
            )

        print(f"  Loaded {len(self.games)} game picks")
        return len(self.games)

    def match_poly_markets(self, poly_markets: list[PolyMarket]) -> int:
        """Match Polymarket markets to model picks, resolving home/away orientation."""
        matched = 0
        for pm in poly_markets:
            # Try both orientations
            key1 = self._game_key(pm.home_team, pm.away_team)
            key2 = self._game_key(pm.away_team, pm.home_team)

            if key1 in self.games:
                # pm.home_team is token[0], and it IS our home team
                self.games[key1].poly_market = pm
                self.games[key1].game_time_utc = pm.game_start_time
                matched += 1
            elif key2 in self.games:
                # pm's team order is flipped — swap token IDs
                pm_fixed = PolyMarket(
                    home_team=pm.away_team,
                    away_team=pm.home_team,
                    home_token_id=pm.away_token_id,
                    away_token_id=pm.home_token_id,
                    game_start_time=pm.game_start_time,
                    condition_id=pm.condition_id,
                    question=pm.question,
                    volume=pm.volume,
                )
                self.games[key2].poly_market = pm_fixed
                self.games[key2].game_time_utc = pm.game_start_time
                matched += 1

        print(f"  Matched {matched}/{len(self.games)} games to Polymarket markets")
        return matched

    # ── Price & Sizing Updates ─────────────────────────────────────────

    def update_prices(self):
        """Fetch live orderbook prices for all matched games."""
        with httpx.Client(timeout=15.0) as client:
            for key, gs in self.games.items():
                if not gs.poly_market:
                    continue

                book = fetch_orderbook(client, gs.poly_market.home_token_id)
                if book.get("error"):
                    continue

                gs.poly_home_bid = book["best_bid"]
                gs.poly_home_ask = book["best_ask"]
                gs.poly_home_mid = book["midpoint"]

                # Compute edge and sizing using mid price
                if gs.poly_home_mid:
                    gs.edge = gs.model_fair - gs.poly_home_mid
                    gs.half_kelly, gs.side = compute_half_kelly(gs.model_fair, gs.poly_home_mid)

                    abs_edge = abs(gs.edge)
                    if abs_edge >= 0.07:
                        gs.conviction = "strong"
                    elif abs_edge >= 0.04:
                        gs.conviction = "lean"
                    elif abs_edge >= 0.02:
                        gs.conviction = "watch"
                    else:
                        gs.conviction = "skip"

                time.sleep(0.15)  # rate limit

    def update_mlb_status(self):
        """Check MLB API for SP changes and lineup confirmations."""
        mlb_games = fetch_mlb_game_status(self.config.target_date)
        self._mlb_games = mlb_games

        for mg in mlb_games:
            key = self._game_key(mg["home_team"], mg["away_team"])
            gs = self.games.get(key)
            if not gs:
                continue

            # Track SP changes
            new_home_sp = mg.get("home_sp", "")
            new_away_sp = mg.get("away_sp", "")

            if gs.last_home_sp and new_home_sp and new_home_sp != gs.last_home_sp:
                print(f"\n  *** SP CHANGE: {key} home SP: {gs.last_home_sp} -> {new_home_sp}")
                gs.quote_halted = True
                gs.halt_reason = f"Home SP changed: {gs.last_home_sp} -> {new_home_sp}"

            if gs.last_away_sp and new_away_sp and new_away_sp != gs.last_away_sp:
                print(f"\n  *** SP CHANGE: {key} away SP: {gs.last_away_sp} -> {new_away_sp}")
                gs.quote_halted = True
                gs.halt_reason = f"Away SP changed: {gs.last_away_sp} -> {new_away_sp}"

            gs.last_home_sp = new_home_sp or gs.last_home_sp
            gs.last_away_sp = new_away_sp or gs.last_away_sp

            if new_home_sp:
                gs.home_sp_confirmed = True
            if new_away_sp:
                gs.away_sp_confirmed = True

            gs.game_time_utc = gs.game_time_utc or mg.get("game_time", "")

            # Check if game has started
            if mg.get("status") in ("Live", "Final"):
                gs.quote_halted = True
                gs.halt_reason = f"Game {mg['status']}"

        # Check lineups (separate API call per game)
        for mg in mlb_games:
            key = self._game_key(mg["home_team"], mg["away_team"])
            gs = self.games.get(key)
            if not gs or (gs.home_lineup_confirmed and gs.away_lineup_confirmed):
                continue

            game_pk = mg.get("game_pk")
            if not game_pk:
                continue

            lineup_status = check_lineup_status(game_pk)
            if lineup_status.get("home_lineup") and not gs.home_lineup_confirmed:
                gs.home_lineup_confirmed = True
                print(f"  LINEUP: {key} home lineup confirmed")
            if lineup_status.get("away_lineup") and not gs.away_lineup_confirmed:
                gs.away_lineup_confirmed = True
                print(f"  LINEUP: {key} away lineup confirmed")

    # ── Quote Management ───────────────────────────────────────────────

    def compute_quotes(self, gs: GameState) -> tuple[float, float, float, float] | None:
        """Compute bid/ask prices and sizes for a game.

        Returns (bid_price, bid_size, ask_price, ask_size) for the HOME token,
        or None if no quote should be posted.
        """
        if not gs.poly_market or gs.quote_halted:
            return None
        if gs.poly_home_mid is None:
            return None
        if abs(gs.edge) < self.config.min_edge:
            return None

        # Half-Kelly determines total desired position (in contracts)
        max_notional = self.config.bankroll * MAX_POSITION_PER_GAME
        kelly_notional = self.config.bankroll * gs.half_kelly
        position_notional = min(kelly_notional, max_notional)

        if position_notional < 1.0:
            return None

        # Compute fair-value-centered quotes
        fair = gs.model_fair
        hs = gs.half_spread

        bid_price = round_to_tick(fair - hs)
        ask_price = round_to_tick(fair + hs)

        # Clamp to valid range
        bid_price = max(0.01, min(0.99, bid_price))
        ask_price = max(0.01, min(0.99, ask_price))

        # Size: we want to accumulate position on the side we like
        # Lean into the side with edge, smaller on the other side
        if gs.side == "BUY_HOME":
            # We want to buy home — bigger bid, smaller ask
            bid_contracts = position_notional / bid_price
            ask_contracts = bid_contracts * 0.3  # token offer to capture spread
        elif gs.side == "BUY_AWAY":
            # We want to sell home (buy away) — bigger ask, smaller bid
            ask_contracts = position_notional / (1 - ask_price)
            bid_contracts = ask_contracts * 0.3
        else:
            return None

        bid_size = round(max(bid_contracts, 1.0), 1)
        ask_size = round(max(ask_contracts, 1.0), 1)

        return bid_price, bid_size, ask_price, ask_size

    async def place_or_update_quotes(self):
        """Place or update two-sided quotes for all active games."""
        for key, gs in self.games.items():
            if not gs.poly_market:
                continue

            quotes = self.compute_quotes(gs)

            # Cancel existing orders if quotes changed or halted
            should_cancel = (
                gs.quote_halted
                or quotes is None
                or (quotes and gs.bid_price != quotes[0])
                or (quotes and gs.ask_price != quotes[2])
            )

            if should_cancel:
                await self._cancel_game_orders(gs)

            if quotes is None or gs.quote_halted:
                continue

            bid_price, bid_size, ask_price, ask_size = quotes

            # Place new orders
            home_token = gs.poly_market.home_token_id

            # Bid: buy home token at bid_price
            if not gs.bid_order_id:
                result = await self._place_order(
                    home_token, "BUY", bid_price, bid_size, key, "BID"
                )
                if result:
                    gs.bid_order_id = result.get("orderID", "")
                    gs.bid_price = bid_price
                    gs.bid_size = bid_size

            # Ask: sell home token at ask_price
            if not gs.ask_order_id:
                result = await self._place_order(
                    home_token, "SELL", ask_price, ask_size, key, "ASK"
                )
                if result:
                    gs.ask_order_id = result.get("orderID", "")
                    gs.ask_price = ask_price
                    gs.ask_size = ask_size

    async def _place_order(
        self, token_id: str, side: str, price: float, size: float,
        game_key: str, label: str
    ) -> dict | None:
        """Place a single order via polymir or log in dry-run mode."""
        if self.config.dry_run:
            print(f"    [DRY RUN] {game_key} {label}: {side} {size:.0f} @ {price:.2f}")
            return {"orderID": f"dry_{game_key}_{label}_{time.time():.0f}"}

        if self._clob_client:
            try:
                result = await self._clob_client.place_order(
                    token_id=token_id,
                    side=side,
                    price=price,
                    size=size,
                )
                order_id = result.get("orderID", "")
                print(f"    ORDER: {game_key} {label}: {side} {size:.0f} @ {price:.2f} → {order_id}")
                return result
            except Exception as e:
                print(f"    ERROR: {game_key} {label}: {e}")
                return None
        return None

    async def _cancel_game_orders(self, gs: GameState):
        """Cancel all open orders for a game."""
        for attr in ("bid_order_id", "ask_order_id"):
            order_id = getattr(gs, attr)
            if not order_id:
                continue
            if not self.config.dry_run and self._clob_client:
                try:
                    await self._clob_client.cancel_order(order_id)
                except Exception:
                    pass
            setattr(gs, attr, None)

        gs.bid_price = None
        gs.ask_price = None

    # ── Display ────────────────────────────────────────────────────────

    def print_dashboard(self):
        """Print current state of all tracked games."""
        now = datetime.now(timezone.utc)
        now_et = now - timedelta(hours=4)

        print(f"\n{'='*110}")
        print(f"  MLB Market Maker — {self.config.target_date} — "
              f"Cycle #{self._cycle_count} — {now_et.strftime('%I:%M:%S%p')} ET — "
              f"{'DRY RUN' if self.config.dry_run else 'LIVE'}")
        print(f"  Bankroll: ${self.config.bankroll:,.0f} | Min edge: {self.config.min_edge:.0%} | "
              f"Poll: {self.config.poll_interval_s:.0f}s")
        print(f"{'='*110}")

        print(f"\n  {'Game':<14s} {'Time':>9s} {'Model':>6s} {'Poly':>6s} {'Edge':>6s} "
              f"{'½K%':>5s} {'Side':>10s} {'Conv':>7s} {'Lineup':>7s} "
              f"{'Bid':>6s} {'Ask':>6s} {'Status':>12s}")
        print(f"  {'-'*108}")

        # Sort by game time
        sorted_games = sorted(
            self.games.values(),
            key=lambda g: g.game_time_utc or "9999",
        )

        total_exposure = 0.0
        active_quotes = 0

        for gs in sorted_games:
            game = f"{gs.away_team}@{gs.home_team}"
            time_str = gs.game_time_et
            model_str = f"{gs.model_fair:.1%}"
            poly_str = f"{gs.poly_home_mid:.1%}" if gs.poly_home_mid else "  ---"
            edge_str = f"{gs.edge:+.1%}" if gs.poly_home_mid else "  ---"
            kelly_str = f"{gs.half_kelly:.1%}" if gs.half_kelly > 0 else "  ---"
            side_str = gs.side.replace("BUY_", "") if gs.side else "---"
            conv_str = gs.conviction.upper() if gs.conviction in ("strong", "lean") else gs.conviction
            lineup_str = "Y" if gs.lineups_confirmed else ("P" if gs.home_lineup_confirmed or gs.away_lineup_confirmed else "N")
            bid_str = f"{gs.bid_price:.2f}" if gs.bid_price else "  ---"
            ask_str = f"{gs.ask_price:.2f}" if gs.ask_price else "  ---"

            if gs.quote_halted:
                status = f"HALTED:{gs.halt_reason[:8]}"
            elif gs.bid_order_id or gs.ask_order_id:
                status = "QUOTING"
                active_quotes += 1
                total_exposure += gs.half_kelly * self.config.bankroll
            elif gs.poly_market is None:
                status = "NO_MKT"
            elif abs(gs.edge) < self.config.min_edge:
                status = "THIN_EDGE"
            else:
                status = "READY"

            print(f"  {game:<14s} {time_str:>9s} {model_str:>6s} {poly_str:>6s} {edge_str:>6s} "
                  f"{kelly_str:>5s} {side_str:>10s} {conv_str:>7s} {lineup_str:>7s} "
                  f"{bid_str:>6s} {ask_str:>6s} {status:>12s}")

        print(f"\n  Active quotes: {active_quotes} | "
              f"Total exposure: ${total_exposure:,.0f} / ${self.config.bankroll:,.0f}")

    # ── Main Loop ──────────────────────────────────────────────────────

    async def run(self):
        """Main bot loop."""
        print(f"\n{'='*70}")
        print(f"  MLB Polymarket Market Maker")
        print(f"  Date: {self.config.target_date}")
        print(f"  Mode: {'DRY RUN' if self.config.dry_run else 'LIVE TRADING'}")
        print(f"  Bankroll: ${self.config.bankroll:,.0f}")
        print(f"{'='*70}")

        # Step 1: Load model picks
        print("\n[1/3] Loading model picks...")
        n = self.load_model_picks()
        if n == 0:
            return

        # Step 2: Discover Polymarket markets
        print("\n[2/3] Discovering Polymarket markets...")
        poly_markets = fetch_poly_mlb_markets(self.config.target_date)
        self.match_poly_markets(poly_markets)

        # Step 3: Initial price fetch
        print("\n[3/3] Fetching initial prices...")
        self.update_prices()
        self.update_mlb_status()
        self.print_dashboard()

        # Initialize trading client
        if not self.config.dry_run and self.config.use_polymir:
            try:
                from polymir.config import APIConfig
                from polymir.live_client import LiveClobClient
                api_config = APIConfig.from_env()
                self._clob_client = LiveClobClient(api_config)
                await self._clob_client.__aenter__()
                print("\n  Connected to Polymarket CLOB (live trading)")
            except Exception as e:
                print(f"\n  WARNING: Could not init live client: {e}")
                print("  Falling back to dry-run mode")
                self.config.dry_run = True

        # Main loop
        self._running = True
        print(f"\n  Starting main loop (poll every {self.config.poll_interval_s}s, Ctrl+C to stop)...")

        try:
            while self._running:
                self._cycle_count += 1

                # Update prices
                self.update_prices()

                # Check MLB API for changes (every 3rd cycle to avoid hammering)
                if self._cycle_count % 3 == 0:
                    self.update_mlb_status()

                # Place/update quotes
                await self.place_or_update_quotes()

                # Display
                self.print_dashboard()

                # Wait
                await asyncio.sleep(self.config.poll_interval_s)

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n\n  Shutting down...")
        finally:
            # Cancel all open orders
            print("  Cancelling all open orders...")
            for gs in self.games.values():
                await self._cancel_game_orders(gs)

            if self._clob_client and not self.config.dry_run:
                await self._clob_client.__aexit__(None, None, None)

            print("  Bot stopped.")

    def stop(self):
        self._running = False


# ── One-shot: Fetch Poly prices and recompute sizing ───────────────────────────

def print_sizing_table(target_date: str, bankroll: float = 5000.0):
    """Fetch Polymarket prices and print half-Kelly sizing table (no trading)."""
    # Load picks
    picks_path = PICKS_DIR / f"picks_{target_date}.json"
    if not picks_path.exists():
        print(f"  ERROR: {picks_path} not found. Run `make picks` first.")
        return

    with open(picks_path) as f:
        data = json.load(f)
    picks = data.get("picks", [])

    # Discover Polymarket markets
    print(f"\nFetching Polymarket MLB markets for {target_date}...")
    poly_markets = fetch_poly_mlb_markets(target_date)

    # Build lookup by team pair
    poly_lookup: dict[tuple, PolyMarket] = {}
    for pm in poly_markets:
        poly_lookup[(pm.home_team, pm.away_team)] = pm
        poly_lookup[(pm.away_team, pm.home_team)] = pm

    # Fetch orderbooks and compute sizing
    print("Fetching orderbooks...\n")
    rows = []

    with httpx.Client(timeout=15.0) as client:
        for p in picks:
            ht = p["home_team"]
            at = p["away_team"]
            fair = p["model_fair"]

            # Find matching Polymarket market
            pm = poly_lookup.get((ht, at)) or poly_lookup.get((at, ht))

            if not pm:
                rows.append({
                    "game": f"{at}@{ht}",
                    "model": fair,
                    "poly_mid": None,
                    "edge": None,
                    "side": "",
                    "half_kelly": 0,
                    "size_usd": 0,
                    "game_time": "",
                    "status": "NO_MKT",
                })
                continue

            # Determine correct home token
            if pm.home_team == ht:
                home_token = pm.home_token_id
            else:
                home_token = pm.away_token_id

            book = fetch_orderbook(client, home_token)
            mid = book.get("midpoint")

            if mid is None:
                rows.append({
                    "game": f"{at}@{ht}",
                    "model": fair,
                    "poly_mid": None,
                    "edge": None,
                    "side": "",
                    "half_kelly": 0,
                    "size_usd": 0,
                    "game_time": pm.game_start_time,
                    "status": "NO_BOOK",
                })
                time.sleep(0.15)
                continue

            edge = fair - mid
            hk, side = compute_half_kelly(fair, mid)
            size_usd = min(bankroll * hk, bankroll * MAX_POSITION_PER_GAME)

            rows.append({
                "game": f"{at}@{ht}",
                "model": fair,
                "poly_mid": mid,
                "poly_bid": book.get("best_bid"),
                "poly_ask": book.get("best_ask"),
                "poly_spread": book.get("spread"),
                "bid_depth": book.get("bid_depth", 0),
                "ask_depth": book.get("ask_depth", 0),
                "edge": edge,
                "side": side.replace("BUY_", ""),
                "half_kelly": hk,
                "size_usd": size_usd,
                "game_time": pm.game_start_time,
                "status": "OK",
            })
            time.sleep(0.15)

    # Sort by game time
    rows.sort(key=lambda r: r.get("game_time") or "9999")

    # Print table
    print(f"{'='*130}")
    print(f"  POLYMARKET HALF-KELLY SIZING — {target_date} — Bankroll: ${bankroll:,.0f}")
    print(f"{'='*130}")

    print(f"\n  {'ET Time':>9s} {'Game':<14s} {'Model':>6s} {'Poly':>6s} {'Bid':>5s} {'Ask':>5s} "
          f"{'Sprd':>5s} {'Edge':>6s} {'Side':>6s} {'½Kelly':>7s} "
          f"{'$Size':>7s} {'BidDpth':>8s} {'AskDpth':>8s}")
    print(f"  {'-'*125}")

    total_size = 0.0
    for r in rows:
        gt = r.get("game_time", "")
        if gt:
            try:
                dt = datetime.fromisoformat(gt.replace("Z", "+00:00") if gt.endswith("Z") else gt)
                et = dt - timedelta(hours=4)
                time_str = et.strftime("%-I:%M%p")
            except (ValueError, TypeError):
                time_str = "TBD"
        else:
            time_str = "TBD"

        game = r["game"]
        model_str = f"{r['model']:.1%}"
        poly_str = f"{r['poly_mid']:.1%}" if r.get("poly_mid") else "  ---"
        bid_str = f"{r['poly_bid']:.2f}" if r.get("poly_bid") else " ---"
        ask_str = f"{r['poly_ask']:.2f}" if r.get("poly_ask") else " ---"
        sprd_str = f"{r['poly_spread']:.2f}" if r.get("poly_spread") else " ---"
        edge_str = f"{r['edge']:+.1%}" if r.get("edge") is not None else "  ---"
        side_str = r.get("side", "---") or "---"
        hk_str = f"{r['half_kelly']:.1%}" if r["half_kelly"] > 0 else "  ---"
        size_str = f"${r['size_usd']:,.0f}" if r["size_usd"] > 0 else "  ---"
        bd_str = f"${r.get('bid_depth', 0):,.0f}" if r.get("bid_depth") else "  ---"
        ad_str = f"${r.get('ask_depth', 0):,.0f}" if r.get("ask_depth") else "  ---"

        total_size += r.get("size_usd", 0)

        print(f"  {time_str:>9s} {game:<14s} {model_str:>6s} {poly_str:>6s} {bid_str:>5s} {ask_str:>5s} "
              f"{sprd_str:>5s} {edge_str:>6s} {side_str:>6s} {hk_str:>7s} "
              f"{size_str:>7s} {bd_str:>8s} {ad_str:>8s}")

    print(f"\n  Total desired exposure: ${total_size:,.0f} / ${bankroll:,.0f} bankroll")
    print(f"  Max per game: ${bankroll * MAX_POSITION_PER_GAME:,.0f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB Polymarket Market-Making Bot")
    parser.add_argument("--date", type=str, required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--bankroll", type=float, default=5000.0, help="Bankroll in USD")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Minimum edge to quote")
    parser.add_argument("--interval", type=float, default=30.0, help="Poll interval in seconds")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Dry run (no real orders)")
    parser.add_argument("--sizing-only", action="store_true", help="Just print sizing table, don't run bot")
    args = parser.parse_args()

    if args.sizing_only:
        print_sizing_table(args.date, args.bankroll)
        return

    config = BotConfig(
        target_date=args.date,
        bankroll=args.bankroll,
        min_edge=args.min_edge,
        poll_interval_s=args.interval,
        dry_run=args.dry_run,
    )

    bot = MLBMarketMaker(config)

    # Handle Ctrl+C gracefully
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler(sig, frame):
        bot.stop()

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        loop.run_until_complete(bot.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
