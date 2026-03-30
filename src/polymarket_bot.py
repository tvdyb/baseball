#!/usr/bin/env python3
"""
Automated pre-game MLB market-making bot for Polymarket.

Uses model fair values from picks.py, fetches live Polymarket orderbooks,
monitors MLB API for lineup/SP changes, and places two-sided quotes with
half-Kelly sizing.

Requires:
  - POLYMARKET_PRIVATE_KEY env var (Polygon wallet)
  - POLYMARKET_API_KEY, POLYMARKET_API_SECRET, POLYMARKET_API_PASSPHRASE env vars
  - py-clob-client package (pip install py-clob-client)
  - Model picks JSON (run `make picks` first)

Usage:
    # Dry run (no real orders)
    python src/polymarket_bot.py --date 2026-03-30 --dry-run

    # Live trading with $5000 bankroll
    python src/polymarket_bot.py --date 2026-03-30 --bankroll 5000

    # Only trade strong-conviction games
    python src/polymarket_bot.py --date 2026-03-30 --min-edge 0.05

    # Debug market discovery chain
    python src/polymarket_bot.py --date 2026-03-30 --sizing-only --debug-discovery
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

# Risk management: volume spike detection
VOL_SPIKE_WINDOW_S = 30         # rolling window for volume tracking
VOL_SPIKE_THRESHOLD = 500       # contracts traded in window → pull quotes
VOL_SPIKE_COOLDOWN_S = 60       # stay pulled for this long after spike

# Risk management: pre-pitch wind-down
WIDEN_SPREAD_MINS = 30          # widen spreads this many min before first pitch
PULL_QUOTES_MINS = 10           # pull all quotes this many min before first pitch

# Risk management: adverse selection / fill tracking
MAX_CONSECUTIVE_FILLS = 3       # if same side fills 3x in a row → pull & reprice
POSITION_LIMIT_CONTRACTS = 0    # 0 = use Kelly sizing, >0 = hard cap per game

# Fast MLB monitoring
MLB_FAST_POLL_S = 8             # poll MLB API every 8s for SP/lineup changes

# Global debug flag (set by --debug-discovery)
DEBUG_DISCOVERY = False


def _debug(msg: str):
    """Print debug message if debug mode is enabled."""
    if DEBUG_DISCOVERY:
        print(f"  [DEBUG] {msg}")


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
    # Gamma-level price hints (may be stale, use CLOB book as source of truth)
    gamma_best_bid: float | None = None
    gamma_best_ask: float | None = None
    neg_risk: bool = False
    accepting_orders: bool = True


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
    poly_spread: float | None = None
    poly_bid_depth: float = 0.0  # $ depth at top 5 levels
    poly_ask_depth: float = 0.0
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
    bid_filled: float = 0.0   # how much of current bid order has filled
    ask_filled: float = 0.0   # how much of current ask order has filled
    net_position: float = 0.0  # +ve = long home, -ve = short home (long away)

    # P&L tracking (from confirmed fills)
    realized_pnl: float = 0.0    # realized P&L from round-trip fills
    cost_basis: float = 0.0      # total $ spent on current position
    fill_history: list = field(default_factory=list)  # [{side, price, size, ts}]

    # Computed
    edge: float = 0.0
    half_kelly: float = 0.0
    side: str = ""  # "BUY_HOME" or "BUY_AWAY"
    conviction: str = "skip"
    quote_halted: bool = False  # halt on SP change until model re-runs
    halt_reason: str = ""

    # Risk management: trade flow / volume spike
    recent_trades: list = field(default_factory=list)  # [(timestamp, size, side)]
    vol_spike_until: float = 0.0  # unix timestamp when cooldown expires
    last_mid: float | None = None  # for mid-move detection
    mid_move_3s: float = 0.0      # mid price change over last snapshot

    # Risk management: fill tracking / adverse selection
    consecutive_bid_fills: int = 0
    consecutive_ask_fills: int = 0
    total_bought: float = 0.0    # contracts bought (home token)
    total_sold: float = 0.0      # contracts sold (home token)
    adverse_halted: bool = False
    adverse_halt_until: float = 0.0  # unix ts

    # Risk management: pre-pitch wind-down
    spread_widened: bool = False
    pre_pitch_pulled: bool = False

    @property
    def lineups_confirmed(self) -> bool:
        return self.home_lineup_confirmed and self.away_lineup_confirmed

    @property
    def half_spread(self) -> float:
        if self.spread_widened:
            return PRE_LINEUP_HALF_SPREAD * 1.5  # 4.5¢ in wind-down zone
        return POST_LINEUP_HALF_SPREAD if self.lineups_confirmed else PRE_LINEUP_HALF_SPREAD

    @property
    def is_vol_spiking(self) -> bool:
        return time.time() < self.vol_spike_until

    @property
    def is_adverse_halted(self) -> bool:
        return self.adverse_halted and time.time() < self.adverse_halt_until

    @property
    def minutes_to_first_pitch(self) -> float | None:
        """Minutes until game start. None if unknown."""
        if not self.game_time_utc:
            return None
        try:
            gst = self.game_time_utc
            if gst.endswith("Z"):
                gst = gst.replace("Z", "+00:00")
            elif " " in gst and "T" not in gst:
                gst = gst.replace(" ", "T", 1)
            if gst.endswith("+00"):
                gst += ":00"
            dt = datetime.fromisoformat(gst)
            delta = (dt - datetime.now(timezone.utc)).total_seconds() / 60
            return delta
        except (ValueError, TypeError):
            return None

    @property
    def game_time_et(self) -> str:
        """Game time in ET for display."""
        if not self.game_time_utc:
            return "TBD"
        try:
            gst = self.game_time_utc
            if gst.endswith("Z"):
                gst = gst.replace("Z", "+00:00")
            elif "+" not in gst and gst[-3:-2] != "-":
                # Handle Polymarket format "2026-03-30 20:10:00+00"
                gst = gst.replace(" ", "T") if "T" not in gst else gst
                if gst.endswith("+00"):
                    gst += ":00"
            dt = datetime.fromisoformat(gst)
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
    debug_discovery: bool = False
    max_loss: float = 500.0  # kill switch: halt all trading if total P&L < -max_loss


# ── Polymarket Market Discovery ────────────────────────────────────────────────

def discover_mlb_series_id() -> str | None:
    """Discover MLB series_id dynamically from /sports endpoint."""
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{GAMMA_API}/sports")
            resp.raise_for_status()
            sports = resp.json()

        for s in sports:
            if s.get("sport", "").lower() == "mlb":
                series_id = str(s.get("series", ""))
                tags = s.get("tags", "")
                _debug(f"/sports → MLB: series={series_id}, tags={tags}")
                return series_id

        _debug(f"/sports → MLB not found in {len(sports)} sports")
    except Exception as e:
        _debug(f"/sports failed: {e}")
    return None


def _json_parse(val):
    """Safely parse a JSON-encoded string field (clobTokenIds, outcomes, etc.)."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
    return val if isinstance(val, list) else []


def fetch_poly_mlb_markets(target_date: str) -> list[PolyMarket]:
    """Fetch active Polymarket MLB moneyline game markets for the target date.

    Discovery chain:
      1. GET /sports → find MLB series_id
      2. GET /events?series_id=...&active=true&closed=false → all active MLB events
      3. Filter markets by: gameStartTime matches target_date, sportsMarketType=moneyline,
         2 team outcomes, acceptingOrders=true
      4. Parse clobTokenIds via json.loads (they're JSON strings)
    """
    # Step 1: Discover series_id
    series_id = discover_mlb_series_id()
    if not series_id:
        print("  WARNING: Could not discover MLB series_id from /sports, falling back to series_id=3")
        series_id = "3"

    # Step 2: Fetch all active MLB events
    markets = []
    offset = 0
    total_events = 0
    total_markets_scanned = 0

    with httpx.Client(timeout=30.0) as client:
        while True:
            params = {
                "series_id": series_id,
                "active": "true",
                "closed": "false",
                "limit": 100,
                "offset": offset,
                "order": "startTime",
                "ascending": "true",
            }
            resp = client.get(f"{GAMMA_API}/events", params=params)
            resp.raise_for_status()
            events = resp.json()

            if not events:
                break

            total_events += len(events)

            for event in events:
                title = event.get("title", "")

                # Skip futures, props, series-level markets
                skip_words = [
                    "Series Winner", "Champion", "MVP", "Cy Young",
                    "Division", "Home Run", "sweep", "record", "props",
                    "CBA", "sale", "Jersey",
                ]
                if any(w.lower() in title.lower() for w in skip_words):
                    _debug(f"SKIP event (title filter): {title}")
                    continue

                for mkt in event.get("markets", []):
                    total_markets_scanned += 1

                    # Filter: must be a moneyline game market
                    smt = mkt.get("sportsMarketType", "")
                    if smt and smt != "moneyline":
                        continue

                    # Filter: must be accepting orders and not closed
                    if mkt.get("closed", False):
                        continue
                    if not mkt.get("acceptingOrders", True):
                        _debug(f"SKIP market (not accepting orders): {mkt.get('question', '')}")
                        continue

                    # Filter: must have exactly 2 team outcomes
                    outcomes = _json_parse(mkt.get("outcomes", []))
                    if len(outcomes) != 2:
                        continue
                    if "Yes" in outcomes or "No" in outcomes:
                        continue

                    t0 = resolve_team_abbr(outcomes[0])
                    t1 = resolve_team_abbr(outcomes[1])
                    if not t0 or not t1 or t0 == t1:
                        _debug(f"SKIP market (team resolve failed): outcomes={outcomes}")
                        continue

                    # Filter: game date must match target
                    game_date = _parse_game_date(event, mkt)
                    if game_date != target_date:
                        continue

                    # Parse clobTokenIds (JSON string!)
                    clob_ids = _json_parse(mkt.get("clobTokenIds", []))
                    if len(clob_ids) != 2:
                        _debug(f"SKIP market (bad clobTokenIds): {mkt.get('clobTokenIds')}")
                        continue

                    neg_risk = bool(mkt.get("negRisk", False))

                    # Extract Gamma-level price hints
                    gamma_bid = mkt.get("bestBid")
                    gamma_ask = mkt.get("bestAsk")
                    try:
                        gamma_bid = float(gamma_bid) if gamma_bid else None
                        gamma_ask = float(gamma_ask) if gamma_ask else None
                    except (ValueError, TypeError):
                        gamma_bid = gamma_ask = None

                    pm = PolyMarket(
                        home_team=t0,  # tentative — resolved in matching
                        away_team=t1,
                        home_token_id=clob_ids[0],
                        away_token_id=clob_ids[1],
                        game_start_time=mkt.get("gameStartTime", ""),
                        condition_id=mkt.get("conditionId", mkt.get("condition_id", "")),
                        question=mkt.get("question", title),
                        volume=float(mkt.get("volumeNum") or mkt.get("volume") or 0),
                        gamma_best_bid=gamma_bid,
                        gamma_best_ask=gamma_ask,
                        neg_risk=neg_risk,
                        accepting_orders=True,
                    )
                    markets.append(pm)

                    _debug(f"FOUND: {t0} vs {t1} | date={game_date} | "
                           f"gammaBid={gamma_bid} gammaAsk={gamma_ask} | "
                           f"negRisk={neg_risk} | vol=${pm.volume:,.0f} | "
                           f"tokens=[{clob_ids[0][:20]}..., {clob_ids[1][:20]}...]")

            if len(events) < 100:
                break
            offset += 100

    print(f"  Discovery: scanned {total_events} events, {total_markets_scanned} markets → "
          f"found {len(markets)} moneyline games for {target_date}")
    return markets


def _parse_game_date(event: dict, market: dict) -> str:
    """Extract game date from Polymarket event/market."""
    gst = market.get("gameStartTime")
    if gst:
        try:
            # Handle formats: "2026-03-30T20:10:00Z" and "2026-03-30 20:10:00+00"
            gst_clean = gst.replace("Z", "+00:00") if gst.endswith("Z") else gst
            if " " in gst_clean and "T" not in gst_clean:
                gst_clean = gst_clean.replace(" ", "T", 1)
            if gst_clean.endswith("+00"):
                gst_clean += ":00"
            dt = datetime.fromisoformat(gst_clean)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    end = market.get("endDate") or event.get("endDate")
    if end:
        try:
            end_clean = end.replace("Z", "+00:00") if end.endswith("Z") else end
            dt = datetime.fromisoformat(end_clean)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    return ""


# ── CLOB Orderbook Fetching ───────────────────────────────────────────────────

def fetch_orderbook(client: httpx.Client, token_id: str) -> dict:
    """Fetch CLOB orderbook for a token.

    IMPORTANT: The CLOB /book endpoint returns bids in ascending price order
    and asks in descending price order. We must sort to find the real
    top-of-book: best bid = highest bid, best ask = lowest ask.

    Returns {best_bid, best_ask, midpoint, spread, bid_depth, ask_depth}.
    """
    try:
        resp = client.get(f"{CLOB_API}/book", params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"best_bid": None, "best_ask": None, "midpoint": None,
                "spread": None, "error": str(e)}

    if "error" in data:
        return {"best_bid": None, "best_ask": None, "midpoint": None,
                "spread": None, "error": data["error"]}

    raw_bids = data.get("bids", [])
    raw_asks = data.get("asks", [])

    # Sort bids descending (highest first), asks ascending (lowest first)
    bids = sorted(
        [(float(b["price"]), float(b["size"])) for b in raw_bids],
        key=lambda x: x[0],
        reverse=True,
    )
    asks = sorted(
        [(float(a["price"]), float(a["size"])) for a in raw_asks],
        key=lambda x: x[0],
    )

    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None

    # Compute midpoint and spread from top-of-book
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
    else:
        mid = None
        spread = None

    # Depth: sum $ value (price * size) of top 5 levels
    bid_depth = sum(p * s for p, s in bids[:5]) if bids else 0.0
    ask_depth = sum(p * s for p, s in asks[:5]) if asks else 0.0

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "midpoint": mid,
        "spread": spread,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "n_bid_levels": len(bids),
        "n_ask_levels": len(asks),
    }


def fetch_clob_midpoint(client: httpx.Client, token_id: str) -> float | None:
    """Fetch CLOB midpoint for a token (cross-check with orderbook).

    The /midpoint endpoint aggregates both the direct book and the complement
    token book, so it may differ from the raw orderbook midpoint.
    """
    try:
        resp = client.get(f"{CLOB_API}/midpoint", params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()
        mid = data.get("mid")
        return float(mid) if mid else None
    except Exception:
        return None


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


# ── Risk Management: Volume Spike Detection ───────────────────────────────────

class TradeFlowMonitor:
    """Monitors Polymarket WebSocket trade feed for volume spikes.

    When trade volume in a rolling window exceeds the threshold for any game,
    it triggers an immediate quote pull for that game. This protects against
    adverse selection when informed flow hits the book.

    Runs as a concurrent async task alongside the main bot loop.
    """

    def __init__(self, games: dict[str, GameState]):
        self._games = games
        self._token_to_game: dict[str, str] = {}  # token_id → game_key
        self._running = False
        self._ws = None

    def build_token_map(self):
        """Map token IDs to game keys for fast lookup on trade events."""
        self._token_to_game.clear()
        for key, gs in self._games.items():
            if gs.poly_market:
                self._token_to_game[gs.poly_market.home_token_id] = key
                self._token_to_game[gs.poly_market.away_token_id] = key

    async def run(self):
        """Connect to WS and monitor trade flow. Pull quotes on volume spikes."""
        self.build_token_map()
        if not self._token_to_game:
            return

        token_ids = list(self._token_to_game.keys())
        self._running = True

        try:
            import websockets
        except ImportError:
            print("  WARNING: websockets not installed, trade flow monitoring disabled")
            return

        ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

        while self._running:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws = ws
                    # Subscribe to all game tokens
                    sub_msg = json.dumps({
                        "type": "subscribe",
                        "assets_ids": token_ids,
                    })
                    await ws.send(sub_msg)
                    print(f"  [FLOW] WebSocket connected, monitoring {len(token_ids)} tokens")

                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        # Process trade messages
                        msg_type = data.get("type", "")
                        if msg_type not in ("trade", "last_trade_price"):
                            continue

                        trades = data.get("trades", [data]) if "trades" in data else [data]
                        for t in trades:
                            self._process_trade(t)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    print(f"  [FLOW] WebSocket error: {e}, reconnecting in 5s...")
                    await asyncio.sleep(5)

    def _process_trade(self, trade: dict):
        """Process a single trade event and check for volume spikes."""
        asset_id = trade.get("asset_id", trade.get("market", ""))
        game_key = self._token_to_game.get(asset_id)
        if not game_key:
            return

        gs = self._games.get(game_key)
        if not gs:
            return

        now = time.time()
        size = float(trade.get("size", 0))
        side = trade.get("side", "")

        # Record trade
        gs.recent_trades.append((now, size, side))

        # Trim old trades outside window
        cutoff = now - VOL_SPIKE_WINDOW_S
        gs.recent_trades = [(t, s, sd) for t, s, sd in gs.recent_trades if t > cutoff]

        # Check total volume in window
        window_vol = sum(s for _, s, _ in gs.recent_trades)

        if window_vol >= VOL_SPIKE_THRESHOLD and not gs.is_vol_spiking:
            gs.vol_spike_until = now + VOL_SPIKE_COOLDOWN_S
            print(f"\n  *** VOL SPIKE: {game_key} — {window_vol:.0f} contracts in {VOL_SPIKE_WINDOW_S}s → PULLING QUOTES")

    def stop(self):
        self._running = False


# ── Risk Management: Fast MLB Monitoring ──────────────────────────────────────

class FastMLBMonitor:
    """Polls MLB API at high frequency for SP changes and lineup drops.

    Runs as a concurrent async task. On detection of a change, immediately
    sets the halt flag on the affected game so the quoting loop cancels orders
    on its next iteration (which should be near-instant since we use asyncio).

    When a game transitions to Live/Final, triggers immediate order cancellation
    via the bot's cancel method (if a bot reference is provided).
    """

    def __init__(self, games: dict[str, GameState], target_date: str,
                 cancel_callback=None):
        self._games = games
        self._target_date = target_date
        self._running = False
        self._game_pks: dict[str, int] = {}  # game_key → game_pk
        self._cancel_callback = cancel_callback  # async fn(GameState) to cancel orders

    async def run(self):
        """High-frequency MLB API polling loop."""
        self._running = True
        cycle = 0
        print(f"  [MLB] Fast monitor started (poll every {MLB_FAST_POLL_S}s)")

        while self._running:
            try:
                await asyncio.sleep(MLB_FAST_POLL_S)
                cycle += 1

                # Fetch schedule (SP + status)
                mlb_games = await asyncio.to_thread(
                    fetch_mlb_game_status, self._target_date
                )

                for mg in mlb_games:
                    key = f"{mg['away_team']}@{mg['home_team']}"
                    gs = self._games.get(key)
                    if not gs:
                        continue

                    # Cache game_pk for lineup checks
                    if mg.get("game_pk"):
                        self._game_pks[key] = mg["game_pk"]

                    # SP change detection
                    new_home_sp = mg.get("home_sp", "")
                    new_away_sp = mg.get("away_sp", "")

                    if gs.last_home_sp and new_home_sp and new_home_sp != gs.last_home_sp:
                        print(f"\n  *** [MLB] SP CHANGE: {key} home: "
                              f"{gs.last_home_sp} → {new_home_sp} → HALTING")
                        gs.quote_halted = True
                        gs.halt_reason = f"SP: {new_home_sp}"

                    if gs.last_away_sp and new_away_sp and new_away_sp != gs.last_away_sp:
                        print(f"\n  *** [MLB] SP CHANGE: {key} away: "
                              f"{gs.last_away_sp} → {new_away_sp} → HALTING")
                        gs.quote_halted = True
                        gs.halt_reason = f"SP: {new_away_sp}"

                    gs.last_home_sp = new_home_sp or gs.last_home_sp
                    gs.last_away_sp = new_away_sp or gs.last_away_sp
                    if new_home_sp:
                        gs.home_sp_confirmed = True
                    if new_away_sp:
                        gs.away_sp_confirmed = True

                    gs.game_time_utc = gs.game_time_utc or mg.get("game_time", "")

                    # Game started — immediately cancel all orders
                    if mg.get("status") in ("Live", "Final"):
                        if not gs.quote_halted:
                            print(f"\n  *** [MLB] GAME {mg['status'].upper()}: {key} → HALTING & CANCELLING")
                            if self._cancel_callback:
                                await self._cancel_callback(gs)
                        gs.quote_halted = True
                        gs.halt_reason = f"Game {mg['status']}"

                # Lineup checks (every 3rd cycle to limit API calls)
                if cycle % 3 == 0:
                    for key, game_pk in self._game_pks.items():
                        gs = self._games.get(key)
                        if not gs or gs.lineups_confirmed:
                            continue

                        status = await asyncio.to_thread(check_lineup_status, game_pk)
                        if status.get("home_lineup") and not gs.home_lineup_confirmed:
                            gs.home_lineup_confirmed = True
                            print(f"  [MLB] LINEUP: {key} home lineup confirmed")
                        if status.get("away_lineup") and not gs.away_lineup_confirmed:
                            gs.away_lineup_confirmed = True
                            print(f"  [MLB] LINEUP: {key} away lineup confirmed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  [MLB] Monitor error: {e}")
                await asyncio.sleep(5)

    def stop(self):
        self._running = False


# ── Risk Management: Pre-Pitch Wind-Down ──────────────────────────────────────

class PrePitchWindDown:
    """Monitors time-to-first-pitch and progressively reduces exposure.

    - WIDEN_SPREAD_MINS before pitch: widen spreads (set spread_widened flag)
    - PULL_QUOTES_MINS before pitch: pull all quotes (set pre_pitch_pulled flag)
    - After first pitch: halt (handled by FastMLBMonitor game status check)
    """

    def __init__(self, games: dict[str, GameState]):
        self._games = games
        self._running = False

    async def run(self):
        self._running = True
        while self._running:
            try:
                await asyncio.sleep(10)  # check every 10s

                for key, gs in self._games.items():
                    mins = gs.minutes_to_first_pitch
                    if mins is None:
                        continue

                    # Progressive wind-down
                    if mins <= PULL_QUOTES_MINS and not gs.pre_pitch_pulled:
                        gs.pre_pitch_pulled = True
                        gs.quote_halted = True
                        gs.halt_reason = f"<{PULL_QUOTES_MINS}m to pitch"
                        print(f"\n  *** [RISK] PRE-PITCH PULL: {key} — "
                              f"{mins:.0f}min to first pitch → PULLING ALL QUOTES")

                    elif mins <= WIDEN_SPREAD_MINS and not gs.spread_widened:
                        gs.spread_widened = True
                        print(f"  [RISK] WIDEN: {key} — {mins:.0f}min to pitch → widening spreads")

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(5)

    def stop(self):
        self._running = False


# ── Core Bot ───────────────────────────────────────────────────────────────────

class MLBMarketMaker:
    """Pre-game market-making bot for Polymarket MLB markets."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.games: dict[str, GameState] = {}  # keyed by "AWAY@HOME"
        self._running = False
        self._killed = False  # kill switch tripped
        self._cycle_count = 0
        self._clob_client = None  # py-clob-client ClobClient or None in dry-run
        self._mlb_games: list[dict] = []  # raw MLB API data for lineup checks
        self._trade_flow: TradeFlowMonitor | None = None
        self._mlb_monitor: FastMLBMonitor | None = None
        self._wind_down: PrePitchWindDown | None = None

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
        """Match Polymarket markets to model picks, resolving home/away orientation.

        Polymarket outcomes list teams as [team_A, team_B] where team_A is
        typically the away team (Polymarket uses "away @ home" ordering from
        the /sports endpoint). We match by team pair regardless of order.
        """
        matched = 0
        for pm in poly_markets:
            # Try both orientations: pm might have (away, home) or (home, away)
            key1 = self._game_key(pm.home_team, pm.away_team)
            key2 = self._game_key(pm.away_team, pm.home_team)

            if key1 in self.games:
                # pm.home_team is token[0], and it IS our home team
                self.games[key1].poly_market = pm
                self.games[key1].game_time_utc = pm.game_start_time
                matched += 1
                _debug(f"MATCH: {key1} → token0={pm.home_team} (home)")
            elif key2 in self.games:
                # pm's team order is flipped — swap token IDs so home_token = home team
                pm_fixed = PolyMarket(
                    home_team=pm.away_team,
                    away_team=pm.home_team,
                    home_token_id=pm.away_token_id,
                    away_token_id=pm.home_token_id,
                    game_start_time=pm.game_start_time,
                    condition_id=pm.condition_id,
                    question=pm.question,
                    volume=pm.volume,
                    gamma_best_bid=1 - pm.gamma_best_ask if pm.gamma_best_ask else None,
                    gamma_best_ask=1 - pm.gamma_best_bid if pm.gamma_best_bid else None,
                    neg_risk=pm.neg_risk,
                    accepting_orders=pm.accepting_orders,
                )
                self.games[key2].poly_market = pm_fixed
                self.games[key2].game_time_utc = pm.game_start_time
                matched += 1
                _debug(f"MATCH (swapped): {key2} → token1={pm.away_team} (home)")
            else:
                _debug(f"UNMATCHED: {pm.home_team} vs {pm.away_team} (no model pick)")

        print(f"  Matched {matched}/{len(self.games)} games to Polymarket markets")
        return matched

    # ── Price & Sizing Updates ─────────────────────────────────────────

    def update_prices(self):
        """Fetch live CLOB prices for all matched games.

        Uses the CLOB /book endpoint with proper bid/ask sorting, plus
        /midpoint as a cross-check (it aggregates complement token books).
        """
        with httpx.Client(timeout=15.0) as client:
            for key, gs in self.games.items():
                if not gs.poly_market:
                    continue

                home_token = gs.poly_market.home_token_id

                # Primary: fetch sorted orderbook
                book = fetch_orderbook(client, home_token)
                if book.get("error"):
                    _debug(f"Book error for {key}: {book['error']}")
                    continue

                # Secondary: CLOB /midpoint as cross-check
                clob_mid = fetch_clob_midpoint(client, home_token)

                # Use book top-of-book if available, else fall back to CLOB midpoint
                book_bid = book["best_bid"]
                book_ask = book["best_ask"]
                book_mid = book["midpoint"]

                # If book mid is wildly off from CLOB mid, prefer CLOB mid
                # (CLOB /midpoint aggregates across complement books)
                if book_mid and clob_mid and abs(book_mid - clob_mid) > 0.05:
                    _debug(f"{key}: book mid {book_mid:.3f} differs from CLOB mid {clob_mid:.3f}, using CLOB")
                    gs.poly_home_mid = clob_mid
                    # Estimate bid/ask from CLOB mid
                    gs.poly_home_bid = clob_mid - 0.005
                    gs.poly_home_ask = clob_mid + 0.005
                elif book_mid:
                    gs.poly_home_bid = book_bid
                    gs.poly_home_ask = book_ask
                    gs.poly_home_mid = book_mid
                elif clob_mid:
                    gs.poly_home_mid = clob_mid
                    gs.poly_home_bid = clob_mid - 0.005
                    gs.poly_home_ask = clob_mid + 0.005
                else:
                    # Last resort: Gamma hints
                    gamma_bid = gs.poly_market.gamma_best_bid
                    gamma_ask = gs.poly_market.gamma_best_ask
                    if gamma_bid and gamma_ask:
                        gs.poly_home_bid = gamma_bid
                        gs.poly_home_ask = gamma_ask
                        gs.poly_home_mid = (gamma_bid + gamma_ask) / 2
                        _debug(f"{key}: using Gamma hints bid={gamma_bid} ask={gamma_ask}")

                gs.poly_spread = book.get("spread")
                gs.poly_bid_depth = book.get("bid_depth", 0)
                gs.poly_ask_depth = book.get("ask_depth", 0)

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

        Risk checks (in order of priority):
          1. quote_halted (SP change, game started, pre-pitch pull)
          2. is_vol_spiking (WebSocket volume spike detected)
          3. is_adverse_halted (consecutive same-side fills)
          4. pre_pitch_pulled (< PULL_QUOTES_MINS before pitch)
        """
        if not gs.poly_market or gs.quote_halted:
            return None
        if gs.is_vol_spiking:
            return None
        if gs.is_adverse_halted:
            return None
        if gs.pre_pitch_pulled:
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

    async def _poll_order_status(self):
        """Poll CLOB API for order status on all active orders.

        Detects fills by comparing size_matched to previously recorded fill amount.
        Updates position, P&L, and adverse selection tracking from real fill data.
        """
        if self.config.dry_run or not self._clob_client:
            return

        for key, gs in self.games.items():
            for side_label, order_attr, filled_attr, price_attr in [
                ("BID", "bid_order_id", "bid_filled", "bid_price"),
                ("ASK", "ask_order_id", "ask_filled", "ask_price"),
            ]:
                order_id = getattr(gs, order_attr)
                if not order_id:
                    continue

                try:
                    order = await asyncio.to_thread(
                        self._clob_client.get_order, order_id
                    )
                except Exception as e:
                    _debug(f"get_order failed for {key} {side_label}: {e}")
                    continue

                if not order:
                    continue

                # Parse fill amount from CLOB response
                # Polymarket returns: size_matched (filled), original_size, status
                size_matched = float(order.get("size_matched", 0) or 0)
                prev_filled = getattr(gs, filled_attr)
                new_fill = size_matched - prev_filled

                if new_fill > 0.01:  # meaningful fill
                    fill_price = float(order.get("price", 0) or getattr(gs, price_attr) or 0)
                    setattr(gs, filled_attr, size_matched)

                    # Update position and P&L
                    if side_label == "BID":
                        # Bought home tokens
                        gs.net_position += new_fill
                        gs.total_bought += new_fill
                        gs.cost_basis += new_fill * fill_price
                        gs.consecutive_bid_fills += 1
                        gs.consecutive_ask_fills = 0
                    else:
                        # Sold home tokens
                        gs.net_position -= new_fill
                        gs.total_sold += new_fill
                        gs.cost_basis -= new_fill * fill_price
                        gs.consecutive_ask_fills += 1
                        gs.consecutive_bid_fills = 0

                    gs.fill_history.append({
                        "side": side_label,
                        "price": fill_price,
                        "size": new_fill,
                        "ts": time.time(),
                    })

                    print(f"  [FILL] {key} {side_label}: {new_fill:.1f} @ {fill_price:.2f} "
                          f"| pos={gs.net_position:+.1f} | bought={gs.total_bought:.1f} sold={gs.total_sold:.1f}")

                # Handle fully filled or cancelled orders — clear the order ID
                status = str(order.get("status", "")).upper()
                original_size = float(order.get("original_size", order.get("size", 0)) or 0)
                if status in ("MATCHED", "CANCELLED", "EXPIRED") or (
                    original_size > 0 and size_matched >= original_size - 0.01
                ):
                    setattr(gs, order_attr, None)
                    setattr(gs, filled_attr, 0.0)

            # Adverse selection check: consecutive same-side fills
            if (gs.consecutive_bid_fills >= MAX_CONSECUTIVE_FILLS or
                    gs.consecutive_ask_fills >= MAX_CONSECUTIVE_FILLS):
                side = "BID" if gs.consecutive_bid_fills >= MAX_CONSECUTIVE_FILLS else "ASK"
                gs.adverse_halted = True
                gs.adverse_halt_until = time.time() + 30
                gs.consecutive_bid_fills = 0
                gs.consecutive_ask_fills = 0
                print(f"\n  *** [RISK] ADVERSE SELECTION: {key} — "
                      f"{MAX_CONSECUTIVE_FILLS}x consecutive {side} fills → "
                      f"halting 30s to reprice")

    def _compute_unrealized_pnl(self, gs: GameState) -> float:
        """Compute unrealized P&L for current position using mid price."""
        if gs.net_position == 0 or gs.poly_home_mid is None:
            return 0.0
        # Mark-to-market: position * mid - cost_basis
        mark = gs.net_position * gs.poly_home_mid
        return mark - gs.cost_basis

    def _check_kill_switch(self) -> bool:
        """Check if total P&L has breached max-loss threshold.

        Returns True if bot should halt all trading.
        """
        total_pnl = 0.0
        for gs in self.games.values():
            total_pnl += gs.realized_pnl + self._compute_unrealized_pnl(gs)

        if total_pnl < -self.config.max_loss:
            print(f"\n  ******* KILL SWITCH: Total P&L ${total_pnl:+,.2f} "
                  f"breached max loss -${self.config.max_loss:,.0f} *******")
            print(f"  ******* HALTING ALL GAMES *******")
            for gs in self.games.values():
                gs.quote_halted = True
                gs.halt_reason = "KILL SWITCH"
            self._killed = True
            return True
        return False

    async def place_or_update_quotes(self):
        """Place or update two-sided quotes for all active games."""
        # Poll real order status and detect fills
        await self._poll_order_status()

        # Check kill switch
        if getattr(self, "_killed", False):
            return
        if self._check_kill_switch():
            return

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

            neg_risk = gs.poly_market.neg_risk

            # Bid: buy home token at bid_price
            if not gs.bid_order_id:
                result = await self._place_order(
                    home_token, "BUY", bid_price, bid_size, key, "BID", neg_risk
                )
                if result:
                    gs.bid_order_id = result.get("orderID") or result.get("id", "")
                    gs.bid_price = bid_price
                    gs.bid_size = bid_size

            # Ask: sell home token at ask_price
            if not gs.ask_order_id:
                result = await self._place_order(
                    home_token, "SELL", ask_price, ask_size, key, "ASK", neg_risk
                )
                if result:
                    gs.ask_order_id = result.get("orderID") or result.get("id", "")
                    gs.ask_price = ask_price
                    gs.ask_size = ask_size

    async def _place_order(
        self, token_id: str, side: str, price: float, size: float,
        game_key: str, label: str, neg_risk: bool = False,
    ) -> dict | None:
        """Place a single order via py-clob-client or log in dry-run mode."""
        if self.config.dry_run:
            print(f"    [DRY RUN] {game_key} {label}: {side} {size:.0f} @ {price:.2f}")
            return {"orderID": f"dry_{game_key}_{label}_{time.time():.0f}"}

        if self._clob_client:
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
                print(f"    ORDER: {game_key} {label}: {side} {size:.0f} @ {price:.2f} -> {order_id}")
                return result
            except Exception as e:
                print(f"    ERROR: {game_key} {label}: {e}")
                return None
        return None

    async def _cancel_game_orders(self, gs: GameState):
        """Cancel all open orders for a game."""
        for attr in ("bid_order_id", "ask_order_id"):
            order_id = getattr(gs, attr)
            if not order_id or order_id.startswith("dry_"):
                continue
            if not self.config.dry_run and self._clob_client:
                try:
                    await asyncio.to_thread(
                        self._clob_client.cancel, order_id
                    )
                except Exception as e:
                    _debug(f"Cancel failed for {order_id}: {e}")
            setattr(gs, attr, None)

        gs.bid_price = None
        gs.ask_price = None
        gs.bid_filled = 0.0
        gs.ask_filled = 0.0

    # ── Display ────────────────────────────────────────────────────────

    def print_dashboard(self):
        """Print current state of all tracked games."""
        now = datetime.now(timezone.utc)
        now_et = now - timedelta(hours=4)

        print(f"\n{'='*125}")
        print(f"  MLB Market Maker — {self.config.target_date} — "
              f"Cycle #{self._cycle_count} — {now_et.strftime('%I:%M:%S%p')} ET — "
              f"{'DRY RUN' if self.config.dry_run else 'LIVE'}")
        print(f"  Bankroll: ${self.config.bankroll:,.0f} | Min edge: {self.config.min_edge:.0%} | "
              f"Poll: {self.config.poll_interval_s:.0f}s")
        print(f"{'='*125}")

        print(f"\n  {'Game':<14s} {'Time':>9s} {'Model':>6s} {'Poly':>6s} {'Sprd':>5s} {'Edge':>6s} "
              f"{'½K%':>5s} {'Side':>6s} {'Conv':>7s} {'LU':>3s} "
              f"{'Bid':>6s} {'Ask':>6s} {'$BidDp':>7s} {'$AskDp':>7s} {'Risk':>8s} {'Status':>12s}")
        print(f"  {'-'*122}")

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
            sprd_str = f"{gs.poly_spread:.2f}" if gs.poly_spread is not None else " ---"
            edge_str = f"{gs.edge:+.1%}" if gs.poly_home_mid else "  ---"
            kelly_str = f"{gs.half_kelly:.1%}" if gs.half_kelly > 0 else "  ---"
            side_str = gs.side.replace("BUY_", "") if gs.side else "---"
            conv_str = gs.conviction.upper() if gs.conviction in ("strong", "lean") else gs.conviction
            lineup_str = "Y" if gs.lineups_confirmed else ("P" if gs.home_lineup_confirmed or gs.away_lineup_confirmed else "N")
            bid_str = f"{gs.bid_price:.2f}" if gs.bid_price else "  ---"
            ask_str = f"{gs.ask_price:.2f}" if gs.ask_price else "  ---"
            bd_str = f"${gs.poly_bid_depth:,.0f}" if gs.poly_bid_depth else "  ---"
            ad_str = f"${gs.poly_ask_depth:,.0f}" if gs.poly_ask_depth else "  ---"

            # Risk flags
            risk_flags = []
            if gs.is_vol_spiking:
                risk_flags.append("VOL")
            if gs.is_adverse_halted:
                risk_flags.append("ADV")
            if gs.spread_widened and not gs.pre_pitch_pulled:
                risk_flags.append("WIDE")
            if gs.pre_pitch_pulled:
                risk_flags.append("PULL")
            mins = gs.minutes_to_first_pitch
            if mins is not None and mins <= 60:
                risk_flags.append(f"{mins:.0f}m")
            risk_str = "|".join(risk_flags) if risk_flags else "OK"

            if gs.quote_halted:
                status = f"HALT:{gs.halt_reason[:9]}"
            elif gs.is_vol_spiking:
                status = "VOL_SPIKE"
            elif gs.is_adverse_halted:
                status = "ADVERSE"
            elif gs.bid_order_id or gs.ask_order_id:
                status = "QUOTING"
                active_quotes += 1
                total_exposure += gs.half_kelly * self.config.bankroll
            elif gs.poly_market is None:
                status = "NO_MKT"
            elif gs.poly_home_mid is None:
                status = "NO_BOOK"
            elif abs(gs.edge) < self.config.min_edge:
                status = "THIN_EDGE"
            else:
                status = "READY"

            print(f"  {game:<14s} {time_str:>9s} {model_str:>6s} {poly_str:>6s} {sprd_str:>5s} {edge_str:>6s} "
                  f"{kelly_str:>5s} {side_str:>6s} {conv_str:>7s} {lineup_str:>3s} "
                  f"{bid_str:>6s} {ask_str:>6s} {bd_str:>7s} {ad_str:>7s} {risk_str:>8s} {status:>12s}")

        # P&L summary
        total_realized = sum(g.realized_pnl for g in self.games.values())
        total_unrealized = sum(self._compute_unrealized_pnl(g) for g in self.games.values())
        total_pnl = total_realized + total_unrealized
        total_bought = sum(g.total_bought for g in self.games.values())
        total_sold = sum(g.total_sold for g in self.games.values())

        # Count active risk events
        vol_spikes = sum(1 for g in self.games.values() if g.is_vol_spiking)
        adverse = sum(1 for g in self.games.values() if g.is_adverse_halted)
        widened = sum(1 for g in self.games.values() if g.spread_widened and not g.pre_pitch_pulled)
        pulled = sum(1 for g in self.games.values() if g.pre_pitch_pulled)

        print(f"\n  Active quotes: {active_quotes} | "
              f"Total exposure: ${total_exposure:,.0f} / ${self.config.bankroll:,.0f}")
        print(f"  P&L: ${total_pnl:+,.2f} (realized ${total_realized:+,.2f} + "
              f"unrealized ${total_unrealized:+,.2f}) | "
              f"Fills: {total_bought:.0f} bought, {total_sold:.0f} sold | "
              f"Kill: -${self.config.max_loss:,.0f}")
        print(f"  Risk: {vol_spikes} vol spikes | {adverse} adverse | "
              f"{widened} widened | {pulled} pre-pitch pulled")

    # ── Main Loop ──────────────────────────────────────────────────────

    async def run(self):
        """Main bot loop with concurrent risk monitors.

        Launches three background async tasks:
          - TradeFlowMonitor: WebSocket volume spike detection
          - FastMLBMonitor: 8s SP/lineup polling
          - PrePitchWindDown: time-based spread widening & quote pulling
        """
        print(f"\n{'='*70}")
        print(f"  MLB Polymarket Market Maker")
        print(f"  Date: {self.config.target_date}")
        print(f"  Mode: {'DRY RUN' if self.config.dry_run else 'LIVE TRADING'}")
        print(f"  Bankroll: ${self.config.bankroll:,.0f}")
        print(f"  Kill switch: -${self.config.max_loss:,.0f}")
        print(f"{'='*70}")

        # Step 1: Load model picks
        print("\n[1/4] Loading model picks...")
        n = self.load_model_picks()
        if n == 0:
            return

        # Step 2: Discover Polymarket markets
        print("\n[2/4] Discovering Polymarket markets...")
        poly_markets = fetch_poly_mlb_markets(self.config.target_date)
        self.match_poly_markets(poly_markets)

        # Step 3: Initial price fetch & MLB status
        print("\n[3/4] Fetching initial prices & MLB status...")
        self.update_prices()
        self.update_mlb_status()
        self.print_dashboard()

        # Step 4: Initialize trading client
        print("\n[4/4] Initializing trading systems...")
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

        # Launch concurrent risk monitors
        self._trade_flow = TradeFlowMonitor(self.games)
        self._mlb_monitor = FastMLBMonitor(
            self.games, self.config.target_date,
            cancel_callback=self._cancel_game_orders,
        )
        self._wind_down = PrePitchWindDown(self.games)

        monitor_tasks = []
        monitor_tasks.append(asyncio.create_task(self._trade_flow.run()))
        monitor_tasks.append(asyncio.create_task(self._mlb_monitor.run()))
        monitor_tasks.append(asyncio.create_task(self._wind_down.run()))

        # Main loop
        self._running = True
        print(f"\n  Starting main loop (poll every {self.config.poll_interval_s}s, Ctrl+C to stop)...")
        print(f"  Risk monitors: WebSocket flow | MLB {MLB_FAST_POLL_S}s poll | Pre-pitch wind-down")

        try:
            while self._running and not self._killed:
                self._cycle_count += 1

                # Update CLOB prices (MLB monitoring is handled by FastMLBMonitor)
                self.update_prices()

                # Place/update quotes (polls order status, checks kill switch)
                await self.place_or_update_quotes()

                # Display
                self.print_dashboard()

                # If killed, break after displaying final state
                if self._killed:
                    break

                # Auto-shutdown: if every game is done or in-progress, exit
                all_done = all(
                    gs.quote_halted or gs.poly_market is None
                    for gs in self.games.values()
                )
                if all_done and self._cycle_count > 1:
                    print("\n  All games started or halted. Shutting down.")
                    break

                # Wait
                await asyncio.sleep(self.config.poll_interval_s)

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n\n  Shutting down...")
        finally:
            # Stop all risk monitors
            self._trade_flow.stop()
            self._mlb_monitor.stop()
            self._wind_down.stop()
            for t in monitor_tasks:
                t.cancel()
            await asyncio.gather(*monitor_tasks, return_exceptions=True)

            # Cancel all open orders
            print("  Cancelling all open orders...")
            for gs in self.games.values():
                await self._cancel_game_orders(gs)

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

    # Build lookup by team pair (both orientations)
    poly_lookup: dict[tuple, PolyMarket] = {}
    for pm in poly_markets:
        poly_lookup[(pm.home_team, pm.away_team)] = pm
        poly_lookup[(pm.away_team, pm.home_team)] = pm

    # Fetch orderbooks and compute sizing
    print("Fetching CLOB orderbooks...\n")
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

            # Primary: CLOB orderbook (properly sorted)
            book = fetch_orderbook(client, home_token)

            # Secondary: CLOB /midpoint (aggregates complement book)
            clob_mid = fetch_clob_midpoint(client, home_token)

            # Determine best mid price
            book_mid = book.get("midpoint")
            if book_mid and clob_mid and abs(book_mid - clob_mid) > 0.05:
                mid = clob_mid  # prefer CLOB mid if book is stale
                _debug(f"{at}@{ht}: book_mid={book_mid:.3f} != clob_mid={clob_mid:.3f}, using CLOB")
            elif book_mid:
                mid = book_mid
            elif clob_mid:
                mid = clob_mid
            elif pm.gamma_best_bid and pm.gamma_best_ask:
                # Last resort: Gamma hints
                if pm.home_team == ht:
                    mid = (pm.gamma_best_bid + pm.gamma_best_ask) / 2
                else:
                    mid = 1 - (pm.gamma_best_bid + pm.gamma_best_ask) / 2
                _debug(f"{at}@{ht}: using Gamma hint mid={mid:.3f}")
            else:
                mid = None

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
          f"{'$Size':>7s} {'$BidDp':>7s} {'$AskDp':>7s}")
    print(f"  {'-'*125}")

    total_size = 0.0
    for r in rows:
        gt = r.get("game_time", "")
        if gt:
            try:
                gt_clean = gt.replace("Z", "+00:00") if gt.endswith("Z") else gt
                if " " in gt_clean and "T" not in gt_clean:
                    gt_clean = gt_clean.replace(" ", "T", 1)
                if gt_clean.endswith("+00"):
                    gt_clean += ":00"
                dt = datetime.fromisoformat(gt_clean)
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
              f"{size_str:>7s} {bd_str:>7s} {ad_str:>7s}")

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
    parser.add_argument("--max-loss", type=float, default=500.0,
                        help="Kill switch: halt all trading if total P&L drops below -$X (default: 500)")
    parser.add_argument("--debug-discovery", action="store_true",
                        help="Print full discovery chain (sports, events, books)")
    args = parser.parse_args()

    global DEBUG_DISCOVERY
    DEBUG_DISCOVERY = args.debug_discovery

    if args.sizing_only:
        print_sizing_table(args.date, args.bankroll)
        return

    # Validate env vars for live trading
    if not args.dry_run:
        required = ["POLYMARKET_PRIVATE_KEY", "POLYMARKET_API_KEY",
                     "POLYMARKET_API_SECRET", "POLYMARKET_API_PASSPHRASE"]
        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            print(f"ERROR: Missing env vars for live trading: {', '.join(missing)}")
            print("Set these or use --dry-run")
            sys.exit(1)

    config = BotConfig(
        target_date=args.date,
        bankroll=args.bankroll,
        min_edge=args.min_edge,
        poll_interval_s=args.interval,
        dry_run=args.dry_run,
        debug_discovery=args.debug_discovery,
        max_loss=args.max_loss,
    )

    bot = MLBMarketMaker(config)

    # Handle Ctrl+C gracefully
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler(sig, frame):
        bot.stop()
        # Also stop risk monitors if they exist
        for attr in ("_trade_flow", "_mlb_monitor", "_wind_down"):
            monitor = getattr(bot, attr, None)
            if monitor:
                monitor.stop()

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        loop.run_until_complete(bot.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
