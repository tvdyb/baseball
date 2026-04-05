#!/usr/bin/env python3
"""
Backtest MC simulator vs Kalshi prediction-market prices.

Two modes:
  1. **Pregame** — run simulator before first pitch, compare to Kalshi's
     pre-game closing price.
  2. **In-game** — reconstruct game states at each half-inning boundary
     from the MLB play-by-play feed, run simulator from those states,
     compare to Kalshi candlestick prices at the matching timestamps.

Usage:
    python src/backtest_vs_kalshi.py --season 2025                     # pregame only
    python src/backtest_vs_kalshi.py --season 2025 --ingame            # pregame + in-game
    python src/backtest_vs_kalshi.py --season 2025 --ingame --max-games 50
    python src/backtest_vs_kalshi.py --season 2025 --pregame-only      # skip in-game
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# torch/multi-output model must be imported BEFORE sklearn to avoid segfault
from multi_output_matchup_model import load_multi_output_models
from simulate import (
    GameState,
    SimConfig,
    load_sim_artifacts,
    load_simulation_context,
    monte_carlo_win_prob,
    ensure_batter_index,
)

from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

from build_transition_matrix import OUTCOME_ORDER
from feature_engineering import _preindex_xrv, _load_hand_models
from predict import MLB_API
from scrape_kalshi import (
    TEAM_MAP as KALSHI_TEAM_MAP,
    parse_event_ticker,
    _find_pregame_price,
    _extract_candle_close,
    _fetch_candles_with_retry,
    API_BASE as KALSHI_API_BASE,
)
from utils import DATA_DIR, XRV_DIR, MODEL_DIR

KALSHI_DIR = DATA_DIR / "kalshi"
GAMES_DIR = DATA_DIR / "games"
AUDIT_DIR = DATA_DIR / "audit"


def log(msg: str):
    print(msg)
    sys.stdout.flush()


# ── Play-by-Play Parsing ─────────────────────────────────────────────────────


@dataclass
class StateSnapshot:
    """Game state at a point in time, plus metadata for matching to Kalshi."""
    timestamp: float           # Unix timestamp
    game_state: GameState
    description: str           # e.g., "Top 5 start" or "Bot 7 end"
    inning_half: str           # e.g., "Top 5", "Bot 7"


def fetch_play_by_play(client: httpx.Client, game_pk: int, retries: int = 3) -> dict | None:
    """Fetch complete play-by-play data for a historical game from the MLB API."""
    import time as _time
    for attempt in range(retries):
        try:
            resp = client.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
                timeout=30.0,
            )
            if resp.status_code == 404:
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                _time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.ConnectError):
            if attempt < retries - 1:
                _time.sleep(2 ** attempt)
                continue
            return None
        except httpx.HTTPStatusError:
            return None
    return None


def extract_half_inning_states(
    pbp_data: dict,
    home_lineup: list[tuple[int, str]] | None = None,
    away_lineup: list[tuple[int, str]] | None = None,
) -> list[StateSnapshot]:
    """Parse MLB play-by-play data into game states at each half-inning boundary.

    Returns a StateSnapshot at the START of each half-inning (state entering
    that half-inning), which is the natural point for running the simulator
    since we're simulating forward from that state.
    """
    all_plays = pbp_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    if not all_plays:
        return []

    # Build lineup position tracking
    home_batting_order = (
        pbp_data.get("liveData", {})
        .get("boxscore", {})
        .get("teams", {})
        .get("home", {})
        .get("battingOrder", [])
    )
    away_batting_order = (
        pbp_data.get("liveData", {})
        .get("boxscore", {})
        .get("teams", {})
        .get("away", {})
        .get("battingOrder", [])
    )

    snapshots = []

    # Track state across plays
    home_score = 0
    away_score = 0
    home_pitchers_used = set()
    away_pitchers_used = set()

    # Group plays by half-inning
    half_innings = defaultdict(list)
    for play in all_plays:
        about = play.get("about", {})
        inning = about.get("inning", 0)
        is_top = about.get("isTopInning", True)
        key = (inning, "Top" if is_top else "Bot")
        half_innings[key].append(play)

    # Sort half-innings in order
    sorted_keys = sorted(half_innings.keys(), key=lambda k: (k[0], 0 if k[1] == "Top" else 1))

    # Track cumulative state between half-innings
    home_lineup_pos = 0
    away_lineup_pos = 0

    for key_idx, (inning, top_bot) in enumerate(sorted_keys):
        plays = half_innings[(inning, top_bot)]
        if not plays:
            continue

        # Get timestamp of first play in this half-inning
        first_play = plays[0]
        ts_str = first_play.get("about", {}).get("startTime", "")
        if not ts_str:
            continue

        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            continue

        # Determine pitcher state from historical data
        # If more than one pitcher has been used on that side, bullpen is active
        home_pitcher_state = "bp" if len(home_pitchers_used) > 1 else "sp"
        away_pitcher_state = "bp" if len(away_pitchers_used) > 1 else "sp"

        # The game state ENTERING this half-inning
        bases = (False, True, False) if inning >= 10 else (False, False, False)
        state = GameState(
            inning=inning,
            top_bottom=top_bot,
            outs=0,
            bases=bases,
            home_score=home_score,
            away_score=away_score,
            home_lineup_pos=home_lineup_pos % 9,
            away_lineup_pos=away_lineup_pos % 9,
            home_pitcher=home_pitcher_state,
            away_pitcher=away_pitcher_state,
        )

        snapshots.append(StateSnapshot(
            timestamp=ts,
            game_state=state,
            description=f"{top_bot} {inning} start",
            inning_half=f"{top_bot} {inning}",
        ))

        # Process plays in this half-inning to update cumulative state
        for play in plays:
            result = play.get("result", {})
            about = play.get("about", {})

            # Track pitchers
            matchup = play.get("matchup", {})
            pitcher_id = matchup.get("pitcher", {}).get("id")
            if pitcher_id:
                if top_bot == "Top":
                    home_pitchers_used.add(pitcher_id)
                else:
                    away_pitchers_used.add(pitcher_id)

            # Track runs
            rbi = result.get("rbi", 0)
            # More accurate: check runners who scored
            runners = play.get("runners", [])
            runs_on_play = sum(
                1 for r in runners
                if r.get("movement", {}).get("end") == "score"
            )

            if top_bot == "Top":
                away_score += runs_on_play
            else:
                home_score += runs_on_play

            # Track lineup advancement (count PA events)
            if result.get("type") in ("atBat",) and about.get("isComplete", False):
                if top_bot == "Top":
                    away_lineup_pos += 1
                else:
                    home_lineup_pos += 1

    return snapshots


# ── Kalshi Candlestick Matching ──────────────────────────────────────────────


def _extract_midpoint(candle: dict) -> float | None:
    """Extract mid price from yes_ask/yes_bid close prices."""
    ask_data = candle.get("yes_ask", {})
    bid_data = candle.get("yes_bid", {})
    ask_close = ask_data.get("close")
    bid_close = bid_data.get("close")
    if ask_close is not None and bid_close is not None:
        try:
            return (float(ask_close) + float(bid_close)) / 2
        except (ValueError, TypeError):
            pass
    # If only one side, use it
    for val in (ask_close, bid_close):
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def fetch_full_candle_series(
    client: httpx.Client,
    market_ticker: str,
    start_ts: int,
    end_ts: int,
) -> list[tuple[float, float]]:
    """Fetch complete candlestick timeseries for a Kalshi market.

    Returns list of (unix_timestamp, close_price) sorted by time.
    """
    # Try historical endpoint first, fall back to live
    candles = _fetch_candles_with_retry(
        client,
        f"{KALSHI_API_BASE}/historical/markets/{market_ticker}/candlesticks",
        {"period_interval": 1, "start_ts": start_ts, "end_ts": end_ts},
    )
    source = "historical"

    if candles is None:
        candles = _fetch_candles_with_retry(
            client,
            f"{KALSHI_API_BASE}/series/KXMLBGAME/markets/{market_ticker}/candlesticks",
            {"period_interval": 1, "start_ts": start_ts, "end_ts": end_ts},
        )
        source = "live"

    if not candles:
        return []

    series = []
    for c in candles:
        price = _extract_candle_close(c.get("price", {}))
        # Fall back to yes_ask/yes_bid midpoint if price.close is None
        if price is None:
            price = _extract_midpoint(c)
        if price is not None and 0.01 <= price <= 0.99:
            series.append((c["end_period_ts"], price))

    series.sort(key=lambda x: x[0])
    return series


def match_candle_to_timestamp(
    candles: list[tuple[float, float]],
    target_ts: float,
    max_gap_seconds: int = 1800,
) -> float | None:
    """Find the candle price closest to (but not after) a target timestamp.

    Returns the close price of the last candle ending before target_ts + max_gap.
    Returns None if no candle is close enough.
    """
    if not candles:
        return None

    # Find candles before or near the target timestamp
    best_price = None
    best_gap = float("inf")

    for ts, price in candles:
        gap = target_ts - ts
        if -max_gap_seconds <= gap <= max_gap_seconds:
            if abs(gap) < abs(best_gap):
                best_gap = gap
                best_price = price

    return best_price


# ── Shared Data Loading ─────────────────────────────────────────────────────


def load_backtest_data(season: int) -> tuple[dict, dict, dict] | None:
    """Load xRV index, matchup models, and lineups for a season.

    Returns (xrv_index, matchup_models, lineups) or None if data is missing.
    Shared by pregame and in-game backtests to avoid duplicate I/O.
    """
    # Load xRV data
    xrv_frames = []
    for yr in range(season - 1, season + 1):
        path = XRV_DIR / f"statcast_xrv_{yr}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "game_type" in df.columns:
                df = df[df["game_type"] == "R"]
            xrv_frames.append(df)
            log(f"  Loaded xRV {yr}: {len(df):,} pitches")
    if not xrv_frames:
        log("  No xRV data found")
        return None
    xrv = pd.concat(xrv_frames, ignore_index=True)
    idx = _preindex_xrv(xrv)

    # Load matchup models
    model_year = season - 1
    matchup_models = _load_hand_models("matchup_model", model_year)
    if not matchup_models:
        matchup_models = _load_hand_models("matchup_model", season)
    if not matchup_models:
        log("  WARNING: No matchup models found — using league-average xRV")

    # Load lineups from JSON cache
    lineups_path = GAMES_DIR / f"lineups_{season}.json"
    lineups = {}
    if lineups_path.exists():
        with open(lineups_path) as f:
            raw = json.load(f)
        for gpk, lu in raw.items():
            home_raw = lu.get("home", lu.get("home_lineup", []))
            away_raw = lu.get("away", lu.get("away_lineup", []))
            home_lu = [(p["player_id"], p.get("bat_side") or "R") for p in home_raw]
            away_lu = [(p["player_id"], p.get("bat_side") or "R") for p in away_raw]
            if home_lu and away_lu:
                lineups[int(gpk)] = {"home": home_lu, "away": away_lu}
        log(f"  Loaded lineups for {len(lineups)} games from cache")

    return idx, matchup_models or {}, lineups


# ── Pregame Backtest ─────────────────────────────────────────────────────────


def run_pregame_backtest(
    season: int,
    base_rates: dict,
    transition_matrix: dict,
    config: SimConfig,
    idx: dict | None = None,
    matchup_models: dict | None = None,
    lineups: dict | None = None,
    mo_models: dict | None = None,
) -> pd.DataFrame:
    """Run MC simulator pregame on all games with Kalshi data.

    Returns DataFrame with columns:
        game_pk, game_date, home_team, away_team, home_win,
        sim_home_wp, kalshi_home_prob, edge, sim_runs_mean, kalshi_implied_total
    """
    # Load Kalshi data
    kalshi_path = KALSHI_DIR / f"kalshi_mlb_{season}.parquet"
    if not kalshi_path.exists():
        log(f"  Kalshi data not found: {kalshi_path}")
        log(f"  Run: python src/scrape_kalshi.py --year {season}")
        return pd.DataFrame()

    kalshi = pd.read_parquet(kalshi_path)
    kalshi["game_date"] = kalshi["game_date"].astype(str)
    log(f"  Loaded {len(kalshi)} Kalshi-priced games")

    # Load games
    games_path = GAMES_DIR / f"games_{season}.parquet"
    if not games_path.exists():
        log(f"  Games file not found: {games_path}")
        return pd.DataFrame()

    games_df = pd.read_parquet(games_path)
    games_df["game_date"] = games_df["game_date"].astype(str)
    if "home_team_abbr" in games_df.columns and "home_team" not in games_df.columns:
        games_df["home_team"] = games_df["home_team_abbr"]
        games_df["away_team"] = games_df["away_team_abbr"]
    games_df = games_df[
        games_df["home_win"].notna()
        & games_df["home_sp_id"].notna()
        & games_df["away_sp_id"].notna()
    ].copy()

    merged = games_df.merge(
        kalshi[["game_date", "home_team", "away_team", "kalshi_home_prob", "volume"]],
        on=["game_date", "home_team", "away_team"],
        how="inner",
    )
    log(f"  {len(merged)} games with both results and Kalshi prices")

    if len(merged) == 0:
        return pd.DataFrame()

    # Fall back to loading data if not provided
    if idx is None or matchup_models is None or lineups is None:
        loaded = load_backtest_data(season)
        if loaded is None:
            return pd.DataFrame()
        idx, matchup_models, lineups = loaded

    # Load park factors and weather for sim adjustments
    from feature_engineering import compute_park_factors
    park_factors = compute_park_factors(merged, season)
    weather_map = {}
    weather_path = GAMES_DIR.parent / "weather" / f"weather_{season}.parquet"
    if weather_path.exists():
        import pandas as _pd
        wdf = _pd.read_parquet(weather_path)
        for _, wr in wdf.iterrows():
            wind_dir = str(wr.get("wind_dir", "")).lower()
            weather_map[int(wr["game_pk"])] = {
                "temperature": wr.get("temperature"),
                "wind_speed": wr.get("wind_speed"),
                "wind_out": int("out" in wind_dir),
                "wind_in": int("in" in wind_dir),
                "is_dome": wr.get("is_dome", 0),
            }

    # For games without cached lineups, fetch from MLB API
    from predict import fetch_lineup
    lineup_client = httpx.Client(timeout=30.0)
    n_fetched = 0

    # Run simulations
    rows = []
    n_skip = 0
    n_total = len(merged)

    for i, (_, row) in enumerate(merged.iterrows()):
        game = row.to_dict()
        gpk = int(game["game_pk"])
        lu = lineups.get(gpk)

        if lu is None or not lu.get("home") or not lu.get("away"):
            # Fetch from API
            lu = fetch_lineup(lineup_client, gpk)
            if lu.get("home") and lu.get("away"):
                lineups[gpk] = lu
                n_fetched += 1
                if n_fetched % 50 == 0:
                    log(f"    Fetched {n_fetched} lineups from API...")
                time.sleep(0.1)
            else:
                n_skip += 1
                continue

        try:
            home_ctx, away_ctx = load_simulation_context(
                game, str(game["game_date"]), lu, idx,
                matchup_models or {}, base_rates,
                mo_models=mo_models, config=config,
            )
            pf = park_factors.get(game.get("home_team", ""), 1.0)
            wx = weather_map.get(gpk)
            if wx and wx.get("is_dome"):
                wx = None

            result = monte_carlo_win_prob(
                home_ctx, away_ctx, GameState(),
                transition_matrix, config,
                park_factor=pf, weather=wx,
            )

            rows.append({
                "game_pk": gpk,
                "game_date": game["game_date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_win": int(game["home_win"]),
                "sim_home_wp": result["home_wp"],
                "kalshi_home_prob": game["kalshi_home_prob"],
                "edge": result["home_wp"] - game["kalshi_home_prob"],
                "sim_total_runs": result["total_runs_mean"],
                "sim_home_runs": result["home_runs_mean"],
                "sim_away_runs": result["away_runs_mean"],
                "kalshi_volume": game.get("volume", 0),
            })
        except Exception as e:
            n_skip += 1
            if n_skip <= 3:
                import traceback
                log(f"    ERROR game {gpk}: {e}")
                traceback.print_exc()
            continue

        if (i + 1) % 100 == 0 or i + 1 == n_total:
            log(f"    Pregame progress: {i + 1}/{n_total} ({len(rows)} done, {n_skip} skipped)")

    lineup_client.close()
    if n_fetched > 0:
        log(f"  Fetched {n_fetched} lineups from MLB API")
    log(f"  Pregame: {len(rows)} predictions, {n_skip} skipped")
    return pd.DataFrame(rows)


# ── In-Game Backtest ─────────────────────────────────────────────────────────


def _load_kalshi_market_tickers(season: int) -> dict[tuple[str, str, str], dict]:
    """Load Kalshi market tickers from the events we already scraped.

    Returns dict mapping (game_date, home_team, away_team) -> {
        "event_ticker": str,
        "home_ticker": str,
        "open_time": str,
        "close_time": str,
    }
    """
    # Re-fetch events from API to get market tickers
    # (The parquet only stores prices, not tickers)
    # For now, reconstruct from event ticker format
    kalshi_path = KALSHI_DIR / f"kalshi_mlb_{season}.parquet"
    if not kalshi_path.exists():
        return {}

    kalshi = pd.read_parquet(kalshi_path)
    result = {}
    for _, row in kalshi.iterrows():
        key = (str(row["game_date"]), row["home_team"], row["away_team"])
        if "event_ticker" in kalshi.columns:
            result[key] = {
                "event_ticker": row["event_ticker"],
            }
    return result


def run_ingame_backtest(
    season: int,
    base_rates: dict,
    transition_matrix: dict,
    config: SimConfig,
    max_games: int = 100,
    pregame_df: pd.DataFrame | None = None,
    idx: dict | None = None,
    matchup_models: dict | None = None,
    lineups: dict | None = None,
    mo_models: dict | None = None,
) -> pd.DataFrame:
    """Run in-game backtest: reconstruct states, match to Kalshi candles, simulate.

    For each game:
      1. Fetch play-by-play from MLB API
      2. Extract game states at half-inning boundaries
      3. Fetch Kalshi candlestick data for the game
      4. Match candle prices to state timestamps
      5. Run MC simulator from each state
      6. Compare simulator WP to Kalshi price

    Returns DataFrame with per-state-point rows.
    """
    # Load games with Kalshi data
    kalshi_path = KALSHI_DIR / f"kalshi_mlb_{season}.parquet"
    if not kalshi_path.exists():
        log(f"  Kalshi data not found: {kalshi_path}")
        return pd.DataFrame()

    kalshi = pd.read_parquet(kalshi_path)
    kalshi["game_date"] = kalshi["game_date"].astype(str)

    games_path = GAMES_DIR / f"games_{season}.parquet"
    if not games_path.exists():
        log(f"  Games file not found: {games_path}")
        return pd.DataFrame()

    games_df = pd.read_parquet(games_path)
    games_df["game_date"] = games_df["game_date"].astype(str)
    if "home_team_abbr" in games_df.columns and "home_team" not in games_df.columns:
        games_df["home_team"] = games_df["home_team_abbr"]
        games_df["away_team"] = games_df["away_team_abbr"]
    games_df = games_df[
        games_df["home_win"].notna()
        & games_df["home_sp_id"].notna()
        & games_df["away_sp_id"].notna()
    ].copy()

    merged = games_df.merge(
        kalshi[["game_date", "home_team", "away_team", "kalshi_home_prob", "event_ticker"]],
        on=["game_date", "home_team", "away_team"],
        how="inner",
    )

    if len(merged) == 0:
        log("  No games with Kalshi event tickers for in-game backtest")
        return pd.DataFrame()

    # Subsample if too many games
    if len(merged) > max_games:
        merged = merged.sample(n=max_games, random_state=42).sort_values("game_date")
        log(f"  Subsampled to {max_games} games for in-game backtest")

    # Fall back to loading data if not provided
    if idx is None or matchup_models is None or lineups is None:
        loaded = load_backtest_data(season)
        if loaded is None:
            return pd.DataFrame()
        idx, matchup_models, lineups = loaded

    lineups = dict(lineups)  # local copy so we can add fetched lineups

    rows = []
    n_games_processed = 0
    n_states_total = 0

    mlb_client = httpx.Client(timeout=30.0)
    kalshi_client = httpx.Client(timeout=30.0)

    try:
        for i, (_, row) in enumerate(merged.iterrows()):
            game = row.to_dict()
            gpk = int(game["game_pk"])
            event_ticker = game.get("event_ticker", "")

            lu = lineups.get(gpk)
            if lu is None or not lu.get("home") or not lu.get("away"):
                continue

            # Build simulation context (shared across all in-game states)
            try:
                home_ctx, away_ctx = load_simulation_context(
                    game, str(game["game_date"]), lu, idx,
                    matchup_models or {}, base_rates,
                    mo_models=mo_models, config=config,
                )
            except Exception as e:
                log(f"    SKIP game {gpk} (context): {type(e).__name__}: {e}")
                continue

            # Fetch play-by-play
            pbp = fetch_play_by_play(mlb_client, gpk)
            if pbp is None:
                continue

            # Extract half-inning states
            snapshots = extract_half_inning_states(pbp)
            if len(snapshots) < 2:
                continue

            # Reconstruct home market ticker from event ticker
            # Event ticker: KXMLBGAME-25JUL31ATLCIN
            # Home market ticker: KXMLBGAME-25JUL31ATLCIN-CIN (home team suffix)
            home_team_kalshi = game["home_team"]
            # Reverse lookup: find Kalshi abbreviation for this team
            reverse_map = {}
            for k, v in KALSHI_TEAM_MAP.items():
                if v not in reverse_map:
                    reverse_map[v] = k
            kalshi_abbr = reverse_map.get(home_team_kalshi, home_team_kalshi)
            home_ticker = f"{event_ticker}-{kalshi_abbr}"

            # Fetch Kalshi candle data for the game period
            # Use game_datetime for time range
            game_dt_str = game.get("game_datetime", "")
            if not game_dt_str:
                # Estimate from game_date — games typically run 18:00-23:00 UTC
                game_date = game["game_date"]
                start_ts = int(datetime.strptime(game_date, "%Y-%m-%d").replace(
                    hour=16, tzinfo=timezone.utc
                ).timestamp())
                end_ts = start_ts + 7 * 3600  # ~7 hours window
            else:
                try:
                    game_dt = datetime.fromisoformat(game_dt_str.replace("Z", "+00:00"))
                    start_ts = int(game_dt.timestamp()) - 3600  # 1hr before
                    end_ts = int(game_dt.timestamp()) + 6 * 3600  # 6hr after
                except (ValueError, TypeError):
                    continue

            candles = fetch_full_candle_series(
                kalshi_client, home_ticker, start_ts, end_ts,
            )

            if not candles:
                continue

            # Match each state snapshot to a Kalshi candle price
            for snap in snapshots:
                kalshi_price = match_candle_to_timestamp(candles, snap.timestamp)
                if kalshi_price is None:
                    continue

                # Run simulator from this state
                try:
                    result = monte_carlo_win_prob(
                        home_ctx, away_ctx, snap.game_state,
                        transition_matrix, config,
                    )
                except Exception as e:
                    log(f"    SKIP {gpk} {snap.inning_half}: {type(e).__name__}: {e}")
                    continue

                # Compute game progress (0.0 = pregame, 1.0 = game over)
                gs = snap.game_state
                half_innings_played = (gs.inning - 1) * 2 + (1 if gs.top_bottom == "Bot" else 0)
                game_progress = min(half_innings_played / 18.0, 1.0)

                rows.append({
                    "game_pk": gpk,
                    "game_date": game["game_date"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "home_win": int(game["home_win"]),
                    "inning": gs.inning,
                    "top_bottom": gs.top_bottom,
                    "inning_half": snap.inning_half,
                    "home_score": gs.home_score,
                    "away_score": gs.away_score,
                    "game_progress": game_progress,
                    "sim_home_wp": result["home_wp"],
                    "kalshi_home_prob": kalshi_price,
                    "edge": result["home_wp"] - kalshi_price,
                    "abs_edge": abs(result["home_wp"] - kalshi_price),
                    "timestamp": snap.timestamp,
                })
                n_states_total += 1

            n_games_processed += 1
            if (n_games_processed) % 10 == 0:
                log(f"    In-game progress: {n_games_processed} games, "
                    f"{n_states_total} state points")

            # Rate-limit API calls
            time.sleep(0.2)

    finally:
        mlb_client.close()
        kalshi_client.close()

    log(f"  In-game: {n_states_total} state points across {n_games_processed} games")
    return pd.DataFrame(rows)


# ── Ensemble Blend ───────────────────────────────────────────────────────────


def blend_with_ensemble(
    ingame_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Blend MC simulator probabilities with the pregame LR+XGB ensemble.

    The ensemble is a better team-strength model (beats the market pregame).
    The MC simulator is a better state-evolution model (handles score, outs,
    bases, matchups). By blending, we get the best of both:
      - Early innings: lean on ensemble (team strength dominates)
      - Late innings: lean on simulator (game state dominates)

    Weight schedule: ensemble weight decays linearly from 1.0 at inning 1
    to 0.0 at inning 9+. This is derived from the observation that the
    simulator only matches or beats the market from inning 7 onward.
    """
    preds_path = AUDIT_DIR / "walk_forward_predictions.csv"
    if not preds_path.exists():
        log(f"  WARNING: {preds_path} not found — cannot blend with ensemble")
        log("  Run `make train` or `python src/win_model.py --walk-forward` first")
        ingame_df["blended_wp"] = ingame_df["sim_home_wp"]
        ingame_df["ens_prob"] = np.nan
        ingame_df["ens_weight"] = 0.0
        return ingame_df

    preds = pd.read_csv(preds_path)

    # Use ens_prob (the LR+XGB blend); fall back to lr_prob if missing
    prob_col = "ens_prob" if "ens_prob" in preds.columns else "lr_prob"
    ens_lookup = dict(zip(preds["game_pk"].astype(int), preds[prob_col].astype(float)))
    log(f"  Loaded {len(ens_lookup)} pregame ensemble predictions from {prob_col}")

    # Match ensemble predictions to in-game rows
    ingame_df["ens_prob"] = ingame_df["game_pk"].map(ens_lookup)
    matched = ingame_df["ens_prob"].notna().sum()
    total = len(ingame_df)
    log(f"  Matched {matched}/{total} state points ({matched/total:.0%})")

    # Compute ensemble weight: decays with inning
    # Inning 1 → 1.0, inning 9+ → 0.0
    # NOTE: Backtesting shows this blend HURTS performance because the
    # pregame probability doesn't incorporate in-game score changes.
    # Included for research/comparison only.
    ingame_df["ens_weight"] = np.clip(1.0 - (ingame_df["inning"] - 1) / 8.0, 0.0, 1.0)

    # Blend: w * ensemble + (1-w) * simulator
    ens = ingame_df["ens_prob"].fillna(ingame_df["sim_home_wp"])  # fallback if no match
    sim = ingame_df["sim_home_wp"]
    w = ingame_df["ens_weight"]
    ingame_df["blended_wp"] = np.clip(w * ens + (1 - w) * sim, 0.001, 0.999)

    # Summary stats
    for inn in range(1, 10):
        mask = ingame_df["inning"] == inn
        if mask.sum() > 0:
            wt = ingame_df.loc[mask, "ens_weight"].iloc[0]
            log(f"    Inning {inn}: ens_weight={wt:.2f}, "
                f"n={mask.sum()}")

    return ingame_df


# ── Metrics & Reporting ──────────────────────────────────────────────────────


def compute_metrics(
    df: pd.DataFrame,
    sim_col: str = "sim_home_wp",
    market_col: str = "kalshi_home_prob",
    label: str = "",
) -> dict:
    """Compute comparison metrics between simulator and market prices."""
    required = [sim_col, market_col, "home_win"]
    if not all(c in df.columns for c in required) or len(df) == 0:
        return {}
    df = df.dropna(subset=required)
    if len(df) == 0:
        return {}

    y = df["home_win"].values
    sim = np.clip(df[sim_col].values, 0.01, 0.99)
    market = np.clip(df[market_col].values, 0.01, 0.99)
    baseline = np.full_like(sim, y.mean())

    sim_ll = log_loss(y, sim)
    market_ll = log_loss(y, market)
    baseline_ll = log_loss(y, baseline)
    sim_bs = brier_score_loss(y, sim)
    market_bs = brier_score_loss(y, market)

    try:
        sim_auc = roc_auc_score(y, sim)
        market_auc = roc_auc_score(y, market)
    except ValueError:
        sim_auc = market_auc = 0.5

    edge = sim - market
    sim_acc = ((sim > 0.5) == y).mean()
    market_acc = ((market > 0.5) == y).mean()

    return {
        "label": label,
        "n": len(df),
        "sim_log_loss": sim_ll,
        "market_log_loss": market_ll,
        "baseline_log_loss": baseline_ll,
        "sim_brier": sim_bs,
        "market_brier": market_bs,
        "sim_auc": sim_auc,
        "market_auc": market_auc,
        "sim_accuracy": sim_acc,
        "market_accuracy": market_acc,
        "mean_edge": edge.mean(),
        "std_edge": edge.std(),
        "mae_vs_market": np.abs(edge).mean(),
        "home_win_rate": y.mean(),
        "sim_mean_pred": sim.mean(),
        "market_mean_pred": market.mean(),
    }


def compute_roi(
    df: pd.DataFrame,
    sim_col: str = "sim_home_wp",
    market_col: str = "kalshi_home_prob",
    edge_thresholds: list[float] = None,
) -> list[dict]:
    """Compute ROI from flat-bet strategy: bet when |sim - market| > threshold."""
    if edge_thresholds is None:
        edge_thresholds = [0.03, 0.05, 0.07, 0.10]

    df = df.dropna(subset=[sim_col, market_col, "home_win"])
    y = df["home_win"].values
    sim = np.clip(df[sim_col].values, 0.01, 0.99)
    market = np.clip(df[market_col].values, 0.01, 0.99)
    edge = sim - market

    results = []
    for threshold in edge_thresholds:
        pnls = []
        for i in range(len(df)):
            if abs(edge[i]) < threshold:
                continue
            # Bet direction: if edge > 0, bet home; if edge < 0, bet away
            if edge[i] > 0:
                cost = market[i] * 100
                pnl = (100 - cost) if y[i] == 1 else -cost
            else:
                cost = (1 - market[i]) * 100
                pnl = (100 - cost) if y[i] == 0 else -cost
            pnls.append(pnl)

        if not pnls:
            results.append({"threshold": threshold, "n_bets": 0})
            continue

        pnls = np.array(pnls)
        total_pnl = pnls.sum()
        n_bets = len(pnls)
        roi = total_pnl / (n_bets * 100)
        wins = (pnls > 0).sum()

        # Sharpe
        if n_bets > 1 and pnls.std() > 0:
            if "game_date" in df.columns:
                dates = pd.to_datetime(df["game_date"])
                days = (dates.max() - dates.min()).days
                bpy = n_bets * (365.0 / max(days, 1))
            else:
                bpy = n_bets
            sharpe = (pnls.mean() / pnls.std()) * np.sqrt(bpy)
        else:
            sharpe = 0.0

        results.append({
            "threshold": threshold,
            "n_bets": n_bets,
            "total_pnl": total_pnl,
            "roi": roi,
            "sharpe": sharpe,
            "win_rate": wins / n_bets,
        })

    return results


def _polymarket_taker_fee(n_contracts: float, price: float, fee_rate: float = 0.03) -> float:
    """Polymarket taker fee: C × feeRate × p × (1-p).

    Sports category = 3%. Fee is symmetric around 50% and approaches
    zero at extreme prices (0% or 100%).
    """
    return abs(n_contracts) * fee_rate * price * (1 - price)


def compute_kelly_rebalancing(
    ingame_df: pd.DataFrame,
    kelly_fractions: list[float] = None,
    min_edges: list[float] = None,
    bankroll: float = 10_000.0,
    max_position_pct: float = 0.10,
    taker_fee_rate: float = 0.03,
    sim_col: str = "sim_home_wp",
) -> list[dict]:
    """Simulate Kelly rebalancing strategy on in-game backtest data.

    Includes Polymarket taker fees: C × feeRate × p × (1-p).

    For each game, at every half-inning boundary:
      1. Compute Kelly-optimal position from model WP vs market price
      2. Rebalance to target position (buy/sell contracts at market price)
      3. At game end, contracts settle at $1 (home wins) or $0

    This compounds small edges across ~18 half-innings per game.

    Returns list of result dicts, one per (kelly_fraction, min_edge) combo.
    """
    if kelly_fractions is None:
        kelly_fractions = [0.25, 0.50, 1.00]
    if min_edges is None:
        min_edges = [0.03, 0.05, 0.07]

    if len(ingame_df) == 0:
        return []

    # Group by game
    games = ingame_df.groupby("game_pk")
    max_exposure = bankroll * max_position_pct

    results = []
    for kf in kelly_fractions:
        for me in min_edges:
            game_pnls = []

            game_dates_list = []
            for gpk, game_df in games:
                game_df = game_df.sort_values("timestamp")
                home_win = game_df.iloc[0]["home_win"]
                if "game_date" in game_df.columns:
                    game_dates_list.append(game_df.iloc[0]["game_date"])

                position = 0.0      # contracts held (positive = home YES)
                cash_spent = 0.0    # total cash invested

                for _, row in game_df.iterrows():
                    sim_wp = row[sim_col]
                    mkt_price = row["kalshi_home_prob"]

                    edge = sim_wp - mkt_price
                    abs_edge = abs(edge)

                    if abs_edge < me:
                        target = 0.0
                    elif edge > 0:
                        # Buy home YES: Kelly = edge / (1 - price)
                        full_kelly = edge / (1 - mkt_price) if mkt_price < 1 else 0
                        dollar_size = min(bankroll * full_kelly * kf, max_exposure)
                        target = dollar_size / mkt_price  # contracts
                        # Cap notional (settlement) exposure too
                        target = min(target, max_exposure)
                    else:
                        # Short home YES: Kelly = |edge| / price
                        full_kelly = abs_edge / mkt_price if mkt_price > 0 else 0
                        dollar_size = min(bankroll * full_kelly * kf, max_exposure)
                        target = -dollar_size / (1 - mkt_price)
                        # Cap notional (settlement) exposure too
                        target = max(target, -max_exposure)

                    # Rebalance: buy/sell to reach target
                    delta = target - position
                    if abs(delta) < 0.01:
                        continue

                    # Taker fee on the rebalance trade
                    fee = _polymarket_taker_fee(delta, mkt_price, taker_fee_rate)

                    if delta > 0:
                        cost = delta * mkt_price + fee
                        cash_spent += cost
                        position += delta
                    else:
                        proceeds = abs(delta) * mkt_price - fee
                        cash_spent -= proceeds
                        position += delta

                # Settlement: home YES contracts pay $1 if home wins, $0 if not
                settle_value = position * (1.0 if home_win else 0.0)
                game_pnl = settle_value - cash_spent
                game_pnls.append(game_pnl)

            if not game_pnls:
                continue

            pnls = np.array(game_pnls)
            total_pnl = pnls.sum()
            n_games = len(pnls)
            # ROI relative to total capital deployed
            total_capital = bankroll * n_games
            roi = total_pnl / total_capital if total_capital > 0 else 0

            # Annualized Sharpe: use actual date span to compute games/year
            if n_games > 1 and pnls.std() > 0:
                if game_dates_list:
                    dates = pd.to_datetime(game_dates_list)
                    n_days = max((dates.max() - dates.min()).days, 1)
                    games_per_year = n_games / n_days * 365.0
                else:
                    # Fallback: ~162 games over ~183 days ≈ 0.89 games/day
                    games_per_year = n_games
                sharpe = (pnls.mean() / pnls.std()) * np.sqrt(games_per_year)
            else:
                sharpe = 0.0

            # Bootstrap 95% CI for ROI
            rng = np.random.RandomState(42)
            boot_rois = []
            for _ in range(1000):
                sample = rng.choice(pnls, size=n_games, replace=True)
                boot_roi = sample.sum() / total_capital
                boot_rois.append(boot_roi)
            boot_rois = sorted(boot_rois)
            ci_lo = boot_rois[25]
            ci_hi = boot_rois[975]
            p_positive = sum(1 for r in boot_rois if r > 0) / len(boot_rois)

            results.append({
                "kelly_pct": kf,
                "min_edge": me,
                "n_games": n_games,
                "total_pnl": total_pnl,
                "roi": roi,
                "sharpe": sharpe,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_positive": p_positive,
            })

    return results


def print_kelly_rebalancing(results: list[dict]):
    """Print Kelly rebalancing results table."""
    if not results:
        return
    log(f"\n  Kelly Rebalancing Strategy (position rebalanced at each half-inning)")
    log(f"  {'Kelly%':>7s} {'MinEdge':>8s} {'Games':>6s} {'TotalPnL':>10s} "
        f"{'ROI':>7s} {'Sharpe':>7s} {'95% CI':>20s} {'P(ROI>0)':>9s}")
    log(f"  {'-'*80}")
    for r in results:
        ci_str = f"[{r['ci_lo']:+.1%}, {r['ci_hi']:+.1%}]"
        log(f"  {r['kelly_pct']:>6.0%} {r['min_edge']:>7.0%} "
            f"{r['n_games']:>6d} ${r['total_pnl']:>+9.0f} "
            f"{r['roi']:>+6.1%} {r['sharpe']:>7.2f} "
            f"{ci_str:>20s} {r['p_positive']:>8.0%}")


def compute_calibration(
    df: pd.DataFrame,
    prob_col: str,
    bins: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Compute calibration bins for a probability column."""
    if bins is None:
        bins = [
            (0.0, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50),
            (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.0),
        ]

    df = df.dropna(subset=[prob_col, "home_win"])
    y = df["home_win"].values
    probs = df[prob_col].values

    rows = []
    for lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() < 5:
            continue
        rows.append({
            "bin": f"[{lo:.2f}, {hi:.2f})",
            "n": mask.sum(),
            "mean_pred": probs[mask].mean(),
            "actual_rate": y[mask].mean(),
            "gap": probs[mask].mean() - y[mask].mean(),
        })
    return rows


# ── Pretty Printing ──────────────────────────────────────────────────────────


def print_header(title: str):
    log(f"\n{'='*70}")
    log(f"  {title}")
    log(f"{'='*70}")


def print_metrics(metrics: dict):
    if not metrics:
        log("  No data")
        return

    log(f"\n  {metrics['label']} — {metrics['n']} observations")
    log(f"  {'Metric':<22s} {'Simulator':>10s} {'Kalshi':>10s} {'Baseline':>10s}")
    log(f"  {'-'*55}")
    log(f"  {'Log Loss':<22s} {metrics['sim_log_loss']:>10.4f} {metrics['market_log_loss']:>10.4f} {metrics['baseline_log_loss']:>10.4f}")
    log(f"  {'Brier Score':<22s} {metrics['sim_brier']:>10.4f} {metrics['market_brier']:>10.4f}")
    log(f"  {'AUC':<22s} {metrics['sim_auc']:>10.4f} {metrics['market_auc']:>10.4f}")
    log(f"  {'Accuracy':<22s} {metrics['sim_accuracy']:>10.1%} {metrics['market_accuracy']:>10.1%}")
    log(f"  {'Mean prediction':<22s} {metrics['sim_mean_pred']:>10.4f} {metrics['market_mean_pred']:>10.4f}")

    log(f"\n  Edge (sim - market): mean={metrics['mean_edge']:+.4f}, "
        f"std={metrics['std_edge']:.4f}, MAE={metrics['mae_vs_market']:.4f}")


def print_roi(roi_results: list[dict], label: str = ""):
    if not roi_results:
        return
    log(f"\n  ROI Simulation — {label} (flat $100 bets)")
    log(f"  {'Edge':>7s} {'Bets':>6s} {'PnL':>10s} {'ROI':>8s} {'Sharpe':>8s} {'WinRate':>8s}")
    log(f"  {'-'*51}")
    for r in roi_results:
        if r["n_bets"] == 0:
            log(f"  {r['threshold']:>6.0%} {0:>6d} {'---':>10s}")
            continue
        log(f"  {r['threshold']:>6.0%} {r['n_bets']:>6d} "
            f"${r['total_pnl']:>+9.0f} {r['roi']:>+7.1%} "
            f"{r['sharpe']:>8.2f} {r['win_rate']:>7.1%}")


def print_calibration(cal_rows: list[dict], label: str = ""):
    if not cal_rows:
        return
    log(f"\n  Calibration — {label}")
    log(f"  {'Bin':<16s} {'N':>5s} {'Predicted':>10s} {'Actual':>10s} {'Gap':>8s}")
    log(f"  {'-'*52}")
    for r in cal_rows:
        log(f"  {r['bin']:<16s} {r['n']:>5d} {r['mean_pred']:>10.3f} "
            f"{r['actual_rate']:>10.3f} {r['gap']:>+8.3f}")


def print_ingame_by_progress(df: pd.DataFrame):
    """Break down in-game results by game progress (early/mid/late)."""
    if len(df) == 0:
        return

    log(f"\n  In-Game Performance by Game Phase")
    log(f"  {'Phase':<12s} {'N':>5s} {'SimLL':>8s} {'MktLL':>8s} {'Δ':>8s} {'MAE':>8s} {'SimAcc':>7s}")
    log(f"  {'-'*60}")

    phases = [
        ("Early (1-3)", lambda gp: gp < 1/3),
        ("Mid (4-6)", lambda gp: (gp >= 1/3) & (gp < 2/3)),
        ("Late (7-9+)", lambda gp: gp >= 2/3),
    ]

    for name, mask_fn in phases:
        mask = mask_fn(df["game_progress"])
        subset = df[mask]
        if len(subset) < 10:
            continue

        y = subset["home_win"].values
        sim = np.clip(subset["sim_home_wp"].values, 0.01, 0.99)
        market = np.clip(subset["kalshi_home_prob"].values, 0.01, 0.99)

        sim_ll = log_loss(y, sim)
        mkt_ll = log_loss(y, market)
        mae = np.abs(sim - market).mean()
        sim_acc = ((sim > 0.5) == y).mean()

        log(f"  {name:<12s} {len(subset):>5d} {sim_ll:>8.4f} {mkt_ll:>8.4f} "
            f"{sim_ll - mkt_ll:>+8.4f} {mae:>8.4f} {sim_acc:>6.1%}")


def print_ingame_by_inning(df: pd.DataFrame):
    """Break down in-game results by inning."""
    if len(df) == 0:
        return

    log(f"\n  In-Game Performance by Inning")
    log(f"  {'Inning':>6s} {'N':>5s} {'SimLL':>8s} {'MktLL':>8s} {'Δ':>8s} {'SimAcc':>7s}")
    log(f"  {'-'*45}")

    for inning in sorted(df["inning"].unique()):
        subset = df[df["inning"] == inning]
        if len(subset) < 5:
            continue

        y = subset["home_win"].values
        sim = np.clip(subset["sim_home_wp"].values, 0.01, 0.99)
        market = np.clip(subset["kalshi_home_prob"].values, 0.01, 0.99)

        try:
            sim_ll = log_loss(y, sim)
            mkt_ll = log_loss(y, market)
        except ValueError:
            continue

        sim_acc = ((sim > 0.5) == y).mean()

        log(f"  {inning:>6d} {len(subset):>5d} {sim_ll:>8.4f} {mkt_ll:>8.4f} "
            f"{sim_ll - mkt_ll:>+8.4f} {sim_acc:>6.1%}")


def print_monthly_breakdown(df: pd.DataFrame, label: str = ""):
    """Monthly breakdown of performance."""
    if len(df) == 0 or "game_date" not in df.columns:
        return

    df = df.copy()
    df["month"] = df["game_date"].str[:7]

    month_names = {
        "03": "Mar", "04": "Apr", "05": "May", "06": "Jun",
        "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct",
    }

    log(f"\n  Monthly Breakdown — {label}")
    log(f"  {'Month':>7s} {'N':>5s} {'SimLL':>8s} {'MktLL':>8s} {'Δ':>8s} "
        f"{'ROI@5%':>8s} {'Bets':>5s}")
    log(f"  {'-'*55}")

    for month in sorted(df["month"].unique()):
        subset = df[df["month"] == month]
        if len(subset) < 10:
            continue

        y = subset["home_win"].values
        sim = np.clip(subset["sim_home_wp"].values, 0.01, 0.99)
        market = np.clip(subset["kalshi_home_prob"].values, 0.01, 0.99)

        try:
            sim_ll = log_loss(y, sim)
            mkt_ll = log_loss(y, market)
        except ValueError:
            continue

        # Compute 5% edge ROI for this month
        edge = sim - market
        pnls = []
        for i in range(len(subset)):
            if abs(edge[i]) < 0.05:
                continue
            if edge[i] > 0:
                cost = market[i] * 100
                pnl = (100 - cost) if y[i] == 1 else -cost
            else:
                cost = (1 - market[i]) * 100
                pnl = (100 - cost) if y[i] == 0 else -cost
            pnls.append(pnl)

        n_bets = len(pnls)
        roi_str = f"{sum(pnls) / (n_bets * 100):>+7.1%}" if n_bets > 0 else "   ---"

        mm = month[-2:]
        name = month_names.get(mm, month)
        log(f"  {name:>7s} {len(subset):>5d} {sim_ll:>8.4f} {mkt_ll:>8.4f} "
            f"{sim_ll - mkt_ll:>+8.4f} {roi_str:>8s} {n_bets:>5d}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Backtest MC simulator vs Kalshi prediction market",
    )
    parser.add_argument("--season", type=int, required=True, help="Season to backtest")
    parser.add_argument("--ingame", action="store_true",
                        help="Include in-game backtest (slower, needs API calls)")
    parser.add_argument("--pregame-only", action="store_true",
                        help="Only run pregame backtest")
    parser.add_argument("--ingame-only", action="store_true",
                        help="Only run in-game backtest (skip pregame)")
    parser.add_argument("--max-games", type=int, default=100,
                        help="Max games for in-game backtest (default: 100)")
    parser.add_argument("--n-sims", type=int, default=1000,
                        help="Sims per game (default: 1000 for speed)")
    parser.add_argument("--n-sims-ingame", type=int, default=500,
                        help="Sims per in-game state (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-mo", action="store_true",
                        help="Skip multi-output model (use log5/xRV only)")
    parser.add_argument("--blend", action="store_true",
                        help="Blend MC sim with pregame LR+XGB ensemble (requires walk_forward_predictions.csv)")
    parser.add_argument("--blend-only", action="store_true",
                        help="Load existing in-game CSV and run blend analysis only (no API calls)")
    args = parser.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Blend-Only Mode (no API calls, no model loading) ──────────────────
    if args.blend_only:
        csv_path = AUDIT_DIR / f"sim_vs_kalshi_ingame_{args.season}.csv"
        if not csv_path.exists():
            log(f"  ERROR: {csv_path} not found. Run in-game backtest first.")
            return
        log(f"Loading existing in-game data from {csv_path}...")
        ingame_df = pd.read_csv(csv_path)
        log(f"  {len(ingame_df)} state points loaded")

        print_header(f"ENSEMBLE BLEND ANALYSIS — {args.season}")
        ingame_df = blend_with_ensemble(ingame_df, args.season)

        if "blended_wp" in ingame_df.columns:
            # Side-by-side metrics
            log("\n  HEAD-TO-HEAD: Sim vs Blend vs Market")
            log(f"  {'':28s} {'Sim':>10s} {'Blend':>10s} {'Market':>10s}")
            log(f"  {'-'*60}")

            sim_m = compute_metrics(ingame_df, sim_col="sim_home_wp", label="sim")
            blend_m = compute_metrics(ingame_df, sim_col="blended_wp", label="blend")

            for key, label in [
                ("sim_log_loss", "Log Loss"),
                ("sim_brier", "Brier Score"),
                ("sim_auc", "AUC"),
            ]:
                sv = sim_m.get(key, 0)
                bv = blend_m.get(key, 0)
                mv = sim_m.get(key.replace("sim_", "market_"), 0)
                log(f"  {label:28s} {sv:10.4f} {bv:10.4f} {mv:10.4f}")

            # By-inning comparison
            log(f"\n  BY-INNING LOG LOSS:")
            log(f"  {'Inning':8s} {'Sim':>10s} {'Blend':>10s} {'Market':>10s} {'Best':>8s}")
            log(f"  {'-'*50}")
            for inn in sorted(ingame_df["inning"].unique()):
                sub = ingame_df[ingame_df["inning"] == inn]
                if len(sub) < 10:
                    continue
                y = sub["home_win"].values
                sim_ll = log_loss(y, np.clip(sub["sim_home_wp"], 0.01, 0.99))
                blend_ll = log_loss(y, np.clip(sub["blended_wp"], 0.01, 0.99))
                mkt_ll = log_loss(y, np.clip(sub["kalshi_home_prob"], 0.01, 0.99))
                best = "SIM" if sim_ll < blend_ll else "BLEND"
                if mkt_ll < min(sim_ll, blend_ll):
                    best = "MKT"
                log(f"  {inn:8d} {sim_ll:10.4f} {blend_ll:10.4f} {mkt_ll:10.4f} {best:>8s}")

            # Kelly rebalancing comparison
            print_header("KELLY REBALANCING — MC SIMULATOR ONLY")
            kelly_sim = compute_kelly_rebalancing(ingame_df, sim_col="sim_home_wp")
            print_kelly_rebalancing(kelly_sim)

            print_header("KELLY REBALANCING — BLENDED SIGNAL")
            kelly_blend = compute_kelly_rebalancing(ingame_df, sim_col="blended_wp")
            print_kelly_rebalancing(kelly_blend)

            # Save updated CSV with blend column
            out_path = AUDIT_DIR / f"sim_vs_kalshi_ingame_{args.season}.csv"
            ingame_df.to_csv(out_path, index=False)
            log(f"\n  Saved updated results to {out_path}")

        return

    # Load multi-output matchup models FIRST (torch must init before large data)
    mo_models = None
    if not args.no_mo:
        mo_season = args.season - 1
        log(f"Loading multi-output matchup models (trained on {mo_season})...")
        try:
            mo_models = load_multi_output_models(mo_season)
            log(f"  Loaded multi-output models: {list(mo_models.keys())}")
        except Exception as e:
            log(f"  Warning: Could not load multi-output models: {e}")
            log("  Falling back to log5-only predictions")
            mo_models = None
    else:
        log("Skipping multi-output model (--no-mo flag)")

    # Load simulation artifacts
    log("Loading simulation artifacts...")
    try:
        base_rates, transition_matrix = load_sim_artifacts()
    except FileNotFoundError as e:
        log(f"Simulation data not found: {e}")
        log("Run `python src/build_transition_matrix.py` first.")
        return

    # Load shared data once for both backtests
    log("Loading xRV, matchup models, and lineups...")
    shared = load_backtest_data(args.season)
    if shared is None:
        log("Cannot proceed without xRV data")
        return
    idx, matchup_models, lineups = shared

    # ── Pregame Backtest ──────────────────────────────────────────────────
    pregame_df = pd.DataFrame()
    if not args.ingame_only:
        print_header(f"PREGAME BACKTEST — {args.season}")

        pregame_config = SimConfig(n_sims=args.n_sims, random_seed=args.seed)
        pregame_df = run_pregame_backtest(
            args.season, base_rates, transition_matrix, pregame_config,
            idx=idx, matchup_models=matchup_models, lineups=lineups,
            mo_models=mo_models,
        )

    if len(pregame_df) > 0:
        # Overall metrics
        metrics = compute_metrics(pregame_df, label="Pregame: Simulator vs Kalshi")
        print_metrics(metrics)

        # ROI
        roi = compute_roi(pregame_df)
        print_roi(roi, "Pregame")

        # Calibration
        print_header("CALIBRATION")
        sim_cal = compute_calibration(pregame_df, "sim_home_wp")
        print_calibration(sim_cal, "Simulator")
        mkt_cal = compute_calibration(pregame_df, "kalshi_home_prob")
        print_calibration(mkt_cal, "Kalshi")

        # Monthly
        print_monthly_breakdown(pregame_df, "Pregame")

        # Volume-weighted analysis (higher-volume markets = more efficient)
        if "kalshi_volume" in pregame_df.columns:
            median_vol = pregame_df["kalshi_volume"].median()
            high_vol = pregame_df[pregame_df["kalshi_volume"] >= median_vol]
            low_vol = pregame_df[pregame_df["kalshi_volume"] < median_vol]

            if len(high_vol) >= 50 and len(low_vol) >= 50:
                print_header("VOLUME SPLIT")
                m_high = compute_metrics(high_vol, label=f"High volume (>= ${median_vol:,.0f})")
                print_metrics(m_high)
                m_low = compute_metrics(low_vol, label=f"Low volume (< ${median_vol:,.0f})")
                print_metrics(m_low)

        # Save pregame results
        out_path = AUDIT_DIR / f"sim_vs_kalshi_pregame_{args.season}.csv"
        pregame_df.to_csv(out_path, index=False)
        log(f"\n  Saved pregame results to {out_path}")

    else:
        log("  No pregame results to report")

    # ── In-Game Backtest ──────────────────────────────────────────────────
    if (args.ingame or args.ingame_only) and not args.pregame_only:
        print_header(f"IN-GAME BACKTEST — {args.season}")

        ingame_config = SimConfig(n_sims=args.n_sims_ingame, random_seed=args.seed)
        ingame_df = run_ingame_backtest(
            args.season, base_rates, transition_matrix,
            ingame_config, max_games=args.max_games, pregame_df=pregame_df,
            idx=idx, matchup_models=matchup_models, lineups=lineups,
            mo_models=mo_models,
        )

        if len(ingame_df) > 0:
            # Overall in-game metrics
            metrics = compute_metrics(ingame_df, label="In-Game: Simulator vs Kalshi")
            print_metrics(metrics)

            # ROI
            roi = compute_roi(ingame_df)
            print_roi(roi, "In-Game")

            # By game phase
            print_ingame_by_progress(ingame_df)

            # By inning
            print_ingame_by_inning(ingame_df)

            # Calibration
            sim_cal = compute_calibration(ingame_df, "sim_home_wp")
            print_calibration(sim_cal, "In-Game Simulator")

            # Kelly rebalancing simulation (sim-only)
            print_header("KELLY REBALANCING — MC SIMULATOR ONLY")
            kelly_results = compute_kelly_rebalancing(ingame_df, sim_col="sim_home_wp")
            print_kelly_rebalancing(kelly_results)

            # ── Ensemble Blend ─────────────────────────────────────────
            if args.blend:
                print_header("ENSEMBLE BLEND — MC SIM + LR+XGB PREGAME")
                ingame_df = blend_with_ensemble(ingame_df, args.season)

                if "blended_wp" in ingame_df.columns:
                    # Side-by-side metrics
                    log("\n  HEAD-TO-HEAD: Sim vs Blend vs Market")
                    log(f"  {'':28s} {'Sim':>10s} {'Blend':>10s} {'Market':>10s}")
                    log(f"  {'-'*60}")

                    sim_m = compute_metrics(ingame_df, sim_col="sim_home_wp",
                                            label="sim")
                    blend_m = compute_metrics(ingame_df, sim_col="blended_wp",
                                              label="blend")

                    for key, label in [
                        ("sim_log_loss", "Log Loss"),
                        ("sim_brier", "Brier Score"),
                        ("sim_auc", "AUC"),
                    ]:
                        sv = sim_m.get(key, 0)
                        bv = blend_m.get(key, 0)
                        mv = sim_m.get(key.replace("sim_", "market_"), 0)
                        best = "←" if key != "sim_auc" and bv < sv else ("←" if key == "sim_auc" and bv > sv else "")
                        log(f"  {label:28s} {sv:10.4f} {bv:10.4f}{best} {mv:10.4f}")

                    # By-inning comparison
                    log(f"\n  BY-INNING LOG LOSS:")
                    log(f"  {'Inning':8s} {'Sim':>10s} {'Blend':>10s} {'Market':>10s} {'Best':>8s}")
                    log(f"  {'-'*50}")
                    for inn in sorted(ingame_df["inning"].unique()):
                        sub = ingame_df[ingame_df["inning"] == inn]
                        if len(sub) < 10:
                            continue
                        y = sub["home_win"].values
                        sim_ll = log_loss(y, np.clip(sub["sim_home_wp"], 0.01, 0.99))
                        blend_ll = log_loss(y, np.clip(sub["blended_wp"], 0.01, 0.99))
                        mkt_ll = log_loss(y, np.clip(sub["kalshi_home_prob"], 0.01, 0.99))
                        best = "SIM" if sim_ll < blend_ll else "BLEND"
                        if mkt_ll < min(sim_ll, blend_ll):
                            best = "MKT"
                        log(f"  {inn:8d} {sim_ll:10.4f} {blend_ll:10.4f} {mkt_ll:10.4f} {best:>8s}")

                    # Kelly rebalancing with blended signal
                    print_header("KELLY REBALANCING — BLENDED SIGNAL")
                    kelly_blend = compute_kelly_rebalancing(
                        ingame_df, sim_col="blended_wp",
                    )
                    print_kelly_rebalancing(kelly_blend)

            # Save
            out_path = AUDIT_DIR / f"sim_vs_kalshi_ingame_{args.season}.csv"
            ingame_df.to_csv(out_path, index=False)
            log(f"\n  Saved in-game results to {out_path}")

            # ── Combined analysis ─────────────────────────────────────────
            print_header("CONVERGENCE ANALYSIS")
            log("  As game progresses, simulator should converge toward Kalshi:")
            log(f"  {'Phase':<12s} {'MAE':>8s} {'Corr':>8s}")
            log(f"  {'-'*30}")

            for lo, hi, name in [(0, 0.33, "Early"), (0.33, 0.66, "Mid"), (0.66, 1.01, "Late")]:
                mask = (ingame_df["game_progress"] >= lo) & (ingame_df["game_progress"] < hi)
                subset = ingame_df[mask]
                if len(subset) < 10:
                    continue
                mae = np.abs(subset["sim_home_wp"] - subset["kalshi_home_prob"]).mean()
                corr = np.corrcoef(subset["sim_home_wp"], subset["kalshi_home_prob"])[0, 1]
                log(f"  {name:<12s} {mae:>8.4f} {corr:>8.4f}")
        else:
            log("  No in-game results to report")

    elif not args.pregame_only:
        log("\n  Skipping in-game backtest (use --ingame to enable)")

    # ── Summary ──────────────────────────────────────────────────────────
    print_header("SUMMARY")
    if len(pregame_df) > 0:
        m = compute_metrics(pregame_df, label="Pregame")
        ll_diff = m["sim_log_loss"] - m["market_log_loss"]
        log(f"  Pregame: {m['n']} games")
        log(f"    Sim log loss:    {m['sim_log_loss']:.4f}")
        log(f"    Kalshi log loss: {m['market_log_loss']:.4f}")
        log(f"    Delta:           {ll_diff:+.4f} ({'worse' if ll_diff > 0 else 'better'})")
        log(f"    Sim AUC:         {m['sim_auc']:.4f}")
        log(f"    Kalshi AUC:      {m['market_auc']:.4f}")


if __name__ == "__main__":
    main()
