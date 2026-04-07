#!/usr/bin/env python3
"""
Backtest: Buy Polymarket MLB Moneyline Gaps vs Sharp Closing Lines
==================================================================

Compares Polymarket pre-game closing prices against devigged sharp book
closing lines. Both snapshots are "at close" (last price before first pitch),
ensuring time-synchronized comparison.

Pipeline:
  1. Fetch all settled Polymarket MLB game events (Gamma API)
  2. For each game, pull trade history (data API), take last pre-game trade
     as the Polymarket closing price
  3. Load sharp closing lines from mlb_odds_dataset.json (FanDuel, Caesars,
     Bet365, DraftKings) and devig via multiplicative method
  4. Match by team + date, compute probability gap
  5. Backtest: buy the Polymarket side where sharp line disagrees by > threshold

Usage:
    python src/backtest_poly_vs_sharp.py
    python src/backtest_poly_vs_sharp.py --threshold 0.04 --min-volume 500
    python src/backtest_poly_vs_sharp.py --scrape-only   # just build the dataset
"""

import argparse
import json
import re
import sys
import time as time_mod
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
POLY_DIR = DATA / "polymarket"
ODDS_PATH = DATA / "odds" / "mlb_odds_dataset.json"
CACHE_PATH = POLY_DIR / "poly_closing_prices.parquet"

GAMMA_API = "https://gamma-api.polymarket.com"
TRADE_API = "https://data-api.polymarket.com"
MLB_TAG_ID = 100381

# Team name normalization (Polymarket full names → abbreviations)
TEAM_TO_ABBR = {
    "Arizona Diamondbacks": "AZ", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
    # Short names from outcomes
    "Diamondbacks": "AZ", "D-backs": "AZ", "Braves": "ATL",
    "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC",
    "White Sox": "CWS", "Reds": "CIN", "Guardians": "CLE",
    "Rockies": "COL", "Tigers": "DET", "Astros": "HOU",
    "Royals": "KC", "Angels": "LAA", "Dodgers": "LAD",
    "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN",
    "Mets": "NYM", "Yankees": "NYY", "Phillies": "PHI",
    "Pirates": "PIT", "Padres": "SD", "Giants": "SF",
    "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB",
    "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH",
}

# SBR/odds dataset team abbreviation mapping
SBR_TEAM_MAP = {
    "TOR": "TOR", "NYY": "NYY", "NYM": "NYM", "BOS": "BOS",
    "TB": "TB", "BAL": "BAL", "CLE": "CLE", "DET": "DET",
    "MIN": "MIN", "CWS": "CWS", "KC": "KC", "HOU": "HOU",
    "TEX": "TEX", "SEA": "SEA", "LAA": "LAA", "OAK": "OAK",
    "ATL": "ATL", "NYM": "NYM", "PHI": "PHI", "MIA": "MIA",
    "WSH": "WSH", "CHC": "CHC", "STL": "STL", "MIL": "MIL",
    "CIN": "CIN", "PIT": "PIT", "LAD": "LAD", "SD": "SD",
    "SF": "SF", "AZ": "AZ", "COL": "COL", "ARI": "AZ",
}


def resolve_abbr(name: str) -> str | None:
    name = name.strip()
    if name in TEAM_TO_ABBR:
        return TEAM_TO_ABBR[name]
    for full, abbr in TEAM_TO_ABBR.items():
        if full in name or name in full:
            return abbr
    return None


# ═══════════════════════════════════════════════════════════════════
# 1. Fetch Polymarket settled games + closing prices
# ═══════════════════════════════════════════════════════════════════

def fetch_settled_mlb_events() -> list[dict]:
    """Fetch all settled MLB game events from Gamma API."""
    all_events = []
    offset = 0
    while True:
        params = {
            "tag_id": MLB_TAG_ID,
            "closed": "true",
            "limit": 100,
            "offset": offset,
        }
        resp = requests.get(f"{GAMMA_API}/events", params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_events.extend(batch)
        if len(batch) < 100:
            break
        offset += 100
        time_mod.sleep(0.05)

    # Filter to game events only
    game_events = [e for e in all_events if " vs" in e.get("title", "").lower()]
    print(f"  Fetched {len(game_events)} settled MLB game events")
    return game_events


def extract_moneyline_markets(events: list[dict]) -> list[dict]:
    """Extract moneyline market info from each game event."""
    games = []
    seen = set()

    for event in events:
        title = event.get("title", "")
        slug = event.get("ticker", "")

        # Skip futures, props
        skip_words = ["champion", "series winner", "mvp", "cy young", "division",
                       "cba", "record", "sweep", "leader", "jersey"]
        if any(w in title.lower() for w in skip_words):
            continue

        # Extract game date from slug
        slug_match = re.search(r"(\d{4}-\d{2}-\d{2})$", slug)
        if not slug_match:
            continue
        game_date = slug_match.group(1)

        for market in event.get("markets", []):
            outcomes_raw = market.get("outcomes", "[]")
            if isinstance(outcomes_raw, str):
                outcomes_raw = json.loads(outcomes_raw)

            if len(outcomes_raw) != 2:
                continue
            if any(x.lower() in ["over", "under", "yes", "no", "yes run", "no run"]
                   for x in outcomes_raw):
                continue

            # Must be two different teams
            t0 = resolve_abbr(outcomes_raw[0])
            t1 = resolve_abbr(outcomes_raw[1])
            if not t0 or not t1 or t0 == t1:
                continue

            # Determine winner from resolved prices
            prices_raw = market.get("outcomePrices", "[]")
            if isinstance(prices_raw, str):
                prices_raw = json.loads(prices_raw)
            try:
                p0, p1 = float(prices_raw[0]), float(prices_raw[1])
            except (ValueError, IndexError, TypeError):
                continue

            # Dedup
            key = (game_date, *sorted([t0, t1]))
            if key in seen:
                continue
            seen.add(key)

            cond_id = market.get("conditionId", "")
            game_start = market.get("gameStartTime", "")

            games.append({
                "game_date": game_date,
                "team0": t0,
                "team1": t1,
                "team0_name": outcomes_raw[0],
                "team1_name": outcomes_raw[1],
                "team0_won": p0 > p1,
                "condition_id": cond_id,
                "game_start_time": game_start,
                "volume": float(market.get("volume", 0) or 0),
                "slug": slug,
            })
            break  # one moneyline per event

    print(f"  Extracted {len(games)} moneyline markets")
    return games


def fetch_closing_price_from_trades(condition_id: str, game_start_str: str,
                                     session: requests.Session) -> dict | None:
    """Get pre-game closing price from trade history.

    Returns dict with team0_close_price and timestamp, or None.
    """
    # Parse game start time
    game_start_ts = None
    if game_start_str:
        try:
            gst = game_start_str.replace("Z", "+00:00")
            if "+" not in gst and "-" not in gst[10:]:
                gst = gst + "+00:00"
            game_dt = datetime.fromisoformat(gst)
            game_start_ts = game_dt.timestamp()
        except (ValueError, TypeError):
            pass

    if not game_start_ts:
        return None

    try:
        resp = session.get(
            f"{TRADE_API}/trades",
            params={"market": condition_id, "limit": 500},
            timeout=15,
        )
        resp.raise_for_status()
        trades = resp.json()
    except Exception:
        return None

    if not trades:
        return None

    # Parse timestamps and filter to pre-game
    pre_game_trades = []
    for t in trades:
        ts = t.get("timestamp")
        if ts is None:
            continue
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
        ts = float(ts)
        price = float(t.get("price", 0))
        if ts < game_start_ts and 0.02 < price < 0.98:
            pre_game_trades.append({
                "ts": ts,
                "price": price,
                "outcome": t.get("outcome", ""),
                "size": float(t.get("size", 0)),
                "side": t.get("side", ""),
            })

    if not pre_game_trades:
        return None

    # Sort by timestamp, take the last pre-game trade
    pre_game_trades.sort(key=lambda x: x["ts"])
    last = pre_game_trades[-1]

    # Also compute VWAP of last 5 minutes for stability
    cutoff = game_start_ts - 300  # 5 min before start
    recent = [t for t in pre_game_trades if t["ts"] >= cutoff]
    if len(recent) >= 3:
        total_size = sum(t["size"] for t in recent)
        if total_size > 0:
            vwap = sum(t["price"] * t["size"] for t in recent) / total_size
        else:
            vwap = last["price"]
    else:
        vwap = last["price"]

    return {
        "close_price": last["price"],
        "close_outcome": last["outcome"],
        "close_ts": last["ts"],
        "vwap_5min": vwap,
        "n_pregame_trades": len(pre_game_trades),
        "minutes_before_start": (game_start_ts - last["ts"]) / 60,
    }


def scrape_polymarket_closing_prices(games: list[dict]) -> pd.DataFrame:
    """Fetch closing prices for all games via trade history API."""
    rows = []
    n_success = 0

    session = requests.Session()

    for i, game in enumerate(games):
        cond_id = game["condition_id"]
        if not cond_id:
            continue

        result = fetch_closing_price_from_trades(
            cond_id, game["game_start_time"], session
        )

        if result:
            # Determine which team the close price refers to
            close_outcome = result["close_outcome"]
            close_abbr = resolve_abbr(close_outcome)

            if close_abbr == game["team0"]:
                team0_prob = result["close_price"]
            elif close_abbr == game["team1"]:
                team0_prob = 1.0 - result["close_price"]
            else:
                # Can't determine mapping — skip
                continue

            rows.append({
                "game_date": game["game_date"],
                "team0": game["team0"],
                "team1": game["team1"],
                "team0_name": game["team0_name"],
                "team1_name": game["team1_name"],
                "team0_won": game["team0_won"],
                "poly_team0_prob": round(team0_prob, 4),
                "poly_team1_prob": round(1.0 - team0_prob, 4),
                "poly_vwap_team0": round(result["vwap_5min"] if close_abbr == game["team0"]
                                         else 1.0 - result["vwap_5min"], 4),
                "volume": game["volume"],
                "n_pregame_trades": result["n_pregame_trades"],
                "minutes_before_start": round(result["minutes_before_start"], 1),
                "slug": game["slug"],
            })
            n_success += 1

        # Rate limit
        time_mod.sleep(0.15)

        if (i + 1) % 100 == 0 or i + 1 == len(games):
            print(f"    {i + 1}/{len(games)} processed ({n_success} with prices)")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Got closing prices for {n_success}/{len(games)} games")
    return df


# ═══════════════════════════════════════════════════════════════════
# 2. Load sharp closing lines
# ═══════════════════════════════════════════════════════════════════

def american_to_implied(odds: float) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    return 0.5


def devig(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    if total == 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def load_sharp_lines() -> pd.DataFrame:
    """Load devigged closing lines from mlb_odds_dataset.json.

    Takes the consensus (average) of all available books' closing lines,
    then devigs. This is the "sharp fair" benchmark.
    """
    if not ODDS_PATH.exists():
        print(f"  ERROR: {ODDS_PATH} not found")
        sys.exit(1)

    print(f"  Loading {ODDS_PATH.name}...")
    with open(ODDS_PATH) as f:
        odds_data = json.load(f)

    rows = []
    for date_str, games in odds_data.items():
        for game in games:
            gv = game.get("gameView", {})
            home_full = gv.get("homeTeam", {}).get("fullName", "")
            away_full = gv.get("awayTeam", {}).get("fullName", "")
            home_short = gv.get("homeTeam", {}).get("shortName", "")
            away_short = gv.get("awayTeam", {}).get("shortName", "")
            home_score = gv.get("homeTeamScore")
            away_score = gv.get("awayTeamScore")
            status = gv.get("gameStatusText", "")

            # Normalize team abbrs
            home_abbr = SBR_TEAM_MAP.get(home_short, home_short)
            away_abbr = SBR_TEAM_MAP.get(away_short, away_short)

            # Get moneyline closing odds — consensus across books
            ml_data = game.get("odds", {}).get("moneyline", [])
            home_implied_list = []
            away_implied_list = []

            for book in ml_data:
                cl = book.get("currentLine", {})
                home_odds = cl.get("homeOdds")
                away_odds = cl.get("awayOdds")
                if home_odds is not None and away_odds is not None:
                    home_implied_list.append(american_to_implied(home_odds))
                    away_implied_list.append(american_to_implied(away_odds))

            if not home_implied_list:
                continue

            # Consensus implied probability (average across books)
            home_raw = np.mean(home_implied_list)
            away_raw = np.mean(away_implied_list)

            # Devig
            home_fair, away_fair = devig(home_raw, away_raw)

            # Determine winner
            if home_score is not None and away_score is not None:
                home_win = int(home_score) > int(away_score)
            else:
                home_win = None

            rows.append({
                "game_date": date_str,
                "home_team": home_abbr,
                "away_team": away_abbr,
                "sharp_home_prob": round(home_fair, 4),
                "sharp_away_prob": round(away_fair, 4),
                "home_win": home_win,
                "n_books": len(home_implied_list),
                "vig": round(home_raw + away_raw - 1.0, 4),
            })

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} games with sharp lines ({df['game_date'].min()} to {df['game_date'].max()})")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3. Match & backtest
# ═══════════════════════════════════════════════════════════════════

def match_poly_to_sharp(poly_df: pd.DataFrame, sharp_df: pd.DataFrame) -> pd.DataFrame:
    """Match Polymarket closing prices to sharp closing lines by team+date."""
    matched = []

    # Build sharp lookup: (date, sorted_teams) -> row
    sharp_lookup = {}
    for _, row in sharp_df.iterrows():
        key = (row["game_date"], *sorted([row["home_team"], row["away_team"]]))
        sharp_lookup[key] = row

    for _, prow in poly_df.iterrows():
        key = (prow["game_date"], *sorted([prow["team0"], prow["team1"]]))
        srow = sharp_lookup.get(key)
        if srow is None:
            continue

        # Align probabilities: both in terms of home team
        home = srow["home_team"]
        away = srow["away_team"]

        if prow["team0"] == home:
            poly_home = prow["poly_team0_prob"]
            poly_away = prow["poly_team1_prob"]
            poly_home_vwap = prow["poly_vwap_team0"]
            team0_is_home = True
        elif prow["team1"] == home:
            poly_home = prow["poly_team1_prob"]
            poly_away = prow["poly_team0_prob"]
            poly_home_vwap = 1.0 - prow["poly_vwap_team0"]
            team0_is_home = False
        else:
            continue

        home_win = srow["home_win"]
        if home_win is None:
            continue

        gap_home = poly_home - srow["sharp_home_prob"]
        gap_away = poly_away - srow["sharp_away_prob"]

        matched.append({
            "game_date": prow["game_date"],
            "home_team": home,
            "away_team": away,
            "home_win": int(home_win),
            "sharp_home_prob": srow["sharp_home_prob"],
            "sharp_away_prob": srow["sharp_away_prob"],
            "poly_home_prob": poly_home,
            "poly_away_prob": poly_away,
            "poly_home_vwap": poly_home_vwap,
            "gap_home": round(gap_home, 4),
            "gap_away": round(gap_away, 4),
            "max_gap": round(max(abs(gap_home), abs(gap_away)), 4),
            "volume": prow["volume"],
            "n_pregame_trades": prow["n_pregame_trades"],
            "n_books": srow["n_books"],
        })

    df = pd.DataFrame(matched).sort_values("game_date").reset_index(drop=True)
    print(f"  Matched {len(df)} games")
    return df


def run_backtest(matched_df: pd.DataFrame, threshold: float,
                 min_volume: float, use_vwap: bool = False):
    """Backtest: when Polymarket price deviates from sharp by > threshold,
    bet the sharp side on Polymarket.

    If poly_home > sharp_home + threshold → poly is overpricing home → buy away
    If poly_away > sharp_away + threshold → poly is overpricing away → buy home
    """
    print(f"\n{'='*80}")
    print(f"  BACKTEST: Sharp vs Polymarket Gap Strategy")
    print(f"  Threshold: {threshold:.0%} | Min volume: ${min_volume:,.0f}")
    print(f"  Price: {'VWAP (5min)' if use_vwap else 'Last trade'}")
    print(f"{'='*80}")

    df = matched_df.copy()
    if min_volume > 0:
        df = df[df["volume"] >= min_volume]
    print(f"  Games after volume filter: {len(df)}")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    poly_home_col = "poly_home_vwap" if use_vwap else "poly_home_prob"

    bets = []
    for _, row in df.iterrows():
        poly_h = row[poly_home_col]
        poly_a = 1.0 - poly_h
        sharp_h = row["sharp_home_prob"]
        sharp_a = row["sharp_away_prob"]

        gap_h = poly_h - sharp_h  # positive = poly overprices home
        gap_a = poly_a - sharp_a  # positive = poly overprices away

        # If poly overprices home → buy away on Polymarket (it's underpriced)
        # If poly overprices away → buy home on Polymarket (it's underpriced)
        if gap_h > threshold and abs(gap_h) >= abs(gap_a):
            # Buy away: price = poly_a, wins if away wins
            side = "away"
            buy_price = poly_a
            edge = gap_h  # how much poly overprices the other side
            won = row["home_win"] == 0
        elif gap_a > threshold:
            # Buy home: price = poly_h, wins if home wins
            side = "home"
            buy_price = poly_h
            edge = gap_a
            won = row["home_win"] == 1
        else:
            continue

        pnl = (1.0 - buy_price) if won else -buy_price

        bets.append({
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "side": side,
            "buy_price": round(buy_price, 3),
            "sharp_fair": round(sharp_h if side == "home" else sharp_a, 3),
            "edge": round(edge, 3),
            "won": int(won),
            "pnl": round(pnl, 4),
            "volume": row["volume"],
        })

    bets_df = pd.DataFrame(bets)
    if bets_df.empty:
        print(f"  No bets at threshold {threshold:.0%}")
        return

    n = len(bets_df)
    wr = bets_df["won"].mean()
    roi = bets_df["pnl"].mean()
    total_pnl = bets_df["pnl"].sum()
    avg_edge = bets_df["edge"].mean()
    avg_price = bets_df["buy_price"].mean()

    print(f"\n  Results:")
    print(f"    Bets: {n}")
    print(f"    Win rate: {wr:.1%}")
    print(f"    Avg edge: {avg_edge:.1%}")
    print(f"    Avg buy price: {avg_price:.3f}")
    print(f"    ROI per bet: {roi:+.2%}")
    print(f"    Total P&L: {total_pnl:+.1f} units (on {n} × $1 bets)")

    # Bootstrap CI
    if n >= 20:
        boot_rois = []
        for _ in range(5000):
            idx = np.random.choice(n, n, replace=True)
            boot_rois.append(bets_df.iloc[idx]["pnl"].mean())
        ci_lo, ci_hi = np.percentile(boot_rois, [2.5, 97.5])
        print(f"    95% CI for ROI: [{ci_lo:+.2%}, {ci_hi:+.2%}]")
        significant = (ci_lo > 0)
        print(f"    Statistically significant: {'YES' if significant else 'NO'}")

    # Monthly breakdown
    bets_df["month"] = pd.to_datetime(bets_df["game_date"]).dt.to_period("M")
    print(f"\n    Monthly:")
    for month, mb in bets_df.groupby("month"):
        mr = mb["pnl"].mean()
        print(f"      {month}: {len(mb):>3} bets, WR={mb['won'].mean():.1%}, "
              f"ROI={mr:+.2%}, PnL={mb['pnl'].sum():+.1f}")

    # By edge bucket
    print(f"\n    By edge bucket:")
    bins = [0, 0.03, 0.05, 0.07, 0.10, 0.15, 1.0]
    bets_df["edge_bucket"] = pd.cut(bets_df["edge"], bins=bins)
    for bucket, gb in bets_df.groupby("edge_bucket", observed=True):
        if len(gb) < 5:
            continue
        print(f"      {bucket}: {len(gb):>3} bets, WR={gb['won'].mean():.1%}, "
              f"ROI={gb['pnl'].mean():+.2%}")

    return bets_df


def sweep_thresholds(matched_df: pd.DataFrame, min_volume: float):
    """Sweep over thresholds to find optimal edge."""
    print(f"\n{'='*80}")
    print(f"  THRESHOLD SWEEP")
    print(f"{'='*80}")
    print(f"  {'Thresh':>7} {'Bets':>6} {'WR':>7} {'ROI':>8} {'PnL':>8} {'AvgEdge':>8}")

    df = matched_df.copy()
    if min_volume > 0:
        df = df[df["volume"] >= min_volume]

    for thresh in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]:
        bets = []
        for _, row in df.iterrows():
            poly_h = row["poly_home_prob"]
            poly_a = row["poly_away_prob"]
            sharp_h = row["sharp_home_prob"]
            sharp_a = row["sharp_away_prob"]
            gap_h = poly_h - sharp_h
            gap_a = poly_a - sharp_a

            if gap_h > thresh and abs(gap_h) >= abs(gap_a):
                buy_price = poly_a
                won = row["home_win"] == 0
            elif gap_a > thresh:
                buy_price = poly_h
                won = row["home_win"] == 1
            else:
                continue
            pnl = (1.0 - buy_price) if won else -buy_price
            bets.append({"won": int(won), "pnl": pnl, "edge": max(gap_h, gap_a)})

        if not bets:
            print(f"  {thresh:>6.0%} {'0':>6}")
            continue

        bdf = pd.DataFrame(bets)
        n = len(bdf)
        wr = bdf["won"].mean()
        roi = bdf["pnl"].mean()
        pnl = bdf["pnl"].sum()
        ae = bdf["edge"].mean()
        print(f"  {thresh:>6.0%} {n:>6} {wr:>6.1%} {roi:>+7.2%} {pnl:>+7.1f} {ae:>7.1%}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Backtest Poly vs Sharp gap strategy")
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--min-volume", type=float, default=500)
    parser.add_argument("--scrape-only", action="store_true",
                        help="Only scrape Polymarket data, don't backtest")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached Polymarket data if available")
    args = parser.parse_args()

    POLY_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Get Polymarket closing prices
    if args.use_cache and CACHE_PATH.exists():
        print("[1] Loading cached Polymarket closing prices...")
        poly_df = pd.read_parquet(CACHE_PATH)
        print(f"  Loaded {len(poly_df)} games from cache")
    else:
        print("[1] Scraping Polymarket settled MLB games...")
        events = fetch_settled_mlb_events()

        print("\n[2] Extracting moneyline markets...")
        games = extract_moneyline_markets(events)

        print(f"\n[3] Fetching pre-game closing prices via trade history...")
        print(f"    This will make ~{len(games)} API calls (~{len(games)*0.15/60:.0f} min)...")
        poly_df = scrape_polymarket_closing_prices(games)

        if len(poly_df) > 0:
            poly_df.to_parquet(CACHE_PATH, index=False)
            print(f"  Cached to {CACHE_PATH}")

    if poly_df.empty:
        print("  No Polymarket data available.")
        return

    print(f"  Polymarket: {len(poly_df)} games, "
          f"{poly_df['game_date'].min()} to {poly_df['game_date'].max()}")

    if args.scrape_only:
        return

    # Step 2: Load sharp lines
    print(f"\n[4] Loading sharp closing lines...")
    sharp_df = load_sharp_lines()

    # Step 3: Match
    print(f"\n[5] Matching Polymarket to sharp lines...")
    matched = match_poly_to_sharp(poly_df, sharp_df)

    if matched.empty:
        print("  No matched games found.")
        return

    # Save matched dataset
    matched_path = POLY_DIR / "poly_vs_sharp_matched.parquet"
    matched.to_parquet(matched_path, index=False)
    print(f"  Saved to {matched_path}")

    # Summary stats
    print(f"\n  Gap statistics:")
    print(f"    Mean |gap|: {matched['max_gap'].mean():.3f}")
    print(f"    Median |gap|: {matched['max_gap'].median():.3f}")
    print(f"    Std |gap|: {matched['max_gap'].std():.3f}")
    print(f"    Games with gap > 4%: {(matched['max_gap'] > 0.04).sum()}")
    print(f"    Games with gap > 7%: {(matched['max_gap'] > 0.07).sum()}")

    # Step 4: Threshold sweep
    sweep_thresholds(matched, args.min_volume)

    # Step 5: Detailed backtest at chosen threshold
    run_backtest(matched, args.threshold, args.min_volume)


if __name__ == "__main__":
    main()
