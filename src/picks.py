#!/usr/bin/env python3
"""
Generate market-making picks for tomorrow's MLB games.

Fetches model probabilities and live Kalshi prices, identifies edges,
and outputs a picks sheet with timing guidance for market making.

Usage:
    python src/picks.py                  # tomorrow's games
    python src/picks.py --date 2026-04-01
    python src/picks.py --refresh        # re-fetch Kalshi prices only (skip model retrain)
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import (
    load_training_data, train_ensemble, predict_games,
    fetch_todays_games, build_live_features,
)
from win_model import EDGE_THRESHOLDS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PICKS_DIR = DATA_DIR / "picks"

# Kalshi API (public, no auth needed for market listing)
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

# ── MLB schedule timing constants ─────────────────────────────────────────────
# All times ET. These are typical patterns, not guarantees.
#
# MARKET LIFECYCLE:
#   ~9pm prior night  : Kalshi posts next-day markets (thin, wide spreads)
#   6-8am ET          : Early morning — markets exist but very thin
#   10-11am ET        : Probable pitchers confirmed on MLB.com (key catalyst)
#   11am-1pm ET       : Lineups posted (usually ~2-3 hrs before first pitch)
#   ~1pm ET           : First pitch window opens (day games)
#   ~6:30-7pm ET      : Main slate first pitches
#
# MARKET-MAKING WINDOWS:
#   AVOID: Before 10am ET — markets too thin, you'll be the only liquidity
#   POST FAIR VALUES: 10am-12pm ET — after SP confirmed, before lineups
#     - Your model has SP quality + team priors but NOT lineup-specific matchups
#     - Quote wide (±3-4¢) since lineups haven't dropped yet
#   TIGHTEN ON LINEUPS: 12pm-2pm ET for day games, 4-6pm ET for night games
#     - Lineups confirm platoon splits and matchup features
#     - This is where vol spikes — other MMs reprice, you should too
#     - Re-run `python src/picks.py --refresh` to update with lineup data
#   CLOSE TO FIRST PITCH: Last 30min before first pitch
#     - Highest volume, tightest spreads, most competition
#     - Your edge is smallest here (market is most efficient)
#
# VOLATILITY CATALYSTS (re-run model when these hit):
#   1. Probable pitcher change (rare but huge — ±5-10¢ move)
#   2. Lineup card posted (~2-3hr pre-game) — ±1-3¢
#   3. Late scratch / injury report — ±2-5¢
#   4. Weather delays / PPD risk — wide spreads, pull quotes


def fetch_kalshi_live_markets(target_date: str) -> pd.DataFrame:
    """Fetch current Kalshi prices for tomorrow's MLB games.

    Uses the public /markets endpoint filtered to KXMLBGAME series.
    Returns DataFrame with columns: event_ticker, home_team, away_team,
    kalshi_yes_price, kalshi_volume, market_ticker.
    """
    rows = []
    cursor = ""

    with httpx.Client(timeout=30.0) as client:
        while True:
            params = {
                "series_ticker": "KXMLBGAME",
                "status": "open",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            resp = client.get(f"{KALSHI_API}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()

            markets = data.get("markets", [])
            if not markets:
                break

            for m in markets:
                ticker = m.get("ticker", "")
                event_ticker = m.get("event_ticker", "")
                # Parse the event ticker to get date and teams
                from scrape_kalshi import parse_event_ticker
                parsed = parse_event_ticker(event_ticker)
                if not parsed:
                    continue
                if parsed["game_date"] != target_date:
                    continue

                # Determine if this is the home team market
                subtitle = m.get("subtitle", "").lower()
                yes_sub = m.get("yes_sub_title", "").lower()
                title = m.get("title", "").lower()

                # Kalshi API returns dollar-denominated fields (e.g. 0.55 = 55¢)
                yes_price = m.get("yes_ask_dollars") or m.get("yes_ask")
                yes_bid = m.get("yes_bid_dollars") or m.get("yes_bid")
                no_ask = m.get("no_ask_dollars") or m.get("no_ask")
                no_bid = m.get("no_bid_dollars") or m.get("no_bid")
                last_price = m.get("last_price_dollars") or m.get("last_price")
                volume = m.get("volume_fp") or m.get("volume") or 0

                # Convert string dollar amounts to float if needed
                for var_name in ["yes_price", "yes_bid", "no_ask", "no_bid", "last_price"]:
                    val = locals()[var_name]
                    if isinstance(val, str):
                        try:
                            locals()[var_name] = float(val)
                        except (ValueError, TypeError):
                            locals()[var_name] = None
                yes_price = float(yes_price) if yes_price else None
                yes_bid = float(yes_bid) if yes_bid else None
                no_ask = float(no_ask) if no_ask else None
                no_bid = float(no_bid) if no_bid else None
                last_price = float(last_price) if last_price else None
                volume = float(volume) if volume else 0

                # Mid price from best bid/ask
                if yes_bid and yes_price:
                    mid = (yes_bid + yes_price) / 2
                elif last_price:
                    mid = last_price
                else:
                    mid = None

                rows.append({
                    "event_ticker": event_ticker,
                    "market_ticker": ticker,
                    "game_date": parsed["game_date"],
                    "home_team": parsed["home_team"],
                    "away_team": parsed["away_team"],
                    "kalshi_yes_bid": yes_bid,
                    "kalshi_yes_ask": yes_price,
                    "kalshi_mid": mid,
                    "kalshi_no_bid": no_bid,
                    "kalshi_no_ask": no_ask,
                    "kalshi_last": last_price,
                    "kalshi_volume": volume,
                    "kalshi_spread": (yes_price - yes_bid) if yes_price and yes_bid else None,
                    "title": m.get("title", ""),
                })

            cursor = data.get("cursor", "")
            if not cursor:
                break

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Deduplicate: keep one market per (home_team, away_team) pair
    # Kalshi has YES/NO markets — we want the home team YES market
    # Group by game and pick the market where the title mentions the home team
    return df


def _identify_home_market(group: pd.DataFrame) -> pd.Series:
    """From a group of markets for the same game, identify the home-team YES market.

    Kalshi tickers end with team abbrev: KXMLBGAME-26MAR30...-DET
    The YES price for that market is the probability of that team winning.
    We want the home team's market so YES = home win prob.
    """
    home = group.iloc[0]["home_team"]
    for _, row in group.iterrows():
        ticker = row.get("market_ticker", "")
        # Ticker suffix after last dash is the team
        team_suffix = ticker.rsplit("-", 1)[-1] if "-" in ticker else ""
        if team_suffix == home:
            return row
    # Fallback: check title
    for _, row in group.iterrows():
        if home.lower() in row.get("title", "").lower():
            return row
    return group.iloc[0]


def generate_picks(target_date: str, lr_only: bool = False) -> pd.DataFrame:
    """Generate picks with model probabilities and live Kalshi prices."""
    print(f"\n{'='*70}")
    print(f"  MLB Market-Making Picks for {target_date}")
    print(f"{'='*70}")

    # ── Model predictions ─────────────────────────────────────────────
    print("\nTraining model...")
    train_df = load_training_data()
    train_df["game_date"] = train_df["game_date"].astype(str)
    n_before = len(train_df)
    train_df = train_df[train_df["game_date"] < target_date]
    if len(train_df) < n_before:
        print(f"  Filtered out {n_before - len(train_df)} games on/after {target_date}")
    print(f"  {len(train_df):,} training games (through {train_df['game_date'].max()})")
    lr, scaler, xgb_model, lr_features, xgb_features, w_lr, train_medians = train_ensemble(
        train_df, lr_only=lr_only
    )

    print(f"\nFetching schedule for {target_date}...")
    with httpx.Client(timeout=15.0) as client:
        games = fetch_todays_games(client, target_date)
        if not games:
            print("  No games found!")
            return pd.DataFrame()

        print(f"  Found {len(games)} games")
        print("\nBuilding features...")
        features_df = build_live_features(games, target_date, client)

    results = predict_games(
        lr, scaler, xgb_model, lr_features, xgb_features,
        features_df, w_lr, train_medians,
    )

    # Attach SP names and confidence
    for col in ["home_sp_name", "away_sp_name", "game_time", "status"]:
        if col in features_df.columns:
            results[col] = features_df[col].values
    if "home_sp_info_confidence" in features_df.columns:
        results["sp_confidence"] = (
            features_df["home_sp_info_confidence"].fillna(0)
            + features_df["away_sp_info_confidence"].fillna(0)
        ) / 2

    # ── Kalshi live prices ────────────────────────────────────────────
    print(f"\nFetching Kalshi live markets...")
    kalshi_df = fetch_kalshi_live_markets(target_date)

    if len(kalshi_df) > 0:
        # For each game, find the home-team market
        home_markets = []
        for (ht, at), group in kalshi_df.groupby(["home_team", "away_team"]):
            home_row = _identify_home_market(group)
            home_markets.append(home_row)
        kalshi_home = pd.DataFrame(home_markets)
        print(f"  Found {len(kalshi_home)} Kalshi markets")

        results = results.merge(
            kalshi_home[["home_team", "away_team", "kalshi_mid", "kalshi_yes_bid",
                         "kalshi_yes_ask", "kalshi_spread", "kalshi_volume",
                         "kalshi_last", "market_ticker"]],
            on=["home_team", "away_team"],
            how="left",
        )
    else:
        print("  No Kalshi markets found (markets may not be posted yet)")
        results["kalshi_mid"] = np.nan
        results["kalshi_yes_bid"] = np.nan
        results["kalshi_yes_ask"] = np.nan
        results["kalshi_spread"] = np.nan
        results["kalshi_volume"] = np.nan
        results["kalshi_last"] = np.nan
        results["market_ticker"] = ""

    # ── Compute edges ─────────────────────────────────────────────────
    results["model_fair"] = results["home_win_prob"]
    results["edge"] = np.where(
        results["kalshi_mid"].notna(),
        results["model_fair"] - results["kalshi_mid"],
        np.nan,
    )
    results["abs_edge"] = results["edge"].abs()

    # ── Suggested quotes ──────────────────────────────────────────────
    # Wide quotes pre-lineup (±3¢), tighter post-lineup (±1.5¢)
    results["quote_bid"] = (results["model_fair"] - 0.03).clip(0.01, 0.99)
    results["quote_ask"] = (results["model_fair"] + 0.03).clip(0.01, 0.99)

    # ── Conviction tier ───────────────────────────────────────────────
    results["conviction"] = pd.cut(
        results["abs_edge"],
        bins=[-1, 0.02, 0.04, 0.07, 1.0],
        labels=["skip", "watch", "lean", "strong"],
    )

    # ── Sort by absolute edge ─────────────────────────────────────────
    results = results.sort_values("abs_edge", ascending=False)

    return results


def display_picks(picks: pd.DataFrame, target_date: str):
    """Print picks in a market-making friendly format."""
    if len(picks) == 0:
        print("  No picks to display.")
        return

    # Parse game times for display
    now_utc = datetime.now(timezone.utc)

    print(f"\n{'='*90}")
    print(f"  PICKS SHEET — {target_date}")
    print(f"  Generated: {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*90}")

    # Summary
    has_kalshi = picks["kalshi_mid"].notna().sum()
    print(f"\n  {len(picks)} games, {has_kalshi} with Kalshi prices")

    if has_kalshi > 0:
        strong = (picks["abs_edge"] >= 0.05).sum()
        lean = ((picks["abs_edge"] >= 0.03) & (picks["abs_edge"] < 0.05)).sum()
        print(f"  Strong edges (≥5%): {strong}")
        print(f"  Lean edges (3-5%): {lean}")

    # Detail table
    print(f"\n  {'Matchup':<25s} {'Pitchers':<30s} {'Model':>6s} {'Kalshi':>6s} "
          f"{'Edge':>6s} {'Spread':>6s} {'Vol':>5s} {'Action':>8s}")
    print(f"  {'-'*98}")

    for _, r in picks.iterrows():
        matchup = f"{r['away_team']} @ {r['home_team']}"
        away_sp = str(r.get("away_sp_name", "TBD"))[:14]
        home_sp = str(r.get("home_sp_name", "TBD"))[:14]
        pitchers = f"{away_sp} v {home_sp}"

        model_str = f"{r['model_fair']:.1%}"
        kalshi_str = f"{r['kalshi_mid']:.1%}" if pd.notna(r.get("kalshi_mid")) else "  ---"
        edge_str = f"{r['edge']:+.1%}" if pd.notna(r.get("edge")) else "  ---"
        spread_str = f"{r['kalshi_spread']:.0%}" if pd.notna(r.get("kalshi_spread")) else "  ---"
        vol_str = f"{int(r['kalshi_volume']):>5d}" if pd.notna(r.get("kalshi_volume")) else "  ---"

        conv = str(r.get("conviction", ""))
        if conv == "strong":
            action = "STRONG"
        elif conv == "lean":
            action = "LEAN"
        elif conv == "watch":
            action = "watch"
        else:
            action = "skip"

        # Confidence warning
        conf = r.get("sp_confidence", 1.0)
        warn = " ⚠" if pd.notna(conf) and conf < 0.3 else ""

        print(f"  {matchup:<25s} {pitchers:<30s} {model_str:>6s} {kalshi_str:>6s} "
              f"{edge_str:>6s} {spread_str:>6s} {vol_str:>5s} {action:>8s}{warn}")

    # Suggested quotes for strong/lean picks
    actionable = picks[picks["conviction"].isin(["strong", "lean"])]
    if len(actionable) > 0:
        print(f"\n{'='*90}")
        print(f"  SUGGESTED QUOTES (pre-lineup, ±3¢ around fair value)")
        print(f"{'='*90}")
        print(f"  {'Matchup':<25s} {'Side':>6s} {'Fair':>6s} {'Bid':>6s} {'Ask':>6s} "
              f"{'Kalshi Bid':>10s} {'Kalshi Ask':>10s}")
        print(f"  {'-'*75}")

        for _, r in actionable.iterrows():
            matchup = f"{r['away_team']} @ {r['home_team']}"
            side = "HOME" if r.get("edge", 0) > 0 else "AWAY"
            fair = f"{r['model_fair']:.1%}"
            bid = f"{r['quote_bid']:.1%}"
            ask = f"{r['quote_ask']:.1%}"
            k_bid = f"{r['kalshi_yes_bid']:.1%}" if pd.notna(r.get("kalshi_yes_bid")) else "---"
            k_ask = f"{r['kalshi_yes_ask']:.1%}" if pd.notna(r.get("kalshi_yes_ask")) else "---"
            print(f"  {matchup:<25s} {side:>6s} {fair:>6s} {bid:>6s} {ask:>6s} "
                  f"{k_bid:>10s} {k_ask:>10s}")

    # Timing guidance
    print(f"\n{'='*90}")
    print(f"  TIMING GUIDE")
    print(f"{'='*90}")
    print(f"  10-11am ET  SPs confirmed on MLB.com → post initial quotes (wide ±3-4¢)")
    print(f"  11am-1pm ET Lineups drop → re-run with --refresh, tighten to ±1.5-2¢")
    print(f"  1-2pm ET    Day game first pitches → highest vol window for day slate")
    print(f"  4-6pm ET    Night lineups drop → re-run, tighten quotes for night slate")
    print(f"  6-7pm ET    Night first pitches → peak volume, tightest spreads")
    print(f"  ")
    print(f"  RE-PRICE ON: SP change, lineup card, injury scratch, weather delay")
    print(f"  PULL QUOTES: PPD announced, SP scratched with no replacement yet")


def save_picks(picks: pd.DataFrame, target_date: str):
    """Save picks to CSV and JSON for downstream consumption."""
    PICKS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = PICKS_DIR / f"picks_{target_date}.csv"
    cols = [
        "game_date", "home_team", "away_team",
        "home_sp_name", "away_sp_name", "game_time",
        "model_fair", "home_win_prob_lr", "home_win_prob_xgb",
        "kalshi_mid", "kalshi_yes_bid", "kalshi_yes_ask",
        "kalshi_spread", "kalshi_volume",
        "edge", "abs_edge", "conviction",
        "quote_bid", "quote_ask",
        "sp_confidence", "market_ticker",
    ]
    out_cols = [c for c in cols if c in picks.columns]
    picks[out_cols].to_csv(csv_path, index=False)
    print(f"\n  Saved CSV: {csv_path}")

    # JSON (for programmatic consumption)
    json_path = PICKS_DIR / f"picks_{target_date}.json"
    records = []
    for _, r in picks.iterrows():
        rec = {
            "game_date": target_date,
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_sp": r.get("home_sp_name", "TBD"),
            "away_sp": r.get("away_sp_name", "TBD"),
            "model_fair": round(float(r["model_fair"]), 4),
            "kalshi_mid": round(float(r["kalshi_mid"]), 4) if pd.notna(r.get("kalshi_mid")) else None,
            "edge": round(float(r["edge"]), 4) if pd.notna(r.get("edge")) else None,
            "conviction": str(r.get("conviction", "")),
            "quote_bid": round(float(r["quote_bid"]), 4),
            "quote_ask": round(float(r["quote_ask"]), 4),
            "market_ticker": r.get("market_ticker", ""),
        }
        records.append(rec)

    with open(json_path, "w") as f:
        json.dump({"date": target_date, "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "picks": records}, f, indent=2)
    print(f"  Saved JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="MLB Market-Making Picks")
    parser.add_argument("--date", type=str,
                        help="Target date (YYYY-MM-DD). Default: tomorrow")
    parser.add_argument("--lr-only", action="store_true",
                        help="Use logistic regression only")
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    picks = generate_picks(target_date, lr_only=args.lr_only)
    if len(picks) == 0:
        return

    display_picks(picks, target_date)
    save_picks(picks, target_date)


if __name__ == "__main__":
    main()
