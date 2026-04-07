#!/usr/bin/env python3
"""
Pinnacle vs Polymarket MLB Moneyline Comparison
================================================

Pulls current MLB moneyline odds from:
  1. Pinnacle (via The Odds API) — devigged to fair probabilities
  2. Polymarket (via Gamma API) — already in probability format

Matches games by team+date, computes implied probability gaps, and flags
any game where the gap exceeds 4 cents (configurable).

Usage:
    export ODDS_API_KEY="your_key_here"
    python src/pinnacle_poly_compare.py
    python src/pinnacle_poly_compare.py --threshold 0.03
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import requests

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
MLB_TAG_ID = 100381

# Polymarket uses full team names; The Odds API does too.
# Build a normalization map for common variations.
TEAM_ALIASES = {
    "arizona diamondbacks": "diamondbacks",
    "atlanta braves": "braves",
    "baltimore orioles": "orioles",
    "boston red sox": "red sox",
    "chicago cubs": "cubs",
    "chicago white sox": "white sox",
    "cincinnati reds": "reds",
    "cleveland guardians": "guardians",
    "colorado rockies": "rockies",
    "detroit tigers": "tigers",
    "houston astros": "astros",
    "kansas city royals": "royals",
    "los angeles angels": "angels",
    "los angeles dodgers": "dodgers",
    "miami marlins": "marlins",
    "milwaukee brewers": "brewers",
    "minnesota twins": "twins",
    "new york mets": "mets",
    "new york yankees": "yankees",
    "oakland athletics": "athletics",
    "philadelphia phillies": "phillies",
    "pittsburgh pirates": "pirates",
    "san diego padres": "padres",
    "san francisco giants": "giants",
    "seattle mariners": "mariners",
    "st. louis cardinals": "cardinals",
    "st louis cardinals": "cardinals",
    "tampa bay rays": "rays",
    "texas rangers": "rangers",
    "toronto blue jays": "blue jays",
    "washington nationals": "nationals",
}


def normalize_team(name: str) -> str:
    """Normalize team name to a short canonical form for matching."""
    lower = name.strip().lower()
    if lower in TEAM_ALIASES:
        return TEAM_ALIASES[lower]
    # Fallback: return last word (usually the mascot)
    return lower.split()[-1]


def game_key(team_a: str, team_b: str, date_str: str) -> str:
    """Create a canonical key: sorted normalized teams + date."""
    teams = sorted([normalize_team(team_a), normalize_team(team_b)])
    return f"{teams[0]}|{teams[1]}|{date_str}"


# ═══════════════════════════════════════════════════════════════════
# 1. Pinnacle via The Odds API
# ═══════════════════════════════════════════════════════════════════

def american_to_implied(odds: float) -> float:
    """Convert American odds to raw implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    return 0.5


def devig_multiplicative(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove vig via multiplicative method (proportional scaling)."""
    total = prob_a + prob_b
    if total == 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def fetch_pinnacle_odds(api_key: str) -> list[dict]:
    """Fetch MLB h2h odds from Pinnacle via The Odds API."""
    params = {
        "apiKey": api_key,
        "bookmakers": "pinnacle",
        "markets": "h2h",
        "oddsFormat": "american",
    }
    resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
    resp.raise_for_status()

    # Log remaining quota
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"  Odds API quota: {remaining} remaining, {used} used")

    games = []
    for event in resp.json():
        home = event["home_team"]
        away = event["away_team"]
        commence = event["commence_time"]

        # Convert UTC commence time to US Eastern for date matching
        # (Polymarket uses ET game dates; a 1am UTC game is a 9pm ET game the day before)
        utc_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        et_dt = utc_dt - timedelta(hours=4)  # EDT = UTC-4
        date_str = et_dt.strftime("%Y-%m-%d")

        # Find Pinnacle bookmaker
        for bk in event.get("bookmakers", []):
            if bk["key"] != "pinnacle":
                continue
            for mkt in bk.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                odds_map = {}
                for outcome in mkt["outcomes"]:
                    odds_map[outcome["name"]] = outcome["price"]

                home_odds = odds_map.get(home)
                away_odds = odds_map.get(away)
                if home_odds is None or away_odds is None:
                    continue

                home_raw = american_to_implied(home_odds)
                away_raw = american_to_implied(away_odds)
                home_fair, away_fair = devig_multiplicative(home_raw, away_raw)

                games.append({
                    "home_team": home,
                    "away_team": away,
                    "date": date_str,
                    "commence": commence,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "home_raw": home_raw,
                    "away_raw": away_raw,
                    "vig": home_raw + away_raw - 1.0,
                    "pin_home_prob": home_fair,
                    "pin_away_prob": away_fair,
                    "key": game_key(home, away, date_str),
                })
    return games


# ═══════════════════════════════════════════════════════════════════
# 2. Polymarket via Gamma API
# ═══════════════════════════════════════════════════════════════════

def fetch_polymarket_mlb() -> list[dict]:
    """Fetch active MLB moneyline markets from Polymarket Gamma API."""
    url = f"{GAMMA_API_BASE}/events"
    events = []
    offset = 0
    while True:
        params = {
            "tag_id": MLB_TAG_ID,
            "active": "true",
            "closed": "false",
            "limit": 100,
            "offset": offset,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        events.extend(batch)
        if len(batch) < 100:
            break
        offset += 100

    games = []
    for event in events:
        title = event.get("title", "")
        slug = event.get("ticker", "") or event.get("slug", "")

        # Extract game date from slug (e.g., "mlb-hou-sea-2026-04-13")
        # The date is the last 10 characters in YYYY-MM-DD format
        slug_match = re.search(r"(\d{4}-\d{2}-\d{2})$", slug)
        if slug_match:
            date_str = slug_match.group(1)
        else:
            # Fallback to startDate (less reliable — it's market creation time)
            start = event.get("startDate", "")
            date_str = start[:10] if start else ""

        # Skip non-game events (futures, props, etc.)
        if " vs" not in title.lower():
            continue

        for market in event.get("markets", []):
            group_title = (market.get("groupItemTitle") or "").strip().lower()
            # Moneyline markets have groupItemTitle="" or "moneyline"
            # Skip spreads, totals, NRFI, etc.
            if group_title and group_title != "moneyline":
                continue

            # Parse outcomes and prices (they're JSON strings)
            try:
                outcomes = json.loads(market.get("outcomes", "[]"))
                prices = json.loads(market.get("outcomePrices", "[]"))
            except (json.JSONDecodeError, TypeError):
                continue

            if len(outcomes) != 2 or len(prices) != 2:
                continue

            # Skip totals/spreads by checking outcome names
            outcome_lower = [o.lower() for o in outcomes]
            if any(x in outcome_lower for x in ["over", "under", "yes", "no"]):
                continue

            # Determine which is home/away from the "vs." title
            # Polymarket title format: "Away Team vs. Home Team"
            # But we can't always tell — match by team names
            team_a, team_b = outcomes[0], outcomes[1]
            price_a, price_b = float(prices[0]), float(prices[1])

            # Try to determine home/away from event title
            # Polymarket: "Team A vs. Team B" — Team B is usually home
            vs_parts = title.replace(" vs. ", "|").replace(" vs ", "|").split("|")
            if len(vs_parts) == 2:
                away_name = vs_parts[0].strip()
                home_name = vs_parts[1].strip()
            else:
                away_name = team_a
                home_name = team_b

            # Map outcomes to home/away
            norm_a = normalize_team(team_a)
            norm_home = normalize_team(home_name)
            if norm_a == norm_home:
                poly_home_prob = price_a
                poly_away_prob = price_b
                home_team = team_a
                away_team = team_b
            else:
                poly_home_prob = price_b
                poly_away_prob = price_a
                home_team = team_b
                away_team = team_a

            games.append({
                "home_team": home_team,
                "away_team": away_team,
                "date": date_str,
                "poly_home_prob": poly_home_prob,
                "poly_away_prob": poly_away_prob,
                "volume": float(market.get("volume", 0) or 0),
                "liquidity": float(market.get("liquidity", 0) or 0),
                "key": game_key(home_team, away_team, date_str),
            })
            break  # Only take the moneyline market per event

    return games


# ═══════════════════════════════════════════════════════════════════
# 3. Match & Compare
# ═══════════════════════════════════════════════════════════════════

def compare(pinnacle_games: list, poly_games: list, threshold: float) -> list[dict]:
    """Match games and compute probability gaps."""
    poly_by_key = {g["key"]: g for g in poly_games}

    matched = []
    for pg in pinnacle_games:
        pm = poly_by_key.get(pg["key"])
        if pm is None:
            continue

        gap_home = pm["poly_home_prob"] - pg["pin_home_prob"]
        gap_away = pm["poly_away_prob"] - pg["pin_away_prob"]
        max_gap = max(abs(gap_home), abs(gap_away))

        matched.append({
            "date": pg["date"],
            "time": pg["commence"][11:16],
            "away": pg["away_team"],
            "home": pg["home_team"],
            "pin_home": pg["pin_home_prob"],
            "pin_away": pg["pin_away_prob"],
            "pin_home_odds": pg["home_odds"],
            "pin_away_odds": pg["away_odds"],
            "pin_vig": pg["vig"],
            "poly_home": pm["poly_home_prob"],
            "poly_away": pm["poly_away_prob"],
            "gap_home": gap_home,
            "gap_away": gap_away,
            "max_gap": max_gap,
            "flag": max_gap >= threshold,
            "poly_volume": pm.get("volume", 0),
            "poly_liquidity": pm.get("liquidity", 0),
        })

    matched.sort(key=lambda x: x["max_gap"], reverse=True)
    return matched


# ═══════════════════════════════════════════════════════════════════
# 4. Display
# ═══════════════════════════════════════════════════════════════════

def print_table(rows: list, threshold: float):
    """Print formatted comparison table."""
    if not rows:
        print("\nNo matched games found.")
        return

    print(f"\n{'='*105}")
    print(f"  Pinnacle (devigged) vs Polymarket — MLB Moneyline")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Threshold: {threshold:.0%} | Matched games: {len(rows)}")
    print(f"{'='*105}")
    print(f"  {'Game':<35} {'Pin H':>6} {'Pin A':>6} {'Poly H':>6} {'Poly A':>6} "
          f"{'Gap H':>7} {'Gap A':>7} {'Flag':>5}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*5}")

    n_flagged = 0
    for r in rows:
        flag_str = " >>>" if r["flag"] else ""
        if r["flag"]:
            n_flagged += 1
        away_short = r["away"].split()[-1][:10]
        home_short = r["home"].split()[-1][:10]
        game_str = f"{away_short} @ {home_short} ({r['time']})"

        print(f"  {game_str:<35} {r['pin_home']:>5.1%} {r['pin_away']:>5.1%} "
              f"{r['poly_home']:>5.1%} {r['poly_away']:>5.1%} "
              f"{r['gap_home']:>+6.1%} {r['gap_away']:>+6.1%} {flag_str}")

    print(f"\n  Flagged: {n_flagged}/{len(rows)} games with gap >= {threshold:.0%}")

    if n_flagged > 0:
        print(f"\n  {'='*70}")
        print(f"  FLAGGED GAMES (gap >= {threshold:.0%}):")
        print(f"  {'='*70}")
        for r in rows:
            if not r["flag"]:
                continue
            # Determine which side has the bigger mispricing
            if abs(r["gap_home"]) >= abs(r["gap_away"]):
                side = "HOME"
                team = r["home"]
                gap = r["gap_home"]
                poly_p = r["poly_home"]
                pin_p = r["pin_home"]
            else:
                side = "AWAY"
                team = r["away"]
                gap = r["gap_away"]
                poly_p = r["poly_away"]
                pin_p = r["pin_away"]

            direction = "POLY OVERPRICED" if gap > 0 else "POLY UNDERPRICED"
            print(f"\n  {r['away']} @ {r['home']}")
            print(f"    {direction}: {team} ({side})")
            print(f"    Pinnacle fair: {pin_p:.1%}  |  Polymarket: {poly_p:.1%}  |  Gap: {gap:+.1%}")
            print(f"    Pinnacle odds: Home {r['pin_home_odds']:+d} / Away {r['pin_away_odds']:+d} (vig: {r['pin_vig']:.1%})")
            print(f"    Poly volume: ${r['poly_volume']:,.0f}  |  Liquidity: ${r['poly_liquidity']:,.0f}")

    print()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compare Pinnacle vs Polymarket MLB odds")
    parser.add_argument("--threshold", type=float, default=0.04,
                        help="Flag games with probability gap >= this (default: 0.04 = 4 cents)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of table")
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ERROR: Set ODDS_API_KEY environment variable")
        print("  Get a free key at https://the-odds-api.com")
        sys.exit(1)

    # Fetch from both sources
    print("[1] Fetching Pinnacle odds via The Odds API...")
    try:
        pin_games = fetch_pinnacle_odds(api_key)
    except requests.HTTPError as e:
        print(f"  ERROR fetching Pinnacle: {e}")
        sys.exit(1)
    print(f"  Found {len(pin_games)} Pinnacle MLB games")

    print("\n[2] Fetching Polymarket MLB markets...")
    try:
        poly_games = fetch_polymarket_mlb()
    except requests.HTTPError as e:
        print(f"  ERROR fetching Polymarket: {e}")
        sys.exit(1)
    print(f"  Found {len(poly_games)} Polymarket MLB games")

    # Match and compare
    print("\n[3] Matching games...")
    matched = compare(pin_games, poly_games, args.threshold)
    print(f"  Matched {len(matched)} games")

    # Unmatched diagnostics
    pin_keys = {g["key"] for g in pin_games}
    poly_keys = {g["key"] for g in poly_games}
    only_pin = pin_keys - poly_keys
    only_poly = poly_keys - pin_keys
    if only_pin:
        print(f"  Pinnacle-only (no Polymarket match): {len(only_pin)}")
    if only_poly:
        print(f"  Polymarket-only (no Pinnacle match): {len(only_poly)}")

    if args.json:
        print(json.dumps(matched, indent=2, default=str))
    else:
        print_table(matched, args.threshold)


if __name__ == "__main__":
    main()
