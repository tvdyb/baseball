#!/usr/bin/env python3
"""
Scrape Polymarket MLB game market data.

Fetches closing probabilities for MLB game markets from Polymarket's
gamma API and CLOB price-history endpoint.

Limitations:
  - Polymarket started daily MLB game markets in 2025.
    For 2024, only ~40 playoff games had markets.
  - The CLOB prices-history endpoint only retains data for ~1 week
    after market resolution. Older markets return empty history.
  - For markets without CLOB price history, the script cannot
    determine pre-game closing probabilities.

Usage:
    python src/scrape_polymarket.py --year 2024
    python src/scrape_polymarket.py --year 2025
    python src/scrape_polymarket.py --year 2026
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POLY_DIR = DATA_DIR / "polymarket"
GAMES_DIR = DATA_DIR / "games"

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

RATE_LIMIT = 0.3  # seconds between CLOB requests

# Polymarket team names → MLB abbreviations
TEAM_NAME_TO_ABBR = {
    # Full names
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "ATH",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
    # Short names (used in outcomes)
    "Diamondbacks": "AZ",
    "D-backs": "AZ",
    "Braves": "ATL",
    "Orioles": "BAL",
    "Red Sox": "BOS",
    "Cubs": "CHC",
    "White Sox": "CWS",
    "Reds": "CIN",
    "Guardians": "CLE",
    "Rockies": "COL",
    "Tigers": "DET",
    "Astros": "HOU",
    "Royals": "KC",
    "Angels": "LAA",
    "Dodgers": "LAD",
    "Marlins": "MIA",
    "Brewers": "MIL",
    "Twins": "MIN",
    "Mets": "NYM",
    "Yankees": "NYY",
    "Phillies": "PHI",
    "Pirates": "PIT",
    "Padres": "SD",
    "Giants": "SF",
    "Mariners": "SEA",
    "Cardinals": "STL",
    "Rays": "TB",
    "Rangers": "TEX",
    "Blue Jays": "TOR",
    "Nationals": "WSH",
}


def resolve_team_abbr(name: str) -> str | None:
    """Resolve a team name (full or short) to MLB abbreviation."""
    name = name.strip()
    if name in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[name]
    # Try substring match
    for full_name, abbr in TEAM_NAME_TO_ABBR.items():
        if full_name in name or name in full_name:
            return abbr
    return None


def is_game_market(market: dict) -> bool:
    """Check if a market is a head-to-head game (not props, series, futures)."""
    outcomes = market.get("outcomes", [])
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)

    # Must have exactly 2 outcomes that are team names (not Yes/No)
    if len(outcomes) != 2:
        return False
    if "Yes" in outcomes or "No" in outcomes:
        return False

    # Both outcomes must map to known teams
    t0 = resolve_team_abbr(outcomes[0])
    t1 = resolve_team_abbr(outcomes[1])
    if not t0 or not t1:
        return False
    # Must be different teams
    if t0 == t1:
        return False

    return True


def is_excluded_event(title: str) -> bool:
    """Exclude futures, series winners, props, etc."""
    excludes = [
        "Series Winner", "Champion", "Winner", "Props",
        "playoffs", "MVP", "Cy Young", "Division",
        "Home Run Leader", "Steals Leader", "RBI Leader",
        "Outcome", "CBA", "sale", "record", "Jersey Jerry",
        "Juan Soto", "sweep", "Sweep",
    ]
    for ex in excludes:
        if ex.lower() in title.lower():
            return True
    return False


def parse_game_date_from_event(event: dict, market: dict) -> str | None:
    """Extract game date from event/market metadata."""
    # Try gameStartTime first (most reliable)
    gst = market.get("gameStartTime")
    if gst:
        try:
            dt = datetime.fromisoformat(
                gst.replace("Z", "+00:00") if gst.endswith("Z") else gst
            )
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

    # Try endDate
    end = market.get("endDate") or event.get("endDate")
    if end:
        try:
            dt = datetime.fromisoformat(
                end.replace("Z", "+00:00") if end.endswith("Z") else end
            )
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

    # Try parsing date from title (e.g., "MLB Games: March 27" or "ATL Braves vs. SD Padres March 27")
    title = market.get("question", "") or event.get("title", "")
    # Match month name + day
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})",
        title,
    )
    if m:
        month_name = m.group(1)
        day = int(m.group(2))
        start_date = event.get("startDate", "")
        year = int(start_date[:4]) if start_date else None
        if year:
            month = datetime.strptime(month_name, "%B").month
            return f"{year}-{month:02d}-{day:02d}"

    # Try startDateIso
    sdi = market.get("startDateIso")
    if sdi:
        return sdi[:10]

    return None


def fetch_all_mlb_events(year: int) -> list[dict]:
    """Fetch all MLB events from Polymarket gamma API with pagination."""
    all_events = []
    offset = 0

    with httpx.Client(timeout=30.0) as client:
        while True:
            params = {
                "tag_slug": "mlb",
                "closed": "true",
                "limit": 100,
                "offset": offset,
            }
            resp = client.get(f"{GAMMA_API}/events", params=params)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            for event in data:
                start = event.get("startDate", "")
                if start[:4] == str(year):
                    all_events.append(event)

            if len(data) < 100:
                break

            offset += 100
            time.sleep(0.1)

    print(f"  Fetched {len(all_events)} MLB events for {year}")
    return all_events


def extract_game_markets(events: list[dict], year: int) -> list[dict]:
    """Extract individual game markets from events."""
    games = []
    seen = set()  # deduplicate by (date, team0, team1)

    for event in events:
        title = event.get("title", "")
        if is_excluded_event(title):
            continue

        markets = event.get("markets", [])
        for market in markets:
            if not is_game_market(market):
                continue

            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)

            outcome_prices = market.get("outcomePrices", [])
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)

            team0_name = outcomes[0]
            team1_name = outcomes[1]
            team0 = resolve_team_abbr(team0_name)
            team1 = resolve_team_abbr(team1_name)

            if not team0 or not team1:
                continue

            game_date = parse_game_date_from_event(event, market)
            if not game_date:
                continue

            # Check year matches
            if not game_date.startswith(str(year)):
                continue

            # Determine winner from resolved prices
            try:
                p0 = float(outcome_prices[0])
                p1 = float(outcome_prices[1])
            except (ValueError, IndexError, TypeError):
                continue

            team0_won = p0 > p1

            # Get clob token IDs for price history
            clob_ids = market.get("clobTokenIds", "")
            if isinstance(clob_ids, str) and clob_ids:
                clob_ids = json.loads(clob_ids)

            # Dedup key
            key = (game_date, *sorted([team0, team1]))
            if key in seen:
                continue
            seen.add(key)

            games.append({
                "game_date": game_date,
                "team0": team0,
                "team1": team1,
                "team0_name": team0_name,
                "team1_name": team1_name,
                "team0_won": team0_won,
                "volume": float(market.get("volumeNum") or market.get("volume") or 0),
                "clob_token_ids": clob_ids if isinstance(clob_ids, list) else [],
                "game_start_time": market.get("gameStartTime"),
                "closed_time": market.get("closedTime"),
                "market_id": market.get("id"),
                "event_title": title,
            })

    print(f"  Found {len(games)} individual game markets")
    return games


def fetch_closing_price(
    client: httpx.Client, token_id: str, game_start_ts: float | None
) -> float | None:
    """Fetch closing price (last pre-game price) from CLOB price history."""
    try:
        resp = client.get(
            f"{CLOB_API}/prices-history",
            params={
                "market": token_id,
                "interval": "1w",
                "fidelity": 10,
            },
        )
        resp.raise_for_status()
        history = resp.json().get("history", [])

        if not history:
            return None

        if game_start_ts:
            # Get last price before game start
            pre_game = [h for h in history if h["t"] < game_start_ts]
            if pre_game:
                return float(pre_game[-1]["p"])

        # Fallback: find last "reasonable" price (between 0.05 and 0.95)
        # Working backwards from the end
        for h in reversed(history):
            p = float(h["p"])
            if 0.05 <= p <= 0.95:
                return p

        # Last resort: first entry
        return float(history[0]["p"])

    except Exception:
        return None


def fetch_all_closing_prices(games: list[dict]) -> list[dict]:
    """Fetch closing prices for all games."""
    results = []
    n_success = 0

    with httpx.Client(timeout=15.0) as client:
        for i, game in enumerate(games):
            token_ids = game.get("clob_token_ids", [])
            if not token_ids:
                game["poly_team0_prob"] = None
                results.append(game)
                continue

            # Parse game start time
            game_start_ts = None
            gst = game.get("game_start_time")
            if gst:
                try:
                    dt = datetime.fromisoformat(
                        gst.replace("Z", "+00:00") if gst.endswith("Z") else gst
                    )
                    game_start_ts = dt.timestamp()
                except (ValueError, TypeError):
                    pass

            # Fetch price for team0 (first token)
            price = fetch_closing_price(client, token_ids[0], game_start_ts)

            if price is not None:
                game["poly_team0_prob"] = round(price, 4)
                game["poly_team1_prob"] = round(1 - price, 4)
                n_success += 1
            else:
                game["poly_team0_prob"] = None
                game["poly_team1_prob"] = None

            results.append(game)
            time.sleep(RATE_LIMIT)

            if (i + 1) % 50 == 0 or i + 1 == len(games):
                print(f"    {i + 1}/{len(games)} processed ({n_success} with prices)")

    print(f"  Got closing prices for {n_success}/{len(games)} games")
    return results


def build_raw_df(games: list[dict]) -> pd.DataFrame:
    """Build raw DataFrame from game data."""
    rows = []
    for g in games:
        if g.get("poly_team0_prob") is None:
            continue
        rows.append({
            "game_date": g["game_date"],
            "team0": g["team0"],
            "team1": g["team1"],
            "team0_won": g["team0_won"],
            "poly_team0_prob": g["poly_team0_prob"],
            "poly_team1_prob": g["poly_team1_prob"],
            "volume": g["volume"],
            "event_title": g["event_title"],
            "market_id": g.get("market_id"),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("game_date").reset_index(drop=True)
    return df


def match_to_game_results(raw_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Match Polymarket data to MLB game results."""
    games_path = GAMES_DIR / f"games_{year}.parquet"
    if not games_path.exists():
        print(f"  WARNING: {games_path} not found, returning raw data")
        return raw_df

    games = pd.read_parquet(games_path)
    games["game_date"] = games["game_date"].astype(str)

    # Build lookup by (date, home_team, away_team) and (date, away_team, home_team)
    # Since Polymarket team order may not match home/away
    matched_rows = []

    for _, row in raw_df.iterrows():
        date = row["game_date"]
        t0 = row["team0"]
        t1 = row["team1"]

        # Try to find the game in results
        mask = (
            (games["game_date"] == date)
            & (
                ((games["home_team_abbr"] == t0) & (games["away_team_abbr"] == t1))
                | ((games["home_team_abbr"] == t1) & (games["away_team_abbr"] == t0))
            )
        )

        matched = games[mask]
        if len(matched) == 0:
            # Try adjacent dates (Polymarket dates can be off by one due to timezone)
            for delta in [-1, 1]:
                alt_date = (
                    pd.Timestamp(date) + pd.Timedelta(days=delta)
                ).strftime("%Y-%m-%d")
                mask2 = (
                    (games["game_date"] == alt_date)
                    & (
                        ((games["home_team_abbr"] == t0) & (games["away_team_abbr"] == t1))
                        | ((games["home_team_abbr"] == t1) & (games["away_team_abbr"] == t0))
                    )
                )
                matched = games[mask2]
                if len(matched) > 0:
                    break

        if len(matched) == 0:
            continue

        # Take the first match (for doubleheaders, this may not be perfect)
        game = matched.iloc[0]
        home_team = game["home_team_abbr"]
        away_team = game["away_team_abbr"]
        home_win = int(game["home_win"])
        game_pk = int(game["game_pk"])

        # Determine poly_home_prob
        if t0 == home_team:
            poly_home_prob = row["poly_team0_prob"]
        else:
            poly_home_prob = row["poly_team1_prob"]

        matched_rows.append({
            "game_date": game["game_date"],
            "team0": t0,
            "team1": t1,
            "team0_won": row["team0_won"],
            "poly_team0_prob": row["poly_team0_prob"],
            "poly_team1_prob": row["poly_team1_prob"],
            "volume": row["volume"],
            "home_team": home_team,
            "away_team": away_team,
            "home_win": home_win,
            "game_pk": game_pk,
            "poly_home_prob": round(poly_home_prob, 4),
        })

    result = pd.DataFrame(matched_rows)
    if len(result) > 0:
        result = result.sort_values("game_date").reset_index(drop=True)
    print(f"  Matched {len(result)}/{len(raw_df)} games to MLB results")
    return result


def main():
    parser = argparse.ArgumentParser(description="Scrape Polymarket MLB market data")
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    POLY_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = POLY_DIR / f"poly_mlb_{args.year}_raw.parquet"
    matched_path = POLY_DIR / f"poly_mlb_{args.year}_matched.parquet"

    # Step 1: Fetch all MLB events
    print(f"\nFetching Polymarket MLB events for {args.year}...")
    events = fetch_all_mlb_events(args.year)

    # Step 2: Extract individual game markets
    print(f"\nExtracting game markets...")
    games = extract_game_markets(events, args.year)

    if not games:
        print(f"\n  No game markets found for {args.year}.")
        print("  Note: Polymarket started offering daily MLB game markets in 2025.")
        print("  For 2024, only a handful of playoff games are available.")
        return

    # Step 3: Fetch closing prices from CLOB
    print(f"\nFetching closing prices from CLOB price history...")
    games = fetch_all_closing_prices(games)

    # Step 4: Build and save raw data
    print(f"\nBuilding raw dataset...")
    raw_df = build_raw_df(games)
    if len(raw_df) == 0:
        print("  No games with valid closing prices found.")
        print("  Price history may be unavailable for older resolved markets.")
        return

    raw_df.to_parquet(raw_path, index=False)
    print(f"  Saved {len(raw_df)} games to {raw_path}")

    # Step 5: Match to game results
    print(f"\nMatching to MLB game results...")
    matched_df = match_to_game_results(raw_df, args.year)

    if len(matched_df) > 0:
        matched_df.to_parquet(matched_path, index=False)
        print(f"  Saved {len(matched_df)} matched games to {matched_path}")

        # Summary
        print(f"\n  Date range: {matched_df['game_date'].min()} to {matched_df['game_date'].max()}")
        if "poly_home_prob" in matched_df.columns:
            print(f"  Avg Poly home prob: {matched_df['poly_home_prob'].mean():.3f}")
            print(f"  Home win rate: {matched_df['home_win'].mean():.3f}")
        print(f"  Avg volume/game: ${matched_df['volume'].mean():,.0f}")
    else:
        print("  No games could be matched to MLB results.")


if __name__ == "__main__":
    main()
