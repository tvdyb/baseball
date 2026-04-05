#!/usr/bin/env python3
"""
Scrape historical MLB O/U closing lines from The Odds API.

Fetches moneyline, totals (O/U), and spread (run line) odds for MLB games.
Prefers DraftKings lines, falls back to FanDuel, then any available bookmaker.

Usage:
    python src/scrape_odds.py --year 2025 --api-key YOUR_KEY
    python src/scrape_odds.py --year 2025  # reads ODDS_API_KEY env var
"""

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd


def log(msg: str):
    print(msg)
    sys.stdout.flush()


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ODDS_DIR = DATA_DIR / "odds"

API_BASE = "https://api.the-odds-api.com/v4"

# Bookmaker preference order: prefer DraftKings, then FanDuel, then anything
BOOKMAKER_PREFERENCE = ["draftkings", "fanduel"]

ODDS_TEAM_MAP = {
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
}


# ---------------------------------------------------------------------------
# Vig removal helpers
# ---------------------------------------------------------------------------

def american_to_prob(price: int) -> float:
    """Convert American odds to implied probability."""
    if price > 0:
        return 100 / (price + 100)
    else:
        return -price / (-price + 100)


def remove_vig(over_price: int, under_price: int) -> tuple[float, float]:
    """Remove vig from over/under prices, return fair probabilities."""
    p_over = american_to_prob(over_price)
    p_under = american_to_prob(under_price)
    total = p_over + p_under
    return p_over / total, p_under / total


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_get_with_retry(
    client: httpx.Client, url: str, params: dict, max_retries: int = 3,
) -> httpx.Response:
    """GET with retry and exponential backoff. Raises on final failure."""
    for attempt in range(max_retries):
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError) as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 3
                log(f"  Retry {attempt + 1}/{max_retries} after error: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise


def _log_quota(resp: httpx.Response):
    """Log API quota info from response headers."""
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if remaining is not None or used is not None:
        log(f"  API quota: {used} used, {remaining} remaining")


def _map_team(full_name: str) -> str | None:
    """Map full team name to our abbreviation."""
    abbr = ODDS_TEAM_MAP.get(full_name)
    if abbr is None:
        log(f"  WARNING: Unknown team name: {full_name}")
    return abbr


# ---------------------------------------------------------------------------
# Bookmaker selection
# ---------------------------------------------------------------------------

def _pick_bookmaker(bookmakers: list[dict]) -> dict | None:
    """Pick the best bookmaker from list, following preference order.

    Returns the bookmaker dict, or None if no bookmakers available.
    """
    if not bookmakers:
        return None

    by_key = {b["key"]: b for b in bookmakers}

    for preferred in BOOKMAKER_PREFERENCE:
        if preferred in by_key:
            return by_key[preferred]

    # Fall back to first available
    return bookmakers[0]


def _extract_market(bookmaker: dict, market_key: str) -> dict | None:
    """Extract a specific market from a bookmaker's markets list."""
    for m in bookmaker.get("markets", []):
        if m["key"] == market_key:
            return m
    return None


# ---------------------------------------------------------------------------
# Parsing odds from API response
# ---------------------------------------------------------------------------

def _parse_game_odds(game: dict) -> dict | None:
    """Parse a single game from the API response into our row format.

    Returns a dict with our column schema, or None if essential data is missing.
    """
    # Map teams
    home_full = game.get("home_team", "")
    away_full = game.get("away_team", "")
    home_team = _map_team(home_full)
    away_team = _map_team(away_full)
    if not home_team or not away_team:
        return None

    # Parse game date from commence_time (ISO 8601)
    commence = game.get("commence_time", "")
    if not commence:
        return None
    try:
        dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        # Convert to Eastern-ish date: subtract 5 hours to handle games
        # that start late at night UTC but are still the same calendar day
        eastern_approx = dt - timedelta(hours=5)
        game_date = eastern_approx.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None

    # Pick best bookmaker
    bookmaker = _pick_bookmaker(game.get("bookmakers", []))
    if not bookmaker:
        return None

    bookmaker_key = bookmaker["key"]

    # Initialize row with defaults
    row = {
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "ou_line": None,
        "ou_over_price": None,
        "ou_under_price": None,
        "ml_home_price": None,
        "ml_away_price": None,
        "spread_line": None,
        "spread_home_price": None,
        "spread_away_price": None,
        "bookmaker": bookmaker_key,
    }

    # Extract totals (O/U)
    totals = _extract_market(bookmaker, "totals")
    if totals:
        for outcome in totals.get("outcomes", []):
            if outcome["name"] == "Over":
                row["ou_line"] = outcome.get("point")
                row["ou_over_price"] = outcome.get("price")
            elif outcome["name"] == "Under":
                row["ou_under_price"] = outcome.get("price")

    # Extract moneyline (h2h)
    h2h = _extract_market(bookmaker, "h2h")
    if h2h:
        for outcome in h2h.get("outcomes", []):
            if outcome["name"] == home_full:
                row["ml_home_price"] = outcome.get("price")
            elif outcome["name"] == away_full:
                row["ml_away_price"] = outcome.get("price")

    # Extract spreads (run line)
    spreads = _extract_market(bookmaker, "spreads")
    if spreads:
        for outcome in spreads.get("outcomes", []):
            if outcome["name"] == home_full:
                row["spread_line"] = outcome.get("point")
                row["spread_home_price"] = outcome.get("price")
            elif outcome["name"] == away_full:
                row["spread_away_price"] = outcome.get("price")

    return row


# ---------------------------------------------------------------------------
# Fetching odds
# ---------------------------------------------------------------------------

def fetch_historical_odds(
    client: httpx.Client, api_key: str, scrape_date: str,
) -> list[dict]:
    """Fetch historical odds for a single date.

    Args:
        client: httpx client
        api_key: The Odds API key
        scrape_date: date string like "2025-07-15"

    Returns:
        List of parsed game row dicts.
    """
    # Use 15:00 UTC (~11am ET) to get closing lines before typical first pitch
    timestamp = f"{scrape_date}T15:00:00Z"

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,totals,spreads",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel",
        "date": timestamp,
    }

    url = f"{API_BASE}/historical/sports/baseball_mlb/odds"

    resp = _api_get_with_retry(client, url, params)
    data = resp.json()

    # The historical endpoint wraps games in a "data" key
    games = data.get("data", [])
    if isinstance(games, dict):
        # Sometimes the response nests further
        games = games.get("data", [])

    rows = []
    for game in games:
        row = _parse_game_odds(game)
        if row:
            rows.append(row)

    return rows, resp


def fetch_current_odds(client: httpx.Client, api_key: str) -> list[dict]:
    """Fetch current/live odds snapshot.

    Returns:
        List of parsed game row dicts.
    """
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,totals,spreads",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel",
    }

    url = f"{API_BASE}/sports/baseball_mlb/odds"

    resp = _api_get_with_retry(client, url, params)
    games = resp.json()

    rows = []
    for game in games:
        row = _parse_game_odds(game)
        if row:
            rows.append(row)

    return rows, resp


# ---------------------------------------------------------------------------
# Season scraping
# ---------------------------------------------------------------------------

def _season_dates(year: int) -> list[str]:
    """Generate all dates in the MLB season (April 1 - October 31)."""
    start = date(year, 4, 1)
    end = date(year, 10, 31)
    dates = []
    current = start
    while current <= end:
        dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates


def _load_existing(output_path: Path) -> set[str]:
    """Load already-scraped dates from existing parquet file."""
    if not output_path.exists():
        return set()
    try:
        df = pd.read_parquet(output_path)
        if "game_date" in df.columns:
            return set(df["game_date"].unique())
    except Exception as e:
        log(f"  WARNING: Could not read existing file: {e}")
    return set()


def scrape_season(api_key: str, year: int, output_path: Path):
    """Scrape historical odds for an entire MLB season.

    Skips dates already present in the output parquet. Appends new data
    incrementally and saves after each batch of dates.
    """
    all_dates = _season_dates(year)
    existing_dates = _load_existing(output_path)

    dates_to_scrape = [d for d in all_dates if d not in existing_dates]

    if not dates_to_scrape:
        log(f"  All {len(all_dates)} dates already scraped for {year}.")
        return

    log(f"  {len(dates_to_scrape)} dates to scrape "
        f"({len(existing_dates)} already done out of {len(all_dates)} total)")

    # Load existing data to append to
    existing_rows = []
    if output_path.exists():
        try:
            existing_df = pd.read_parquet(output_path)
            existing_rows = existing_df.to_dict("records")
        except Exception:
            pass

    all_rows = list(existing_rows)
    new_games = 0
    api_errors = 0
    dates_with_no_games = 0
    first_quota_logged = False

    with httpx.Client(timeout=30.0) as client:
        for i, scrape_date in enumerate(dates_to_scrape):
            try:
                rows, resp = fetch_historical_odds(client, api_key, scrape_date)

                if not first_quota_logged:
                    _log_quota(resp)
                    first_quota_logged = True

                if rows:
                    all_rows.extend(rows)
                    new_games += len(rows)
                else:
                    dates_with_no_games += 1

                if (i + 1) % 10 == 0 or i + 1 == len(dates_to_scrape):
                    log(f"  [{i + 1}/{len(dates_to_scrape)}] "
                        f"date={scrape_date}, {len(rows)} games, "
                        f"total new={new_games}")

            except httpx.HTTPStatusError as e:
                api_errors += 1
                status = e.response.status_code
                if status == 422:
                    # Date not available (e.g., before API coverage starts)
                    if (i + 1) % 50 == 0:
                        log(f"  [{i + 1}/{len(dates_to_scrape)}] "
                            f"date={scrape_date}: not available (422)")
                elif status == 429:
                    log(f"  Rate limited at date={scrape_date}. "
                        f"Saving progress and stopping.")
                    _log_quota(e.response)
                    break
                elif status == 401:
                    log(f"  ERROR: Invalid API key (401). Aborting.")
                    break
                else:
                    log(f"  ERROR on {scrape_date}: HTTP {status}")
            except Exception as e:
                api_errors += 1
                log(f"  ERROR on {scrape_date}: {e}")

            # Rate limit: 0.5s between requests
            time.sleep(0.5)

            # Save progress every 50 dates
            if (i + 1) % 50 == 0 and new_games > 0:
                _save_dataframe(all_rows, output_path)
                log(f"  Checkpoint saved: {len(all_rows)} total rows")

    # Final save
    if new_games > 0:
        _save_dataframe(all_rows, output_path)

    # Log final quota
    log(f"\n  Scraping complete:")
    log(f"    New games: {new_games}")
    log(f"    Dates with no games: {dates_with_no_games}")
    log(f"    API errors: {api_errors}")


def _save_dataframe(rows: list[dict], output_path: Path):
    """Save rows to parquet, deduplicating by (game_date, home_team, away_team)."""
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return

    # Deduplicate: keep last occurrence (most recent scrape wins)
    df = df.drop_duplicates(
        subset=["game_date", "home_team", "away_team"],
        keep="last",
    )
    df = df.sort_values(["game_date", "home_team"]).reset_index(drop=True)

    # Cast numeric columns
    for col in ["ou_line", "ou_over_price", "ou_under_price",
                 "ml_home_price", "ml_away_price",
                 "spread_line", "spread_home_price", "spread_away_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_parquet(output_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape MLB odds from The Odds API"
    )
    parser.add_argument("--year", type=int, default=2025,
                        help="Season year to scrape (default: 2025)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="The Odds API key (or set ODDS_API_KEY env var)")
    parser.add_argument("--current", action="store_true",
                        help="Fetch current/live odds instead of historical")
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        log("ERROR: No API key provided. Use --api-key or set ODDS_API_KEY env var.")
        sys.exit(1)

    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ODDS_DIR / f"odds_mlb_{args.year}.parquet"

    if args.current:
        log(f"\nFetching current MLB odds...")
        with httpx.Client(timeout=30.0) as client:
            rows, resp = fetch_current_odds(client, api_key)
            _log_quota(resp)

        if rows:
            _save_dataframe(rows, output_path)
            log(f"  Saved {len(rows)} games to {output_path}")
        else:
            log("  No games found.")
    else:
        log(f"\nScraping historical MLB odds for {args.year}...")
        scrape_season(api_key, args.year, output_path)

    # Summary stats
    if output_path.exists():
        df = pd.read_parquet(output_path)
        log(f"\nDataset summary ({output_path}):")
        log(f"  Total games: {len(df)}")
        if len(df) > 0:
            log(f"  Date range: {df['game_date'].min()} -> {df['game_date'].max()}")
            log(f"  Unique dates: {df['game_date'].nunique()}")

            has_ou = df["ou_line"].notna().sum()
            has_ml = df["ml_home_price"].notna().sum()
            has_spread = df["spread_line"].notna().sum()
            log(f"  Games with O/U: {has_ou} ({has_ou / len(df):.1%})")
            log(f"  Games with ML: {has_ml} ({has_ml / len(df):.1%})")
            log(f"  Games with spread: {has_spread} ({has_spread / len(df):.1%})")

            if has_ou > 0:
                log(f"  Avg O/U line: {df['ou_line'].mean():.2f}")

            bookmaker_counts = df["bookmaker"].value_counts()
            log(f"  Bookmaker breakdown:")
            for bk, count in bookmaker_counts.items():
                log(f"    {bk}: {count} ({count / len(df):.1%})")


if __name__ == "__main__":
    main()
