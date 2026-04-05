#!/usr/bin/env python3
"""
Download and parse MLB O/U (over/under) closing lines from SportsBookReview.

Data source: Pre-built JSON dataset from ArnavSaraogi/mlb-odds-scraper
covering 2021-04-01 through 2025-08-16, with live scraping fallback for
dates beyond that range.

The live scraper hits SBR's Next.js __NEXT_DATA__ endpoint:
  https://www.sportsbookreview.com/betting-odds/mlb-baseball/totals/full-game/?date=YYYY-MM-DD
  https://www.sportsbookreview.com/betting-odds/mlb-baseball/?date=YYYY-MM-DD

Outputs: data/odds/sbr_mlb_2025.parquet

Usage:
    python src/scrape_sbr_odds.py                      # uses JSON dataset
    python src/scrape_sbr_odds.py --scrape-live        # scrape SBR live for any gaps
    python src/scrape_sbr_odds.py --year 2025          # filter to a specific year
"""

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ODDS_DIR = DATA_DIR / "odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

JSON_DATASET = ODDS_DIR / "mlb_odds_dataset.json"
OUTPUT_PARQUET = ODDS_DIR / "sbr_mlb_2025.parquet"

# ---------------------------------------------------------------------------
# Team name mapping: SBR shortName -> our canonical abbreviation
# SBR uses these non-standard codes; all others match already.
# ---------------------------------------------------------------------------

SBR_TO_CANONICAL: dict[str, str] = {
    "CHW": "CWS",   # Chicago White Sox
    "WAS": "WSH",   # Washington Nationals
    # All others (AZ, ATH, SD, SF, etc.) are identical in both systems
}


def canonicalize(abbr: str) -> str:
    return SBR_TO_CANONICAL.get(abbr, abbr)


# ---------------------------------------------------------------------------
# Sportsbook preference for selecting the "canonical" line
# We prefer DraftKings, then FanDuel, then BetMGM, then first available.
# ---------------------------------------------------------------------------

BOOK_PREFERENCE = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet"]


def pick_best_line(odds_list: list[dict], line_type: str) -> dict | None:
    """Return the best sportsbook's opening/closing line dict."""
    if not odds_list:
        return None
    by_book = {o["sportsbook"]: o for o in odds_list}
    for book in BOOK_PREFERENCE:
        if book in by_book:
            return by_book[book]
    # Fallback: first entry
    return odds_list[0]


# ---------------------------------------------------------------------------
# Parse a single game dict from the JSON dataset
# ---------------------------------------------------------------------------

def parse_game(game: dict, game_date: str) -> dict | None:
    """Extract structured row from a game dict. Returns None if incomplete."""
    gv = game.get("gameView", {})
    odds = game.get("odds", {})

    away_abbr = canonicalize(gv.get("awayTeam", {}).get("shortName", ""))
    home_abbr = canonicalize(gv.get("homeTeam", {}).get("shortName", ""))
    if not away_abbr or not home_abbr:
        return None

    away_score = gv.get("awayTeamScore")
    home_score = gv.get("homeTeamScore")
    status = gv.get("gameStatusText", "")

    # Totals lines
    totals = odds.get("totals", [])
    best_total = pick_best_line(totals, "totals")
    ou_open = ou_close = over_open_odds = under_open_odds = over_close_odds = under_close_odds = None
    if best_total:
        ol = best_total.get("openingLine", {})
        cl = best_total.get("currentLine", {})
        ou_open = ol.get("total")
        ou_close = cl.get("total")
        over_open_odds = ol.get("overOdds")
        under_open_odds = ol.get("underOdds")
        over_close_odds = cl.get("overOdds")
        under_close_odds = cl.get("underOdds")

    # Moneyline
    moneylines = odds.get("moneyline", [])
    best_ml = pick_best_line(moneylines, "moneyline")
    home_ml_open = away_ml_open = home_ml_close = away_ml_close = None
    if best_ml:
        ol = best_ml.get("openingLine", {})
        cl = best_ml.get("currentLine", {})
        home_ml_open = ol.get("homeOdds")
        away_ml_open = ol.get("awayOdds")
        home_ml_close = cl.get("homeOdds")
        away_ml_close = cl.get("awayOdds")

    # Compute total runs scored if game is final
    total_runs = None
    if status in ("Final", "F") and away_score is not None and home_score is not None:
        try:
            total_runs = int(away_score) + int(home_score)
        except (ValueError, TypeError):
            pass

    # O/U result: 1=over, 0=under, 0.5=push
    ou_result = None
    if total_runs is not None and ou_close is not None:
        if total_runs > ou_close:
            ou_result = 1.0
        elif total_runs < ou_close:
            ou_result = 0.0
        else:
            ou_result = 0.5  # push

    return {
        "game_date": game_date,
        "away_team": away_abbr,
        "home_team": home_abbr,
        "away_score": away_score,
        "home_score": home_score,
        "total_runs": total_runs,
        "status": status,
        "ou_open": ou_open,
        "ou_close": ou_close,
        "over_open_odds": over_open_odds,
        "under_open_odds": under_open_odds,
        "over_close_odds": over_close_odds,
        "under_close_odds": under_close_odds,
        "home_ml_open": home_ml_open,
        "away_ml_open": away_ml_open,
        "home_ml_close": home_ml_close,
        "away_ml_close": away_ml_close,
        "ou_result": ou_result,
        "book_totals": best_total["sportsbook"] if best_total else None,
        "book_ml": best_ml["sportsbook"] if best_ml else None,
    }


# ---------------------------------------------------------------------------
# Load from JSON dataset
# ---------------------------------------------------------------------------

def load_from_json_dataset(year: int | None = None) -> list[dict]:
    """Parse the local JSON dataset. Filter to year if specified."""
    if not JSON_DATASET.exists():
        print(f"JSON dataset not found at {JSON_DATASET}", file=sys.stderr)
        return []

    print(f"Loading JSON dataset from {JSON_DATASET} ...")
    with open(JSON_DATASET) as f:
        data = json.load(f)

    rows = []
    dates = sorted(data.keys())
    if year:
        dates = [d for d in dates if d.startswith(str(year))]
    print(f"  Dates to process: {len(dates)}")

    for game_date in dates:
        games = data[game_date]
        for game in games:
            row = parse_game(game, game_date)
            if row:
                rows.append(row)

    print(f"  Parsed {len(rows)} games from {len(dates)} dates")
    return rows


# ---------------------------------------------------------------------------
# Live scraper for SBR website (for dates not in the JSON dataset)
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

SBR_BASE = "https://www.sportsbookreview.com/betting-odds/mlb-baseball"


def _fetch_sbr_page(client: httpx.Client, date_str: str, endpoint: str = "totals") -> dict | None:
    """Fetch one SBR page and return the parsed __NEXT_DATA__ JSON."""
    if endpoint == "totals":
        url = f"{SBR_BASE}/totals/full-game/?date={date_str}"
    else:
        url = f"{SBR_BASE}/?date={date_str}"

    try:
        resp = client.get(url, headers=HEADERS, timeout=20, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Warning: fetch failed for {date_str} ({endpoint}): {e}", file=sys.stderr)
        return None

    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        resp.text,
        re.DOTALL,
    )
    if not m:
        return None
    return json.loads(m.group(1))


def _extract_game_rows(next_data: dict) -> list[dict]:
    """Pull gameRows list from parsed __NEXT_DATA__."""
    try:
        return (
            next_data["props"]["pageProps"]["oddsTables"][0]["oddsTableModel"]["gameRows"]
        )
    except (KeyError, IndexError, TypeError):
        return []


def scrape_live_date(client: httpx.Client, game_date: str) -> list[dict]:
    """Scrape SBR live for a single date string (YYYY-MM-DD)."""
    rows_by_game: dict[str, dict] = {}

    # Fetch totals page
    totals_data = _fetch_sbr_page(client, game_date, "totals")
    if totals_data:
        for gr in _extract_game_rows(totals_data):
            gv = gr.get("gameView", {})
            away = canonicalize(gv.get("awayTeam", {}).get("shortName", ""))
            home = canonicalize(gv.get("homeTeam", {}).get("shortName", ""))
            if not away or not home:
                continue
            key = f"{away}@{home}"
            # Filter out None entries from oddsViews
            odds_views = [v for v in gr.get("oddsViews", []) if v is not None]
            rows_by_game[key] = {
                "game_date": game_date,
                "away_team": away,
                "home_team": home,
                "away_score": gv.get("awayTeamScore"),
                "home_score": gv.get("homeTeamScore"),
                "status": gv.get("gameStatusText", ""),
                "_totals_views": odds_views,
                "_ml_views": [],
            }

    # Fetch moneyline page
    ml_data = _fetch_sbr_page(client, game_date, "moneyline")
    if ml_data:
        for gr in _extract_game_rows(ml_data):
            gv = gr.get("gameView", {})
            away = canonicalize(gv.get("awayTeam", {}).get("shortName", ""))
            home = canonicalize(gv.get("homeTeam", {}).get("shortName", ""))
            key = f"{away}@{home}"
            # Filter out None entries from oddsViews
            odds_views = [v for v in gr.get("oddsViews", []) if v is not None]
            if key not in rows_by_game:
                rows_by_game[key] = {
                    "game_date": game_date,
                    "away_team": away,
                    "home_team": home,
                    "away_score": gv.get("awayTeamScore"),
                    "home_score": gv.get("homeTeamScore"),
                    "status": gv.get("gameStatusText", ""),
                    "_totals_views": [],
                    "_ml_views": [],
                }
            rows_by_game[key]["_ml_views"] = odds_views

    # Convert to structured rows
    results = []
    for key, raw in rows_by_game.items():
        # Totals
        totals_views = [
            {
                "sportsbook": v["sportsbook"],
                "openingLine": v.get("openingLine", {}),
                "currentLine": v.get("currentLine", {}),
            }
            for v in raw.get("_totals_views", [])
        ]
        best_t = pick_best_line(totals_views, "totals")

        # ML
        ml_views = [
            {
                "sportsbook": v["sportsbook"],
                "openingLine": v.get("openingLine", {}),
                "currentLine": v.get("currentLine", {}),
            }
            for v in raw.get("_ml_views", [])
        ]
        best_ml = pick_best_line(ml_views, "moneyline")

        ou_open = ou_close = over_open_odds = under_open_odds = None
        over_close_odds = under_close_odds = None
        if best_t:
            ou_open = (best_t.get("openingLine") or {}).get("total")
            ou_close = (best_t.get("currentLine") or {}).get("total")
            over_open_odds = (best_t.get("openingLine") or {}).get("overOdds")
            under_open_odds = (best_t.get("openingLine") or {}).get("underOdds")
            over_close_odds = (best_t.get("currentLine") or {}).get("overOdds")
            under_close_odds = (best_t.get("currentLine") or {}).get("underOdds")

        home_ml_open = away_ml_open = home_ml_close = away_ml_close = None
        if best_ml:
            home_ml_open = (best_ml.get("openingLine") or {}).get("homeOdds")
            away_ml_open = (best_ml.get("openingLine") or {}).get("awayOdds")
            home_ml_close = (best_ml.get("currentLine") or {}).get("homeOdds")
            away_ml_close = (best_ml.get("currentLine") or {}).get("awayOdds")

        away_score = raw.get("away_score")
        home_score = raw.get("home_score")
        status = raw.get("status", "")
        total_runs = None
        if status in ("Final", "F") and away_score is not None and home_score is not None:
            try:
                total_runs = int(away_score) + int(home_score)
            except (ValueError, TypeError):
                pass

        ou_result = None
        if total_runs is not None and ou_close is not None:
            if total_runs > ou_close:
                ou_result = 1.0
            elif total_runs < ou_close:
                ou_result = 0.0
            else:
                ou_result = 0.5

        results.append({
            "game_date": raw["game_date"],
            "away_team": raw["away_team"],
            "home_team": raw["home_team"],
            "away_score": away_score,
            "home_score": home_score,
            "total_runs": total_runs,
            "status": status,
            "ou_open": ou_open,
            "ou_close": ou_close,
            "over_open_odds": over_open_odds,
            "under_open_odds": under_open_odds,
            "over_close_odds": over_close_odds,
            "under_close_odds": under_close_odds,
            "home_ml_open": home_ml_open,
            "away_ml_open": away_ml_open,
            "home_ml_close": home_ml_close,
            "away_ml_close": away_ml_close,
            "ou_result": ou_result,
            "book_totals": best_t["sportsbook"] if best_t else None,
            "book_ml": best_ml["sportsbook"] if best_ml else None,
        })

    return results


def scrape_live_range(start_date: date, end_date: date) -> list[dict]:
    """Scrape SBR live for a date range (inclusive)."""
    rows = []
    current = start_date
    with httpx.Client() as client:
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            print(f"  Scraping {date_str} ...")
            day_rows = scrape_live_date(client, date_str)
            rows.extend(day_rows)
            time.sleep(1.0)  # be polite
            current += timedelta(days=1)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch SBR MLB O/U odds data")
    parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
    parser.add_argument(
        "--scrape-live",
        action="store_true",
        help="Also scrape SBR live for any dates not in the JSON dataset",
    )
    parser.add_argument(
        "--start-date",
        help="Start date for live scraping (YYYY-MM-DD). Defaults to day after JSON dataset ends.",
    )
    parser.add_argument(
        "--end-date",
        help="End date for live scraping (YYYY-MM-DD). Defaults to yesterday.",
    )
    args = parser.parse_args()

    year = args.year

    # Step 1: Load from JSON dataset
    rows = load_from_json_dataset(year=year)

    # Step 2: Optionally scrape live for gaps
    if args.scrape_live:
        # Find the last date in the JSON dataset for this year
        if JSON_DATASET.exists():
            with open(JSON_DATASET) as f:
                data = json.load(f)
            year_dates = sorted(d for d in data.keys() if d.startswith(str(year)))
            if year_dates:
                last_json_date = datetime.strptime(year_dates[-1], "%Y-%m-%d").date()
            else:
                last_json_date = date(year, 1, 1)
        else:
            last_json_date = date(year, 1, 1)

        if args.start_date:
            start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        else:
            start = last_json_date + timedelta(days=1)

        if args.end_date:
            end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        else:
            end = date.today() - timedelta(days=1)

        if start <= end:
            print(f"Scraping SBR live from {start} to {end} ...")
            live_rows = scrape_live_range(start, end)
            print(f"  Got {len(live_rows)} live rows")
            rows.extend(live_rows)
        else:
            print(f"No live scraping needed (start={start} > end={end})")

    if not rows:
        print("No data collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Build DataFrame
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    # Sort and deduplicate (prefer later entries which have final scores)
    df = df.sort_values(["game_date", "away_team", "home_team"])
    df = df.drop_duplicates(subset=["game_date", "away_team", "home_team"], keep="last")
    df = df.reset_index(drop=True)

    # Convert types
    float_cols = [
        "ou_open", "ou_close",
        "over_open_odds", "under_open_odds", "over_close_odds", "under_close_odds",
        "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close",
        "ou_result",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    int_cols = ["away_score", "home_score", "total_runs"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Step 4: Report
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(df)} games for {year}")
    print(f"{'='*60}")
    print(f"\nDate range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))
    print(f"\nO/U close coverage: {df['ou_close'].notna().sum()} / {len(df)} games ({df['ou_close'].notna().mean():.1%})")
    print(f"ML close coverage:  {df['home_ml_close'].notna().sum()} / {len(df)} games ({df['home_ml_close'].notna().mean():.1%})")
    print(f"\nO/U line stats:")
    print(df["ou_close"].describe())
    print(f"\nO/U result distribution (1=over, 0=under, 0.5=push):")
    if df["ou_result"].notna().any():
        print(df["ou_result"].value_counts(dropna=True).sort_index())
    print(f"\nTeams found (home):")
    print(sorted(df["home_team"].unique()))
    print(f"\nTop sportsbooks used for totals:")
    print(df["book_totals"].value_counts().head(10))

    # Step 5: Save
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved to {OUTPUT_PARQUET}")

    return df


if __name__ == "__main__":
    main()
