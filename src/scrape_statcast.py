#!/usr/bin/env python3
"""
Scrape pitch-level Statcast data from Baseball Savant via pybaseball.

Each season is ~700K+ pitches. We scrape day-by-day to stay under the
30K row limit per query, cache everything locally as parquet files, and
can resume if interrupted.

Usage:
    python src/scrape_statcast.py                    # all seasons 2017-2025
    python src/scrape_statcast.py --seasons 2024 2025  # specific seasons
    python src/scrape_statcast.py --resume             # skip already-scraped days
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pybaseball

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "statcast"

# MLB season date ranges (opening day through World Series)
# Using generous bounds — pybaseball returns empty for off-days
SEASON_DATES = {
    2017: (date(2017, 4, 2), date(2017, 11, 1)),
    2018: (date(2018, 3, 29), date(2018, 10, 28)),
    2019: (date(2019, 3, 20), date(2019, 10, 30)),
    2020: (date(2020, 7, 23), date(2020, 10, 27)),  # COVID shortened
    2021: (date(2021, 4, 1), date(2021, 11, 2)),
    2022: (date(2022, 4, 7), date(2022, 11, 5)),
    2023: (date(2023, 3, 30), date(2023, 11, 1)),
    2024: (date(2024, 3, 20), date(2024, 10, 30)),
    2025: (date(2025, 3, 18), date(2025, 10, 29)),
    2026: (date(2026, 3, 26), date(2026, 10, 28)),
}

# Columns we definitely need for the model. We'll keep everything but
# this defines the minimum viable set.
CORE_COLUMNS = [
    # identifiers
    "game_pk", "game_date", "game_year", "at_bat_number", "pitch_number",
    "pitcher", "batter", "home_team", "away_team",
    "inning", "inning_topbot",

    # pitch characteristics
    "pitch_type", "release_speed", "release_spin_rate",
    "release_extension", "release_pos_x", "release_pos_z",
    "pfx_x", "pfx_z", "plate_x", "plate_z",
    "effective_speed", "spin_axis",

    # count / game state
    "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b",
    "home_score", "away_score",

    # result
    "description", "events", "type",  # B/S/X
    "zone", "stand", "p_throws",

    # batted ball
    "launch_speed", "launch_angle", "hit_distance_sc",
    "hc_x", "hc_y", "bb_type",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom",
    "babip_value", "iso_value",

    # fielding
    "if_fielding_alignment", "of_fielding_alignment",

    # bat tracking (2024+)
    "bat_speed", "swing_length",

    # game metadata
    "game_type",  # R=regular, F/D/L/W=postseason
]


def scrape_day(dt: date) -> pd.DataFrame | None:
    """Scrape a single day of Statcast data."""
    dt_str = dt.strftime("%Y-%m-%d")
    try:
        df = pybaseball.statcast(dt_str, dt_str, parallel=False, verbose=False)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        print(f"    ERROR on {dt_str}: {e}")
        return None


def scrape_season(year: int, resume: bool = True):
    """Scrape an entire season day-by-day, saving to parquet."""
    if year not in SEASON_DATES:
        print(f"No date range defined for {year}")
        return

    season_dir = DATA_DIR / str(year)
    season_dir.mkdir(parents=True, exist_ok=True)
    combined_path = DATA_DIR / f"statcast_{year}.parquet"

    start, end = SEASON_DATES[year]

    # If scraping current/future season, cap at today
    today = date.today()
    if end > today:
        end = today - timedelta(days=1)

    if start > today:
        print(f"  {year} hasn't started yet, skipping")
        return

    total_days = (end - start).days + 1
    total_pitches = 0
    day_files = []

    print(f"\n{'='*60}")
    print(f"Season {year}: {start} → {end} ({total_days} days)")
    print(f"{'='*60}")

    dt = start
    while dt <= end:
        dt_str = dt.strftime("%Y-%m-%d")
        day_path = season_dir / f"{dt_str}.parquet"

        if resume and day_path.exists():
            try:
                df = pd.read_parquet(day_path)
                n = len(df)
                if n > 0:
                    total_pitches += n
                    day_files.append(day_path)
                dt += timedelta(days=1)
                continue
            except Exception:
                pass  # re-scrape corrupted files

        df = scrape_day(dt)

        if df is not None and not df.empty:
            n = len(df)
            total_pitches += n
            df.to_parquet(day_path, index=False)
            day_files.append(day_path)
            days_done = (dt - start).days + 1
            print(f"  {dt_str}: {n:,} pitches  "
                  f"[{days_done}/{total_days} days, {total_pitches:,} total]")
        else:
            # Save empty marker so resume skips this day
            pd.DataFrame().to_parquet(day_path, index=False)

        dt += timedelta(days=1)

        # Be polite to Baseball Savant
        time.sleep(0.2)

    # Combine all days into one season file
    if day_files:
        print(f"\n  Combining {len(day_files)} day files...")
        chunks = []
        for fp in sorted(day_files):
            try:
                chunk = pd.read_parquet(fp)
                if not chunk.empty:
                    chunks.append(chunk)
            except Exception:
                continue

        if chunks:
            season_df = pd.concat(chunks, ignore_index=True)

            # Keep only columns that exist in the data
            available = [c for c in CORE_COLUMNS if c in season_df.columns]
            extra = [c for c in season_df.columns if c not in CORE_COLUMNS]

            # Save full dataset (all columns)
            season_df.to_parquet(combined_path, index=False)

            print(f"  Season {year}: {len(season_df):,} total pitches, "
                  f"{len(season_df.columns)} columns")
            print(f"  Saved to {combined_path}")
            print(f"  Core columns present: {len(available)}/{len(CORE_COLUMNS)}")
            if extra:
                print(f"  Extra columns: {len(extra)}")

    return total_pitches


def main():
    parser = argparse.ArgumentParser(description="Scrape Statcast data")
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=list(SEASON_DATES.keys()),
                        help="Seasons to scrape (default: all)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-scraped days (default: True)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-scrape everything")
    args = parser.parse_args()

    resume = not args.no_resume

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Statcast Scraper")
    print(f"Seasons: {args.seasons}")
    print(f"Resume mode: {resume}")
    print(f"Output dir: {DATA_DIR}")

    grand_total = 0
    for year in sorted(args.seasons):
        count = scrape_season(year, resume=resume)
        if count:
            grand_total += count

    print(f"\n{'='*60}")
    print(f"Done. {grand_total:,} total pitches across {len(args.seasons)} seasons.")


if __name__ == "__main__":
    main()
