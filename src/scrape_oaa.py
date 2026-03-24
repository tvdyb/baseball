#!/usr/bin/env python3
"""
Scrape per-player OAA (Outs Above Average) from Baseball Savant.

Uses pybaseball's statcast_outs_above_average() for positions 3-9
(1B through RF; catchers not supported by the OAA leaderboard).

Output: data/oaa/oaa_all.parquet
  Columns: player_id, season, position, oaa, fielding_runs_prevented
"""

import argparse
from pathlib import Path

import pandas as pd
from pybaseball import statcast_outs_above_average

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OAA_DIR = DATA_DIR / "oaa"

POSITIONS = list(range(3, 10))  # 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF


def scrape_oaa(years: list[int]):
    OAA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OAA_DIR / "oaa_all.parquet"

    frames = []
    for year in sorted(years):
        for pos in POSITIONS:
            print(f"  {year} pos={pos}...", end=" ")
            try:
                df = statcast_outs_above_average(year, pos=str(pos), min_att=1)
                if df is not None and len(df) > 0:
                    df = df.rename(columns={
                        "outs_above_average": "oaa",
                    })
                    df["season"] = year
                    df["position"] = pos
                    # Keep only the columns we need
                    keep = ["player_id", "season", "position", "oaa", "fielding_runs_prevented"]
                    keep = [c for c in keep if c in df.columns]
                    frames.append(df[keep])
                    print(f"{len(df)} players")
                else:
                    print("empty")
            except Exception as e:
                print(f"error: {e}")

    if not frames:
        print("No OAA data scraped")
        return

    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(output_path, index=False)
    print(f"\nSaved {len(result)} rows to {output_path}")
    print(f"  Seasons: {sorted(result['season'].unique())}")
    print(f"  Positions: {sorted(result['position'].unique())}")
    print(f"  Unique players: {result['player_id'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description="Scrape OAA data")
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=list(range(2017, 2026)))
    args = parser.parse_args()

    print("Scraping OAA data from Baseball Savant")
    scrape_oaa(args.seasons)
    print("Done.")


if __name__ == "__main__":
    main()
