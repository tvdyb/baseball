#!/usr/bin/env python3
"""
Scrape per-player sprint speed data from Baseball Savant.
This is a yearly aggregate (ft/sec in fastest 1-sec window).
We join this onto pitch-level data by batter_id + season.
"""

from pathlib import Path
import pandas as pd
import pybaseball

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def scrape_sprint_speed(years: list[int]):
    out_dir = DATA_DIR / "sprint_speed"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for year in years:
        print(f"Fetching sprint speed for {year}...")
        try:
            df = pybaseball.statcast_sprint_speed(year, min_opp=1)
            df["season"] = year
            df = df.rename(columns={"player_id": "batter", "sprint_speed": "sprint_speed"})
            df = df[["batter", "season", "sprint_speed"]].copy()
            all_dfs.append(df)
            print(f"  {len(df)} players")
        except Exception as e:
            print(f"  Error: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = out_dir / "sprint_speed_all.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"\nSaved {len(combined)} records to {out_path}")
        return combined


if __name__ == "__main__":
    scrape_sprint_speed(list(range(2017, 2026)))
