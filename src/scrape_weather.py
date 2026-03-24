#!/usr/bin/env python3
"""
Scrape weather data for MLB games from the game feed endpoint.

Weather fields: temperature, wind speed, wind direction, condition (dome/outdoor).
Used as features in the win probability model — temperature and wind affect
ball carry and run environment.

Usage:
    python src/scrape_weather.py --seasons 2017 2018 2019 2020 2021 2022 2023 2024 2025
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GAMES_DIR = DATA_DIR / "games"
WEATHER_DIR = DATA_DIR / "weather"
MLB_API = "https://statsapi.mlb.com/api/v1.1"

# Parks with retractable or fixed roofs
DOME_VENUES = {
    # Fixed roofs
    "Tropicana Field",
    "Rogers Centre",       # retractable but often closed
    "T-Mobile Park",       # retractable
    "Minute Maid Park",    # retractable
    "Miller Park",         # retractable (now AmFam)
    "American Family Field",
    "Chase Field",         # retractable
    "Marlins Park",        # retractable
    "loanDepot park",
    "Globe Life Field",    # fixed roof
    "Daikin Park",         # retractable (formerly Minute Maid)
}


def parse_wind(wind_str: str) -> dict:
    """Parse wind string like '10 mph, L To R' into speed and direction."""
    if not wind_str or wind_str == "":
        return {"wind_speed": 0.0, "wind_dir": "none"}

    parts = wind_str.split(",")
    speed = 0.0
    direction = "none"

    if parts:
        speed_part = parts[0].strip().lower()
        try:
            speed = float(speed_part.replace("mph", "").strip())
        except (ValueError, AttributeError):
            speed = 0.0

    if len(parts) > 1:
        direction = parts[1].strip().lower()

    return {"wind_speed": speed, "wind_dir": direction}


async def fetch_game_weather(client: httpx.AsyncClient, game_pk: int) -> dict | None:
    """Fetch weather for a single game from the game feed."""
    try:
        resp = await client.get(f"{MLB_API}/game/{game_pk}/feed/live")
        resp.raise_for_status()
        data = resp.json()

        game_data = data.get("gameData", {})
        weather = game_data.get("weather", {})
        venue = game_data.get("venue", {})

        if not weather:
            return None

        wind = parse_wind(weather.get("wind", ""))
        venue_name = venue.get("name", "")

        return {
            "game_pk": game_pk,
            "temperature": float(weather.get("temp", 0)) if weather.get("temp") else None,
            "wind_speed": wind["wind_speed"],
            "wind_dir": wind["wind_dir"],
            "condition": weather.get("condition", ""),
            "venue_name": venue_name,
            "is_dome": int(venue_name in DOME_VENUES or
                          weather.get("condition", "").lower() in ("dome", "roof closed")),
        }
    except Exception:
        return None


async def scrape_weather_for_season(year: int, max_concurrent: int = 20):
    """Scrape weather for all games in a season."""
    games_path = GAMES_DIR / f"games_{year}.parquet"
    if not games_path.exists():
        print(f"  No games file for {year}")
        return

    games = pd.read_parquet(games_path)
    game_pks = games["game_pk"].tolist()

    output_path = WEATHER_DIR / f"weather_{year}.parquet"

    # Check for existing data to resume
    existing_pks = set()
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        existing_pks = set(existing["game_pk"].tolist())
        remaining = [pk for pk in game_pks if pk not in existing_pks]
        print(f"  {year}: {len(existing_pks)} already scraped, {len(remaining)} remaining")
        if not remaining:
            print(f"  {year}: All done!")
            return
        game_pks = remaining
    else:
        print(f"  {year}: Scraping weather for {len(game_pks)} games")

    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(client, pk):
        async with semaphore:
            result = await fetch_game_weather(client, pk)
            return result

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Process in batches for progress reporting
        batch_size = 200
        for batch_start in range(0, len(game_pks), batch_size):
            batch = game_pks[batch_start:batch_start + batch_size]
            tasks = [fetch_with_limit(client, pk) for pk in batch]
            batch_results = await asyncio.gather(*tasks)

            for r in batch_results:
                if r is not None:
                    results.append(r)

            done = min(batch_start + batch_size, len(game_pks))
            print(f"    {done}/{len(game_pks)} fetched ({len(results)} with weather)")

    if results:
        new_df = pd.DataFrame(results)

        # Merge with existing
        if existing_pks and output_path.exists():
            existing = pd.read_parquet(output_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
            new_df = new_df.drop_duplicates(subset=["game_pk"])

        WEATHER_DIR.mkdir(parents=True, exist_ok=True)
        new_df.to_parquet(output_path, index=False)
        print(f"  Saved {len(new_df)} weather records to {output_path}")
    else:
        print(f"  No weather data retrieved for {year}")


def main():
    parser = argparse.ArgumentParser(description="Scrape MLB weather data")
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=list(range(2017, 2026)))
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    print("MLB Weather Scraper")
    print(f"Seasons: {args.seasons}")

    for year in sorted(args.seasons):
        print(f"\n{'='*40}")
        print(f"Season {year}")
        print(f"{'='*40}")
        asyncio.run(scrape_weather_for_season(year, max_concurrent=args.concurrency))

    print("\nDone.")


if __name__ == "__main__":
    main()
