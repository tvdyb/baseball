#!/usr/bin/env python3
"""
Scrape game-level data: schedules, results, starting pitchers, lineups.

This gives us:
  - Game outcomes (the label we're predicting)
  - Starting pitcher for each game
  - Park/venue for park factors
  - Home/away designation

Uses the MLB Stats API (statsapi.mlb.com) which is free and has no auth.
"""

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GAMES_DIR = DATA_DIR / "games"
MLB_API = "https://statsapi.mlb.com/api/v1"

SEASON_DATES = {
    2017: (date(2017, 4, 2), date(2017, 11, 1)),
    2018: (date(2018, 3, 29), date(2018, 10, 28)),
    2019: (date(2019, 3, 20), date(2019, 10, 30)),
    2020: (date(2020, 7, 23), date(2020, 10, 27)),
    2021: (date(2021, 4, 1), date(2021, 11, 2)),
    2022: (date(2022, 4, 7), date(2022, 11, 5)),
    2023: (date(2023, 3, 30), date(2023, 11, 1)),
    2024: (date(2024, 3, 20), date(2024, 10, 30)),
    2025: (date(2025, 3, 18), date(2025, 10, 29)),
}


def fetch_schedule(client: httpx.Client, start: date, end: date) -> list[dict]:
    """Fetch games from the MLB Stats API schedule endpoint."""
    resp = client.get(
        f"{MLB_API}/schedule",
        params={
            "sportId": 1,  # MLB
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": end.strftime("%Y-%m-%d"),
            "hydrate": "probablePitcher,venue,team,linescore",
            "gameType": "R,F,D,L,W",  # regular + postseason
        },
    )
    resp.raise_for_status()
    data = resp.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            game = parse_game(g)
            if game:
                games.append(game)
    return games


def parse_game(g: dict) -> dict | None:
    """Parse a single game from the MLB API response."""
    status = g.get("status", {}).get("detailedState", "")
    if status not in ("Final", "Completed Early", "Game Over"):
        return None

    home = g.get("teams", {}).get("home", {})
    away = g.get("teams", {}).get("away", {})
    venue = g.get("venue", {})
    linescore = g.get("linescore", {})

    home_team = home.get("team", {})
    away_team = away.get("team", {})

    home_pitcher = home.get("probablePitcher", {})
    away_pitcher = away.get("probablePitcher", {})

    home_score = home.get("score", 0)
    away_score = away.get("score", 0)

    return {
        "game_pk": g.get("gamePk"),
        "game_date": g.get("officialDate", g.get("gameDate", "")[:10]),
        "game_type": g.get("gameType", ""),
        "season": g.get("seasonDisplay", ""),
        "status": status,

        # teams
        "home_team_id": home_team.get("id"),
        "home_team_name": home_team.get("name", ""),
        "home_team_abbr": home_team.get("abbreviation", ""),
        "away_team_id": away_team.get("id"),
        "away_team_name": away_team.get("name", ""),
        "away_team_abbr": away_team.get("abbreviation", ""),

        # scores
        "home_score": home_score,
        "away_score": away_score,
        "home_win": int(home_score > away_score),

        # starting pitchers
        "home_sp_id": home_pitcher.get("id"),
        "home_sp_name": home_pitcher.get("fullName", ""),
        "away_sp_id": away_pitcher.get("id"),
        "away_sp_name": away_pitcher.get("fullName", ""),

        # venue
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name", ""),

        # innings
        "innings": linescore.get("currentInning", 9),

        # day/night
        "day_night": g.get("dayNight", ""),
    }


def fetch_boxscore_lineups(client: httpx.Client, game_pk: int) -> dict:
    """Fetch starting lineups from boxscore endpoint."""
    try:
        resp = client.get(f"{MLB_API}/game/{game_pk}/boxscore")
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {"home_lineup": [], "away_lineup": []}

    lineups = {"home_lineup": [], "away_lineup": []}

    for side in ("home", "away"):
        team_data = data.get("teams", {}).get(side, {})
        batting_order = team_data.get("battingOrder", [])
        players = team_data.get("players", {})

        lineup = []
        for player_id in batting_order:
            key = f"ID{player_id}"
            p = players.get(key, {})
            person = p.get("person", {})
            position = p.get("position", {})
            lineup.append({
                "player_id": player_id,
                "name": person.get("fullName", ""),
                "position": position.get("abbreviation", ""),
                "bat_side": p.get("batSide", {}).get("code", ""),
            })

        lineups[f"{side}_lineup"] = lineup

    return lineups


def scrape_season_games(year: int, include_lineups: bool = False):
    """Scrape all games for a season."""
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GAMES_DIR / f"games_{year}.parquet"
    lineups_path = GAMES_DIR / f"lineups_{year}.json"

    if year not in SEASON_DATES:
        print(f"No dates for {year}")
        return

    start, end = SEASON_DATES[year]
    today = date.today()
    if end > today:
        end = today - timedelta(days=1)
    if start > today:
        print(f"  {year} hasn't started yet")
        return

    print(f"\nSeason {year}: {start} → {end}")

    with httpx.Client(timeout=30.0) as client:
        # Fetch in monthly chunks
        all_games = []
        dt = start
        while dt <= end:
            chunk_end = min(dt + timedelta(days=30), end)
            games = fetch_schedule(client, dt, chunk_end)
            all_games.extend(games)
            print(f"  {dt} → {chunk_end}: {len(games)} games")
            dt = chunk_end + timedelta(days=1)
            time.sleep(0.3)

        if not all_games:
            print(f"  No games found for {year}")
            return

        df = pd.DataFrame(all_games)
        df = df.drop_duplicates(subset=["game_pk"])
        df.to_parquet(output_path, index=False)
        print(f"  Saved {len(df)} games to {output_path}")

        # Optionally fetch lineups (slower — one API call per game)
        if include_lineups:
            print(f"  Fetching lineups for {len(df)} games...")
            lineups = {}
            for i, row in df.iterrows():
                gpk = row["game_pk"]
                lu = fetch_boxscore_lineups(client, gpk)
                lineups[str(gpk)] = lu
                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(df)} games")
                time.sleep(0.15)

            with open(lineups_path, "w") as f:
                json.dump(lineups, f)
            print(f"  Saved lineups to {lineups_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape MLB game data")
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=list(SEASON_DATES.keys()))
    parser.add_argument("--lineups", action="store_true",
                        help="Also fetch starting lineups (slow)")
    args = parser.parse_args()

    print("MLB Game Scraper")
    print(f"Seasons: {args.seasons}")

    for year in sorted(args.seasons):
        scrape_season_games(year, include_lineups=args.lineups)

    print("\nDone.")


if __name__ == "__main__":
    main()
