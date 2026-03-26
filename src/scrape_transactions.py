#!/usr/bin/env python3
"""
Scrape MLB roster transactions (trades, signings, waivers) from the MLB Stats API.

Output: data/rosters/transactions_{year}.parquet
  Columns: date, player_id, player_name, from_team, to_team, transaction_type

Used to build trade-deadline features: teams that acquire talent in July
tend to outperform their pre-deadline projections.
"""

import argparse
import time
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROSTER_DIR = DATA_DIR / "rosters"
MLB_API = "https://statsapi.mlb.com/api/v1"

# Team ID -> abbreviation mapping (from MLB API)
TEAM_ABBR = {}


def _fetch_team_abbrs(client: httpx.Client) -> dict[int, str]:
    """Fetch team ID to abbreviation mapping."""
    resp = client.get(f"{MLB_API}/teams", params={"sportId": 1})
    resp.raise_for_status()
    data = resp.json()
    mapping = {}
    for team in data.get("teams", []):
        mapping[team["id"]] = team.get("abbreviation", "")
    return mapping


def scrape_transactions(year: int, client: httpx.Client) -> pd.DataFrame:
    """Scrape all roster-relevant transactions for a season."""
    global TEAM_ABBR
    if not TEAM_ABBR:
        TEAM_ABBR = _fetch_team_abbrs(client)

    # Fetch trades and other acquisitions
    # Focus on trade deadline window but get full season for completeness
    tx_types = "Trade,Waiver Claim,Free Agent Signing"
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    print(f"  Fetching transactions for {year}...")
    resp = client.get(
        f"{MLB_API}/transactions",
        params={
            "startDate": start_date,
            "endDate": end_date,
        },
    )
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for tx in data.get("transactions", []):
        tx_date = tx.get("date", "")
        tx_type = tx.get("typeDesc", "")

        # Only care about player movements between teams
        if tx_type not in ("Trade", "Waiver Claim", "Free Agent Signing",
                           "Purchased From", "Claimed From"):
            continue

        player = tx.get("person", {})
        from_team = tx.get("fromTeam", {})
        to_team = tx.get("toTeam", {})

        if not player.get("id") or not to_team.get("id"):
            continue

        rows.append({
            "date": tx_date[:10] if tx_date else "",
            "player_id": player["id"],
            "player_name": player.get("fullName", ""),
            "from_team_id": from_team.get("id"),
            "from_team": TEAM_ABBR.get(from_team.get("id"), ""),
            "to_team_id": to_team.get("id"),
            "to_team": TEAM_ABBR.get(to_team.get("id"), ""),
            "transaction_type": tx_type,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df[df["date"] != ""]
        df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Scrape MLB transactions")
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=list(range(2017, 2026)))
    args = parser.parse_args()

    ROSTER_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30.0) as client:
        for year in sorted(args.seasons):
            print(f"\nScraping transactions for {year}...")
            df = scrape_transactions(year, client)

            if len(df) > 0:
                out_path = ROSTER_DIR / f"transactions_{year}.parquet"
                df.to_parquet(out_path, index=False)
                print(f"  Saved {len(df)} transactions to {out_path}")

                # Summary
                trades = df[df["transaction_type"] == "Trade"]
                july_trades = trades[trades["date"].str[:7] == f"{year}-07"]
                print(f"  Trades: {len(trades)} total, {len(july_trades)} in July")
                print(f"  Teams involved: {df['to_team'].nunique()}")
            else:
                print(f"  No transactions found for {year}")

            time.sleep(0.5)

    print("\nDone.")


if __name__ == "__main__":
    main()
