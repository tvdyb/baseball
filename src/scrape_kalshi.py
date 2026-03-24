#!/usr/bin/env python3
"""
Scrape Kalshi MLB game market data for backtesting.

Fetches:
  1. All settled KXMLBGAME markets (team, result, volume)
  2. Pre-game closing prices via candlestick endpoint

Usage:
    python src/scrape_kalshi.py --year 2025
"""

import argparse
import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KALSHI_DIR = DATA_DIR / "kalshi"
API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi team abbreviations → our abbreviations
TEAM_MAP = {
    "LAD": "LAD", "TOR": "TOR", "NYY": "NYY", "NYM": "NYM",
    "BOS": "BOS", "HOU": "HOU", "ATL": "ATL", "SD": "SD",
    "SF": "SF", "SEA": "SEA", "MIN": "MIN", "CLE": "CLE",
    "BAL": "BAL", "TB": "TB", "TEX": "TEX", "AZ": "ARI",
    "MIL": "MIL", "CHC": "CHC", "CWS": "CWS", "COL": "COL",
    "PIT": "PIT", "CIN": "CIN", "STL": "STL", "KC": "KC",
    "PHI": "PHI", "MIA": "MIA", "DET": "DET", "WSH": "WSH",
    "LAA": "LAA", "ATH": "OAK",
}

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_event_ticker(event_ticker: str) -> dict | None:
    """Parse KXMLBGAME-25JUL31ATLCIN into date + teams."""
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None

    code = parts[1]  # e.g. "25JUL31ATLCIN"

    # Extract year prefix (2 digits), month (3 letters), day (1-2 digits), then teams
    m = re.match(r"(\d{2})([A-Z]{3})(\d{1,2})(.+)", code)
    if not m:
        return None

    year = 2000 + int(m.group(1))
    month = MONTH_MAP.get(m.group(2))
    day = int(m.group(3))
    team_str = m.group(4)

    if not month:
        return None

    # Parse teams - try known abbreviations (2 or 3 chars each)
    away_team = None
    home_team = None
    for away_len in [3, 2]:
        away_try = team_str[:away_len]
        home_try = team_str[away_len:]
        if away_try in TEAM_MAP and home_try in TEAM_MAP:
            away_team = TEAM_MAP[away_try]
            home_team = TEAM_MAP[home_try]
            break

    if not away_team:
        return None

    try:
        game_date = f"{year}-{month:02d}-{day:02d}"
    except ValueError:
        return None

    return {
        "game_date": game_date,
        "away_team": away_team,
        "home_team": home_team,
    }


def fetch_all_markets(year: int) -> list[dict]:
    """Fetch all settled KXMLBGAME markets, paginating through results."""
    all_markets = []
    cursor = ""

    with httpx.Client(timeout=30.0) as client:
        while True:
            params = {
                "series_ticker": "KXMLBGAME",
                "status": "settled",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            resp = client.get(f"{API_BASE}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()

            markets = data.get("markets", [])
            if not markets:
                break

            all_markets.extend(markets)
            cursor = data.get("cursor", "")

            # Check if we've gone past our target year
            last_ticker = markets[-1]["event_ticker"]
            parsed = parse_event_ticker(last_ticker)
            if parsed and int(parsed["game_date"][:4]) < year:
                break

            if not cursor:
                break

            time.sleep(0.1)

    print(f"  Fetched {len(all_markets)} total markets")
    return all_markets


def group_markets_by_event(markets: list[dict], year: int) -> dict:
    """Group markets by event (game) and filter to target year."""
    events = {}

    for m in markets:
        event_ticker = m["event_ticker"]
        parsed = parse_event_ticker(event_ticker)
        if not parsed:
            continue
        if int(parsed["game_date"][:4]) != year:
            continue

        if event_ticker not in events:
            events[event_ticker] = {
                **parsed,
                "event_ticker": event_ticker,
                "markets": {},
            }

        # Figure out which team this market is for
        ticker_suffix = m["ticker"].split("-")[-1]
        team = TEAM_MAP.get(ticker_suffix, ticker_suffix)

        events[event_ticker]["markets"][team] = {
            "ticker": m["ticker"],
            "result": m.get("result"),
            "volume": float(m.get("volume_fp", 0)),
            "open_interest": float(m.get("open_interest_fp", 0)),
            "open_time": m.get("open_time"),
            "close_time": m.get("close_time"),
            "expected_expiration_time": m.get("expected_expiration_time"),
        }

    return events


async def fetch_pregame_price(
    client: httpx.AsyncClient,
    ticker: str,
    open_time: str,
    close_time: str,
    semaphore: asyncio.Semaphore,
) -> float | None:
    """Fetch the pre-game closing price using candlesticks.

    Strategy: MLB games last ~3 hours. The pre-game price is the last
    candle at 3-4 hours before close_time, before in-game action moves
    prices to extremes and volume spikes.
    """
    async with semaphore:
        try:
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            open_dt = datetime.fromisoformat(open_time.replace("Z", "+00:00"))

            start_ts = int(open_dt.timestamp())
            end_ts = int(close_dt.timestamp())

            resp = await client.get(
                f"{API_BASE}/series/KXMLBGAME/markets/{ticker}/candlesticks",
                params={
                    "period_interval": 60,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                },
            )
            resp.raise_for_status()
            candles = resp.json().get("candlesticks", [])

            if not candles:
                return None

            close_ts = close_dt.timestamp()

            # Strategy 1: Find candle closest to 3.5 hours before close
            # (approximate game start time)
            target_ts = close_ts - 3.5 * 3600
            best_candle = None
            best_dist = float("inf")

            for c in candles:
                price_data = c.get("price", {})
                close_p = price_data.get("close_dollars")
                if close_p is None:
                    continue
                close_p = float(close_p)

                # Must be a reasonable probability (not post-settlement)
                if close_p < 0.03 or close_p > 0.97:
                    continue

                candle_ts = c["end_period_ts"]
                dist = abs(candle_ts - target_ts)

                # Only consider candles BEFORE the target (pre-game)
                # Allow some slack: up to 1h after target
                if candle_ts <= target_ts + 3600 and dist < best_dist:
                    best_dist = dist
                    best_candle = close_p

            if best_candle is not None:
                return best_candle

            # Strategy 2: Use the last candle before a volume spike
            # (volume > 5x median signals in-game trading)
            volumes = []
            for c in candles:
                vol = float(c.get("volume_fp", "0"))
                if vol > 0:
                    volumes.append(vol)
            if volumes:
                median_vol = sorted(volumes)[len(volumes) // 2]
                threshold = max(median_vol * 5, 1000)

                for c in reversed(candles):
                    vol = float(c.get("volume_fp", "0"))
                    price_data = c.get("price", {})
                    close_p = price_data.get("close_dollars")
                    if close_p is None:
                        continue
                    close_p = float(close_p)
                    if 0.03 < close_p < 0.97 and vol < threshold:
                        return close_p

            # Strategy 3: First candle with any data
            for c in candles:
                price_data = c.get("price", {})
                close_p = price_data.get("close_dollars")
                if close_p is not None:
                    close_p = float(close_p)
                    if 0.03 < close_p < 0.97:
                        return close_p

            return None

        except Exception:
            return None


async def fetch_all_pregame_prices(events: dict, max_concurrent: int = 15) -> dict:
    """Fetch pre-game prices for all events."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = {}

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Create tasks for home team market of each event
        fetch_tasks = []
        event_keys = []

        for event_ticker, event in events.items():
            home_team = event["home_team"]
            home_market = event["markets"].get(home_team)

            if not home_market:
                # Try the other team
                for team, mkt in event["markets"].items():
                    home_market = mkt
                    break

            if not home_market or not home_market.get("open_time") or not home_market.get("close_time"):
                continue

            fetch_tasks.append(
                fetch_pregame_price(
                    client,
                    home_market["ticker"],
                    home_market["open_time"],
                    home_market["close_time"],
                    semaphore,
                )
            )
            event_keys.append(event_ticker)

        # Process in batches
        results = {}
        batch_size = 100
        for i in range(0, len(fetch_tasks), batch_size):
            batch = fetch_tasks[i:i + batch_size]
            batch_keys = event_keys[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)

            for key, price in zip(batch_keys, batch_results):
                if price is not None:
                    results[key] = price

            done = min(i + batch_size, len(fetch_tasks))
            print(f"    {done}/{len(fetch_tasks)} prices fetched ({len(results)} valid)")

    return results


def build_dataset(events: dict, prices: dict) -> pd.DataFrame:
    """Build final dataset with market prices and outcomes."""
    rows = []

    for event_ticker, event in events.items():
        home_team = event["home_team"]
        away_team = event["away_team"]

        # Determine winner
        home_market = event["markets"].get(home_team)
        away_market = event["markets"].get(away_team)

        if not home_market or not away_market:
            continue

        home_won = home_market.get("result") == "yes"

        # Get pre-game price
        home_price = prices.get(event_ticker)
        if home_price is None:
            continue

        # If we got the away team's ticker price, flip it
        if home_market["ticker"] != event_ticker.split("|")[0] if "|" in event_ticker else True:
            pass  # home_price is already for home team

        total_volume = home_market["volume"] + away_market["volume"]

        rows.append({
            "game_date": event["game_date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_win": int(home_won),
            "kalshi_home_prob": home_price,
            "kalshi_away_prob": 1 - home_price,
            "volume": total_volume,
            "event_ticker": event_ticker,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("game_date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Scrape Kalshi MLB market data")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--concurrency", type=int, default=15)
    args = parser.parse_args()

    KALSHI_DIR.mkdir(parents=True, exist_ok=True)
    output_path = KALSHI_DIR / f"kalshi_mlb_{args.year}.parquet"

    print(f"\nFetching Kalshi MLB markets for {args.year}...")
    markets = fetch_all_markets(args.year)

    print(f"\nGrouping by event...")
    events = group_markets_by_event(markets, args.year)
    print(f"  {len(events)} games found")

    print(f"\nFetching pre-game prices...")
    prices = asyncio.run(fetch_all_pregame_prices(events, max_concurrent=args.concurrency))
    print(f"  Got prices for {len(prices)}/{len(events)} games")

    print(f"\nBuilding dataset...")
    df = build_dataset(events, prices)
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df)} games to {output_path}")

    # Summary
    print(f"\n  Date range: {df['game_date'].min()} → {df['game_date'].max()}")
    print(f"  Avg Kalshi home prob: {df['kalshi_home_prob'].mean():.3f}")
    print(f"  Kalshi home win rate: {df['home_win'].mean():.3f}")
    print(f"  Avg volume/game: ${df['volume'].mean():,.0f}")


if __name__ == "__main__":
    main()
