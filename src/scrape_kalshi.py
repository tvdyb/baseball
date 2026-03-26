#!/usr/bin/env python3
"""
Scrape Kalshi MLB game market data for backtesting.

Fetches:
  1. All settled KXMLBGAME markets (team, result, volume)
  2. Pre-game closing prices via previous_price_dollars + candlestick fallback

Usage:
    python src/scrape_kalshi.py --year 2025
"""

import argparse
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
    "BAL": "BAL", "TB": "TB", "TEX": "TEX", "AZ": "AZ",
    "MIL": "MIL", "CHC": "CHC", "CWS": "CWS", "COL": "COL",
    "PIT": "PIT", "CIN": "CIN", "STL": "STL", "KC": "KC",
    "PHI": "PHI", "MIA": "MIA", "DET": "DET", "WSH": "WSH",
    "LAA": "LAA", "ATH": "ATH", "ARI": "AZ",
}

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# All known Kalshi team abbreviations (2-3 chars)
KNOWN_TEAMS = set(TEAM_MAP.keys())


def parse_event_ticker(event_ticker: str) -> dict | None:
    """Parse KXMLBGAME-25JUL31ATLCIN or KXMLBGAME-26MAR252005NYYSF."""
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None

    code = parts[1]  # e.g. "25AUG26DETATH" or "26MAR252005NYYSF"

    # Extract: 2-digit year, 3-letter month, then the rest
    m = re.match(r"(\d{2})([A-Z]{3})(.+)", code)
    if not m:
        return None

    year = 2000 + int(m.group(1))
    month = MONTH_MAP.get(m.group(2))
    remainder = m.group(3)  # e.g. "26DETATH" or "252005NYYSF"

    if not month:
        return None

    # Try to parse day + optional time + teams
    # Day is 1-2 digits, then optional 4-digit time (HHMM), then team codes
    best = None
    for day_len in [2, 1]:
        if len(remainder) < day_len:
            continue
        day_str = remainder[:day_len]
        if not day_str.isdigit():
            continue
        day = int(day_str)
        if day < 1 or day > 31:
            continue

        rest = remainder[day_len:]

        # Try with 4-digit time prefix
        for time_len in [4, 0]:
            if time_len > 0 and len(rest) >= time_len and rest[:time_len].isdigit():
                team_str = rest[time_len:]
            elif time_len == 0:
                # Skip if rest starts with digits (it's probably a time code)
                if rest and rest[0].isdigit():
                    continue
                team_str = rest
            else:
                continue

            # Parse teams from team_str
            away_team, home_team = _parse_teams(team_str)
            if away_team and home_team:
                try:
                    game_date = f"{year}-{month:02d}-{day:02d}"
                    # Validate date
                    datetime(year, month, day)
                    best = {
                        "game_date": game_date,
                        "away_team": away_team,
                        "home_team": home_team,
                    }
                    return best
                except ValueError:
                    continue

    return best


def _parse_teams(team_str: str) -> tuple[str | None, str | None]:
    """Parse 'DETATH' or 'NYYSF' into (away, home) abbreviations."""
    # Try all splits: away_len in [2, 3], home is the rest
    for away_len in [3, 2]:
        if len(team_str) < away_len + 2:
            continue
        away_try = team_str[:away_len]
        home_try = team_str[away_len:]
        if away_try in KNOWN_TEAMS and home_try in KNOWN_TEAMS:
            return TEAM_MAP[away_try], TEAM_MAP[home_try]
    return None, None


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
            "volume": float(m.get("volume_fp") or 0),
            "open_interest": float(m.get("open_interest_fp") or 0),
            "open_time": m.get("open_time"),
            "close_time": m.get("close_time"),
            "previous_price": m.get("previous_price_dollars"),
            "last_price": m.get("last_price_dollars"),
        }

    return events


def fetch_pregame_price_candle(
    client: httpx.Client,
    ticker: str,
    open_time: str,
    close_time: str,
) -> float | None:
    """Fetch the pre-game closing price using candlesticks.

    Strategies (in order):
    1. Last candle with reasonable price before game-time region
    2. Walk backwards to find last stable price
    3. Any candle with reasonable price
    """
    try:
        close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        open_dt = datetime.fromisoformat(open_time.replace("Z", "+00:00"))

        start_ts = int(open_dt.timestamp())
        end_ts = int(close_dt.timestamp())

        resp = client.get(
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

        # Collect candles with valid prices
        valid = []
        for c in candles:
            price_data = c.get("price", {})
            close_p = price_data.get("close_dollars")
            if close_p is None:
                continue
            close_p = float(close_p)
            vol = float(c.get("volume_fp") or 0)
            valid.append((c["end_period_ts"], close_p, vol))

        if not valid:
            return None

        # Strategy 1: Find candles in the pre-game window
        # Game starts ~3-4h before market close
        game_start_ts = close_ts - 3.5 * 3600

        # Get last reasonable price before game start
        pre_game = [
            (ts, p, v) for ts, p, v in valid
            if ts <= game_start_ts + 1800 and 0.08 <= p <= 0.92
        ]
        if pre_game:
            return pre_game[-1][1]  # last one (closest to game start)

        # Strategy 2: Walk backwards from end, find last "stable" price
        # (before in-game moves push to extremes)
        for ts, p, v in reversed(valid):
            if 0.10 <= p <= 0.90:
                return p

        # Strategy 3: Any reasonable price
        for ts, p, v in valid:
            if 0.05 <= p <= 0.95:
                return p

        return None

    except Exception:
        return None


def fetch_all_pregame_prices(events: dict) -> dict:
    """Fetch pre-game prices for all events.

    Uses previous_price_dollars first, falls back to candlestick extraction.
    """
    # First pass: use previous_price_dollars where available
    results = {}
    need_candles = []

    for event_ticker, event in events.items():
        home_team = event["home_team"]
        home_market = event["markets"].get(home_team)

        if not home_market:
            continue

        prev_price = home_market.get("previous_price")
        if prev_price is not None:
            prev_price = float(prev_price)
            if 0.05 <= prev_price <= 0.95:
                results[event_ticker] = prev_price
                continue

        # Need candlestick fallback
        if home_market.get("open_time") and home_market.get("close_time"):
            need_candles.append((event_ticker, home_market))

    print(f"    {len(results)} from previous_price_dollars")
    print(f"    {len(need_candles)} need candlestick extraction")

    if not need_candles:
        return results

    # Second pass: synchronous candlestick extraction
    candle_found = 0
    with httpx.Client(timeout=15.0) as client:
        for i, (event_ticker, market) in enumerate(need_candles):
            price = fetch_pregame_price_candle(
                client,
                market["ticker"],
                market["open_time"],
                market["close_time"],
            )
            if price is not None:
                results[event_ticker] = price
                candle_found += 1

            if (i + 1) % 100 == 0 or i + 1 == len(need_candles):
                print(f"    candles: {i + 1}/{len(need_candles)} ({candle_found} found)")

            time.sleep(0.05)  # rate limit

    return results


def build_dataset(events: dict, prices: dict) -> pd.DataFrame:
    """Build final dataset with market prices and outcomes."""
    rows = []

    for event_ticker, event in events.items():
        home_team = event["home_team"]
        away_team = event["away_team"]

        home_market = event["markets"].get(home_team)
        away_market = event["markets"].get(away_team)

        if not home_market or not away_market:
            continue

        home_won = home_market.get("result") == "yes"

        home_price = prices.get(event_ticker)
        if home_price is None:
            continue

        total_volume = home_market["volume"] + away_market["volume"]

        rows.append({
            "game_date": event["game_date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_win": int(home_won),
            "kalshi_home_prob": round(home_price, 4),
            "kalshi_away_prob": round(1 - home_price, 4),
            "volume": total_volume,
            "event_ticker": event_ticker,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("game_date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Scrape Kalshi MLB market data")
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    KALSHI_DIR.mkdir(parents=True, exist_ok=True)
    output_path = KALSHI_DIR / f"kalshi_mlb_{args.year}.parquet"

    print(f"\nFetching Kalshi MLB markets for {args.year}...")
    markets = fetch_all_markets(args.year)

    print(f"\nGrouping by event...")
    events = group_markets_by_event(markets, args.year)
    print(f"  {len(events)} games found")

    print(f"\nFetching pre-game prices...")
    prices = fetch_all_pregame_prices(events)
    print(f"  Got prices for {len(prices)}/{len(events)} games")

    print(f"\nBuilding dataset...")
    df = build_dataset(events, prices)
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df)} games to {output_path}")

    if len(df) > 0:
        print(f"\n  Date range: {df['game_date'].min()} → {df['game_date'].max()}")
        print(f"  Avg Kalshi home prob: {df['kalshi_home_prob'].mean():.3f}")
        print(f"  Home win rate: {df['home_win'].mean():.3f}")
        print(f"  Avg volume/game: ${df['volume'].mean():,.0f}")

        # Monthly breakdown
        df["month"] = df["game_date"].str[:7]
        print(f"\n  By month:")
        for month in sorted(df["month"].unique()):
            n = len(df[df["month"] == month])
            print(f"    {month}: {n}")


if __name__ == "__main__":
    main()
