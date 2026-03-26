#!/usr/bin/env python3
"""
Scrape Kalshi MLB game market data for backtesting.

Fetches:
  1. All settled KXMLBGAME markets (team, result, volume)
  2. Pre-game closing prices using actual first-pitch timestamps
     from games parquet + candlestick endpoint

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
GAMES_DIR = DATA_DIR / "games"
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

KNOWN_TEAMS = set(TEAM_MAP.keys())


def parse_event_ticker(event_ticker: str) -> dict | None:
    """Parse KXMLBGAME-25JUL31ATLCIN or KXMLBGAME-26MAR252005NYYSF."""
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None

    code = parts[1]

    m = re.match(r"(\d{2})([A-Z]{3})(.+)", code)
    if not m:
        return None

    year = 2000 + int(m.group(1))
    month = MONTH_MAP.get(m.group(2))
    remainder = m.group(3)

    if not month:
        return None

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

        for time_len in [4, 0]:
            if time_len > 0 and len(rest) >= time_len and rest[:time_len].isdigit():
                team_str = rest[time_len:]
            elif time_len == 0:
                if rest and rest[0].isdigit():
                    continue
                team_str = rest
            else:
                continue

            away_team, home_team = _parse_teams(team_str)
            if away_team and home_team:
                try:
                    game_date = f"{year}-{month:02d}-{day:02d}"
                    datetime(year, month, day)
                    return {
                        "game_date": game_date,
                        "away_team": away_team,
                        "home_team": home_team,
                    }
                except ValueError:
                    continue

    return None


def _parse_teams(team_str: str) -> tuple[str | None, str | None]:
    """Parse 'DETATH' or 'NYYSF' into (away, home) abbreviations."""
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


def load_game_start_times(year: int) -> dict:
    """Load actual first-pitch timestamps from games parquet.

    Returns dict mapping (game_date, home_team, away_team) → unix timestamp.
    """
    games_path = GAMES_DIR / f"games_{year}.parquet"
    if not games_path.exists():
        print(f"  WARNING: {games_path} not found — will use fallback timing")
        return {}

    df = pd.read_parquet(games_path)
    df["game_date"] = df["game_date"].astype(str)

    start_times = {}
    for _, row in df.iterrows():
        gdt = row.get("game_datetime", "")
        if not gdt:
            continue
        try:
            dt = datetime.fromisoformat(gdt.replace("Z", "+00:00"))
            key = (row["game_date"], row["home_team_abbr"], row["away_team_abbr"])
            start_times[key] = dt.timestamp()
        except (ValueError, TypeError):
            continue

    print(f"  Loaded {len(start_times)} game start times from {games_path}")
    return start_times


def fetch_pregame_price_candle(
    client: httpx.Client,
    ticker: str,
    open_time: str,
    close_time: str,
    game_start_ts: float,
) -> float | None:
    """Fetch the pre-game closing price using candlesticks.

    Uses the actual game start timestamp to find the last candle with a
    reasonable price (0.08-0.92) that ends BEFORE first pitch.
    No fallback strategies — if there's no pre-game price, return None.
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

        # Collect candles with valid prices
        valid = []
        for c in candles:
            price_data = c.get("price", {})
            close_p = price_data.get("close_dollars")
            if close_p is None:
                continue
            valid.append((c["end_period_ts"], float(close_p)))

        if not valid:
            return None

        # Find last candle that ends before game start (with 5-min buffer)
        # and has a reasonable price (0.08-0.92)
        cutoff_ts = game_start_ts + 300  # 5-minute buffer
        pre_game = [
            (ts, p) for ts, p in valid
            if ts <= cutoff_ts and 0.08 <= p <= 0.92
        ]
        if pre_game:
            return pre_game[-1][1]  # last one (closest to first pitch)

        return None

    except Exception:
        return None


def fetch_all_pregame_prices(
    events: dict, game_start_times: dict
) -> dict:
    """Fetch pre-game prices for all events.

    Uses previous_price_dollars first (validated against game start time),
    then falls back to candlestick extraction using actual first-pitch times.
    """
    results = {}
    need_candles = []
    no_start_time = 0
    prev_price_used = 0
    prev_price_stale = 0

    for event_ticker, event in events.items():
        home_team = event["home_team"]
        away_team = event["away_team"]
        game_date = event["game_date"]
        home_market = event["markets"].get(home_team)

        if not home_market:
            continue

        # Look up actual game start time
        gst_key = (game_date, home_team, away_team)
        game_start_ts = game_start_times.get(gst_key)

        if game_start_ts is None:
            no_start_time += 1
            # Can't validate previous_price or anchor candles without start time
            # Skip this game
            continue

        # Try previous_price_dollars, but validate it's not stale
        prev_price = home_market.get("previous_price")
        close_time = home_market.get("close_time", "")
        if prev_price is not None and close_time:
            prev_price = float(prev_price)
            if 0.05 <= prev_price <= 0.95:
                # Validate: market close_time should be within ~6h of game start
                try:
                    close_dt = datetime.fromisoformat(
                        close_time.replace("Z", "+00:00")
                    )
                    hours_diff = abs(close_dt.timestamp() - game_start_ts) / 3600
                    if hours_diff <= 6:
                        results[event_ticker] = prev_price
                        prev_price_used += 1
                        continue
                    else:
                        prev_price_stale += 1
                except (ValueError, TypeError):
                    pass

        # Need candlestick fallback
        if home_market.get("open_time") and home_market.get("close_time"):
            need_candles.append((event_ticker, home_market, game_start_ts))

    print(f"    {prev_price_used} from previous_price_dollars")
    print(f"    {prev_price_stale} previous_price rejected (stale)")
    print(f"    {no_start_time} skipped (no game start time)")
    print(f"    {len(need_candles)} need candlestick extraction")

    if not need_candles:
        return results

    # Synchronous candlestick extraction
    candle_found = 0
    with httpx.Client(timeout=15.0) as client:
        for i, (event_ticker, market, game_start_ts) in enumerate(need_candles):
            price = fetch_pregame_price_candle(
                client,
                market["ticker"],
                market["open_time"],
                market["close_time"],
                game_start_ts,
            )
            if price is not None:
                results[event_ticker] = price
                candle_found += 1

            if (i + 1) % 100 == 0 or i + 1 == len(need_candles):
                print(
                    f"    candles: {i + 1}/{len(need_candles)} ({candle_found} found)"
                )

            time.sleep(0.05)

    no_price = len(need_candles) - candle_found
    print(f"    {no_price} games had no pre-game candle price")

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

    print(f"\nLoading game start times...")
    game_start_times = load_game_start_times(args.year)

    print(f"\nFetching pre-game prices...")
    prices = fetch_all_pregame_prices(events, game_start_times)
    print(f"  Got prices for {len(prices)}/{len(events)} games")

    print(f"\nBuilding dataset...")
    df = build_dataset(events, prices)
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df)} games to {output_path}")

    if len(df) > 0:
        print(f"\n  Date range: {df['game_date'].min()} -> {df['game_date'].max()}")
        print(f"  Avg Kalshi home prob: {df['kalshi_home_prob'].mean():.3f}")
        print(f"  Home win rate: {df['home_win'].mean():.3f}")
        print(f"  Avg volume/game: ${df['volume'].mean():,.0f}")

        df["month"] = df["game_date"].str[:7]
        print(f"\n  By month:")
        for month in sorted(df["month"].unique()):
            n = len(df[df["month"] == month])
            print(f"    {month}: {n}")


if __name__ == "__main__":
    main()
