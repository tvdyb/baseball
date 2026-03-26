#!/usr/bin/env python3
"""
Scrape Kalshi MLB game market data for backtesting.

Fetches:
  1. All settled KXMLBGAME markets (team, result, volume)
  2. Pre-game closing prices using actual first-pitch timestamps
     from games parquet + candlestick endpoints (historical + live)

Usage:
    python src/scrape_kalshi.py --year 2025
"""

import argparse
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd


def log(msg: str):
    print(msg)
    sys.stdout.flush()

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

    with httpx.Client(timeout=60.0) as client:
        while True:
            params = {
                "series_ticker": "KXMLBGAME",
                "status": "settled",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            for attempt in range(3):
                try:
                    resp = client.get(f"{API_BASE}/markets", params=params)
                    resp.raise_for_status()
                    break
                except (httpx.TimeoutException, httpx.HTTPStatusError):
                    if attempt < 2:
                        time.sleep((attempt + 1) * 3)
                    else:
                        raise

            data = resp.json()

            markets = data.get("markets", [])
            if not markets:
                break

            all_markets.extend(markets)
            cursor = data.get("cursor", "")

            if not cursor:
                break

            time.sleep(0.3)

    log(f"  Fetched {len(all_markets)} total markets")
    return all_markets


def fetch_historical_cutoff() -> float:
    """Fetch the archival cutoff timestamp from Kalshi.

    Markets settled before this timestamp are archived and must use
    the /historical/ endpoint for candlestick data.
    """
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{API_BASE}/historical/cutoff")
        resp.raise_for_status()
        data = resp.json()

    ts_str = data["market_settled_ts"]
    cutoff_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    log(f"  Historical cutoff: {ts_str}")
    return cutoff_dt.timestamp()


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
        }

    return events


def load_game_start_times(year: int) -> dict:
    """Load actual first-pitch timestamps from games parquet.

    Returns dict mapping (game_date, home_team, away_team) → unix timestamp.
    """
    games_path = GAMES_DIR / f"games_{year}.parquet"
    if not games_path.exists():
        log(f"  WARNING: {games_path} not found — will use fallback timing")
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

    log(f"  Loaded {len(start_times)} game start times from {games_path}")
    return start_times


def _extract_candle_close(price_data: dict) -> float | None:
    """Extract close price from a candlestick, handling both API schemas.

    Live endpoint uses: price.close_dollars
    Historical endpoint uses: price.close
    """
    # Try live schema first, then historical
    close_p = price_data.get("close_dollars")
    if close_p is None:
        close_p = price_data.get("close")
    if close_p is None:
        return None
    return float(close_p)


def _find_pregame_price(candles: list[dict], game_start_ts: float) -> float | None:
    """Extract the last pre-game closing price from a list of candles.

    Finds the last candle ending before game_start_ts + 5min buffer
    with a reasonable price (0.08-0.92).
    """
    valid = []
    for c in candles:
        price_data = c.get("price", {})
        close_p = _extract_candle_close(price_data)
        if close_p is None:
            continue
        valid.append((c["end_period_ts"], close_p))

    if not valid:
        return None

    cutoff_ts = game_start_ts + 300  # 5-minute buffer
    pre_game = [
        (ts, p) for ts, p in valid
        if ts <= cutoff_ts and 0.08 <= p <= 0.92
    ]
    if pre_game:
        return pre_game[-1][1]  # last one (closest to first pitch)

    return None


def _fetch_candles_with_retry(
    client: httpx.Client, url: str, params: dict, max_retries: int = 3,
) -> list[dict] | None:
    """Fetch candlesticks with retry and exponential backoff.

    Returns list of candles on success, None on permanent failure.
    """
    for attempt in range(max_retries):
        try:
            resp = client.get(url, params=params)
            if resp.status_code == 404:
                return None  # Market not found on this endpoint
            resp.raise_for_status()
            return resp.json().get("candlesticks", [])
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt < max_retries - 1:
                backoff = (attempt + 1) * 2
                time.sleep(backoff)
            else:
                return None
        except Exception:
            return None
    return None


def fetch_pregame_prices_individual(
    client: httpx.Client,
    markets_to_fetch: list[tuple[str, dict, float]],
) -> tuple[dict, dict]:
    """Fetch pre-game prices for markets via individual candlestick requests.

    Tries historical endpoint first, falls back to live endpoint.

    Returns:
        (results dict, stats dict)
    """
    results = {}
    stats = {
        "found_hourly": 0, "found_daily": 0,
        "no_candles": 0, "no_pregame": 0, "error": 0,
        "via_historical": 0, "via_live": 0,
    }

    for i, (event_ticker, market, game_start_ts) in enumerate(markets_to_fetch):
        ticker = market["ticker"]
        try:
            open_dt = datetime.fromisoformat(market["open_time"].replace("Z", "+00:00"))
            close_dt = datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
            start_ts = int(open_dt.timestamp())
            end_ts = int(close_dt.timestamp())
            params = {"period_interval": 60, "start_ts": start_ts, "end_ts": end_ts}

            # Try historical endpoint first
            candles = _fetch_candles_with_retry(
                client, f"{API_BASE}/historical/markets/{ticker}/candlesticks", params,
            )
            source = "historical"

            # Fall back to live endpoint if historical returns 404 or None
            if candles is None:
                candles = _fetch_candles_with_retry(
                    client, f"{API_BASE}/series/KXMLBGAME/markets/{ticker}/candlesticks", params,
                )
                source = "live"

            if candles is None:
                stats["error"] += 1
                time.sleep(0.1)
                continue

            price = _find_pregame_price(candles, game_start_ts)
            if price is not None:
                results[event_ticker] = price
                stats["found_hourly"] += 1
                stats[f"via_{source}"] += 1
            elif candles:
                # Had candles but none were pre-game or in range
                # Try daily candles as fallback for illiquid markets
                daily_params = {"period_interval": 1440, "start_ts": start_ts, "end_ts": end_ts}
                if source == "historical":
                    daily_url = f"{API_BASE}/historical/markets/{ticker}/candlesticks"
                else:
                    daily_url = f"{API_BASE}/series/KXMLBGAME/markets/{ticker}/candlesticks"
                daily = _fetch_candles_with_retry(client, daily_url, daily_params)
                if daily:
                    price = _find_pregame_price(daily, game_start_ts)
                    if price is not None:
                        results[event_ticker] = price
                        stats["found_daily"] += 1
                        stats[f"via_{source}"] += 1
                    else:
                        stats["no_pregame"] += 1
                else:
                    stats["no_pregame"] += 1
                time.sleep(0.1)
            else:
                stats["no_candles"] += 1

        except Exception:
            stats["error"] += 1

        if (i + 1) % 100 == 0 or i + 1 == len(markets_to_fetch):
            found = stats["found_hourly"] + stats["found_daily"]
            log(f"    progress: {i + 1}/{len(markets_to_fetch)} ({found} found)")

        time.sleep(0.1)

    return results, stats


def fetch_pregame_prices_live_batch(
    client: httpx.Client,
    markets_to_fetch: list[tuple[str, dict, float]],
) -> tuple[dict, dict]:
    """Fetch pre-game prices for live markets via batch /markets/candlesticks.

    Batches up to 60 tickers per request.

    Returns:
        (results dict, stats dict)
    """
    results = {}
    stats = {"found_hourly": 0, "found_daily": 0, "no_candles": 0, "no_pregame": 0, "error": 0}

    if not markets_to_fetch:
        return results, stats

    # Build lookup: ticker -> (event_ticker, game_start_ts, open_ts, end_ts)
    ticker_info = {}
    for event_ticker, market, game_start_ts in markets_to_fetch:
        ticker = market["ticker"]
        try:
            open_dt = datetime.fromisoformat(market["open_time"].replace("Z", "+00:00"))
            close_dt = datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
            ticker_info[ticker] = (event_ticker, game_start_ts, int(open_dt.timestamp()), int(close_dt.timestamp()))
        except (ValueError, TypeError):
            stats["error"] += 1

    tickers = list(ticker_info.keys())
    batch_size = 60

    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start:batch_start + batch_size]

        # Find the overall time range for this batch
        min_ts = min(ticker_info[t][2] for t in batch)
        max_ts = max(ticker_info[t][3] for t in batch)

        try:
            resp = client.get(
                f"{API_BASE}/markets/candlesticks",
                params={
                    "market_tickers": ",".join(batch),
                    "series_ticker": "KXMLBGAME",
                    "period_interval": 60,
                    "start_ts": min_ts,
                    "end_ts": max_ts,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Response: {"markets": [{"market_ticker": "...", "candlesticks": [...]}]}
            by_ticker = {}
            for m in data.get("markets", []):
                by_ticker[m["market_ticker"]] = m.get("candlesticks", [])

            for ticker in batch:
                event_ticker, game_start_ts, _, _ = ticker_info[ticker]
                candles = by_ticker.get(ticker, [])
                if not candles:
                    stats["no_candles"] += 1
                    continue

                price = _find_pregame_price(candles, game_start_ts)
                if price is not None:
                    results[event_ticker] = price
                    stats["found_hourly"] += 1
                else:
                    stats["no_pregame"] += 1

        except Exception:
            stats["error"] += len(batch)

        found = stats["found_hourly"] + stats["found_daily"]
        processed = min(batch_start + batch_size, len(tickers))
        log(f"    live batch: {processed}/{len(tickers)} ({found} found)")
        time.sleep(0.2)

    return results, stats


def fetch_all_pregame_prices(
    events: dict, game_start_times: dict, cutoff_ts: float,
) -> dict:
    """Fetch pre-game prices for all events.

    Routes requests to historical or live endpoint based on the archival cutoff.
    Always uses candlesticks — never previous_price_dollars.
    """
    all_markets = []
    live_markets = []
    no_start_time = 0
    no_home_market = 0

    for event_ticker, event in events.items():
        home_team = event["home_team"]
        away_team = event["away_team"]
        game_date = event["game_date"]
        home_market = event["markets"].get(home_team)

        if not home_market:
            no_home_market += 1
            continue

        if not home_market.get("open_time") or not home_market.get("close_time"):
            no_home_market += 1
            continue

        # Look up actual game start time
        gst_key = (game_date, home_team, away_team)
        game_start_ts = game_start_times.get(gst_key)

        if game_start_ts is None:
            no_start_time += 1
            continue

        # Route to batch (live) or individual (historical/fallback)
        try:
            close_dt = datetime.fromisoformat(
                home_market["close_time"].replace("Z", "+00:00")
            )
            if close_dt.timestamp() >= cutoff_ts:
                live_markets.append((event_ticker, home_market, game_start_ts))
            else:
                all_markets.append((event_ticker, home_market, game_start_ts))
        except (ValueError, TypeError):
            no_home_market += 1

    log(f"    {len(all_markets)} archived markets (individual requests)")
    log(f"    {len(live_markets)} live markets (batch eligible)")
    log(f"    {no_start_time} skipped (no game start time)")
    log(f"    {no_home_market} skipped (no home market/times)")

    results = {}

    with httpx.Client(timeout=30.0) as client:
        # Fetch archived markets individually (with retry + live fallback)
        if all_markets:
            log(f"\n  Fetching archived markets ({len(all_markets)})...")
            ind_results, ind_stats = fetch_pregame_prices_individual(
                client, all_markets,
            )
            results.update(ind_results)
            log(f"    → {ind_stats['found_hourly']} hourly, "
                  f"{ind_stats['found_daily']} daily, "
                  f"{ind_stats['no_candles']} no candles, "
                  f"{ind_stats['no_pregame']} no pre-game price, "
                  f"{ind_stats['error']} errors")
            log(f"    → {ind_stats['via_historical']} via historical, "
                  f"{ind_stats['via_live']} via live fallback")

        # Fetch live markets — try batch first, fall back to individual
        if live_markets:
            log(f"\n  Fetching live markets ({len(live_markets)})...")
            live_results, live_stats = fetch_pregame_prices_live_batch(
                client, live_markets,
            )
            results.update(live_results)

            # For any that failed in batch, try individual requests
            failed = [
                (et, m, gs) for et, m, gs in live_markets
                if et not in results
            ]
            if failed:
                log(f"    Batch missed {len(failed)}, trying individual...")
                ind_results, ind_stats = fetch_pregame_prices_individual(
                    client, failed,
                )
                results.update(ind_results)
                live_stats["found_hourly"] += ind_stats["found_hourly"]
                live_stats["found_daily"] += ind_stats["found_daily"]
                live_stats["error"] = ind_stats["error"]

            log(f"    → {live_stats['found_hourly']} hourly, "
                  f"{live_stats['found_daily']} daily, "
                  f"{live_stats['no_candles']} no candles, "
                  f"{live_stats['no_pregame']} no pre-game price, "
                  f"{live_stats['error']} errors")

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

    log(f"\nFetching Kalshi MLB markets for {args.year}...")
    markets = fetch_all_markets(args.year)

    log(f"\nFetching historical cutoff...")
    cutoff_ts = fetch_historical_cutoff()

    log(f"\nGrouping by event...")
    events = group_markets_by_event(markets, args.year)
    log(f"  {len(events)} games found")

    log(f"\nLoading game start times...")
    game_start_times = load_game_start_times(args.year)

    log(f"\nFetching pre-game prices...")
    prices = fetch_all_pregame_prices(events, game_start_times, cutoff_ts)
    log(f"\n  Got prices for {len(prices)}/{len(events)} games")

    log(f"\nBuilding dataset...")
    df = build_dataset(events, prices)
    df.to_parquet(output_path, index=False)
    log(f"  Saved {len(df)} games to {output_path}")

    if len(df) > 0:
        log(f"\n  Date range: {df['game_date'].min()} -> {df['game_date'].max()}")
        log(f"  Avg Kalshi home prob: {df['kalshi_home_prob'].mean():.3f}")
        log(f"  Home win rate: {df['home_win'].mean():.3f}")
        log(f"  Avg volume/game: ${df['volume'].mean():,.0f}")

        # Price distribution
        probs = df["kalshi_home_prob"]
        log(f"\n  Price distribution:")
        for lo, hi in [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
            n = ((probs >= lo) & (probs < hi)).sum()
            log(f"    [{lo:.1f}, {hi:.1f}): {n} ({n/len(probs):.1%})")

        df["month"] = df["game_date"].str[:7]
        log(f"\n  By month:")
        for month in sorted(df["month"].unique()):
            n = len(df[df["month"] == month])
            log(f"    {month}: {n}")


if __name__ == "__main__":
    main()
