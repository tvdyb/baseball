#!/usr/bin/env python3
"""
Scrape Kalshi MLB game market data for backtesting.

Fetches:
  1. All settled KXMLBGAME markets (team, result, volume)
     from both the regular and historical market listing endpoints
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

# Kalshi team abbreviations → MLB Stats API abbreviations (canonical form).
# For 2025+: Oakland Athletics = "ATH" (not "OAK"), Arizona = "AZ" (not "ARI").
# Both the MLB Stats API and our feature pipeline use ATH and AZ.
# Kalshi tickers use "ARI" for Arizona, so we map ARI → AZ.
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

            # Strip trailing doubleheader markers: G1/G2 or just 1/2
            team_str_clean = re.sub(r"G?\d$", "", team_str)

            away_team, home_team = _parse_teams(team_str_clean)
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


def _api_get_with_retry(
    client: httpx.Client, url: str, params: dict, max_retries: int = 3,
) -> httpx.Response:
    """GET with retry and exponential backoff. Raises on final failure."""
    for attempt in range(max_retries):
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError):
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


def fetch_events(year: int) -> list[dict]:
    """Fetch all settled KXMLBGAME events via the Events API.

    Paginates GET /events?series_ticker=KXMLBGAME&status=settled&with_nested_markets=true.
    For events missing nested markets (can happen with archived events),
    fetches individually via GET /events/{event_ticker}?with_nested_markets=true
    or falls back to GET /historical/markets?event_ticker={et}.
    """
    all_events = []
    cursor = ""
    page = 0

    with httpx.Client(timeout=60.0) as client:
        # Phase 1: Paginate the events endpoint
        while True:
            params = {
                "series_ticker": "KXMLBGAME",
                "status": "settled",
                "with_nested_markets": "true",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            resp = _api_get_with_retry(client, f"{API_BASE}/events", params)
            data = resp.json()

            events = data.get("events", [])
            if not events:
                break

            all_events.extend(events)
            page += 1
            cursor = data.get("cursor", "")

            if not cursor:
                break

            time.sleep(0.3)

        log(f"  Fetched {len(all_events)} events from events endpoint")

        # Phase 2: For events missing nested markets, fetch individually
        missing_markets = [
            e for e in all_events
            if not e.get("markets")
        ]
        if missing_markets:
            log(f"  {len(missing_markets)} events missing nested markets, fetching individually...")
            fetched = 0
            for e in missing_markets:
                et = e["event_ticker"]
                try:
                    resp = _api_get_with_retry(
                        client,
                        f"{API_BASE}/events/{et}",
                        {"with_nested_markets": "true"},
                    )
                    event_data = resp.json().get("event", {})
                    markets = event_data.get("markets", [])
                    if markets:
                        e["markets"] = markets
                        fetched += 1
                    else:
                        # Fall back to historical markets endpoint
                        resp2 = _api_get_with_retry(
                            client,
                            f"{API_BASE}/historical/markets",
                            {"event_ticker": et},
                        )
                        hist_markets = resp2.json().get("markets", [])
                        if hist_markets:
                            e["markets"] = hist_markets
                            fetched += 1
                except Exception:
                    pass
                time.sleep(0.2)
            log(f"    Resolved {fetched}/{len(missing_markets)} events")

        # Phase 3: Supplement with /markets endpoint to catch events the
        # Events API misses (known gap: ~11 events not returned by /events)
        known_event_tickers = {e["event_ticker"] for e in all_events}
        log(f"  Checking /markets endpoint for additional events...")
        supplement_markets: dict[str, list[dict]] = {}  # event_ticker -> [markets]
        supplement_cursor = ""

        while True:
            params = {
                "series_ticker": "KXMLBGAME",
                "status": "settled",
                "limit": 200,
            }
            if supplement_cursor:
                params["cursor"] = supplement_cursor

            resp = _api_get_with_retry(client, f"{API_BASE}/markets", params)
            data = resp.json()
            markets = data.get("markets", [])
            if not markets:
                break

            for m in markets:
                et = m.get("event_ticker", "")
                if et and et not in known_event_tickers:
                    if et not in supplement_markets:
                        supplement_markets[et] = []
                    supplement_markets[et].append(m)

            supplement_cursor = data.get("cursor", "")
            if not supplement_cursor:
                break
            time.sleep(0.3)

        for et, mkts in supplement_markets.items():
            all_events.append({"event_ticker": et, "markets": mkts})

        if supplement_markets:
            log(f"  Found {len(supplement_markets)} additional events from /markets endpoint")

    return all_events


def fetch_all_markets(year: int) -> list[dict]:
    """Fetch all settled KXMLBGAME markets via the Events API.

    Returns a flat list of market dicts (same format as before) for
    compatibility with downstream code.
    """
    events = fetch_events(year)

    # Flatten: extract nested markets from each event
    all_markets = []
    seen_tickers = set()
    events_without_markets = []

    for e in events:
        event_ticker = e.get("event_ticker", "")
        markets = e.get("markets", [])
        if not markets:
            events_without_markets.append(event_ticker)
            continue

        for m in markets:
            # Ensure event_ticker is set on each market
            if "event_ticker" not in m:
                m["event_ticker"] = event_ticker
            t = m.get("ticker", "")
            if t and t not in seen_tickers:
                seen_tickers.add(t)
                all_markets.append(m)

    if events_without_markets:
        log(f"  WARNING: {len(events_without_markets)} events had no markets:")
        for et in events_without_markets[:10]:
            log(f"    {et}")
        if len(events_without_markets) > 10:
            log(f"    ... and {len(events_without_markets) - 10} more")

    log(f"  {len(all_markets)} total markets from {len(events)} events")
    return all_markets


def fetch_historical_cutoff() -> float:
    """Fetch the archival cutoff timestamp from Kalshi.

    Markets settled before this timestamp are archived and must use
    the /historical/ endpoint for candlestick data.
    """
    with httpx.Client(timeout=30.0) as client:
        resp = _api_get_with_retry(client, f"{API_BASE}/historical/cutoff", {})
        data = resp.json()

    ts_str = data["market_settled_ts"]
    cutoff_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    log(f"  Historical cutoff: {ts_str}")
    return cutoff_dt.timestamp()


def group_markets_by_event(markets: list[dict], year: int) -> dict:
    """Group markets by event (game) and filter to target year."""
    events = {}
    parse_failures = []
    unknown_suffixes = []
    wrong_year = 0
    kalshi_teams_seen = set()

    for m in markets:
        event_ticker = m["event_ticker"]
        parsed = parse_event_ticker(event_ticker)
        if not parsed:
            if event_ticker not in {e for e, _ in parse_failures}:
                parse_failures.append((event_ticker, m["ticker"]))
            continue
        if int(parsed["game_date"][:4]) != year:
            wrong_year += 1
            continue

        if event_ticker not in events:
            events[event_ticker] = {
                **parsed,
                "event_ticker": event_ticker,
                "markets": {},
            }

        ticker_suffix = m["ticker"].split("-")[-1]
        team = TEAM_MAP.get(ticker_suffix)
        if team is None:
            unknown_suffixes.append((ticker_suffix, m["ticker"], event_ticker))
            team = ticker_suffix  # use raw suffix as fallback
        kalshi_teams_seen.add(team)

        events[event_ticker]["markets"][team] = {
            "ticker": m["ticker"],
            "result": m.get("result"),
            "volume": float(m.get("volume_fp") or m.get("volume") or 0),
            "open_interest": float(m.get("open_interest_fp") or m.get("open_interest") or 0),
            "open_time": m.get("open_time"),
            "close_time": m.get("close_time"),
            # Store fallback price fields for rescue when candles unavailable
            "previous_price": float(p) if (p := m.get("previous_price_dollars") or m.get("previous_price")) else None,
            "last_price": float(p) if (p := m.get("last_price_dollars") or m.get("last_price")) else None,
        }

    if parse_failures:
        log(f"  WARNING: {len(parse_failures)} event tickers failed to parse:")
        for et, ticker in parse_failures[:10]:
            log(f"    {et} (market: {ticker})")
        if len(parse_failures) > 10:
            log(f"    ... and {len(parse_failures) - 10} more")

    if unknown_suffixes:
        log(f"  WARNING: {len(unknown_suffixes)} markets with unknown team suffix:")
        for suffix, ticker, et in unknown_suffixes[:10]:
            log(f"    suffix={suffix} ticker={ticker} event={et}")

    if wrong_year:
        log(f"  Filtered out {wrong_year} markets from other years")

    log(f"  Kalshi team abbreviations seen: {sorted(kalshi_teams_seen)}")

    return events


def load_game_start_times(year: int) -> dict:
    """Load actual first-pitch timestamps from games parquet.

    Returns dict mapping (game_date, home_team, away_team) → list of unix
    timestamps. Multiple entries for doubleheaders.
    """
    games_path = GAMES_DIR / f"games_{year}.parquet"
    if not games_path.exists():
        log(f"  WARNING: {games_path} not found — will use fallback timing")
        return {}

    df = pd.read_parquet(games_path)
    df["game_date"] = df["game_date"].astype(str)

    start_times: dict[tuple, list[float]] = {}
    doubleheaders = 0
    for _, row in df.iterrows():
        gdt = row.get("game_datetime", "")
        if not gdt:
            continue
        try:
            dt = datetime.fromisoformat(gdt.replace("Z", "+00:00"))
            key = (row["game_date"], row["home_team_abbr"], row["away_team_abbr"])
            if key not in start_times:
                start_times[key] = []
            else:
                doubleheaders += 1
            start_times[key].append(dt.timestamp())
        except (ValueError, TypeError):
            continue

    # Sort each list so earliest game is first
    for key in start_times:
        start_times[key].sort()

    all_teams = set()
    for (_, h, a) in start_times:
        all_teams.add(h)
        all_teams.add(a)
    log(f"  Loaded {len(start_times)} matchups ({doubleheaders} doubleheader games) "
        f"from {games_path}")
    log(f"  Games parquet team abbreviations: {sorted(all_teams)}")
    return start_times


def _pick_best_start_time(
    start_times_list: list[float], market_open_time: str, market_close_time: str,
) -> float:
    """For doubleheaders, pick the game start time closest to the market's timing.

    The market open_time and close_time bracket the game. We pick the start time
    that's closest to the market close_time (since markets close around game end).
    """
    if len(start_times_list) == 1:
        return start_times_list[0]

    # Parse market close time as reference
    try:
        ref_ts = datetime.fromisoformat(
            market_close_time.replace("Z", "+00:00")
        ).timestamp()
    except (ValueError, TypeError):
        try:
            ref_ts = datetime.fromisoformat(
                market_open_time.replace("Z", "+00:00")
            ).timestamp()
        except (ValueError, TypeError):
            return start_times_list[0]

    # Pick the game start time closest to the market close time
    return min(start_times_list, key=lambda ts: abs(ts - ref_ts))


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


def _find_pregame_price(
    candles: list[dict], game_start_ts: float,
) -> tuple[float | None, str]:
    """Extract the last pre-game closing price from a list of candles.

    Finds the last candle ending before game_start_ts + 5min buffer
    with a reasonable price (0.04-0.96).

    Returns (price, reason) where reason is one of:
        "found", "no_valid_candles", "no_pregame_candles", "filtered_by_range"
    """
    valid = []
    for c in candles:
        price_data = c.get("price", {})
        close_p = _extract_candle_close(price_data)
        if close_p is None:
            continue
        valid.append((c["end_period_ts"], close_p))

    if not valid:
        return None, "no_valid_candles"

    cutoff_ts = game_start_ts + 300  # 5-minute buffer
    pre_game_all = [
        (ts, p) for ts, p in valid
        if ts <= cutoff_ts
    ]
    if not pre_game_all:
        return None, "no_pregame_candles"

    pre_game = [(ts, p) for ts, p in pre_game_all if 0.04 <= p <= 0.96]
    if pre_game:
        return pre_game[-1][1], "found"

    # Had pre-game candles but all were outside 0.04-0.96
    return None, "filtered_by_range"


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
        "no_candles": 0, "no_pregame": 0, "filtered_by_range": 0,
        "error": 0, "via_historical": 0, "via_live": 0,
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

            price, reason = _find_pregame_price(candles, game_start_ts)
            if price is not None:
                results[event_ticker] = price
                stats["found_hourly"] += 1
                stats[f"via_{source}"] += 1
            elif reason == "filtered_by_range":
                stats["filtered_by_range"] += 1
            elif candles:
                # Had candles but none were pre-game
                # Try daily candles as fallback for illiquid markets
                daily_params = {"period_interval": 1440, "start_ts": start_ts, "end_ts": end_ts}
                if source == "historical":
                    daily_url = f"{API_BASE}/historical/markets/{ticker}/candlesticks"
                else:
                    daily_url = f"{API_BASE}/series/KXMLBGAME/markets/{ticker}/candlesticks"
                daily = _fetch_candles_with_retry(client, daily_url, daily_params)
                if daily:
                    price, reason = _find_pregame_price(daily, game_start_ts)
                    if price is not None:
                        results[event_ticker] = price
                        stats["found_daily"] += 1
                        stats[f"via_{source}"] += 1
                    elif reason == "filtered_by_range":
                        stats["filtered_by_range"] += 1
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
    stats = {
        "found_hourly": 0, "found_daily": 0,
        "no_candles": 0, "no_pregame": 0, "filtered_by_range": 0, "error": 0,
    }

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

                price, reason = _find_pregame_price(candles, game_start_ts)
                if price is not None:
                    results[event_ticker] = price
                    stats["found_hourly"] += 1
                elif reason == "filtered_by_range":
                    stats["filtered_by_range"] += 1
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
) -> tuple[dict, dict]:
    """Fetch pre-game prices for all events.

    Routes requests to historical or live endpoint based on the archival cutoff.

    Returns:
        (prices dict, funnel stats dict)
    """
    all_markets = []
    live_markets = []
    drop_no_home_market = []
    drop_no_times = []
    drop_no_start_time = []
    drop_close_parse = []

    for event_ticker, event in events.items():
        home_team = event["home_team"]
        away_team = event["away_team"]
        game_date = event["game_date"]
        home_market = event["markets"].get(home_team)

        if not home_market:
            drop_no_home_market.append(
                f"game_date={game_date}, home={home_team}, away={away_team} "
                f"(event: {event_ticker}, market keys: {list(event['markets'].keys())})"
            )
            continue

        if not home_market.get("open_time") or not home_market.get("close_time"):
            drop_no_times.append(
                f"game_date={game_date}, home={home_team}, away={away_team} "
                f"(event: {event_ticker}, open_time={home_market.get('open_time')}, "
                f"close_time={home_market.get('close_time')})"
            )
            continue

        # Look up actual game start time (handles doubleheaders)
        gst_key = (game_date, home_team, away_team)
        start_times_list = game_start_times.get(gst_key)

        if not start_times_list:
            drop_no_start_time.append(
                f"game_date={game_date}, home={home_team}, away={away_team} "
                f"(event: {event_ticker})"
            )
            continue

        # For doubleheaders, pick the start time closest to this market's timing
        game_start_ts = _pick_best_start_time(
            start_times_list,
            home_market["open_time"],
            home_market["close_time"],
        )

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
            drop_close_parse.append(
                f"game_date={game_date}, home={home_team} "
                f"(close_time={home_market.get('close_time')})"
            )

    log(f"    {len(all_markets)} archived markets (individual requests)")
    log(f"    {len(live_markets)} live markets (batch eligible)")

    if drop_no_home_market:
        log(f"    {len(drop_no_home_market)} skipped (no home market):")
        for msg in drop_no_home_market[:5]:
            log(f"      {msg}")
        if len(drop_no_home_market) > 5:
            log(f"      ... and {len(drop_no_home_market) - 5} more")

    if drop_no_times:
        log(f"    {len(drop_no_times)} skipped (no open/close time):")
        for msg in drop_no_times[:5]:
            log(f"      {msg}")
        if len(drop_no_times) > 5:
            log(f"      ... and {len(drop_no_times) - 5} more")

    if drop_no_start_time:
        log(f"    {len(drop_no_start_time)} skipped (no game start time):")
        for msg in drop_no_start_time[:5]:
            log(f"      {msg}")
        if len(drop_no_start_time) > 5:
            log(f"      ... and {len(drop_no_start_time) - 5} more")

    if drop_close_parse:
        log(f"    {len(drop_close_parse)} skipped (close_time parse error):")
        for msg in drop_close_parse[:3]:
            log(f"      {msg}")

    results = {}
    total_filtered_by_range = 0

    with httpx.Client(timeout=30.0) as client:
        # Fetch archived markets individually (with retry + live fallback)
        if all_markets:
            log(f"\n  Fetching archived markets ({len(all_markets)})...")
            ind_results, ind_stats = fetch_pregame_prices_individual(
                client, all_markets,
            )
            results.update(ind_results)
            total_filtered_by_range += ind_stats.get("filtered_by_range", 0)
            log(f"    → {ind_stats['found_hourly']} hourly, "
                  f"{ind_stats['found_daily']} daily, "
                  f"{ind_stats['no_candles']} no candles, "
                  f"{ind_stats['no_pregame']} no pre-game price, "
                  f"{ind_stats.get('filtered_by_range', 0)} filtered by range, "
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
                live_stats["filtered_by_range"] = live_stats.get("filtered_by_range", 0) + ind_stats.get("filtered_by_range", 0)

            total_filtered_by_range += live_stats.get("filtered_by_range", 0)
            log(f"    → {live_stats['found_hourly']} hourly, "
                  f"{live_stats['found_daily']} daily, "
                  f"{live_stats['no_candles']} no candles, "
                  f"{live_stats['no_pregame']} no pre-game price, "
                  f"{live_stats.get('filtered_by_range', 0)} filtered by range, "
                  f"{live_stats['error']} errors")

    funnel = {
        "with_home_market": len(events) - len(drop_no_home_market),
        "drop_no_home_market": len(drop_no_home_market),
        "drop_no_times": len(drop_no_times),
        "drop_no_start_time": len(drop_no_start_time),
        "drop_close_parse": len(drop_close_parse),
        "routed_to_fetch": len(all_markets) + len(live_markets),
        "with_candle_price": len(results),
        "filtered_by_range": total_filtered_by_range,
    }

    return results, funnel


def build_dataset(events: dict, prices: dict) -> tuple[pd.DataFrame, dict]:
    """Build final dataset with market prices and outcomes.

    For events without candle prices, falls back to previous_price or
    last_price from the market object.

    Returns:
        (DataFrame, build stats dict)
    """
    rows = []
    no_home_or_away = 0
    no_price = 0
    rescued_by_fallback = 0

    for event_ticker, event in events.items():
        home_team = event["home_team"]
        away_team = event["away_team"]

        home_market = event["markets"].get(home_team)
        away_market = event["markets"].get(away_team)

        if not home_market or not away_market:
            no_home_or_away += 1
            continue

        home_won = home_market.get("result") == "yes"

        home_price = prices.get(event_ticker)
        price_source = "candle"

        # Fallback: use previous_price or last_price from market object
        if home_price is None:
            fallback = home_market.get("previous_price") or home_market.get("last_price")
            if fallback is not None and 0.04 <= fallback <= 0.96:
                home_price = fallback
                price_source = "fallback"
                rescued_by_fallback += 1

        if home_price is None:
            no_price += 1
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
            "price_source": price_source,
        })

    if rescued_by_fallback:
        log(f"  Rescued {rescued_by_fallback} games via fallback price (previous_price/last_price)")

    build_stats = {
        "no_home_or_away_market": no_home_or_away,
        "no_price": no_price,
        "rescued_by_fallback": rescued_by_fallback,
    }

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("game_date").reset_index(drop=True)
    return df, build_stats


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
    prices, funnel = fetch_all_pregame_prices(events, game_start_times, cutoff_ts)
    log(f"\n  Got prices for {len(prices)}/{len(events)} games")

    log(f"\nBuilding dataset...")
    df, build_stats = build_dataset(events, prices)
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

    # Pipeline funnel
    n_events = len(events)
    n_parse_ok = n_events  # already filtered in group_markets_by_event
    n_with_home = funnel["with_home_market"]
    n_with_times = n_with_home - funnel["drop_no_times"]
    n_with_start = n_with_times - funnel["drop_no_start_time"]
    n_routed = funnel["routed_to_fetch"]
    n_candle_price = funnel["with_candle_price"]
    n_final = len(df) if len(df) > 0 else 0

    log(f"\nPipeline funnel:")
    log(f"  Events after parse/year: {n_events:>5}")
    log(f"  With home market:        {n_with_home:>5}  ({funnel['drop_no_home_market']} no home market)")
    log(f"  With open/close time:    {n_with_times:>5}  ({funnel['drop_no_times']} missing times)")
    log(f"  With game start time:    {n_with_start:>5}  ({funnel['drop_no_start_time']} no start time)")
    log(f"  Routed to fetch:         {n_routed:>5}  ({funnel['drop_close_parse']} close_time parse err)")
    log(f"  With candle price:       {n_candle_price:>5}  ({funnel.get('filtered_by_range', 0)} filtered by 0.04-0.96)")
    log(f"  Rescued by fallback:     {build_stats['rescued_by_fallback']:>5}")
    log(f"  In final dataset:        {n_final:>5}  ({build_stats['no_home_or_away_market']} no away market, {build_stats['no_price']} no price)")

    # Coverage diagnostic: compare against games parquet
    log(f"\nCoverage diagnostic:")
    games_path = GAMES_DIR / f"games_{args.year}.parquet"
    if games_path.exists():
        games_df = pd.read_parquet(games_path)
        games_df["game_date"] = games_df["game_date"].astype(str)
        total_mlb = len(games_df)
        kalshi_events = len(events)
        kalshi_priced = len(df) if len(df) > 0 else 0

        log(f"  MLB games (parquet): {total_mlb}")
        log(f"  Kalshi events found: {kalshi_events}")
        log(f"  Kalshi with prices:  {kalshi_priced}")
        log(f"  Missing from Kalshi: {total_mlb - kalshi_events}")

        # Find specific missing games
        kalshi_keys = set()
        for et, ev in events.items():
            kalshi_keys.add((ev["game_date"], ev["home_team"], ev["away_team"]))

        missing = []
        for _, row in games_df.iterrows():
            key = (row["game_date"], row["home_team_abbr"], row["away_team_abbr"])
            if key not in kalshi_keys:
                missing.append(key)

        if missing:
            # Group by date range
            missing_dates = sorted(set(m[0] for m in missing))
            if missing_dates:
                log(f"  Missing date range: {missing_dates[0]} -> {missing_dates[-1]}")
                # Show first/last few
                early = [m for m in missing if m[0] <= missing_dates[min(5, len(missing_dates)-1)]]
                late = [m for m in missing if m[0] >= missing_dates[max(0, len(missing_dates)-6)]]
                if early:
                    log(f"  First missing games:")
                    for d, h, a in sorted(early)[:5]:
                        log(f"    {d}: {a} @ {h}")
                if late and late != early:
                    log(f"  Last missing games:")
                    for d, h, a in sorted(late)[-5:]:
                        log(f"    {d}: {a} @ {h}")
    else:
        log(f"  {games_path} not found — cannot compare")


if __name__ == "__main__":
    main()
