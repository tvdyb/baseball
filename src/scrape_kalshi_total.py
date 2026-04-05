#!/usr/bin/env python3
"""
Scrape Kalshi KXMLBTOTAL (over/under total runs) market data.

NRFI markets do NOT exist on Kalshi as of 2026-04.
KXMLBTOTAL markets offer multiple binary strikes (e.g. "Over 7.5 runs").
We scrape pre-game prices for all strikes to derive:
  - implied_total: the strike where P(over) ≈ 50%, interpolated
  - over_N_5_prob: P(over N.5) at each strike for 6.5-9.5

Markets exist from 2025 playoffs (Oct 2025) and the 2026 season.

Usage:
    python src/scrape_kalshi_total.py --year 2026
    python src/scrape_kalshi_total.py --year 2025  # only ~40 games (playoffs + WS)
"""

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

# Reuse helpers from ML scraper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_kalshi import (
    KALSHI_DIR,
    GAMES_DIR,
    API_BASE,
    MONTH_MAP,
    TEAM_MAP,
    KNOWN_TEAMS,
    log,
    parse_event_ticker,
    _api_get_with_retry,
    fetch_historical_cutoff,
    load_game_start_times,
    _pick_best_start_time,
    _find_pregame_price,
    _fetch_candles_with_retry,
)

SERIES_TICKER = "KXMLBTOTAL"

# All possible strikes Kalshi uses (floor_strike = N means "over N runs")
# From inspection: 2.5 through 13.5 seen, but most liquidity at 6.5-9.5
ALL_STRIKES = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]

# Strikes we report individual probabilities for (most liquid range)
REPORT_STRIKES = [6.5, 7.5, 8.5, 9.5]


def fetch_total_events(year: int) -> list[dict]:
    """Fetch all settled KXMLBTOTAL events with nested markets."""
    all_events: list[dict] = []
    cursor = ""

    with httpx.Client(timeout=60.0) as client:
        # Phase 1: events endpoint
        while True:
            params = {
                "series_ticker": SERIES_TICKER,
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
            cursor = data.get("cursor", "")
            if not cursor:
                break
            time.sleep(0.3)

        log(f"  Fetched {len(all_events)} events from events endpoint")

        # Phase 2: resolve missing nested markets
        missing = [e for e in all_events if not e.get("markets")]
        if missing:
            log(f"  {len(missing)} events missing nested markets, resolving...")
            resolved = 0
            for e in missing:
                et = e["event_ticker"]
                try:
                    # Try event detail endpoint first
                    resp = _api_get_with_retry(
                        client, f"{API_BASE}/events/{et}",
                        {"with_nested_markets": "true"},
                    )
                    mkts = resp.json().get("event", {}).get("markets", [])
                    if mkts:
                        e["markets"] = mkts
                        resolved += 1
                        time.sleep(0.15)
                        continue

                    # Fall back to historical markets endpoint
                    resp2 = _api_get_with_retry(
                        client, f"{API_BASE}/historical/markets",
                        {"event_ticker": et},
                    )
                    hist_mkts = resp2.json().get("markets", [])
                    if hist_mkts:
                        e["markets"] = hist_mkts
                        resolved += 1
                except Exception:
                    pass
                time.sleep(0.2)
            log(f"    Resolved {resolved}/{len(missing)}")

        # Phase 3: supplement from /markets endpoint (catch any gaps)
        known_tickers = {e["event_ticker"] for e in all_events}
        supplement: dict[str, list[dict]] = {}
        supp_cursor = ""
        while True:
            params = {
                "series_ticker": SERIES_TICKER,
                "status": "settled",
                "limit": 200,
            }
            if supp_cursor:
                params["cursor"] = supp_cursor
            resp = _api_get_with_retry(client, f"{API_BASE}/markets", params)
            data = resp.json()
            mkts = data.get("markets", [])
            if not mkts:
                break
            for m in mkts:
                et = m.get("event_ticker", "")
                if et and et not in known_tickers:
                    supplement.setdefault(et, []).append(m)
            supp_cursor = data.get("cursor", "")
            if not supp_cursor:
                break
            time.sleep(0.3)

        if supplement:
            for et, mkts in supplement.items():
                all_events.append({"event_ticker": et, "markets": mkts})
            log(f"  +{len(supplement)} events from /markets supplement")

    return all_events


def parse_total_event_ticker(event_ticker: str) -> dict | None:
    """Parse KXMLBTOTAL-26MAR311940LAACHC → game_date, home_team, away_team.

    Re-uses parse_event_ticker from scrape_kalshi but strips the KXMLBTOTAL prefix.
    """
    # Rewrite to use the same series-agnostic code path
    # parse_event_ticker expects KXMLBGAME-... format but just uses parts[1]
    # So we can fake the prefix
    fake_ticker = "KXMLBGAME-" + event_ticker.split("-", 1)[1]
    return parse_event_ticker(fake_ticker)


def group_total_events_by_game(
    events: list[dict], year: int,
) -> dict[str, dict]:
    """Group KXMLBTOTAL events by game, extracting per-strike market info.

    Returns dict: event_ticker → {game_date, home_team, away_team,
                                   markets: {strike → market_dict}}
    """
    by_event: dict[str, dict] = {}
    parse_failures = []
    wrong_year = 0

    for e in events:
        event_ticker = e.get("event_ticker", "")
        parsed = parse_total_event_ticker(event_ticker)
        if not parsed:
            parse_failures.append(event_ticker)
            continue
        if int(parsed["game_date"][:4]) != year:
            wrong_year += 1
            continue

        markets_by_strike: dict[float, dict] = {}
        for m in e.get("markets", []):
            strike = m.get("floor_strike")
            if strike is None:
                continue
            strike = float(strike)
            markets_by_strike[strike] = {
                "ticker": m["ticker"],
                "result": m.get("result"),
                "volume": float(m.get("volume_fp") or m.get("volume") or 0),
                "open_time": m.get("open_time"),
                "close_time": m.get("close_time"),
                "last_price": float(p) if (p := m.get("last_price_dollars") or m.get("last_price")) else None,
                "previous_price": float(p) if (p := m.get("previous_price_dollars") or m.get("previous_price")) else None,
            }

        if not markets_by_strike:
            continue

        by_event[event_ticker] = {
            **parsed,
            "event_ticker": event_ticker,
            "markets": markets_by_strike,
        }

    if parse_failures:
        log(f"  WARNING: {len(parse_failures)} event tickers failed to parse:")
        for et in parse_failures[:5]:
            log(f"    {et}")

    if wrong_year:
        log(f"  Filtered {wrong_year} events from other years")

    log(f"  {len(by_event)} total game events")
    return by_event


def _interpolate_implied_total(strike_prices: dict[float, float]) -> float | None:
    """Find the implied over/under line by interpolating where P(over) = 0.5.

    strike_prices: {strike → P(over strike)} e.g. {7.5: 0.62, 8.5: 0.47}
    Returns the implied total (e.g. 8.1), or None if not determinable.
    """
    if not strike_prices:
        return None

    sorted_strikes = sorted(strike_prices.keys())

    # Find adjacent strikes bracketing 0.5
    for i in range(len(sorted_strikes) - 1):
        s_lo = sorted_strikes[i]
        s_hi = sorted_strikes[i + 1]
        p_lo = strike_prices[s_lo]  # P(over s_lo) — higher prob
        p_hi = strike_prices[s_hi]  # P(over s_hi) — lower prob

        if p_lo >= 0.5 >= p_hi:
            # Linear interpolation between strikes
            # At s_lo: prob = p_lo; at s_hi: prob = p_hi
            if p_lo == p_hi:
                return (s_lo + s_hi) / 2
            frac = (p_lo - 0.5) / (p_lo - p_hi)
            return round(s_lo + frac * (s_hi - s_lo), 3)

    # If all probs above 0.5, implied total > max strike
    # If all probs below 0.5, implied total < min strike
    return None


def fetch_total_pregame_prices(
    events: dict[str, dict],
    game_start_times: dict,
    cutoff_ts: float,
) -> dict[str, dict[float, float]]:
    """Fetch pre-game prices for all strikes of each KXMLBTOTAL event.

    Returns: event_ticker → {strike → pregame_price}
    """
    # Build flat list of (event_ticker, strike, market, game_start_ts)
    to_fetch: list[tuple[str, float, dict, float]] = []

    no_start_time = 0
    no_times = 0
    fallback_used = 0

    for event_ticker, event in events.items():
        home_team = event["home_team"]
        away_team = event["away_team"]
        game_date = event["game_date"]

        gst_key = (game_date, home_team, away_team)
        start_times_list = game_start_times.get(gst_key)

        for strike, market in event["markets"].items():
            if not market.get("open_time") or not market.get("close_time"):
                no_times += 1
                continue

            if start_times_list:
                # Use actual game start time from games parquet
                game_start_ts = _pick_best_start_time(
                    start_times_list,
                    market["open_time"],
                    market["close_time"],
                )
            else:
                # Fallback: use market close_time as proxy for game start.
                # Total markets close early when the game ends (can_close_early=True),
                # so close_time is a reasonable upper bound. We use close_time - 30min
                # as an estimated game start to avoid grabbing in-game candles.
                try:
                    close_dt = datetime.fromisoformat(
                        market["close_time"].replace("Z", "+00:00")
                    )
                    game_start_ts = close_dt.timestamp() - 10800  # -3h buffer (avg game ~2h40m)
                    fallback_used += 1
                except (ValueError, TypeError):
                    no_start_time += 1
                    continue

            to_fetch.append((event_ticker, strike, market, game_start_ts))

        if not start_times_list and not any(
            market.get("open_time") and market.get("close_time")
            for market in event["markets"].values()
        ):
            no_start_time += 1

    log(f"  {no_start_time} events skipped (no game start time or close_time)")
    log(f"  {no_times} markets skipped (no open/close time)")
    if fallback_used:
        log(f"  {fallback_used} strike-markets using close_time as game start proxy (no games parquet)")
    log(f"  {len(to_fetch)} (event, strike) pairs to fetch")

    # Split into archived vs live
    archived: list[tuple[str, float, dict, float]] = []
    live: list[tuple[str, float, dict, float]] = []

    for item in to_fetch:
        event_ticker, strike, market, game_start_ts = item
        try:
            close_dt = datetime.fromisoformat(
                market["close_time"].replace("Z", "+00:00")
            )
            if close_dt.timestamp() >= cutoff_ts:
                live.append(item)
            else:
                archived.append(item)
        except (ValueError, TypeError):
            archived.append(item)

    log(f"  {len(archived)} archived, {len(live)} live")

    # Results: event_ticker → {strike → price}
    results: dict[str, dict[float, float]] = {}

    def _store(event_ticker: str, strike: float, price: float):
        results.setdefault(event_ticker, {})[strike] = price

    def _fetch_single(
        client: httpx.Client,
        event_ticker: str,
        strike: float,
        market: dict,
        game_start_ts: float,
        source_hint: str = "auto",
    ):
        ticker = market["ticker"]
        try:
            open_dt = datetime.fromisoformat(market["open_time"].replace("Z", "+00:00"))
            close_dt = datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
            start_ts = int(open_dt.timestamp())
            end_ts = int(close_dt.timestamp())
            params = {"period_interval": 60, "start_ts": start_ts, "end_ts": end_ts}

            # Try historical first
            candles = _fetch_candles_with_retry(
                client, f"{API_BASE}/historical/markets/{ticker}/candlesticks", params,
            )
            if candles is None:
                # Fall back to live series endpoint
                candles = _fetch_candles_with_retry(
                    client,
                    f"{API_BASE}/series/{SERIES_TICKER}/markets/{ticker}/candlesticks",
                    params,
                )

            if candles is None:
                return

            price, reason = _find_pregame_price(candles, game_start_ts)
            if price is not None:
                _store(event_ticker, strike, price)
            elif reason == "no_candles" or not candles:
                pass  # no data available
            else:
                # Try daily candles as fallback
                daily_params = {"period_interval": 1440, "start_ts": start_ts, "end_ts": end_ts}
                daily = _fetch_candles_with_retry(
                    client,
                    f"{API_BASE}/historical/markets/{ticker}/candlesticks",
                    daily_params,
                )
                if daily is None:
                    daily = _fetch_candles_with_retry(
                        client,
                        f"{API_BASE}/series/{SERIES_TICKER}/markets/{ticker}/candlesticks",
                        daily_params,
                    )
                if daily:
                    price, _ = _find_pregame_price(daily, game_start_ts)
                    if price is not None:
                        _store(event_ticker, strike, price)
        except Exception:
            pass

    with httpx.Client(timeout=30.0) as client:
        # Archived: individual requests
        if archived:
            log(f"\n  Fetching archived markets ({len(archived)} strike-markets)...")
            for i, (event_ticker, strike, market, game_start_ts) in enumerate(archived):
                _fetch_single(client, event_ticker, strike, market, game_start_ts)
                if (i + 1) % 200 == 0 or i + 1 == len(archived):
                    n_events_with_data = len(results)
                    n_strikes_with_data = sum(len(v) for v in results.values())
                    log(f"    progress: {i + 1}/{len(archived)} "
                        f"({n_events_with_data} events, {n_strikes_with_data} strikes)")
                time.sleep(0.1)

        # Live: batch by event (all strikes of one game together) to minimize requests
        if live:
            log(f"\n  Fetching live markets ({len(live)} strike-markets)...")
            # Group live items by event_ticker
            live_by_event: dict[str, list[tuple[float, dict, float]]] = {}
            for event_ticker, strike, market, game_start_ts in live:
                live_by_event.setdefault(event_ticker, []).append(
                    (strike, market, game_start_ts)
                )

            processed = 0
            for event_ticker, items in live_by_event.items():
                tickers = [m["ticker"] for _, m, _ in items]
                min_ts = min(
                    int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
                    for _, m, _ in items
                    if m.get("open_time")
                )
                max_ts = max(
                    int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp())
                    for _, m, _ in items
                    if m.get("close_time")
                )

                try:
                    resp = client.get(
                        f"{API_BASE}/markets/candlesticks",
                        params={
                            "market_tickers": ",".join(tickers),
                            "series_ticker": SERIES_TICKER,
                            "period_interval": 60,
                            "start_ts": min_ts,
                            "end_ts": max_ts,
                        },
                    )
                    resp.raise_for_status()
                    by_ticker = {
                        m["market_ticker"]: m.get("candlesticks", [])
                        for m in resp.json().get("markets", [])
                    }

                    for strike, market, game_start_ts in items:
                        ticker = market["ticker"]
                        candles = by_ticker.get(ticker, [])
                        if candles:
                            price, _ = _find_pregame_price(candles, game_start_ts)
                            if price is not None:
                                _store(event_ticker, strike, price)

                except Exception:
                    # Fall back to individual requests for this event
                    for strike, market, game_start_ts in items:
                        _fetch_single(
                            client, event_ticker, strike, market, game_start_ts
                        )
                        time.sleep(0.1)

                processed += len(items)
                if processed % 100 == 0 or processed >= len(live):
                    n_events_with_data = len(results)
                    log(f"    live progress: {processed}/{len(live)} "
                        f"({n_events_with_data} events)")
                time.sleep(0.15)

    n_strikes_total = sum(len(v) for v in results.values())
    log(f"\n  Prices fetched: {len(results)} events, {n_strikes_total} strike-prices")
    return results


def build_total_dataset(
    events: dict[str, dict],
    prices: dict[str, dict[float, float]],
) -> pd.DataFrame:
    """Build DataFrame with implied totals and per-strike probabilities.

    Columns:
        game_date, home_team, away_team, event_ticker
        implied_total      — interpolated O/U line from market prices
        over_{N}5_prob     — P(over N.5 runs) for N in 6,7,8,9
        actual_total       — None (populated later from games parquet)
        volume             — total dollar volume across all strikes
        n_strikes          — number of strikes with price data
    """
    rows = []
    no_price = 0

    for event_ticker, event in events.items():
        strike_prices = prices.get(event_ticker)

        # Fallback: use last_price from market objects if no candle data
        if not strike_prices:
            fallback_prices: dict[float, float] = {}
            for strike, mkt in event["markets"].items():
                p = mkt.get("last_price") or mkt.get("previous_price")
                if p is not None and 0.04 <= float(p) <= 0.96:
                    fallback_prices[float(strike)] = float(p)
            if fallback_prices:
                strike_prices = fallback_prices
            else:
                no_price += 1
                continue

        implied_total = _interpolate_implied_total(strike_prices)

        # Per-strike probabilities for report strikes
        over_probs = {}
        for n in [6, 7, 8, 9]:
            strike = float(n) + 0.5
            p = strike_prices.get(strike)
            over_probs[f"over_{n}5_prob"] = round(p, 4) if p is not None else None

        # Total volume across all strikes
        total_volume = sum(
            mkt["volume"]
            for mkt in event["markets"].values()
        )

        # Actual outcome: derive from result fields
        actual_total = None
        settled_strikes = {
            s: mkt["result"]
            for s, mkt in event["markets"].items()
            if mkt.get("result") in ("yes", "no")
        }
        if settled_strikes:
            # All strikes with result=yes resolved over; find boundary
            yes_strikes = {s for s, r in settled_strikes.items() if r == "yes"}
            no_strikes = {s for s, r in settled_strikes.items() if r == "no"}
            if yes_strikes and no_strikes:
                # Max yes_strike < min no_strike → actual total between them
                max_yes = max(yes_strikes)
                min_no = min(no_strikes)
                # actual total is in (max_yes, min_no], so it's min_no - 0.5 + something
                # e.g. max_yes=7.5 (over=yes), min_no=8.5 (over=no) → total in [8,8]
                actual_total = float(max_yes) + 0.5  # conservatively the low end
            elif yes_strikes and not no_strikes:
                actual_total = float(max(yes_strikes)) + 0.5  # all over; total > max strike
            elif no_strikes and not yes_strikes:
                actual_total = float(min(no_strikes)) - 0.5  # all under; total < min strike

        rows.append({
            "game_date": event["game_date"],
            "home_team": event["home_team"],
            "away_team": event["away_team"],
            "event_ticker": event_ticker,
            "implied_total": round(implied_total, 3) if implied_total is not None else None,
            "actual_total": actual_total,
            "volume": total_volume,
            "n_strikes": len(strike_prices),
            **over_probs,
        })

    if no_price:
        log(f"  {no_price} events skipped (no price data)")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("game_date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Kalshi KXMLBTOTAL (over/under) market data"
    )
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    KALSHI_DIR.mkdir(parents=True, exist_ok=True)
    output_path = KALSHI_DIR / f"kalshi_total_{args.year}.parquet"

    log(f"\nFetching Kalshi KXMLBTOTAL markets for {args.year}...")
    events_raw = fetch_total_events(args.year)

    log(f"\nGrouping events by game...")
    events = group_total_events_by_game(events_raw, args.year)

    log(f"\nFetching historical cutoff...")
    cutoff_ts = fetch_historical_cutoff()

    log(f"\nLoading game start times...")
    game_start_times = load_game_start_times(args.year)

    log(f"\nFetching pre-game prices...")
    prices = fetch_total_pregame_prices(events, game_start_times, cutoff_ts)

    log(f"\nBuilding dataset...")
    df = build_total_dataset(events, prices)
    df.to_parquet(output_path, index=False)
    log(f"  Saved {len(df)} games to {output_path}")

    if len(df) > 0:
        log(f"\n  Date range: {df['game_date'].min()} -> {df['game_date'].max()}")

        if df["implied_total"].notna().any():
            log(f"  Implied total: mean={df['implied_total'].mean():.2f}, "
                f"min={df['implied_total'].min():.2f}, "
                f"max={df['implied_total'].max():.2f}")
            log(f"  Implied total coverage: {df['implied_total'].notna().sum()}/{len(df)}")

        if df["actual_total"].notna().any():
            log(f"  Actual total: mean={df['actual_total'].mean():.1f}")

        log(f"  Avg volume/game: ${df['volume'].mean():,.0f}")
        log(f"  Avg strikes with price data: {df['n_strikes'].mean():.1f}")

        df["month"] = df["game_date"].str[:7]
        log(f"\n  By month:")
        for month in sorted(df["month"].unique()):
            sub = df[df["month"] == month]
            n_total = df["implied_total"].notna().sum() if "implied_total" in df.columns else 0
            log(f"    {month}: {len(sub)} games")

        # Show distribution of implied totals
        if df["implied_total"].notna().any():
            totals = df["implied_total"].dropna()
            log(f"\n  Implied total distribution:")
            for lo, hi in [(5, 7), (7, 8), (8, 9), (9, 10), (10, 12)]:
                n = ((totals >= lo) & (totals < hi)).sum()
                log(f"    [{lo}, {hi}): {n} ({n/len(totals):.1%})")

        # Calibration check (if actual totals available)
        valid = df[df["implied_total"].notna() & df["actual_total"].notna()].copy()
        if len(valid) > 10:
            valid["over_actual"] = (valid["actual_total"] >= valid["implied_total"]).astype(int)
            log(f"\n  Calibration (over/under at implied total, N={len(valid)}):")
            log(f"    Over rate: {valid['over_actual'].mean():.3f} (expect ~0.500)")


if __name__ == "__main__":
    main()
