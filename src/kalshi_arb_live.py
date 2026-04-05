#!/usr/bin/env python3
"""
DK vs Kalshi Moneyline Arbitrage — Live Daily Scanner
======================================================
Fetches today's MLB games, current Kalshi prices, and DraftKings ML odds,
then surfaces bets where DK's implied probability exceeds Kalshi's price by
at least the configured edge threshold (default 7%).

Usage:
    # Live scan (requires ODDS_API_KEY env var for DK odds)
    python src/kalshi_arb_live.py

    # Specify bankroll and edge threshold
    python src/kalshi_arb_live.py --bankroll 5000 --edge-threshold 0.07

    # Use a specific date instead of today
    python src/kalshi_arb_live.py --date 2025-07-15

    # Accept manually-entered DK lines instead of fetching via API
    python src/kalshi_arb_live.py --manual-dk

    # Dry-run with hardcoded sample data (no live API calls)
    python src/kalshi_arb_live.py --dry-run

    # Output machine-readable JSON (for downstream automation)
    python src/kalshi_arb_live.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Optional: pretty table output via pandas; fall back to plain text
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
MLB_API    = "https://statsapi.mlb.com/api/v1"

# Odds-API endpoint for live DK MLB moneylines (free tier: 500 req/month)
ODDS_API_ENDPOINT = (
    "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    "?apiKey={key}&regions=us&markets=h2h&bookmakers=draftkings&oddsFormat=american"
)

DEFAULT_EDGE_THRESHOLD = 0.07   # 7% minimum DK-implied minus Kalshi price
DEFAULT_KELLY_FRACTION = 0.50   # half-Kelly
DEFAULT_BANKROLL       = 1000.0 # dollars

# Team abbreviation mappings: Odds-API team names → our internal abbreviations
# Kalshi uses the same abbreviations we use elsewhere (ATH not OAK, AZ not ARI)
ODDS_API_TO_ABB: dict[str, str] = {
    "Arizona Diamondbacks":    "AZ",
    "Atlanta Braves":          "ATL",
    "Baltimore Orioles":       "BAL",
    "Boston Red Sox":          "BOS",
    "Chicago Cubs":            "CHC",
    "Chicago White Sox":       "CWS",
    "Cincinnati Reds":         "CIN",
    "Cleveland Guardians":     "CLE",
    "Colorado Rockies":        "COL",
    "Detroit Tigers":          "DET",
    "Houston Astros":          "HOU",
    "Kansas City Royals":      "KC",
    "Los Angeles Angels":      "LAA",
    "Los Angeles Dodgers":     "LAD",
    "Miami Marlins":           "MIA",
    "Milwaukee Brewers":       "MIL",
    "Minnesota Twins":         "MIN",
    "New York Mets":           "NYM",
    "New York Yankees":        "NYY",
    "Oakland Athletics":       "ATH",
    "Athletics":               "ATH",
    "Philadelphia Phillies":   "PHI",
    "Pittsburgh Pirates":      "PIT",
    "San Diego Padres":        "SD",
    "San Francisco Giants":    "SF",
    "Seattle Mariners":        "SEA",
    "St. Louis Cardinals":     "STL",
    "Tampa Bay Rays":          "TB",
    "Texas Rangers":           "TEX",
    "Toronto Blue Jays":       "TOR",
    "Washington Nationals":    "WSH",
}

# Kalshi ticker suffix → internal abbreviation (mirrors scrape_kalshi.py)
KALSHI_TEAM_MAP: dict[str, str] = {
    "LAD": "LAD", "TOR": "TOR", "NYY": "NYY", "NYM": "NYM",
    "BOS": "BOS", "HOU": "HOU", "ATL": "ATL", "SD":  "SD",
    "SF":  "SF",  "SEA": "SEA", "MIN": "MIN", "CLE": "CLE",
    "BAL": "BAL", "TB":  "TB",  "TEX": "TEX", "AZ":  "AZ",
    "MIL": "MIL", "CHC": "CHC", "CWS": "CWS", "COL": "COL",
    "PIT": "PIT", "CIN": "CIN", "STL": "STL", "KC":  "KC",
    "PHI": "PHI", "MIA": "MIA", "DET": "DET", "WSH": "WSH",
    "LAA": "LAA", "ATH": "ATH", "ARI": "AZ",
}

MONTH_ABBR: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GameOdds:
    """Holds all pricing data for a single game."""
    game_date:       str
    home_team:       str
    away_team:       str
    game_time:       str = ""        # ISO-8601 UTC string from MLB API
    home_sp:         str = "TBD"
    away_sp:         str = "TBD"

    # Kalshi prices (probabilities, 0-1)
    kalshi_home_prob: Optional[float] = None
    kalshi_away_prob: Optional[float] = None
    kalshi_home_ticker: str = ""
    kalshi_event_ticker: str = ""
    kalshi_source:   str = ""        # "ask", "last_trade", "mid", "fallback"

    # DraftKings lines (American odds → vig-removed probability)
    dk_home_ml:      Optional[int]   = None
    dk_away_ml:      Optional[int]   = None
    dk_home_prob:    Optional[float] = None
    dk_away_prob:    Optional[float] = None

    # Computed
    edge_home:       Optional[float] = None
    edge_away:       Optional[float] = None


@dataclass
class ArbPick:
    """A qualifying arbitrage opportunity."""
    game_date:       str
    matchup:         str             # "AWAY @ HOME"
    game_time:       str
    home_sp:         str
    away_sp:         str
    side:            str             # "home" or "away"
    bet_team:        str             # team abbreviation to bet on
    edge:            float           # DK_implied - Kalshi_price (positive = edge)
    kalshi_price:    float           # the price we pay on Kalshi (probability)
    dk_implied:      float           # vig-removed DK probability for this side
    dk_ml:           int             # raw DK American odds
    kelly_fraction:  float           # half-Kelly fraction of bankroll
    stake:           float           # dollar amount to bet (given bankroll)
    ev_per_100:      float           # expected value per $100 wagered
    bankroll:        float           # bankroll used for sizing


# ---------------------------------------------------------------------------
# Math helpers (mirrors kalshi_arb.py)
# ---------------------------------------------------------------------------

def american_to_prob(odds: float) -> float:
    """Raw implied probability from American odds (includes vig)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def remove_vig(p_home: float, p_away: float) -> tuple[float, float]:
    """Normalize raw implied probs to sum to 1 (remove overround)."""
    total = p_home + p_away
    if total == 0:
        return 0.5, 0.5
    return p_home / total, p_away / total


def kelly_fraction(p_win: float, kalshi_price: float,
                   fraction: float = 0.5) -> float:
    """
    Half-Kelly (by default) fraction of bankroll to wager.

    For a Kalshi binary contract at price `kalshi_price`:
        net_odds b = (1 - kalshi_price) / kalshi_price
        Kelly f    = (p_win * b - (1 - p_win)) / b
        Half-Kelly = f * fraction
    """
    if kalshi_price <= 0 or kalshi_price >= 1 or p_win <= 0:
        return 0.0
    b = (1.0 - kalshi_price) / kalshi_price
    f = (p_win * b - (1.0 - p_win)) / b
    return max(0.0, f * fraction)


def ev_per_100(p_win: float, kalshi_price: float) -> float:
    """
    Expected value per $100 wagered at `kalshi_price`.

    A $100 bet on a Kalshi binary contract at price `kalshi_price` wins
    $100 * (1 - kalshi_price) / kalshi_price if correct, loses $100 otherwise.
    EV = p_win * payout - (1 - p_win) * 100
    """
    if kalshi_price <= 0 or kalshi_price >= 1:
        return 0.0
    payout = 100.0 * (1.0 - kalshi_price) / kalshi_price
    return p_win * payout - (1.0 - p_win) * 100.0


# ---------------------------------------------------------------------------
# 1. Fetch today's MLB schedule
# ---------------------------------------------------------------------------

def fetch_todays_games(client: httpx.Client, target_date: str) -> list[dict]:
    """
    Fetch scheduled MLB games for `target_date` via the MLB Stats API.

    Reuses the same logic as predict.py::fetch_todays_games so the team
    abbreviations and game structure are identical across the pipeline.
    """
    resp = client.get(
        f"{MLB_API}/schedule",
        params={
            "sportId":   1,
            "date":      target_date,
            "hydrate":   "probablePitcher,venue,team,linescore",
            "gameType":  "R",
        },
    )
    resp.raise_for_status()
    data = resp.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            home_team    = home.get("team", {})
            away_team    = away.get("team", {})
            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})

            games.append({
                "game_pk":        g.get("gamePk"),
                "game_date":      g.get("officialDate", target_date),
                "status":         g.get("status", {}).get("abstractGameState", ""),
                "game_time":      g.get("gameDate", ""),
                "home_team":      home_team.get("abbreviation", ""),
                "away_team":      away_team.get("abbreviation", ""),
                "home_team_name": home_team.get("name", ""),
                "away_team_name": away_team.get("name", ""),
                "home_sp_name":   home_pitcher.get("fullName", "TBD"),
                "away_sp_name":   away_pitcher.get("fullName", "TBD"),
            })
    return games


# ---------------------------------------------------------------------------
# 2. Fetch Kalshi prices
# ---------------------------------------------------------------------------

def _parse_event_ticker_date(event_ticker: str) -> Optional[str]:
    """
    Extract game date (YYYY-MM-DD) from a KXMLBGAME event ticker.

    Ticker format: KXMLBGAME-25JUL31ATLCIN  or  KXMLBGAME-26MAR252005NYYSF
    """
    import re
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None
    code = parts[1]
    m = re.match(r"(\d{2})([A-Z]{3})(\d{1,2})", code)
    if not m:
        return None
    year  = 2000 + int(m.group(1))
    month = MONTH_ABBR.get(m.group(2))
    day   = int(m.group(3))
    if not month or day < 1 or day > 31:
        return None
    try:
        datetime(year, month, day)
        return f"{year}-{month:02d}-{day:02d}"
    except ValueError:
        return None


def _parse_teams_from_suffix(team_str: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse 'DETATH' or 'NYYSF' into (away_abbr, home_abbr).
    Tries 3-char then 2-char splits for each side.
    """
    known = set(KALSHI_TEAM_MAP.keys())
    for away_len in [3, 2]:
        if len(team_str) < away_len + 2:
            continue
        away_try = team_str[:away_len]
        home_try = team_str[away_len:]
        if away_try in known and home_try in known:
            return KALSHI_TEAM_MAP[away_try], KALSHI_TEAM_MAP[home_try]
    return None, None


def fetch_kalshi_prices(
    client: httpx.Client,
    target_date: str,
    verbose: bool = False,
) -> dict[tuple[str, str], dict]:
    """
    Fetch current Kalshi prices for all active KXMLBGAME markets on `target_date`.

    Returns a dict keyed by (home_team, away_team) → price info dict:
        {
            "kalshi_home_prob":   float,
            "kalshi_away_prob":   float,
            "home_ticker":        str,
            "event_ticker":       str,
            "source":             str,   # "ask", "last_trade", "mid", "fallback"
        }

    Strategy for obtaining the current price:
      1. Use best_ask (the current offer price) — this is what we'd actually pay.
      2. Fall back to last_price if ask is unavailable.
      3. Fall back to (yes_bid + yes_ask) / 2 if available as mid.
      4. Fall back to previous_price as last resort.

    Only open (active) markets are fetched since we care about today's pre-game lines.
    """
    results: dict[tuple[str, str], dict] = {}
    import re

    # Paginate through open KXMLBGAME markets
    cursor = ""
    page   = 0
    raw_markets: list[dict] = []

    while True:
        params: dict = {
            "series_ticker": "KXMLBGAME",
            "status":        "open",
            "limit":         200,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = client.get(f"{KALSHI_API}/markets", params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            print(f"  [kalshi] HTTP {exc.response.status_code} fetching markets page {page}")
            break
        except httpx.RequestError as exc:
            print(f"  [kalshi] Network error fetching markets: {exc}")
            break

        data    = resp.json()
        markets = data.get("markets", [])
        if not markets:
            break

        raw_markets.extend(markets)
        page  += 1
        cursor = data.get("cursor", "")
        if not cursor:
            break
        time.sleep(0.2)

    if verbose:
        print(f"  [kalshi] Fetched {len(raw_markets)} open KXMLBGAME markets across {page} pages")

    # Group markets by event, then extract prices
    event_groups: dict[str, dict] = {}   # event_ticker → {"home": m, "away": m, ...}

    for m in raw_markets:
        et = m.get("event_ticker", "")
        if not et.startswith("KXMLBGAME"):
            continue

        # Filter to today's date
        mkt_date = _parse_event_ticker_date(et)
        if mkt_date != target_date:
            continue

        event_groups.setdefault(et, {})

        # The last segment of the market ticker is the team abbreviation
        ticker = m.get("ticker", "")
        suffix = ticker.split("-")[-1]
        team   = KALSHI_TEAM_MAP.get(suffix, suffix)
        event_groups[et][team] = m

    if verbose:
        print(f"  [kalshi] Found {len(event_groups)} events for {target_date}")

    for et, team_markets in event_groups.items():
        # Parse teams from the event ticker to determine home/away
        # e.g. KXMLBGAME-25JUL31ATLCIN → away=ATL, home=CIN
        parts = et.split("-")
        if len(parts) < 2:
            continue
        code = parts[1]
        # Strip year+month+day (+optional time)
        team_str_match = re.search(r"\d{2}[A-Z]{3}\d{1,2}(?:\d{4})?([A-Z]+)", code)
        if not team_str_match:
            continue
        team_str = team_str_match.group(1)
        # Strip trailing doubleheader markers G1/G2 or trailing digit
        team_str = re.sub(r"G?\d$", "", team_str)
        away_team, home_team = _parse_teams_from_suffix(team_str)
        if not away_team or not home_team:
            continue

        home_market = team_markets.get(home_team)
        away_market = team_markets.get(away_team)
        if not home_market:
            continue

        # Extract the best available price for the home team
        def _extract_price(mkt: dict) -> tuple[Optional[float], str]:
            """Return (probability, source_label) for a market dict."""
            # Best ask = the price we would pay to buy YES contracts
            ask = mkt.get("yes_ask")
            if ask is not None:
                p = float(ask)
                if 0.01 <= p <= 0.99:
                    return p / 100.0, "ask"  # Kalshi quotes in cents (0-99)

            # Some endpoints return prices already as 0-1 floats
            ask_float = mkt.get("yes_ask_price")
            if ask_float is not None:
                p = float(ask_float)
                if 0.01 <= p <= 0.99:
                    return p, "ask"

            # Mid-market
            bid = mkt.get("yes_bid")
            if bid is not None and ask is not None:
                mid = (float(bid) + float(ask)) / 2.0
                if 0.01 <= mid <= 0.99:
                    return mid / 100.0, "mid"

            # Last trade price
            last = mkt.get("last_price")
            if last is not None:
                p = float(last)
                # Could be in cents (0-99) or 0-1; normalize
                if p > 1.0:
                    p /= 100.0
                if 0.01 <= p <= 0.99:
                    return p, "last_trade"

            # previous_price as fallback
            prev = mkt.get("previous_price")
            if prev is not None:
                p = float(prev)
                if p > 1.0:
                    p /= 100.0
                if 0.01 <= p <= 0.99:
                    return p, "fallback"

            return None, "none"

        home_price, source = _extract_price(home_market)
        if home_price is None:
            continue

        results[(home_team, away_team)] = {
            "kalshi_home_prob":  round(home_price, 4),
            "kalshi_away_prob":  round(1.0 - home_price, 4),
            "home_ticker":       home_market.get("ticker", ""),
            "away_ticker":       away_market.get("ticker", "") if away_market else "",
            "event_ticker":      et,
            "source":            source,
        }

    return results


# ---------------------------------------------------------------------------
# 3. Fetch DraftKings odds via the-odds-api.com
# ---------------------------------------------------------------------------

def fetch_dk_odds_via_api(
    api_key: str,
    client: httpx.Client,
    verbose: bool = False,
) -> dict[tuple[str, str], dict]:
    """
    Fetch current DraftKings MLB moneylines from the-odds-api.com.

    Returns dict keyed by (home_team_abbr, away_team_abbr) → {
        "home_ml": int,
        "away_ml": int,
        "home_prob": float,
        "away_prob": float,
        "game_time": str,
    }

    Free-tier callers have ~500 requests/month; each call fetches all games.
    The key is expected in ODDS_API_KEY environment variable.
    """
    url = (
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        f"?apiKey={api_key}&regions=us&markets=h2h"
        "&bookmakers=draftkings&oddsFormat=american"
    )

    try:
        resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"  [dk] Odds-API HTTP {exc.response.status_code}: {exc.response.text[:200]}")
        return {}
    except httpx.RequestError as exc:
        print(f"  [dk] Odds-API network error: {exc}")
        return {}

    events = resp.json()
    if verbose:
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        print(f"  [dk] Odds-API requests used: {used}, remaining: {remaining}")
        print(f"  [dk] Found {len(events)} events from Odds-API")

    results: dict[tuple[str, str], dict] = {}

    for event in events:
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        home_abbr = ODDS_API_TO_ABB.get(home_name)
        away_abbr = ODDS_API_TO_ABB.get(away_name)

        if not home_abbr or not away_abbr:
            if verbose:
                print(f"  [dk] Unknown team names: '{home_name}' / '{away_name}'")
            continue

        # Find DraftKings bookmaker in the bookmakers list
        dk_book = next(
            (b for b in event.get("bookmakers", []) if b.get("key") == "draftkings"),
            None,
        )
        if not dk_book:
            continue

        # h2h market has two outcomes: home and away
        h2h = next(
            (m for m in dk_book.get("markets", []) if m.get("key") == "h2h"),
            None,
        )
        if not h2h:
            continue

        home_ml = away_ml = None
        for outcome in h2h.get("outcomes", []):
            name = outcome.get("name", "")
            price = outcome.get("price")
            if name == home_name:
                home_ml = int(price)
            elif name == away_name:
                away_ml = int(price)

        if home_ml is None or away_ml is None:
            continue

        # Reject extreme lines (>±500) — likely in-game / stale data
        if abs(home_ml) > 500 or abs(away_ml) > 500:
            if verbose:
                print(f"  [dk] Skipping {away_abbr}@{home_abbr}: "
                      f"extreme lines {away_ml}/{home_ml}")
            continue

        raw_home = american_to_prob(home_ml)
        raw_away = american_to_prob(away_ml)
        dk_home_prob, dk_away_prob = remove_vig(raw_home, raw_away)

        results[(home_abbr, away_abbr)] = {
            "home_ml":   home_ml,
            "away_ml":   away_ml,
            "home_prob": round(dk_home_prob, 4),
            "away_prob": round(dk_away_prob, 4),
            "game_time": event.get("commence_time", ""),
        }

    return results


def fetch_dk_odds_manual() -> dict[tuple[str, str], dict]:
    """
    Prompt the user to enter DraftKings ML lines for each game interactively.

    Useful when the Odds-API key is unavailable or the user wants to override.
    Returns the same structure as fetch_dk_odds_via_api.
    """
    print("\nManual DK line entry. Type 'done' when finished.")
    print("Format: AWAY HOME AWAY_ML HOME_ML  (e.g.  NYY BOS -130 +110)")
    print("American odds: favorite is negative, underdog is positive.\n")

    results: dict[tuple[str, str], dict] = {}

    while True:
        try:
            raw = input("  Enter line (or 'done'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() in ("done", "q", "quit", "exit", ""):
            break

        parts = raw.split()
        if len(parts) != 4:
            print("  Expected: AWAY HOME AWAY_ML HOME_ML")
            continue

        away_abbr, home_abbr = parts[0].upper(), parts[1].upper()
        try:
            away_ml = int(parts[2])
            home_ml = int(parts[3])
        except ValueError:
            print("  ML lines must be integers (e.g. -130 or +110)")
            continue

        if abs(home_ml) > 500 or abs(away_ml) > 500:
            print("  Warning: extreme lines (|ML|>500) excluded — skipping.")
            continue

        raw_home = american_to_prob(home_ml)
        raw_away = american_to_prob(away_ml)
        dk_home_prob, dk_away_prob = remove_vig(raw_home, raw_away)

        results[(home_abbr, away_abbr)] = {
            "home_ml":   home_ml,
            "away_ml":   away_ml,
            "home_prob": round(dk_home_prob, 4),
            "away_prob": round(dk_away_prob, 4),
            "game_time": "",
        }
        print(f"  Recorded: {away_abbr}@{home_abbr}  "
              f"DK_home={dk_home_prob:.3f}  DK_away={dk_away_prob:.3f}")

    return results


# ---------------------------------------------------------------------------
# 4. Merge + compute edges
# ---------------------------------------------------------------------------

def build_game_odds(
    mlb_games:     list[dict],
    kalshi_prices: dict[tuple[str, str], dict],
    dk_odds:       dict[tuple[str, str], dict],
) -> list[GameOdds]:
    """
    Join MLB schedule, Kalshi prices, and DK odds into a list of GameOdds.

    Key logic:
    - Use (home_team, away_team) as join key.
    - Only populate Kalshi / DK fields when available; None means no data.
    - Compute edge_home / edge_away only when both sources are present.
    """
    games_out: list[GameOdds] = []

    for g in mlb_games:
        home = g["home_team"]
        away = g["away_team"]
        key  = (home, away)

        go = GameOdds(
            game_date = g["game_date"],
            home_team = home,
            away_team = away,
            game_time = g.get("game_time", ""),
            home_sp   = g.get("home_sp_name", "TBD"),
            away_sp   = g.get("away_sp_name", "TBD"),
        )

        # Kalshi
        k = kalshi_prices.get(key)
        if k:
            go.kalshi_home_prob   = k["kalshi_home_prob"]
            go.kalshi_away_prob   = k["kalshi_away_prob"]
            go.kalshi_home_ticker = k.get("home_ticker", "")
            go.kalshi_event_ticker = k.get("event_ticker", "")
            go.kalshi_source      = k.get("source", "")

        # DraftKings
        dk = dk_odds.get(key)
        if dk:
            go.dk_home_ml   = dk["home_ml"]
            go.dk_away_ml   = dk["away_ml"]
            go.dk_home_prob = dk["home_prob"]
            go.dk_away_prob = dk["away_prob"]
            # Prefer Odds-API game time if MLB API didn't return one
            if not go.game_time and dk.get("game_time"):
                go.game_time = dk["game_time"]

        # Edges
        if go.kalshi_home_prob is not None and go.dk_home_prob is not None:
            go.edge_home = round(go.dk_home_prob - go.kalshi_home_prob, 4)
            go.edge_away = round(go.dk_away_prob - go.kalshi_away_prob, 4)

        games_out.append(go)

    return games_out


# ---------------------------------------------------------------------------
# 5. Select qualifying bets
# ---------------------------------------------------------------------------

def select_arb_picks(
    games:          list[GameOdds],
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    kelly_mult:     float = DEFAULT_KELLY_FRACTION,
    bankroll:       float = DEFAULT_BANKROLL,
) -> list[ArbPick]:
    """
    For each game with both Kalshi and DK prices, select the side (if any)
    whose edge exceeds `edge_threshold`.

    If both sides qualify (unusual), pick the higher-edge side.

    Returns a list of ArbPick objects sorted by descending edge.
    """
    picks: list[ArbPick] = []

    for go in games:
        if go.edge_home is None:
            continue  # missing data

        best_side  = None
        best_edge  = edge_threshold  # must beat threshold to qualify

        if go.edge_home > best_edge:
            best_side = "home"
            best_edge = go.edge_home
        if go.edge_away is not None and go.edge_away > best_edge:
            best_side = "away"
            best_edge = go.edge_away

        if best_side is None:
            continue

        if best_side == "home":
            kalshi_p = go.kalshi_home_prob
            dk_p     = go.dk_home_prob
            dk_ml    = go.dk_home_ml
            bet_team = go.home_team
        else:
            kalshi_p = go.kalshi_away_prob
            dk_p     = go.dk_away_prob
            dk_ml    = go.dk_away_ml
            bet_team = go.away_team

        kf    = kelly_fraction(dk_p, kalshi_p, fraction=kelly_mult)
        stake = round(bankroll * kf, 2)
        ev    = round(ev_per_100(dk_p, kalshi_p), 2)

        # Format game time as "HH:MM ET" for display
        game_time_str = ""
        if go.game_time:
            try:
                dt_utc = datetime.fromisoformat(go.game_time.replace("Z", "+00:00"))
                # Convert to ET (UTC-4 during EDT, UTC-5 during EST)
                # Use fixed -4 offset (EDT) for MLB season; close enough for display
                et_offset = -4
                dt_et = dt_utc.replace(tzinfo=None)
                et_hour = (dt_utc.hour + et_offset) % 24
                game_time_str = f"{et_hour:02d}:{dt_utc.minute:02d} ET"
            except (ValueError, AttributeError):
                game_time_str = go.game_time

        picks.append(ArbPick(
            game_date    = go.game_date,
            matchup      = f"{go.away_team} @ {go.home_team}",
            game_time    = game_time_str,
            home_sp      = go.home_sp,
            away_sp      = go.away_sp,
            side         = best_side,
            bet_team     = bet_team,
            edge         = round(best_edge, 4),
            kalshi_price = round(kalshi_p, 4),
            dk_implied   = round(dk_p, 4),
            dk_ml        = dk_ml or 0,
            kelly_fraction = round(kf, 4),
            stake        = stake,
            ev_per_100   = ev,
            bankroll     = bankroll,
        ))

    picks.sort(key=lambda p: p.edge, reverse=True)
    return picks


# ---------------------------------------------------------------------------
# 6. Display
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt_ml(ml: int) -> str:
    return f"+{ml}" if ml > 0 else str(ml)


def print_game_table(games: list[GameOdds], verbose: bool = False):
    """Print a summary table of all games with available prices."""
    header = (
        f"  {'Matchup':<26s}  {'Kalshi Home':>11s}  "
        f"{'DK Home':>7s}  {'Edge Home':>9s}  {'Edge Away':>9s}  Kalshi src"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for go in games:
        matchup = f"{go.away_team} @ {go.home_team}"

        k_home  = _fmt_pct(go.kalshi_home_prob) if go.kalshi_home_prob else "   n/a"
        dk_home = _fmt_pct(go.dk_home_prob)     if go.dk_home_prob     else "   n/a"

        edge_home_str = "  n/a"
        edge_away_str = "  n/a"
        if go.edge_home is not None:
            edge_home_str = f"{go.edge_home * 100:+.1f}%"
            edge_away_str = f"{go.edge_away * 100:+.1f}%"

        src = go.kalshi_source or "-"

        print(
            f"  {matchup:<26s}  {k_home:>11s}  "
            f"{dk_home:>7s}  {edge_home_str:>9s}  {edge_away_str:>9s}  {src}"
        )

    # Legend
    if verbose:
        print()
        print("  Edge = DK_implied - Kalshi_price  (positive → DK thinks this side more likely)")
        print("  Kalshi src: ask = current offer price, last_trade, mid, fallback = previous_price")


def print_picks_table(picks: list[ArbPick], edge_threshold: float, bankroll: float):
    """Print the qualifying arb picks in a clean table."""
    if not picks:
        print(f"\n  No qualifying picks found at {edge_threshold*100:.0f}% edge threshold.")
        return

    print(f"\n  {'Matchup':<26s}  {'Bet':>5s}  {'Side':>5s}  "
          f"{'Edge':>6s}  {'DK ML':>7s}  {'Kal$':>6s}  "
          f"{'DK%':>6s}  {'½-Kelly':>8s}  {'Stake':>8s}  {'EV/100':>7s}")
    print("  " + "-" * 110)

    for p in picks:
        print(
            f"  {p.matchup:<26s}  {p.bet_team:>5s}  {p.side:>5s}  "
            f"{p.edge*100:>5.1f}%  {_fmt_ml(p.dk_ml):>7s}  {p.kalshi_price*100:>5.1f}¢  "
            f"{p.dk_implied*100:>5.1f}%  {p.kelly_fraction*100:>7.2f}%  "
            f"${p.stake:>7.2f}  ${p.ev_per_100:>6.2f}"
        )

    total_stake = sum(p.stake for p in picks)
    total_ev    = sum(p.ev_per_100 * p.stake / 100 for p in picks)
    print("  " + "-" * 110)
    print(f"  {len(picks)} pick(s)   Total stake: ${total_stake:.2f}  "
          f"Total EV: ${total_ev:.2f}  Bankroll: ${bankroll:.0f}")


def print_picks_detail(picks: list[ArbPick]):
    """Print verbose per-pick detail blocks."""
    for i, p in enumerate(picks, 1):
        print(f"\n  --- Pick #{i}: {p.matchup} ---")
        print(f"  Bet:          {p.bet_team} ({p.side})")
        print(f"  Game time:    {p.game_time or 'TBD'}")
        print(f"  Pitchers:     {p.away_sp} (away) vs {p.home_sp} (home)")
        print(f"  DK ML:        {_fmt_ml(p.dk_ml)}  →  DK implied: {_fmt_pct(p.dk_implied)}")
        print(f"  Kalshi price: {p.kalshi_price*100:.1f}¢  ({_fmt_pct(p.kalshi_price)})")
        print(f"  Edge:         {p.edge*100:.2f}%")
        print(f"  Half-Kelly:   {p.kelly_fraction*100:.2f}% of ${p.bankroll:.0f} = ${p.stake:.2f}")
        print(f"  EV per $100:  ${p.ev_per_100:.2f}")


# ---------------------------------------------------------------------------
# 7. Dry-run sample data
# ---------------------------------------------------------------------------

def make_dry_run_data(target_date: str) -> tuple[list[dict], dict, dict]:
    """
    Return (mlb_games, kalshi_prices, dk_odds) with hardcoded sample values
    that exercise the full pipeline without any live API calls.

    Edge scenarios covered:
      - LAD vs SF: LAD edge_home = 0.09 (qualifies at 7%)
      - NYY vs BOS: no qualifying edge
      - HOU vs TEX: TEX edge_away = 0.08 (qualifies)
      - ATL vs PHI: Kalshi data missing (no Kalshi market)
    """
    mlb_games = [
        {"game_pk": 1, "game_date": target_date, "status": "Preview",
         "game_time": "2025-07-15T23:10:00Z",
         "home_team": "LAD", "away_team": "SF",
         "home_team_name": "Los Angeles Dodgers", "away_team_name": "San Francisco Giants",
         "home_sp_name": "Yoshinobu Yamamoto", "away_sp_name": "Logan Webb"},
        {"game_pk": 2, "game_date": target_date, "status": "Preview",
         "game_time": "2025-07-15T23:05:00Z",
         "home_team": "BOS", "away_team": "NYY",
         "home_team_name": "Boston Red Sox", "away_team_name": "New York Yankees",
         "home_sp_name": "Tanner Houck", "away_sp_name": "Gerrit Cole"},
        {"game_pk": 3, "game_date": target_date, "status": "Preview",
         "game_time": "2025-07-15T20:10:00Z",
         "home_team": "TEX", "away_team": "HOU",
         "home_team_name": "Texas Rangers", "away_team_name": "Houston Astros",
         "home_sp_name": "Jon Gray", "away_sp_name": "Framber Valdez"},
        {"game_pk": 4, "game_date": target_date, "status": "Preview",
         "game_time": "2025-07-15T19:20:00Z",
         "home_team": "PHI", "away_team": "ATL",
         "home_team_name": "Philadelphia Phillies", "away_team_name": "Atlanta Braves",
         "home_sp_name": "Zack Wheeler", "away_sp_name": "Spencer Strider"},
    ]

    kalshi_prices = {
        ("LAD", "SF"):  {"kalshi_home_prob": 0.60, "kalshi_away_prob": 0.40,
                         "home_ticker": "KXMLBGAME-25JUL15SFLAB-LAD",
                         "event_ticker": "KXMLBGAME-25JUL15SFLAD",
                         "source": "ask"},
        ("BOS", "NYY"): {"kalshi_home_prob": 0.46, "kalshi_away_prob": 0.54,
                         "home_ticker": "KXMLBGAME-25JUL15NYYBOS-BOS",
                         "event_ticker": "KXMLBGAME-25JUL15NYYBOS",
                         "source": "ask"},
        ("TEX", "HOU"): {"kalshi_home_prob": 0.44, "kalshi_away_prob": 0.56,
                         "home_ticker": "KXMLBGAME-25JUL15HOUTEX-TEX",
                         "event_ticker": "KXMLBGAME-25JUL15HOUTEX",
                         "source": "mid"},
        # ATL @ PHI intentionally missing from Kalshi
    }

    dk_odds = {
        # LAD heavily favored; DK home_prob = 0.69 vs Kalshi 0.60 → +9% edge home
        ("LAD", "SF"):  {"home_ml": -220, "away_ml": +185,
                         "home_prob": 0.690, "away_prob": 0.310, "game_time": ""},
        # NYY vs BOS: no meaningful edge
        ("BOS", "NYY"): {"home_ml": +120, "away_ml": -140,
                         "home_prob": 0.456, "away_prob": 0.544, "game_time": ""},
        # HOU @ TEX: DK_away_prob = 0.64 vs Kalshi 0.56 → +8% edge away (HOU)
        ("TEX", "HOU"): {"home_ml": +145, "away_ml": -170,
                         "home_prob": 0.407, "away_prob": 0.593, "game_time": ""},
        # PHI @ ATL: DK has prices but Kalshi market absent
        ("PHI", "ATL"): {"home_ml": -165, "away_ml": +140,
                         "home_prob": 0.622, "away_prob": 0.378, "game_time": ""},
    }

    return mlb_games, kalshi_prices, dk_odds


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------

def run(
    target_date:    str,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    kelly_mult:     float = DEFAULT_KELLY_FRACTION,
    bankroll:       float = DEFAULT_BANKROLL,
    manual_dk:      bool  = False,
    dry_run:        bool  = False,
    as_json:        bool  = False,
    verbose:        bool  = False,
) -> list[ArbPick]:
    """
    Full pipeline: fetch data → merge → compute edges → select picks → display.

    Returns the list of ArbPick objects (also printed to stdout unless as_json).
    """
    separator = "=" * 72

    if not as_json:
        print(separator)
        print(f"  DK vs Kalshi Arb Scanner  |  {target_date}  |  "
              f"bankroll ${bankroll:.0f}  |  edge {edge_threshold*100:.0f}%")
        print(separator)

    # ── Data acquisition ─────────────────────────────────────────────────────

    if dry_run:
        if not as_json:
            print("\n[DRY RUN] Using sample data — no live API calls.")
        mlb_games, kalshi_prices, dk_odds = make_dry_run_data(target_date)

    else:
        with httpx.Client(timeout=20.0) as client:
            # MLB schedule
            if not as_json:
                print(f"\nFetching MLB schedule for {target_date}...")
            try:
                mlb_games = fetch_todays_games(client, target_date)
                if not as_json:
                    print(f"  Found {len(mlb_games)} game(s)")
            except Exception as exc:
                print(f"  ERROR fetching MLB schedule: {exc}")
                return []

            if not mlb_games:
                print("  No games scheduled — nothing to scan.")
                return []

            # Kalshi prices
            if not as_json:
                print(f"\nFetching Kalshi prices...")
            try:
                kalshi_prices = fetch_kalshi_prices(client, target_date, verbose=verbose)
                if not as_json:
                    print(f"  Got prices for {len(kalshi_prices)} game(s)")
            except Exception as exc:
                print(f"  ERROR fetching Kalshi prices: {exc}")
                kalshi_prices = {}

        # DK odds (outside httpx.Client so manual input works in terminal)
        if manual_dk:
            if not as_json:
                print("\nManual DK odds entry:")
            dk_odds = fetch_dk_odds_manual()
        else:
            odds_api_key = os.environ.get("ODDS_API_KEY", "")
            if not odds_api_key:
                print(
                    "\n  WARNING: ODDS_API_KEY not set. "
                    "Set it with: export ODDS_API_KEY=your_key\n"
                    "  Get a free key at https://the-odds-api.com\n"
                    "  Re-run with --manual-dk to enter DK lines manually.\n"
                    "  Continuing with no DK data — no picks will be generated.\n"
                )
                dk_odds = {}
            else:
                if not as_json:
                    print("\nFetching DraftKings odds via Odds-API...")
                with httpx.Client(timeout=15.0) as client:
                    dk_odds = fetch_dk_odds_via_api(
                        odds_api_key, client, verbose=verbose
                    )
                if not as_json:
                    print(f"  Got DK lines for {len(dk_odds)} game(s)")

    # ── Merge & compute edges ────────────────────────────────────────────────

    games = build_game_odds(mlb_games, kalshi_prices, dk_odds)

    # ── Game price overview ──────────────────────────────────────────────────

    if not as_json:
        n_both   = sum(1 for g in games if g.kalshi_home_prob and g.dk_home_prob)
        n_kalshi = sum(1 for g in games if g.kalshi_home_prob)
        n_dk     = sum(1 for g in games if g.dk_home_prob)

        print(f"\n{'─'*72}")
        print(f"  All Games  ({n_kalshi} Kalshi  |  {n_dk} DK  |  {n_both} both)")
        print(f"{'─'*72}")
        print_game_table(games, verbose=verbose)

    # ── Select qualifying picks ──────────────────────────────────────────────

    picks = select_arb_picks(games, edge_threshold, kelly_mult, bankroll)

    # ── Output ───────────────────────────────────────────────────────────────

    if as_json:
        payload = {
            "scan_date":      target_date,
            "run_at":         datetime.now(timezone.utc).isoformat(),
            "edge_threshold": edge_threshold,
            "kelly_mult":     kelly_mult,
            "bankroll":       bankroll,
            "picks":          [asdict(p) for p in picks],
            "n_games":        len(games),
            "n_with_both":    sum(1 for g in games if g.kalshi_home_prob and g.dk_home_prob),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"\n{'─'*72}")
        print(f"  Qualifying Picks  (edge ≥ {edge_threshold*100:.0f}%,  ½-Kelly,  bankroll ${bankroll:.0f})")
        print(f"{'─'*72}")
        print_picks_table(picks, edge_threshold, bankroll)

        if picks and verbose:
            print(f"\n{'─'*72}")
            print("  Pick Detail")
            print(f"{'─'*72}")
            print_picks_detail(picks)

        print()

    return picks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DK vs Kalshi moneyline arb scanner — live daily pick engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--date", type=str, default=None,
        help="Target date YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--bankroll", type=float, default=DEFAULT_BANKROLL,
        help=f"Bankroll in dollars for sizing (default: {DEFAULT_BANKROLL:.0f})",
    )
    p.add_argument(
        "--edge-threshold", type=float, default=DEFAULT_EDGE_THRESHOLD,
        help=f"Minimum edge to qualify (default: {DEFAULT_EDGE_THRESHOLD})",
    )
    p.add_argument(
        "--kelly-fraction", type=float, default=DEFAULT_KELLY_FRACTION,
        help=f"Kelly multiplier — 0.5=half-Kelly (default: {DEFAULT_KELLY_FRACTION})",
    )
    p.add_argument(
        "--manual-dk", action="store_true",
        help="Enter DK moneylines manually instead of fetching via Odds-API",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Use hardcoded sample data (no live API calls)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Output picks as machine-readable JSON",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print extra detail (game table legend, per-pick blocks)",
    )
    return p


def main():
    args    = _build_parser().parse_args()
    today   = date.today().strftime("%Y-%m-%d")
    target  = args.date or today

    picks = run(
        target_date    = target,
        edge_threshold = args.edge_threshold,
        kelly_mult     = args.kelly_fraction,
        bankroll       = args.bankroll,
        manual_dk      = args.manual_dk,
        dry_run        = args.dry_run,
        as_json        = args.json,
        verbose        = args.verbose,
    )

    # Exit 0 always — non-zero only on unhandled exceptions
    sys.exit(0)


if __name__ == "__main__":
    main()
