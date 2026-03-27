#!/usr/bin/env python3
"""
Scrape preseason projection data from FanGraphs.

Fetches:
  1. Projected team standings (win totals)
  2. Pitcher-level Steamer projections (ERA, FIP, K/9, BB/9, IP, WAR)

Usage:
    python src/scrape_projections.py --seasons 2018 2019 2020 2021 2022 2023 2024 2025
"""

import argparse
import logging
import time
from pathlib import Path

import httpx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FanGraphs abbreviation -> canonical abbreviation used in games parquet
# ---------------------------------------------------------------------------
TEAM_MAP: dict[str, str] = {
    # Teams whose FanGraphs abbrev differs from our canonical form
    "OAK": "ATH",
    "ARI": "AZ",
    "CHW": "CWS",
    "WAS": "WSH",
    "WSN": "WSH",
    "TBR": "TB",
    "SDP": "SD",
    "SFG": "SF",
    "KCR": "KC",
    # Teams that are already canonical (identity mappings for safety)
    "ATH": "ATH",
    "AZ": "AZ",
    "CWS": "CWS",
    "WSH": "WSH",
    "TB": "TB",
    "SD": "SD",
    "SF": "SF",
    "KC": "KC",
    "LAA": "LAA",
    "LAD": "LAD",
    "NYM": "NYM",
    "NYY": "NYY",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "STL": "STL",
    "TEX": "TEX",
    "TOR": "TOR",
}

DATA_DIR = Path("data/projections")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.fangraphs.com/",
}

REQUEST_TIMEOUT = 30.0
SLEEP_SECONDS = 2


def _normalize_team(abbr: str) -> str:
    """Map a FanGraphs team abbreviation to our canonical form."""
    if abbr is None:
        return ""
    abbr = abbr.strip().upper()
    return TEAM_MAP.get(abbr, abbr)


def _get_json(client: httpx.Client, url: str) -> dict | list | None:
    """GET *url* and return parsed JSON, or None on any failure."""
    try:
        resp = client.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        log.warning("HTTP %s for %s", exc.response.status_code, url)
    except httpx.RequestError as exc:
        log.warning("Request error for %s: %s", url, exc)
    except Exception as exc:
        log.warning("Unexpected error fetching %s: %s", url, exc)
    return None


# ── Team projections ──────────────────────────────────────────────────────


_TEAM_STANDINGS_URLS = [
    "https://www.fangraphs.com/api/depth-charts/standings/{year}",
    "https://www.fangraphs.com/api/projections/rest-of-season/standings?season={year}",
    "https://www.fangraphs.com/api/depth-charts/standings?season={year}",
]


def _scrape_team_projections(client: httpx.Client, year: int) -> pd.DataFrame | None:
    """Try multiple FanGraphs endpoints for projected standings."""
    for url_template in _TEAM_STANDINGS_URLS:
        url = url_template.format(year=year)
        log.info("Trying team standings URL: %s", url)
        data = _get_json(client, url)
        if data is None:
            time.sleep(SLEEP_SECONDS)
            continue

        # The response can be a list of team dicts or a dict with a nested list.
        teams: list[dict] = []
        if isinstance(data, list):
            teams = data
        elif isinstance(data, dict):
            # Look for a list inside common keys
            for key in ("teams", "standings", "data", "Teams", "Standings"):
                if key in data and isinstance(data[key], list):
                    teams = data[key]
                    break
            if not teams:
                # Maybe the dict values are division lists
                for v in data.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        teams.extend(v)

        if not teams:
            log.warning("No team records parsed from %s", url)
            time.sleep(SLEEP_SECONDS)
            continue

        rows = []
        for t in teams:
            # Try various key names FanGraphs has used
            abbr = (
                t.get("abbr")
                or t.get("Abbr")
                or t.get("team")
                or t.get("Team")
                or t.get("teamAbbr")
                or t.get("ShortName")
                or ""
            )
            wins = (
                t.get("w")
                or t.get("W")
                or t.get("wins")
                or t.get("Wins")
                or t.get("projW")
                or t.get("ProjW")
            )
            losses = (
                t.get("l")
                or t.get("L")
                or t.get("losses")
                or t.get("Losses")
                or t.get("projL")
                or t.get("ProjL")
            )
            if abbr and wins is not None and losses is not None:
                w = float(wins)
                lo = float(losses)
                wpct = w / (w + lo) if (w + lo) > 0 else 0.0
                rows.append(
                    {
                        "team_abbr": _normalize_team(str(abbr)),
                        "projected_wins": round(w, 1),
                        "projected_losses": round(lo, 1),
                        "projected_wpct": round(wpct, 4),
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            log.info(
                "Fetched %d team projections for %d from %s", len(df), year, url
            )
            return df

        log.warning("Could not extract win/loss from %s", url)
        time.sleep(SLEEP_SECONDS)

    log.warning("All team-standings endpoints failed for %d", year)
    return None


# ── Pitcher projections ───────────────────────────────────────────────────

_PITCHER_PROJECTION_URLS = [
    (
        "https://www.fangraphs.com/api/projections"
        "?type=steamer&stats=pit&pos=all&team=0&players=0&season={year}"
    ),
    (
        "https://www.fangraphs.com/api/projections"
        "?type=steamer&stats=pit&pos=all&team=0&players=0&lg=all&season={year}"
    ),
    (
        "https://www.fangraphs.com/api/projections"
        "?type=steamerr&stats=pit&pos=all&team=0&players=0&season={year}"
    ),
    (
        "https://www.fangraphs.com/api/projections"
        "?type=zips&stats=pit&pos=all&team=0&players=0&season={year}"
    ),
]


def _extract_pitcher_rows(data: list[dict]) -> list[dict]:
    """Pull the columns we care about from a list of pitcher dicts."""
    rows = []
    for p in data:
        fg_id = p.get("playerid") or p.get("PlayerId") or p.get("playerIds") or ""
        name = p.get("PlayerName") or p.get("Name") or p.get("playerName") or ""
        team = (
            p.get("Team")
            or p.get("team")
            or p.get("TeamAbbr")
            or p.get("teamAbbr")
            or ""
        )
        era = p.get("ERA") or p.get("era")
        fip = p.get("FIP") or p.get("fip")
        k9 = p.get("K/9") or p.get("K9") or p.get("k9") or p.get("SO9")
        bb9 = p.get("BB/9") or p.get("BB9") or p.get("bb9")
        ip = p.get("IP") or p.get("ip") or p.get("Inn")
        war = p.get("WAR") or p.get("war") or p.get("fWAR")

        if not name:
            continue

        rows.append(
            {
                "fg_id": str(fg_id),
                "name": str(name),
                "team": _normalize_team(str(team)) if team else "",
                "projected_era": _safe_float(era),
                "projected_fip": _safe_float(fip),
                "projected_k9": _safe_float(k9),
                "projected_bb9": _safe_float(bb9),
                "projected_ip": _safe_float(ip),
                "projected_war": _safe_float(war),
            }
        )
    return rows


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return round(float(val), 3)
    except (ValueError, TypeError):
        return None


def _scrape_pitcher_projections(
    client: httpx.Client, year: int
) -> pd.DataFrame | None:
    """Fetch Steamer pitcher projections from FanGraphs API."""
    for url_template in _PITCHER_PROJECTION_URLS:
        url = url_template.format(year=year)
        log.info("Trying pitcher projections URL: %s", url)
        data = _get_json(client, url)
        if data is None:
            time.sleep(SLEEP_SECONDS)
            continue

        pitchers: list[dict] = []
        if isinstance(data, list):
            pitchers = data
        elif isinstance(data, dict):
            for key in ("data", "players", "Pitchers", "pitchers", "Data"):
                if key in data and isinstance(data[key], list):
                    pitchers = data[key]
                    break

        if not pitchers:
            log.warning("No pitcher records from %s", url)
            time.sleep(SLEEP_SECONDS)
            continue

        rows = _extract_pitcher_rows(pitchers)
        if rows:
            df = pd.DataFrame(rows)
            log.info(
                "Fetched %d pitcher projections for %d from %s",
                len(df),
                year,
                url,
            )
            return df

        log.warning("Could not extract pitcher stats from %s", url)
        time.sleep(SLEEP_SECONDS)

    # ── Fallback: pybaseball ──────────────────────────────────────────────
    log.info("API endpoints exhausted for %d; trying pybaseball fallback", year)
    try:
        from pybaseball import fg_pitching_data  # type: ignore[import-untyped]

        raw = fg_pitching_data(year, qual=0)
        if raw is not None and not raw.empty:
            rows = []
            for _, p in raw.iterrows():
                rows.append(
                    {
                        "fg_id": str(p.get("IDfg", "")),
                        "name": str(p.get("Name", "")),
                        "team": _normalize_team(str(p.get("Team", ""))),
                        "projected_era": _safe_float(p.get("ERA")),
                        "projected_fip": _safe_float(p.get("FIP")),
                        "projected_k9": _safe_float(p.get("K/9")),
                        "projected_bb9": _safe_float(p.get("BB/9")),
                        "projected_ip": _safe_float(p.get("IP")),
                        "projected_war": _safe_float(p.get("WAR")),
                    }
                )
            if rows:
                df = pd.DataFrame(rows)
                log.info(
                    "Fetched %d pitcher rows for %d via pybaseball", len(df), year
                )
                return df
    except ImportError:
        log.warning("pybaseball not installed; cannot use fallback")
    except Exception as exc:
        log.warning("pybaseball fallback failed for %d: %s", year, exc)

    log.warning("All pitcher-projection sources failed for %d", year)
    return None


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape FanGraphs preseason projections"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(range(2018, 2026)),
        help="Seasons to fetch (default: 2018-2025)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output file already exists",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True) as client:
        for year in sorted(args.seasons):
            # ── Team projections ──────────────────────────────────────
            team_path = DATA_DIR / f"team_projections_{year}.parquet"
            if team_path.exists() and not args.force:
                log.info("Skipping team projections %d (cached: %s)", year, team_path)
            else:
                log.info("Fetching team projections for %d ...", year)
                team_df = _scrape_team_projections(client, year)
                if team_df is not None and not team_df.empty:
                    team_df.to_parquet(team_path, index=False)
                    log.info("Saved %s (%d rows)", team_path, len(team_df))
                else:
                    log.warning("No team projections saved for %d", year)

            time.sleep(SLEEP_SECONDS)

            # ── Pitcher projections ───────────────────────────────────
            pitcher_path = DATA_DIR / f"pitcher_projections_{year}.parquet"
            if pitcher_path.exists() and not args.force:
                log.info(
                    "Skipping pitcher projections %d (cached: %s)", year, pitcher_path
                )
            else:
                log.info("Fetching pitcher projections for %d ...", year)
                pitcher_df = _scrape_pitcher_projections(client, year)
                if pitcher_df is not None and not pitcher_df.empty:
                    pitcher_df.to_parquet(pitcher_path, index=False)
                    log.info("Saved %s (%d rows)", pitcher_path, len(pitcher_df))
                else:
                    log.warning("No pitcher projections saved for %d", year)

            time.sleep(SLEEP_SECONDS)

    log.info("Done.")


if __name__ == "__main__":
    main()
