#!/usr/bin/env python3
"""Plot model vs Kalshi win probability trace for a single game."""
import argparse
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multi_output_matchup_model import load_multi_output_models
from simulate import (
    GameState,
    SimConfig,
    load_sim_artifacts,
    load_simulation_context,
    monte_carlo_win_prob,
)
from backtest_vs_kalshi import (
    fetch_play_by_play,
    extract_half_inning_states,
    fetch_full_candle_series,
    match_candle_to_timestamp,
    load_backtest_data,
)
from predict import fetch_lineup
from utils import DATA_DIR

AUDIT_DIR = DATA_DIR / "audit"


def main():
    parser = argparse.ArgumentParser(description="Plot game win probability trace")
    parser.add_argument("--game-pk", type=int, required=True)
    parser.add_argument("--kalshi-ticker", type=str, default=None,
                        help="Kalshi market ticker for home team win (auto-detected if omitted)")
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    game_pk = args.game_pk

    # ── Load models and data ──────────────────────────────────────────────
    print("Loading multi-output models...")
    try:
        mo_models = load_multi_output_models(args.season - 1)
    except Exception:
        mo_models = None

    print("Loading simulation artifacts...")
    base_rates, transition_matrix = load_sim_artifacts()

    print("Loading backtest data...")
    shared = load_backtest_data(args.season)
    if shared is None:
        print("Cannot load xRV data")
        return
    idx, matchup_models, lineups = shared

    config = SimConfig(n_sims=args.n_sims, random_seed=42)

    # ── Fetch play-by-play ────────────────────────────────────────────────
    print(f"Fetching play-by-play for game {game_pk}...")
    client = httpx.Client(timeout=30.0)
    pbp = fetch_play_by_play(client, game_pk)
    if pbp is None:
        print("Could not fetch play-by-play data")
        return

    game_data = pbp.get("gameData", {})
    teams = game_data.get("teams", {})
    home_abbr = teams.get("home", {}).get("abbreviation", "HOME")
    away_abbr = teams.get("away", {}).get("abbreviation", "AWAY")
    game_date = game_data.get("datetime", {}).get("officialDate", "")
    home_name = teams.get("home", {}).get("teamName", home_abbr)
    away_name = teams.get("away", {}).get("teamName", away_abbr)

    print(f"  {away_abbr} @ {home_abbr} on {game_date}")

    # Get game info dict for load_simulation_context
    home_sp_id = None
    away_sp_id = None
    players = game_data.get("players", {})
    boxscore = pbp.get("liveData", {}).get("boxscore", {})

    # Try to get SP from boxscore pitchers
    for side, var in [("home", "home_sp_id"), ("away", "away_sp_id")]:
        pitchers = (
            boxscore.get("teams", {}).get(side, {}).get("pitchers", [])
        )
        if pitchers:
            if var == "home_sp_id":
                home_sp_id = pitchers[0]
            else:
                away_sp_id = pitchers[0]

    game_info = {
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "home_sp_id": home_sp_id or 0,
        "away_sp_id": away_sp_id or 0,
    }

    # ── Get lineup ────────────────────────────────────────────────────────
    lu = lineups.get(game_pk)
    if lu is None:
        print("  Fetching lineup from API...")
        lu = fetch_lineup(client, game_pk)
    if not lu or not lu.get("home") or not lu.get("away"):
        print("  Could not get lineup")
        return

    # ── Build simulation context ──────────────────────────────────────────
    print("Building simulation context...")
    try:
        home_ctx, away_ctx = load_simulation_context(
            game_info, str(game_date), lu, idx,
            matchup_models or {}, base_rates,
            mo_models=mo_models, config=config,
        )
    except Exception as e:
        print(f"  Error building context: {e}")
        return

    # ── Extract half-inning states ────────────────────────────────────────
    snapshots = extract_half_inning_states(pbp)
    print(f"  {len(snapshots)} half-inning state points")

    if len(snapshots) < 2:
        print("  Not enough state points")
        return

    # ── Run simulator at each state ───────────────────────────────────────
    print(f"Running {args.n_sims} sims at each state point...")
    sim_points = []  # (timestamp, home_wp, label)

    for i, snap in enumerate(snapshots):
        state = snap.game_state
        result = monte_carlo_win_prob(
            home_ctx, away_ctx, state, transition_matrix, config,
        )
        home_wp = result["home_wp"]
        label = snap.inning_half
        sim_points.append((snap.timestamp, home_wp, label))

        if (i + 1) % 5 == 0 or i == len(snapshots) - 1:
            score_str = f"{away_abbr} {state.away_score} - {home_abbr} {state.home_score}"
            print(f"    {label:>8}: {score_str}  →  {home_abbr} WP = {home_wp:.1%}")

    # ── Fetch Kalshi candles ──────────────────────────────────────────────
    kalshi_ticker = args.kalshi_ticker
    kalshi_points = []

    if kalshi_ticker is None:
        # Try to auto-detect
        kalshi_client = httpx.Client(timeout=30.0)
        resp = kalshi_client.get(
            "https://api.elections.kalshi.com/trade-api/v2/events",
            params={"series_ticker": "KXMLBGAME", "status": "open", "limit": 100},
        )
        if resp.status_code == 200:
            for e in resp.json().get("events", []):
                t = e.get("event_ticker", "")
                if home_abbr in t and away_abbr in t:
                    kalshi_ticker = t + f"-{home_abbr}"
                    print(f"  Auto-detected Kalshi ticker: {kalshi_ticker}")
                    break
        # Also try closed events
        if kalshi_ticker is None:
            resp = kalshi_client.get(
                "https://api.elections.kalshi.com/trade-api/v2/events",
                params={"series_ticker": "KXMLBGAME", "status": "closed", "limit": 100},
            )
            if resp.status_code == 200:
                for e in resp.json().get("events", []):
                    t = e.get("event_ticker", "")
                    if home_abbr in t and away_abbr in t and game_date.replace("-", "")[-4:] in t:
                        kalshi_ticker = t + f"-{home_abbr}"
                        print(f"  Auto-detected Kalshi ticker: {kalshi_ticker}")
                        break

    if kalshi_ticker:
        print(f"Fetching Kalshi candles for {kalshi_ticker}...")
        start_ts = int(sim_points[0][0]) - 3600
        end_ts = int(sim_points[-1][0]) + 3600
        candle_series = fetch_full_candle_series(client, kalshi_ticker, start_ts, end_ts)
        print(f"  {len(candle_series)} candles fetched")

        # Match candles to each sim timestamp
        for ts, home_wp, label in sim_points:
            price = match_candle_to_timestamp(candle_series, ts)
            if price is not None:
                kalshi_points.append((ts, price, label))
    else:
        print("  Could not detect Kalshi ticker — plotting model only")

    # ── Plot ──────────────────────────────────────────────────────────────
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(14, 7))

    # Convert timestamps to datetime
    sim_times = [datetime.fromtimestamp(t, tz=timezone.utc) for t, _, _ in sim_points]
    sim_wps = [wp for _, wp, _ in sim_points]

    ax.plot(sim_times, sim_wps, "o-", color="#2563EB", linewidth=2.5,
            markersize=6, label="Model", zorder=5)

    if kalshi_points:
        k_times = [datetime.fromtimestamp(t, tz=timezone.utc) for t, _, _ in kalshi_points]
        k_wps = [wp for _, wp, _ in kalshi_points]
        ax.plot(k_times, k_wps, "s--", color="#DC2626", linewidth=2,
                markersize=5, label="Kalshi", zorder=4)

    # 50% line
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    # Annotate score changes
    prev_score = (0, 0)
    for ts, wp, label in sim_points:
        snap = next(s for s in snapshots if abs(s.timestamp - ts) < 1)
        curr_score = (snap.game_state.away_score, snap.game_state.home_score)
        if curr_score != prev_score:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            score_text = f"{curr_score[0]}-{curr_score[1]}"
            ax.annotate(score_text, (dt, wp),
                       textcoords="offset points", xytext=(0, 12),
                       fontsize=8, ha="center", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
            prev_score = curr_score

    # Add inning labels on x-axis
    for ts, wp, label in sim_points:
        if "Top" in label:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            inning_num = label.split()[-1]
            ax.annotate(inning_num, (dt, -0.02),
                       fontsize=8, ha="center", color="gray",
                       annotation_clip=False)

    ax.set_ylabel(f"{home_name} Win Probability", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p", tz=timezone.utc))
    ax.tick_params(axis="x", rotation=30)

    ax.set_title(
        f"{away_abbr} @ {home_abbr} — {game_date}\nWin Probability Trace",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = AUDIT_DIR / f"game_trace_{away_abbr}_{home_abbr}_{game_date.replace('-', '')}.png"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
