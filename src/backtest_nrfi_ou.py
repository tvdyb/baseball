#!/usr/bin/env python3
"""
Backtest MC simulator on NRFI (No Run First Inning) and Over/Under predictions.

Evaluates how well the MC sim predicts:
  1. NRFI — P(no runs scored in the 1st inning)
  2. Over/Under — P(total runs > line)

Usage:
    python src/backtest_nrfi_ou.py --season 2025 --n-sims 5000
    python src/backtest_nrfi_ou.py --season 2025 --n-sims 5000 --max-games 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_vs_kalshi import load_backtest_data, load_sim_artifacts
from predict import fetch_lineup
from simulate import (
    GameState,
    SimConfig,
    load_simulation_context,
    monte_carlo_win_prob,
)
from utils import DATA_DIR

import httpx


def log(msg: str) -> None:
    print(msg, flush=True)


def compute_actual_first_inning(season: int) -> pd.DataFrame:
    """Compute actual first-inning runs per game from Statcast pitch data.

    Returns DataFrame with columns: game_pk, away_1st_runs, home_1st_runs, nrfi
    """
    sc_path = DATA_DIR / "statcast" / f"statcast_{season}.parquet"
    if not sc_path.exists():
        log(f"  Statcast data not found: {sc_path}")
        return pd.DataFrame()

    sc = pd.read_parquet(
        sc_path,
        columns=["game_pk", "inning", "inning_topbot", "post_home_score", "post_away_score"],
    )
    first = sc[sc["inning"] == 1]

    # Max post_away_score after top of 1st = away runs in 1st inning
    top1 = first[first["inning_topbot"] == "Top"].groupby("game_pk")["post_away_score"].max()
    # Max post_home_score after bot of 1st = home runs in 1st inning
    bot1 = first[first["inning_topbot"] == "Bot"].groupby("game_pk")["post_home_score"].max()

    both = pd.DataFrame({"away_1st_runs": top1, "home_1st_runs": bot1}).dropna()
    both["nrfi"] = ((both["away_1st_runs"] == 0) & (both["home_1st_runs"] == 0)).astype(int)
    both = both.reset_index()
    return both


def run_backtest(
    season: int,
    n_sims: int = 5000,
    max_games: int | None = None,
) -> dict:
    """Run NRFI + O/U backtest on a season's games."""

    log(f"=== NRFI + O/U Backtest — {season} season ===")
    log(f"  MC sims per game: {n_sims}")

    # Load actual first-inning data
    actual_1st = compute_actual_first_inning(season)
    if actual_1st.empty:
        return {}
    log(f"  Actual first-inning data: {len(actual_1st)} games")
    log(f"  Actual NRFI rate: {actual_1st['nrfi'].mean():.3f}")

    # Load game results
    games_path = DATA_DIR / "games" / f"games_{season}.parquet"
    games_df = pd.read_parquet(games_path)
    games_df["game_date"] = games_df["game_date"].astype(str)
    if "home_team_abbr" in games_df.columns and "home_team" not in games_df.columns:
        games_df["home_team"] = games_df["home_team_abbr"]
        games_df["away_team"] = games_df["away_team_abbr"]
    games_df = games_df[
        games_df["home_win"].notna()
        & games_df["home_sp_id"].notna()
        & games_df["away_sp_id"].notna()
    ].copy()
    games_df["actual_total"] = games_df["home_score"] + games_df["away_score"]

    # Merge with first-inning data
    games_df = games_df.merge(actual_1st, on="game_pk", how="inner")
    log(f"  Games with first-inning + result data: {len(games_df)}")

    if max_games:
        games_df = games_df.head(max_games)
        log(f"  Limited to {max_games} games")

    # Load sim artifacts
    base_rates, transition_matrix = load_sim_artifacts()
    loaded = load_backtest_data(season)
    if loaded is None:
        log("  Failed to load backtest data")
        return {}
    idx, matchup_models, lineups = loaded

    # Load park factors and weather
    from feature_engineering import compute_park_factors
    park_factors = compute_park_factors(games_df, season)
    log(f"  Park factors: {len(park_factors)} venues (range {min(park_factors.values()):.2f}-{max(park_factors.values()):.2f})")

    weather_map = {}
    weather_path = DATA_DIR / "weather" / f"weather_{season}.parquet"
    if weather_path.exists():
        wdf = pd.read_parquet(weather_path)
        for _, wr in wdf.iterrows():
            wind_dir = str(wr.get("wind_dir", "")).lower()
            weather_map[int(wr["game_pk"])] = {
                "temperature": wr.get("temperature"),
                "wind_speed": wr.get("wind_speed"),
                "wind_out": int("out" in wind_dir),
                "wind_in": int("in" in wind_dir),
                "is_dome": wr.get("is_dome", 0),
            }
        log(f"  Weather data: {len(weather_map)} games")

    config = SimConfig(n_sims=n_sims, random_seed=42)
    client = httpx.Client(timeout=30.0)

    rows = []
    n_skip = 0
    n_total = len(games_df)

    for i, (_, row) in enumerate(games_df.iterrows()):
        game = row.to_dict()
        gpk = int(game["game_pk"])
        lu = lineups.get(gpk)

        if lu is None or not lu.get("home") or not lu.get("away"):
            lu = fetch_lineup(client, gpk)
            if lu.get("home") and lu.get("away"):
                lineups[gpk] = lu
                time.sleep(0.1)
            else:
                n_skip += 1
                continue

        try:
            home_ctx, away_ctx = load_simulation_context(
                game, str(game["game_date"]), lu, idx,
                matchup_models or {}, base_rates,
                config=config,
            )
            # Park factor for this game's venue
            pf = park_factors.get(game["home_team"], 1.0)
            # Weather for this game
            wx = weather_map.get(gpk)
            # Skip weather adjustments for dome stadiums
            if wx and wx.get("is_dome"):
                wx = None

            result = monte_carlo_win_prob(
                home_ctx, away_ctx, GameState(),
                transition_matrix, config,
                park_factor=pf, weather=wx,
            )

            rows.append({
                "game_pk": gpk,
                "game_date": game["game_date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                # NRFI
                "sim_nrfi_prob": result.get("nrfi_prob", np.nan),
                "actual_nrfi": int(game["nrfi"]),
                # O/U
                "sim_total_mean": result["total_runs_mean"],
                "actual_total": int(game["actual_total"]),
                # Win prob
                "sim_home_wp": result["home_wp"],
                "actual_home_win": int(game["home_win"]),
            })
        except Exception as e:
            n_skip += 1
            if n_skip <= 5:
                import traceback
                log(f"    ERROR game {gpk}: {e}")
                traceback.print_exc()
            continue

        if (i + 1) % 50 == 0 or i + 1 == n_total:
            log(f"  Progress: {i + 1}/{n_total} ({len(rows)} done, {n_skip} skipped)")

    client.close()
    df = pd.DataFrame(rows)
    if df.empty:
        log("  No results")
        return {}

    log(f"\n{'='*60}")
    log(f"RESULTS: {len(df)} games simulated ({n_skip} skipped)")
    log(f"{'='*60}")

    # ── NRFI Analysis ──────────────────────────────────────
    nrfi_valid = df.dropna(subset=["sim_nrfi_prob"])
    if len(nrfi_valid) > 0:
        log(f"\n── NRFI Analysis ({len(nrfi_valid)} games) ──")
        log(f"  Actual NRFI rate: {nrfi_valid['actual_nrfi'].mean():.3f}")
        log(f"  Mean sim P(NRFI): {nrfi_valid['sim_nrfi_prob'].mean():.3f}")

        # Brier score
        brier = ((nrfi_valid["sim_nrfi_prob"] - nrfi_valid["actual_nrfi"]) ** 2).mean()
        # Baseline Brier (predict actual rate for every game)
        base_rate = nrfi_valid["actual_nrfi"].mean()
        brier_baseline = ((base_rate - nrfi_valid["actual_nrfi"]) ** 2).mean()
        log(f"  Brier score (MC sim):  {brier:.4f}")
        log(f"  Brier score (baseline): {brier_baseline:.4f}")
        log(f"  Brier skill score:     {1 - brier / brier_baseline:.4f}")

        # Calibration bins
        log(f"\n  Calibration (P(NRFI) bins):")
        log(f"  {'Bin':>12}  {'N':>5}  {'Pred':>6}  {'Actual':>6}")
        bins = [0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0]
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (nrfi_valid["sim_nrfi_prob"] >= lo) & (nrfi_valid["sim_nrfi_prob"] < hi)
            subset = nrfi_valid[mask]
            if len(subset) >= 5:
                log(f"  [{lo:.2f},{hi:.2f})  {len(subset):>5}  {subset['sim_nrfi_prob'].mean():.3f}  {subset['actual_nrfi'].mean():.3f}")

        # If we bet NRFI when sim says P(NRFI) > 0.55, how do we do?
        log(f"\n  NRFI Betting (flat $100):")
        for thresh in [0.52, 0.55, 0.58, 0.60]:
            bets = nrfi_valid[nrfi_valid["sim_nrfi_prob"] >= thresh]
            if len(bets) > 0:
                wins = bets["actual_nrfi"].sum()
                # NRFI typically pays ~even money (-110 vig), model as -110
                profit = wins * 100 - (len(bets) - wins) * 110
                roi = profit / (len(bets) * 110) * 100
                log(f"    P(NRFI) >= {thresh:.2f}: {len(bets)} bets, {wins} wins ({wins/len(bets):.1%}), ROI: {roi:+.1f}%")

    # ── O/U Analysis ───────────────────────────────────────
    log(f"\n── Over/Under Analysis ({len(df)} games) ──")
    log(f"  Actual mean total:   {df['actual_total'].mean():.2f}")
    log(f"  Sim mean total:      {df['sim_total_mean'].mean():.2f}")
    log(f"  Sim bias (sim-actual): {df['sim_total_mean'].mean() - df['actual_total'].mean():+.2f}")
    log(f"  MAE: {(df['sim_total_mean'] - df['actual_total']).abs().mean():.2f}")
    log(f"  RMSE: {np.sqrt(((df['sim_total_mean'] - df['actual_total'])**2).mean()):.2f}")

    # Correlation
    corr = df["sim_total_mean"].corr(df["actual_total"])
    log(f"  Correlation: {corr:.3f}")

    # O/U accuracy at standard lines
    log(f"\n  O/U accuracy at standard lines:")
    log(f"  {'Line':>6}  {'N_over':>6}  {'Pred%':>6}  {'Act%':>6}  {'Acc':>6}")
    for line in [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]:
        # Use sim's nearest-half line for each game
        pred_over = (df["sim_total_mean"] > line).astype(int)
        actual_over = (df["actual_total"] > line).astype(int)
        # Push (exactly on line) — skip for .5 lines
        if line == int(line):
            mask = df["actual_total"] != line
            pred_over = pred_over[mask]
            actual_over = actual_over[mask]
        acc = (pred_over == actual_over).mean()
        n_over = actual_over.sum()
        log(f"  {line:>6.1f}  {n_over:>6}  {pred_over.mean():>6.3f}  {actual_over.mean():>6.3f}  {acc:>6.3f}")

    # Per-game O/U with game-specific line (nearest 0.5 to sim prediction)
    df["sim_line"] = (df["sim_total_mean"] * 2).round() / 2
    df["pred_over"] = (df["sim_total_mean"] > df["sim_line"]).astype(int)
    # Actually this is always 0 or 0.5 diff, let's use a proper approach:
    # For each game, the "line" is the sim's predicted mean rounded to nearest .5
    # Then check if actual went over or under that line
    df["actual_over"] = (df["actual_total"] > df["sim_line"]).astype(int)
    df["actual_under"] = (df["actual_total"] < df["sim_line"]).astype(int)
    df["push"] = (df["actual_total"] == df["sim_line"]).astype(int)
    no_push = df[df["push"] == 0]
    over_pct = no_push["actual_over"].mean()
    log(f"\n  Game-specific line (sim mean → nearest .5):")
    log(f"  Pushes: {df['push'].sum()} ({df['push'].mean():.1%})")
    log(f"  Over rate (ex pushes): {over_pct:.3f}")
    log(f"  Under rate: {1-over_pct:.3f}")

    # ── Moneyline (for comparison) ─────────────────────────
    log(f"\n── Moneyline (for reference) ──")
    brier_ml = ((df["sim_home_wp"] - df["actual_home_win"]) ** 2).mean()
    base_ml = ((df["actual_home_win"].mean() - df["actual_home_win"]) ** 2).mean()
    log(f"  Brier (MC sim): {brier_ml:.4f}")
    log(f"  Brier (baseline): {base_ml:.4f}")
    log(f"  Brier skill: {1 - brier_ml / base_ml:.4f}")

    # ── Market comparison (if odds data available) ────────
    odds_path = DATA_DIR / "odds" / f"odds_mlb_{season}.parquet"
    kalshi_path = DATA_DIR / "kalshi" / f"kalshi_mlb_{season}.parquet"

    for mkt_name, mkt_path in [("Odds API", odds_path), ("Kalshi", kalshi_path)]:
        if not mkt_path.exists():
            continue
        mkt = pd.read_parquet(mkt_path)
        mkt["game_date"] = mkt["game_date"].astype(str)
        merged = df.merge(mkt, on=["game_date", "home_team", "away_team"], how="inner")
        if len(merged) == 0:
            continue

        log(f"\n── vs {mkt_name} Market ({len(merged)} games) ──")

        # ML comparison
        if "kalshi_home_prob" in merged.columns:
            mkt_brier = ((merged["kalshi_home_prob"] - merged["actual_home_win"]) ** 2).mean()
            sim_brier = ((merged["sim_home_wp"] - merged["actual_home_win"]) ** 2).mean()
            log(f"  ML Brier — {mkt_name}: {mkt_brier:.4f}, Sim: {sim_brier:.4f}")

        # O/U comparison (if we have ou_line)
        if "ou_line" in merged.columns:
            ou_valid = merged.dropna(subset=["ou_line"])
            if len(ou_valid) > 10:
                # Sim prediction vs market line
                sim_over = (ou_valid["sim_total_mean"] > ou_valid["ou_line"]).astype(int)
                actual_over = (ou_valid["actual_total"] > ou_valid["ou_line"]).astype(int)
                # Remove pushes
                no_push = ou_valid["actual_total"] != ou_valid["ou_line"]
                sim_over_np = sim_over[no_push]
                actual_over_np = actual_over[no_push]
                acc = (sim_over_np == actual_over_np).mean()
                log(f"  O/U vs market line: {len(sim_over_np)} bets, {acc:.3f} accuracy")

                # ROI at -110 vig
                wins = (sim_over_np == actual_over_np).sum()
                losses = len(sim_over_np) - wins
                roi = (wins * 100 - losses * 110) / (len(sim_over_np) * 110) * 100
                log(f"  O/U ROI (flat -110): {roi:+.1f}%")

                # Edge-based betting: only bet when sim disagrees with line by > 0.5 runs
                for edge_thresh in [0.3, 0.5, 0.7, 1.0]:
                    sim_edge = ou_valid["sim_total_mean"] - ou_valid["ou_line"]
                    bet_over = ou_valid[sim_edge > edge_thresh]
                    bet_under = ou_valid[sim_edge < -edge_thresh]
                    if len(bet_over) > 5:
                        ov_win = (bet_over["actual_total"] > bet_over["ou_line"]).sum()
                        ov_roi = (ov_win * 100 - (len(bet_over) - ov_win) * 110) / (len(bet_over) * 110) * 100
                        log(f"    Over edge>{edge_thresh:.1f}: {len(bet_over)} bets, {ov_win/len(bet_over):.1%} win, ROI: {ov_roi:+.1f}%")
                    if len(bet_under) > 5:
                        un_win = (bet_under["actual_total"] < bet_under["ou_line"]).sum()
                        un_roi = (un_win * 100 - (len(bet_under) - un_win) * 110) / (len(bet_under) * 110) * 100
                        log(f"    Under edge>{edge_thresh:.1f}: {len(bet_under)} bets, {un_win/len(bet_under):.1%} win, ROI: {un_roi:+.1f}%")

    # ── Walk-forward split: first 30% calibration, last 70% test ──
    n_cal = int(len(df) * 0.3)
    if n_cal >= 50 and len(df) - n_cal >= 100:
        cal_df = df.iloc[:n_cal]
        test_df = df.iloc[n_cal:]
        log(f"\n── Walk-forward split: {n_cal} cal / {len(test_df)} test ──")

        # Test set NRFI
        tv = test_df.dropna(subset=["sim_nrfi_prob"])
        if len(tv) > 0:
            tb = ((tv["sim_nrfi_prob"] - tv["actual_nrfi"]) ** 2).mean()
            tbase = ((tv["actual_nrfi"].mean() - tv["actual_nrfi"]) ** 2).mean()
            log(f"  Test NRFI Brier skill: {1 - tb / tbase:.4f}")

        # Test set O/U
        for line in [8.5, 9.0, 9.5]:
            pred = (test_df["sim_total_mean"] > line).astype(int)
            actual = (test_df["actual_total"] > line).astype(int)
            if line == int(line):
                mask = test_df["actual_total"] != line
                pred, actual = pred[mask], actual[mask]
            acc = (pred == actual).mean()
            log(f"  Test O/U @{line}: {acc:.3f}")

        # Test set ML
        tml = ((test_df["sim_home_wp"] - test_df["actual_home_win"]) ** 2).mean()
        tbase_ml = ((test_df["actual_home_win"].mean() - test_df["actual_home_win"]) ** 2).mean()
        log(f"  Test ML Brier skill: {1 - tml / tbase_ml:.4f}")

        # Test set O/U correlation
        log(f"  Test O/U correlation: {test_df['sim_total_mean'].corr(test_df['actual_total']):.3f}")

    # Save results
    out_dir = DATA_DIR / "backtest"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"nrfi_ou_backtest_{season}.parquet"
    df.to_parquet(out_path, index=False)
    log(f"\n  Results saved to {out_path}")

    return {
        "n_games": len(df),
        "nrfi_brier": float(brier) if len(nrfi_valid) > 0 else None,
        "nrfi_brier_baseline": float(brier_baseline) if len(nrfi_valid) > 0 else None,
        "ou_mae": float((df["sim_total_mean"] - df["actual_total"]).abs().mean()),
        "ou_corr": float(corr),
        "ml_brier": float(brier_ml),
    }


def main():
    parser = argparse.ArgumentParser(description="NRFI + O/U MC sim backtest")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    run_backtest(args.season, args.n_sims, args.max_games)


if __name__ == "__main__":
    main()
