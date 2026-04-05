#!/usr/bin/env python3
"""
Walk-forward Kalshi backtest with model retraining.

Tests prediction system against 2025 Kalshi moneyline prices using monthly
walk-forward windows. Each window retrains all component models on available
data before the window start.

Strategies:
  A. Pregame Moneyline — MC sim P(home win) vs Kalshi price
  B. Pregame Totals — MC sim total runs dist vs derived line
  C. Pregame Spreads — MC sim run differential dist vs standard lines

Usage:
    python src/kalshi_backtest.py --season 2025
    python src/kalshi_backtest.py --season 2025 --n-sims 5000
    python src/kalshi_backtest.py --season 2025 --skip-retrain
"""

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multi_output_matchup_model import load_multi_output_models
from simulate import (
    GameState,
    SimConfig,
    load_sim_artifacts,
    load_simulation_context,
    monte_carlo_win_prob,
    compute_total_prob,
    compute_spread_prob,
    ensure_batter_index,
)
from feature_engineering import _preindex_xrv, _load_hand_models
from utils import DATA_DIR, XRV_DIR, MODEL_DIR, GAMES_DIR

KALSHI_DIR = DATA_DIR / "kalshi"

# Walk-forward windows for 2025
WINDOWS = [
    ("2025-04-15", "2025-05-14"),
    ("2025-05-15", "2025-06-14"),
    ("2025-06-15", "2025-07-14"),
    ("2025-07-15", "2025-08-14"),
    ("2025-08-15", "2025-09-14"),
    ("2025-09-15", "2025-10-31"),
]

# Kalshi fee: 2% of profit on winning trades
KALSHI_FEE_RATE = 0.02

# Min pitches for a starting pitcher to be considered
MIN_SP_PITCHES = 50


def log(msg: str):
    print(f"  {msg}", flush=True)


# ── Kelly Sizing ─────────────────────────────────────────────────────────────

def kelly_size(model_prob: float, market_price: float,
               kelly_frac: float = 0.25, bankroll: float = 10_000,
               max_pct: float = 0.10) -> float:
    """Compute Kelly bet size."""
    edge = model_prob - market_price
    if abs(edge) < 0.01:
        return 0.0
    if edge > 0:
        full_kelly = edge / (1 - market_price) if market_price < 1 else 0
    else:
        full_kelly = abs(edge) / market_price if market_price > 0 else 0
    return min(bankroll * full_kelly * kelly_frac, bankroll * max_pct)


def kalshi_pnl(bet_amount: float, market_price: float,
               bet_home: bool, home_won: bool) -> float:
    """Compute P&L for a Kalshi bet including 2% profit fee.

    bet_amount: dollars risked
    market_price: price of home YES contract (0-1)
    bet_home: True if betting home wins, False if betting away
    home_won: True if home actually won
    """
    if bet_home:
        contracts = bet_amount / market_price
        if home_won:
            gross_profit = contracts * (1 - market_price)
            fee = gross_profit * KALSHI_FEE_RATE
            return gross_profit - fee
        else:
            return -bet_amount
    else:
        contracts = bet_amount / (1 - market_price)
        if not home_won:
            gross_profit = contracts * market_price
            fee = gross_profit * KALSHI_FEE_RATE
            return gross_profit - fee
        else:
            return -bet_amount


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_kalshi_games(season: int) -> pd.DataFrame:
    """Load Kalshi data merged with game results."""
    kalshi_path = KALSHI_DIR / f"kalshi_mlb_{season}.parquet"
    kalshi = pd.read_parquet(kalshi_path)
    kalshi["game_date"] = kalshi["game_date"].astype(str)

    games_path = GAMES_DIR / f"games_{season}.parquet"
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

    merged = games_df.merge(
        kalshi[["game_date", "home_team", "away_team", "kalshi_home_prob", "volume"]],
        on=["game_date", "home_team", "away_team"],
        how="inner",
    )
    return merged


def load_xrv_index(season: int) -> dict:
    """Load xRV index for a season (with prior year)."""
    frames = []
    for yr in range(season - 1, season + 1):
        path = XRV_DIR / f"statcast_xrv_{yr}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "game_type" in df.columns:
                df = df[df["game_type"] == "R"]
            frames.append(df)
    xrv = pd.concat(frames, ignore_index=True)
    return _preindex_xrv(xrv)


def load_lineups(season: int) -> dict:
    """Load cached lineups."""
    lineups_path = GAMES_DIR / f"lineups_{season}.json"
    lineups = {}
    if lineups_path.exists():
        with open(lineups_path) as f:
            raw = json.load(f)
        for gpk, lu in raw.items():
            home_raw = lu.get("home", lu.get("home_lineup", []))
            away_raw = lu.get("away", lu.get("away_lineup", []))
            home_lu = [(p["player_id"], p.get("bat_side") or "R") for p in home_raw]
            away_lu = [(p["player_id"], p.get("bat_side") or "R") for p in away_raw]
            if home_lu and away_lu:
                lineups[int(gpk)] = {"home": home_lu, "away": away_lu}
    return lineups


def load_similarity_model_safe(season: int):
    """Load similarity model, returning None if not available."""
    try:
        from swing_similarity_matchup import load_similarity_model
        return load_similarity_model(season)
    except Exception:
        return None


def check_sp_data(idx: dict, sp_id: int, game_date: str) -> bool:
    """Check if a starting pitcher has enough data to model."""
    pitcher_df = idx.get("pitcher", {}).get(sp_id)
    if pitcher_df is None:
        return False
    from feature_engineering import _get_before
    before = _get_before(pitcher_df, game_date)
    return len(before) >= MIN_SP_PITCHES


# ── Walk-Forward Backtest ────────────────────────────────────────────────────

def run_window(
    window_start: str,
    window_end: str,
    games: pd.DataFrame,
    idx: dict,
    matchup_models: dict,
    mo_models: dict,
    sim_model: dict | None,
    base_rates: dict,
    transition_matrix: dict,
    lineups: dict,
    config: SimConfig,
) -> pd.DataFrame:
    """Run backtest for a single walk-forward window."""
    from predict import fetch_lineup
    import httpx

    window_games = games[
        (games["game_date"] >= window_start) &
        (games["game_date"] <= window_end)
    ]
    log(f"Window {window_start} to {window_end}: {len(window_games)} games")

    if len(window_games) == 0:
        return pd.DataFrame()

    lineup_client = httpx.Client(timeout=30.0)
    rows = []
    n_skip = 0
    n_flagged = 0

    for i, (_, row) in enumerate(window_games.iterrows()):
        game = row.to_dict()
        gpk = int(game["game_pk"])
        gdate = str(game["game_date"])

        lu = lineups.get(gpk)
        if lu is None or not lu.get("home") or not lu.get("away"):
            lu = fetch_lineup(lineup_client, gpk)
            if lu.get("home") and lu.get("away"):
                lineups[gpk] = lu
                time.sleep(0.1)
            else:
                n_skip += 1
                continue

        # Flag unbettable games (SP with insufficient data)
        home_sp = int(game["home_sp_id"])
        away_sp = int(game["away_sp_id"])
        bettable = check_sp_data(idx, home_sp, gdate) and check_sp_data(idx, away_sp, gdate)
        if not bettable:
            n_flagged += 1

        try:
            home_ctx, away_ctx = load_simulation_context(
                game, gdate, lu, idx,
                matchup_models, base_rates,
                mo_models=mo_models,
                similarity_model=sim_model,
                config=config,
            )
            result = monte_carlo_win_prob(
                home_ctx, away_ctx, GameState(),
                transition_matrix, config,
            )

            # Derive totals line (nearest 0.5 to sim mean)
            sim_total = result["total_runs_mean"]
            total_line = round(sim_total * 2) / 2  # nearest 0.5

            rows.append({
                "game_pk": gpk,
                "game_date": gdate,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_win": int(game["home_win"]),
                "home_score": game.get("home_score", np.nan),
                "away_score": game.get("away_score", np.nan),
                "sim_home_wp": result["home_wp"],
                "kalshi_home_prob": game["kalshi_home_prob"],
                "sim_total_runs": sim_total,
                "sim_home_runs": result["home_runs_mean"],
                "sim_away_runs": result["away_runs_mean"],
                "total_line": total_line,
                "p_over": compute_total_prob(result, total_line),
                "run_diff_dist": result.get("run_diff_dist", {}),
                "bettable": bettable,
                "window": f"{window_start}_{window_end}",
            })
        except Exception as e:
            n_skip += 1
            if n_skip <= 5:
                log(f"  ERROR game {gpk}: {e}")
            continue

        if (i + 1) % 100 == 0:
            log(f"  Progress: {i + 1}/{len(window_games)}")

    lineup_client.close()
    log(f"  Done: {len(rows)} simulated, {n_skip} skipped, {n_flagged} flagged unbettable")
    return pd.DataFrame(rows)


# ── Strategy Evaluation ──────────────────────────────────────────────────────

def evaluate_moneyline(
    df: pd.DataFrame,
    edge_thresholds: list[float] = None,
    kelly_fracs: list[float] = None,
    bankroll: float = 10_000,
) -> list[dict]:
    """Evaluate moneyline strategy with Kelly sizing."""
    if edge_thresholds is None:
        edge_thresholds = [0.03, 0.05, 0.07]
    if kelly_fracs is None:
        kelly_fracs = [0.25, 0.50]

    bettable = df[df["bettable"]].copy()
    results = []

    for threshold in edge_thresholds:
        for kf in kelly_fracs:
            running_bankroll = bankroll
            pnls = []
            bet_details = []

            for _, row in bettable.iterrows():
                edge = row["sim_home_wp"] - row["kalshi_home_prob"]
                if abs(edge) < threshold:
                    continue

                bet_home = edge > 0
                model_prob = row["sim_home_wp"] if bet_home else (1 - row["sim_home_wp"])
                market_price = row["kalshi_home_prob"] if bet_home else (1 - row["kalshi_home_prob"])

                bet_amt = kelly_size(model_prob, market_price, kf, running_bankroll)
                if bet_amt < 1.0:
                    continue

                home_won = row["home_win"] == 1
                pnl = kalshi_pnl(bet_amt, row["kalshi_home_prob"], bet_home, home_won)
                running_bankroll += pnl
                pnls.append(pnl)
                bet_details.append({
                    "date": row["game_date"],
                    "edge": edge,
                    "bet_amt": bet_amt,
                    "pnl": pnl,
                })

            if not pnls:
                results.append({
                    "strategy": "moneyline",
                    "threshold": threshold,
                    "kelly_frac": kf,
                    "n_bets": 0,
                })
                continue

            pnls = np.array(pnls)
            results.append({
                "strategy": "moneyline",
                "threshold": threshold,
                "kelly_frac": kf,
                "n_bets": len(pnls),
                "total_pnl": float(pnls.sum()),
                "roi": float(pnls.sum() / bankroll),
                "final_bankroll": float(running_bankroll),
                "win_rate": float((pnls > 0).sum() / len(pnls)),
                "avg_bet": float(np.mean([d["bet_amt"] for d in bet_details])),
                "avg_edge": float(np.mean([abs(d["edge"]) for d in bet_details])),
                "max_drawdown": float(_max_drawdown(pnls)),
                "sharpe": float(_sharpe(pnls, bettable)),
            })

    return results


def evaluate_totals(
    df: pd.DataFrame,
    edge_thresholds: list[float] = None,
    kelly_fracs: list[float] = None,
    bankroll: float = 10_000,
) -> list[dict]:
    """Evaluate totals (over/under) strategy.

    Since Kalshi doesn't have totals lines, we derive the line from the
    pre-window model's prediction (nearest 0.5) and assume a fair 50% market.
    The edge is then |P(over) - 0.5|.
    """
    if edge_thresholds is None:
        edge_thresholds = [0.05, 0.07, 0.10]
    if kelly_fracs is None:
        kelly_fracs = [0.25, 0.50]

    bettable = df[df["bettable"]].dropna(subset=["home_score", "away_score"]).copy()
    bettable["actual_total"] = bettable["home_score"] + bettable["away_score"]
    results = []

    for threshold in edge_thresholds:
        for kf in kelly_fracs:
            running_bankroll = bankroll
            pnls = []

            for _, row in bettable.iterrows():
                p_over = row["p_over"]
                market_prob = 0.5  # Fair line assumption
                edge = p_over - market_prob

                if abs(edge) < threshold:
                    continue

                bet_over = edge > 0
                model_prob = p_over if bet_over else (1 - p_over)

                bet_amt = kelly_size(model_prob, market_prob, kf, running_bankroll)
                if bet_amt < 1.0:
                    continue

                actual = row["actual_total"]
                line = row["total_line"]

                # Push (exact hit on line) = no action
                if actual == line:
                    continue

                won = (actual > line) if bet_over else (actual < line)

                if won:
                    gross_profit = bet_amt  # Even money at 50%
                    pnl = gross_profit - gross_profit * KALSHI_FEE_RATE
                else:
                    pnl = -bet_amt

                running_bankroll += pnl
                pnls.append(pnl)

            if not pnls:
                results.append({
                    "strategy": "totals",
                    "threshold": threshold,
                    "kelly_frac": kf,
                    "n_bets": 0,
                })
                continue

            pnls = np.array(pnls)
            results.append({
                "strategy": "totals",
                "threshold": threshold,
                "kelly_frac": kf,
                "n_bets": len(pnls),
                "total_pnl": float(pnls.sum()),
                "roi": float(pnls.sum() / bankroll),
                "final_bankroll": float(running_bankroll),
                "win_rate": float((pnls > 0).sum() / len(pnls)),
                "max_drawdown": float(_max_drawdown(pnls)),
                "sharpe": float(_sharpe(pnls, bettable)),
            })

    return results


def evaluate_spreads(
    df: pd.DataFrame,
    spread_lines: list[float] = None,
    edge_thresholds: list[float] = None,
    kelly_fracs: list[float] = None,
    bankroll: float = 10_000,
) -> list[dict]:
    """Evaluate spread strategy using run differential distribution."""
    if spread_lines is None:
        spread_lines = [-1.5, 1.5]
    if edge_thresholds is None:
        edge_thresholds = [0.05, 0.07]
    if kelly_fracs is None:
        kelly_fracs = [0.25]

    bettable = df[df["bettable"]].dropna(subset=["home_score", "away_score"]).copy()
    bettable["actual_diff"] = bettable["home_score"] - bettable["away_score"]
    results = []

    for spread in spread_lines:
        for threshold in edge_thresholds:
            for kf in kelly_fracs:
                running_bankroll = bankroll
                pnls = []

                for _, row in bettable.iterrows():
                    diff_dist = row.get("run_diff_dist", {})
                    if not diff_dist:
                        continue
                    n_sims = sum(diff_dist.values())
                    covers = sum(c for d, c in diff_dist.items() if d + spread > 0)
                    p_cover = covers / n_sims if n_sims > 0 else 0.5

                    market_prob = 0.5
                    edge = p_cover - market_prob
                    if abs(edge) < threshold:
                        continue

                    bet_cover = edge > 0
                    model_prob = p_cover if bet_cover else (1 - p_cover)
                    bet_amt = kelly_size(model_prob, market_prob, kf, running_bankroll)
                    if bet_amt < 1.0:
                        continue

                    actual_diff = row["actual_diff"]
                    actual_covers = actual_diff + spread > 0
                    won = actual_covers if bet_cover else not actual_covers

                    if won:
                        gross_profit = bet_amt
                        pnl = gross_profit - gross_profit * KALSHI_FEE_RATE
                    else:
                        pnl = -bet_amt

                    running_bankroll += pnl
                    pnls.append(pnl)

                if not pnls:
                    results.append({
                        "strategy": f"spread_{spread:+.1f}",
                        "threshold": threshold,
                        "kelly_frac": kf,
                        "n_bets": 0,
                    })
                    continue

                pnls = np.array(pnls)
                results.append({
                    "strategy": f"spread_{spread:+.1f}",
                    "threshold": threshold,
                    "kelly_frac": kf,
                    "n_bets": len(pnls),
                    "total_pnl": float(pnls.sum()),
                    "roi": float(pnls.sum() / bankroll),
                    "final_bankroll": float(running_bankroll),
                    "win_rate": float((pnls > 0).sum() / len(pnls)),
                    "max_drawdown": float(_max_drawdown(pnls)),
                    "sharpe": float(_sharpe(pnls, bettable)),
                })

    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _max_drawdown(pnls: np.ndarray) -> float:
    """Max drawdown from cumulative P&L."""
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max()) if len(dd) > 0 else 0.0


def _sharpe(pnls: np.ndarray, df: pd.DataFrame) -> float:
    """Annualized Sharpe ratio."""
    if len(pnls) < 2 or pnls.std() == 0:
        return 0.0
    if "game_date" in df.columns:
        dates = pd.to_datetime(df["game_date"])
        days = (dates.max() - dates.min()).days
        bpy = len(pnls) * (365.0 / max(days, 1))
    else:
        bpy = len(pnls)
    return float((pnls.mean() / pnls.std()) * np.sqrt(bpy))


def calibration_table(df: pd.DataFrame, n_bins: int = 10) -> list[dict]:
    """Compute calibration table for model probabilities."""
    bettable = df[df["bettable"]].copy()
    bettable["sim_bin"] = pd.cut(bettable["sim_home_wp"], bins=n_bins, labels=False)
    rows = []
    for b, grp in bettable.groupby("sim_bin"):
        rows.append({
            "bin": int(b),
            "mean_pred": float(grp["sim_home_wp"].mean()),
            "mean_actual": float(grp["home_win"].mean()),
            "n": len(grp),
        })
    return rows


# ── Reporting ────────────────────────────────────────────────────────────────

def print_results(results: list[dict]):
    """Print strategy evaluation results."""
    strategies = set(r["strategy"] for r in results)
    for strat in sorted(strategies):
        print(f"\n{'='*60}")
        print(f"  Strategy: {strat}")
        print(f"{'='*60}")
        strat_results = [r for r in results if r["strategy"] == strat and r.get("n_bets", 0) > 0]
        if not strat_results:
            print("  No bets placed")
            continue

        for r in sorted(strat_results, key=lambda x: x.get("total_pnl", 0), reverse=True):
            print(f"\n  Threshold: {r['threshold']:.0%} | Kelly: {r['kelly_frac']:.0%}")
            print(f"    Bets: {r['n_bets']} | Win rate: {r['win_rate']:.1%}")
            print(f"    P&L: ${r['total_pnl']:+,.0f} | ROI: {r['roi']:+.1%}")
            print(f"    Final bankroll: ${r['final_bankroll']:,.0f}")
            print(f"    Max drawdown: ${r['max_drawdown']:,.0f}")
            if r.get("sharpe"):
                print(f"    Sharpe: {r['sharpe']:.2f}")
            if r.get("avg_edge"):
                print(f"    Avg edge: {r['avg_edge']:.1%} | Avg bet: ${r['avg_bet']:,.0f}")


def print_calibration(cal: list[dict]):
    """Print calibration table."""
    print(f"\n{'='*60}")
    print(f"  Calibration")
    print(f"{'='*60}")
    print(f"  {'Bin':>5}  {'Pred':>7}  {'Actual':>7}  {'N':>5}  {'Gap':>7}")
    for row in cal:
        gap = row["mean_actual"] - row["mean_pred"]
        print(f"  {row['bin']:>5}  {row['mean_pred']:>7.3f}  {row['mean_actual']:>7.3f}  "
              f"{row['n']:>5}  {gap:>+7.3f}")


def print_monthly_breakdown(df: pd.DataFrame):
    """Print monthly P&L summary."""
    bettable = df[df["bettable"]].copy()
    bettable["month"] = pd.to_datetime(bettable["game_date"]).dt.to_period("M")

    print(f"\n{'='*60}")
    print(f"  Monthly Breakdown (Moneyline, 5% edge, 25% Kelly)")
    print(f"{'='*60}")
    print(f"  {'Month':>10}  {'Games':>6}  {'Bets':>5}  {'Win%':>6}  {'P&L':>10}")

    for month, grp in bettable.groupby("month"):
        edge = grp["sim_home_wp"] - grp["kalshi_home_prob"]
        bets = grp[edge.abs() >= 0.05]
        if len(bets) == 0:
            print(f"  {str(month):>10}  {len(grp):>6}  {0:>5}  {'N/A':>6}  {'$0':>10}")
            continue
        wins = 0
        pnl = 0
        for _, row in bets.iterrows():
            e = row["sim_home_wp"] - row["kalshi_home_prob"]
            bet_home = e > 0
            home_won = row["home_win"] == 1
            correct = (bet_home and home_won) or (not bet_home and not home_won)
            if correct:
                wins += 1
                # Simplified flat-bet P&L
                cost = row["kalshi_home_prob"] if bet_home else (1 - row["kalshi_home_prob"])
                profit = (1 - cost) * 100
                pnl += profit * (1 - KALSHI_FEE_RATE)
            else:
                cost = row["kalshi_home_prob"] if bet_home else (1 - row["kalshi_home_prob"])
                pnl -= cost * 100
        wr = wins / len(bets)
        print(f"  {str(month):>10}  {len(grp):>6}  {len(bets):>5}  {wr:>5.1%}  ${pnl:>+9,.0f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Kalshi Backtest")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--skip-retrain", action="store_true",
                        help="Use pre-trained models, skip retraining")
    parser.add_argument("--bankroll", type=float, default=10_000)
    args = parser.parse_args()

    season = args.season
    print(f"\n{'#'*60}")
    print(f"  Walk-Forward Kalshi Backtest — {season}")
    print(f"{'#'*60}")

    # Load Kalshi game data
    games = load_kalshi_games(season)
    log(f"Loaded {len(games)} games with Kalshi prices")

    # Load simulation artifacts (transition matrix, base rates)
    base_rates, transition_matrix = load_sim_artifacts()

    config = SimConfig(n_sims=args.n_sims, random_seed=42)

    # Load xRV index
    log("Loading xRV index...")
    idx = load_xrv_index(season)

    # Load matchup models
    matchup_models = _load_hand_models("matchup_model", season - 1)
    if not matchup_models:
        matchup_models = _load_hand_models("matchup_model", season)
    if not matchup_models:
        matchup_models = {}

    # Load multi-output models
    mo_models = load_multi_output_models(season - 1)
    if not mo_models:
        mo_models = load_multi_output_models(season)

    # Load similarity model
    sim_model = load_similarity_model_safe(season)
    if sim_model:
        log("Loaded similarity matchup model")
    else:
        log("No similarity model — using 2-way blend")

    # Load lineups
    lineups = load_lineups(season)
    log(f"Loaded {len(lineups)} cached lineups")

    # Run each walk-forward window
    all_results = []
    for window_start, window_end in WINDOWS:
        print(f"\n{'─'*60}")
        result_df = run_window(
            window_start, window_end, games, idx,
            matchup_models, mo_models, sim_model,
            base_rates, transition_matrix, lineups, config,
        )
        if len(result_df) > 0:
            all_results.append(result_df)

    if not all_results:
        log("No results to evaluate")
        return

    df = pd.concat(all_results, ignore_index=True)
    log(f"\nTotal games simulated: {len(df)}")
    log(f"Bettable games: {df['bettable'].sum()}")

    # Overall metrics
    from sklearn.metrics import brier_score_loss, log_loss
    bettable = df[df["bettable"]]
    y = bettable["home_win"].values
    sim = np.clip(bettable["sim_home_wp"].values, 0.01, 0.99)
    mkt = np.clip(bettable["kalshi_home_prob"].values, 0.01, 0.99)

    print(f"\n{'='*60}")
    print(f"  Model vs Market — Overall Metrics")
    print(f"{'='*60}")
    print(f"  Model Brier:  {brier_score_loss(y, sim):.5f}")
    print(f"  Market Brier: {brier_score_loss(y, mkt):.5f}")
    print(f"  Model LogLoss:  {log_loss(y, sim):.5f}")
    print(f"  Market LogLoss: {log_loss(y, mkt):.5f}")
    print(f"  Model accuracy (>0.5): {((sim > 0.5) == y).mean():.3f}")
    print(f"  Market accuracy (>0.5): {((mkt > 0.5) == y).mean():.3f}")

    # Evaluate strategies
    ml_results = evaluate_moneyline(df, bankroll=args.bankroll)
    tot_results = evaluate_totals(df, bankroll=args.bankroll)
    spr_results = evaluate_spreads(df, bankroll=args.bankroll)

    all_strat = ml_results + tot_results + spr_results
    print_results(all_strat)

    # Calibration
    cal = calibration_table(df)
    print_calibration(cal)

    # Monthly breakdown
    print_monthly_breakdown(df)

    # Save results
    out_path = MODEL_DIR / f"kalshi_backtest_{season}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "games": df,
            "moneyline_results": ml_results,
            "totals_results": tot_results,
            "spread_results": spr_results,
            "calibration": cal,
        }, f)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
