#!/usr/bin/env python3
"""
Ensemble predictions: blends MC simulator + win model (ML) and MC sim + LGB (O/U).

Strategy:
  - Moneyline: blend sim_home_wp (from MC sim) with ensemble win-model probability
  - O/U:       blend sim_total_mean with LGB predicted total runs

Calibration set: first 30% of games (chronological)
Test set: last 70%

Outputs:
  - Brier scores for ML: sim-only, win-model-only, ensemble
  - O/U accuracy: sim-only, LGB-only, ensemble
  - Edge-based ROI vs Kalshi for the ML ensemble
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SIM_PATH    = ROOT / "data/backtest/nrfi_ou_backtest_2025.parquet"
WIN_PATH    = ROOT / "data/audit/walk_forward_predictions.csv"
LGB_PATH    = ROOT / "data/backtest/total_runs_lgb_2025.parquet"
KALSHI_PATH = ROOT / "data/kalshi/kalshi_mlb_2025.parquet"

CLAMP = 1e-4   # probability floor/ceiling


# ── helpers ────────────────────────────────────────────────────────────────

def clamp(p: np.ndarray) -> np.ndarray:
    return np.clip(p, CLAMP, 1 - CLAMP)


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, clamp(y_prob)))


def logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(log_loss(y_true, clamp(y_prob)))


def ou_accuracy(pred_over_prob: np.ndarray, actual_over: np.ndarray,
                threshold: float = 0.5) -> float:
    """Fraction of non-push games called correctly at the given threshold."""
    pred = (pred_over_prob > threshold).astype(int)
    return float((pred == actual_over).mean())


def p_over_from_mean_std(mean: np.ndarray, std: np.ndarray,
                          line: np.ndarray) -> np.ndarray:
    """Normal CDF P(total > line) given mean/std."""
    from scipy.stats import norm
    return norm.sf(line, loc=mean, scale=std)


def ou_from_line(df: pd.DataFrame, pred_total: np.ndarray) -> np.ndarray:
    """Return P(over) for each game using the game's sim_line as the O/U line."""
    # Use the residual std from LGB when available, else a fixed prior
    if "lgb_residual_std" in df.columns:
        std = df["lgb_residual_std"].fillna(4.5).values
    else:
        std = np.full(len(df), 4.5)
    return p_over_from_mean_std(pred_total, std, df["sim_line"].values)


# ── data loading ───────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Merge all sources on game_pk (and date/team for Kalshi)."""

    sim = pd.read_parquet(SIM_PATH)
    # Drop push games for O/U evaluation
    sim = sim[sim["push"] == 0].copy()
    sim = sim.sort_values("game_date").reset_index(drop=True)

    # Win model walk-forward predictions (2025 season)
    wf = pd.read_csv(WIN_PATH)
    wf_2025 = wf[wf["season"] == 2025][
        ["game_pk", "lr_prob", "xgb_prob", "ens_prob", "ens_calibrated"]
    ].rename(columns={
        "lr_prob":        "wm_lr_prob",
        "xgb_prob":       "wm_xgb_prob",
        "ens_prob":       "wm_ens_prob",
        "ens_calibrated": "wm_cal_prob",
    })
    sim = sim.merge(wf_2025, on="game_pk", how="left")

    # LGB total runs predictions
    lgb = pd.read_parquet(LGB_PATH)[[
        "game_pk", "lgb_pred_total", "lgb_residual_std",
        "lgb_p_over_7.5", "lgb_p_over_8.0", "lgb_p_over_8.5",
        "lgb_p_over_9.0", "lgb_p_over_9.5", "lgb_p_over_10.0",
    ]]
    sim = sim.merge(lgb, on="game_pk", how="left")

    # Kalshi market data — merge on date + home/away team
    kal = pd.read_parquet(KALSHI_PATH)[[
        "game_date", "home_team", "away_team",
        "kalshi_home_prob", "kalshi_away_prob", "volume",
    ]]
    kal["game_date"] = kal["game_date"].astype(str)
    sim["game_date_str"] = sim["game_date"].astype(str)
    sim = sim.merge(
        kal,
        left_on=["game_date_str", "home_team", "away_team"],
        right_on=["game_date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_kal"),
    )
    sim = sim.drop(columns=["game_date_str", "game_date_kal"], errors="ignore")

    n_kalshi = sim["kalshi_home_prob"].notna().sum()
    print(f"Loaded {len(sim)} games (push-free)  |  Kalshi coverage: {n_kalshi}/{len(sim)}")
    return sim


# ── grid-search optimal weights ───────────────────────────────────────────

def grid_search_weights(y_true: np.ndarray,
                         *prob_arrays,
                         metric: str = "brier",
                         n_steps: int = 101) -> np.ndarray:
    """
    Grid-search convex combination weights for 2 or 3 probability arrays.
    metric: 'brier' or 'logloss'
    Returns weight vector summing to 1.
    """
    n_models = len(prob_arrays)
    best_w = np.ones(n_models) / n_models
    best_loss = np.inf

    loss_fn = brier if metric == "brier" else logloss

    if n_models == 2:
        for i in range(n_steps):
            w1 = i / (n_steps - 1)
            blend = w1 * prob_arrays[0] + (1 - w1) * prob_arrays[1]
            loss = loss_fn(y_true, blend)
            if loss < best_loss:
                best_loss = loss
                best_w = np.array([w1, 1 - w1])
    elif n_models == 3:
        for i in range(n_steps):
            for j in range(n_steps - i):
                w1 = i / (n_steps - 1)
                w2 = j / (n_steps - 1)
                w3 = 1 - w1 - w2
                if w3 < 0:
                    continue
                blend = w1 * prob_arrays[0] + w2 * prob_arrays[1] + w3 * prob_arrays[2]
                loss = loss_fn(y_true, blend)
                if loss < best_loss:
                    best_loss = loss
                    best_w = np.array([w1, w2, w3])
    else:
        raise ValueError(f"Unsupported n_models={n_models}")

    return best_w


# ── O/U ensemble helpers ───────────────────────────────────────────────────

def find_best_lgb_line_column(df: pd.DataFrame, line_col: str = "sim_line") -> str:
    """
    For each game find which discrete LGB over-prob column matches the actual line.
    Returns the most common matching column name (for the calibration set).
    """
    line_values = df[line_col].unique()
    # LGB columns cover 7.5, 8.0, 8.5, 9.0, 9.5, 10.0
    available_lines = [7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    col_map = {l: f"lgb_p_over_{l}" for l in available_lines}

    # For each game: pick closest available LGB line
    def pick_lgb_prob(row):
        line = row[line_col]
        closest = min(available_lines, key=lambda l: abs(l - line))
        col = col_map[closest]
        if col in df.columns:
            return row[col]
        return np.nan

    return df.apply(pick_lgb_prob, axis=1)


# ── ROI vs Kalshi ─────────────────────────────────────────────────────────

def edge_roi(df: pd.DataFrame,
             pred_prob: np.ndarray,
             edge_thresholds: list = None) -> pd.DataFrame:
    """
    For games where Kalshi is available, compute edge = our_prob - kalshi_prob.
    Bet 1 unit on home when edge > threshold, away when edge < -threshold.
    Assumes fair Kalshi pricing (no vig deduction) for simplicity.
    """
    if edge_thresholds is None:
        edge_thresholds = [0.02, 0.03, 0.05, 0.07, 0.10]

    mask = df["kalshi_home_prob"].notna().values
    if mask.sum() == 0:
        print("  No Kalshi data for ROI analysis.")
        return pd.DataFrame()

    df_k = df[mask].copy()
    pp = pred_prob[mask]
    y = df_k["actual_home_win"].values
    k_home = df_k["kalshi_home_prob"].values

    edge = pp - k_home  # positive = we think home is more likely than market

    rows = []
    for thresh in edge_thresholds:
        # Home bets
        home_bets = edge > thresh
        # Away bets
        away_bets = edge < -thresh

        # P&L: bet on home → win if home wins; bet on away → win if home loses
        # payout at fair odds: win = (1 - k_home)/k_home units per unit risked
        home_pnl = np.where(
            home_bets,
            np.where(y == 1,
                     (1 - k_home) / np.maximum(k_home, 0.01),   # win
                     -1.0),                                        # loss
            0.0
        )
        away_pnl = np.where(
            away_bets,
            np.where(y == 0,
                     k_home / np.maximum(1 - k_home, 0.01),      # win
                     -1.0),                                        # loss
            0.0
        )

        n_bets = int(home_bets.sum() + away_bets.sum())
        total_pnl = float(home_pnl.sum() + away_pnl.sum())
        roi = total_pnl / max(n_bets, 1)

        n_wins = int(
            (home_bets & (y == 1)).sum() +
            (away_bets & (y == 0)).sum()
        )
        wr = n_wins / max(n_bets, 1)

        rows.append({
            "edge_threshold": thresh,
            "n_bets": n_bets,
            "n_wins": n_wins,
            "win_rate": wr,
            "total_pnl_units": total_pnl,
            "roi_per_bet": roi,
        })

    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ENSEMBLE PREDICTIONS  |  MLB 2025")
    print("=" * 70)

    df = load_data()

    # Chronological calibration (30%) / test (70%) split
    n_cal = int(len(df) * 0.30)
    cal = df.iloc[:n_cal].copy()
    test = df.iloc[n_cal:].copy()
    print(f"\nCalibration set: {len(cal)} games  |  Test set: {len(test)} games")
    print(f"  Cal dates:  {cal['game_date'].min()} → {cal['game_date'].max()}")
    print(f"  Test dates: {test['game_date'].min()} → {test['game_date'].max()}")

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 1: MONEYLINE (win probability)
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("SECTION 1: MONEYLINE ENSEMBLE")
    print("─" * 70)

    # Components
    sim_wp_cal  = cal["sim_home_wp"].values
    wm_cal      = cal["wm_cal_prob"].fillna(0.5).values
    y_win_cal   = cal["actual_home_win"].values

    sim_wp_test = test["sim_home_wp"].values
    wm_test     = test["wm_cal_prob"].fillna(0.5).values
    y_win_test  = test["actual_home_win"].values

    # Grid-search weights on calibration set
    ml_weights = grid_search_weights(
        y_win_cal, sim_wp_cal, wm_cal, metric="brier"
    )
    print(f"\nOptimal ML blend (cal set, Brier):")
    print(f"  w_sim = {ml_weights[0]:.3f}  |  w_win_model = {ml_weights[1]:.3f}")

    ens_ml_test = ml_weights[0] * sim_wp_test + ml_weights[1] * wm_test

    print("\n── Test-set Moneyline Metrics ──────────────────────────────────────")
    baseline_rate = y_win_test.mean()
    baseline_brier = brier(y_win_test, np.full(len(y_win_test), baseline_rate))

    rows_ml = []
    for label, probs in [
        ("Sim only (sim_home_wp)", sim_wp_test),
        ("Win model only (ens_calibrated)", wm_test),
        ("Ensemble (blend)", ens_ml_test),
    ]:
        b = brier(y_win_test, probs)
        ll = logloss(y_win_test, probs)
        try:
            auc = roc_auc_score(y_win_test, probs)
        except Exception:
            auc = float("nan")
        rows_ml.append({"Model": label, "Brier": b, "LogLoss": ll, "AUC": auc})
        print(f"  {label}")
        print(f"    Brier={b:.4f} (baseline={baseline_brier:.4f})  "
              f"LogLoss={ll:.4f}  AUC={auc:.4f}")

    ml_df = pd.DataFrame(rows_ml)

    # Correlation of the two signals
    corr = np.corrcoef(sim_wp_test, wm_test)[0, 1]
    print(f"\n  Pearson correlation (sim_wp vs win_model): {corr:.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 2: OVER/UNDER ENSEMBLE
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("SECTION 2: O/U ENSEMBLE")
    print("─" * 70)

    # Sim P(over) — derived from pred_over flag, but recompute from mean/std for blend
    # The sim already has pred_over (binary). We'll use sim_total_mean vs sim_line as a probability.
    # For calibration we need continuous probabilities, so we compute them from a normal approx.
    sim_std_prior = 4.5   # fallback std dev for sim totals

    sim_p_over_cal  = ou_from_line(cal,  cal["sim_total_mean"].values)
    sim_p_over_test = ou_from_line(test, test["sim_total_mean"].values)

    lgb_p_over_cal  = find_best_lgb_line_column(cal)
    lgb_p_over_test = find_best_lgb_line_column(test)

    y_over_cal  = cal["actual_over"].values
    y_over_test = test["actual_over"].values

    # Drop rows with NaN LGB probs
    cal_valid  = lgb_p_over_cal.notna()
    test_valid = lgb_p_over_test.notna()

    ou_weights = grid_search_weights(
        y_over_cal[cal_valid],
        sim_p_over_cal[cal_valid],
        lgb_p_over_cal[cal_valid].values,
        metric="brier",
    )
    print(f"\nOptimal O/U blend (cal set, Brier):")
    print(f"  w_sim = {ou_weights[0]:.3f}  |  w_lgb = {ou_weights[1]:.3f}")

    ens_ou_test = (
        ou_weights[0] * sim_p_over_test[test_valid] +
        ou_weights[1] * lgb_p_over_test[test_valid].values
    )

    print("\n── Test-set O/U Metrics ────────────────────────────────────────────")
    baseline_ou_rate = y_over_test[test_valid].mean()
    baseline_ou_brier = brier(y_over_test[test_valid],
                               np.full(test_valid.sum(), baseline_ou_rate))

    rows_ou = []
    for label, probs in [
        ("Sim only (normal approx)",  sim_p_over_test[test_valid]),
        ("LGB only (matched line)",   lgb_p_over_test[test_valid].values),
        ("Ensemble (blend)",          ens_ou_test),
    ]:
        b    = brier(y_over_test[test_valid], probs)
        acc  = ou_accuracy(probs, y_over_test[test_valid])
        rows_ou.append({"Model": label, "Brier": b, "Accuracy": acc})
        print(f"  {label}")
        print(f"    Brier={b:.4f} (baseline={baseline_ou_brier:.4f})  "
              f"O/U Accuracy={acc:.4f}")

    ou_df = pd.DataFrame(rows_ou)

    # Also report simple binary O/U from pred_over (original sim flag)
    acc_sim_flag = ou_accuracy(
        test["pred_over"].values.astype(float),
        y_over_test
    )
    print(f"\n  [Reference] Sim binary pred_over accuracy (all test games): "
          f"{acc_sim_flag:.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 3: ROI vs KALSHI
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("SECTION 3: EDGE-BASED ROI vs KALSHI  (test set)")
    print("─" * 70)

    print("\n  -- Sim-only --")
    roi_sim = edge_roi(test, sim_wp_test)
    if not roi_sim.empty:
        print(roi_sim.to_string(index=False))

    print("\n  -- Win-model-only --")
    roi_wm = edge_roi(test, wm_test)
    if not roi_wm.empty:
        print(roi_wm.to_string(index=False))

    print("\n  -- Ensemble (ML blend) --")
    roi_ens = edge_roi(test, ens_ml_test)
    if not roi_ens.empty:
        print(roi_ens.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nMoneyline Brier Scores (test set):")
    for _, row in ml_df.iterrows():
        marker = " <-- best" if row["Brier"] == ml_df["Brier"].min() else ""
        print(f"  {row['Model']:<45}  Brier={row['Brier']:.4f}{marker}")

    print(f"\nO/U Brier Scores (test set, valid LGB rows={test_valid.sum()}):")
    for _, row in ou_df.iterrows():
        marker = " <-- best" if row["Brier"] == ou_df["Brier"].min() else ""
        print(f"  {row['Model']:<45}  Brier={row['Brier']:.4f}  "
              f"Acc={row['Accuracy']:.4f}{marker}")

    print(f"\nSignal correlation (sim_wp vs win_model): {corr:.4f}")
    print(f"ML ensemble weights: sim={ml_weights[0]:.2f}  win_model={ml_weights[1]:.2f}")
    print(f"O/U ensemble weights: sim={ou_weights[0]:.2f}  lgb={ou_weights[1]:.2f}")

    print(f"\nKalshi ROI — Ensemble (edge > 5%):")
    if not roi_ens.empty:
        row5 = roi_ens[roi_ens["edge_threshold"] == 0.05]
        if not row5.empty:
            r = row5.iloc[0]
            print(f"  {int(r['n_bets'])} bets  |  Win rate {r['win_rate']:.1%}  |  "
                  f"ROI {r['roi_per_bet']:+.3f} units/bet")


if __name__ == "__main__":
    main()
