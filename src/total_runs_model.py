#!/usr/bin/env python3
"""
Total game runs regression model.

Predicts total_runs = home_score + away_score using the pregame feature
matrix from feature_engineering.py.  Complements the MC simulator by
providing a fast, gradient-boosted point estimate + distributional
information for over/under betting.

Walk-forward design:
  - Train on 2024, predict 2025 (cross-season)
  - Expanding window within 2025 (train on early 2025, predict later)

Uses LightGBM regression with both point (L2) and quantile objectives.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import pearsonr, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FEATURES_DIR, DATA_DIR

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

# Differential features (same as win model — measure home-away gap)
DIFF_FEATURES = [
    "diff_sp_xrv_mean",
    "diff_sp_k_rate",
    "diff_sp_bb_rate",
    "diff_sp_avg_velo",
    "diff_sp_rest_days",
    "diff_sp_overperf",
    "diff_sp_overperf_recent",
    "diff_bp_xrv_mean",
    "diff_bp_fatigue_score",
    "diff_bp_matchup_xrv_mean",
    "diff_hit_xrv_mean",
    "diff_hit_xrv_contact",
    "diff_hit_k_rate",
    "diff_def_xrv_delta",
    "diff_matchup_xrv_mean",
    "diff_matchup_xrv_sum",
    "diff_arsenal_matchup_xrv_mean",
    "diff_arsenal_matchup_xrv_sum",
    "diff_bp_arsenal_matchup_xrv_mean",
    "diff_platoon_pct",
    "diff_recent_form",
    "diff_sp_xrv_vs_lineup",
    "diff_sp_velo_trend",
    "diff_sp_spin_trend",
    "diff_sp_xrv_trend",
    "diff_sp_transition_entropy",
    "diff_oaa_rate",
    "diff_team_prior",
    "diff_sp_context_xrv",
    "diff_trade_net",
    "diff_trade_pitcher_xrv",
    "diff_projected_wpct",
    "diff_sp_projected_era",
    "diff_sp_projected_war",
    "diff_adjusted_team_prior",
    "diff_sp_stuff_score",
    "diff_sp_location_score",
    "diff_sp_sequencing_score",
    "diff_sp_composite_score",
    "diff_hit_swing_z",
    "diff_hit_iz_contact",
    "diff_hit_chase_contact",
    "diff_hit_foul_fight",
    "diff_hit_bip_iz",
]

# Non-differential (level) features relevant to TOTAL runs
# These affect total scoring environment, not just home-away gap
LEVEL_FEATURES = [
    # Park & weather
    "park_factor",
    "temperature",
    "wind_speed",
    "is_dome",
    "wind_out",
    "wind_in",
    # Both SPs — total runs depends on quality of BOTH pitchers
    "home_sp_xrv_mean",
    "away_sp_xrv_mean",
    "home_sp_k_rate",
    "away_sp_k_rate",
    "home_sp_bb_rate",
    "away_sp_bb_rate",
    "home_sp_composite_score",
    "away_sp_composite_score",
    # Both lineups
    "home_hit_xrv_mean",
    "away_hit_xrv_mean",
    "home_hit_k_rate",
    "away_hit_k_rate",
    # Both bullpens
    "home_bp_xrv_mean",
    "away_bp_xrv_mean",
    "home_bp_fatigue_score",
    "away_bp_fatigue_score",
    # Sample size / confidence
    "home_sp_n_pitches",
    "away_sp_n_pitches",
    "home_sp_info_confidence",
    "away_sp_info_confidence",
    "home_sp_pitch_mix_entropy",
    "away_sp_pitch_mix_entropy",
    # Season context
    "days_into_season",
    "home_team_games_played",
    "away_team_games_played",
]

# SUM features: for total runs, the SUM of both teams' offensive/pitching
# quality matters more than the DIFF
SUM_FEATURES_TO_CREATE = [
    # (feature_base, home_col, away_col) — we create home+away sum
    ("sum_sp_xrv", "home_sp_xrv_mean", "away_sp_xrv_mean"),
    ("sum_sp_k_rate", "home_sp_k_rate", "away_sp_k_rate"),
    ("sum_sp_bb_rate", "home_sp_bb_rate", "away_sp_bb_rate"),
    ("sum_sp_composite", "home_sp_composite_score", "away_sp_composite_score"),
    ("sum_hit_xrv", "home_hit_xrv_mean", "away_hit_xrv_mean"),
    ("sum_hit_k_rate", "home_hit_k_rate", "away_hit_k_rate"),
    ("sum_bp_xrv", "home_bp_xrv_mean", "away_bp_xrv_mean"),
    ("sum_bp_fatigue", "home_bp_fatigue_score", "away_bp_fatigue_score"),
]

ALL_FEATURES = DIFF_FEATURES + LEVEL_FEATURES

# LightGBM hyperparameters
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}

LGB_QUANTILE_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------------

def load_features(years: list[int]) -> pd.DataFrame:
    """Load and concatenate game features for multiple years."""
    frames = []
    for year in years:
        path = FEATURES_DIR / f"game_features_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["season"] = year
            frames.append(df)
            print(f"  Loaded {year}: {len(df)} games")
        else:
            print(f"  {path} not found, skipping")
    if not frames:
        raise FileNotFoundError("No feature files found")
    return pd.concat(frames, ignore_index=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build feature matrix for total runs prediction.

    Returns (X, feature_names).
    """
    X = pd.DataFrame(index=df.index)

    # Differential features
    for col in DIFF_FEATURES:
        if col in df.columns:
            X[col] = df[col]

    # Level features
    for col in LEVEL_FEATURES:
        if col in df.columns:
            X[col] = df[col]

    # Create SUM features (total pitching/hitting environment)
    for name, home_col, away_col in SUM_FEATURES_TO_CREATE:
        if home_col in df.columns and away_col in df.columns:
            X[name] = df[home_col].fillna(0) + df[away_col].fillna(0)

    # Fill NaNs: diffs -> 0, counts -> median, rest -> 0
    for col in X.columns:
        if col.startswith("diff_"):
            X[col] = X[col].fillna(0)
        elif col in ("home_sp_n_pitches", "away_sp_n_pitches"):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(0)

    feature_names = list(X.columns)
    return X, feature_names


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lgb_point(X_train: pd.DataFrame, y_train: np.ndarray,
                    X_val: pd.DataFrame = None, y_val: np.ndarray = None,
                    ) -> lgb.Booster:
    """Train LightGBM point regression (L2/MAE)."""
    dtrain = lgb.Dataset(X_train, label=y_train)
    callbacks = [lgb.log_evaluation(period=0)]  # silent

    if X_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=500,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
    else:
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=300,
                          callbacks=callbacks)
    return model


def train_lgb_quantile(X_train: pd.DataFrame, y_train: np.ndarray,
                       alpha: float,
                       X_val: pd.DataFrame = None,
                       y_val: np.ndarray = None) -> lgb.Booster:
    """Train LightGBM quantile regression at the given alpha."""
    params = LGB_QUANTILE_PARAMS.copy()
    params["alpha"] = alpha
    dtrain = lgb.Dataset(X_train, label=y_train)

    if X_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
    else:
        model = lgb.train(params, dtrain, num_boost_round=300,
                          callbacks=[lgb.log_evaluation(period=0)])
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         label: str = "") -> dict:
    """Compute regression metrics for total runs predictions."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    corr, pval = pearsonr(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    print(f"\n  {label}")
    print(f"    MAE:         {mae:.3f}")
    print(f"    RMSE:        {rmse:.3f}")
    print(f"    Correlation: {corr:.4f}  (p={pval:.2e})")
    print(f"    Bias:        {bias:+.3f}")
    print(f"    Actual mean: {y_true.mean():.2f},  Pred mean: {y_pred.mean():.2f}")

    return {"mae": mae, "rmse": rmse, "corr": corr, "bias": bias}


def evaluate_ou(y_true: np.ndarray, y_pred: np.ndarray,
                lines: list[float] = None,
                label: str = "",
                p_over: np.ndarray = None) -> dict:
    """
    Evaluate over/under accuracy at fixed lines.

    If p_over is provided, use it as P(total > line).
    Otherwise derive from point prediction + Gaussian residual.
    """
    if lines is None:
        lines = [7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

    # Estimate residual std from data if no quantile model
    residual_std = np.std(y_true - y_pred)

    print(f"\n  {label} O/U Accuracy:")
    results = {}
    for line in lines:
        actual_over = y_true > line
        actual_under = y_true < line
        pushes = y_true == line
        non_push = ~pushes

        if non_push.sum() == 0:
            continue

        # Predict over/under
        if p_over is not None:
            pred_over = p_over > 0.5
        else:
            # Use Gaussian: P(total > line) = 1 - Phi((line - pred) / std)
            p = 1.0 - norm.cdf(line, loc=y_pred, scale=residual_std)
            pred_over = p > 0.5

        # Accuracy on non-push games
        correct = ((pred_over & actual_over) | (~pred_over & actual_under))[non_push]
        accuracy = correct.mean()
        n_games = non_push.sum()
        n_over = actual_over.sum()
        n_push = pushes.sum()

        print(f"    Line {line:>4.1f}: {accuracy:.1%} "
              f"({correct.sum()}/{n_games} non-push, "
              f"{n_over} over, {n_push} push)")
        results[f"ou_{line}_acc"] = accuracy
        results[f"ou_{line}_n"] = n_games

    return results


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward(verbose: bool = True) -> pd.DataFrame:
    """
    Walk-forward evaluation:
      1. Train on 2024, predict all 2025.
      2. Expanding window within 2025 (monthly cutoffs).
    """
    print("=" * 70)
    print("TOTAL RUNS REGRESSION MODEL — Walk-Forward Evaluation")
    print("=" * 70)

    # Load data
    df = load_features([2024, 2025])
    df["total_runs"] = df["home_score"] + df["away_score"]
    df["game_date"] = pd.to_datetime(df["game_date"])

    train_2024 = df[df["season"] == 2024].sort_values("game_date").copy()
    test_2025 = df[df["season"] == 2025].sort_values("game_date").copy()

    print(f"\n  2024 games: {len(train_2024)}")
    print(f"  2025 games: {len(test_2025)}")
    print(f"  2024 total runs: mean={train_2024['total_runs'].mean():.2f}, "
          f"std={train_2024['total_runs'].std():.2f}")
    print(f"  2025 total runs: mean={test_2025['total_runs'].mean():.2f}, "
          f"std={test_2025['total_runs'].std():.2f}")

    # Prepare features
    X_train_24, feat_names = prepare_features(train_2024)
    y_train_24 = train_2024["total_runs"].values

    X_test_25, _ = prepare_features(test_2025)
    # Ensure same columns
    for col in feat_names:
        if col not in X_test_25.columns:
            X_test_25[col] = 0
    X_test_25 = X_test_25[feat_names]
    y_test_25 = test_2025["total_runs"].values

    print(f"  Features: {len(feat_names)}")

    # ------------------------------------------------------------------
    # Model 1: Train 2024 → Predict 2025 (point regression)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL 1: Train 2024 → Predict 2025 (LightGBM Regression)")
    print("=" * 70)

    # Hold out last 20% of 2024 for early stopping
    n_24 = len(X_train_24)
    val_size = int(n_24 * 0.2)
    X_tr = X_train_24.iloc[:n_24 - val_size]
    y_tr = y_train_24[:n_24 - val_size]
    X_vl = X_train_24.iloc[n_24 - val_size:]
    y_vl = y_train_24[n_24 - val_size:]

    model_point = train_lgb_point(X_tr, y_tr, X_vl, y_vl)
    pred_25 = model_point.predict(X_test_25)

    # Evaluate
    metrics_point = evaluate_predictions(y_test_25, pred_25,
                                         "2024→2025 Point Regression")
    ou_metrics = evaluate_ou(y_test_25, pred_25, label="2024→2025 Point")

    # Feature importance
    importance = model_point.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_names, importance), key=lambda x: -x[1])
    total_imp = sum(importance)
    print("\n  Top 20 features (gain):")
    for feat, imp in feat_imp[:20]:
        print(f"    {feat}: {imp / total_imp:.1%}")

    # ------------------------------------------------------------------
    # Model 2: Quantile regression for P(over)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL 2: Quantile Regression (multiple quantiles)")
    print("=" * 70)

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    quantile_models = {}
    quantile_preds = {}
    for q in quantiles:
        qmodel = train_lgb_quantile(X_tr, y_tr, alpha=q, X_val=X_vl, y_val=y_vl)
        quantile_models[q] = qmodel
        quantile_preds[q] = qmodel.predict(X_test_25)

    # Use quantile predictions to estimate P(over) at each line
    # Interpolate from the quantile predictions
    print("\n  Quantile predictions sample (first 10 games):")
    print(f"    {'Actual':>7s}", end="")
    for q in quantiles:
        print(f"  Q{q:.0%}".rjust(8), end="")
    print(f"  {'Point':>7s}")
    for i in range(min(10, len(y_test_25))):
        print(f"    {y_test_25[i]:7.0f}", end="")
        for q in quantiles:
            print(f"  {quantile_preds[q][i]:7.2f}", end="")
        print(f"  {pred_25[i]:7.2f}")

    # Derive P(over line) from quantile interpolation
    lines = [7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    print("\n  O/U via Quantile Interpolation:")
    for line in lines:
        # For each game, estimate P(total > line) by interpolating quantiles
        q_vals = np.array(quantiles)
        p_over_arr = np.zeros(len(y_test_25))
        for i in range(len(y_test_25)):
            preds_i = np.array([quantile_preds[q][i] for q in quantiles])
            # If line < smallest quantile pred, P(over) is high
            # If line > largest quantile pred, P(over) is low
            if line <= preds_i[0]:
                p_over_arr[i] = 1.0 - q_vals[0] * 0.5  # extrapolate
            elif line >= preds_i[-1]:
                p_over_arr[i] = (1.0 - q_vals[-1]) * 0.5  # extrapolate
            else:
                # Interpolate: find where line falls in quantile predictions
                p_over_arr[i] = 1.0 - np.interp(line, preds_i, q_vals)

        actual_over = y_test_25 > line
        actual_under = y_test_25 < line
        pushes = y_test_25 == line
        non_push = ~pushes
        pred_over = p_over_arr > 0.5
        if non_push.sum() > 0:
            correct = ((pred_over & actual_over) | (~pred_over & actual_under))[non_push]
            accuracy = correct.mean()
            print(f"    Line {line:>4.1f}: {accuracy:.1%} "
                  f"({correct.sum()}/{non_push.sum()} non-push)")

    # ------------------------------------------------------------------
    # Model 3: Expanding window within 2025
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL 3: Expanding Window within 2025")
    print("=" * 70)

    # Use 2024 + early 2025 to predict later 2025
    all_data = pd.concat([train_2024, test_2025]).sort_values("game_date")
    X_all, _ = prepare_features(all_data)
    for col in feat_names:
        if col not in X_all.columns:
            X_all[col] = 0
    X_all = X_all[feat_names]
    y_all = all_data["total_runs"].values
    dates_all = all_data["game_date"].values

    # Monthly cutoffs in 2025
    cutoffs = pd.to_datetime(["2025-04-01", "2025-05-01", "2025-06-01",
                               "2025-07-01", "2025-08-01"])
    # Only use cutoffs where we have data on both sides
    max_date = pd.Timestamp(all_data["game_date"].max())
    cutoffs = [c for c in cutoffs if c < max_date]

    expanding_results = []
    for cutoff in cutoffs:
        train_mask = dates_all < np.datetime64(cutoff)
        test_mask = dates_all >= np.datetime64(cutoff)
        if train_mask.sum() < 100 or test_mask.sum() < 50:
            continue

        X_tr_exp = X_all[train_mask]
        y_tr_exp = y_all[train_mask]
        X_te_exp = X_all[test_mask]
        y_te_exp = y_all[test_mask]

        # Val: last 20% of training
        n_tr = len(X_tr_exp)
        val_n = int(n_tr * 0.2)
        model_exp = train_lgb_point(
            X_tr_exp.iloc[:n_tr - val_n], y_tr_exp[:n_tr - val_n],
            X_tr_exp.iloc[n_tr - val_n:], y_tr_exp[n_tr - val_n:]
        )
        pred_exp = model_exp.predict(X_te_exp)

        m = evaluate_predictions(y_te_exp, pred_exp,
                                 f"Expanding to {cutoff.strftime('%Y-%m-%d')}")
        ou_m = evaluate_ou(y_te_exp, pred_exp,
                           label=f"Expanding to {cutoff.strftime('%Y-%m-%d')}")
        m.update(ou_m)
        m["cutoff"] = cutoff
        m["n_train"] = train_mask.sum()
        m["n_test"] = test_mask.sum()
        expanding_results.append(m)

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING PREDICTIONS")
    print("=" * 70)

    out_df = test_2025[["game_pk", "game_date", "home_team", "away_team",
                         "home_score", "away_score", "total_runs"]].copy()
    out_df["lgb_pred_total"] = pred_25
    out_df["lgb_residual_std"] = np.std(y_test_25 - pred_25)

    # Add quantile predictions
    for q in quantiles:
        out_df[f"lgb_q{int(q*100):02d}"] = quantile_preds[q]

    # P(over) at standard lines using Gaussian from point model
    residual_std = np.std(y_test_25 - pred_25)
    for line in [7.5, 8.0, 8.5, 9.0, 9.5, 10.0]:
        out_df[f"lgb_p_over_{line}"] = 1.0 - norm.cdf(
            line, loc=pred_25, scale=residual_std
        )

    out_path = DATA_DIR / "backtest" / "total_runs_lgb_2025.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"  Saved {len(out_df)} predictions to {out_path}")

    return out_df


# ---------------------------------------------------------------------------
# Ensemble with MC sim
# ---------------------------------------------------------------------------

def ensemble_with_mc_sim():
    """
    Blend LGB regression predictions with MC simulator predictions.
    Reports which blend weight gives best O/U accuracy.
    """
    print("\n" + "=" * 70)
    print("ENSEMBLE: LGB Regression + MC Simulator")
    print("=" * 70)

    # Load predictions
    lgb_path = DATA_DIR / "backtest" / "total_runs_lgb_2025.parquet"
    mc_path = DATA_DIR / "backtest" / "nrfi_ou_backtest_2025.parquet"

    if not lgb_path.exists():
        print("  LGB predictions not found — run walk_forward() first")
        return
    if not mc_path.exists():
        print("  MC backtest not found")
        return

    lgb_df = pd.read_parquet(lgb_path)
    mc_df = pd.read_parquet(mc_path)

    print(f"  LGB predictions: {len(lgb_df)} games")
    print(f"  MC predictions:  {len(mc_df)} games")

    # Merge on game_pk
    merged = mc_df.merge(
        lgb_df[["game_pk", "lgb_pred_total", "lgb_residual_std"]],
        on="game_pk", how="inner"
    )
    print(f"  Matched games:   {len(merged)}")

    if len(merged) == 0:
        print("  No matching games found!")
        return

    actual = merged["actual_total"].values
    sim_pred = merged["sim_total_mean"].values
    lgb_pred = merged["lgb_pred_total"].values
    sim_line = merged["sim_line"].values

    # Individual model metrics
    print("\n  --- MC Simulator alone ---")
    evaluate_predictions(actual, sim_pred, "MC Sim")

    print("\n  --- LGB Regression alone ---")
    evaluate_predictions(actual, lgb_pred, "LGB Regression")

    # Blend at various weights
    print("\n  --- Blend Weights Search ---")
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    blend_results = []

    for w_lgb in weights:
        w_sim = 1.0 - w_lgb
        blended = w_lgb * lgb_pred + w_sim * sim_pred
        mae = np.mean(np.abs(actual - blended))
        rmse = np.sqrt(np.mean((actual - blended) ** 2))
        corr, _ = pearsonr(actual, blended)

        # O/U accuracy at each game's actual sim_line
        residual_std = np.std(actual - blended)
        line_results = {}
        for line in [7.5, 8.0, 8.5, 9.0, 9.5]:
            mask_line = sim_line == line
            if mask_line.sum() < 5:
                continue
            a = actual[mask_line]
            b = blended[mask_line]
            over = a > line
            under = a < line
            push = a == line
            non_push = ~push
            if non_push.sum() == 0:
                continue
            pred_over = b > line
            correct = ((pred_over & over) | (~pred_over & under))[non_push]
            line_results[f"ou_{line}"] = correct.mean()

        # Overall O/U accuracy using each game's own line
        over = actual > sim_line
        under = actual < sim_line
        push = actual == sim_line
        non_push = ~push
        if non_push.sum() > 0:
            pred_over = blended > sim_line
            correct_overall = ((pred_over & over) | (~pred_over & under))[non_push]
            ou_overall = correct_overall.mean()
        else:
            ou_overall = np.nan

        result = {
            "w_lgb": w_lgb,
            "w_sim": w_sim,
            "mae": mae,
            "rmse": rmse,
            "corr": corr,
            "ou_vs_line": ou_overall,
        }
        result.update(line_results)
        blend_results.append(result)

    blend_df = pd.DataFrame(blend_results)
    print("\n  Blend results:")
    print(f"    {'w_LGB':>6s} {'w_SIM':>6s} {'MAE':>6s} {'RMSE':>6s} "
          f"{'Corr':>7s} {'O/U%':>6s}")
    for _, row in blend_df.iterrows():
        print(f"    {row['w_lgb']:6.1f} {row['w_sim']:6.1f} "
              f"{row['mae']:6.3f} {row['rmse']:6.3f} "
              f"{row['corr']:7.4f} {row['ou_vs_line']:6.1%}")

    # Best blend
    best_idx = blend_df["ou_vs_line"].idxmax()
    best = blend_df.iloc[best_idx]
    print(f"\n  Best O/U accuracy: {best['ou_vs_line']:.1%} "
          f"at w_LGB={best['w_lgb']:.1f}, w_SIM={best['w_sim']:.1f}")

    best_corr_idx = blend_df["corr"].idxmax()
    best_corr = blend_df.iloc[best_corr_idx]
    print(f"  Best correlation:  {best_corr['corr']:.4f} "
          f"at w_LGB={best_corr['w_lgb']:.1f}, w_SIM={best_corr['w_sim']:.1f}")

    best_mae_idx = blend_df["mae"].idxmin()
    best_mae = blend_df.iloc[best_mae_idx]
    print(f"  Best MAE:          {best_mae['mae']:.3f} "
          f"at w_LGB={best_mae['w_lgb']:.1f}, w_SIM={best_mae['w_sim']:.1f}")

    return blend_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Total Runs Regression Model (LightGBM)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run walk-forward evaluation")
    parser.add_argument("--ensemble", action="store_true",
                        help="Ensemble with MC sim backtest")
    parser.add_argument("--all", action="store_true",
                        help="Run evaluation + ensemble")
    args = parser.parse_args()

    if args.all or args.evaluate:
        walk_forward()
    if args.all or args.ensemble:
        ensemble_with_mc_sim()
    if not (args.all or args.evaluate or args.ensemble):
        print("Specify --evaluate, --ensemble, or --all")


if __name__ == "__main__":
    main()
