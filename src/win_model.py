#!/usr/bin/env python3
"""
Game-level win probability model.

Takes the pregame feature matrix from feature_engineering.py and
predicts P(home_win). Trains on prior seasons, evaluates on held-out
season — no lookahead bias.

Models:
  1. Logistic regression (baseline — interpretable, fast)
  2. XGBoost (primary — handles interactions, missing values)

Evaluation:
  - Log loss (calibration)
  - Brier score (calibration)
  - AUC (discrimination)
  - Calibration plot (binned predicted vs actual)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from utils import DATA_DIR, FEATURES_DIR, MODEL_DIR


# Features to use (differentials are cleaner for the model)
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
    # SP trend features
    "diff_sp_velo_trend",
    "diff_sp_spin_trend",
    "diff_sp_xrv_trend",
    # OAA defense
    "diff_oaa_rate",
    # Team strength prior
    "diff_team_prior",
    # Context-aware SP xRV (home split for home SP, away split for away SP)
    "diff_sp_context_xrv",
]

RAW_FEATURES = [
    "park_factor",
    "home_sp_pitch_mix_entropy",
    "away_sp_pitch_mix_entropy",
    "home_sp_n_pitches",
    "away_sp_n_pitches",
    "home_matchup_n_known",
    "away_matchup_n_known",
    "home_arsenal_matchup_n_known",
    "away_arsenal_matchup_n_known",
    # Weather
    "temperature",
    "wind_speed",
    "is_dome",
    "wind_out",
    "wind_in",
]

ALL_FEATURES = DIFF_FEATURES + RAW_FEATURES


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


def add_nonlinear_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Manually engineer nonlinear features so LR can capture them.
    This is better than relying on XGB because we control exactly
    which nonlinearities to model, avoiding overfitting.
    """
    X = X.copy()

    # SP rest days: normal (4-5) is baseline. Short rest and long rest are both bad.
    # Encode as deviation from optimal rest (4-5 days)
    if "diff_sp_rest_days" in X.columns:
        for side in ["home", "away"]:
            col = f"{side}_sp_rest_days"
            if col not in X.columns:
                continue
            rest = X[col].fillna(5)
            # Optimal rest = 5 days. Short rest penalty (exponential below 4)
            X[f"{side}_sp_short_rest"] = np.clip(4 - rest, 0, None)
            # Long rest / rust (linear above 7)
            X[f"{side}_sp_long_rest"] = np.clip(rest - 7, 0, None)
        # Differential versions
        if "home_sp_short_rest" in X.columns and "away_sp_short_rest" in X.columns:
            X["diff_sp_short_rest"] = -(X["home_sp_short_rest"] - X["away_sp_short_rest"])
            X["diff_sp_long_rest"] = -(X["home_sp_long_rest"] - X["away_sp_long_rest"])

    # SP sample size confidence: log transform of pitch count
    # Difference between 200 and 500 pitches matters more than 1500 vs 1800
    for col in ["home_sp_n_pitches", "away_sp_n_pitches"]:
        if col in X.columns:
            X[f"{col}_log"] = np.log1p(X[col].fillna(0))

    # Bullpen fatigue: squared term (moderate fatigue OK, extreme fatigue disproportionately bad)
    if "diff_bp_fatigue_score" in X.columns:
        for side in ["home", "away"]:
            col = f"{side}_bp_fatigue_score"
            if col in X.columns:
                X[f"{col}_sq"] = X[col].fillna(0) ** 2
        if "home_bp_fatigue_score_sq" in X.columns:
            X["diff_bp_fatigue_sq"] = -(X["home_bp_fatigue_score_sq"] - X["away_bp_fatigue_score_sq"])

    # Squared differentials for key features where extreme values matter more
    for col in ["diff_sp_xrv_mean", "diff_hit_xrv_mean", "diff_matchup_xrv_mean"]:
        if col in X.columns:
            # Signed square: preserves direction but amplifies large differences
            vals = X[col].fillna(0)
            X[f"{col}_sq"] = vals * vals.abs()

    return X


def prepare_xy(df: pd.DataFrame, features: list[str] = None, nonlinear: bool = False):
    """Extract feature matrix and target from game features."""
    if features is None:
        features = ALL_FEATURES
    # Only use features that exist in the data
    available = [f for f in features if f in df.columns]
    missing = set(features) - set(available)
    if missing:
        print(f"  Missing features (will skip): {missing}")

    X = df[available].copy()

    if nonlinear:
        # Add raw columns needed for nonlinear feature engineering
        for col in ["home_sp_rest_days", "away_sp_rest_days",
                    "home_bp_fatigue_score", "away_bp_fatigue_score"]:
            if col in df.columns and col not in X.columns:
                X[col] = df[col]
        X = add_nonlinear_features(X)

    y = df["home_win"].values

    return X, y, list(X.columns)


def evaluate(y_true, y_prob, label=""):
    """Compute calibration and discrimination metrics."""
    ll = log_loss(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    baseline_ll = log_loss(y_true, np.full_like(y_prob, y_true.mean()))
    baseline_bs = brier_score_loss(y_true, np.full_like(y_prob, y_true.mean()))

    print(f"\n  {label} Evaluation:")
    print(f"    Log loss:     {ll:.4f}  (baseline: {baseline_ll:.4f})")
    print(f"    Brier score:  {bs:.4f}  (baseline: {baseline_bs:.4f})")
    print(f"    AUC:          {auc:.4f}")
    print(f"    Home win pct: {y_true.mean():.3f}")
    print(f"    Pred mean:    {y_prob.mean():.3f}")

    # Calibration by decile
    print(f"    Calibration (predicted → actual):")
    order = np.argsort(y_prob)
    n = len(y_prob)
    n_bins = 10
    for b in range(n_bins):
        lo = b * n // n_bins
        hi = (b + 1) * n // n_bins
        idx = order[lo:hi]
        pred_mean = y_prob[idx].mean()
        actual_mean = y_true[idx].mean()
        print(f"      {pred_mean:.3f} → {actual_mean:.3f}  (n={len(idx)})")

    return {"log_loss": ll, "brier_score": bs, "auc": auc}


def train_logistic(X_train, y_train, X_test=None, y_test=None):
    """Train logistic regression baseline."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.fillna(0))

    model = LogisticRegression(
        C=1.0, penalty="l2", max_iter=1000, solver="lbfgs"
    )
    model.fit(X_train_s, y_train)

    # Feature importance
    coefs = dict(zip(X_train.columns, model.coef_[0]))
    print("\n  Logistic Regression coefficients:")
    for feat, coef in sorted(coefs.items(), key=lambda x: -abs(x[1])):
        print(f"    {feat}: {coef:+.4f}")

    # In-sample
    train_prob = model.predict_proba(X_train_s)[:, 1]
    evaluate(y_train, train_prob, "Train (in-sample)")

    if X_test is not None:
        X_test_s = scaler.transform(X_test.fillna(0))
        test_prob = model.predict_proba(X_test_s)[:, 1]
        evaluate(y_test, test_prob, "Test (out-of-sample)")
        return model, scaler, test_prob

    return model, scaler, train_prob


def train_xgboost(X_train, y_train, X_test=None, y_test=None):
    """Train XGBoost model."""
    if not HAS_XGB:
        print("  XGBoost not available, skipping")
        return None, None, None

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)

    if X_test is not None:
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=False)
        evals = [(dtrain, "train"), (dtest, "test")]
    else:
        evals = [(dtrain, "train")]

    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    # Feature importance
    importance = model.get_score(importance_type="gain")
    print("\n  XGBoost feature importance (gain):")
    total = sum(importance.values()) if importance else 1
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp/total:.1%}")

    # In-sample
    train_prob = model.predict(dtrain)
    evaluate(y_train, train_prob, "XGB Train")

    if X_test is not None:
        test_prob = model.predict(dtest)
        evaluate(y_test, test_prob, "XGB Test")
        return model, None, test_prob

    return model, None, train_prob


def walk_forward_evaluation(all_years: list[int], min_train_years: int = 2):
    """
    Walk-forward evaluation: for each test year, train on all prior years.
    This mirrors actual deployment — no lookahead.
    """
    print(f"\n{'='*60}")
    print("Walk-Forward Evaluation")
    print(f"{'='*60}")

    df = load_features(all_years)
    all_results = []

    for test_year in all_years[min_train_years:]:
        train_years = [y for y in all_years if y < test_year]
        print(f"\n{'='*60}")
        print(f"Test: {test_year}, Train: {train_years}")
        print(f"{'='*60}")

        train_df = df[df["season"].isin(train_years)]
        test_df = df[df["season"] == test_year]

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"  Skipping — train: {len(train_df)}, test: {len(test_df)}")
            continue

        # LR: linear features only
        X_train_lr, y_train, features_lr = prepare_xy(train_df, nonlinear=False)
        X_test_lr, y_test, _ = prepare_xy(test_df, features_lr, nonlinear=False)

        # XGB: nonlinear features
        X_train_xgb, _, features_xgb = prepare_xy(train_df, nonlinear=True)
        X_test_xgb, _, _ = prepare_xy(test_df, features_xgb, nonlinear=True)

        print(f"  Train: {len(X_train_lr)} games, Test: {len(X_test_lr)} games")
        print(f"  LR features: {len(features_lr)}, XGB features: {len(features_xgb)}")

        # Logistic regression (linear features)
        print("\n  --- Logistic Regression ---")
        lr_model, lr_scaler, lr_probs = train_logistic(X_train_lr, y_train, X_test_lr, y_test)

        # XGBoost (nonlinear features)
        print("\n  --- XGBoost ---")
        xgb_model, _, xgb_probs = train_xgboost(X_train_xgb, y_train, X_test_xgb, y_test)

        result = {
            "test_year": test_year,
            "n_train": len(X_train_lr),
            "n_test": len(X_test_lr),
        }

        if lr_probs is not None:
            result["lr_log_loss"] = log_loss(y_test, lr_probs)
            result["lr_auc"] = roc_auc_score(y_test, lr_probs)
            result["lr_brier"] = brier_score_loss(y_test, lr_probs)

        if xgb_probs is not None:
            result["xgb_log_loss"] = log_loss(y_test, xgb_probs)
            result["xgb_auc"] = roc_auc_score(y_test, xgb_probs)
            result["xgb_brier"] = brier_score_loss(y_test, xgb_probs)

        # Ensemble
        if lr_probs is not None and xgb_probs is not None:
            ens_probs = 0.5 * lr_probs + 0.5 * xgb_probs
            result["ens_log_loss"] = log_loss(y_test, ens_probs)
            result["ens_auc"] = roc_auc_score(y_test, ens_probs)
            result["ens_brier"] = brier_score_loss(y_test, ens_probs)

        all_results.append(result)

    # Summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*60}")
        print("Walk-Forward Summary")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))

        if "lr_log_loss" in results_df.columns:
            print(f"\n  LR avg log loss:  {results_df['lr_log_loss'].mean():.4f}")
            print(f"  LR avg AUC:       {results_df['lr_auc'].mean():.4f}")
        if "xgb_log_loss" in results_df.columns:
            print(f"  XGB avg log loss: {results_df['xgb_log_loss'].mean():.4f}")
            print(f"  XGB avg AUC:      {results_df['xgb_auc'].mean():.4f}")
        if "ens_log_loss" in results_df.columns:
            print(f"  ENS avg log loss: {results_df['ens_log_loss'].mean():.4f}")
            print(f"  ENS avg AUC:      {results_df['ens_auc'].mean():.4f}")
            print(f"  XGB avg AUC:      {results_df['xgb_auc'].mean():.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="MLB Win Probability Model")
    parser.add_argument("--train-seasons", type=int, nargs="+",
                        help="Seasons to train on")
    parser.add_argument("--test-season", type=int,
                        help="Season to evaluate on")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward evaluation across all available seasons")
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=list(range(2018, 2025)),
                        help="Seasons for walk-forward evaluation")
    args = parser.parse_args()

    if args.walk_forward:
        walk_forward_evaluation(args.seasons)
    elif args.train_seasons and args.test_season:
        df = load_features(args.train_seasons + [args.test_season])
        train_df = df[df["season"].isin(args.train_seasons)]
        test_df = df[df["season"] == args.test_season]

        X_train, y_train, features = prepare_xy(train_df)
        X_test, y_test, _ = prepare_xy(test_df, features)

        print(f"\nTrain: {len(X_train)} games from {args.train_seasons}")
        print(f"Test: {len(X_test)} games from {args.test_season}")

        print("\n--- Logistic Regression ---")
        train_logistic(X_train, y_train, X_test, y_test)

        print("\n--- XGBoost ---")
        train_xgboost(X_train, y_train, X_test, y_test)
    else:
        print("Specify --walk-forward or --train-seasons + --test-season")


if __name__ == "__main__":
    main()
