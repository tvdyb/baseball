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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from utils import FEATURES_DIR

# XGBoost hyperparameters (single source of truth — imported by predict.py, audit.py)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": 0,
}

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
    "diff_sp_transition_entropy",
    # OAA defense
    "diff_oaa_rate",
    # Team strength prior
    "diff_team_prior",
    # Context-aware SP xRV (home split for home SP, away split for away SP)
    "diff_sp_context_xrv",
    # Trade deadline features
    "diff_trade_net",
    "diff_trade_pitcher_xrv",
    # Preseason projections
    "diff_projected_wpct",
    "diff_sp_projected_era",
    "diff_sp_projected_war",
    # Roster-adjusted team prior
    "diff_adjusted_team_prior",
    # Pitcher stuff model scores
    "diff_sp_stuff_score",
    "diff_sp_location_score",
    "diff_sp_sequencing_score",
    "diff_sp_composite_score",
    # Hitter eval scores
    "diff_hit_swing_z",
    "diff_hit_iz_contact",
    "diff_hit_chase_contact",
    "diff_hit_foul_fight",
    "diff_hit_bip_iz",
]

RAW_FEATURES = [
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
    # SP confidence
    "home_sp_info_confidence",
    "away_sp_info_confidence",
]

ALL_FEATURES = DIFF_FEATURES + RAW_FEATURES

# Edge thresholds for ROI simulation (single source of truth)
EDGE_THRESHOLDS = [0.03, 0.05, 0.07, 0.10]

# Feature groups for ablation (single source of truth — imported by audit.py)
FEATURE_GROUPS = {
    "base_hitting": [
        "diff_hit_xrv_mean", "diff_hit_xrv_contact", "diff_hit_k_rate",
    ],
    "sp_quality": [
        "diff_sp_xrv_mean", "diff_sp_k_rate", "diff_sp_bb_rate",
        "diff_sp_avg_velo", "diff_sp_rest_days",
        "diff_sp_overperf", "diff_sp_overperf_recent",
        "diff_sp_context_xrv",
        "home_sp_pitch_mix_entropy", "away_sp_pitch_mix_entropy",
        "home_sp_n_pitches", "away_sp_n_pitches",
        "home_sp_info_confidence", "away_sp_info_confidence",
    ],
    "bullpen": [
        "diff_bp_xrv_mean", "diff_bp_fatigue_score",
        "diff_bp_matchup_xrv_mean", "diff_bp_arsenal_matchup_xrv_mean",
    ],
    "matchup_sparse": [
        "diff_matchup_xrv_mean", "diff_matchup_xrv_sum",
        "home_matchup_n_known", "away_matchup_n_known",
    ],
    "matchup_arsenal": [
        "diff_arsenal_matchup_xrv_mean", "diff_arsenal_matchup_xrv_sum",
        "home_arsenal_matchup_n_known", "away_arsenal_matchup_n_known",
    ],
    "weather": [
        "temperature", "wind_speed", "is_dome", "wind_out", "wind_in",
    ],
    "defense": [
        "diff_oaa_rate", "diff_def_xrv_delta",
    ],
    "team_prior": [
        "diff_team_prior", "diff_adjusted_team_prior",
    ],
    "projections": [
        "diff_projected_wpct", "diff_sp_projected_era", "diff_sp_projected_war",
    ],
    "sp_trends": [
        "diff_sp_velo_trend", "diff_sp_spin_trend",
        "diff_sp_xrv_trend", "diff_sp_transition_entropy",
    ],
    "trades": [
        "diff_trade_net", "diff_trade_pitcher_xrv",
    ],
    "context": [
        "days_into_season", "park_factor",
        "home_team_games_played", "away_team_games_played",
    ],
    "form": [
        "diff_recent_form",
    ],
    "platoon": [
        "diff_platoon_pct", "diff_sp_xrv_vs_lineup",
    ],
}

# Count-like columns that should be filled with median (not 0) for LR
_COUNT_FEATURES = {
    "home_sp_n_pitches", "away_sp_n_pitches",
    "home_matchup_n_known", "away_matchup_n_known",
    "home_arsenal_matchup_n_known", "away_arsenal_matchup_n_known",
}


def _smart_fillna(X: pd.DataFrame, train_medians: dict = None) -> tuple[pd.DataFrame, dict]:
    """Smart NaN filling for logistic regression.

    - diff_ columns: fill with 0 (assume equal when unknown)
    - Count columns (sp_n_pitches, matchup_n_known, etc.): fill with training median
    - Everything else: fill with 0

    Returns (filled_X, medians_dict) where medians_dict can be reused for test data.
    """
    X = X.copy()
    medians = {}

    for col in X.columns:
        if col.startswith("diff_"):
            X[col] = X[col].fillna(0)
        elif col in _COUNT_FEATURES:
            if train_medians and col in train_medians:
                med = train_medians[col]
            else:
                med = X[col].median()
                if pd.isna(med):
                    med = 0
            medians[col] = med
            X[col] = X[col].fillna(med)
        else:
            X[col] = X[col].fillna(0)

    return X, medians


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

    # --- Step 7: Early-season interaction features ---
    if "days_into_season" in X.columns:
        days = X["days_into_season"].fillna(30)
        # early_season_mask: 1.0 on Opening Day, 0.0 after 60 days
        early_mask = np.clip(1 - days / 60, 0, 1)

        # Interact key features with early-season mask
        for col in ["diff_sp_xrv_mean", "diff_hit_xrv_mean",
                     "diff_team_prior", "diff_projected_wpct"]:
            if col in X.columns:
                X[f"{col}_x_early"] = X[col].fillna(0) * early_mask

        # Blended prior: projections early, xRV later (z-scored to comparable scales)
        if "diff_projected_wpct" in X.columns and "diff_sp_xrv_mean" in X.columns:
            proj = X["diff_projected_wpct"].fillna(0)
            xrv = X["diff_sp_xrv_mean"].fillna(0)
            proj_z = (proj - proj.mean()) / (proj.std() + 1e-9)
            xrv_z = (xrv - xrv.mean()) / (xrv.std() + 1e-9)
            X["prior_dominance"] = early_mask * proj_z + (1 - early_mask) * xrv_z

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
                    "home_bp_fatigue_score", "away_bp_fatigue_score",
                    "days_into_season"]:
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
    X_train_filled, train_medians = _smart_fillna(X_train)
    X_train_s = scaler.fit_transform(X_train_filled)

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
        X_test_filled, _ = _smart_fillna(X_test, train_medians)
        X_test_s = scaler.transform(X_test_filled)
        test_prob = model.predict_proba(X_test_s)[:, 1]
        evaluate(y_test, test_prob, "Test (out-of-sample)")
        return model, scaler, test_prob

    return model, scaler, train_prob


def train_xgboost(X_train, y_train, X_test=None, y_test=None):
    """Train XGBoost model."""
    if not HAS_XGB:
        print("  XGBoost not available, skipping")
        return None, None, None

    # Split training: last 20% as validation for early stopping (chronological)
    n = len(X_train)
    val_size = int(n * 0.2)
    X_tr = X_train.iloc[:n - val_size]
    y_tr = y_train[:n - val_size]
    X_val = X_train.iloc[n - val_size:]
    y_val = y_train[n - val_size:]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, "train"), (dval, "val")]

    model = xgb.train(
        XGB_PARAMS, dtrain,
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

    # Report on full training set and test set
    dtrain_full = xgb.DMatrix(X_train, label=y_train)
    train_prob = model.predict(dtrain_full)
    evaluate(y_train, train_prob, "XGB Train")

    if X_test is not None:
        dtest = xgb.DMatrix(X_test, label=y_test)
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

        # Ensemble — learn optimal blend via K-fold OOF on training data
        if lr_probs is not None and xgb_probs is not None:
            from scipy.optimize import minimize_scalar

            tscv = TimeSeriesSplit(n_splits=5)  # forward-only time-series folds
            lr_oof = np.zeros(len(y_train))
            xgb_oof = np.zeros(len(y_train))

            for fold_train, fold_val in tscv.split(X_train_lr):
                # LR fold
                X_ft_filled, fm = _smart_fillna(X_train_lr.iloc[fold_train])
                X_fv_filled, _ = _smart_fillna(X_train_lr.iloc[fold_val], fm)
                sc_fold = StandardScaler()
                lr_fold = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
                lr_fold.fit(sc_fold.fit_transform(X_ft_filled), y_train[fold_train])
                lr_oof[fold_val] = lr_fold.predict_proba(sc_fold.transform(X_fv_filled))[:, 1]

                # XGB fold
                X_ft_xgb = X_train_xgb.iloc[fold_train]
                X_fv_xgb = X_train_xgb.iloc[fold_val]
                dt_fold = xgb.DMatrix(X_ft_xgb, label=y_train[fold_train])
                dv_fold = xgb.DMatrix(X_fv_xgb, label=y_train[fold_val])
                xgb_fold = xgb.train(XGB_PARAMS, dt_fold, num_boost_round=200,
                                      evals=[(dt_fold, "t"), (dv_fold, "v")],
                                      early_stopping_rounds=30, verbose_eval=0)
                xgb_oof[fold_val] = xgb_fold.predict(dv_fold)

            def blend_loss(w):
                blended = np.clip(w * lr_oof + (1 - w) * xgb_oof, 1e-6, 1 - 1e-6)
                return log_loss(y_train, blended)

            opt = minimize_scalar(blend_loss, bounds=(0.1, 0.9), method="bounded")
            w_lr = opt.x
            print(f"\n  Learned ensemble weight: LR={w_lr:.2f}, XGB={1-w_lr:.2f}")

            ens_probs = w_lr * lr_probs + (1 - w_lr) * xgb_probs
            result["ens_log_loss"] = log_loss(y_test, ens_probs)
            result["ens_auc"] = roc_auc_score(y_test, ens_probs)
            result["ens_brier"] = brier_score_loss(y_test, ens_probs)

            # Per-year betting analysis
            if ens_probs is not None:
                pnls = []
                for i in range(len(y_test)):
                    p = ens_probs[i]
                    if max(p, 1-p) >= 0.55:
                        bet_home = p >= 0.5
                        if bet_home:
                            pnl = 100 if y_test[i] == 1 else -100
                        else:
                            pnl = 100 if y_test[i] == 0 else -100
                        pnls.append(pnl)
                if pnls:
                    roi = sum(pnls) / (len(pnls) * 100)
                    wr = sum(1 for p in pnls if p > 0) / len(pnls)
                    result["n_bets"] = len(pnls)
                    result["roi_vs_baseline"] = roi
                    result["win_rate"] = wr

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
        if "roi_vs_baseline" in results_df.columns:
            print(f"\n  Per-Year Betting (confidence >= 55%):")
            for _, row in results_df.iterrows():
                if "n_bets" in row and pd.notna(row.get("n_bets")):
                    print(f"    {int(row['test_year'])}: {int(row['n_bets'])} bets, "
                          f"ROI={row['roi_vs_baseline']:+.1%}, "
                          f"WR={row['win_rate']:.1%}")

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
