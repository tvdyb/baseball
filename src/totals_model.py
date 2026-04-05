#!/usr/bin/env python3
"""
Totals (over/under) prediction model for MLB games.

Predicts expected total runs using a Ridge + XGBoost regression ensemble,
trained separately on home and away runs then summed. Converts to P(over line)
via the empirical CDF of cross-validated residuals.

Key improvements over v1:
  - Batted ball quality features (barrel rate, hard hit rate)
  - Nonlinear feature engineering (rest penalties, fatigue interactions)
  - TimeSeriesSplit (no future leakage) + minimize_scalar for blend weight
  - Separate home/away run models for asymmetric effects

Usage:
    python src/totals_model.py --backtest 2025
    python src/totals_model.py --date 2026-04-04
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import FEATURES_DIR

# Features for totals prediction — both sides' absolute quality matters.
TOTALS_FEATURES = [
    # Park and environment
    "park_factor",
    "temperature",
    "wind_speed",
    "wind_out",
    "wind_in",
    "is_dome",
    "is_night",
    # Home SP quality
    "home_sp_xrv_mean",
    "home_sp_k_rate",
    "home_sp_bb_rate",
    "home_sp_xrv_std",
    # Away SP quality
    "away_sp_xrv_mean",
    "away_sp_k_rate",
    "away_sp_bb_rate",
    "away_sp_xrv_std",
    # Home offense
    "home_hit_xrv_mean",
    "home_hit_xrv_contact",
    "home_hit_k_rate",
    # Away offense
    "away_hit_xrv_mean",
    "away_hit_xrv_contact",
    "away_hit_k_rate",
    # Batted ball quality — offense
    "home_hit_hard_hit_rate",
    "home_hit_barrel_rate",
    "away_hit_hard_hit_rate",
    "away_hit_barrel_rate",
    # Batted ball quality — pitching allowed
    "home_pitch_hard_hit_rate",
    "home_pitch_barrel_rate",
    "away_pitch_hard_hit_rate",
    "away_pitch_barrel_rate",
    # Bullpens (both contribute to total)
    "home_bp_xrv_mean",
    "away_bp_xrv_mean",
    "home_bp_fatigue_score",
    "away_bp_fatigue_score",
    "home_bp_recent_ip",
    "away_bp_recent_ip",
    # Defense
    "home_def_xrv_delta",
    "away_def_xrv_delta",
    # SP rest
    "home_sp_rest_days",
    "away_sp_rest_days",
    # Matchup quality
    "home_matchup_xrv_mean",
    "away_matchup_xrv_mean",
    # Differentials
    "diff_sp_xrv_mean",
    "diff_hit_xrv_mean",
    "diff_bp_xrv_mean",
]


def load_totals_training_data(exclude_year: int = None) -> pd.DataFrame:
    """Load feature files with total_runs target."""
    frames = []
    for path in sorted(FEATURES_DIR.glob("game_features_*.parquet")):
        year = int(path.stem.split("_")[-1])
        if exclude_year and year >= exclude_year:
            continue
        df = pd.read_parquet(path)
        df["season"] = year
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No feature files found")

    df = pd.concat(frames, ignore_index=True)
    df["total_runs"] = df["home_score"] + df["away_score"]
    df = df.dropna(subset=["total_runs"])
    return df


def _smart_fillna(df: pd.DataFrame, medians: dict = None) -> tuple[pd.DataFrame, dict]:
    """Fill NaN with column medians (0.0 if all-NaN). Returns (filled_df, medians_used)."""
    if medians is None:
        medians = {}
        for c in df.columns:
            if df[c].dtype in ("float64", "float32", "int64"):
                med = df[c].median()
                medians[c] = med if pd.notna(med) else 0.0
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(medians.get(c, 0.0))
    return df, medians


def _get_available_features(df: pd.DataFrame) -> list[str]:
    """Return features from TOTALS_FEATURES that exist in df."""
    return [f for f in TOTALS_FEATURES if f in df.columns]


def add_totals_nonlinear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer nonlinear features for totals prediction.

    Follows the pattern from win_model.py:add_nonlinear_features() but
    adapted for predicting total runs (both sides' scoring).
    """
    df = df.copy()

    # SP rest: short rest (<4 days) is a blowup risk for totals
    for side in ["home", "away"]:
        col = f"{side}_sp_rest_days"
        if col in df.columns:
            rest = df[col].fillna(5)
            df[f"{side}_sp_short_rest"] = np.clip(4 - rest, 0, None)

    # BP fatigue squared: extreme fatigue is disproportionately bad
    for side in ["home", "away"]:
        col = f"{side}_bp_fatigue_score"
        if col in df.columns:
            df[f"{col}_sq"] = df[col].fillna(0) ** 2

    # Offense-vs-pitching mismatch: high barrel rate vs low SP K rate
    home_br = df.get("home_hit_barrel_rate")
    away_br = df.get("away_hit_barrel_rate")
    home_k = df.get("home_sp_k_rate")
    away_k = df.get("away_sp_k_rate")
    if home_br is not None and away_k is not None:
        # Away hitters barrel rate vs home SP K rate
        df["away_barrel_vs_home_k"] = df["away_hit_barrel_rate"].fillna(0) * (1 - df["home_sp_k_rate"].fillna(0.2))
    if away_br is not None and home_k is not None:
        df["home_barrel_vs_away_k"] = df["home_hit_barrel_rate"].fillna(0) * (1 - df["away_sp_k_rate"].fillna(0.2))

    # Combined bullpen quality (both pens matter for late-game totals)
    if "home_bp_xrv_mean" in df.columns and "away_bp_xrv_mean" in df.columns:
        df["combined_bp_quality"] = df["home_bp_xrv_mean"].fillna(0) + df["away_bp_xrv_mean"].fillna(0)

    # Park factor * wind interaction
    if "park_factor" in df.columns and "wind_speed" in df.columns:
        df["park_wind"] = df["park_factor"].fillna(1.0) * df["wind_speed"].fillna(0)

    # Park factor * wind_out (strongest environmental effect on totals)
    if "park_factor" in df.columns and "wind_out" in df.columns:
        df["park_wind_out"] = df["park_factor"].fillna(1.0) * df["wind_out"].fillna(0)

    return df


# Names of nonlinear features so we can add them to the feature list dynamically
NONLINEAR_FEATURE_NAMES = [
    "home_sp_short_rest", "away_sp_short_rest",
    "home_bp_fatigue_score_sq", "away_bp_fatigue_score_sq",
    "away_barrel_vs_home_k", "home_barrel_vs_away_k",
    "combined_bp_quality",
    "park_wind", "park_wind_out",
]


def _train_single_target(X: pd.DataFrame, y: np.ndarray, scaler: StandardScaler,
                          xgb_params: dict) -> tuple:
    """Train Ridge + XGBoost for a single target (home_runs or away_runs).

    Returns (ridge, xgb_model, oof_ridge, oof_xgb).
    """
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit

    X_scaled = scaler.transform(X)

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    # XGBoost — early stopping on tail split
    xgb_model = xgb.XGBRegressor(**xgb_params)
    n_val = max(100, len(X) // 5)
    xgb_model.fit(
        X.iloc[:-n_val], y[:-n_val],
        eval_set=[(X.iloc[-n_val:], y[-n_val:])],
        verbose=False,
    )

    # TimeSeriesSplit OOF predictions
    tscv = TimeSeriesSplit(n_splits=5)
    oof_ridge = np.full(len(y), np.nan)
    oof_xgb = np.full(len(y), np.nan)

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y[train_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)

        r = Ridge(alpha=1.0)
        r.fit(X_tr_s, y_tr)

        xg = xgb.XGBRegressor(**xgb_params)
        n_es = max(50, len(X_tr) // 5)
        xg.fit(
            X_tr.iloc[:-n_es], y_tr[:-n_es],
            eval_set=[(X_tr.iloc[-n_es:], y_tr[-n_es:])],
            verbose=False,
        )

        oof_ridge[val_idx] = r.predict(X_va_s)
        oof_xgb[val_idx] = xg.predict(X_va)

    return ridge, xgb_model, oof_ridge, oof_xgb


def train_totals_model(train_df: pd.DataFrame) -> dict:
    """Train home/away Ridge + XGBoost regression ensembles for total runs.

    Returns a dict with keys:
        home: (ridge, xgb_model)
        away: (ridge, xgb_model)
        scaler: StandardScaler (shared)
        features: list of feature names
        w_ridge: float blend weight
        train_medians: dict
        sorted_residuals: np.ndarray
    """
    from scipy.optimize import minimize_scalar

    base_features = _get_available_features(train_df)

    # Apply nonlinear feature engineering
    X_full = train_df[base_features].copy()
    X_full, train_medians = _smart_fillna(X_full)
    X_full = add_totals_nonlinear_features(X_full)

    # Final feature list = base + nonlinear that exist
    features = base_features + [f for f in NONLINEAR_FEATURE_NAMES if f in X_full.columns]
    X = X_full[features].copy()

    y_home = train_df["home_score"].values
    y_away = train_df["away_score"].values
    y_total = train_df["total_runs"].values

    # Shared scaler
    scaler = StandardScaler()
    scaler.fit(X)

    xgb_params = dict(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=10,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=20,
        random_state=42,
    )

    # Train home and away models
    ridge_home, xgb_home, oof_ridge_home, oof_xgb_home = _train_single_target(X, y_home, scaler, xgb_params)
    ridge_away, xgb_away, oof_ridge_away, oof_xgb_away = _train_single_target(X, y_away, scaler, xgb_params)

    # Optimize blend weight on combined OOF total runs
    valid = ~(np.isnan(oof_ridge_home) | np.isnan(oof_xgb_home)
              | np.isnan(oof_ridge_away) | np.isnan(oof_xgb_away))

    def neg_rmse(w):
        oof_home = w * oof_ridge_home[valid] + (1 - w) * oof_xgb_home[valid]
        oof_away = w * oof_ridge_away[valid] + (1 - w) * oof_xgb_away[valid]
        oof_total = oof_home + oof_away
        return np.sqrt(np.mean((oof_total - y_total[valid]) ** 2))

    opt = minimize_scalar(neg_rmse, bounds=(0.0, 1.0), method="bounded")
    w_ridge = opt.x

    # Build OOF residuals for empirical CDF
    oof_home_blend = w_ridge * oof_ridge_home + (1 - w_ridge) * oof_xgb_home
    oof_away_blend = w_ridge * oof_ridge_away + (1 - w_ridge) * oof_xgb_away
    oof_total_blend = oof_home_blend + oof_away_blend
    oof_residuals = y_total[valid] - oof_total_blend[valid]
    sorted_residuals = np.sort(oof_residuals)
    oof_rmse = np.sqrt(np.mean(oof_residuals ** 2))

    # In-sample RMSE for overfitting gauge
    X_scaled = scaler.transform(X)
    is_home = w_ridge * ridge_home.predict(X_scaled) + (1 - w_ridge) * xgb_home.predict(X)
    is_away = w_ridge * ridge_away.predict(X_scaled) + (1 - w_ridge) * xgb_away.predict(X)
    is_total = is_home + is_away
    is_rmse = np.sqrt(np.mean((is_total - y_total) ** 2))

    print(f"  Totals model: Ridge weight={w_ridge:.2f}, "
          f"in-sample RMSE={is_rmse:.3f}, CV RMSE={oof_rmse:.3f}, "
          f"n_features={len(features)}, n_residuals={len(sorted_residuals)}")
    print(f"  OOF pred std: {oof_total_blend[valid].std():.3f}, "
          f"actual std: {y_total[valid].std():.3f}")

    return {
        "home": (ridge_home, xgb_home),
        "away": (ridge_away, xgb_away),
        "scaler": scaler,
        "features": features,
        "w_ridge": w_ridge,
        "train_medians": train_medians,
        "sorted_residuals": sorted_residuals,
    }


def predict_total_runs(
    model: dict,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict total runs for games.

    Accepts either new dict format or legacy tuple format.
    Returns df with total_runs_pred column.
    """
    # Support legacy tuple format for backwards compatibility
    if isinstance(model, tuple):
        ridge, scaler, xgb_model, features, w_ridge, train_medians, _sorted_residuals = model
        available = [f for f in features if f in features_df.columns]
        X = features_df[available].copy()
        X, _ = _smart_fillna(X, train_medians)
        for f in features:
            if f not in X.columns:
                X[f] = train_medians.get(f, 0.0)
        X = X[features]
        X_scaled = scaler.transform(X)
        blend = w_ridge * ridge.predict(X_scaled) + (1 - w_ridge) * xgb_model.predict(X)
        result = features_df.copy()
        result["total_runs_pred"] = blend
        return result

    # New dict format
    features = model["features"]
    scaler = model["scaler"]
    w_ridge = model["w_ridge"]
    train_medians = model["train_medians"]
    ridge_home, xgb_home = model["home"]
    ridge_away, xgb_away = model["away"]

    # Prepare features with nonlinear engineering
    base_features = [f for f in TOTALS_FEATURES if f in features_df.columns]
    X = features_df[base_features].copy()
    X, _ = _smart_fillna(X, train_medians)
    X = add_totals_nonlinear_features(X)

    # Ensure all model features are present
    for f in features:
        if f not in X.columns:
            X[f] = train_medians.get(f, 0.0)
    X = X[features]

    X_scaled = scaler.transform(X)
    home_pred = w_ridge * ridge_home.predict(X_scaled) + (1 - w_ridge) * xgb_home.predict(X)
    away_pred = w_ridge * ridge_away.predict(X_scaled) + (1 - w_ridge) * xgb_away.predict(X)

    result = features_df.copy()
    result["total_runs_pred"] = home_pred + away_pred
    result["home_runs_pred"] = home_pred
    result["away_runs_pred"] = away_pred
    return result


def prob_over(total_pred: float, line: float,
              sorted_residuals: np.ndarray | None = None) -> float:
    """Compute P(total > line) using empirical residual distribution.

    Given prediction mu and cross-validated residuals r_i:
        P(total > line) = P(mu + r > line) = P(r > line - mu)
                        = fraction of residuals > (line - mu)

    This avoids the Poisson assumption, which is badly wrong for MLB totals
    (actual variance/mean ~ 2.4, heavy right skew).

    Falls back to a Normal approximation if residuals are not provided
    (e.g. for the MC simulator's compute_total_prob, which uses its own
    empirical distribution from simulated games).
    """
    if sorted_residuals is None:
        # Fallback: should not be used in production — only for backwards compat
        from scipy.stats import norm
        sigma = 4.5  # approximate OOS RMSE
        return float(1.0 - norm.cdf(line, loc=total_pred, scale=sigma))

    threshold = line - total_pred
    # Binary search in sorted residuals
    idx = np.searchsorted(sorted_residuals, threshold, side="right")
    return float(1.0 - idx / len(sorted_residuals))


def backtest_totals(year: int):
    """Backtest totals model on a season."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    print(f"\n{'='*60}")
    print(f"  Totals Model Backtest — {year}")
    print(f"{'='*60}")

    train_df = load_totals_training_data(exclude_year=year)
    print(f"  Training on {len(train_df):,} games")

    model = train_totals_model(train_df)
    sorted_residuals = model["sorted_residuals"]
    features = model["features"]

    test_path = FEATURES_DIR / f"game_features_{year}.parquet"
    if not test_path.exists():
        print(f"  No features for {year}")
        return

    test_df = pd.read_parquet(test_path)
    test_df["total_runs"] = test_df["home_score"] + test_df["away_score"]
    test_df = test_df.dropna(subset=["total_runs"])
    print(f"  Testing on {len(test_df):,} games")

    result = predict_total_runs(model, test_df)

    y_true = result["total_runs"].values
    y_pred = result["total_runs_pred"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    actual_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - actual_mean) ** 2)

    print(f"\n  Regression Metrics:")
    print(f"    RMSE:         {rmse:.3f}")
    print(f"    MAE:          {mae:.3f}")
    print(f"    R²:           {r2:.4f}")
    print(f"    Actual mean:  {actual_mean:.2f}")
    print(f"    Predicted mean: {pred_mean:.2f}")
    print(f"    Pred std:     {y_pred.std():.3f}")

    # Calibration: P(over) at various lines
    print(f"\n  Over/Under Calibration (empirical residual CDF):")
    print(f"  {'Line':>6s} {'P(over)':>8s} {'Actual%':>8s} {'Bias':>7s} {'Brier':>8s}")
    print(f"  {'-'*45}")

    for line in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]:
        p_over = np.array([prob_over(p, line, sorted_residuals) for p in y_pred])
        actual_over = (y_true > line).astype(float)
        brier = np.mean((p_over - actual_over) ** 2)
        bias = p_over.mean() - actual_over.mean()
        print(f"  {line:>6.1f} {p_over.mean():>7.1%} {actual_over.mean():>7.1%} "
              f"{bias:>+6.1%} {brier:>8.4f}")

    # Quantile calibration at 8.5
    print(f"\n  Quantile Calibration (line=8.5):")
    p_over = np.array([prob_over(p, 8.5, sorted_residuals) for p in y_pred])
    actual_over = (y_true > 8.5).astype(float)
    bins = [(0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.0)]
    for lo, hi in bins:
        mask = (p_over >= lo) & (p_over < hi)
        if mask.sum() < 10:
            continue
        act = actual_over[mask].mean()
        mod = p_over[mask].mean()
        print(f"    P(over)∈[{lo:.2f},{hi:.2f}): model={mod:.1%}  actual={act:.1%}  "
              f"N={mask.sum():4d}  gap={mod-act:+.1%}")

    # Feature importance from XGBoost (home model)
    _, xgb_home = model["home"]
    print(f"\n  Top 15 Features (XGBoost importance, home model):")
    importances = xgb_home.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    for i, j in enumerate(idx):
        print(f"    {i+1:>2d}. {features[j]:<35s} {importances[j]:.4f}")


def predict_date(target_date: str):
    """Predict totals for today's games."""
    from predict import (
        fetch_todays_games, build_live_features, load_training_data,
    )
    import httpx

    print(f"\n{'='*60}")
    print(f"  Totals Predictions for {target_date}")
    print(f"{'='*60}")

    # Train totals model
    print("\n  Training totals model...")
    train_df = load_totals_training_data()
    train_df["game_date"] = train_df["game_date"].astype(str)
    train_df = train_df[train_df["game_date"] < target_date]
    model = train_totals_model(train_df)
    sorted_residuals = model["sorted_residuals"]

    # Fetch and build features
    print(f"\n  Fetching games for {target_date}...")
    with httpx.Client(timeout=15.0) as client:
        games = fetch_todays_games(client, target_date)
        if not games:
            print("  No games found")
            return
        print(f"  Found {len(games)} games")
        features_df = build_live_features(games, target_date, client)

    result = predict_total_runs(model, features_df)

    print(f"\n  {'Matchup':<25s} {'Pred':>6s} {'O 7.5':>6s} {'O 8.5':>6s} "
          f"{'O 9.5':>6s} {'O 10.5':>7s}")
    print(f"  {'-'*58}")

    for _, r in result.iterrows():
        matchup = f"{r.get('away_team', '?')} @ {r.get('home_team', '?')}"
        pred = r["total_runs_pred"]
        print(f"  {matchup:<25s} {pred:>5.1f} "
              f"{prob_over(pred, 7.5, sorted_residuals):>5.1%} "
              f"{prob_over(pred, 8.5, sorted_residuals):>5.1%} "
              f"{prob_over(pred, 9.5, sorted_residuals):>5.1%} "
              f"{prob_over(pred, 10.5, sorted_residuals):>6.1%}")


def main():
    parser = argparse.ArgumentParser(description="MLB Totals Prediction Model")
    parser.add_argument("--backtest", type=int, nargs="+",
                        help="Backtest on season(s)")
    parser.add_argument("--date", type=str,
                        help="Predict totals for date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.backtest:
        for year in args.backtest:
            backtest_totals(year)
    elif args.date:
        predict_date(args.date)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
