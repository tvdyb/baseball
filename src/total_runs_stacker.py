"""
Total Runs Stacking Model
=========================
A second-stage LightGBM model that combines:
  1. MC simulator outputs (sim_total_mean, sim_home_wp, sim_nrfi_prob)
  2. LightGBM first-stage total-runs predictions + quantile outputs
  3. Pregame features (SP quality, park factors, weather, team offense)

Also trains quantile regressors to predict P(total > line) directly for
O/U betting at lines 7.5, 8.0, 8.5, 9.0, 9.5.

Walk-forward: train on first 200 games (chronological), predict remaining
games with monthly expanding-window recalibration.

Saves: data/backtest/total_runs_stacker_2025.parquet
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
SIM_PATH   = "data/backtest/nrfi_ou_backtest_2025.parquet"
FEAT_PATH  = "data/features/game_features_2025.parquet"
LGB_PATH   = "data/backtest/total_runs_lgb_2025.parquet"
OUT_PATH   = "data/backtest/total_runs_stacker_2025.parquet"
LINES      = [7.5, 8.0, 8.5, 9.0, 9.5]

# ─── Selected pregame features ────────────────────────────────────────────────
PREGAME_COLS = [
    # SP quality
    "home_sp_xrv_mean", "home_sp_xrv_std", "home_sp_k_rate", "home_sp_bb_rate",
    "home_sp_avg_velo", "home_sp_stuff_score", "home_sp_composite_score",
    "home_sp_projected_era", "home_sp_overperf_recent", "home_sp_xrv_vs_lineup",
    "away_sp_xrv_mean", "away_sp_xrv_std", "away_sp_k_rate", "away_sp_bb_rate",
    "away_sp_avg_velo", "away_sp_stuff_score", "away_sp_composite_score",
    "away_sp_projected_era", "away_sp_overperf_recent", "away_sp_xrv_vs_lineup",
    # Bullpen
    "home_bp_xrv_mean", "home_bp_fatigue_score",
    "away_bp_xrv_mean", "away_bp_fatigue_score",
    # Hitting
    "home_hit_xrv_mean", "home_hit_xrv_contact", "home_hit_k_rate",
    "home_hit_hard_hit_rate", "home_hit_barrel_rate",
    "away_hit_xrv_mean", "away_hit_xrv_contact", "away_hit_k_rate",
    "away_hit_hard_hit_rate", "away_hit_barrel_rate",
    # Matchup
    "home_matchup_xrv_mean", "away_matchup_xrv_mean",
    "home_arsenal_matchup_xrv_mean", "away_arsenal_matchup_xrv_mean",
    # Park / weather
    "park_factor", "temperature", "wind_speed", "is_dome", "wind_out", "wind_in",
    "is_night",
    # Team priors
    "home_team_prior", "away_team_prior",
    "home_recent_form", "away_recent_form",
    "days_into_season",
]

# ─── Sim features (available for 500 games) ───────────────────────────────────
SIM_COLS = ["sim_total_mean", "sim_home_wp", "sim_nrfi_prob", "sim_line"]

# ─── LGB meta-features ────────────────────────────────────────────────────────
LGB_META_COLS = [
    "lgb_pred_total", "lgb_q10", "lgb_q25", "lgb_q50", "lgb_q75", "lgb_q90",
    "lgb_p_over_7.5", "lgb_p_over_8.0", "lgb_p_over_8.5",
    "lgb_p_over_9.0", "lgb_p_over_9.5",
]

# ─── LGB hyperparameters ─────────────────────────────────────────────────────
BASE_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=5,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Tighter regularization for classifier (less overfitting on binary target)
CLS_PARAMS = dict(BASE_PARAMS)
CLS_PARAMS.update(dict(
    n_estimators=300,
    num_leaves=15,
    max_depth=4,
    min_child_samples=20,
    reg_lambda=2.0,
))


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Data loading & merging
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data …")

    feat = pd.read_parquet(FEAT_PATH)
    feat["game_date"] = pd.to_datetime(feat["game_date"])
    feat["total_runs"] = feat["home_score"] + feat["away_score"]

    lgb_df = pd.read_parquet(LGB_PATH)
    lgb_df["game_date"] = pd.to_datetime(lgb_df["game_date"])

    sim = pd.read_parquet(SIM_PATH)
    sim["game_date"] = pd.to_datetime(sim["game_date"])

    keep_feat = ["game_pk", "game_date", "home_team", "away_team",
                 "home_score", "away_score", "total_runs"] + PREGAME_COLS
    keep_feat = [c for c in keep_feat if c in feat.columns]

    df = feat[keep_feat].copy()
    df = df.merge(lgb_df[["game_pk"] + LGB_META_COLS], on="game_pk", how="left")
    df = df.merge(sim[["game_pk"] + SIM_COLS], on="game_pk", how="left")

    df = df.sort_values("game_date").reset_index(drop=True)
    df = df.dropna(subset=["total_runs"]).reset_index(drop=True)

    print(f"  Total games: {len(df)}")
    print(f"  With sim:    {df['sim_total_mean'].notna().sum()}")
    print(f"  With LGB:    {df['lgb_pred_total'].notna().sum()}")
    print(f"  Date range:  {df.game_date.min().date()} – {df.game_date.max().date()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Feature set
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_cols(df):
    cols = []
    for c in PREGAME_COLS + LGB_META_COLS + SIM_COLS:
        if c in df.columns:
            cols.append(c)
    return cols


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Walk-forward with monthly expanding windows
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_predict_block(X_train, y_train, X_test, line=None):
    """
    Train one regressor (line=None) or one binary classifier (line=float).
    Returns predictions for X_test.
    """
    medians = X_train.median()
    X_tr = X_train.fillna(medians)
    X_te = X_test.fillna(medians)

    if line is None:
        model = lgb.LGBMRegressor(**BASE_PARAMS)
        model.fit(X_tr, y_train)
        return model.predict(X_te), model
    else:
        y_cls = (y_train > line).astype(int)
        if y_cls.sum() < 5 or (~y_cls.astype(bool)).sum() < 5:
            return np.full(len(X_te), np.nan), None
        model = lgb.LGBMClassifier(**CLS_PARAMS)
        model.fit(X_tr, y_cls)
        return model.predict_proba(X_te)[:, 1], model


def walk_forward_predict(df, feature_cols, initial_train=200):
    n = len(df)
    X = df[feature_cols]
    y = df["total_runs"].values

    pred_total   = np.full(n, np.nan)
    pred_ou = {line: np.full(n, np.nan) for line in LINES}

    df_copy = df.copy()
    df_copy["year_month"] = df_copy["game_date"].dt.to_period("M")
    test_months = df_copy.iloc[initial_train:]["year_month"].unique()

    print(f"\nWalk-forward: train={initial_train}, predict={n-initial_train} "
          f"across {len(test_months)} months")

    train_end = initial_train

    for month in test_months:
        month_mask = (df_copy["year_month"] == month) & (df_copy.index >= initial_train)
        month_idx  = df_copy.index[month_mask].tolist()
        if not month_idx:
            continue

        X_train = X.iloc[:train_end]
        y_train = y[:train_end]
        X_test  = X.iloc[month_idx]

        preds, _ = train_and_predict_block(X_train, y_train, X_test, line=None)
        pred_total[month_idx] = preds

        for line in LINES:
            p, _ = train_and_predict_block(X_train, y_train, X_test, line=line)
            pred_ou[line][month_idx] = p

        print(f"  {month}: trained on {train_end:>4} → predicting {len(month_idx):>3}")
        train_end = month_idx[-1] + 1

    return pred_total, pred_ou


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Quantile regression (direct P(total > line) without binary classification)
# ═══════════════════════════════════════════════════════════════════════════════

def ou_from_quantile_regression(df, feature_cols, initial_train=200):
    """
    Train a quantile LGB regressor at quantile α to get P(Y > line) = 1 - α.
    We estimate P(Y > 8.5) ≈ 1 - F(8.5) using the predicted quantile.

    More directly: fit a regressor that predicts the total, then use the
    predicted distribution (assumed Normal with mean=pred, std=residual std)
    to estimate P(Y > line).

    This version uses LGB's native quantile objective for q10, q50, q90
    then interpolates probabilities.
    """
    n = len(df)
    X = df[feature_cols]
    y = df["total_runs"].values

    df_copy = df.copy()
    df_copy["year_month"] = df_copy["game_date"].dt.to_period("M")
    test_months = df_copy.iloc[initial_train:]["year_month"].unique()

    # Store quantile predictions
    q_preds = {q: np.full(n, np.nan) for q in [0.10, 0.25, 0.50, 0.75, 0.90]}

    print("\nQuantile regression walk-forward …")
    train_end = initial_train

    for month in test_months:
        month_mask = (df_copy["year_month"] == month) & (df_copy.index >= initial_train)
        month_idx  = df_copy.index[month_mask].tolist()
        if not month_idx:
            continue

        X_train = X.iloc[:train_end]
        y_train = y[:train_end]
        X_test  = X.iloc[month_idx]
        medians = X_train.median()
        X_tr = X_train.fillna(medians)
        X_te = X_test.fillna(medians)

        for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
            qparams = dict(BASE_PARAMS)
            qparams.update(dict(objective="quantile", alpha=q, n_estimators=300))
            model = lgb.LGBMRegressor(**qparams)
            model.fit(X_tr, y_train)
            q_preds[q][month_idx] = model.predict(X_te)

        train_end = month_idx[-1] + 1

    # Convert quantile predictions to P(total > line) via linear interpolation
    # P(Y > line) ≈ 1 - F(line) where F is estimated from quantile predictions
    # We interpolate between known quantile points
    qu_prob = {}
    quantile_levels = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    q_matrix = np.column_stack([q_preds[q] for q in quantile_levels])

    for line in LINES:
        probs = np.full(n, np.nan)
        valid = ~np.any(np.isnan(q_matrix), axis=1)
        for i in np.where(valid)[0]:
            q_vals = q_matrix[i]
            # Linear interp: F(line) = quantile level where q_val = line
            # If line < q_vals[0]: F(line) < 0.10
            # If line > q_vals[-1]: F(line) > 0.90
            cdf_at_line = np.interp(line, q_vals, quantile_levels,
                                    left=0.0, right=1.0)
            probs[i] = 1.0 - cdf_at_line
        qu_prob[line] = probs

    return qu_prob


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def ou_accuracy(actual, pred_prob, line):
    """Accuracy betting over if prob > 0.5. Pushes excluded."""
    pushes = (actual == line)
    valid  = ~pushes & ~np.isnan(pred_prob)
    if valid.sum() == 0:
        return np.nan
    over_correct = (actual[valid] > line) == (pred_prob[valid] > 0.5)
    return over_correct.mean()


def regression_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]
    mae  = mean_absolute_error(y_t, y_p)
    rmse = mean_squared_error(y_t, y_p) ** 0.5
    corr = pearsonr(y_t, y_p)[0]
    return dict(n=int(mask.sum()), mae=mae, rmse=rmse, corr=corr)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Feature importance (on initial training set)
# ═══════════════════════════════════════════════════════════════════════════════

def feature_importance(df, feature_cols, initial_train=200):
    X = df[feature_cols].iloc[:initial_train]
    y = df["total_runs"].values[:initial_train]
    medians = X.median()
    model = lgb.LGBMRegressor(**BASE_PARAMS)
    model.fit(X.fillna(medians), y)
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    df = load_data()
    feature_cols = get_feature_cols(df)
    INITIAL_TRAIN = 200

    print(f"\nFeatures: {len(feature_cols)} columns")

    # ── Walk-forward ─────────────────────────────────────────────────────────
    pred_total, pred_ou = walk_forward_predict(df, feature_cols, INITIAL_TRAIN)

    # ── Quantile regression ──────────────────────────────────────────────────
    qu_prob = ou_from_quantile_regression(df, feature_cols, INITIAL_TRAIN)

    # ── Attach predictions ───────────────────────────────────────────────────
    df["stk_pred_total"] = pred_total
    for line in LINES:
        col = f"stk_p_over_{line}"
        df[col] = pred_ou[line]
        df[f"stk_qr_p_over_{line}"] = qu_prob[line]

    # Test set
    test_mask = (df.index >= INITIAL_TRAIN) & df["stk_pred_total"].notna()
    df_test = df[test_mask].copy()
    y_test  = df_test["total_runs"].values

    # ── Print results ────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("EVALUATION — TEST SET")
    print(f"{'═'*68}")
    print(f"Games: {len(df_test)}  |  "
          f"{df_test.game_date.min().date()} – {df_test.game_date.max().date()}")

    # Regression
    print("\n── Regression Metrics ──────────────────────────────────────────────")
    rows = []
    for label, col in [("Stacker", "stk_pred_total"),
                        ("LGB-only", "lgb_pred_total"),
                        ("Sim-only", "sim_total_mean")]:
        if col not in df_test.columns:
            continue
        m = regression_metrics(y_test, df_test[col].values)
        rows.append((label, m))
        print(f"  {label:<12}: MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
              f"r={m['corr']:.3f}  (n={m['n']})")

    # O/U accuracy — classifier
    print("\n── O/U Accuracy (Binary Classifier) ───────────────────────────────")
    print(f"  {'Line':>5}  {'Stacker':>8}  {'LGB-only':>9}  {'Naïve(>50%)':>12}  N")
    print("  " + "-" * 52)
    ou_acc_results = {}
    for line in LINES:
        stk_acc = ou_accuracy(y_test, df_test[f"stk_p_over_{line}"].values, line)
        lgb_acc = ou_accuracy(y_test, df_test[f"lgb_p_over_{line}"].values, line) \
                  if f"lgb_p_over_{line}" in df_test.columns else np.nan
        # Naïve: always bet the more common side
        naive_over = (y_test > line).mean()
        naive_acc  = max(naive_over, 1 - naive_over)
        n_valid    = (~(y_test == line)).sum()
        print(f"  {line:>5.1f}  {stk_acc:>8.3f}  {lgb_acc:>9.3f}  "
              f"{naive_acc:>12.3f}  {n_valid}")
        ou_acc_results[line] = dict(stk=stk_acc, lgb=lgb_acc, naive=naive_acc)

    # O/U accuracy — quantile regression
    print("\n── O/U Accuracy (Quantile Regression → P(over)) ───────────────────")
    print(f"  {'Line':>5}  {'QR-Stacker':>10}  {'LGB-only':>9}  {'Naïve':>8}  N")
    print("  " + "-" * 48)
    for line in LINES:
        qr_acc  = ou_accuracy(y_test, df_test[f"stk_qr_p_over_{line}"].values, line)
        lgb_acc = ou_accuracy(y_test, df_test[f"lgb_p_over_{line}"].values, line) \
                  if f"lgb_p_over_{line}" in df_test.columns else np.nan
        naive_acc = max((y_test > line).mean(), 1 - (y_test > line).mean())
        n_valid   = (~(y_test == line)).sum()
        print(f"  {line:>5.1f}  {qr_acc:>10.3f}  {lgb_acc:>9.3f}  "
              f"{naive_acc:>8.3f}  {n_valid}")

    # Calibration
    print("\n── Calibration: P(over line) ───────────────────────────────────────")
    print(f"  {'Line':>5}  {'Actual%':>8}  {'Cls-mean':>9}  {'QR-mean':>8}  {'LGB-mean':>9}")
    print("  " + "-" * 52)
    for line in LINES:
        actual_frac = (y_test > line).mean()
        cls_mean    = df_test[f"stk_p_over_{line}"].mean()
        qr_mean     = df_test[f"stk_qr_p_over_{line}"].mean()
        lgb_mean    = df_test[f"lgb_p_over_{line}"].mean() \
                      if f"lgb_p_over_{line}" in df_test.columns else np.nan
        print(f"  {line:>5.1f}  {actual_frac:>8.3f}  {cls_mean:>9.3f}  "
              f"{qr_mean:>8.3f}  {lgb_mean:>9.3f}")

    # Feature importance
    print("\n── Top-20 Feature Importances ──────────────────────────────────────")
    imp = feature_importance(df, feature_cols, INITIAL_TRAIN)
    for name, score in imp.head(20).items():
        tag = " ◀ SIM" if name in SIM_COLS else (" ◀ LGB" if name in LGB_META_COLS else "")
        print(f"  {name:<44} {score:>6.0f}{tag}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    save_base = ["game_pk", "game_date", "home_team", "away_team",
                 "home_score", "away_score", "total_runs",
                 "sim_total_mean", "sim_home_wp", "sim_nrfi_prob", "sim_line",
                 "lgb_pred_total",
                 "lgb_p_over_7.5", "lgb_p_over_8.0", "lgb_p_over_8.5",
                 "lgb_p_over_9.0", "lgb_p_over_9.5",
                 "stk_pred_total"]
    for line in LINES:
        save_base += [f"stk_p_over_{line}", f"stk_qr_p_over_{line}"]
    save_cols = [c for c in save_base if c in df.columns]
    df[save_cols].to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}  ({len(df)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("SUMMARY")
    print(f"{'═'*68}")
    stk_m = rows[0][1] if rows else {}
    lgb_m = rows[1][1] if len(rows) > 1 else {}
    if stk_m and lgb_m:
        delta_mae  = stk_m["mae"]  - lgb_m["mae"]
        delta_rmse = stk_m["rmse"] - lgb_m["rmse"]
        delta_r    = stk_m["corr"] - lgb_m["corr"]
        print(f"  Regression (stacker vs LGB-only):")
        print(f"    MAE  {stk_m['mae']:.3f} vs {lgb_m['mae']:.3f}  Δ={delta_mae:+.3f}")
        print(f"    RMSE {stk_m['rmse']:.3f} vs {lgb_m['rmse']:.3f}  Δ={delta_rmse:+.3f}")
        print(f"    r    {stk_m['corr']:.3f} vs {lgb_m['corr']:.3f}  Δ={delta_r:+.3f}")
    print(f"\n  O/U accuracy (stacker vs LGB-only):")
    for line in LINES:
        r = ou_acc_results.get(line, {})
        stk_a = r.get("stk", np.nan)
        lgb_a = r.get("lgb", np.nan)
        if not np.isnan(stk_a):
            delta = stk_a - lgb_a if not np.isnan(lgb_a) else np.nan
            delta_str = f"Δ={delta:+.3f}" if not np.isnan(delta) else ""
            print(f"    Line {line}: {stk_a:.3f} vs {lgb_a:.3f}  {delta_str}")


if __name__ == "__main__":
    main()
