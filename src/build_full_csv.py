#!/usr/bin/env python3
"""
Build the full 2025 comparison CSV with game metadata, model features,
model win probabilities, Kalshi pre-game closing lines, and actual results.

Usage:
    python src/build_full_csv.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from win_model import ALL_FEATURES, XGB_PARAMS, _smart_fillna, add_nonlinear_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_DIR = DATA_DIR / "features"
OUTPUT_PATH = DATA_DIR / "full_2025_comparison.csv"


def load_features(years):
    frames = []
    for year in years:
        path = FEATURES_DIR / f"game_features_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["season"] = year
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prepare_xgb_features(df):
    """Prepare XGB feature matrix with nonlinear features (NaN handled by XGB)."""
    available = [f for f in ALL_FEATURES if f in df.columns]
    X = df[available].copy()
    for col in ["home_sp_rest_days", "away_sp_rest_days",
                "home_bp_fatigue_score", "away_bp_fatigue_score",
                "days_into_season"]:
        if col in df.columns and col not in X.columns:
            X[col] = df[col]
    return add_nonlinear_features(X)


def train_ensemble(train_df):
    y = train_df["home_win"].values

    # LR: linear features → _smart_fillna → scaler
    available = [f for f in ALL_FEATURES if f in train_df.columns]
    X_lr_raw = train_df[available].copy()
    X_filled, train_medians = _smart_fillna(X_lr_raw)
    scaler = StandardScaler()
    X_lr = scaler.fit_transform(X_filled)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(X_lr, y)

    # XGB: nonlinear features → DMatrix (NaN native)
    X_xgb = _prepare_xgb_features(train_df)
    xgb_features = list(X_xgb.columns)

    xgb_model = None
    w_lr = 0.5
    if HAS_XGB:
        # Chronological val split for early stopping
        n = len(X_xgb)
        val_size = int(n * 0.2)
        dtrain = xgb.DMatrix(X_xgb.iloc[:n - val_size], label=y[:n - val_size])
        dval = xgb.DMatrix(X_xgb.iloc[n - val_size:], label=y[n - val_size:])
        xgb_model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=500,
                               evals=[(dtrain, "train"), (dval, "val")],
                               early_stopping_rounds=50, verbose_eval=50)

        # Learn blend weight via OOF predictions (not in-sample) to avoid leakage
        from scipy.optimize import minimize_scalar
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=5, shuffle=False)  # chronological folds
        lr_oof = np.zeros(len(y))
        xgb_oof = np.zeros(len(y))

        for fold_train, fold_val in kf.split(X_lr_raw):
            # LR fold
            Xf_filled, fm = _smart_fillna(X_lr_raw.iloc[fold_train])
            Xv_filled, _ = _smart_fillna(X_lr_raw.iloc[fold_val], fm)
            sc_f = StandardScaler()
            lr_f = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            lr_f.fit(sc_f.fit_transform(Xf_filled), y[fold_train])
            lr_oof[fold_val] = lr_f.predict_proba(sc_f.transform(Xv_filled))[:, 1]

            # XGB fold
            Xf_xgb = X_xgb.iloc[fold_train]
            Xv_xgb = X_xgb.iloc[fold_val]
            df_fold = xgb.DMatrix(Xf_xgb, label=y[fold_train])
            dv_fold = xgb.DMatrix(Xv_xgb, label=y[fold_val])
            xgb_f = xgb.train(XGB_PARAMS, df_fold, num_boost_round=200,
                              evals=[(df_fold, "t"), (dv_fold, "v")],
                              early_stopping_rounds=30, verbose_eval=0)
            xgb_oof[fold_val] = xgb_f.predict(dv_fold)

        def blend_loss(w):
            blended = np.clip(w * lr_oof + (1 - w) * xgb_oof, 1e-6, 1 - 1e-6)
            return log_loss(y, blended)

        opt = minimize_scalar(blend_loss, bounds=(0.0, 1.0), method="bounded")
        w_lr = opt.x
        print(f"  Learned blend (OOF): LR={w_lr:.2f}, XGB={1-w_lr:.2f}")

    return lr, scaler, xgb_model, available, xgb_features, w_lr, train_medians


def predict(lr, scaler, xgb_model, lr_features, xgb_features, test_df, w_lr=0.5, train_medians=None):
    # LR path
    available_lr = [f for f in lr_features if f in test_df.columns]
    X_lr_raw = test_df[available_lr].copy()
    X_filled, _ = _smart_fillna(X_lr_raw, train_medians)
    for col in lr_features:
        if col not in X_filled.columns:
            X_filled[col] = 0
    X_filled = X_filled[lr_features]
    X_lr = scaler.transform(X_filled)
    lr_probs = lr.predict_proba(X_lr)[:, 1]

    # XGB path
    if xgb_model and HAS_XGB:
        X_xgb = _prepare_xgb_features(test_df)
        for col in xgb_features:
            if col not in X_xgb.columns:
                X_xgb[col] = np.nan
        X_xgb = X_xgb[xgb_features]
        dtest = xgb.DMatrix(X_xgb)
        xgb_probs = xgb_model.predict(dtest)
        ens_probs = w_lr * lr_probs + (1 - w_lr) * xgb_probs
    else:
        ens_probs = lr_probs

    return ens_probs


def compute_bet_pnl(edge, market_prob, outcome, fee_pct=0.02):
    """PnL for a flat $100 notional bet with fees."""
    if edge > 0:
        cost = market_prob * 100
        fee = cost * fee_pct
        return (100 - cost - fee) if outcome == 1 else (-cost - fee)
    else:
        cost = (1 - market_prob) * 100
        fee = cost * fee_pct
        return (100 - cost - fee) if outcome == 0 else (-cost - fee)


def main():
    # ── Train model on 2018-2024 ───────────────────────────────────────
    print("Loading training data (2018-2024)...")
    train_df = load_features(list(range(2018, 2025)))
    print(f"  {len(train_df)} training games")

    print("\nTraining ensemble model...")
    lr, scaler, xgb_model, lr_features, xgb_features, w_lr, train_medians = train_ensemble(train_df)
    print(f"  LR features: {len(lr_features)}, XGB features: {len(xgb_features)}")

    # ── Load 2025 features ─────────────────────────────────────────────
    print("\nLoading 2025 features...")
    test_df = load_features([2025])
    print(f"  {len(test_df)} games, {len(test_df.columns)} columns")

    # ── Load player names from games parquet ─────────────────────────
    print("\nLoading player/game metadata...")
    games_path = DATA_DIR / "games" / "games_2025.parquet"
    games_meta = pd.read_parquet(games_path)
    games_meta["game_date"] = games_meta["game_date"].astype(str)
    player_cols = ["game_date", "home_team_abbr", "away_team_abbr",
                   "home_team_name", "away_team_name",
                   "home_sp_name", "away_sp_name", "venue_name"]
    games_meta = games_meta[player_cols].rename(columns={
        "home_team_abbr": "home_team", "away_team_abbr": "away_team",
    })
    test_df = test_df.merge(games_meta, on=["game_date", "home_team", "away_team"], how="left")
    print(f"  Matched {test_df['home_sp_name'].notna().sum()}/{len(test_df)} with pitcher names")

    # ── Generate predictions ───────────────────────────────────────────
    print("\nGenerating model predictions...")
    model_probs = predict(lr, scaler, xgb_model, lr_features, xgb_features, test_df, w_lr, train_medians)
    test_df["model_home_prob"] = model_probs

    # ── Load Kalshi data ───────────────────────────────────────────────
    print("\nLoading Kalshi data...")
    kalshi_path = DATA_DIR / "kalshi" / "kalshi_mlb_2025.parquet"
    kalshi = pd.read_parquet(kalshi_path)
    kalshi["game_date"] = kalshi["game_date"].astype(str)
    print(f"  {len(kalshi)} Kalshi games")

    # ── Join ───────────────────────────────────────────────────────────
    print("\nJoining features + model + Kalshi...")
    test_df["game_date"] = test_df["game_date"].astype(str)

    merged = test_df.merge(
        kalshi[["game_date", "home_team", "away_team",
                "kalshi_home_prob", "kalshi_away_prob", "volume"]].rename(
            columns={"volume": "kalshi_volume"}
        ),
        on=["game_date", "home_team", "away_team"],
        how="left",
    )

    has_kalshi = merged["kalshi_home_prob"].notna()
    merged["model_edge"] = np.where(
        has_kalshi,
        merged["model_home_prob"] - merged["kalshi_home_prob"],
        np.nan,
    )

    # ── Identify feature columns ───────────────────────────────────────
    meta_cols = [
        "game_date", "home_team", "away_team",
        "home_team_name", "away_team_name",
        "home_sp_name", "away_sp_name", "venue_name",
        "home_win",
        "model_home_prob", "kalshi_home_prob", "model_edge", "kalshi_volume",
    ]
    non_feature_cols = {
        "game_pk", "game_date", "home_team", "away_team", "home_win",
        "home_score", "away_score", "season",
        "home_team_name", "away_team_name",
        "home_sp_name", "away_sp_name", "venue_name",
        "model_home_prob", "kalshi_home_prob", "kalshi_away_prob",
        "kalshi_volume", "model_edge",
    }
    feature_cols = [c for c in merged.columns if c not in non_feature_cols]

    # Prefix feature columns with feat_ to avoid collisions
    feat_rename = {c: f"feat_{c}" for c in feature_cols}
    output_cols = meta_cols + [f"feat_{c}" for c in feature_cols]

    out = merged.rename(columns=feat_rename)[output_cols]
    out = out.sort_values("game_date").reset_index(drop=True)

    # ── Save CSV ───────────────────────────────────────────────────────
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out)} games to {OUTPUT_PATH}")

    # ── Summary stats ──────────────────────────────────────────────────
    n_total = len(out)
    n_both = has_kalshi.sum()
    print(f"\n{'='*60}")
    print(f"  Summary: {n_total} total games, {n_both} with Kalshi prices")
    print(f"{'='*60}")

    y = out["home_win"].values
    m_probs = np.clip(out["model_home_prob"].values, 0.01, 0.99)

    print(f"\n  All {n_total} games (model only):")
    print(f"    Log loss:    {log_loss(y, m_probs):.4f}")
    print(f"    Brier score: {brier_score_loss(y, m_probs):.4f}")
    print(f"    AUC:         {roc_auc_score(y, m_probs):.4f}")

    # Kalshi subset
    k_mask = out["kalshi_home_prob"].notna()
    if k_mask.sum() > 0:
        y_k = out.loc[k_mask, "home_win"].values
        m_k = np.clip(out.loc[k_mask, "model_home_prob"].values, 0.01, 0.99)
        k_k = np.clip(out.loc[k_mask, "kalshi_home_prob"].values, 0.01, 0.99)
        edge = m_k - k_k

        print(f"\n  {k_mask.sum()} games with both model + Kalshi:")
        print(f"    {'Metric':<20s} {'Model':>10s} {'Kalshi':>10s}")
        print(f"    {'-'*42}")
        print(f"    {'Log loss':<20s} {log_loss(y_k, m_k):>10.4f} {log_loss(y_k, k_k):>10.4f}")
        print(f"    {'Brier score':<20s} {brier_score_loss(y_k, m_k):>10.4f} {brier_score_loss(y_k, k_k):>10.4f}")
        print(f"    {'AUC':<20s} {roc_auc_score(y_k, m_k):>10.4f} {roc_auc_score(y_k, k_k):>10.4f}")

        print(f"\n  Edge analysis:")
        print(f"    Mean absolute edge: {np.abs(edge).mean():.4f}")
        print(f"    Games with |edge| > 3%: {(np.abs(edge) > 0.03).sum()} ({(np.abs(edge) > 0.03).mean():.1%})")
        print(f"    Correlation (model, Kalshi): {np.corrcoef(m_k, k_k)[0, 1]:.4f}")

        # Simulated flat-bet ROI
        for threshold in [0.03, 0.05]:
            pnls = []
            for i in range(len(edge)):
                if abs(edge[i]) >= threshold:
                    pnls.append(compute_bet_pnl(edge[i], k_k[i], y_k[i]))
            if pnls:
                total = sum(pnls)
                roi = total / (len(pnls) * 100)
                print(f"    ROI at {threshold:.0%} edge: {roi:+.1%} on {len(pnls)} bets (${total:+,.0f})")


if __name__ == "__main__":
    main()
