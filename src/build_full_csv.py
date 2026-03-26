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
from win_model import DIFF_FEATURES, RAW_FEATURES, ALL_FEATURES

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


def train_ensemble(train_df):
    available = [f for f in ALL_FEATURES if f in train_df.columns]
    X = train_df[available].copy()
    y = train_df["home_win"].values

    scaler = StandardScaler()
    X_lr = scaler.fit_transform(X.fillna(0))
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(X_lr, y)

    xgb_model = None
    w_lr = 0.5
    if HAS_XGB:
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "binary:logistic", "eval_metric": "logloss",
            "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 50,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "verbosity": 0,
        }
        xgb_model = xgb.train(params, dtrain, num_boost_round=100)

        from scipy.optimize import minimize_scalar
        lr_train_probs = lr.predict_proba(X_lr)[:, 1]
        xgb_train_probs = xgb_model.predict(dtrain)

        def blend_loss(w):
            blended = w * lr_train_probs + (1 - w) * xgb_train_probs
            blended = np.clip(blended, 1e-6, 1 - 1e-6)
            return log_loss(y, blended)

        opt = minimize_scalar(blend_loss, bounds=(0.0, 1.0), method="bounded")
        w_lr = opt.x
        print(f"  Learned blend: LR={w_lr:.2f}, XGB={1-w_lr:.2f}")

    return lr, scaler, xgb_model, available, w_lr


def predict(lr, scaler, xgb_model, features, test_df, w_lr=0.5):
    available = [f for f in features if f in test_df.columns]
    X = test_df[available].copy()

    X_lr = scaler.transform(X.fillna(0))
    lr_probs = lr.predict_proba(X_lr)[:, 1]

    if xgb_model and HAS_XGB:
        dtest = xgb.DMatrix(X)
        xgb_probs = xgb_model.predict(dtest)
        ens_probs = w_lr * lr_probs + (1 - w_lr) * xgb_probs
    else:
        xgb_probs = lr_probs
        ens_probs = lr_probs

    return ens_probs


def compute_bet_pnl(edge, market_prob, outcome):
    if edge > 0:
        fair_odds = 1 / market_prob
        return (fair_odds - 1) * 100 if outcome == 1 else -100.0
    else:
        fair_odds = 1 / (1 - market_prob)
        return (fair_odds - 1) * 100 if outcome == 0 else -100.0


def main():
    # ── Train model on 2018-2024 ───────────────────────────────────────
    print("Loading training data (2018-2024)...")
    train_df = load_features(list(range(2018, 2025)))
    print(f"  {len(train_df)} training games")

    print("\nTraining ensemble model...")
    lr, scaler, xgb_model, features, w_lr = train_ensemble(train_df)
    print(f"  Features used: {len(features)}")

    # ── Load 2025 features ─────────────────────────────────────────────
    print("\nLoading 2025 features...")
    test_df = load_features([2025])
    print(f"  {len(test_df)} games, {len(test_df.columns)} columns")

    # ── Generate predictions ───────────────────────────────────────────
    print("\nGenerating model predictions...")
    model_probs = predict(lr, scaler, xgb_model, features, test_df, w_lr)
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
        "game_date", "home_team", "away_team", "home_win",
        "model_home_prob", "kalshi_home_prob", "model_edge", "kalshi_volume",
    ]
    non_feature_cols = {
        "game_pk", "game_date", "home_team", "away_team", "home_win",
        "home_score", "away_score", "season",
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
