#!/usr/bin/env python3
"""Train LGB on 2024+2025 and predict 2026 game win probabilities."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kalshi_clean_backtest import (
    extract_dk_lines_from_json, aggregate_nn_features,
    get_feature_cols, american_to_prob,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main():
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import TimeSeriesSplit
    from tqdm import tqdm

    # ── Load 2024 features + DK + NN ──
    print("Loading 2024 features...")
    feat_2024 = pd.read_parquet(DATA / "features" / "game_features_2024.parquet")
    feat_2024["game_date"] = pd.to_datetime(feat_2024["game_date"]).dt.strftime("%Y-%m-%d")

    dk_2024 = extract_dk_lines_from_json(2024)
    if not dk_2024.empty:
        mk = ["game_date", "home_team", "away_team"]
        feat_2024 = feat_2024.merge(dk_2024[mk + ["dk_home_prob"]], on=mk, how="left")

    nn_2024_path = DATA / "features" / "nn_features_2024.parquet"
    if nn_2024_path.exists():
        nn_raw = pd.read_parquet(nn_2024_path)
        nn_raw["game_date"] = pd.to_datetime(nn_raw["game_date"]).dt.strftime("%Y-%m-%d")
        nn_agg = aggregate_nn_features(nn_raw)
        nn_agg["game_date"] = pd.to_datetime(nn_agg["game_date"]).dt.strftime("%Y-%m-%d")
        mk = ["game_date", "home_team", "away_team"]
        nn_cols = [c for c in nn_agg.columns if c.startswith("nn_") or c in mk]
        feat_2024 = feat_2024.merge(nn_agg[nn_cols], on=mk, how="left")

    # ── Load ALL 2025 features + DK + NN (full season for training) ──
    print("Loading 2025 features...")
    feat_2025 = pd.read_parquet(DATA / "features" / "game_features_2025.parquet")
    feat_2025["game_date"] = pd.to_datetime(feat_2025["game_date"]).dt.strftime("%Y-%m-%d")

    nn_2025_path = DATA / "features" / "nn_features_2025.parquet"
    if nn_2025_path.exists():
        nn_raw = pd.read_parquet(nn_2025_path)
        nn_raw["game_date"] = pd.to_datetime(nn_raw["game_date"]).dt.strftime("%Y-%m-%d")
        nn_agg = aggregate_nn_features(nn_raw)
        nn_agg["game_date"] = pd.to_datetime(nn_agg["game_date"]).dt.strftime("%Y-%m-%d")
        mk = ["game_date", "home_team", "away_team"]
        nn_cols = [c for c in nn_agg.columns if c.startswith("nn_") or c in mk]
        feat_2025 = feat_2025.merge(nn_agg[nn_cols], on=mk, how="left")

    dk_2025 = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    dk_2025["game_date"] = pd.to_datetime(dk_2025["game_date"]).dt.strftime("%Y-%m-%d")
    dk_2025["dk_home_raw"] = american_to_prob(dk_2025["home_ml_close"])
    dk_2025["dk_away_raw"] = american_to_prob(dk_2025["away_ml_close"])
    dk_2025["dk_total"] = dk_2025["dk_home_raw"] + dk_2025["dk_away_raw"]
    dk_2025["dk_home_prob"] = dk_2025["dk_home_raw"] / dk_2025["dk_total"]
    mk = ["game_date", "home_team", "away_team"]
    feat_2025 = feat_2025.merge(dk_2025[mk + ["dk_home_prob"]].dropna(), on=mk, how="left")

    # ── Load 2026 features + DK + NN ──
    print("Loading 2026 features...")
    feat_2026 = pd.read_parquet(DATA / "features" / "game_features_2026.parquet")
    feat_2026["game_date"] = pd.to_datetime(feat_2026["game_date"]).dt.strftime("%Y-%m-%d")

    nn_2026_path = DATA / "features" / "nn_features_2026.parquet"
    if nn_2026_path.exists():
        nn_raw = pd.read_parquet(nn_2026_path)
        nn_raw["game_date"] = pd.to_datetime(nn_raw["game_date"]).dt.strftime("%Y-%m-%d")
        nn_agg = aggregate_nn_features(nn_raw)
        nn_agg["game_date"] = pd.to_datetime(nn_agg["game_date"]).dt.strftime("%Y-%m-%d")
        mk = ["game_date", "home_team", "away_team"]
        nn_cols = [c for c in nn_agg.columns if c.startswith("nn_") or c in mk]
        feat_2026 = feat_2026.merge(nn_agg[nn_cols], on=mk, how="left")

    # Merge DK 2026 odds
    dk_2026 = pd.read_parquet(DATA / "odds" / "sbr_ml_2026.parquet")
    dk_2026["game_date"] = pd.to_datetime(dk_2026["game_date"]).dt.strftime("%Y-%m-%d")
    mk = ["game_date", "home_team", "away_team"]
    feat_2026 = feat_2026.merge(dk_2026[mk + ["dk_home_prob"]].dropna(), on=mk, how="left")

    # ── Combine training data: ALL of 2024 + ALL of 2025 ──
    train_feat = pd.concat([feat_2024, feat_2025], ignore_index=True)
    print(f"  Training games: {len(train_feat)} (2024: {len(feat_2024)}, 2025: {len(feat_2025)})")

    # ── Feature selection ──
    feat_cols = get_feature_cols(train_feat)
    # Only keep features present in 2026
    common_cols = [c for c in feat_cols if c in feat_2026.columns]
    feat_cols = common_cols

    X_train = train_feat[feat_cols].copy()
    y_train = train_feat["home_win"].values.astype(int)
    missing_frac = X_train.isnull().mean()
    keep_cols = missing_frac[missing_frac < 0.3].index.tolist()
    feat_cols = keep_cols
    X_train = X_train[feat_cols]

    print(f"  Features: {len(feat_cols)}")

    # ── Optuna tuning ──
    print("\nTuning LGB (60 trials)...")

    def objective(trial):
        params = {
            "objective": "binary", "verbose": -1, "random_state": 42, "n_jobs": -1,
            "n_estimators": trial.suggest_int("n_estimators", 20, 250),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "num_leaves": trial.suggest_int("num_leaves", 3, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 30.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
        }
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        from sklearn.metrics import log_loss
        for tr_idx, va_idx in tscv.split(X_train):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            scores.append(log_loss(y_train[va_idx], m.predict_proba(X_train.iloc[va_idx])[:, 1]))
        return np.mean(scores)

    pbar = tqdm(total=60, desc="Optuna")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=60, callbacks=[lambda s, t: pbar.update(1)])
    pbar.close()

    best = study.best_params
    best.update({"objective": "binary", "verbose": -1, "random_state": 42, "n_jobs": -1})
    model = lgb.LGBMClassifier(**best)
    model.fit(X_train, y_train)
    print(f"  Best CV log-loss: {study.best_value:.5f}")

    # Save feature importances
    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_path = DATA / "features" / "lgb_feature_importances.parquet"
    imp_df.to_parquet(imp_path, index=False)
    print(f"  Saved feature importances to {imp_path}")

    # ── Predict 2026 ──
    X_2026 = feat_2026[feat_cols].copy()
    feat_2026["lgb_home_prob"] = model.predict_proba(X_2026)[:, 1]
    feat_2026["lgb_away_prob"] = 1 - feat_2026["lgb_home_prob"]

    # Filter to regular season (exclude spring training)
    feat_2026 = feat_2026[feat_2026["game_date"] >= "2026-03-25"].copy()

    print(f"\n  2026 predictions: {len(feat_2026)} games")
    print(f"  LGB prob range: {feat_2026['lgb_home_prob'].min():.3f} - {feat_2026['lgb_home_prob'].max():.3f}")
    print(f"  LGB prob mean: {feat_2026['lgb_home_prob'].mean():.3f}")

    # Save predictions
    out_cols = ["game_date", "home_team", "away_team", "home_win",
                "dk_home_prob", "lgb_home_prob", "lgb_away_prob"]
    out = feat_2026[out_cols].copy()
    out_path = DATA / "features" / "lgb_predictions_2026.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Saved to {out_path}")

    # Show sample
    print("\n  Sample predictions:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
