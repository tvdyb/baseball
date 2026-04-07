#!/usr/bin/env python3
"""Export detailed bet log with teams, pitchers, and all odds sources."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kalshi_clean_backtest import (
    load_all_data, extract_dk_lines_from_json, aggregate_nn_features,
    get_feature_cols, american_to_prob, compute_bets,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VAL_CUTOFF = "2025-07-01"


def main():
    print("Loading data...")
    df = load_all_data()

    # Train LGB model (same as kalshi_clean_backtest)
    print("\nTraining LGB model...")
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import TimeSeriesSplit

    feat_2024 = pd.read_parquet(DATA / "features" / "game_features_2024.parquet")
    feat_2024["game_date"] = pd.to_datetime(feat_2024["game_date"]).dt.strftime("%Y-%m-%d")

    # DK 2024 lines
    dk_2024 = extract_dk_lines_from_json(2024)
    if not dk_2024.empty:
        mk = ["game_date", "home_team", "away_team"]
        feat_2024 = feat_2024.merge(dk_2024[mk + ["dk_home_prob"]], on=mk, how="left")

    # NN lineup features 2024
    nn_2024_path = DATA / "features" / "nn_features_2024.parquet"
    if nn_2024_path.exists():
        nn_raw = pd.read_parquet(nn_2024_path)
        nn_raw["game_date"] = pd.to_datetime(nn_raw["game_date"]).dt.strftime("%Y-%m-%d")
        nn_agg = aggregate_nn_features(nn_raw)
        nn_agg["game_date"] = pd.to_datetime(nn_agg["game_date"]).dt.strftime("%Y-%m-%d")
        mk = ["game_date", "home_team", "away_team"]
        nn_cols = [c for c in nn_agg.columns if c.startswith("nn_") or c in mk]
        feat_2024 = feat_2024.merge(nn_agg[nn_cols], on=mk, how="left")

    # 2025 early training data
    feat_2025_all = pd.read_parquet(DATA / "features" / "game_features_2025.parquet")
    feat_2025_all["game_date"] = pd.to_datetime(feat_2025_all["game_date"]).dt.strftime("%Y-%m-%d")

    nn_2025_path = DATA / "features" / "nn_features_2025.parquet"
    if nn_2025_path.exists():
        nn_raw = pd.read_parquet(nn_2025_path)
        nn_raw["game_date"] = pd.to_datetime(nn_raw["game_date"]).dt.strftime("%Y-%m-%d")
        nn_agg = aggregate_nn_features(nn_raw)
        nn_agg["game_date"] = pd.to_datetime(nn_agg["game_date"]).dt.strftime("%Y-%m-%d")
        mk = ["game_date", "home_team", "away_team"]
        nn_cols = [c for c in nn_agg.columns if c.startswith("nn_") or c in mk]
        feat_2025_all = feat_2025_all.merge(nn_agg[nn_cols], on=mk, how="left")

    # DK 2025 lines for training
    dk_2025 = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    dk_2025["game_date"] = pd.to_datetime(dk_2025["game_date"]).dt.strftime("%Y-%m-%d")
    dk_2025["dk_home_raw"] = american_to_prob(dk_2025["home_ml_close"])
    dk_2025["dk_away_raw"] = american_to_prob(dk_2025["away_ml_close"])
    dk_2025["dk_total"] = dk_2025["dk_home_raw"] + dk_2025["dk_away_raw"]
    dk_2025["dk_home_prob"] = dk_2025["dk_home_raw"] / dk_2025["dk_total"]
    mk = ["game_date", "home_team", "away_team"]
    feat_2025_all = feat_2025_all.merge(dk_2025[mk + ["dk_home_prob"]].dropna(), on=mk, how="left")

    feat_2025_early = feat_2025_all[feat_2025_all["game_date"] < "2025-04-16"].copy()
    train_feat = pd.concat([feat_2024, feat_2025_early], ignore_index=True)

    feat_cols = get_feature_cols(train_feat)
    common_cols = [c for c in feat_cols if c in df.columns]
    feat_cols = common_cols

    X_train = train_feat[feat_cols].copy()
    y_train = train_feat["home_win"].values.astype(int)
    missing_frac = X_train.isnull().mean()
    keep_cols = missing_frac[missing_frac < 0.3].index.tolist()
    feat_cols = keep_cols
    X_train = X_train[feat_cols]

    print(f"  Features: {len(feat_cols)}, Training games: {len(X_train)}")

    # Optuna tuning (60 trials)
    from tqdm import tqdm

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
        for tr_idx, va_idx in tscv.split(X_train):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            from sklearn.metrics import log_loss
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

    # Predict on all 2025
    df["lgb_home_prob"] = model.predict_proba(df[feat_cols])[:, 1]

    # Generate bets on test period
    test_df = df[df["game_date"] >= VAL_CUTOFF].copy()
    bets = compute_bets(test_df, "lgb_home_prob", min_edge=0.10, kelly_frac=0.25)
    print(f"\n  Test bets: {len(bets)}")

    # Merge pitcher names from games data (drop_duplicates to handle doubleheaders)
    games = pd.read_parquet(DATA / "games" / "games_2025.parquet")
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")
    games_info = games[["game_date", "home_team_abbr", "away_team_abbr",
                         "home_sp_name", "away_sp_name"]].copy()
    games_info.rename(columns={"home_team_abbr": "home_team", "away_team_abbr": "away_team"}, inplace=True)
    games_info = games_info.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")

    bets = bets.merge(games_info, on=["game_date", "home_team", "away_team"], how="left")

    # Merge Polymarket closing prices
    poly_path = DATA / "polymarket" / "poly_vs_sharp_matched.parquet"
    if poly_path.exists():
        poly = pd.read_parquet(poly_path)
        poly = poly[(poly["poly_home_prob"] >= 0.18) & (poly["poly_home_prob"] <= 0.82)]
        poly_slim = poly[["game_date", "home_team", "away_team", "poly_home_prob"]].copy()
        poly_slim = poly_slim.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")
        bets = bets.merge(poly_slim, on=["game_date", "home_team", "away_team"], how="left")
    else:
        bets["poly_home_prob"] = np.nan

    # Merge DK and LGB probs from test_df
    odds_cols = test_df[["game_date", "home_team", "away_team",
                          "dk_home_prob", "lgb_home_prob",
                          "kalshi_home_prob", "home_ml_close", "away_ml_close"]].copy()
    odds_cols = odds_cols.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")
    bets = bets.merge(odds_cols, on=["game_date", "home_team", "away_team"], how="left")

    # Build clean output
    out = pd.DataFrame()
    out["date"] = bets["game_date"]
    out["home"] = bets["home_team"]
    out["away"] = bets["away_team"]
    out["home_sp"] = bets["home_sp_name"].fillna("")
    out["away_sp"] = bets["away_sp_name"].fillna("")
    out["bet_side"] = bets["side"]
    out["dk_home"] = bets["dk_home_prob"].round(3)
    out["dk_away"] = (1 - bets["dk_home_prob"]).round(3)
    out["dk_home_ml"] = bets["home_ml_close"]
    out["dk_away_ml"] = bets["away_ml_close"]
    out["poly_home"] = bets["poly_home_prob"].round(3)
    out["poly_away"] = (1 - bets["poly_home_prob"]).where(bets["poly_home_prob"].notna()).round(3)
    out["lgb_home"] = bets["lgb_home_prob"].round(3)
    out["lgb_away"] = (1 - bets["lgb_home_prob"]).round(3)
    out["kalshi_home"] = bets["kalshi_home_prob"].round(3)
    out["kalshi_away"] = (1 - bets["kalshi_home_prob"]).round(3)
    out["edge"] = bets["edge"].round(3)
    out["won"] = bets["won"]
    out["pnl"] = bets["pnl_flat"].round(3)
    out["result"] = bets["home_win"].map({1: "home_win", 0: "away_win"})

    out = out.sort_values("date").reset_index(drop=True)

    # Save
    out_path = ROOT / "outputs" / "bet_log_2025.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\n  Saved {len(out)} bets to {out_path}")
    print(f"\n  Summary: {len(out)} bets, WR={out['won'].mean():.1%}, ROI={out['pnl'].mean():+.1%}, PnL={out['pnl'].sum():+.1f}")
    print(f"  Bets with Polymarket odds: {out['poly_home'].notna().sum()}")
    print(f"\n  Sample rows:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
