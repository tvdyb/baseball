#!/usr/bin/env python3
"""
Plot LGB Model Bets for 2025 Test Period
=========================================

Loads data and trains the LGB model identically to kalshi_clean_backtest.py,
then generates a two-panel chart of test-period bets (edge > 10%):
  - Top: scatter of LGB prob vs Kalshi price, colored by win/loss
  - Bottom: cumulative P&L over time (flat $1 bets)

Saves to: outputs/bets_2025.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kalshi_clean_backtest import (
    load_all_data,
    extract_dk_lines_from_json,
    aggregate_nn_features,
    get_feature_cols,
    compute_bets,
    american_to_prob,
    VAL_CUTOFF,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

MIN_EDGE = 0.10  # 10% edge threshold


def train_win_model_fast(feat_2024, feat_2025_val, n_trials=60):
    """Train LGB win model — same as kalshi_clean_backtest but with fewer Optuna trials."""
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    feat_cols = get_feature_cols(feat_2024)
    common_cols = [c for c in feat_cols if c in feat_2025_val.columns]
    feat_cols = common_cols

    X_train = feat_2024[feat_cols].copy()
    y_train = feat_2024["home_win"].values.astype(int)

    # Drop columns with >30% missing
    missing_frac = X_train.isnull().mean()
    keep_cols = missing_frac[missing_frac < 0.3].index.tolist()
    feat_cols = keep_cols
    X_train = X_train[feat_cols]

    print(f"  Training features: {len(feat_cols)}, Training games: {len(X_train)}")

    def objective(trial):
        params = {
            "objective": "binary",
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
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        }
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for tr_idx, va_idx in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, y_tr)
            from sklearn.metrics import log_loss
            preds = m.predict_proba(X_va)[:, 1]
            scores.append(log_loss(y_va, preds))
        return np.mean(scores)

    from tqdm import tqdm
    pbar = tqdm(total=n_trials, desc="Optuna HP tuning")
    study = optuna.create_study(direction="minimize")

    def callback(study, trial):
        pbar.update(1)
        pbar.set_description(f"Best: {study.best_value:.5f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    pbar.close()

    best = study.best_params
    best.update({"objective": "binary", "verbose": -1, "random_state": 42, "n_jobs": -1})
    print(f"  Best CV log-loss: {study.best_value:.5f}")

    model = lgb.LGBMClassifier(**best)
    model.fit(X_train, y_train)

    return model, feat_cols


def main():
    print("=" * 70)
    print("PLOT LGB BETS — 2025 Test Period")
    print("=" * 70)

    # ── Load data (same as kalshi_clean_backtest) ──
    print("\n[1] Loading data...")
    df = load_all_data()

    # ── Train LGB model (same pipeline, fewer trials) ──
    print("\n[2] Training LGB win model...")
    feat_2024_path = DATA / "features" / "game_features_2024.parquet"
    feat_2025_path = DATA / "features" / "game_features_2025.parquet"

    if not feat_2024_path.exists():
        print("  ERROR: No 2024 features available — cannot train LGB model")
        sys.exit(1)

    feat_2024 = pd.read_parquet(feat_2024_path)
    feat_2024["game_date"] = pd.to_datetime(feat_2024["game_date"]).dt.strftime("%Y-%m-%d")

    # Merge DK 2024 lines
    dk_2024 = extract_dk_lines_from_json(2024)
    if not dk_2024.empty:
        mk = ["game_date", "home_team", "away_team"]
        feat_2024 = feat_2024.merge(dk_2024[mk + ["dk_home_prob"]], on=mk, how="left")
        print(f"  2024: {feat_2024['dk_home_prob'].notna().sum()}/{len(feat_2024)} with DK lines")

    # Merge NN lineup features into 2024
    nn_feat_2024_path = DATA / "features" / "nn_features_2024.parquet"
    if nn_feat_2024_path.exists():
        nn_raw_2024 = pd.read_parquet(nn_feat_2024_path)
        nn_raw_2024["game_date"] = pd.to_datetime(nn_raw_2024["game_date"]).dt.strftime("%Y-%m-%d")
        nn_agg_2024 = aggregate_nn_features(nn_raw_2024)
        nn_agg_2024["game_date"] = pd.to_datetime(nn_agg_2024["game_date"]).dt.strftime("%Y-%m-%d")
        mk = ["game_date", "home_team", "away_team"]
        nn_agg_cols = [c for c in nn_agg_2024.columns if c.startswith("nn_") or c in mk]
        nn_agg_2024 = nn_agg_2024[nn_agg_cols].copy()
        feat_2024 = feat_2024.merge(nn_agg_2024, on=mk, how="left")

    # Add early 2025 to training
    if feat_2025_path.exists():
        feat_2025_all = pd.read_parquet(feat_2025_path)
        feat_2025_all["game_date"] = pd.to_datetime(feat_2025_all["game_date"]).dt.strftime("%Y-%m-%d")

        # Merge NN features into 2025 training data
        nn_feat_2025_path = DATA / "features" / "nn_features_2025.parquet"
        if nn_feat_2025_path.exists():
            nn_raw_2025 = pd.read_parquet(nn_feat_2025_path)
            nn_raw_2025["game_date"] = pd.to_datetime(nn_raw_2025["game_date"]).dt.strftime("%Y-%m-%d")
            nn_agg_2025 = aggregate_nn_features(nn_raw_2025)
            nn_agg_2025["game_date"] = pd.to_datetime(nn_agg_2025["game_date"]).dt.strftime("%Y-%m-%d")
            mk = ["game_date", "home_team", "away_team"]
            nn_agg_cols = [c for c in nn_agg_2025.columns if c.startswith("nn_") or c in mk]
            nn_agg_2025 = nn_agg_2025[nn_agg_cols].copy()
            feat_2025_all = feat_2025_all.merge(nn_agg_2025, on=mk, how="left")

        # Merge DK 2025 lines
        dk_2025_path = DATA / "odds" / "sbr_mlb_2025.parquet"
        if dk_2025_path.exists():
            dk_2025 = pd.read_parquet(dk_2025_path)
            dk_2025["game_date"] = pd.to_datetime(dk_2025["game_date"]).dt.strftime("%Y-%m-%d")
            dk_2025["dk_home_raw"] = american_to_prob(dk_2025["home_ml_close"])
            dk_2025["dk_away_raw"] = american_to_prob(dk_2025["away_ml_close"])
            dk_2025["dk_total"] = dk_2025["dk_home_raw"] + dk_2025["dk_away_raw"]
            dk_2025["dk_home_prob"] = dk_2025["dk_home_raw"] / dk_2025["dk_total"]
            mk = ["game_date", "home_team", "away_team"]
            feat_2025_all = feat_2025_all.merge(dk_2025[mk + ["dk_home_prob"]].dropna(), on=mk, how="left")

        feat_2025_early = feat_2025_all[feat_2025_all["game_date"] < "2025-04-16"].copy()
        train_feat = pd.concat([feat_2024, feat_2025_early], ignore_index=True)
        print(f"  Training: {len(feat_2024)} (2024) + {len(feat_2025_early)} (early 2025) = {len(train_feat)}")
    else:
        train_feat = feat_2024
        print(f"  Training: {len(feat_2024)} (2024 only)")

    model, feat_cols = train_win_model_fast(train_feat, df, n_trials=60)

    # Generate LGB predictions for all 2025 games
    if all(c in df.columns for c in feat_cols):
        X_2025 = df[feat_cols].copy()
        df["lgb_home_prob"] = model.predict_proba(X_2025)[:, 1]
        print(f"  LGB predictions: {df['lgb_home_prob'].notna().sum()} games")
    else:
        print("  ERROR: Feature mismatch — cannot generate predictions")
        sys.exit(1)

    # ── Filter to test period ──
    test_df = df[df["game_date"] >= VAL_CUTOFF].copy()
    print(f"\n[3] Test period: {len(test_df)} games ({test_df['game_date'].min()} to {test_df['game_date'].max()})")

    # ── Generate bets at 10% edge ──
    bets = compute_bets(test_df, "lgb_home_prob", min_edge=MIN_EDGE)
    print(f"  Bets at edge>{MIN_EDGE:.0%}: {len(bets)}")

    if bets.empty:
        print("  No bets generated — nothing to plot.")
        sys.exit(0)

    n_bets = len(bets)
    win_rate = bets["won"].mean()
    total_pnl = bets["pnl_flat"].sum()
    roi = bets["pnl_flat"].mean()

    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Flat ROI: {roi:+.1%}")
    print(f"  Total P&L: {total_pnl:+.2f} units")

    # ── Build chart ──
    print("\n[4] Creating chart...")
    bets["game_date_dt"] = pd.to_datetime(bets["game_date"])
    bets = bets.sort_values("game_date_dt").reset_index(drop=True)
    bets["cum_pnl"] = bets["pnl_flat"].cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle(
        f"LGB Model Bets — 2025 Test Period (>={VAL_CUTOFF})\n"
        f"N={n_bets} bets | Win Rate={win_rate:.1%} | ROI={roi:+.1%} | Total P&L={total_pnl:+.2f} units",
        fontsize=14, fontweight="bold",
    )

    # ── Top panel: scatter of LGB prob vs Kalshi price ──
    wins = bets[bets["won"] == 1]
    losses = bets[bets["won"] == 0]

    ax1.scatter(wins["our_prob"], wins["kalshi_price"], c="green", alpha=0.6,
                edgecolors="darkgreen", s=40, label=f"Win ({len(wins)})", zorder=3)
    ax1.scatter(losses["our_prob"], losses["kalshi_price"], c="red", alpha=0.6,
                edgecolors="darkred", s=40, label=f"Loss ({len(losses)})", zorder=3)

    # Diagonal y=x reference line
    lims = [
        min(bets["our_prob"].min(), bets["kalshi_price"].min()) - 0.02,
        max(bets["our_prob"].max(), bets["kalshi_price"].max()) + 0.02,
    ]
    ax1.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="y = x")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("LGB Model Probability", fontsize=12)
    ax1.set_ylabel("Kalshi Price", fontsize=12)
    ax1.set_title("LGB Model vs Kalshi Price — Each Bet", fontsize=13)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Bottom panel: cumulative P&L ──
    ax2.plot(bets["game_date_dt"], bets["cum_pnl"], color="steelblue", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax2.fill_between(bets["game_date_dt"], bets["cum_pnl"], 0,
                     where=bets["cum_pnl"] >= 0, color="green", alpha=0.15)
    ax2.fill_between(bets["game_date_dt"], bets["cum_pnl"], 0,
                     where=bets["cum_pnl"] < 0, color="red", alpha=0.15)
    ax2.set_xlabel("Game Date", fontsize=12)
    ax2.set_ylabel("Cumulative P&L ($1 flat bets)", fontsize=12)
    ax2.set_title("Cumulative P&L (Flat $1 Bets)", fontsize=13)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = OUTPUT / "bets_2025.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart saved to: {out_path}")


if __name__ == "__main__":
    main()
