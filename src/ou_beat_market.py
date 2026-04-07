#!/usr/bin/env python3
"""
O/U Model: Beat the Market on Total Runs
=========================================

Goal: Build a model that predicts total runs better than DK closing lines,
then trade against Polymarket/Kalshi O/U contracts where DK-like pricing
is the benchmark.

Key insight: DK P(over) BSS is only +0.0008 vs base rate, and the market
systematically overprices overs by 3-5%. A model that corrects this bias
AND captures game-specific pitcher/lineup factors can generate edge.

Approach:
  1. LightGBM regression predicting total_runs
  2. Include ou_close as a feature (we're predicting the residual)
  3. Convert predictions to P(over line) using empirical residual CDF
  4. Compare model P(over) vs DK devigged P(over) — this is the edge
  5. Edge-based betting with Kelly sizing

Zero look-ahead bias:
  TRAIN:      2024 season
  VALIDATION: 2025-04-01 to 2025-06-30
  TEST:       2025-07-01 to 2025-10-29

Usage:
    python src/ou_beat_market.py
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VAL_CUTOFF = "2025-07-01"

# ═══════════════════════════════════════════════════════════════════
# Feature lists — focused on total-runs prediction
# ═══════════════════════════════════════════════════════════════════

# Level features: both sides matter for total runs (not just differential)
LEVEL_FEATURES = [
    # Park & weather — biggest total-runs drivers
    "park_factor", "temperature", "wind_speed", "is_dome", "wind_out", "wind_in",
    # Both SPs — total runs depends on quality of BOTH pitchers
    "home_sp_xrv_mean", "away_sp_xrv_mean",
    "home_sp_k_rate", "away_sp_k_rate",
    "home_sp_bb_rate", "away_sp_bb_rate",
    "home_sp_composite_score", "away_sp_composite_score",
    "home_sp_stuff_score", "away_sp_stuff_score",
    "home_sp_xrv_std", "away_sp_xrv_std",
    "home_sp_xrv_trend", "away_sp_xrv_trend",
    "home_sp_rest_days", "away_sp_rest_days",
    "home_sp_overperf", "away_sp_overperf",
    "home_sp_overperf_recent", "away_sp_overperf_recent",
    "home_sp_n_pitches", "away_sp_n_pitches",
    "home_sp_avg_velo", "away_sp_avg_velo",
    "home_sp_velo_trend", "away_sp_velo_trend",
    # Both lineups
    "home_hit_xrv_mean", "away_hit_xrv_mean",
    "home_hit_k_rate", "away_hit_k_rate",
    "home_hit_barrel_rate", "away_hit_barrel_rate",
    "home_hit_hard_hit_rate", "away_hit_hard_hit_rate",
    "home_hit_xrv_contact", "away_hit_xrv_contact",
    # Both bullpens
    "home_bp_xrv_mean", "away_bp_xrv_mean",
    "home_bp_fatigue_score", "away_bp_fatigue_score",
    # Matchups
    "home_matchup_xrv_mean", "away_matchup_xrv_mean",
    # Season context
    "days_into_season", "is_night",
]

# Sum features — total scoring environment
SUM_FEATURES = [
    "sum_sp_xrv", "sum_sp_k_rate", "sum_sp_bb_rate",
    "sum_hit_xrv", "sum_bp_xrv",
    "sum_sp_composite", "sum_sp_stuff",
    "sum_hit_barrel", "sum_hit_hard_hit",
]

# Market features — use the line itself as a feature
MARKET_FEATURES = [
    "ou_close",
    "ou_movement",     # ou_close - ou_open
    "dk_over_fair",    # devigged P(over)
]


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def american_to_prob(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100),
                    np.where(odds > 0, 100 / (odds + 100), np.nan))


def load_data(year):
    """Load features + DK O/U odds for a given year."""
    feat = pd.read_parquet(DATA / "features" / f"game_features_{year}.parquet")
    feat["game_date"] = pd.to_datetime(feat["game_date"]).dt.strftime("%Y-%m-%d")

    dk = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    dk["game_date"] = pd.to_datetime(dk["game_date"]).dt.strftime("%Y-%m-%d")

    # Only keep games from the target year
    if year == 2024:
        dk = dk[dk["game_date"] < "2025-01-01"]
    elif year == 2025:
        dk = dk[dk["game_date"] >= "2025-01-01"]

    dk = dk.dropna(subset=["ou_close", "total_runs", "over_close_odds", "under_close_odds"])

    # Devig
    dk["dk_over_raw"] = american_to_prob(dk["over_close_odds"])
    dk["dk_under_raw"] = american_to_prob(dk["under_close_odds"])
    dk["dk_vig"] = dk["dk_over_raw"] + dk["dk_under_raw"]
    dk["dk_over_fair"] = dk["dk_over_raw"] / dk["dk_vig"]
    dk["ou_movement"] = dk["ou_close"] - dk["ou_open"]

    dk_cols = ["game_date", "home_team", "away_team", "ou_close", "ou_open",
               "ou_movement", "dk_over_fair", "over_close_odds", "under_close_odds",
               "total_runs"]

    merged = feat.merge(dk[dk_cols].drop_duplicates(),
                       on=["game_date", "home_team", "away_team"], how="inner")
    merged["total_runs"] = merged["total_runs"].astype(float)
    merged["actual_over"] = (merged["total_runs"] > merged["ou_close"]).astype(int)
    merged["is_push"] = merged["total_runs"] == merged["ou_close"]

    # Build sum features
    if "home_sp_xrv_mean" in merged.columns and "away_sp_xrv_mean" in merged.columns:
        merged["sum_sp_xrv"] = merged["home_sp_xrv_mean"].fillna(0) + merged["away_sp_xrv_mean"].fillna(0)
    if "home_sp_k_rate" in merged.columns and "away_sp_k_rate" in merged.columns:
        merged["sum_sp_k_rate"] = merged["home_sp_k_rate"].fillna(0) + merged["away_sp_k_rate"].fillna(0)
    if "home_sp_bb_rate" in merged.columns and "away_sp_bb_rate" in merged.columns:
        merged["sum_sp_bb_rate"] = merged["home_sp_bb_rate"].fillna(0) + merged["away_sp_bb_rate"].fillna(0)
    if "home_hit_xrv_mean" in merged.columns and "away_hit_xrv_mean" in merged.columns:
        merged["sum_hit_xrv"] = merged["home_hit_xrv_mean"].fillna(0) + merged["away_hit_xrv_mean"].fillna(0)
    if "home_bp_xrv_mean" in merged.columns and "away_bp_xrv_mean" in merged.columns:
        merged["sum_bp_xrv"] = merged["home_bp_xrv_mean"].fillna(0) + merged["away_bp_xrv_mean"].fillna(0)
    if "home_sp_composite_score" in merged.columns and "away_sp_composite_score" in merged.columns:
        merged["sum_sp_composite"] = merged["home_sp_composite_score"].fillna(0) + merged["away_sp_composite_score"].fillna(0)
    if "home_sp_stuff_score" in merged.columns and "away_sp_stuff_score" in merged.columns:
        merged["sum_sp_stuff"] = merged["home_sp_stuff_score"].fillna(0) + merged["away_sp_stuff_score"].fillna(0)
    if "home_hit_barrel_rate" in merged.columns and "away_hit_barrel_rate" in merged.columns:
        merged["sum_hit_barrel"] = merged["home_hit_barrel_rate"].fillna(0) + merged["away_hit_barrel_rate"].fillna(0)
    if "home_hit_hard_hit_rate" in merged.columns and "away_hit_hard_hit_rate" in merged.columns:
        merged["sum_hit_hard_hit"] = merged["home_hit_hard_hit_rate"].fillna(0) + merged["away_hit_hard_hit_rate"].fillna(0)

    print(f"  {year}: {len(merged)} games with features + O/U odds")
    return merged


def get_feature_cols(df):
    """Get all available feature columns."""
    all_feats = LEVEL_FEATURES + SUM_FEATURES + MARKET_FEATURES
    available = [c for c in all_feats if c in df.columns]
    # Drop features with >40% missing
    good = [c for c in available if df[c].notna().mean() > 0.6]
    return good


# ═══════════════════════════════════════════════════════════════════
# Model Training
# ═══════════════════════════════════════════════════════════════════

def train_ou_model(train_df, feat_cols):
    """Train LGB regression on total_runs with Optuna TSCV on training data only."""
    X_train = train_df[feat_cols].copy()
    y_train = train_df["total_runs"].values

    print(f"  Features: {len(feat_cols)}, Training games: {len(X_train)}")

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "n_estimators": trial.suggest_int("n_estimators", 30, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "num_leaves": trial.suggest_int("num_leaves", 4, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 30.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 50.0, log=True),
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
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_va)
            scores.append(mean_absolute_error(y_va, preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=150, show_progress_bar=True)

    best = study.best_params
    best.update({"objective": "regression", "metric": "mae",
                 "verbose": -1, "random_state": 42, "n_jobs": -1})
    print(f"  Best CV MAE: {study.best_value:.3f}")

    # Train final model on ALL training data
    model = lgb.LGBMRegressor(**best)
    model.fit(X_train, y_train)

    # Compute residual distribution from training data (for CDF conversion)
    train_preds = model.predict(X_train)
    residuals = y_train - train_preds
    residual_std = np.std(residuals)
    print(f"  Residual std: {residual_std:.2f}")

    return model, residual_std


def predict_p_over(model, df, feat_cols, residual_std):
    """Convert model predictions to P(over line) using Gaussian CDF."""
    preds = model.predict(df[feat_cols])
    lines = df["ou_close"].values

    # P(total > line) = 1 - Phi((line + 0.5 - pred) / sigma)
    # The +0.5 accounts for the half-run in the line (e.g., over 8.5 means total >= 9)
    p_over = 1.0 - norm.cdf(lines + 0.5, loc=preds, scale=residual_std)
    return preds, p_over


# ═══════════════════════════════════════════════════════════════════
# Betting Logic
# ═══════════════════════════════════════════════════════════════════

def compute_ou_bets(df, model_p_over_col, min_edge, kelly_frac=0.25):
    """
    Generate O/U bets. Compare model P(over) vs DK devigged P(over).
    If model says more likely over than DK → bet over.
    If model says more likely under than DK → bet under.
    """
    bets = []
    for _, row in df.iterrows():
        if row["is_push"]:
            continue
        model_p = row[model_p_over_col]
        mkt_p = row["dk_over_fair"]
        if pd.isna(model_p) or pd.isna(mkt_p):
            continue

        edge_over = model_p - mkt_p
        edge_under = (1 - model_p) - (1 - mkt_p)  # = mkt_p - model_p = -edge_over

        if edge_over > min_edge:
            side = "over"
            edge = edge_over
            our_p = model_p
            buy_price = mkt_p  # proxy for Kalshi/Poly price
            won = row["actual_over"] == 1
        elif -edge_over > min_edge:  # edge_under > min_edge
            side = "under"
            edge = -edge_over
            our_p = 1 - model_p
            buy_price = 1 - mkt_p
            won = row["actual_over"] == 0
        else:
            continue

        # P&L: buy at market price, get $1 if right
        pnl = (1.0 - buy_price) if won else -buy_price

        # Kelly
        b = (1.0 - buy_price) / buy_price if buy_price > 0 else 0
        q = 1.0 - our_p
        kelly_f = max(0.0, (our_p * b - q) / b) * kelly_frac if b > 0 else 0
        kelly_f = min(kelly_f, 0.05)

        bets.append({
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "ou_line": row["ou_close"],
            "side": side,
            "edge": edge,
            "our_prob": our_p,
            "market_price": buy_price,
            "actual_total": row["total_runs"],
            "pnl_flat": pnl,
            "kelly_f": kelly_f,
            "won": int(won),
        })
    return pd.DataFrame(bets)


def run_backtest(bets_df, starting_bankroll=10000):
    """Run Kelly-sized backtest."""
    if bets_df.empty:
        return {"n_bets": 0, "sharpe": 0, "flat_roi": 0, "win_rate": 0,
                "total_return": 0, "max_dd": 0}

    bankroll = starting_bankroll
    daily_pnl = []
    for date, day_bets in bets_df.groupby("game_date"):
        day_total = 0
        for _, bet in day_bets.iterrows():
            stake = bankroll * bet["kelly_f"]
            if stake <= 0:
                continue
            if bet["won"]:
                dollar_pnl = stake * (1.0 - bet["market_price"]) / bet["market_price"]
            else:
                dollar_pnl = -stake
            bankroll += dollar_pnl
            day_total += dollar_pnl
        daily_pnl.append(day_total)

    daily_pnl = np.array(daily_pnl)
    n = len(bets_df)
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(180) if daily_pnl.std() > 0 else 0

    equity = np.array([starting_bankroll] + list(np.cumsum(daily_pnl) + starting_bankroll))
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min()

    return {
        "n_bets": n,
        "win_rate": bets_df["won"].mean(),
        "flat_roi": bets_df["pnl_flat"].mean(),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": (bankroll - starting_bankroll) / starting_bankroll,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("O/U BEAT THE MARKET — Total Runs Model vs DK Lines")
    print("=" * 70)

    # ── Load data ──
    print("\n[1] Loading data...")
    # 2024 for training, 2025 for val+test
    # Note: SBR only has 2025 DK odds, so we use 2025 data split temporally
    df_2025 = load_data(2025)

    # Filter to regular season (Apr 1+)
    df_2025 = df_2025[df_2025["game_date"] >= "2025-04-01"].copy()
    df_2025 = df_2025.sort_values("game_date").reset_index(drop=True)

    feat_cols = get_feature_cols(df_2025)
    print(f"  Available features: {len(feat_cols)}")

    # ── Temporal split ──
    train_mask = df_2025["game_date"] < VAL_CUTOFF
    # Use first 60% of pre-July data for training, last 40% for within-period validation
    pre_july = df_2025[train_mask]
    train_cutoff_idx = int(len(pre_july) * 0.6)
    train_dates = pre_july.iloc[:train_cutoff_idx]["game_date"].max()

    train_df = df_2025[df_2025["game_date"] <= train_dates].copy()
    val_df = df_2025[(df_2025["game_date"] > train_dates) & (df_2025["game_date"] < VAL_CUTOFF)].copy()
    test_df = df_2025[df_2025["game_date"] >= VAL_CUTOFF].copy()

    print(f"\n[2] Temporal splits:")
    print(f"  Train: {len(train_df)} games ({train_df['game_date'].min()} to {train_df['game_date'].max()})")
    print(f"  Val:   {len(val_df)} games ({val_df['game_date'].min()} to {val_df['game_date'].max()})")
    print(f"  Test:  {len(test_df)} games ({test_df['game_date'].min()} to {test_df['game_date'].max()})")

    # ── Train model ──
    print("\n[3] Training O/U regression model (Optuna, 150 trials)...")
    model, residual_std = train_ou_model(train_df, feat_cols)

    # ── Generate predictions ──
    print("\n[4] Generating predictions...")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        preds, p_over = predict_p_over(model, split_df, feat_cols, residual_std)
        split_df["model_pred_total"] = preds
        split_df["model_p_over"] = p_over
        # Clip to reasonable range
        split_df["model_p_over"] = split_df["model_p_over"].clip(0.05, 0.95)

        # Quality metrics
        non_push = ~split_df["is_push"]
        if non_push.sum() > 50:
            y = split_df.loc[non_push, "actual_over"].values
            p_model = split_df.loc[non_push, "model_p_over"].values
            p_dk = split_df.loc[non_push, "dk_over_fair"].values

            brier_model = brier_score_loss(y, p_model)
            brier_dk = brier_score_loss(y, p_dk)
            brier_naive = brier_score_loss(y, np.full(len(y), y.mean()))
            bss_model = 1 - brier_model / brier_naive
            bss_dk = 1 - brier_dk / brier_naive
            bss_vs_dk = 1 - brier_model / brier_dk

            corr_model = np.corrcoef(preds, split_df["total_runs"].values)[0, 1]
            corr_dk = np.corrcoef(split_df["ou_close"].values, split_df["total_runs"].values)[0, 1]
            mae_model = mean_absolute_error(split_df["total_runs"], preds)
            mae_dk = mean_absolute_error(split_df["total_runs"], split_df["ou_close"])

            print(f"\n  {split_name} ({len(split_df)} games):")
            print(f"    Total runs — Model: corr={corr_model:.4f} MAE={mae_model:.2f} | DK: corr={corr_dk:.4f} MAE={mae_dk:.2f}")
            print(f"    P(over) BSS — Model: {bss_model:+.4f} | DK: {bss_dk:+.4f} | Model vs DK: {bss_vs_dk:+.4f}")
            print(f"    Actual over rate: {y.mean():.4f} | Model mean P(over): {p_model.mean():.4f} | DK mean P(over): {p_dk.mean():.4f}")

    # ── Naive under strategy (exploit market bias) ──
    print("\n[5] Naive under strategy (always bet under at DK odds)...")
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        non_push = split_df[~split_df["is_push"]]
        under_wr = (non_push["actual_over"] == 0).mean()
        # P&L: buy under at (1 - dk_over_fair) price
        under_pnl = non_push.apply(
            lambda r: (1.0 - (1 - r["dk_over_fair"])) if r["actual_over"] == 0 else -(1 - r["dk_over_fair"]),
            axis=1
        )
        print(f"  {split_name}: {len(non_push)} games, under WR={under_wr:.3f}, under ROI={under_pnl.mean():+.3f}")

    # ── Validation: optimize edge threshold ──
    print("\n[6] Validation: optimizing edge threshold...")
    print(f"  {'Edge':>6} {'Kelly':>6} {'Bets':>6} {'WR':>7} {'ROI':>8} {'Sharpe':>8} {'Over%':>7}")

    best_score = -999
    best_edge = None
    best_kelly = None

    for me in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        for kf in [0.10, 0.15, 0.20, 0.25, 0.30]:
            bets = compute_ou_bets(val_df, "model_p_over", min_edge=me, kelly_frac=kf)
            if len(bets) < 15:
                continue
            m = run_backtest(bets)
            score = m["sharpe"] * min(1.0, len(bets) / 50.0)
            over_pct = (bets["side"] == "over").mean() if len(bets) > 0 else 0

            if me in [0.01, 0.02, 0.03, 0.05, 0.08] and kf == 0.25:
                print(f"  {me:>5.0%} {kf:>5.0%} {m['n_bets']:>6} {m['win_rate']:>6.1%} "
                      f"{m['flat_roi']:>+7.1%} {m['sharpe']:>7.2f} {over_pct:>6.1%}")

            if score > best_score:
                best_score = score
                best_edge = me
                best_kelly = kf

    if best_edge is None:
        print("  No profitable configuration found on validation.")
        return

    # Show best config on validation
    bets_val = compute_ou_bets(val_df, "model_p_over", min_edge=best_edge, kelly_frac=best_kelly)
    m_val = run_backtest(bets_val)
    print(f"\n  BEST: edge>{best_edge:.0%} kelly={best_kelly:.0%}")
    print(f"    Val: {m_val['n_bets']} bets, WR={m_val['win_rate']:.1%}, ROI={m_val['flat_roi']:+.1%}, Sharpe={m_val['sharpe']:.2f}")

    # Side breakdown
    if len(bets_val) > 0:
        for side in ["over", "under"]:
            sb = bets_val[bets_val["side"] == side]
            if len(sb) > 0:
                print(f"    {side}: {len(sb)} bets, WR={sb['won'].mean():.1%}, ROI={sb['pnl_flat'].mean():+.1%}")

    # ── Test evaluation (FIXED params) ──
    print("\n[7] Test evaluation (fixed params from validation)...")
    bets_test = compute_ou_bets(test_df, "model_p_over", min_edge=best_edge, kelly_frac=best_kelly)
    m_test = run_backtest(bets_test)

    # Bootstrap CI
    if len(bets_test) >= 15:
        boot_rois = [bets_test.iloc[np.random.choice(len(bets_test), len(bets_test), replace=True)]["pnl_flat"].mean()
                    for _ in range(2000)]
        ci_lo, ci_hi = np.percentile(boot_rois, [2.5, 97.5])
        ci_str = f"[{ci_lo:+.1%}, {ci_hi:+.1%}]"
    else:
        ci_str = "N/A"

    print(f"  Test: {m_test['n_bets']} bets, WR={m_test['win_rate']:.1%}, ROI={m_test['flat_roi']:+.1%} {ci_str}")
    print(f"  Sharpe: {m_test['sharpe']:.2f}, Kelly Return: {m_test['total_return']:+.1%}, Max DD: {m_test['max_dd']:.1%}")

    # Side breakdown
    if len(bets_test) > 0:
        for side in ["over", "under"]:
            sb = bets_test[bets_test["side"] == side]
            if len(sb) > 0:
                print(f"    {side}: {len(sb)} bets, WR={sb['won'].mean():.1%}, ROI={sb['pnl_flat'].mean():+.1%}, avg_edge={sb['edge'].mean():.1%}")

        # Monthly
        bets_test["month"] = pd.to_datetime(bets_test["game_date"]).dt.to_period("M")
        print(f"  Monthly:")
        for month, mb in bets_test.groupby("month"):
            print(f"    {month}: {len(mb)} bets, WR={mb['won'].mean():.1%}, ROI={mb['pnl_flat'].mean():+.1%}")

    # ── Permutation test ──
    if len(bets_test) >= 15:
        print(f"\n  Permutation test (5,000 shuffles)...")
        observed_roi = bets_test["pnl_flat"].mean()
        perm_rois = []
        for _ in range(5000):
            shuffled = test_df.copy()
            shuffled["actual_over"] = np.random.permutation(shuffled["actual_over"].values)
            pb = compute_ou_bets(shuffled, "model_p_over", min_edge=best_edge, kelly_frac=best_kelly)
            perm_rois.append(pb["pnl_flat"].mean() if len(pb) > 0 else 0)
        perm_rois = np.array(perm_rois)
        p_value = (perm_rois >= observed_roi).mean()
        print(f"    Observed ROI: {observed_roi:+.1%}")
        print(f"    P-value: {p_value:.4f}")
        print(f"    Significant: {'YES' if p_value < 0.05 else 'NO'}")

    # ── Also test: model WITHOUT market features (pure Statcast) ──
    print("\n[8] Ablation: model WITHOUT market features (pure Statcast)...")
    pure_feats = [c for c in feat_cols if c not in MARKET_FEATURES]
    if len(pure_feats) > 10:
        model_pure, res_std_pure = train_ou_model(train_df, pure_feats)
        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            preds_p, p_over_p = predict_p_over(model_pure, split_df, pure_feats, res_std_pure)
            split_df["pure_p_over"] = np.clip(p_over_p, 0.05, 0.95)

            non_push = ~split_df["is_push"]
            if non_push.sum() > 50:
                y = split_df.loc[non_push, "actual_over"].values
                brier_pure = brier_score_loss(y, split_df.loc[non_push, "pure_p_over"].values)
                brier_dk = brier_score_loss(y, split_df.loc[non_push, "dk_over_fair"].values)
                bss_vs_dk = 1 - brier_pure / brier_dk
                print(f"  {split_name}: Pure Statcast BSS vs DK: {bss_vs_dk:+.4f}")

        bets_pure = compute_ou_bets(test_df, "pure_p_over", min_edge=best_edge, kelly_frac=best_kelly)
        if len(bets_pure) > 0:
            m_pure = run_backtest(bets_pure)
            print(f"  Pure Statcast test: {m_pure['n_bets']} bets, WR={m_pure['win_rate']:.1%}, "
                  f"ROI={m_pure['flat_roi']:+.1%}, Sharpe={m_pure['sharpe']:.2f}")

    # ── Feature importance ──
    print("\n[9] Top feature importances:")
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    for feat_name, importance in imp.head(15).items():
        print(f"    {feat_name:<45} {importance:>6}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: LightGBM regression → P(over) via Gaussian CDF")
    print(f"  Training: {len(train_df)} games, Optuna 150 trials, 5-fold TSCV")
    print(f"  Edge threshold: {best_edge:.0%} (validation-selected)")
    print(f"  Kelly fraction: {best_kelly:.0%}")
    print(f"  Validation: {m_val['n_bets']} bets, ROI={m_val['flat_roi']:+.1%}, Sharpe={m_val['sharpe']:.2f}")
    print(f"  Test:       {m_test['n_bets']} bets, ROI={m_test['flat_roi']:+.1%}, Sharpe={m_test['sharpe']:.2f} {ci_str}")


if __name__ == "__main__":
    main()
