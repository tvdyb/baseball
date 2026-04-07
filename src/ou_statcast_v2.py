#!/usr/bin/env python3
"""
O/U Statcast-Only Model v2
===========================

Key insight from v1: The Statcast-only model (no market features) outperforms
the market-feature model (Sharpe 0.93 vs -0.16). This makes sense because:
  - DK is a better predictor than our model (higher corr, lower MAE)
  - Including ou_close makes the model learn to copy DK, destroying edge
  - Statcast features capture pitching/hitting quality DK DOESN'T fully price

Strategy: Use Statcast features to identify games where P(under) is highest,
then bet under at DK prices where the market's systematic over-bias is amplified.

Approach A: Regression → predict total runs → P(over) via CDF → edge vs DK
Approach B: Classification → directly predict P(under > line) → edge vs DK
Approach C: Conditional under → predict when DK's over-pricing is worst

All with proper temporal splits and permutation testing.

Zero look-ahead bias:
  TRAIN:      Apr 1 – May 25, 2025
  VALIDATION: May 26 – Jun 30, 2025
  TEST:       Jul 1 – Sep 28, 2025
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VAL_CUTOFF = "2025-07-01"

# ═══════════════════════════════════════════════════════════════════
# Features — Statcast only, no market
# ═══════════════════════════════════════════════════════════════════

LEVEL_FEATURES = [
    "park_factor", "temperature", "wind_speed", "is_dome", "wind_out", "wind_in",
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
    "home_hit_xrv_mean", "away_hit_xrv_mean",
    "home_hit_k_rate", "away_hit_k_rate",
    "home_hit_barrel_rate", "away_hit_barrel_rate",
    "home_hit_hard_hit_rate", "away_hit_hard_hit_rate",
    "home_hit_xrv_contact", "away_hit_xrv_contact",
    "home_bp_xrv_mean", "away_bp_xrv_mean",
    "home_bp_fatigue_score", "away_bp_fatigue_score",
    "home_matchup_xrv_mean", "away_matchup_xrv_mean",
    "days_into_season", "is_night",
]

SUM_FEATURES = [
    "sum_sp_xrv", "sum_sp_k_rate", "sum_sp_bb_rate",
    "sum_hit_xrv", "sum_bp_xrv",
    "sum_sp_composite", "sum_sp_stuff",
    "sum_hit_barrel", "sum_hit_hard_hit",
]


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def american_to_prob(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100),
                    np.where(odds > 0, 100 / (odds + 100), np.nan))


def load_data():
    feat = pd.read_parquet(DATA / "features" / "game_features_2025.parquet")
    feat["game_date"] = pd.to_datetime(feat["game_date"]).dt.strftime("%Y-%m-%d")

    dk = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    dk["game_date"] = pd.to_datetime(dk["game_date"]).dt.strftime("%Y-%m-%d")
    dk = dk[dk["game_date"] >= "2025-01-01"]
    dk = dk.dropna(subset=["ou_close", "total_runs", "over_close_odds", "under_close_odds"])

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
    sum_pairs = [
        ("sum_sp_xrv", "home_sp_xrv_mean", "away_sp_xrv_mean"),
        ("sum_sp_k_rate", "home_sp_k_rate", "away_sp_k_rate"),
        ("sum_sp_bb_rate", "home_sp_bb_rate", "away_sp_bb_rate"),
        ("sum_hit_xrv", "home_hit_xrv_mean", "away_hit_xrv_mean"),
        ("sum_bp_xrv", "home_bp_xrv_mean", "away_bp_xrv_mean"),
        ("sum_sp_composite", "home_sp_composite_score", "away_sp_composite_score"),
        ("sum_sp_stuff", "home_sp_stuff_score", "away_sp_stuff_score"),
        ("sum_hit_barrel", "home_hit_barrel_rate", "away_hit_barrel_rate"),
        ("sum_hit_hard_hit", "home_hit_hard_hit_rate", "away_hit_hard_hit_rate"),
    ]
    for name, h, a in sum_pairs:
        if h in merged.columns and a in merged.columns:
            merged[name] = merged[h].fillna(0) + merged[a].fillna(0)

    # Interaction features: best pitcher quality * worst lineup quality
    if "home_sp_composite_score" in merged.columns and "away_sp_composite_score" in merged.columns:
        merged["best_sp_composite"] = merged[["home_sp_composite_score", "away_sp_composite_score"]].max(axis=1)
        merged["worst_sp_composite"] = merged[["home_sp_composite_score", "away_sp_composite_score"]].min(axis=1)
    if "home_sp_k_rate" in merged.columns and "away_sp_k_rate" in merged.columns:
        merged["max_sp_k_rate"] = merged[["home_sp_k_rate", "away_sp_k_rate"]].max(axis=1)
    if "home_hit_barrel_rate" in merged.columns and "away_hit_barrel_rate" in merged.columns:
        merged["max_hit_barrel"] = merged[["home_hit_barrel_rate", "away_hit_barrel_rate"]].max(axis=1)

    merged = merged[merged["game_date"] >= "2025-04-01"].copy()
    merged = merged.sort_values("game_date").reset_index(drop=True)

    print(f"  Loaded: {len(merged)} games (Apr 1 – Sep 28)")
    return merged


def get_feature_cols(df, extra=[]):
    all_feats = LEVEL_FEATURES + SUM_FEATURES + extra
    available = [c for c in all_feats if c in df.columns]
    return [c for c in available if df[c].notna().mean() > 0.6]


# ═══════════════════════════════════════════════════════════════════
# Model Training
# ═══════════════════════════════════════════════════════════════════

def train_regression(train_df, feat_cols, n_trials=100):
    """Train LGB regression on total_runs."""
    X = train_df[feat_cols].copy()
    y = train_df["total_runs"].values

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "n_estimators": trial.suggest_int("n_estimators", 30, 250),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "num_leaves": trial.suggest_int("num_leaves", 4, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 30.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
            "verbose": -1, "random_state": 42, "n_jobs": -1,
        }
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X.iloc[tr_idx], y[tr_idx])
            scores.append(mean_absolute_error(y[va_idx], m.predict(X.iloc[va_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({"objective": "regression", "metric": "mae",
                 "verbose": -1, "random_state": 42, "n_jobs": -1})
    model = lgb.LGBMRegressor(**best)
    model.fit(X, y)

    residuals = y - model.predict(X)
    res_std = np.std(residuals)
    print(f"    Best CV MAE: {study.best_value:.3f}, residual std: {res_std:.2f}")
    return model, res_std


def train_classifier(train_df, feat_cols, n_trials=100):
    """Train LGB binary classifier: P(under hits)."""
    X = train_df[feat_cols].copy()
    y = (train_df["actual_over"] == 0).astype(int).values  # 1 = under hits

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": trial.suggest_int("n_estimators", 30, 250),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "num_leaves": trial.suggest_int("num_leaves", 4, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 30.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
            "verbose": -1, "random_state": 42, "n_jobs": -1,
        }
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            m = lgb.LGBMClassifier(**params)
            m.fit(X.iloc[tr_idx], y[tr_idx])
            preds = m.predict_proba(X.iloc[va_idx])[:, 1]
            scores.append(log_loss(y[va_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({"objective": "binary", "metric": "binary_logloss",
                 "verbose": -1, "random_state": 42, "n_jobs": -1})
    model = lgb.LGBMClassifier(**best)
    model.fit(X, y)
    print(f"    Best CV logloss: {study.best_value:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════

def predict_regression_p_over(model, df, feat_cols, res_std):
    preds = model.predict(df[feat_cols])
    lines = df["ou_close"].values
    p_over = 1.0 - norm.cdf(lines + 0.5, loc=preds, scale=res_std)
    return preds, p_over


def predict_classifier_p_under(model, df, feat_cols):
    return model.predict_proba(df[feat_cols])[:, 1]


# ═══════════════════════════════════════════════════════════════════
# Betting
# ═══════════════════════════════════════════════════════════════════

def compute_bets(df, p_over_col, min_edge, kelly_frac=0.25, under_only=False):
    """Generate O/U bets comparing model P(over) vs DK devigged P(over)."""
    bets = []
    for _, row in df.iterrows():
        if row["is_push"]:
            continue
        model_p_over = row[p_over_col]
        mkt_p_over = row["dk_over_fair"]
        if pd.isna(model_p_over) or pd.isna(mkt_p_over):
            continue

        edge_over = model_p_over - mkt_p_over
        edge_under = -edge_over

        if not under_only and edge_over > min_edge:
            side, edge, our_p = "over", edge_over, model_p_over
            buy_price = mkt_p_over
            won = row["actual_over"] == 1
        elif edge_under > min_edge:
            side, edge, our_p = "under", edge_under, 1 - model_p_over
            buy_price = 1 - mkt_p_over
            won = row["actual_over"] == 0
        else:
            continue

        pnl = (1.0 - buy_price) if won else -buy_price

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


def backtest(bets_df, starting_bankroll=10000):
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
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(180) if daily_pnl.std() > 0 else 0

    equity = np.array([starting_bankroll] + list(np.cumsum(daily_pnl) + starting_bankroll))
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min()

    return {
        "n_bets": len(bets_df),
        "win_rate": bets_df["won"].mean(),
        "flat_roi": bets_df["pnl_flat"].mean(),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": (bankroll - starting_bankroll) / starting_bankroll,
    }


def permutation_test(test_df, p_over_col, min_edge, kelly_frac, n_perms=5000):
    """Permutation test: shuffle actual outcomes, recompute ROI."""
    real_bets = compute_bets(test_df, p_over_col, min_edge=min_edge, kelly_frac=kelly_frac)
    if len(real_bets) < 10:
        return None, None, None
    observed_roi = real_bets["pnl_flat"].mean()

    perm_rois = []
    for _ in range(n_perms):
        shuffled = test_df.copy()
        shuffled["actual_over"] = np.random.permutation(shuffled["actual_over"].values)
        # Recompute is_push (stays same since line doesn't change)
        pb = compute_bets(shuffled, p_over_col, min_edge=min_edge, kelly_frac=kelly_frac)
        perm_rois.append(pb["pnl_flat"].mean() if len(pb) > 0 else 0)
    perm_rois = np.array(perm_rois)
    p_value = (perm_rois >= observed_roi).mean()
    return observed_roi, p_value, perm_rois


def bootstrap_ci(bets_df, n_boot=2000):
    if len(bets_df) < 10:
        return None, None
    rois = [bets_df.iloc[np.random.choice(len(bets_df), len(bets_df), replace=True)]["pnl_flat"].mean()
            for _ in range(n_boot)]
    return np.percentile(rois, 2.5), np.percentile(rois, 97.5)


def print_results(label, bets_df, m):
    ci_lo, ci_hi = bootstrap_ci(bets_df)
    ci_str = f"[{ci_lo:+.1%}, {ci_hi:+.1%}]" if ci_lo is not None else ""
    print(f"    {label}: {m['n_bets']} bets, WR={m['win_rate']:.1%}, "
          f"ROI={m['flat_roi']:+.1%} {ci_str}, Sharpe={m['sharpe']:.2f}, "
          f"MaxDD={m['max_dd']:.1%}")
    if len(bets_df) > 0:
        for side in ["over", "under"]:
            sb = bets_df[bets_df["side"] == side]
            if len(sb) > 0:
                print(f"      {side}: {len(sb)} bets, WR={sb['won'].mean():.1%}, "
                      f"ROI={sb['pnl_flat'].mean():+.1%}, avg_edge={sb['edge'].mean():.1%}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("O/U STATCAST MODEL v2 — Beat DK Lines")
    print("=" * 70)

    # ── Load data ──
    print("\n[1] Loading data...")
    df = load_data()

    # ── Temporal split ──
    pre_july = df[df["game_date"] < VAL_CUTOFF]
    train_cutoff_idx = int(len(pre_july) * 0.6)
    train_dates = pre_july.iloc[:train_cutoff_idx]["game_date"].max()

    train_df = df[df["game_date"] <= train_dates].copy()
    val_df = df[(df["game_date"] > train_dates) & (df["game_date"] < VAL_CUTOFF)].copy()
    test_df = df[df["game_date"] >= VAL_CUTOFF].copy()

    # Remove pushes from val/test for cleaner evaluation
    val_df = val_df[~val_df["is_push"]].copy()
    test_df = test_df[~test_df["is_push"]].copy()

    print(f"\n[2] Splits:")
    print(f"  Train: {len(train_df)} ({train_df['game_date'].min()} – {train_df['game_date'].max()})")
    print(f"  Val:   {len(val_df)} ({val_df['game_date'].min()} – {val_df['game_date'].max()})")
    print(f"  Test:  {len(test_df)} ({test_df['game_date'].min()} – {test_df['game_date'].max()})")

    base_feats = get_feature_cols(df)
    extra_feats = ["best_sp_composite", "worst_sp_composite", "max_sp_k_rate", "max_hit_barrel"]
    feat_cols = get_feature_cols(df, extra=[f for f in extra_feats if f in df.columns])
    print(f"  Features: {len(feat_cols)}")

    # Market baseline
    for name, sdf in [("Val", val_df), ("Test", test_df)]:
        under_wr = (sdf["actual_over"] == 0).mean()
        under_pnl = sdf.apply(
            lambda r: r["dk_over_fair"] if r["actual_over"] == 0 else -(1 - r["dk_over_fair"]), axis=1
        )
        print(f"  {name} naive under: WR={under_wr:.1%}, ROI={under_pnl.mean():+.1%}")

    # ════════════════════════════════════════════════════════════════
    # APPROACH A: Regression → P(over) via CDF
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("APPROACH A: Regression → Gaussian CDF → P(over)")
    print("=" * 70)

    print("\n  Training regression model...")
    model_reg, res_std = train_regression(train_df, feat_cols, n_trials=100)

    for name, sdf in [("val", val_df), ("test", test_df)]:
        preds, p_over = predict_regression_p_over(model_reg, sdf, feat_cols, res_std)
        sdf["reg_p_over"] = np.clip(p_over, 0.05, 0.95)
        sdf["reg_pred"] = preds

        corr = np.corrcoef(preds, sdf["total_runs"].values)[0, 1]
        mae = mean_absolute_error(sdf["total_runs"], preds)
        corr_dk = np.corrcoef(sdf["ou_close"].values, sdf["total_runs"].values)[0, 1]
        mae_dk = mean_absolute_error(sdf["total_runs"], sdf["ou_close"])
        print(f"  {name}: Model corr={corr:.3f} MAE={mae:.2f} | DK corr={corr_dk:.3f} MAE={mae_dk:.2f}")

    # Validation sweep
    print("\n  Edge threshold sweep (validation):")
    print(f"    {'Edge':>5} {'Kelly':>6} {'Mode':>8} {'Bets':>5} {'WR':>6} {'ROI':>7} {'Sharpe':>7}")

    best_score, best_cfg = -999, None
    for me in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        for kf in [0.10, 0.15, 0.20, 0.25]:
            for under_only in [False, True]:
                bets = compute_bets(val_df, "reg_p_over", min_edge=me, kelly_frac=kf, under_only=under_only)
                if len(bets) < 20:
                    continue
                m = backtest(bets)
                score = m["sharpe"] * min(1.0, len(bets) / 50.0)
                mode = "under" if under_only else "both"
                if me in [0.02, 0.05, 0.08, 0.10, 0.15] and kf == 0.25:
                    print(f"    {me:>4.0%} {kf:>5.0%} {mode:>8} {m['n_bets']:>5} "
                          f"{m['win_rate']:>5.1%} {m['flat_roi']:>+6.1%} {m['sharpe']:>6.2f}")
                if score > best_score:
                    best_score = score
                    best_cfg = {"edge": me, "kelly": kf, "under_only": under_only}

    if best_cfg is None:
        print("  No profitable config found.")
        return

    # Best config on validation
    bv = compute_bets(val_df, "reg_p_over", min_edge=best_cfg["edge"],
                      kelly_frac=best_cfg["kelly"], under_only=best_cfg["under_only"])
    mv = backtest(bv)
    mode_str = "under-only" if best_cfg["under_only"] else "both"
    print(f"\n  BEST: edge>{best_cfg['edge']:.0%} kelly={best_cfg['kelly']:.0%} mode={mode_str}")
    print_results("Val", bv, mv)

    # Test
    bt = compute_bets(test_df, "reg_p_over", min_edge=best_cfg["edge"],
                      kelly_frac=best_cfg["kelly"], under_only=best_cfg["under_only"])
    mt = backtest(bt)
    print_results("Test", bt, mt)

    # Monthly breakdown
    if len(bt) > 0:
        bt["month"] = pd.to_datetime(bt["game_date"]).dt.to_period("M")
        print("    Monthly:")
        for month, mb in bt.groupby("month"):
            print(f"      {month}: {len(mb)} bets, WR={mb['won'].mean():.1%}, ROI={mb['pnl_flat'].mean():+.1%}")

    # Permutation test
    print("\n  Permutation test (5,000 shuffles)...")
    obs_roi, p_val, _ = permutation_test(
        test_df, "reg_p_over", best_cfg["edge"], best_cfg["kelly"], n_perms=5000
    )
    if obs_roi is not None:
        print(f"    Observed ROI: {obs_roi:+.1%}, p-value: {p_val:.4f}, "
              f"Significant: {'YES' if p_val < 0.05 else 'NO'}")

    # ════════════════════════════════════════════════════════════════
    # APPROACH B: Classification → P(under)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("APPROACH B: Classification → P(under)")
    print("=" * 70)

    # Include ou_close as a feature for the classifier (it's a strong predictor of under)
    # This is NOT look-ahead — it's the pre-game closing line
    cls_feats = feat_cols + (["ou_close"] if "ou_close" in df.columns else [])
    cls_feats = [c for c in cls_feats if c in train_df.columns and train_df[c].notna().mean() > 0.6]

    print(f"\n  Training classifier (features: {len(cls_feats)})...")
    model_cls = train_classifier(train_df[~train_df["is_push"]], cls_feats, n_trials=100)

    for name, sdf in [("val", val_df), ("test", test_df)]:
        p_under = predict_classifier_p_under(model_cls, sdf, cls_feats)
        sdf["cls_p_over"] = np.clip(1 - p_under, 0.05, 0.95)

        y = (sdf["actual_over"] == 0).astype(int).values
        brier_cls = brier_score_loss(y, p_under)
        brier_dk = brier_score_loss(y, 1 - sdf["dk_over_fair"].values)
        bss = 1 - brier_cls / brier_dk
        print(f"  {name}: Classifier BSS vs DK: {bss:+.4f}")

    # Validation sweep for classifier
    print("\n  Edge threshold sweep (validation):")
    best_score_b, best_cfg_b = -999, None
    for me in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        for kf in [0.10, 0.15, 0.20, 0.25]:
            for under_only in [False, True]:
                bets = compute_bets(val_df, "cls_p_over", min_edge=me, kelly_frac=kf, under_only=under_only)
                if len(bets) < 20:
                    continue
                m = backtest(bets)
                score = m["sharpe"] * min(1.0, len(bets) / 50.0)
                if score > best_score_b:
                    best_score_b = score
                    best_cfg_b = {"edge": me, "kelly": kf, "under_only": under_only}

    if best_cfg_b:
        bv_b = compute_bets(val_df, "cls_p_over", min_edge=best_cfg_b["edge"],
                           kelly_frac=best_cfg_b["kelly"], under_only=best_cfg_b["under_only"])
        mv_b = backtest(bv_b)
        mode_str_b = "under-only" if best_cfg_b["under_only"] else "both"
        print(f"\n  BEST: edge>{best_cfg_b['edge']:.0%} kelly={best_cfg_b['kelly']:.0%} mode={mode_str_b}")
        print_results("Val", bv_b, mv_b)

        bt_b = compute_bets(test_df, "cls_p_over", min_edge=best_cfg_b["edge"],
                           kelly_frac=best_cfg_b["kelly"], under_only=best_cfg_b["under_only"])
        mt_b = backtest(bt_b)
        print_results("Test", bt_b, mt_b)

        # Monthly
        if len(bt_b) > 0:
            bt_b["month"] = pd.to_datetime(bt_b["game_date"]).dt.to_period("M")
            print("    Monthly:")
            for month, mb in bt_b.groupby("month"):
                print(f"      {month}: {len(mb)} bets, WR={mb['won'].mean():.1%}, ROI={mb['pnl_flat'].mean():+.1%}")

        # Permutation
        print("\n  Permutation test (5,000 shuffles)...")
        obs_b, p_b, _ = permutation_test(
            test_df, "cls_p_over", best_cfg_b["edge"], best_cfg_b["kelly"], n_perms=5000
        )
        if obs_b is not None:
            print(f"    Observed ROI: {obs_b:+.1%}, p-value: {p_b:.4f}, "
                  f"Significant: {'YES' if p_b < 0.05 else 'NO'}")

    # ════════════════════════════════════════════════════════════════
    # APPROACH C: Ensemble (avg of regression + classifier)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("APPROACH C: Ensemble (avg regression + classifier)")
    print("=" * 70)

    for sdf in [val_df, test_df]:
        sdf["ens_p_over"] = np.clip((sdf["reg_p_over"] + sdf["cls_p_over"]) / 2, 0.05, 0.95)

    # Validation sweep
    best_score_c, best_cfg_c = -999, None
    for me in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        for kf in [0.10, 0.15, 0.20, 0.25]:
            for under_only in [False, True]:
                bets = compute_bets(val_df, "ens_p_over", min_edge=me, kelly_frac=kf, under_only=under_only)
                if len(bets) < 20:
                    continue
                m = backtest(bets)
                score = m["sharpe"] * min(1.0, len(bets) / 50.0)
                if score > best_score_c:
                    best_score_c = score
                    best_cfg_c = {"edge": me, "kelly": kf, "under_only": under_only}

    if best_cfg_c:
        bv_c = compute_bets(val_df, "ens_p_over", min_edge=best_cfg_c["edge"],
                           kelly_frac=best_cfg_c["kelly"], under_only=best_cfg_c["under_only"])
        mv_c = backtest(bv_c)
        mode_str_c = "under-only" if best_cfg_c["under_only"] else "both"
        print(f"\n  BEST: edge>{best_cfg_c['edge']:.0%} kelly={best_cfg_c['kelly']:.0%} mode={mode_str_c}")
        print_results("Val", bv_c, mv_c)

        bt_c = compute_bets(test_df, "ens_p_over", min_edge=best_cfg_c["edge"],
                           kelly_frac=best_cfg_c["kelly"], under_only=best_cfg_c["under_only"])
        mt_c = backtest(bt_c)
        print_results("Test", bt_c, mt_c)

        # Permutation
        print("\n  Permutation test (5,000 shuffles)...")
        obs_c, p_c, _ = permutation_test(
            test_df, "ens_p_over", best_cfg_c["edge"], best_cfg_c["kelly"], n_perms=5000
        )
        if obs_c is not None:
            print(f"    Observed ROI: {obs_c:+.1%}, p-value: {p_c:.4f}, "
                  f"Significant: {'YES' if p_c < 0.05 else 'NO'}")

    # ════════════════════════════════════════════════════════════════
    # Feature importance
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Regression model)")
    print("=" * 70)
    imp = pd.Series(model_reg.feature_importances_, index=feat_cols).sort_values(ascending=False)
    for feat_name, importance in imp.head(15).items():
        print(f"  {feat_name:<45} {importance:>6}")

    # ════════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Approach A (Regression):     Test Sharpe={mt['sharpe']:.2f}, ROI={mt['flat_roi']:+.1%}, n={mt['n_bets']}")
    if best_cfg_b:
        print(f"  Approach B (Classifier):     Test Sharpe={mt_b['sharpe']:.2f}, ROI={mt_b['flat_roi']:+.1%}, n={mt_b['n_bets']}")
    if best_cfg_c:
        print(f"  Approach C (Ensemble):       Test Sharpe={mt_c['sharpe']:.2f}, ROI={mt_c['flat_roi']:+.1%}, n={mt_c['n_bets']}")


if __name__ == "__main__":
    main()
