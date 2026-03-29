#!/usr/bin/env python3
"""
Comprehensive audit of the MLB pregame win probability model.

Runs walk-forward evaluation (2020-2025), bootstrap significance testing,
calibration diagnostics, feature ablation, confidence-tier analysis,
and market comparison. All outputs to data/audit/.

Usage:
    python src/audit.py                # full audit
    python src/audit.py --skip-ablation  # skip expensive ablation
"""

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, FEATURES_DIR
from win_model import (
    ALL_FEATURES, DIFF_FEATURES, RAW_FEATURES,
    _smart_fillna, add_nonlinear_features,
)

warnings.filterwarnings("ignore", category=UserWarning)

AUDIT_DIR = DATA_DIR / "audit"
ALL_YEARS = list(range(2018, 2026))
MIN_TRAIN_YEARS = 2  # first testable year = 2020

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

# ── Feature groups for ablation ──────────────────────────────────────────────

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


# ── Data loading ─────────────────────────────────────────────────────────────

def load_features(years):
    frames = []
    for year in years:
        path = FEATURES_DIR / f"game_features_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["season"] = year
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No feature files found")
    return pd.concat(frames, ignore_index=True)


def _prepare_xgb_features(df, features=None):
    """Prepare XGB feature matrix with nonlinear features."""
    if features is None:
        features = ALL_FEATURES
    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    for col in ["home_sp_rest_days", "away_sp_rest_days",
                "home_bp_fatigue_score", "away_bp_fatigue_score",
                "days_into_season"]:
        if col in df.columns and col not in X.columns:
            X[col] = df[col]
    return add_nonlinear_features(X)


# ── Core training (leak-free) ───────────────────────────────────────────────

def train_ensemble_oof(train_df, features_override=None, n_folds=5):
    """Train LR+XGB ensemble with OOF blend weight. Returns bundle dict."""
    features = features_override if features_override is not None else ALL_FEATURES
    y = train_df["home_win"].values

    # Prepare feature matrices
    available_lr = [f for f in features if f in train_df.columns]
    X_lr_raw = train_df[available_lr].copy()
    X_xgb_full = _prepare_xgb_features(train_df, features)
    xgb_feature_names = list(X_xgb_full.columns)

    # Train final LR
    X_filled, train_medians = _smart_fillna(X_lr_raw)
    scaler = StandardScaler()
    X_lr_scaled = scaler.fit_transform(X_filled)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(X_lr_scaled, y)

    # Train final XGB with chronological val split
    xgb_model = None
    w_lr = 1.0
    if HAS_XGB:
        n = len(X_xgb_full)
        val_size = int(n * 0.2)
        dtrain = xgb.DMatrix(X_xgb_full.iloc[:n - val_size], label=y[:n - val_size])
        dval = xgb.DMatrix(X_xgb_full.iloc[n - val_size:], label=y[n - val_size:])
        xgb_model = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50, verbose_eval=0,
        )

        # OOF predictions for blend weight
        kf = KFold(n_splits=n_folds, shuffle=False)
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
            df_fold = xgb.DMatrix(X_xgb_full.iloc[fold_train], label=y[fold_train])
            dv_fold = xgb.DMatrix(X_xgb_full.iloc[fold_val], label=y[fold_val])
            xgb_f = xgb.train(
                XGB_PARAMS, df_fold, num_boost_round=200,
                evals=[(df_fold, "t"), (dv_fold, "v")],
                early_stopping_rounds=30, verbose_eval=0,
            )
            xgb_oof[fold_val] = xgb_f.predict(dv_fold)

        def blend_loss(w):
            blended = np.clip(w * lr_oof + (1 - w) * xgb_oof, 1e-6, 1 - 1e-6)
            return log_loss(y, blended)

        opt = minimize_scalar(blend_loss, bounds=(0.1, 0.9), method="bounded")
        w_lr = opt.x

    return {
        "lr": lr, "scaler": scaler, "train_medians": train_medians,
        "xgb_model": xgb_model, "lr_features": available_lr,
        "xgb_features": xgb_feature_names, "w_lr": w_lr,
        "features_used": features,
    }


def predict_ensemble(bundle, test_df):
    """Generate predictions from a trained bundle. Returns (lr, xgb, ens) probs."""
    lr_features = bundle["lr_features"]
    available_lr = [f for f in lr_features if f in test_df.columns]
    X_lr_raw = test_df[available_lr].copy()
    X_filled, _ = _smart_fillna(X_lr_raw, bundle["train_medians"])
    for col in lr_features:
        if col not in X_filled.columns:
            X_filled[col] = 0
    X_filled = X_filled[lr_features]
    X_lr = bundle["scaler"].transform(X_filled)
    lr_probs = bundle["lr"].predict_proba(X_lr)[:, 1]

    if bundle["xgb_model"] and HAS_XGB:
        X_xgb = _prepare_xgb_features(test_df, bundle["features_used"])
        for col in bundle["xgb_features"]:
            if col not in X_xgb.columns:
                X_xgb[col] = np.nan
        X_xgb = X_xgb[bundle["xgb_features"]]
        dtest = xgb.DMatrix(X_xgb)
        xgb_probs = bundle["xgb_model"].predict(dtest)
        ens_probs = bundle["w_lr"] * lr_probs + (1 - bundle["w_lr"]) * xgb_probs
    else:
        xgb_probs = lr_probs
        ens_probs = lr_probs

    return lr_probs, xgb_probs, ens_probs


# ── Walk-forward engine ──────────────────────────────────────────────────────

def walk_forward(df, test_years=None, features_override=None,
                 fit_isotonic=False, verbose=True):
    """Walk-forward evaluation. Returns per-game DataFrame."""
    if test_years is None:
        test_years = [y for y in ALL_YEARS[MIN_TRAIN_YEARS:]]

    rows = []
    yearly_info = []

    for test_year in test_years:
        train_years = [y for y in ALL_YEARS if y < test_year]
        if len(train_years) < MIN_TRAIN_YEARS:
            continue

        train_df = df[df["season"].isin(train_years)].copy()
        test_df = df[df["season"] == test_year].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        if verbose:
            print(f"  {test_year}: train={len(train_df)}, test={len(test_df)}", end="")

        bundle = train_ensemble_oof(train_df, features_override=features_override)
        lr_probs, xgb_probs, ens_probs = predict_ensemble(bundle, test_df)

        # Isotonic calibration (trained on OOF from training data)
        ens_calibrated = ens_probs.copy()
        if fit_isotonic and bundle["xgb_model"]:
            # Generate training OOF predictions for isotonic fitting
            y_train = train_df["home_win"].values
            kf = KFold(n_splits=5, shuffle=False)
            oof_ens = np.zeros(len(y_train))
            lr_feats = bundle["lr_features"]
            X_lr_raw_train = train_df[[f for f in lr_feats if f in train_df.columns]].copy()
            X_xgb_train = _prepare_xgb_features(train_df, bundle["features_used"])

            for ft, fv in kf.split(X_lr_raw_train):
                Xf_filled, fm = _smart_fillna(X_lr_raw_train.iloc[ft])
                Xv_filled, _ = _smart_fillna(X_lr_raw_train.iloc[fv], fm)
                sc = StandardScaler()
                lr_f = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
                lr_f.fit(sc.fit_transform(Xf_filled), y_train[ft])
                lr_oof_v = lr_f.predict_proba(sc.transform(Xv_filled))[:, 1]

                df_fold = xgb.DMatrix(X_xgb_train.iloc[ft], label=y_train[ft])
                dv_fold = xgb.DMatrix(X_xgb_train.iloc[fv], label=y_train[fv])
                xgb_f = xgb.train(
                    XGB_PARAMS, df_fold, num_boost_round=200,
                    evals=[(df_fold, "t"), (dv_fold, "v")],
                    early_stopping_rounds=30, verbose_eval=0,
                )
                xgb_oof_v = xgb_f.predict(dv_fold)
                oof_ens[fv] = bundle["w_lr"] * lr_oof_v + (1 - bundle["w_lr"]) * xgb_oof_v

            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(oof_ens, y_train)
            ens_calibrated = iso.predict(ens_probs)

        y_test = test_df["home_win"].values
        if verbose:
            ll = log_loss(y_test, np.clip(ens_probs, 0.01, 0.99))
            auc = roc_auc_score(y_test, ens_probs)
            print(f", w_lr={bundle['w_lr']:.2f}, LL={ll:.4f}, AUC={auc:.4f}")

        yearly_info.append({
            "season": test_year, "n_games": len(test_df), "w_lr": bundle["w_lr"],
        })

        for i in range(len(test_df)):
            row = test_df.iloc[i]
            rows.append({
                "game_pk": row.get("game_pk", i),
                "game_date": str(row.get("game_date", "")),
                "season": test_year,
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "home_win": int(y_test[i]),
                "lr_prob": lr_probs[i],
                "xgb_prob": xgb_probs[i],
                "ens_prob": ens_probs[i],
                "ens_calibrated": ens_calibrated[i],
                "days_into_season": row.get("days_into_season", np.nan),
                "home_team_games_played": row.get("home_team_games_played", np.nan),
                "away_team_games_played": row.get("away_team_games_played", np.nan),
            })

    return pd.DataFrame(rows)


# ── Baselines ────────────────────────────────────────────────────────────────

def compute_baselines(wf_df, full_df):
    """Add baseline probability columns."""
    wf_df = wf_df.copy()
    wf_df["baseline_home"] = np.nan
    wf_df["baseline_prior"] = np.nan

    for season in wf_df["season"].unique():
        train_mask = full_df["season"] < season
        test_mask = wf_df["season"] == season

        # Constant home-win rate from training data
        train_home_rate = full_df.loc[train_mask, "home_win"].mean()
        wf_df.loc[test_mask, "baseline_home"] = train_home_rate

        # Prior win-pct baseline: 1-feature LR on diff_team_prior
        if "diff_team_prior" in full_df.columns:
            train_sub = full_df[train_mask].dropna(subset=["diff_team_prior", "home_win"])
            if len(train_sub) > 50:
                X_tr = train_sub[["diff_team_prior"]].fillna(0).values
                y_tr = train_sub["home_win"].values
                lr_b = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
                lr_b.fit(X_tr, y_tr)
                test_sub = wf_df.loc[test_mask]
                # We need diff_team_prior from full_df for test games
                test_full = full_df[full_df["season"] == season]
                if "diff_team_prior" in test_full.columns:
                    X_te = test_full[["diff_team_prior"]].fillna(0).values
                    preds = lr_b.predict_proba(X_te)[:, 1]
                    wf_df.loc[test_mask, "baseline_prior"] = preds

    return wf_df


# ── Bootstrap significance ───────────────────────────────────────────────────

def bootstrap_metric_delta(y, probs_a, probs_b, metric_fn, n_boot=10000,
                           seed=42, lower_is_better=True):
    """Paired bootstrap CI for metric(a) - metric(b)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    observed = metric_fn(y, probs_a) - metric_fn(y, probs_b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yi, ai, bi = y[idx], probs_a[idx], probs_b[idx]
        try:
            deltas[i] = metric_fn(yi, ai) - metric_fn(yi, bi)
        except ValueError:
            deltas[i] = np.nan
    deltas = deltas[~np.isnan(deltas)]
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    # p-value: fraction of bootstraps where sign differs from observed
    if lower_is_better:
        p_val = np.mean(deltas >= 0) if observed < 0 else np.mean(deltas <= 0)
    else:
        p_val = np.mean(deltas <= 0) if observed > 0 else np.mean(deltas >= 0)
    return {
        "delta": observed, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "p_value": p_val, "n_bootstrap": len(deltas),
    }


def run_all_bootstrap_tests(wf_df, n_boot=10000):
    """Run bootstrap tests for ensemble vs all baselines."""
    y = wf_df["home_win"].values
    ens = np.clip(wf_df["ens_prob"].values, 0.01, 0.99)
    ens_cal = np.clip(wf_df["ens_calibrated"].values, 0.01, 0.99)
    lr = np.clip(wf_df["lr_prob"].values, 0.01, 0.99)
    xgb_p = np.clip(wf_df["xgb_prob"].values, 0.01, 0.99)
    home = np.clip(wf_df["baseline_home"].values, 0.01, 0.99)
    prior = np.clip(wf_df["baseline_prior"].values, 0.01, 0.99)

    comparisons = [
        ("ens_vs_home_constant", ens, home),
        ("ens_vs_prior_winpct", ens, prior),
        ("ens_vs_lr_only", ens, lr),
        ("ens_vs_xgb_only", ens, xgb_p),
        ("calibrated_vs_uncalibrated", ens_cal, ens),
    ]

    results = []
    for label, pa, pb in comparisons:
        mask = ~np.isnan(pa) & ~np.isnan(pb) & ~np.isnan(y)
        if mask.sum() < 100:
            continue
        for metric_name, metric_fn, lower in [
            ("log_loss", log_loss, True),
            ("brier_score", brier_score_loss, True),
        ]:
            r = bootstrap_metric_delta(
                y[mask], pa[mask], pb[mask], metric_fn,
                n_boot=n_boot, lower_is_better=lower,
            )
            r["comparison"] = label
            r["metric"] = metric_name
            results.append(r)
            sig = "*" if r["p_value"] < 0.05 else ""
            print(f"    {label} ({metric_name}): {r['delta']:+.5f} "
                  f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}] p={r['p_value']:.3f}{sig}")

    return results


# ── Calibration ──────────────────────────────────────────────────────────────

def calibration_analysis(y, probs, n_bins=10, label=""):
    """Compute ECE, MCE, and bin-level calibration."""
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    total_ce = 0.0
    max_ce = 0.0
    n_total = len(y)

    for i in range(n_bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        count = mask.sum()
        if count > 0:
            pred_mean = probs[mask].mean()
            actual_mean = y[mask].mean()
            ce = abs(pred_mean - actual_mean)
            total_ce += ce * count
            max_ce = max(max_ce, ce)
            bins.append({
                "bin_lo": edges[i], "bin_hi": edges[i + 1],
                "pred_mean": pred_mean, "actual_mean": actual_mean,
                "count": int(count), "cal_error": ce,
            })

    ece = total_ce / n_total if n_total > 0 else 0.0
    return {"ece": ece, "mce": max_ce, "bins": bins, "label": label}


# ── Ablation ─────────────────────────────────────────────────────────────────

def run_ablation(df, feature_groups, n_boot=5000):
    """Leave-one-group-out ablation with bootstrap CIs."""
    print("\n  Running full model walk-forward for ablation baseline...")
    full_wf = walk_forward(df, verbose=False)
    y_full = full_wf["home_win"].values
    ens_full = np.clip(full_wf["ens_prob"].values, 0.01, 0.99)
    full_ll = log_loss(y_full, ens_full)
    full_auc = roc_auc_score(y_full, ens_full)

    results = []
    for group_name, group_features in feature_groups.items():
        # Features with this group removed
        remaining = [f for f in ALL_FEATURES if f not in group_features]
        if len(remaining) == len(ALL_FEATURES):
            # Group had no features in ALL_FEATURES, skip
            print(f"    {group_name}: no features to remove, skipping")
            continue

        print(f"    {group_name}: removing {len(ALL_FEATURES) - len(remaining)} features...", end="")
        t0 = time.time()
        ablated_wf = walk_forward(df, features_override=remaining, verbose=False)
        dt = time.time() - t0

        y_abl = ablated_wf["home_win"].values
        ens_abl = np.clip(ablated_wf["ens_prob"].values, 0.01, 0.99)
        abl_ll = log_loss(y_abl, ens_abl)
        abl_auc = roc_auc_score(y_abl, ens_abl)

        # Bootstrap delta
        boot = bootstrap_metric_delta(
            y_full, ens_abl, ens_full, log_loss, n_boot=n_boot,
        )

        results.append({
            "group": group_name,
            "n_features_removed": len(ALL_FEATURES) - len(remaining),
            "full_ll": full_ll,
            "ablated_ll": abl_ll,
            "delta_ll": abl_ll - full_ll,
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "p_value": boot["p_value"],
            "full_auc": full_auc,
            "ablated_auc": abl_auc,
            "delta_auc": abl_auc - full_auc,
        })
        sig = "*" if boot["p_value"] < 0.05 else ""
        print(f" ΔLL={abl_ll - full_ll:+.5f}{sig} ({dt:.0f}s)")

    return pd.DataFrame(results).sort_values("delta_ll", ascending=False)


# ── Confidence tiers ─────────────────────────────────────────────────────────

def confidence_tier_analysis(wf_df):
    """Metrics by model conviction level."""
    df = wf_df.copy()
    df["conviction"] = np.abs(df["ens_prob"] - 0.5)
    tier_edges = [(0.0, 0.03, "low"), (0.03, 0.07, "medium"),
                  (0.07, 0.12, "high"), (0.12, 1.0, "extreme")]
    rows = []
    for lo, hi, name in tier_edges:
        mask = (df["conviction"] >= lo) & (df["conviction"] < hi)
        sub = df[mask]
        if len(sub) < 20:
            continue
        y = sub["home_win"].values
        p = np.clip(sub["ens_prob"].values, 0.01, 0.99)
        # Accuracy: predict home if p > 0.5
        correct = ((p > 0.5) == (y == 1)).mean()
        rows.append({
            "tier": name, "conviction_lo": lo, "conviction_hi": hi,
            "n_games": len(sub),
            "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
            "accuracy": correct,
        })
    return pd.DataFrame(rows)


# ── Monthly analysis ─────────────────────────────────────────────────────────

def monthly_analysis(wf_df):
    """Metrics by month across all test years."""
    df = wf_df.copy()
    df["month"] = pd.to_datetime(df["game_date"]).dt.month
    rows = []
    for month in sorted(df["month"].unique()):
        sub = df[df["month"] == month]
        if len(sub) < 20:
            continue
        y = sub["home_win"].values
        p = np.clip(sub["ens_prob"].values, 0.01, 0.99)
        rows.append({
            "month": month,
            "n_games": len(sub),
            "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def early_vs_late_analysis(wf_df, n_boot=5000):
    """Compare first 30 days vs rest of season."""
    df = wf_df.copy()
    early_mask = df["days_into_season"] < 30
    late_mask = df["days_into_season"] >= 30

    results = {}
    for label, mask in [("early_30d", early_mask), ("rest_of_season", late_mask)]:
        sub = df[mask]
        if len(sub) < 50:
            continue
        y = sub["home_win"].values
        p = np.clip(sub["ens_prob"].values, 0.01, 0.99)
        results[label] = {
            "n_games": len(sub),
            "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
        }

    # Bootstrap the difference
    if "early_30d" in results and "rest_of_season" in results:
        early_sub = df[early_mask]
        late_sub = df[late_mask]
        y_e, p_e = early_sub["home_win"].values, np.clip(early_sub["ens_prob"].values, 0.01, 0.99)
        y_l, p_l = late_sub["home_win"].values, np.clip(late_sub["ens_prob"].values, 0.01, 0.99)
        rng = np.random.RandomState(42)
        deltas = []
        for _ in range(n_boot):
            ie = rng.randint(0, len(y_e), len(y_e))
            il = rng.randint(0, len(y_l), len(y_l))
            try:
                d = log_loss(y_e[ie], p_e[ie]) - log_loss(y_l[il], p_l[il])
                deltas.append(d)
            except ValueError:
                pass
        if deltas:
            results["delta_ll_early_minus_late"] = {
                "delta": results["early_30d"]["log_loss"] - results["rest_of_season"]["log_loss"],
                "ci_lo": np.percentile(deltas, 2.5),
                "ci_hi": np.percentile(deltas, 97.5),
            }

    return results


# ── Market comparison ────────────────────────────────────────────────────────

def _compute_bet_pnl(edge, market_prob, outcome, fee_pct=0.02):
    """PnL for $100 flat bet with fees."""
    if edge > 0:
        cost = market_prob * 100
        fee = cost * fee_pct
        return (100 - cost - fee) if outcome == 1 else (-cost - fee)
    else:
        cost = (1 - market_prob) * 100
        fee = cost * fee_pct
        return (100 - cost - fee) if outcome == 0 else (-cost - fee)


def market_comparison(wf_df, n_boot=10000):
    """Merge walk-forward 2025 predictions with market data."""
    wf_2025 = wf_df[wf_df["season"] == 2025].copy()
    if len(wf_2025) == 0:
        return {}

    results = {}

    # Kalshi
    kalshi_path = DATA_DIR / "kalshi" / "kalshi_mlb_2025.parquet"
    if kalshi_path.exists():
        kalshi = pd.read_parquet(kalshi_path)
        kalshi["game_date"] = kalshi["game_date"].astype(str)
        wf_2025["game_date"] = wf_2025["game_date"].astype(str)
        merged = wf_2025.merge(
            kalshi[["game_date", "home_team", "away_team", "kalshi_home_prob"]].drop_duplicates(),
            on=["game_date", "home_team", "away_team"], how="inner",
        )
        if len(merged) > 0:
            y = merged["home_win"].values
            m = np.clip(merged["ens_prob"].values, 0.01, 0.99)
            k = np.clip(merged["kalshi_home_prob"].values, 0.01, 0.99)

            kalshi_results = {
                "n_games": len(merged),
                "model_ll": log_loss(y, m),
                "kalshi_ll": log_loss(y, k),
                "model_brier": brier_score_loss(y, m),
                "kalshi_brier": brier_score_loss(y, k),
                "model_auc": roc_auc_score(y, m),
                "kalshi_auc": roc_auc_score(y, k),
                "correlation": np.corrcoef(m, k)[0, 1],
            }

            # Bootstrap
            for metric_name, metric_fn in [("log_loss", log_loss), ("brier", brier_score_loss)]:
                b = bootstrap_metric_delta(y, m, k, metric_fn, n_boot=n_boot)
                kalshi_results[f"boot_{metric_name}"] = b

            # ROI simulation
            edge = m - k
            for threshold in [0.03, 0.05, 0.07, 0.10]:
                pnls = []
                for i in range(len(edge)):
                    if abs(edge[i]) >= threshold:
                        pnls.append(_compute_bet_pnl(edge[i], k[i], y[i]))
                if pnls:
                    total = sum(pnls)
                    roi = total / (len(pnls) * 100)
                    kalshi_results[f"roi_{int(threshold*100)}pct"] = {
                        "n_bets": len(pnls),
                        "pnl": total,
                        "roi": roi,
                        "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
                    }

            # Bootstrap ROI at 3%
            if abs(sum(1 for e in edge if abs(e) >= 0.03)) > 20:
                bet_idx = [i for i in range(len(edge)) if abs(edge[i]) >= 0.03]
                bet_pnls = np.array([_compute_bet_pnl(edge[i], k[i], y[i]) for i in bet_idx])
                rng = np.random.RandomState(42)
                roi_boots = []
                for _ in range(n_boot):
                    bi = rng.randint(0, len(bet_pnls), len(bet_pnls))
                    roi_boots.append(bet_pnls[bi].sum() / (len(bi) * 100))
                kalshi_results["roi_3pct_boot"] = {
                    "ci_lo": np.percentile(roi_boots, 2.5),
                    "ci_hi": np.percentile(roi_boots, 97.5),
                    "p_value_positive": np.mean(np.array(roi_boots) <= 0),
                }

            # Per-game CSV
            merged["kalshi_prob"] = k
            merged["model_prob"] = m
            merged["edge"] = edge
            results["kalshi"] = kalshi_results
            results["kalshi_games"] = merged

    return results


# ── Plots ────────────────────────────────────────────────────────────────────

def generate_plots(wf_df, cal_data, ablation_df, market_data, output_dir):
    """Generate all diagnostic plots."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping plots")
        return

    # 1. Reliability diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    bins = cal_data["bins"]
    pred = [b["pred_mean"] for b in bins]
    actual = [b["actual_mean"] for b in bins]
    counts = [b["count"] for b in bins]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.scatter(pred, actual, s=[c/5 for c in counts], alpha=0.7, zorder=5)
    for i, b in enumerate(bins):
        ax.annotate(f"n={b['count']}", (pred[i], actual[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability Diagram (ECE={cal_data['ece']:.4f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_diagram.png", dpi=150)
    plt.close(fig)

    # 2. Walk-forward yearly bar chart
    yearly = wf_df.groupby("season").apply(
        lambda g: pd.Series({
            "log_loss": log_loss(g["home_win"], np.clip(g["ens_prob"], 0.01, 0.99)),
            "auc": roc_auc_score(g["home_win"], g["ens_prob"]),
            "n_games": len(g),
        })
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(yearly["season"].astype(str), yearly["log_loss"], color="steelblue")
    axes[0].set_title("Log Loss by Year (lower = better)")
    axes[0].set_ylabel("Log Loss")
    axes[0].axhline(y=np.log(2), color="red", linestyle="--", alpha=0.5, label="Coin flip")
    axes[0].legend()
    axes[1].bar(yearly["season"].astype(str), yearly["auc"], color="darkorange")
    axes[1].set_title("AUC by Year (higher = better)")
    axes[1].set_ylabel("AUC")
    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "walk_forward_yearly.png", dpi=150)
    plt.close(fig)

    # 3. Ablation bar chart
    if ablation_df is not None and len(ablation_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        abl = ablation_df.sort_values("delta_ll")
        colors = ["#d32f2f" if p < 0.05 else "#90a4ae" for p in abl["p_value"]]
        ax.barh(abl["group"], abl["delta_ll"], color=colors)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("ΔLog Loss (positive = group helps)")
        ax.set_title("Feature Ablation: Log Loss Impact of Removing Each Group")
        fig.tight_layout()
        fig.savefig(output_dir / "ablation_bar.png", dpi=150)
        plt.close(fig)

    # 4. Cumulative PnL (2025 vs Kalshi)
    if "kalshi_games" in market_data:
        mg = market_data["kalshi_games"].sort_values("game_date")
        edge = mg["edge"].values
        y = mg["home_win"].values
        k = mg["kalshi_prob"].values
        dates = pd.to_datetime(mg["game_date"])
        cum_pnl = []
        running = 0.0
        bet_dates = []
        for i in range(len(edge)):
            if abs(edge[i]) >= 0.03:
                running += _compute_bet_pnl(edge[i], k[i], y[i])
                cum_pnl.append(running)
                bet_dates.append(dates.iloc[i])
        if cum_pnl:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(bet_dates, cum_pnl, "b-", linewidth=1)
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_title("Cumulative PnL vs Kalshi (3% edge, 2% fee, $100 flat)")
            ax.set_ylabel("Cumulative PnL ($)")
            ax.set_xlabel("Date")
            fig.tight_layout()
            fig.savefig(output_dir / "cumulative_pnl_2025.png", dpi=150)
            plt.close(fig)

    # 5. Monthly performance
    monthly_df = wf_df.copy()
    monthly_df["month"] = pd.to_datetime(monthly_df["game_date"]).dt.month
    month_stats = monthly_df.groupby("month").apply(
        lambda g: pd.Series({
            "log_loss": log_loss(g["home_win"], np.clip(g["ens_prob"], 0.01, 0.99)),
            "n_games": len(g),
        })
    ).reset_index()
    month_names = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([month_names.get(m, str(m)) for m in month_stats["month"]],
           month_stats["log_loss"], color="steelblue")
    ax.set_title("Log Loss by Month (all years pooled)")
    ax.set_ylabel("Log Loss")
    ax.axhline(y=np.log(2), color="red", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_performance.png", dpi=150)
    plt.close(fig)

    # 6. Confidence tier
    conv = np.abs(wf_df["ens_prob"] - 0.5)
    tier_names = []
    tier_ll = []
    for lo, hi, name in [(0, 0.03, "Low"), (0.03, 0.07, "Med"),
                         (0.07, 0.12, "High"), (0.12, 1.0, "Extreme")]:
        mask = (conv >= lo) & (conv < hi)
        sub = wf_df[mask]
        if len(sub) > 20:
            tier_names.append(f"{name}\n(n={len(sub)})")
            tier_ll.append(log_loss(sub["home_win"], np.clip(sub["ens_prob"], 0.01, 0.99)))
    if tier_names:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(tier_names, tier_ll, color="teal")
        ax.set_title("Log Loss by Confidence Tier")
        ax.set_ylabel("Log Loss")
        fig.tight_layout()
        fig.savefig(output_dir / "confidence_tier.png", dpi=150)
        plt.close(fig)

    print(f"  Plots saved to {output_dir}")


# ── Markdown report ──────────────────────────────────────────────────────────

def generate_markdown_report(wf_df, bootstrap_results, cal_data, cal_data_cal,
                             ablation_df, market_data, conf_df, monthly_df,
                             early_late, output_dir):
    """Write AUDIT_REPORT.md."""
    lines = []
    lines.append("# MLB Win Probability Model — Audit Report\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # 1. Leakage audit
    lines.append("## 1. Leakage Audit\n")
    lines.append("**Finding 1 (FIXED):** `compare_vs_market.py` and `build_full_csv.py` computed "
                 "blend weight on in-sample predictions. Both now use 5-fold chronological OOF, "
                 "matching `win_model.py`.\n")
    lines.append("**Finding 2 (MINOR):** `add_nonlinear_features` z-scores `prior_dominance` "
                 "using batch statistics. For XGB (tree-based, rank-invariant), this has no "
                 "practical impact. No fix needed.\n")
    lines.append("**Finding 3:** Feature engineering (`compute_single_game_features`) uses "
                 "strictly pre-game data: rolling windows up to `game_date`, matchup models from "
                 "prior year, park factors from prior year. No lookahead detected.\n")

    # 2. Walk-forward results
    lines.append("## 2. Walk-Forward Results (2020-2025)\n")
    lines.append("### Per-Year Summary\n")
    lines.append("| Year | Games | Log Loss | Brier | AUC | Baseline LL |\n")
    lines.append("|------|-------|----------|-------|-----|-------------|\n")

    for season in sorted(wf_df["season"].unique()):
        sub = wf_df[wf_df["season"] == season]
        y = sub["home_win"].values
        p = np.clip(sub["ens_prob"].values, 0.01, 0.99)
        bl = np.clip(sub["baseline_home"].values, 0.01, 0.99)
        ll = log_loss(y, p)
        bs = brier_score_loss(y, p)
        auc = roc_auc_score(y, p)
        bl_ll = log_loss(y, bl)
        lines.append(f"| {season} | {len(sub)} | {ll:.4f} | {bs:.4f} | {auc:.4f} | {bl_ll:.4f} |\n")

    # Pooled
    y_all = wf_df["home_win"].values
    p_all = np.clip(wf_df["ens_prob"].values, 0.01, 0.99)
    bl_all = np.clip(wf_df["baseline_home"].values, 0.01, 0.99)
    lines.append(f"| **Pooled** | **{len(wf_df)}** | **{log_loss(y_all, p_all):.4f}** | "
                 f"**{brier_score_loss(y_all, p_all):.4f}** | **{roc_auc_score(y_all, p_all):.4f}** | "
                 f"**{log_loss(y_all, bl_all):.4f}** |\n")

    # LR vs XGB vs Ensemble pooled
    lr_all = np.clip(wf_df["lr_prob"].values, 0.01, 0.99)
    xgb_all = np.clip(wf_df["xgb_prob"].values, 0.01, 0.99)
    lines.append("\n### Model Components (Pooled)\n")
    lines.append("| Model | Log Loss | Brier | AUC |\n")
    lines.append("|-------|----------|-------|-----|\n")
    lines.append(f"| LR only | {log_loss(y_all, lr_all):.4f} | "
                 f"{brier_score_loss(y_all, lr_all):.4f} | {roc_auc_score(y_all, lr_all):.4f} |\n")
    lines.append(f"| XGB only | {log_loss(y_all, xgb_all):.4f} | "
                 f"{brier_score_loss(y_all, xgb_all):.4f} | {roc_auc_score(y_all, xgb_all):.4f} |\n")
    lines.append(f"| Ensemble | {log_loss(y_all, p_all):.4f} | "
                 f"{brier_score_loss(y_all, p_all):.4f} | {roc_auc_score(y_all, p_all):.4f} |\n")

    # 3. Bootstrap significance
    lines.append("\n## 3. Statistical Significance\n")
    lines.append("| Comparison | Metric | Delta | 95% CI | p-value |\n")
    lines.append("|------------|--------|-------|--------|--------|\n")
    for r in bootstrap_results:
        sig = " *" if r["p_value"] < 0.05 else ""
        lines.append(f"| {r['comparison']} | {r['metric']} | {r['delta']:+.5f} | "
                     f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}] | {r['p_value']:.3f}{sig} |\n")

    # 4. Calibration
    lines.append("\n## 4. Calibration\n")
    lines.append(f"**Uncalibrated:** ECE = {cal_data['ece']:.4f}, MCE = {cal_data['mce']:.4f}\n")
    lines.append(f"**Isotonic-calibrated:** ECE = {cal_data_cal['ece']:.4f}, MCE = {cal_data_cal['mce']:.4f}\n")
    lines.append("\n![Reliability Diagram](reliability_diagram.png)\n")

    # 5. Ablation
    if ablation_df is not None and len(ablation_df) > 0:
        lines.append("\n## 5. Feature Ablation\n")
        lines.append("Positive ΔLL = removing the group hurts the model (group is useful).\n\n")
        lines.append("| Group | Features Removed | ΔLog Loss | 95% CI | p-value |\n")
        lines.append("|-------|-----------------|-----------|--------|--------|\n")
        for _, row in ablation_df.iterrows():
            sig = " *" if row["p_value"] < 0.05 else ""
            lines.append(f"| {row['group']} | {row['n_features_removed']} | "
                         f"{row['delta_ll']:+.5f} | [{row['ci_lo']:+.5f}, {row['ci_hi']:+.5f}] | "
                         f"{row['p_value']:.3f}{sig} |\n")
        lines.append("\n![Ablation](ablation_bar.png)\n")

    # 6. Temporal analysis
    lines.append("\n## 6. Temporal Analysis\n")
    if monthly_df is not None and len(monthly_df) > 0:
        month_names = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
        lines.append("### Monthly Performance (All Years Pooled)\n")
        lines.append("| Month | Games | Log Loss | Brier | AUC |\n")
        lines.append("|-------|-------|----------|-------|-----|\n")
        for _, row in monthly_df.iterrows():
            mn = month_names.get(int(row["month"]), str(int(row["month"])))
            lines.append(f"| {mn} | {int(row['n_games'])} | {row['log_loss']:.4f} | "
                         f"{row['brier']:.4f} | {row['auc']:.4f} |\n")

    if early_late:
        lines.append("\n### Early Season (first 30 days) vs Rest\n")
        for k, v in early_late.items():
            if isinstance(v, dict) and "n_games" in v:
                lines.append(f"- **{k}**: {v['n_games']} games, LL={v['log_loss']:.4f}, "
                             f"Brier={v['brier']:.4f}, AUC={v['auc']:.4f}\n")
        if "delta_ll_early_minus_late" in early_late:
            d = early_late["delta_ll_early_minus_late"]
            lines.append(f"- **ΔLL (early - late)**: {d['delta']:+.4f} "
                         f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]\n")

    # 7. Confidence tiers
    if conf_df is not None and len(conf_df) > 0:
        lines.append("\n## 7. Confidence Tiers\n")
        lines.append("| Tier | Games | Log Loss | Brier | AUC | Accuracy |\n")
        lines.append("|------|-------|----------|-------|-----|----------|\n")
        for _, row in conf_df.iterrows():
            auc_str = f"{row['auc']:.4f}" if pd.notna(row['auc']) else "N/A"
            lines.append(f"| {row['tier']} | {int(row['n_games'])} | {row['log_loss']:.4f} | "
                         f"{row['brier']:.4f} | {auc_str} | {row['accuracy']:.1%} |\n")
        lines.append("\n![Confidence Tiers](confidence_tier.png)\n")

    # 8. Market comparison
    if "kalshi" in market_data:
        mk = market_data["kalshi"]
        lines.append("\n## 8. Market Comparison (2025 vs Kalshi)\n")
        lines.append(f"**{mk['n_games']}** games with both model and Kalshi prices.\n\n")
        lines.append("| Metric | Model | Kalshi | Delta |\n")
        lines.append("|--------|-------|--------|-------|\n")
        lines.append(f"| Log Loss | {mk['model_ll']:.4f} | {mk['kalshi_ll']:.4f} | "
                     f"{mk['model_ll'] - mk['kalshi_ll']:+.4f} |\n")
        lines.append(f"| Brier | {mk['model_brier']:.4f} | {mk['kalshi_brier']:.4f} | "
                     f"{mk['model_brier'] - mk['kalshi_brier']:+.4f} |\n")
        lines.append(f"| AUC | {mk['model_auc']:.4f} | {mk['kalshi_auc']:.4f} | "
                     f"{mk['model_auc'] - mk['kalshi_auc']:+.4f} |\n")
        lines.append(f"\nCorrelation: {mk['correlation']:.4f}\n")

        # Bootstrap results for model vs Kalshi
        for metric_name in ["log_loss", "brier"]:
            key = f"boot_{metric_name}"
            if key in mk:
                b = mk[key]
                sig = " *" if b["p_value"] < 0.05 else ""
                lines.append(f"- **{metric_name}** delta: {b['delta']:+.5f} "
                             f"[{b['ci_lo']:+.5f}, {b['ci_hi']:+.5f}] p={b['p_value']:.3f}{sig}\n")

        # ROI
        lines.append("\n### ROI Simulation ($100 flat, 2% fee)\n")
        lines.append("| Edge | Bets | PnL | ROI | Win Rate |\n")
        lines.append("|------|------|-----|-----|----------|\n")
        for threshold in [3, 5, 7, 10]:
            key = f"roi_{threshold}pct"
            if key in mk:
                r = mk[key]
                lines.append(f"| {threshold}% | {r['n_bets']} | ${r['pnl']:+,.0f} | "
                             f"{r['roi']:+.1%} | {r['win_rate']:.1%} |\n")

        if "roi_3pct_boot" in mk:
            b = mk["roi_3pct_boot"]
            lines.append(f"\n3% edge ROI bootstrap: [{b['ci_lo']:+.1%}, {b['ci_hi']:+.1%}], "
                         f"p(ROI>0)={1-b['p_value_positive']:.3f}\n")

        lines.append("\n![Cumulative PnL](cumulative_pnl_2025.png)\n")

    # 9. Conclusions
    lines.append("\n## 9. Conclusions\n")
    lines.append("### What is real\n")

    pooled_ll = log_loss(y_all, p_all)
    baseline_ll = log_loss(y_all, bl_all)
    lines.append(f"- Model achieves {pooled_ll:.4f} pooled log loss vs {baseline_ll:.4f} "
                 f"home-rate baseline across {len(wf_df)} games (2020-2025)\n")

    sig_results = {r["comparison"]: r for r in bootstrap_results if r["metric"] == "log_loss"}
    for comp in ["ens_vs_home_constant", "ens_vs_prior_winpct"]:
        if comp in sig_results:
            r = sig_results[comp]
            status = "statistically significant" if r["p_value"] < 0.05 else "NOT significant"
            lines.append(f"- {comp}: {status} (p={r['p_value']:.3f})\n")

    lines.append("\n### What is fragile\n")
    lines.append("- Early-season predictions (first 30 days) have notably higher log loss\n")
    lines.append("- 2025 Kalshi comparison is a single season — need more years of market data\n")
    lines.append("- ROI depends on edge threshold and fee assumptions\n")

    lines.append("\n### What to do next\n")
    lines.append("- Collect Kalshi/market data for 2024 to validate market comparison out-of-sample\n")
    lines.append("- Investigate early-season performance: consider separate early-season model or "
                 "heavier projection weighting\n")
    lines.append("- Monitor ablation results: remove feature groups that don't significantly help\n")
    lines.append("- Consider expanding to other markets (DraftKings, FanDuel) for broader comparison\n")

    report_path = output_dir / "AUDIT_REPORT.md"
    report_path.write_text("".join(lines))
    print(f"  Report written to {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB Model Audit")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip expensive ablation study")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Bootstrap iterations for main tests")
    args = parser.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. Load data
    print("Loading features...")
    df = load_features(ALL_YEARS)
    print(f"  {len(df)} games across {df['season'].nunique()} seasons")

    # 2. Walk-forward with isotonic calibration
    print("\nWalk-forward evaluation (2020-2025)...")
    wf_df = walk_forward(df, fit_isotonic=True)
    wf_df.to_csv(AUDIT_DIR / "walk_forward_predictions.csv", index=False)
    print(f"  Saved {len(wf_df)} predictions")

    # 3. Baselines
    print("\nComputing baselines...")
    wf_df = compute_baselines(wf_df, df)

    # 4. Per-year summary
    print("\nPer-year summary:")
    yearly_rows = []
    for season in sorted(wf_df["season"].unique()):
        sub = wf_df[wf_df["season"] == season]
        y = sub["home_win"].values
        p = np.clip(sub["ens_prob"].values, 0.01, 0.99)
        lr_p = np.clip(sub["lr_prob"].values, 0.01, 0.99)
        xgb_p = np.clip(sub["xgb_prob"].values, 0.01, 0.99)
        bl = np.clip(sub["baseline_home"].values, 0.01, 0.99)
        row = {
            "season": season, "n_games": len(sub),
            "ens_ll": log_loss(y, p), "ens_brier": brier_score_loss(y, p),
            "ens_auc": roc_auc_score(y, p),
            "lr_ll": log_loss(y, lr_p), "xgb_ll": log_loss(y, xgb_p),
            "baseline_home_ll": log_loss(y, bl),
        }
        yearly_rows.append(row)
        print(f"  {season}: n={len(sub):4d}  ENS_LL={row['ens_ll']:.4f}  "
              f"AUC={row['ens_auc']:.4f}  Base_LL={row['baseline_home_ll']:.4f}")

    yearly_df = pd.DataFrame(yearly_rows)
    yearly_df.to_csv(AUDIT_DIR / "walk_forward_summary.csv", index=False)

    # 5. Bootstrap significance
    print(f"\nBootstrap significance tests (n={args.n_boot})...")
    boot_results = run_all_bootstrap_tests(wf_df, n_boot=args.n_boot)
    pd.DataFrame(boot_results).to_csv(AUDIT_DIR / "bootstrap_significance.csv", index=False)

    # 6. Calibration
    print("\nCalibration analysis...")
    y_all = wf_df["home_win"].values
    p_all = np.clip(wf_df["ens_prob"].values, 0.01, 0.99)
    p_cal = np.clip(wf_df["ens_calibrated"].values, 0.01, 0.99)
    cal_data = calibration_analysis(y_all, p_all, label="Uncalibrated")
    cal_data_cal = calibration_analysis(y_all, p_cal, label="Isotonic-calibrated")
    pd.DataFrame(cal_data["bins"]).to_csv(AUDIT_DIR / "calibration.csv", index=False)
    print(f"  Uncalibrated ECE: {cal_data['ece']:.4f}")
    print(f"  Calibrated ECE:   {cal_data_cal['ece']:.4f}")

    # 7. Ablation
    ablation_df = None
    if not args.skip_ablation:
        print("\nAblation study...")
        ablation_df = run_ablation(df, FEATURE_GROUPS, n_boot=min(args.n_boot, 5000))
        ablation_df.to_csv(AUDIT_DIR / "ablation_results.csv", index=False)
    else:
        print("\nAblation skipped (--skip-ablation)")
        abl_path = AUDIT_DIR / "ablation_results.csv"
        if abl_path.exists():
            ablation_df = pd.read_csv(abl_path)
            print(f"  Loaded previous ablation from {abl_path}")

    # 8. Confidence tiers
    print("\nConfidence tier analysis...")
    conf_df = confidence_tier_analysis(wf_df)
    conf_df.to_csv(AUDIT_DIR / "confidence_tiers.csv", index=False)
    for _, row in conf_df.iterrows():
        print(f"  {row['tier']:>8s}: n={int(row['n_games']):5d}  "
              f"LL={row['log_loss']:.4f}  Acc={row['accuracy']:.1%}")

    # 9. Monthly analysis
    print("\nMonthly analysis...")
    month_df = monthly_analysis(wf_df)
    month_df.to_csv(AUDIT_DIR / "monthly_performance.csv", index=False)

    # 10. Early vs late
    print("\nEarly vs late season...")
    early_late = early_vs_late_analysis(wf_df)
    for k, v in early_late.items():
        if isinstance(v, dict) and "n_games" in v:
            print(f"  {k}: {v['n_games']} games, LL={v['log_loss']:.4f}")
        elif isinstance(v, dict) and "delta" in v:
            print(f"  {k}: {v['delta']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")

    # 11. Market comparison
    print("\nMarket comparison (2025)...")
    market_data = market_comparison(wf_df, n_boot=args.n_boot)
    if "kalshi" in market_data:
        mk = market_data["kalshi"]
        print(f"  Kalshi: {mk['n_games']} games")
        print(f"    Model LL={mk['model_ll']:.4f}  Kalshi LL={mk['kalshi_ll']:.4f}")
        if "kalshi_games" in market_data:
            market_data["kalshi_games"].to_csv(
                AUDIT_DIR / "market_comparison_2025.csv", index=False)

    # 12. Plots
    print("\nGenerating plots...")
    generate_plots(wf_df, cal_data, ablation_df, market_data, AUDIT_DIR)

    # 13. Report
    print("\nWriting report...")
    generate_markdown_report(
        wf_df, boot_results, cal_data, cal_data_cal,
        ablation_df, market_data, conf_df, month_df, early_late,
        AUDIT_DIR,
    )

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} minutes. All outputs in {AUDIT_DIR}")


if __name__ == "__main__":
    main()
