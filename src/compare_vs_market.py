#!/usr/bin/env python3
"""
Compare model predictions vs prediction market prices (Polymarket, Kalshi).

Runs walk-forward evaluation on 2025 and merges with market data to compute:
  - Log loss / Brier score comparison (model vs market)
  - Edge analysis: when model disagrees with market, who's right?
  - ROI simulation: bet when model has edge over market
  - Calibration comparison
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
from win_model import DIFF_FEATURES, RAW_FEATURES, ALL_FEATURES, add_nonlinear_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_DIR = DATA_DIR / "features"


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

        # Learn optimal blend weight
        from scipy.optimize import minimize_scalar
        lr_train_probs = lr.predict_proba(X_lr)[:, 1]
        xgb_train_probs = xgb_model.predict(dtrain)

        def blend_loss(w):
            blended = w * lr_train_probs + (1 - w) * xgb_train_probs
            blended = np.clip(blended, 1e-6, 1 - 1e-6)
            return log_loss(y, blended)

        opt = minimize_scalar(blend_loss, bounds=(0.1, 0.9), method="bounded")
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

    return lr_probs, xgb_probs, ens_probs


def evaluate_vs_market(results_df, model_col, market_col, label=""):
    """Compare model vs market predictions."""
    df = results_df.dropna(subset=[model_col, market_col, "home_win"])
    if len(df) == 0:
        print(f"  No overlapping games for {label}")
        return

    y = df["home_win"].values
    model_probs = df[model_col].values
    market_probs = df[market_col].values

    # Clip to avoid log(0)
    model_probs = np.clip(model_probs, 0.01, 0.99)
    market_probs = np.clip(market_probs, 0.01, 0.99)

    model_ll = log_loss(y, model_probs)
    market_ll = log_loss(y, market_probs)
    model_bs = brier_score_loss(y, model_probs)
    market_bs = brier_score_loss(y, market_probs)
    model_auc = roc_auc_score(y, model_probs)
    market_auc = roc_auc_score(y, market_probs)
    baseline_ll = log_loss(y, np.full_like(model_probs, y.mean()))

    print(f"\n{'='*60}")
    print(f"  {label} — {len(df)} games")
    print(f"{'='*60}")
    print(f"  {'Metric':<20s} {'Model':>10s} {'Market':>10s} {'Baseline':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'Log Loss':<20s} {model_ll:>10.4f} {market_ll:>10.4f} {baseline_ll:>10.4f}")
    print(f"  {'Brier Score':<20s} {model_bs:>10.4f} {market_bs:>10.4f}")
    print(f"  {'AUC':<20s} {model_auc:>10.4f} {market_auc:>10.4f}")

    # Calibration by quintile
    print(f"\n  Calibration (model):")
    for lo, hi in [(0.0, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 1.0)]:
        mask = (model_probs >= lo) & (model_probs < hi)
        if mask.sum() > 0:
            print(f"    [{lo:.0%}-{hi:.0%}): pred={model_probs[mask].mean():.3f} "
                  f"actual={y[mask].mean():.3f} n={mask.sum()}")

    print(f"\n  Calibration (market):")
    for lo, hi in [(0.0, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 1.0)]:
        mask = (market_probs >= lo) & (market_probs < hi)
        if mask.sum() > 0:
            print(f"    [{lo:.0%}-{hi:.0%}): pred={market_probs[mask].mean():.3f} "
                  f"actual={y[mask].mean():.3f} n={mask.sum()}")

    # Edge analysis: when model disagrees with market
    edge = model_probs - market_probs
    print(f"\n  Edge Analysis (model - market):")
    print(f"    Mean edge: {edge.mean():+.4f}")
    print(f"    Std edge:  {edge.std():.4f}")

    for threshold in [0.03, 0.05, 0.07, 0.10]:
        # Model says home more likely than market
        home_edge = edge >= threshold
        if home_edge.sum() > 0:
            win_rate = y[home_edge].mean()
            avg_market = market_probs[home_edge].mean()
            avg_model = model_probs[home_edge].mean()
            print(f"\n    Model likes HOME by >={threshold:.0%}: {home_edge.sum()} games")
            print(f"      Actual win rate: {win_rate:.1%} (market implied: {avg_market:.1%}, model: {avg_model:.1%})")

        # Model says away more likely than market
        away_edge = edge <= -threshold
        if away_edge.sum() > 0:
            # For away edge, home_win=0 is the "correct" bet
            away_win_rate = 1 - y[away_edge].mean()
            avg_market = (1 - market_probs[away_edge]).mean()
            avg_model = (1 - model_probs[away_edge]).mean()
            print(f"    Model likes AWAY by >={threshold:.0%}: {away_edge.sum()} games")
            print(f"      Actual win rate: {away_win_rate:.1%} (market implied: {avg_market:.1%}, model: {avg_model:.1%})")

    # ROI simulation (flat betting, no vig for simplicity)
    print(f"\n  ROI Simulation (flat $100 bets, no vig):")
    for min_edge in [0.03, 0.05, 0.07, 0.10]:
        bets = 0
        pnl = 0.0
        for i in range(len(df)):
            e = edge[i]
            if abs(e) >= min_edge:
                bets += 1
                # Bet on model's preferred side
                if e > 0:  # bet home
                    fair_odds = 1 / market_probs[i]
                    pnl += (fair_odds - 1) * 100 if y[i] == 1 else -100
                else:  # bet away
                    fair_odds = 1 / (1 - market_probs[i])
                    pnl += (fair_odds - 1) * 100 if y[i] == 0 else -100
        if bets > 0:
            roi = pnl / (bets * 100)
            print(f"    Edge >= {min_edge:.0%}: {bets} bets, PnL=${pnl:+.0f}, ROI={roi:+.1%}")


def main():
    print("Loading training data (2018-2024)...")
    train_df = load_features(list(range(2018, 2025)))
    print(f"  {len(train_df)} training games")

    print("\nTraining ensemble model...")
    lr, scaler, xgb_model, features, w_lr = train_ensemble(train_df)
    print(f"  Features used: {len(features)}")

    print("\nLoading 2025 test data...")
    test_df = load_features([2025])
    print(f"  {len(test_df)} test games")

    # Generate predictions
    lr_probs, xgb_probs, ens_probs = predict(lr, scaler, xgb_model, features, test_df, w_lr)
    test_df["model_prob"] = ens_probs
    test_df["model_prob_lr"] = lr_probs
    test_df["model_prob_xgb"] = xgb_probs

    # Model-only evaluation
    y = test_df["home_win"].values
    print(f"\n  Model-only (2025, {len(test_df)} games):")
    print(f"    ENS log_loss={log_loss(y, ens_probs):.4f}, AUC={roc_auc_score(y, ens_probs):.4f}")
    if HAS_XGB:
        print(f"    LR  log_loss={log_loss(y, lr_probs):.4f}, AUC={roc_auc_score(y, lr_probs):.4f}")
        print(f"    XGB log_loss={log_loss(y, xgb_probs):.4f}, AUC={roc_auc_score(y, xgb_probs):.4f}")

    # Load Polymarket data
    poly_path = DATA_DIR / "polymarket" / "poly_mlb_2025_matched.parquet"
    if poly_path.exists():
        poly = pd.read_parquet(poly_path)
        # Merge on game_pk
        merged = test_df.merge(
            poly[["game_pk", "poly_home_prob"]].drop_duplicates("game_pk"),
            on="game_pk", how="inner"
        )
        print(f"\n  Polymarket overlap: {len(merged)} games")
        evaluate_vs_market(merged, "model_prob", "poly_home_prob", "Model vs Polymarket (ENS)")

        # Monthly breakdown
        merged["month"] = pd.to_datetime(merged["game_date"]).dt.month
        month_names = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
        print(f"\n{'='*60}")
        print("  Monthly Breakdown: Model vs Polymarket")
        print(f"{'='*60}")
        print(f"  {'Month':<6} {'N':>5} {'Model LL':>10} {'Poly LL':>10} {'Delta':>8} {'Model AUC':>10} {'Poly AUC':>10}")
        print(f"  {'-'*55}")
        for month in sorted(merged["month"].unique()):
            m = merged[merged["month"] == month]
            y_m = m["home_win"].values
            mp = np.clip(m["model_prob"].values, 0.01, 0.99)
            pp = np.clip(m["poly_home_prob"].values, 0.01, 0.99)
            try:
                mll = log_loss(y_m, mp)
                pll = log_loss(y_m, pp)
            except ValueError:
                continue
            try:
                mauc = roc_auc_score(y_m, mp)
                pauc = roc_auc_score(y_m, pp)
            except ValueError:
                mauc = pauc = float("nan")
            name = month_names.get(month, str(month))
            print(f"  {name:<6} {len(m):>5} {mll:>10.4f} {pll:>10.4f} {mll-pll:>+8.4f} {mauc:>10.4f} {pauc:>10.4f}")
    else:
        print("\n  No Polymarket data found")

    # Load Kalshi data
    kalshi_path = DATA_DIR / "kalshi" / "kalshi_mlb_2025.parquet"
    if kalshi_path.exists():
        kalshi = pd.read_parquet(kalshi_path)
        kalshi["game_date"] = kalshi["game_date"].astype(str)
        # Merge on game_date + teams
        test_df["game_date_str"] = test_df["game_date"].astype(str)
        merged_k = test_df.merge(
            kalshi[["game_date", "home_team", "away_team", "kalshi_home_prob"]].drop_duplicates(),
            left_on=["game_date_str", "home_team", "away_team"],
            right_on=["game_date", "home_team", "away_team"],
            how="inner",
            suffixes=("", "_kalshi"),
        )
        print(f"\n  Kalshi overlap: {len(merged_k)} games")
        if len(merged_k) > 0:
            evaluate_vs_market(merged_k, "model_prob", "kalshi_home_prob", "Model vs Kalshi (ENS)")
    else:
        print("\n  No Kalshi data found")


if __name__ == "__main__":
    main()
