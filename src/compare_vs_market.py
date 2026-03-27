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
from win_model import DIFF_FEATURES, RAW_FEATURES, ALL_FEATURES, add_nonlinear_features, _smart_fillna

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_DIR = DATA_DIR / "features"
ANALYSIS_DIR = DATA_DIR / "analysis"

MONTH_NAMES = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul",
               8: "Aug", 9: "Sep", 10: "Oct"}
EDGE_THRESHOLDS = [0.03, 0.05, 0.07, 0.10]


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
    X_filled, train_medians = _smart_fillna(X)
    X_lr = scaler.fit_transform(X_filled)
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

    return lr, scaler, xgb_model, available, w_lr, train_medians


def predict(lr, scaler, xgb_model, features, test_df, w_lr=0.5, train_medians=None):
    available = [f for f in features if f in test_df.columns]
    X = test_df[available].copy()

    X_filled, _ = _smart_fillna(X, train_medians)
    X_lr = scaler.transform(X_filled)
    lr_probs = lr.predict_proba(X_lr)[:, 1]

    if xgb_model and HAS_XGB:
        dtest = xgb.DMatrix(X)
        xgb_probs = xgb_model.predict(dtest)
        ens_probs = w_lr * lr_probs + (1 - w_lr) * xgb_probs
    else:
        xgb_probs = lr_probs
        ens_probs = lr_probs

    return lr_probs, xgb_probs, ens_probs


def _sharpe(pnls, bets_per_year=500):
    """Annualized Sharpe ratio from an array of per-bet PnLs."""
    if len(pnls) < 2:
        return 0.0
    mean = np.mean(pnls)
    std = np.std(pnls, ddof=1)
    if std < 1e-9:
        return 0.0
    return (mean / std) * np.sqrt(min(len(pnls), bets_per_year))


def _compute_bet_pnl(edge, market_prob, outcome):
    """Compute PnL for a single flat $100 bet at fair odds (no vig)."""
    if edge > 0:
        # Bet home
        fair_odds = 1 / market_prob
        return (fair_odds - 1) * 100 if outcome == 1 else -100.0
    else:
        # Bet away
        fair_odds = 1 / (1 - market_prob)
        return (fair_odds - 1) * 100 if outcome == 0 else -100.0


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

    for threshold in EDGE_THRESHOLDS:
        home_edge = edge >= threshold
        if home_edge.sum() > 0:
            win_rate = y[home_edge].mean()
            avg_market = market_probs[home_edge].mean()
            avg_model = model_probs[home_edge].mean()
            print(f"\n    Model likes HOME by >={threshold:.0%}: {home_edge.sum()} games")
            print(f"      Actual win rate: {win_rate:.1%} (market implied: {avg_market:.1%}, model: {avg_model:.1%})")

        away_edge = edge <= -threshold
        if away_edge.sum() > 0:
            away_win_rate = 1 - y[away_edge].mean()
            avg_market = (1 - market_probs[away_edge]).mean()
            avg_model = (1 - model_probs[away_edge]).mean()
            print(f"    Model likes AWAY by >={threshold:.0%}: {away_edge.sum()} games")
            print(f"      Actual win rate: {away_win_rate:.1%} (market implied: {avg_market:.1%}, model: {avg_model:.1%})")

    # ROI simulation (flat $100 bets)
    print(f"\n  ROI Simulation (flat $100 bets):")
    print(f"  {'Edge':>7s} {'Bets':>6s} {'PnL':>10s} {'ROI':>8s} {'Sharpe':>8s} {'WinRate':>8s}")
    print(f"  {'-'*51}")
    for min_edge in EDGE_THRESHOLDS:
        pnls = []
        for i in range(len(df)):
            e = edge[i]
            if abs(e) >= min_edge:
                pnls.append(_compute_bet_pnl(e, market_probs[i], y[i]))
        if pnls:
            total_pnl = sum(pnls)
            n_bets = len(pnls)
            roi = total_pnl / (n_bets * 100)
            sharpe = _sharpe(np.array(pnls))
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / n_bets
            print(f"  {min_edge:>6.0%} {n_bets:>6d} ${total_pnl:>+9.0f} "
                  f"{roi:>+7.1%} {sharpe:>8.2f} {wr:>7.1%}")


def detailed_roi_analysis(df, model_col, market_col, label=""):
    """Monthly ROI breakdown, edge decomposition, and per-bet CSV export."""
    df = df.dropna(subset=[model_col, market_col, "home_win"]).copy()
    if len(df) == 0:
        return

    y = df["home_win"].values
    model_probs = np.clip(df[model_col].values, 0.01, 0.99)
    market_probs = np.clip(df[market_col].values, 0.01, 0.99)
    edge = model_probs - market_probs
    dates = pd.to_datetime(df["game_date"])
    months = dates.dt.month.values
    edge_default = 0.05

    # ── Monthly ROI table ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Monthly ROI Breakdown: {label}")
    print(f"{'='*80}")
    print(f"  {'Mon':<5s} {'Games':>5s} {'Bets':>5s} {'PnL':>9s} {'ROI':>7s} "
          f"{'Sharpe':>7s} {'WR':>6s} {'MLL':>7s} {'MktLL':>7s} {'Δ':>7s}")
    print(f"  {'-'*73}")

    for month in sorted(set(months)):
        mask = months == month
        if mask.sum() < 5:
            continue
        m_edge = edge[mask]
        m_model = model_probs[mask]
        m_market = market_probs[mask]
        m_y = y[mask]
        name = MONTH_NAMES.get(month, str(month))

        try:
            mll = log_loss(m_y, m_model)
            mkll = log_loss(m_y, m_market)
        except ValueError:
            continue

        pnls = []
        for i in range(len(m_edge)):
            if abs(m_edge[i]) >= edge_default:
                pnls.append(_compute_bet_pnl(m_edge[i], m_market[i], m_y[i]))

        n_bets = len(pnls)
        if n_bets > 0:
            total_pnl = sum(pnls)
            roi = total_pnl / (n_bets * 100)
            sharpe = _sharpe(np.array(pnls))
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / n_bets
            print(f"  {name:<5s} {mask.sum():>5d} {n_bets:>5d} ${total_pnl:>+8.0f} "
                  f"{roi:>+6.1%} {sharpe:>7.2f} {wr:>5.1%} "
                  f"{mll:>7.4f} {mkll:>7.4f} {mll-mkll:>+7.4f}")
        else:
            print(f"  {name:<5s} {mask.sum():>5d} {0:>5d} {'---':>9s} "
                  f"{'---':>7s} {'---':>7s} {'---':>6s} "
                  f"{mll:>7.4f} {mkll:>7.4f} {mll-mkll:>+7.4f}")

    # ── Monthly edge decomposition ───────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Monthly Edge Decomposition")
    print(f"{'='*80}")
    print(f"  {'Mon':<5s} {'AvgEdge':>8s} {'HomeFav':>8s} {'AwayFav':>8s} "
          f"{'Dog':>8s} {'HitRate':>8s}")
    print(f"  {'-'*47}")

    for month in sorted(set(months)):
        mask = months == month
        if mask.sum() < 10:
            continue
        m_edge = edge[mask]
        m_model = model_probs[mask]
        m_market = market_probs[mask]
        m_y = y[mask]
        name = MONTH_NAMES.get(month, str(month))

        bet_mask = np.abs(m_edge) >= edge_default
        if bet_mask.sum() == 0:
            continue

        avg_edge = np.abs(m_edge[bet_mask]).mean()

        # Home favorite bets: edge > 0 and market > 0.5
        hf = (m_edge >= edge_default) & (m_market > 0.5)
        hf_wr = m_y[hf].mean() if hf.sum() > 0 else float("nan")

        # Away favorite bets: edge < 0 and market < 0.5
        af = (m_edge <= -edge_default) & (m_market < 0.5)
        af_wr = (1 - m_y[af]).mean() if af.sum() > 0 else float("nan")

        # Underdog bets
        dog = ((m_edge >= edge_default) & (m_market <= 0.5)) | \
              ((m_edge <= -edge_default) & (m_market >= 0.5))
        dog_wr_vals = []
        for i in np.where(dog)[0]:
            if m_edge[i] > 0:
                dog_wr_vals.append(m_y[i])
            else:
                dog_wr_vals.append(1 - m_y[i])
        dog_wr = np.mean(dog_wr_vals) if dog_wr_vals else float("nan")

        # Overall hit rate
        hit = 0
        total = 0
        for i in range(len(m_edge)):
            if abs(m_edge[i]) >= edge_default:
                total += 1
                if m_edge[i] > 0 and m_y[i] == 1:
                    hit += 1
                elif m_edge[i] < 0 and m_y[i] == 0:
                    hit += 1
        hit_rate = hit / total if total > 0 else float("nan")

        print(f"  {name:<5s} {avg_edge:>7.1%} "
              f"{hf_wr:>7.1%} {af_wr:>7.1%} {dog_wr:>7.1%} {hit_rate:>7.1%}")

    # ── Per-bet CSV ──────────────────────────────────────────────────────
    _write_pnl_csv(df, model_probs, market_probs, edge, y, label)


def _write_pnl_csv(df, model_probs, market_probs, edge, y, label):
    """Dump per-bet results to CSV for external analysis / equity curve plotting."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    cumulative = 0.0
    for i in range(len(df)):
        if abs(edge[i]) < 0.03:
            continue
        bet_side = "home" if edge[i] > 0 else "away"
        result = int((edge[i] > 0 and y[i] == 1) or (edge[i] < 0 and y[i] == 0))
        pnl = _compute_bet_pnl(edge[i], market_probs[i], y[i])
        cumulative += pnl
        rows.append({
            "game_date": df.iloc[i].get("game_date", ""),
            "home_team": df.iloc[i].get("home_team", ""),
            "away_team": df.iloc[i].get("away_team", ""),
            "model_prob": model_probs[i],
            "market_prob": market_probs[i],
            "edge": edge[i],
            "bet_side": bet_side,
            "result": result,
            "pnl": pnl,
            "cumulative_pnl": cumulative,
        })

    out = pd.DataFrame(rows)
    safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path = ANALYSIS_DIR / f"pnl_{safe_label}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  Per-bet CSV written to {out_path} ({len(out)} rows)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading training data (2018-2024)...")
    train_df = load_features(list(range(2018, 2025)))
    print(f"  {len(train_df)} training games")

    print("\nTraining ensemble model...")
    lr, scaler, xgb_model, features, w_lr, train_medians = train_ensemble(train_df)
    print(f"  Features used: {len(features)}")

    print("\nLoading 2025 test data...")
    test_df = load_features([2025])
    print(f"  {len(test_df)} test games")

    # Generate predictions
    lr_probs, xgb_probs, ens_probs = predict(lr, scaler, xgb_model, features, test_df, w_lr, train_medians)
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
        merged = test_df.merge(
            poly[["game_pk", "poly_home_prob"]].drop_duplicates("game_pk"),
            on="game_pk", how="inner"
        )
        print(f"\n  Polymarket overlap: {len(merged)} games")
        evaluate_vs_market(merged, "model_prob", "poly_home_prob", "Model vs Polymarket (ENS)")
        detailed_roi_analysis(merged, "model_prob", "poly_home_prob", "Model vs Polymarket")
    else:
        print("\n  No Polymarket data found")

    # Load Kalshi data
    kalshi_path = DATA_DIR / "kalshi" / "kalshi_mlb_2025.parquet"
    if kalshi_path.exists():
        kalshi = pd.read_parquet(kalshi_path)
        kalshi["game_date"] = kalshi["game_date"].astype(str)
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
            detailed_roi_analysis(merged_k, "model_prob", "kalshi_home_prob", "Model vs Kalshi")
    else:
        print("\n  No Kalshi data found")


if __name__ == "__main__":
    main()
