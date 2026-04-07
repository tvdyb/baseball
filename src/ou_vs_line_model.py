"""
O/U vs Line Model: Predict OVER/UNDER outcome directly against the DK closing line.

Key design decisions:
  1. Bet signal is EDGE-BASED: model P(over/under) vs market's devigged P(over/under).
     Only bet when our edge exceeds the vig. A model saying P(under)=0.55 on a -200 under
     (implied 66.7%) is a NEGATIVE edge — we would never bet that.
  2. Hyperparameters are optimized via Optuna with time-series cross-validation on the
     training data — not hand-picked.
  3. Payouts use actual per-game DK closing odds, not flat -110.

Training strategy: Train on 2023-2024, validate walk-forward on 2025.
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────
# 1. Data loading
# ──────────────────────────────────────────────────────────────────────
def load_dk_lines_from_json(year: int) -> pd.DataFrame:
    """Extract DK closing O/U lines from the JSON odds dataset for a given year."""
    json_path = ROOT / "data" / "odds" / "mlb_odds_dataset.json"
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for date_str, games in data.items():
        if not date_str.startswith(str(year)):
            continue
        for g in games:
            gv = g["gameView"]
            totals = g["odds"].get("totals", [])
            dk = [t for t in totals if t["sportsbook"] == "draftkings"]
            if not dk:
                continue
            dk = dk[0]
            current = dk.get("currentLine", {})
            opening = dk.get("openingLine", {})
            ou_close = current.get("total")
            ou_open = opening.get("total")
            if ou_close is None:
                continue

            rows.append({
                "game_date": date_str,
                "home_team": gv["homeTeam"]["shortName"],
                "away_team": gv["awayTeam"]["shortName"],
                "home_score": gv.get("homeTeamScore"),
                "away_score": gv.get("awayTeamScore"),
                "ou_open": ou_open,
                "ou_close": ou_close,
                "over_close_odds": current.get("overOdds"),
                "under_close_odds": current.get("underOdds"),
            })

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    team_map = {"OAK": "ATH", "ARI": "AZ", "WSN": "WSH"}
    df["home_team"] = df["home_team"].replace(team_map)
    df["away_team"] = df["away_team"].replace(team_map)
    return df


def build_dataset(year: int, odds_df: pd.DataFrame = None) -> pd.DataFrame:
    """Build a single year's dataset by merging features, odds, and results."""
    feat_path = ROOT / "data" / "features" / f"game_features_{year}.parquet"
    features = pd.read_parquet(feat_path)
    features["game_date"] = pd.to_datetime(features["game_date"]).dt.strftime("%Y-%m-%d")

    if odds_df is None:
        odds_path = ROOT / "data" / "odds" / "sbr_mlb_2025.parquet"
        odds_df = pd.read_parquet(odds_path)
        odds_df["game_date"] = pd.to_datetime(odds_df["game_date"]).dt.strftime("%Y-%m-%d")

    merged = features.merge(
        odds_df[["game_date", "home_team", "away_team", "ou_close", "ou_open",
                 "over_close_odds", "under_close_odds"]].drop_duplicates(),
        on=["game_date", "home_team", "away_team"],
        how="inner",
    )

    merged["total_runs"] = merged["home_score"].astype(float) + merged["away_score"].astype(float)
    merged["is_push"] = merged["total_runs"] == merged["ou_close"]
    merged["over"] = (merged["total_runs"] > merged["ou_close"]).astype(int)

    print(f"  {year}: {len(features)} features, {len(odds_df)} odds rows -> {len(merged)} merged, "
          f"{merged['is_push'].sum()} pushes")
    return merged


# ──────────────────────────────────────────────────────────────────────
# 2. Odds helpers
# ──────────────────────────────────────────────────────────────────────
def american_to_prob(odds):
    """Convert American odds (scalar or array) to raw implied probability."""
    odds = pd.to_numeric(odds, errors="coerce") if not isinstance(odds, (int, float)) else odds
    return np.where(odds < 0, -odds / (-odds + 100), 100 / (odds + 100))


def american_to_decimal(odds):
    """Convert American odds (scalar or array) to decimal odds."""
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, 1.0 + 100.0 / np.abs(odds),
                    np.where(odds > 0, 1.0 + odds / 100.0, np.nan))


def devig(over_odds, under_odds):
    """Remove vig → return (fair_over_prob, fair_under_prob)."""
    p_over = american_to_prob(over_odds)
    p_under = american_to_prob(under_odds)
    total = p_over + p_under
    return p_over / total, p_under / total


# ──────────────────────────────────────────────────────────────────────
# 3. Feature engineering
# ──────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add derived features and return (df, feature_columns)."""
    df = df.copy()

    df["dk_ou_line"] = df["ou_close"]
    df["line_movement"] = df["ou_close"] - df["ou_open"]

    df["implied_over_prob"] = american_to_prob(df["over_close_odds"])
    df["implied_under_prob"] = american_to_prob(df["under_close_odds"])
    df["implied_vig"] = df["implied_over_prob"] + df["implied_under_prob"] - 1
    imp_total = df["implied_over_prob"] + df["implied_under_prob"]
    df["devigged_over_prob"] = df["implied_over_prob"] / imp_total
    df["devigged_under_prob"] = df["implied_under_prob"] / imp_total

    if "home_sp_xrv_mean" in df.columns and "away_sp_xrv_mean" in df.columns:
        df["sp_quality_sum"] = df["home_sp_xrv_mean"] + df["away_sp_xrv_mean"]
        df["sp_quality_diff"] = df["away_sp_xrv_mean"] - df["home_sp_xrv_mean"]

    if "home_hit_barrel_rate" in df.columns:
        df["lineup_power_sum"] = (
            df["home_hit_barrel_rate"].fillna(0) + df["away_hit_barrel_rate"].fillna(0)
        )

    if "home_bp_xrv_mean" in df.columns:
        df["bp_quality_sum"] = df["home_bp_xrv_mean"].fillna(0) + df["away_bp_xrv_mean"].fillna(0)
        df["bp_fatigue_sum"] = (
            df["home_bp_fatigue_score"].fillna(0) + df["away_bp_fatigue_score"].fillna(0)
        )

    if "home_sp_projected_era" in df.columns and "away_sp_projected_era" in df.columns:
        avg_era = (df["home_sp_projected_era"].fillna(4.5) + df["away_sp_projected_era"].fillna(4.5)) / 2
        df["line_vs_sp_era"] = df["dk_ou_line"] - avg_era * 2

    exclude = {
        "game_pk", "game_date", "home_team", "away_team", "home_win",
        "home_score", "away_score", "total_runs", "is_push", "over",
        "ou_close", "ou_open", "over_close_odds", "under_close_odds",
        "ou_result", "status", "book_totals", "book_ml",
        "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close",
        "is_home",
        # Keep devigged probs for edge computation but exclude from features
        # (they ARE used as features — the model should see what the market thinks)
    }
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype in ("float64", "int64", "float32", "int32", "bool")]

    return df, feature_cols


# ──────────────────────────────────────────────────────────────────────
# 4. Edge-based ROI evaluation
# ──────────────────────────────────────────────────────────────────────
def compute_edge_roi(model_p_over: np.ndarray,
                     actuals: np.ndarray,
                     mkt_p_over: np.ndarray,
                     over_odds_am: np.ndarray,
                     under_odds_am: np.ndarray,
                     min_edge: float = 0.02) -> dict:
    """Compute ROI by betting only when model edge exceeds market implied prob + min_edge.

    For each game:
      - model_p_over is our estimated P(over)
      - mkt_p_over is the devigged market P(over)
      - If model_p_over - mkt_p_over > min_edge → bet OVER at actual over odds
      - If mkt_p_over - model_p_over > min_edge → bet UNDER at actual under odds
      - Otherwise → no bet
    """
    model_p_under = 1.0 - model_p_over
    mkt_p_under = 1.0 - mkt_p_over

    over_edge = model_p_over - mkt_p_over
    under_edge = model_p_under - mkt_p_under

    over_dec = american_to_decimal(over_odds_am)
    under_dec = american_to_decimal(under_odds_am)

    # Fallback for missing odds
    fallback_dec = 1.0 + 100.0 / 110.0  # -110
    over_dec = np.where(np.isfinite(over_dec), over_dec, fallback_dec)
    under_dec = np.where(np.isfinite(under_dec), under_dec, fallback_dec)

    total_pnl = 0.0
    n_over = 0
    n_under = 0
    wins = 0
    per_bet_pnl = []

    for i in range(len(model_p_over)):
        if over_edge[i] > min_edge:
            # Bet OVER
            n_over += 1
            won = actuals[i] == 1
            pnl = (over_dec[i] - 1.0) if won else -1.0
            total_pnl += pnl
            per_bet_pnl.append(pnl)
            if won:
                wins += 1
        elif under_edge[i] > min_edge:
            # Bet UNDER
            n_under += 1
            won = actuals[i] == 0
            pnl = (under_dec[i] - 1.0) if won else -1.0
            total_pnl += pnl
            per_bet_pnl.append(pnl)
            if won:
                wins += 1

    n_total = n_over + n_under
    if n_total == 0:
        return {"min_edge": min_edge, "n_bets": 0, "roi": np.nan, "win_pct": np.nan}

    return {
        "min_edge": min_edge,
        "n_bets": n_total,
        "n_over": n_over,
        "n_under": n_under,
        "wins": wins,
        "win_pct": wins / n_total,
        "profit_units": total_pnl,
        "roi": total_pnl / n_total,
        "per_bet_pnl": per_bet_pnl,
    }


def bootstrap_edge_roi(model_p_over, actuals, mkt_p_over, over_odds, under_odds,
                        min_edge, n_boot=2000):
    """Bootstrap CI for edge-based ROI."""
    rng = np.random.default_rng(42)
    rois = []
    n = len(model_p_over)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        result = compute_edge_roi(model_p_over[idx], actuals[idx], mkt_p_over[idx],
                                  over_odds[idx], under_odds[idx], min_edge)
        if result["n_bets"] > 0 and not np.isnan(result["roi"]):
            rois.append(result["roi"])
    if not rois:
        return np.nan, np.nan, np.nan
    rois = np.array(rois)
    return np.median(rois), np.percentile(rois, 2.5), np.percentile(rois, 97.5)


# ──────────────────────────────────────────────────────────────────────
# 5. Optuna hyperparameter optimization
# ──────────────────────────────────────────────────────────────────────
def optimize_lgb(X_train: pd.DataFrame, y_train: np.ndarray,
                 n_trials: int = 100, n_splits: int = 4) -> dict:
    """Find optimal LightGBM hyperparameters using Optuna + TimeSeriesSplit CV."""

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 30, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "num_leaves": trial.suggest_int("num_leaves", 4, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(log_loss(y_val, preds))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["objective"] = "binary"
    best["metric"] = "binary_logloss"
    best["verbose"] = -1
    best["random_state"] = 42
    return best, study


# ──────────────────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("O/U vs Line Model — Edge-Based with Optimized Hyperparameters")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────
    print("\n[1] Loading data...")
    dk_2024 = load_dk_lines_from_json(2024)
    dk_2023 = load_dk_lines_from_json(2023)
    print(f"  2024 DK lines: {len(dk_2024)} games")
    print(f"  2023 DK lines: {len(dk_2023)} games")

    ds_2024 = build_dataset(2024, dk_2024)
    ds_2025 = build_dataset(2025)

    try:
        ds_2023 = build_dataset(2023, dk_2023)
        has_2023 = len(ds_2023) > 100
    except Exception as e:
        print(f"  2023 features not available: {e}")
        has_2023 = False
        ds_2023 = pd.DataFrame()

    # ── Build training set ────────────────────────────────────────
    print("\n[2] Building datasets...")
    if has_2023:
        train_full = pd.concat([ds_2023, ds_2024], ignore_index=True)
        print(f"  Training: 2023-2024 ({len(train_full)} games)")
    else:
        train_full = ds_2024
        print(f"  Training: 2024 only ({len(train_full)} games)")

    train_full = train_full[~train_full["is_push"]].copy()
    ds_2025_nopush = ds_2025[~ds_2025["is_push"]].copy()
    print(f"  After pushes: train={len(train_full)}, test_2025={len(ds_2025_nopush)}")

    train_full, feat_cols_train = engineer_features(train_full)
    ds_2025_nopush, feat_cols_test = engineer_features(ds_2025_nopush)

    feat_cols = sorted(set(feat_cols_train) & set(feat_cols_test))
    print(f"  Feature columns: {len(feat_cols)}")

    # ── Base rates ────────────────────────────────────────────────
    print(f"\n[3] Base rates: train over={train_full['over'].mean():.3f}, "
          f"test over={ds_2025_nopush['over'].mean():.3f}")

    X_train = train_full[feat_cols].copy()
    y_train = train_full["over"].values

    # Sort training data by date for proper time-series CV
    train_dates = pd.to_datetime(train_full["game_date"])
    date_order = train_dates.argsort()
    X_train = X_train.iloc[date_order].reset_index(drop=True)
    y_train = y_train[date_order]

    X_test = ds_2025_nopush[feat_cols].copy()
    y_test = ds_2025_nopush["over"].values
    naive_brier = brier_score_loss(y_test, np.full(len(y_test), y_test.mean()))

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # ── Optuna hyperparameter optimization ─────────────────────────
    print("\n" + "=" * 70)
    print("[4] OPTUNA HYPERPARAMETER OPTIMIZATION (TimeSeriesSplit CV)")
    print("=" * 70)

    best_params, study = optimize_lgb(X_train, y_train, n_trials=150, n_splits=5)

    print(f"\n  Best CV log-loss: {study.best_value:.5f}")
    print(f"  Best params:")
    for k, v in sorted(best_params.items()):
        if k in ("objective", "metric", "verbose", "random_state"):
            continue
        print(f"    {k:25s} = {v}")

    # ── Train final model with best params ─────────────────────────
    print("\n" + "=" * 70)
    print("[5] LIGHTGBM (Optuna-optimized)")
    print("=" * 70)

    lgb_model = lgb.LGBMClassifier(**best_params)
    lgb_model.fit(X_train, y_train)
    lgb_preds_raw = lgb_model.predict_proba(X_test)[:, 1]

    lgb_brier = brier_score_loss(y_test, lgb_preds_raw)
    lgb_auc = roc_auc_score(y_test, lgb_preds_raw)
    print(f"  Brier: {lgb_brier:.4f} (BSS: {1 - lgb_brier/naive_brier:.4f})")
    print(f"  AUC:   {lgb_auc:.4f}")

    # Feature importance
    imp = pd.DataFrame({"feature": feat_cols, "importance": lgb_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print("\n  Top 20 features:")
    for _, row in imp.head(20).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:6.0f}")

    # ── Walk-forward with isotonic calibration ─────────────────────
    print("\n" + "=" * 70)
    print("[6] EXPANDING WINDOW WALK-FORWARD (monthly, edge-based eval)")
    print("=" * 70)

    ds_2025_sorted = ds_2025_nopush.sort_values("game_date").reset_index(drop=True)
    ds_2025_sorted["month"] = pd.to_datetime(ds_2025_sorted["game_date"]).dt.to_period("M")
    months = sorted(ds_2025_sorted["month"].unique())
    print(f"  Months: {[str(m) for m in months]}")

    expanding_preds = []       # model P(over), calibrated
    expanding_actuals = []     # 1=over, 0=under
    expanding_mkt_over = []    # devigged market P(over)
    expanding_over_odds = []   # American over odds
    expanding_under_odds = []  # American under odds
    expanding_dates = []

    for i, test_month in enumerate(months):
        if i < 2:
            continue
        cal_mask = ds_2025_sorted["month"] < test_month
        test_mask = ds_2025_sorted["month"] == test_month
        if cal_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        cal_chunk = ds_2025_sorted[cal_mask].copy()
        test_chunk = ds_2025_sorted[test_mask].copy()
        cal_chunk, _ = engineer_features(cal_chunk)
        test_chunk, _ = engineer_features(test_chunk)

        X_c = cal_chunk[feat_cols]
        y_c = cal_chunk["over"].values
        X_t = test_chunk[feat_cols]
        y_t = test_chunk["over"].values

        # Isotonic calibration on prior months
        cal_p = lgb_model.predict_proba(X_c)[:, 1]
        iso_m = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso_m.fit(cal_p, y_c)

        test_p = lgb_model.predict_proba(X_t)[:, 1]
        test_cal = iso_m.predict(test_p)

        expanding_preds.extend(test_cal)
        expanding_actuals.extend(y_t)
        expanding_dates.extend(test_chunk["game_date"].values)

        # Market probabilities and odds for edge computation
        expanding_mkt_over.extend(test_chunk["devigged_over_prob"].values)
        expanding_over_odds.extend(test_chunk["over_close_odds"].values)
        expanding_under_odds.extend(test_chunk["under_close_odds"].values)

        month_brier = brier_score_loss(y_t, test_cal)
        month_naive = brier_score_loss(y_t, np.full(len(y_t), y_t.mean()))
        month_auc = roc_auc_score(y_t, test_cal) if len(np.unique(y_t)) > 1 else np.nan
        print(f"  {test_month}: n={len(y_t)}, Brier={month_brier:.4f} "
              f"(BSS={1-month_brier/month_naive:.4f}), AUC={month_auc:.4f}")

    model_p_over = np.array(expanding_preds)
    actuals = np.array(expanding_actuals)
    mkt_p_over = np.array(expanding_mkt_over, dtype=float)
    over_odds = np.array(expanding_over_odds, dtype=float)
    under_odds = np.array(expanding_under_odds, dtype=float)

    if len(model_p_over) > 0:
        exp_brier = brier_score_loss(actuals, model_p_over)
        exp_naive = brier_score_loss(actuals, np.full(len(actuals), actuals.mean()))
        exp_auc = roc_auc_score(actuals, model_p_over)
        print(f"\n  Overall: Brier={exp_brier:.4f} (BSS={1-exp_brier/exp_naive:.4f}), AUC={exp_auc:.4f}")

    # ── EDGE-BASED ROI Analysis ───────────────────────────────────
    print("\n" + "=" * 70)
    print("[7] EDGE-BASED ROI (model prob vs market devigged prob, actual DK odds)")
    print("=" * 70)

    print(f"  Games in test: {len(model_p_over)}")
    print(f"  Mean model P(over): {model_p_over.mean():.4f}")
    print(f"  Mean market P(over): {mkt_p_over.mean():.4f}")
    print(f"  Actual over rate: {actuals.mean():.4f}")

    edge_over = model_p_over - mkt_p_over
    edge_under = (1 - model_p_over) - (1 - mkt_p_over)
    print(f"  Mean edge (over): {edge_over.mean():+.4f}")
    print(f"  Edge std: {edge_over.std():.4f}")
    print(f"  Games with |edge| > 2%: {(np.abs(edge_over) > 0.02).sum()}")
    print(f"  Games with |edge| > 5%: {(np.abs(edge_over) > 0.05).sum()}")

    edge_thresholds = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15]

    print(f"\n  {'Edge':>6s} {'N_bets':>7s} {'N_over':>7s} {'N_under':>8s} "
          f"{'Win%':>7s} {'ROI':>8s} {'Units':>8s} {'Boot_Med':>9s} {'95% CI':>22s}")
    print("  " + "-" * 100)

    for me in edge_thresholds:
        result = compute_edge_roi(model_p_over, actuals, mkt_p_over,
                                  over_odds, under_odds, min_edge=me)
        if result["n_bets"] == 0:
            print(f"  {me:6.2f} {'0':>7s} {'—':>7s} {'—':>8s} {'—':>7s} {'—':>8s} {'—':>8s}")
            continue
        med_roi, ci_lo, ci_hi = bootstrap_edge_roi(
            model_p_over, actuals, mkt_p_over, over_odds, under_odds, me)
        print(f"  {me:6.2f} {result['n_bets']:7d} {result['n_over']:7d} {result['n_under']:8d} "
              f"{result['win_pct']:7.3f} {result['roi']:+8.3f} {result['profit_units']:+8.1f} "
              f"{med_roi:+9.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]")

    # ── Monthly breakdown at best threshold ───────────────────────
    print("\n" + "=" * 70)
    print("[8] MONTHLY BREAKDOWN (edge > 0.02)")
    print("=" * 70)

    dates_arr = np.array(expanding_dates)
    months_arr = pd.to_datetime(dates_arr).to_period("M")
    for m in sorted(months_arr.unique()):
        mask = months_arr == m
        if mask.sum() == 0:
            continue
        r = compute_edge_roi(model_p_over[mask], actuals[mask], mkt_p_over[mask],
                             over_odds[mask], under_odds[mask], min_edge=0.02)
        if r["n_bets"] > 0:
            print(f"  {m}: {r['n_bets']:4d} bets, {r['win_pct']:.3f} win%, "
                  f"ROI={r['roi']:+.3f}, PnL={r['profit_units']:+.1f}u")

    # ── Calibration: model vs market ──────────────────────────────
    print("\n" + "=" * 70)
    print("[9] CALIBRATION: model P(over) vs actual, by market-prob bucket")
    print("=" * 70)

    bins = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    print(f"  {'Mkt P(over)':>12s} {'N':>6s} {'Model_mean':>11s} {'Mkt_mean':>9s} "
          f"{'Actual':>7s} {'Model_edge':>11s}")
    for j in range(len(bins) - 1):
        mask = (mkt_p_over >= bins[j]) & (mkt_p_over < bins[j+1])
        if mask.sum() < 10:
            continue
        model_mean = model_p_over[mask].mean()
        mkt_mean = mkt_p_over[mask].mean()
        actual_mean = actuals[mask].mean()
        edge = model_mean - mkt_mean
        print(f"  [{bins[j]:.2f},{bins[j+1]:.2f}) {mask.sum():6d} {model_mean:11.4f} {mkt_mean:9.4f} "
              f"{actual_mean:7.4f} {edge:+11.4f}")

    # ── Permutation test ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[10] PERMUTATION TEST (edge > 0.02)")
    print("=" * 70)

    observed = compute_edge_roi(model_p_over, actuals, mkt_p_over,
                                over_odds, under_odds, min_edge=0.02)
    observed_roi = observed["roi"] if observed["n_bets"] > 0 else 0

    rng = np.random.default_rng(123)
    n_perm = 5000
    perm_rois = []
    for _ in range(n_perm):
        shuffled = rng.permutation(actuals)
        r = compute_edge_roi(model_p_over, shuffled, mkt_p_over,
                             over_odds, under_odds, min_edge=0.02)
        if r["n_bets"] > 0 and not np.isnan(r["roi"]):
            perm_rois.append(r["roi"])
    perm_rois = np.array(perm_rois)
    p_value = (perm_rois >= observed_roi).mean() if len(perm_rois) > 0 else 1.0
    print(f"  Observed: ROI={observed_roi:+.3f} ({observed['n_bets']} bets)")
    print(f"  Permutation mean: {perm_rois.mean():+.3f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")

    # ── Market bias ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[11] MARKET BIAS")
    print("=" * 70)
    print(f"  Actual over rate: {actuals.mean():.3f}")
    print(f"  Market implied over: {mkt_p_over.mean():.3f}")
    print(f"  Difference: {actuals.mean() - mkt_p_over.mean():+.3f}")

    # ── Blind under baseline ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("[12] BLIND UNDER vs MODEL-SELECTED UNDER")
    print("=" * 70)

    under_dec_all = american_to_decimal(under_odds)
    under_dec_all = np.where(np.isfinite(under_dec_all), under_dec_all, 1.0 + 100.0/110.0)
    blind_won = actuals == 0
    blind_pnl = np.sum(np.where(blind_won, under_dec_all - 1.0, -1.0))
    blind_roi = blind_pnl / len(actuals)
    print(f"  Blind under (all {len(actuals)} games): win%={blind_won.mean():.3f}, "
          f"ROI={blind_roi:+.3f}")

    if observed["n_bets"] > 0:
        print(f"  Model under (edge>2%, {observed['n_under']} bets): "
              f"win%={observed['win_pct']:.3f}, ROI={observed['roi']:+.3f}")
        print(f"  Model lift: {observed['roi'] - blind_roi:+.3f} ROI")

    # ── Save predictions ──────────────────────────────────────────
    out_df = pd.DataFrame({
        "game_date": expanding_dates,
        "model_p_over": model_p_over,
        "mkt_p_over": mkt_p_over,
        "over_close_odds": over_odds,
        "under_close_odds": under_odds,
        "actual_over": actuals,
        "edge_over": model_p_over - mkt_p_over,
        "edge_under": (1 - model_p_over) - (1 - mkt_p_over),
    })
    out_path = ROOT / "data" / "backtest" / "ou_edge_predictions_2025.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"\n  Predictions saved to {out_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
