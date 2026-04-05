"""
O/U vs Line Model: Predict OVER/UNDER outcome directly against the DK closing line.

Instead of predicting total runs and comparing, this model takes the DK line AS A FEATURE
alongside pregame features. The target is binary: 1 = over, 0 = under.
Pushes (total_runs == line) are excluded.

Training strategy: Train on 2024 full season, validate walk-forward on 2025.
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────
# 1. Parse 2024 DK lines from JSON
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
            if gv.get("gameType") not in (None, "R", "Regular Season"):
                # Try to only include regular season
                pass
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

            home_team = gv["homeTeam"]["shortName"]
            away_team = gv["awayTeam"]["shortName"]
            home_score = gv.get("homeTeamScore")
            away_score = gv.get("awayTeamScore")

            rows.append({
                "game_date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "ou_open": ou_open,
                "ou_close": ou_close,
                "over_close_odds": current.get("overOdds"),
                "under_close_odds": current.get("underOdds"),
            })

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    # Fix team name mappings (JSON might differ from our standard)
    team_map = {"OAK": "ATH", "ARI": "AZ", "WSN": "WSH"}
    df["home_team"] = df["home_team"].replace(team_map)
    df["away_team"] = df["away_team"].replace(team_map)
    return df


# ──────────────────────────────────────────────────────────────────────
# 2. Merge features + odds + results
# ──────────────────────────────────────────────────────────────────────
def build_dataset(year: int, odds_df: pd.DataFrame = None) -> pd.DataFrame:
    """Build a single year's dataset by merging features, odds, and results."""
    feat_path = ROOT / "data" / "features" / f"game_features_{year}.parquet"
    features = pd.read_parquet(feat_path)
    features["game_date"] = pd.to_datetime(features["game_date"]).dt.strftime("%Y-%m-%d")

    if odds_df is None:
        # Use parquet for 2025
        odds_path = ROOT / "data" / "odds" / "sbr_mlb_2025.parquet"
        odds_df = pd.read_parquet(odds_path)
        odds_df["game_date"] = pd.to_datetime(odds_df["game_date"]).dt.strftime("%Y-%m-%d")

    # Merge on date + teams
    merged = features.merge(
        odds_df[["game_date", "home_team", "away_team", "ou_close", "ou_open",
                 "over_close_odds", "under_close_odds"]].drop_duplicates(),
        on=["game_date", "home_team", "away_team"],
        how="inner",
    )

    # Compute total runs & target
    merged["total_runs"] = merged["home_score"].astype(float) + merged["away_score"].astype(float)
    merged["is_push"] = merged["total_runs"] == merged["ou_close"]
    merged["over"] = (merged["total_runs"] > merged["ou_close"]).astype(int)

    print(f"  {year}: {len(features)} features, {len(odds_df)} odds rows -> {len(merged)} merged, "
          f"{merged['is_push'].sum()} pushes")

    return merged


# ──────────────────────────────────────────────────────────────────────
# 3. Feature engineering
# ──────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add derived features and return (df, feature_columns)."""
    df = df.copy()

    # Core derived features
    df["dk_ou_line"] = df["ou_close"]
    df["line_movement"] = df["ou_close"] - df["ou_open"]

    # Implied over probability from closing odds (American -> implied prob)
    def american_to_prob(odds):
        odds = pd.to_numeric(odds, errors="coerce")
        prob = np.where(odds < 0, -odds / (-odds + 100), 100 / (odds + 100))
        return prob

    df["implied_over_prob"] = american_to_prob(df["over_close_odds"])
    df["implied_under_prob"] = american_to_prob(df["under_close_odds"])
    df["implied_vig"] = df["implied_over_prob"] + df["implied_under_prob"] - 1
    # De-vigged over prob
    df["devigged_over_prob"] = df["implied_over_prob"] / (df["implied_over_prob"] + df["implied_under_prob"])

    # SP quality interactions with line
    if "home_sp_xrv_mean" in df.columns and "away_sp_xrv_mean" in df.columns:
        df["sp_quality_sum"] = df["home_sp_xrv_mean"] + df["away_sp_xrv_mean"]
        df["sp_quality_diff"] = df["away_sp_xrv_mean"] - df["home_sp_xrv_mean"]

    # Lineup power sum
    if "home_hit_barrel_rate" in df.columns:
        df["lineup_power_sum"] = (
            df["home_hit_barrel_rate"].fillna(0) + df["away_hit_barrel_rate"].fillna(0)
        )

    # Bullpen quality sum
    if "home_bp_xrv_mean" in df.columns:
        df["bp_quality_sum"] = df["home_bp_xrv_mean"].fillna(0) + df["away_bp_xrv_mean"].fillna(0)
        df["bp_fatigue_sum"] = (
            df["home_bp_fatigue_score"].fillna(0) + df["away_bp_fatigue_score"].fillna(0)
        )

    # Line vs SP expectation: higher = market expects more runs than SPs suggest
    if "home_sp_projected_era" in df.columns and "away_sp_projected_era" in df.columns:
        avg_era = (df["home_sp_projected_era"].fillna(4.5) + df["away_sp_projected_era"].fillna(4.5)) / 2
        df["line_vs_sp_era"] = df["dk_ou_line"] - avg_era * 2  # rough: 2*ERA ~ total runs

    # Select feature columns (exclude target / identifiers / raw scores)
    exclude = {
        "game_pk", "game_date", "home_team", "away_team", "home_win",
        "home_score", "away_score", "total_runs", "is_push", "over",
        "ou_close", "ou_open", "over_close_odds", "under_close_odds",
        "ou_result", "status", "book_totals", "book_ml",
        "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close",
        "is_home",  # always 1
    }
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64", "float32", "int32", "bool")]

    return df, feature_cols


# ──────────────────────────────────────────────────────────────────────
# 4. Evaluation helpers
# ──────────────────────────────────────────────────────────────────────
def _american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    elif odds > 0:
        return 1.0 + odds / 100.0
    return np.nan


def compute_roi(preds: np.ndarray, actuals: np.ndarray, threshold: float,
                juice: float = -110,
                over_odds: np.ndarray | None = None,
                under_odds: np.ndarray | None = None) -> dict:
    """Compute ROI at a given confidence threshold for over AND under bets.

    If per-game ``over_odds`` / ``under_odds`` arrays (American format) are
    provided, payouts are computed from the actual DK closing odds for each
    bet.  Otherwise falls back to the flat ``juice`` assumption.
    """
    # Over bets: predict > threshold
    over_mask = preds > threshold
    # Under bets: predict < (1 - threshold)
    under_mask = preds < (1 - threshold)

    n_over = int(over_mask.sum())
    n_under = int(under_mask.sum())
    n_total = n_over + n_under

    if n_total == 0:
        return {"threshold": threshold, "n_bets": 0, "roi": np.nan, "win_pct": np.nan}

    # Compute PnL per bet using actual odds when available
    total_pnl = 0.0
    total_wins = 0

    if n_over > 0:
        over_won = actuals[over_mask] == 1
        total_wins += int(over_won.sum())
        if over_odds is not None:
            over_dec = np.array([_american_to_decimal(o) for o in over_odds[over_mask]])
            # NaN odds → fallback to flat juice
            fallback_dec = 1.0 + 100.0 / abs(juice) if juice < 0 else 1.0 + juice / 100.0
            over_dec = np.where(np.isnan(over_dec), fallback_dec, over_dec)
            total_pnl += np.sum(np.where(over_won, over_dec - 1.0, -1.0))
        else:
            payout = 100.0 / abs(juice) if juice < 0 else juice / 100.0
            total_pnl += int(over_won.sum()) * payout - int((~over_won).sum()) * 1.0

    if n_under > 0:
        under_won = actuals[under_mask] == 0
        total_wins += int(under_won.sum())
        if under_odds is not None:
            under_dec = np.array([_american_to_decimal(o) for o in under_odds[under_mask]])
            fallback_dec = 1.0 + 100.0 / abs(juice) if juice < 0 else 1.0 + juice / 100.0
            under_dec = np.where(np.isnan(under_dec), fallback_dec, under_dec)
            total_pnl += np.sum(np.where(under_won, under_dec - 1.0, -1.0))
        else:
            payout = 100.0 / abs(juice) if juice < 0 else juice / 100.0
            total_pnl += int(under_won.sum()) * payout - int((~under_won).sum()) * 1.0

    win_pct = total_wins / n_total
    roi = total_pnl / n_total

    return {
        "threshold": threshold,
        "n_bets": n_total,
        "n_over": n_over,
        "n_under": n_under,
        "wins": total_wins,
        "win_pct": win_pct,
        "profit_units": total_pnl,
        "roi": roi,
    }


def bootstrap_roi(preds, actuals, threshold, n_boot=2000, juice=-110,
                   over_odds=None, under_odds=None):
    """Bootstrap CI for ROI at a threshold."""
    rng = np.random.default_rng(42)
    rois = []
    n = len(preds)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        oo = over_odds[idx] if over_odds is not None else None
        uo = under_odds[idx] if under_odds is not None else None
        result = compute_roi(preds[idx], actuals[idx], threshold, juice,
                             over_odds=oo, under_odds=uo)
        if not np.isnan(result["roi"]):
            rois.append(result["roi"])
    if not rois:
        return np.nan, np.nan, np.nan
    rois = np.array(rois)
    return np.median(rois), np.percentile(rois, 2.5), np.percentile(rois, 97.5)


# ──────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("O/U vs Line Model: Direct OVER/UNDER prediction against DK line")
    print("=" * 70)

    # ── Load data ───────────────────────────────────────────────────
    print("\n[1] Loading data...")
    dk_2024 = load_dk_lines_from_json(2024)
    print(f"  2024 DK lines from JSON: {len(dk_2024)} games")

    # Also load 2023
    dk_2023 = load_dk_lines_from_json(2023)
    print(f"  2023 DK lines from JSON: {len(dk_2023)} games")

    ds_2024 = build_dataset(2024, dk_2024)
    ds_2025 = build_dataset(2025)

    # Also try 2023
    try:
        ds_2023 = build_dataset(2023, dk_2023)
        has_2023 = len(ds_2023) > 100
    except Exception as e:
        print(f"  2023 features not available: {e}")
        has_2023 = False
        ds_2023 = pd.DataFrame()

    # ── Build combined training set ────────────────────────────────
    print("\n[2] Building datasets...")
    if has_2023:
        train_full = pd.concat([ds_2023, ds_2024], ignore_index=True)
        print(f"  Training data: 2023-2024 ({len(train_full)} games)")
    else:
        train_full = ds_2024
        print(f"  Training data: 2024 only ({len(train_full)} games)")

    # Exclude pushes
    train_full = train_full[~train_full["is_push"]].copy()
    ds_2025_nopush = ds_2025[~ds_2025["is_push"]].copy()
    print(f"  After excluding pushes: train={len(train_full)}, test_2025={len(ds_2025_nopush)}")

    # Engineer features
    train_full, feat_cols_train = engineer_features(train_full)
    ds_2025_nopush, feat_cols_test = engineer_features(ds_2025_nopush)

    # Use intersection of feature columns
    feat_cols = sorted(set(feat_cols_train) & set(feat_cols_test))
    print(f"  Feature columns: {len(feat_cols)}")

    # ── Check base rate ────────────────────────────────────────────
    print("\n[3] Base rates...")
    train_over_rate = train_full["over"].mean()
    test_over_rate = ds_2025_nopush["over"].mean()
    print(f"  Train over rate: {train_over_rate:.3f}")
    print(f"  Test 2025 over rate: {test_over_rate:.3f}")

    # ── Prepare arrays ──────────────────────────────────────────────
    X_train = train_full[feat_cols].copy()
    y_train = train_full["over"].values
    X_test = ds_2025_nopush[feat_cols].copy()
    y_test = ds_2025_nopush["over"].values
    test_dates = ds_2025_nopush["game_date"].values

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  NaN rates in train: {X_train.isna().mean().mean():.3f}")
    print(f"  NaN rates in test: {X_test.isna().mean().mean():.3f}")

    # ── Model A: Logistic Regression baseline ──────────────────────
    print("\n" + "=" * 70)
    print("[4] LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    # Fill NaN: use median, then 0 for any remaining (all-NaN columns)
    train_medians = X_train.median()
    X_train_filled = X_train.fillna(train_medians).fillna(0)
    X_test_filled = X_test.fillna(train_medians).fillna(0)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_filled),
        columns=feat_cols,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_filled),
        columns=feat_cols,
    )

    lr = LogisticRegression(C=0.01, max_iter=2000, solver="saga", penalty="l1")
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict_proba(X_test_scaled)[:, 1]

    lr_brier = brier_score_loss(y_test, lr_preds)
    lr_logloss = log_loss(y_test, lr_preds)
    lr_auc = roc_auc_score(y_test, lr_preds)
    naive_brier = brier_score_loss(y_test, np.full(len(y_test), y_test.mean()))

    print(f"  Brier score:  {lr_brier:.4f}  (naive: {naive_brier:.4f}, BSS: {1 - lr_brier/naive_brier:.4f})")
    print(f"  Log loss:     {lr_logloss:.4f}")
    print(f"  AUC:          {lr_auc:.4f}")

    # Top LR coefficients
    coef_df = pd.DataFrame({"feature": feat_cols, "coef": lr.coef_[0]})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    nonzero = coef_df[coef_df["coef"] != 0]
    print(f"\n  Non-zero LR coefficients: {len(nonzero)} / {len(feat_cols)}")
    print("  Top 15:")
    for _, row in nonzero.head(15).iterrows():
        print(f"    {row['feature']:40s} {row['coef']:+.4f}")

    # ── Model B: LightGBM ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[5] LIGHTGBM CLASSIFIER")
    print("=" * 70)

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 80,
        "max_depth": 3,
        "num_leaves": 8,
        "learning_rate": 0.05,
        "min_child_samples": 50,
        "reg_alpha": 2.0,
        "reg_lambda": 2.0,
        "colsample_bytree": 0.5,
        "subsample": 0.8,
        "subsample_freq": 1,
        "verbose": -1,
        "random_state": 42,
    }

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)
    lgb_preds_raw = lgb_model.predict_proba(X_test)[:, 1]

    lgb_brier = brier_score_loss(y_test, lgb_preds_raw)
    lgb_logloss = log_loss(y_test, lgb_preds_raw)
    lgb_auc = roc_auc_score(y_test, lgb_preds_raw)

    print(f"  Brier score:  {lgb_brier:.4f}  (BSS: {1 - lgb_brier/naive_brier:.4f})")
    print(f"  Log loss:     {lgb_logloss:.4f}")
    print(f"  AUC:          {lgb_auc:.4f}")

    # Feature importance
    imp = pd.DataFrame({"feature": feat_cols, "importance": lgb_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print("\n  Top 20 LightGBM features:")
    for _, row in imp.head(20).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:6.0f}")

    # ── Model C: LightGBM with isotonic calibration ────────────────
    print("\n" + "=" * 70)
    print("[6] LIGHTGBM + ISOTONIC CALIBRATION (walk-forward within 2025)")
    print("=" * 70)

    # Walk-forward: train on 2024 (+ optional 2023), calibrate on first 40% of 2025, test on rest
    ds_2025_sorted = ds_2025_nopush.sort_values("game_date").reset_index(drop=True)
    n_2025 = len(ds_2025_sorted)
    cal_end = int(n_2025 * 0.4)

    cal_set = ds_2025_sorted.iloc[:cal_end]
    test_set = ds_2025_sorted.iloc[cal_end:]
    print(f"  Calibration set: first 40% of 2025 = {len(cal_set)} games")
    print(f"  Test set: last 60% of 2025 = {len(test_set)} games")
    print(f"  Cal date range: {cal_set['game_date'].min()} to {cal_set['game_date'].max()}")
    print(f"  Test date range: {test_set['game_date'].min()} to {test_set['game_date'].max()}")

    cal_set, _ = engineer_features(cal_set)
    test_set, _ = engineer_features(test_set)

    X_cal = cal_set[feat_cols]
    y_cal = cal_set["over"].values
    X_test_wf = test_set[feat_cols]
    y_test_wf = test_set["over"].values

    # Calibrate using isotonic regression on calibration set predictions
    lgb_cal_preds = lgb_model.predict_proba(X_cal)[:, 1]

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(lgb_cal_preds, y_cal)

    lgb_test_raw = lgb_model.predict_proba(X_test_wf)[:, 1]
    lgb_test_cal = iso.predict(lgb_test_raw)

    wf_brier_raw = brier_score_loss(y_test_wf, lgb_test_raw)
    wf_brier_cal = brier_score_loss(y_test_wf, lgb_test_cal)
    wf_naive = brier_score_loss(y_test_wf, np.full(len(y_test_wf), y_test_wf.mean()))
    wf_auc_raw = roc_auc_score(y_test_wf, lgb_test_raw)
    wf_auc_cal = roc_auc_score(y_test_wf, lgb_test_cal)

    print(f"\n  Walk-forward results (last 60% of 2025):")
    print(f"  Naive Brier:       {wf_naive:.4f}")
    print(f"  LGB raw Brier:     {wf_brier_raw:.4f}  (BSS: {1 - wf_brier_raw/wf_naive:.4f})")
    print(f"  LGB cal Brier:     {wf_brier_cal:.4f}  (BSS: {1 - wf_brier_cal/wf_naive:.4f})")
    print(f"  LGB raw AUC:       {wf_auc_raw:.4f}")
    print(f"  LGB cal AUC:       {wf_auc_cal:.4f}")

    # ── Expanding window within 2025 ──────────────────────────────
    print("\n" + "=" * 70)
    print("[7] EXPANDING WINDOW WALK-FORWARD (monthly chunks in 2025)")
    print("=" * 70)

    ds_2025_sorted["month"] = pd.to_datetime(ds_2025_sorted["game_date"]).dt.to_period("M")
    months = sorted(ds_2025_sorted["month"].unique())
    print(f"  Months in 2025: {[str(m) for m in months]}")

    expanding_preds = []
    expanding_actuals = []
    expanding_dates = []
    expanding_over_odds = []
    expanding_under_odds = []

    for i, test_month in enumerate(months):
        if i < 2:  # need at least 2 months of 2025 data for calibration
            continue
        # Train: 2024 (already trained LGB) + calibrate on 2025 data up to this month
        cal_mask = ds_2025_sorted["month"] < test_month
        test_mask = ds_2025_sorted["month"] == test_month

        if cal_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        cal_chunk = ds_2025_sorted[cal_mask]
        test_chunk = ds_2025_sorted[test_mask]
        cal_chunk, _ = engineer_features(cal_chunk)
        test_chunk, _ = engineer_features(test_chunk)

        X_c = cal_chunk[feat_cols]
        y_c = cal_chunk["over"].values
        X_t = test_chunk[feat_cols]
        y_t = test_chunk["over"].values

        # Re-calibrate
        cal_p = lgb_model.predict_proba(X_c)[:, 1]
        iso_m = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso_m.fit(cal_p, y_c)

        test_p = lgb_model.predict_proba(X_t)[:, 1]
        test_cal = iso_m.predict(test_p)

        expanding_preds.extend(test_cal)
        expanding_actuals.extend(y_t)
        expanding_dates.extend(test_chunk["game_date"].values)
        # Collect actual DK closing odds for this chunk
        if "over_close_odds" in test_chunk.columns:
            expanding_over_odds.extend(test_chunk["over_close_odds"].values)
            expanding_under_odds.extend(test_chunk["under_close_odds"].values)
        else:
            expanding_over_odds.extend([np.nan] * len(y_t))
            expanding_under_odds.extend([np.nan] * len(y_t))

        month_brier = brier_score_loss(y_t, test_cal)
        month_naive = brier_score_loss(y_t, np.full(len(y_t), y_t.mean()))
        month_auc = roc_auc_score(y_t, test_cal) if len(np.unique(y_t)) > 1 else np.nan
        print(f"  {test_month}: n={len(y_t)}, Brier={month_brier:.4f} "
              f"(naive={month_naive:.4f}, BSS={1-month_brier/month_naive:.4f}), AUC={month_auc:.4f}")

    expanding_preds = np.array(expanding_preds)
    expanding_actuals = np.array(expanding_actuals)

    if len(expanding_preds) > 0:
        exp_brier = brier_score_loss(expanding_actuals, expanding_preds)
        exp_naive = brier_score_loss(expanding_actuals, np.full(len(expanding_actuals), expanding_actuals.mean()))
        exp_auc = roc_auc_score(expanding_actuals, expanding_preds)
        print(f"\n  Overall expanding window:")
        print(f"  Brier: {exp_brier:.4f} (naive: {exp_naive:.4f}, BSS: {1-exp_brier/exp_naive:.4f})")
        print(f"  AUC:   {exp_auc:.4f}")

    # ── ROI Analysis ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[8] ROI ANALYSIS (on walk-forward test set)")
    print("=" * 70)

    # Use the expanding window predictions (most realistic)
    if len(expanding_preds) == 0:
        print("  No expanding window predictions available, using full test set")
        roi_preds = lgb_preds_raw
        roi_actuals = y_test
        roi_over_odds = None
        roi_under_odds = None
    else:
        roi_preds = expanding_preds
        roi_actuals = expanding_actuals
        roi_over_odds = np.array(expanding_over_odds, dtype=float)
        roi_under_odds = np.array(expanding_under_odds, dtype=float)

    has_real_odds = roi_over_odds is not None and np.isfinite(roi_over_odds).sum() > 0
    if has_real_odds:
        n_real = np.isfinite(roi_under_odds).sum()
        print(f"  Using actual DK closing odds for {n_real}/{len(roi_preds)} games (fallback -110 for rest)")
    else:
        print("  No actual DK odds available — using flat -110 assumption")

    breakeven = 110 / 210  # ~0.5238 at -110
    print(f"  Break-even win rate at -110: {breakeven:.4f}")

    thresholds = [0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.62, 0.65]

    # Show both actual-odds and flat-110 for comparison
    for label, oo, uo in [("Actual DK odds", roi_over_odds, roi_under_odds),
                          ("Flat -110", None, None)]:
        if label == "Actual DK odds" and not has_real_odds:
            continue
        print(f"\n  --- {label} ---")
        print(f"  {'Thresh':>7s} {'N_bets':>7s} {'N_over':>7s} {'N_under':>7s} "
              f"{'Win%':>7s} {'ROI':>8s} {'Units':>8s} {'Boot_Med':>9s} {'95% CI':>20s}")
        print("  " + "-" * 95)

        for t in thresholds:
            result = compute_roi(roi_preds, roi_actuals, t,
                                 over_odds=oo, under_odds=uo)
            if result["n_bets"] == 0:
                continue
            med_roi, ci_lo, ci_hi = bootstrap_roi(roi_preds, roi_actuals, t,
                                                   over_odds=oo, under_odds=uo)
            print(f"  {t:7.2f} {result['n_bets']:7d} {result['n_over']:7d} {result['n_under']:7d} "
                  f"{result['win_pct']:7.3f} {result['roi']:+8.3f} {result['profit_units']:+8.1f} "
                  f"{med_roi:+9.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]")

    # ── Calibration check ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[9] CALIBRATION CHECK")
    print("=" * 70)

    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for label, preds_arr, actuals_arr in [
        ("LGB raw (full 2025)", lgb_preds_raw, y_test),
        ("Expanding window cal", roi_preds, roi_actuals),
    ]:
        print(f"\n  {label}:")
        print(f"  {'Bin':>12s} {'N':>6s} {'Pred_mean':>10s} {'Actual':>8s} {'Gap':>8s}")
        digitized = np.digitize(preds_arr, bins) - 1
        for b in range(len(bin_centers)):
            mask = digitized == b
            if mask.sum() == 0:
                continue
            pred_mean = preds_arr[mask].mean()
            actual_mean = actuals_arr[mask].mean()
            gap = actual_mean - pred_mean
            print(f"  {bin_centers[b]:10.2f}   {mask.sum():5d} {pred_mean:10.4f} {actual_mean:8.4f} {gap:+8.4f}")

    # ── Line-only model (sanity check) ─────────────────────────────
    print("\n" + "=" * 70)
    print("[10] ABLATION: LINE-ONLY vs FULL MODEL")
    print("=" * 70)

    # Line-only features
    line_feats = ["dk_ou_line", "line_movement", "devigged_over_prob", "implied_vig"]
    available_line_feats = [f for f in line_feats if f in feat_cols]

    lgb_line = lgb.LGBMClassifier(**lgb_params)
    lgb_line.fit(X_train[available_line_feats], y_train)
    line_preds = lgb_line.predict_proba(X_test[available_line_feats])[:, 1]
    line_brier = brier_score_loss(y_test, line_preds)
    line_auc = roc_auc_score(y_test, line_preds)

    # No-line features
    no_line_feats = [f for f in feat_cols if f not in line_feats]
    lgb_noline = lgb.LGBMClassifier(**lgb_params)
    lgb_noline.fit(X_train[no_line_feats], y_train)
    noline_preds = lgb_noline.predict_proba(X_test[no_line_feats])[:, 1]
    noline_brier = brier_score_loss(y_test, noline_preds)
    noline_auc = roc_auc_score(y_test, noline_preds)

    print(f"  {'Model':>25s} {'Brier':>8s} {'BSS':>8s} {'AUC':>8s}")
    print(f"  {'Naive':>25s} {naive_brier:8.4f} {'0.000':>8s} {'0.500':>8s}")
    print(f"  {'Line-only (4 feats)':>25s} {line_brier:8.4f} {1-line_brier/naive_brier:8.4f} {line_auc:8.4f}")
    print(f"  {'No-line (pregame only)':>25s} {noline_brier:8.4f} {1-noline_brier/naive_brier:8.4f} {noline_auc:8.4f}")
    print(f"  {'Full model':>25s} {lgb_brier:8.4f} {1-lgb_brier/naive_brier:8.4f} {lgb_auc:8.4f}")
    print(f"  {'Logistic Regression':>25s} {lr_brier:8.4f} {1-lr_brier/naive_brier:8.4f} {lr_auc:8.4f}")

    # ── Over/Under bias analysis ───────────────────────────────────
    print("\n" + "=" * 70)
    print("[11] MARKET BIAS: Does the market systematically over/under-price?")
    print("=" * 70)

    for label, df_check in [("Train (2024)", train_full), ("Test (2025)", ds_2025_nopush)]:
        over_rate = df_check["over"].mean()
        devigged = df_check.get("devigged_over_prob")
        if devigged is None:
            df_check, _ = engineer_features(df_check)
            devigged = df_check["devigged_over_prob"]
        market_over = devigged.mean()
        print(f"  {label}: actual_over={over_rate:.3f}, market_implied_over={market_over:.3f}, "
              f"diff={over_rate - market_over:+.3f}")

    # ── By line bucket ─────────────────────────────────────────────
    print("\n  Over rate by O/U line bucket (2025 test):")
    ds_2025_nopush_eng, _ = engineer_features(ds_2025_nopush)
    ds_2025_nopush_eng["line_bucket"] = pd.cut(ds_2025_nopush_eng["dk_ou_line"],
                                                 bins=[6, 7, 7.5, 8, 8.5, 9, 9.5, 10, 13])
    bucket_stats = ds_2025_nopush_eng.groupby("line_bucket", observed=True).agg(
        n=("over", "size"),
        over_rate=("over", "mean"),
        mean_devigged=("devigged_over_prob", "mean"),
    )
    print(f"  {'Line bucket':>15s} {'N':>6s} {'Over%':>8s} {'Mkt_impl':>9s} {'Edge':>8s}")
    for idx, row in bucket_stats.iterrows():
        edge = row["over_rate"] - row["mean_devigged"]
        print(f"  {str(idx):>15s} {row['n']:6.0f} {row['over_rate']:8.3f} {row['mean_devigged']:9.3f} {edge:+8.3f}")

    # ── Permutation test for ROI ──────────────────────────────────
    print("\n" + "=" * 70)
    print("[12] PERMUTATION TEST: Is model ROI statistically significant?")
    print("=" * 70)

    best_thresh = 0.55  # use a fixed threshold for the test
    observed = compute_roi(roi_preds, roi_actuals, best_thresh)
    observed_roi = observed["roi"] if not np.isnan(observed["roi"]) else 0

    rng = np.random.default_rng(123)
    n_perm = 5000
    perm_rois = []
    for _ in range(n_perm):
        shuffled = rng.permutation(roi_actuals)
        r = compute_roi(roi_preds, shuffled, best_thresh)
        if not np.isnan(r["roi"]):
            perm_rois.append(r["roi"])
    perm_rois = np.array(perm_rois)
    p_value = (perm_rois >= observed_roi).mean()
    print(f"  Threshold: {best_thresh}")
    print(f"  Observed ROI: {observed_roi:+.3f} ({observed['n_bets']} bets)")
    print(f"  Permutation mean ROI: {perm_rois.mean():+.3f}")
    print(f"  P-value (one-sided): {p_value:.4f}")
    print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")

    # ── Blind under baseline comparison ───────────────────────────
    print("\n" + "=" * 70)
    print("[13] BLIND UNDER BASELINE COMPARISON")
    print("=" * 70)
    blind_under_wins = (roi_actuals == 0).sum()
    blind_n = len(roi_actuals)
    blind_win_pct = blind_under_wins / blind_n
    blind_profit = blind_under_wins * (100 / 110) - (blind_n - blind_under_wins)
    blind_roi = blind_profit / blind_n
    print(f"  Blind under (all {blind_n} games): win%={blind_win_pct:.3f}, ROI={blind_roi:+.3f}")
    print(f"  Model selected under ({observed['n_under']} bets): win%={observed['win_pct']:.3f}, ROI={observed_roi:+.3f}")
    print(f"  Model lift vs blind: {observed['win_pct'] - blind_win_pct:+.3f} win%, {observed_roi - blind_roi:+.3f} ROI")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
