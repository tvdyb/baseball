#!/usr/bin/env python3
"""
Unified Daily Prediction & Betting Strategy
============================================
Combines all profitable edges into one walk-forward backtest:

  1. DK-vs-Kalshi ML Arbitrage  (7% edge threshold)
  2. O/U Direct Classifier Unders  (P(under) > 0.55)
  3. DK Away Underdog Anomaly  (blind: away dog DK implied 35-40%)
  4. Win Model Pick-em Home Bets  (DK 48-52%, model home edge > 3%)
  5. NRFI Bets  (P(NRFI) >= 0.57 at -120 odds)
  6. YRFI Bets  (P(NRFI) <= 0.48 at +100 odds)

Hard filters:
  - No division games on ML
  - No away favorites
  - Cap model-DK disagreement at 10%
  - Reduce September allocations

Usage:
    python src/unified_strategy.py
"""

import json
import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

# ═══════════════════════════════════════════════════════════════════════
# Division data
# ═══════════════════════════════════════════════════════════════════════
DIVISIONS = {
    "NYY": "AL East", "BOS": "AL East", "TOR": "AL East",
    "TB":  "AL East", "BAL": "AL East",
    "CWS": "AL Central", "CLE": "AL Central", "DET": "AL Central",
    "KC":  "AL Central", "MIN": "AL Central",
    "HOU": "AL West", "LAA": "AL West", "OAK": "AL West",
    "SEA": "AL West", "TEX": "AL West", "ATH": "AL West",
    "ATL": "NL East", "MIA": "NL East", "NYM": "NL East",
    "PHI": "NL East", "WSH": "NL East",
    "CHC": "NL Central", "CIN": "NL Central", "MIL": "NL Central",
    "PIT": "NL Central", "STL": "NL Central",
    "ARI": "NL West", "AZ": "NL West", "COL": "NL West",
    "LAD": "NL West", "SD":  "NL West", "SF":  "NL West",
}


# ═══════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════
def american_to_prob(odds):
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


def remove_vig(p1, p2):
    total = p1 + p2
    if total == 0 or pd.isna(total):
        return np.nan, np.nan
    return p1 / total, p2 / total


def decimal_from_american(odds):
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / (-odds)


def kelly_fraction(p, b):
    """Kelly fraction: f = (p*b - q) / b where q = 1-p."""
    if b <= 0 or p <= 0:
        return 0.0
    q = 1.0 - p
    f = (p * b - q) / b
    return max(0.0, f)


# ═══════════════════════════════════════════════════════════════════════
# 1. Data Loading
# ═══════════════════════════════════════════════════════════════════════
def load_all_data():
    """Load and merge all data sources needed for the four strategies."""
    print("Loading data sources...")

    # -- DK odds (ML + O/U) --
    odds = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    odds["game_date"] = pd.to_datetime(odds["game_date"]).dt.strftime("%Y-%m-%d")
    odds = odds[odds["status"].str.startswith("Final", na=False)].copy()
    odds = odds.dropna(subset=["home_ml_close", "away_ml_close"]).copy()

    # Vig-free ML probs
    odds["dk_home_raw"] = odds["home_ml_close"].apply(american_to_prob)
    odds["dk_away_raw"] = odds["away_ml_close"].apply(american_to_prob)
    pairs = odds.apply(lambda r: pd.Series(remove_vig(r["dk_home_raw"], r["dk_away_raw"])), axis=1)
    odds["dk_home_prob"] = pairs[0]
    odds["dk_away_prob"] = pairs[1]
    odds["home_ml_dec"] = odds["home_ml_close"].apply(decimal_from_american)
    odds["away_ml_dec"] = odds["away_ml_close"].apply(decimal_from_american)
    odds["home_win"] = (odds["home_score"] > odds["away_score"]).astype(int)
    odds["total_runs"] = odds["home_score"].astype(float) + odds["away_score"].astype(float)

    # Filter out extreme lines (in-game artefacts)
    extreme = (odds["home_ml_close"].abs() > 500) | (odds["away_ml_close"].abs() > 500)
    odds = odds[~extreme].copy()

    # Division info
    odds["home_div"] = odds["home_team"].map(DIVISIONS)
    odds["away_div"] = odds["away_team"].map(DIVISIONS)
    odds["is_division"] = (odds["home_div"] == odds["away_div"]).astype(int)

    # Month / September flag
    odds["month"] = pd.to_datetime(odds["game_date"]).dt.month
    odds["is_september"] = (odds["month"] == 9).astype(int)

    # O/U implied probs
    odds["implied_over"] = odds["over_close_odds"].apply(american_to_prob)
    odds["implied_under"] = odds["under_close_odds"].apply(american_to_prob)
    total_imp = odds["implied_over"] + odds["implied_under"]
    odds["devigged_over"] = odds["implied_over"] / total_imp
    odds["devigged_under"] = odds["implied_under"] / total_imp
    odds["under_ml_dec"] = odds["under_close_odds"].apply(decimal_from_american)

    # Dedup
    keys = ["game_date", "home_team", "away_team"]
    odds = odds.drop_duplicates(subset=keys, keep="first")
    print(f"  DK odds: {len(odds)} completed games")

    # -- Kalshi prices --
    kal = pd.read_parquet(DATA / "kalshi" / "kalshi_mlb_2025.parquet")
    kal["game_date"] = kal["game_date"].astype(str)
    kal = kal.sort_values("volume", ascending=False).drop_duplicates(subset=keys, keep="first")
    print(f"  Kalshi: {len(kal)} games")

    # -- Win model walk-forward predictions --
    wf = pd.read_csv(DATA / "audit" / "walk_forward_predictions.csv")
    wf["game_date"] = pd.to_datetime(wf["game_date"]).dt.strftime("%Y-%m-%d")
    wf = wf[wf["season"] == 2025].copy()
    wf["model_prob"] = wf["ens_calibrated"].fillna(wf["ens_prob"])
    wf = wf.drop_duplicates(subset=keys, keep="first")
    print(f"  Win model predictions (2025): {len(wf)}")

    # -- Merge everything onto odds base --
    df = odds.copy()

    # Merge Kalshi
    df = df.merge(
        kal[["game_date", "home_team", "away_team",
             "kalshi_home_prob", "kalshi_away_prob"]],
        on=keys, how="left",
    )
    print(f"  Games with Kalshi prices: {df['kalshi_home_prob'].notna().sum()}")

    # Merge win model
    df = df.merge(
        wf[["game_date", "home_team", "away_team", "model_prob"]],
        on=keys, how="left",
    )
    print(f"  Games with model predictions: {df['model_prob'].notna().sum()}")

    # -- NRFI/YRFI model predictions --
    nrfi_path = DATA / "backtest" / "nrfi_lgb_2025.parquet"
    if nrfi_path.exists():
        nrfi = pd.read_parquet(nrfi_path)
        nrfi["game_date"] = pd.to_datetime(nrfi["game_date"]).dt.strftime("%Y-%m-%d")
        nrfi = nrfi.drop_duplicates(subset=keys, keep="first")
        df = df.merge(
            nrfi[["game_date", "home_team", "away_team", "nrfi_lgb_prob", "actual_nrfi"]],
            on=keys, how="left",
        )
        print(f"  Games with NRFI predictions: {df['nrfi_lgb_prob'].notna().sum()}")
    else:
        print(f"  WARNING: NRFI predictions not found at {nrfi_path}")
        df["nrfi_lgb_prob"] = np.nan
        df["actual_nrfi"] = np.nan

    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Total merged games: {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. O/U Model (train on 2024, predict 2025)
# ═══════════════════════════════════════════════════════════════════════
def load_dk_lines_from_json(year):
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
            if ou_close is None:
                continue
            home_team = gv["homeTeam"]["shortName"]
            away_team = gv["awayTeam"]["shortName"]
            rows.append({
                "game_date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": gv.get("homeTeamScore"),
                "away_score": gv.get("awayTeamScore"),
                "ou_open": opening.get("total"),
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


def build_ou_dataset(year, odds_df=None):
    feat_path = ROOT / "data" / "features" / f"game_features_{year}.parquet"
    features = pd.read_parquet(feat_path)
    features["game_date"] = pd.to_datetime(features["game_date"]).dt.strftime("%Y-%m-%d")

    if odds_df is None:
        odds_df = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
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
    return merged


def engineer_ou_features(df):
    df = df.copy()
    df["dk_ou_line"] = df["ou_close"]
    df["line_movement"] = df["ou_close"] - df["ou_open"]

    def _american_to_prob(odds):
        odds = pd.to_numeric(odds, errors="coerce")
        return np.where(odds < 0, -odds / (-odds + 100), 100 / (odds + 100))

    df["implied_over_prob"] = _american_to_prob(df["over_close_odds"])
    df["implied_under_prob"] = _american_to_prob(df["under_close_odds"])
    df["implied_vig"] = df["implied_over_prob"] + df["implied_under_prob"] - 1
    df["devigged_over_prob"] = df["implied_over_prob"] / (df["implied_over_prob"] + df["implied_under_prob"])

    if "home_sp_xrv_mean" in df.columns and "away_sp_xrv_mean" in df.columns:
        df["sp_quality_sum"] = df["home_sp_xrv_mean"] + df["away_sp_xrv_mean"]
        df["sp_quality_diff"] = df["away_sp_xrv_mean"] - df["home_sp_xrv_mean"]
    if "home_hit_barrel_rate" in df.columns:
        df["lineup_power_sum"] = df["home_hit_barrel_rate"].fillna(0) + df["away_hit_barrel_rate"].fillna(0)
    if "home_bp_xrv_mean" in df.columns:
        df["bp_quality_sum"] = df["home_bp_xrv_mean"].fillna(0) + df["away_bp_xrv_mean"].fillna(0)
        df["bp_fatigue_sum"] = df["home_bp_fatigue_score"].fillna(0) + df["away_bp_fatigue_score"].fillna(0)
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
    }
    feat_cols = [c for c in df.columns if c not in exclude
                 and df[c].dtype in ("float64", "int64", "float32", "int32", "bool")]
    return df, feat_cols


def train_ou_model():
    """Train O/U LightGBM model on 2023-2024, return model + features + 2025 predictions."""
    print("\nTraining O/U model...")

    # Load training data
    dk_2024 = load_dk_lines_from_json(2024)
    dk_2023 = load_dk_lines_from_json(2023)

    ds_2024 = build_ou_dataset(2024, dk_2024)
    ds_2025 = build_ou_dataset(2025)

    try:
        ds_2023 = build_ou_dataset(2023, dk_2023)
        has_2023 = len(ds_2023) > 100
    except Exception:
        has_2023 = False
        ds_2023 = pd.DataFrame()

    if has_2023:
        train_full = pd.concat([ds_2023, ds_2024], ignore_index=True)
    else:
        train_full = ds_2024

    train_full = train_full[~train_full["is_push"]].copy()
    ds_2025_nopush = ds_2025[~ds_2025["is_push"]].copy()

    train_full, feat_cols_train = engineer_ou_features(train_full)
    ds_2025_nopush, feat_cols_test = engineer_ou_features(ds_2025_nopush)
    feat_cols = sorted(set(feat_cols_train) & set(feat_cols_test))

    X_train = train_full[feat_cols]
    y_train = train_full["over"].values

    lgb_params = {
        "objective": "binary", "metric": "binary_logloss",
        "n_estimators": 80, "max_depth": 3, "num_leaves": 8,
        "learning_rate": 0.05, "min_child_samples": 50,
        "reg_alpha": 2.0, "reg_lambda": 2.0,
        "colsample_bytree": 0.5, "subsample": 0.8,
        "subsample_freq": 1, "verbose": -1, "random_state": 42,
    }

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train)

    # Walk-forward calibration: use first 40% of 2025 as calibration
    ds_sorted = ds_2025_nopush.sort_values("game_date").reset_index(drop=True)
    n = len(ds_sorted)
    cal_end = int(n * 0.4)

    ds_sorted, _ = engineer_ou_features(ds_sorted)

    cal_preds = model.predict_proba(ds_sorted.iloc[:cal_end][feat_cols])[:, 1]
    cal_actuals = ds_sorted.iloc[:cal_end]["over"].values

    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(cal_preds, cal_actuals)

    # Predict on ALL of 2025 (will only use post-calibration period in backtest)
    all_preds_raw = model.predict_proba(ds_sorted[feat_cols])[:, 1]
    all_preds_cal = iso.predict(all_preds_raw)

    # Build output: game_date, home_team, away_team, p_over (calibrated)
    ou_preds = ds_sorted[["game_date", "home_team", "away_team"]].copy()
    ou_preds["p_over_cal"] = all_preds_cal
    ou_preds["p_under_cal"] = 1.0 - all_preds_cal
    ou_preds["ou_cal_start_idx"] = cal_end  # mark calibration boundary

    print(f"  O/U model trained: {len(X_train)} training games, "
          f"{cal_end} calibration, {n - cal_end} test period")
    print(f"  O/U calibration date boundary: {ds_sorted.iloc[cal_end]['game_date']}")

    return ou_preds, ds_sorted.iloc[cal_end]["game_date"]


# ═══════════════════════════════════════════════════════════════════════
# 3. Strategy Implementations
# ═══════════════════════════════════════════════════════════════════════
def strategy_kalshi_arb(row):
    """Strategy 1: DK-vs-Kalshi ML arbitrage. Returns list of bet dicts."""
    if pd.isna(row.get("kalshi_home_prob")) or pd.isna(row.get("kalshi_away_prob")):
        return []

    # Hard filters
    if row["is_division"] == 1:
        return []

    edge_home = row["dk_home_prob"] - row["kalshi_home_prob"]
    edge_away = row["dk_away_prob"] - row["kalshi_away_prob"]

    threshold = 0.07
    bets = []

    # Pick the side with highest edge above threshold
    best_side = None
    best_edge = threshold

    if edge_home > best_edge:
        best_side = "home"
        best_edge = edge_home
    if edge_away > best_edge:
        best_side = "away"
        best_edge = edge_away

    if best_side is None:
        return []

    # Hard filter: no away favorites
    if best_side == "away" and row["dk_away_prob"] >= 0.50:
        return []

    if best_side == "home":
        kalshi_prob = row["kalshi_home_prob"]
        our_prob = row["dk_home_prob"]
        won = int(row["home_win"] == 1)
        # Kalshi payout
        pnl = (1.0 - kalshi_prob) if won else -kalshi_prob
        b = (1.0 - kalshi_prob) / kalshi_prob  # decimal odds - 1
    else:
        kalshi_prob = row["kalshi_away_prob"]
        our_prob = row["dk_away_prob"]
        won = int(row["home_win"] == 0)
        pnl = (1.0 - kalshi_prob) if won else -kalshi_prob
        b = (1.0 - kalshi_prob) / kalshi_prob

    bets.append({
        "strategy": "kalshi_arb",
        "side": best_side,
        "edge": best_edge,
        "our_prob": our_prob,
        "odds_b": b,
        "won": won,
        "pnl_flat": pnl,
        "market": "kalshi",
    })
    return bets


def strategy_ou_under(row):
    """Strategy 2: O/U classifier unders when P(under) > 0.55."""
    if pd.isna(row.get("p_under_cal")):
        return []

    threshold = 0.55
    if row["p_under_cal"] < threshold:
        return []

    # Check if total is available for result
    if pd.isna(row.get("ou_close")) or pd.isna(row.get("total_runs")):
        return []

    # Push excluded
    if row["total_runs"] == row["ou_close"]:
        return []

    won = int(row["total_runs"] < row["ou_close"])

    # Payout from DK under closing odds
    under_dec = row.get("under_ml_dec", np.nan)
    if pd.isna(under_dec) or under_dec <= 1.0:
        # Fallback: -110 standard juice
        under_dec = 1.0 + 100.0 / 110.0

    b = under_dec - 1.0  # net payout per $1
    pnl = b if won else -1.0

    bets = [{
        "strategy": "ou_under",
        "side": "under",
        "edge": row["p_under_cal"] - 0.50,
        "our_prob": row["p_under_cal"],
        "odds_b": b,
        "won": won,
        "pnl_flat": pnl,
        "market": "dk",
    }]
    return bets


def strategy_away_dog_anomaly(row):
    """Strategy 3: Blind bet away underdogs when DK implied 35-40%."""
    dk_away = row["dk_away_prob"]
    if pd.isna(dk_away):
        return []

    # Hard filters
    if row["is_division"] == 1:
        return []
    if not (0.35 <= dk_away <= 0.40):
        return []

    # This is by definition an away underdog, so no "away favorite" filter needed.

    won = int(row["home_win"] == 0)
    away_dec = row["away_ml_dec"]
    if pd.isna(away_dec) or away_dec <= 1.0:
        return []

    b = away_dec - 1.0
    pnl = b if won else -1.0

    bets = [{
        "strategy": "away_dog",
        "side": "away",
        "edge": 0.05,  # estimated systematic edge
        "our_prob": 0.42,  # approximate true prob (DK underprices these)
        "odds_b": b,
        "won": won,
        "pnl_flat": pnl,
        "market": "dk",
    }]
    return bets


def strategy_pickem_home(row):
    """Strategy 4: Win model pick-em home bets (DK 48-52%, model edge > 3%)."""
    if pd.isna(row.get("model_prob")) or pd.isna(row.get("dk_home_prob")):
        return []

    dk_home = row["dk_home_prob"]

    # Must be pick-em range
    if not (0.48 <= dk_home <= 0.52):
        return []

    # Hard filters
    if row["is_division"] == 1:
        return []

    model_edge = row["model_prob"] - dk_home
    # Edge must be > 3% and < 10% (cap disagreement)
    if model_edge <= 0.03 or model_edge > 0.10:
        return []

    won = int(row["home_win"] == 1)
    home_dec = row["home_ml_dec"]
    if pd.isna(home_dec) or home_dec <= 1.0:
        return []

    b = home_dec - 1.0
    pnl = b if won else -1.0

    bets = [{
        "strategy": "pickem_home",
        "side": "home",
        "edge": model_edge,
        "our_prob": row["model_prob"],
        "odds_b": b,
        "won": won,
        "pnl_flat": pnl,
        "market": "dk",
    }]
    return bets


def strategy_nrfi(row):
    """Strategy 5: NRFI bet when P(NRFI) >= 0.57 at -120 odds."""
    if pd.isna(row.get("nrfi_lgb_prob")) or pd.isna(row.get("actual_nrfi")):
        return []

    p_nrfi = row["nrfi_lgb_prob"]
    if p_nrfi < 0.57:
        return []

    won = int(row["actual_nrfi"] == 1)

    # NRFI at -120 odds: risk 1.20 to win 1.00 -> decimal = 1 + 100/120 = 1.8333
    b = 100.0 / 120.0  # net payout per $1 risked = 0.8333
    pnl = b if won else -1.0

    bets = [{
        "strategy": "nrfi",
        "side": "nrfi",
        "edge": p_nrfi - (120.0 / 220.0),  # edge over breakeven 54.5%
        "our_prob": p_nrfi,
        "odds_b": b,
        "won": won,
        "pnl_flat": pnl,
        "market": "dk",
    }]
    return bets


def strategy_yrfi(row):
    """Strategy 6: YRFI bet when P(NRFI) <= 0.48 at +100 odds."""
    if pd.isna(row.get("nrfi_lgb_prob")) or pd.isna(row.get("actual_nrfi")):
        return []

    p_nrfi = row["nrfi_lgb_prob"]
    if p_nrfi > 0.48:
        return []

    won = int(row["actual_nrfi"] == 0)

    # YRFI at +100 odds: risk 1.00 to win 1.00 -> decimal = 2.00
    b = 1.0  # net payout per $1 risked
    p_yrfi = 1.0 - p_nrfi
    pnl = b if won else -1.0

    bets = [{
        "strategy": "yrfi",
        "side": "yrfi",
        "edge": p_yrfi - 0.50,  # edge over breakeven 50%
        "our_prob": p_yrfi,
        "odds_b": b,
        "won": won,
        "pnl_flat": pnl,
        "market": "dk",
    }]
    return bets


# ═══════════════════════════════════════════════════════════════════════
# 4. Walk-Forward Backtest Engine
# ═══════════════════════════════════════════════════════════════════════
STRATEGY_FUNCS = {
    "kalshi_arb": strategy_kalshi_arb,
    "ou_under": strategy_ou_under,
    "away_dog": strategy_away_dog_anomaly,
    "pickem_home": strategy_pickem_home,
    "nrfi": strategy_nrfi,
    "yrfi": strategy_yrfi,
}

# Quarter-Kelly for each; asymmetric NRFI/YRFI sizing
KELLY_FRACTIONS = {
    "kalshi_arb": 0.25,
    "ou_under": 0.25,
    "away_dog": 0.25,
    "pickem_home": 0.15,  # smaller — not yet significant
    "nrfi": 0.15,         # weaker signal
    "yrfi": 0.25,         # stronger signal
}

MAX_BET_FRACTION = 0.05  # cap any single bet at 5% of bankroll
SEPTEMBER_MULTIPLIER = 0.5


def run_backtest(df, starting_bankroll=10_000):
    """
    Walk-forward backtest across all strategies with Kelly sizing.
    Each game date: evaluate all strategies, size with fractional Kelly,
    update bankroll.
    """
    all_bets = []
    bankroll = starting_bankroll
    equity_trace = [{"date": df["game_date"].iloc[0], "bankroll": bankroll}]

    for _, row in df.iterrows():
        game_date = row["game_date"]
        is_sep = row.get("is_september", 0) == 1

        for strat_name, strat_func in STRATEGY_FUNCS.items():
            bets = strat_func(row)
            for bet in bets:
                # Kelly sizing
                p = bet["our_prob"]
                b = bet["odds_b"]
                kelly_mult = KELLY_FRACTIONS[strat_name]
                f_full = kelly_fraction(p, b)
                f = f_full * kelly_mult

                # Cap at max fraction
                f = min(f, MAX_BET_FRACTION)

                # September reduction
                if is_sep:
                    f *= SEPTEMBER_MULTIPLIER

                if f <= 0:
                    continue

                stake = bankroll * f
                if bet["won"]:
                    pnl_dollar = stake * b
                else:
                    pnl_dollar = -stake

                bankroll += pnl_dollar

                bet.update({
                    "game_date": game_date,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "kelly_f": f,
                    "stake": stake,
                    "pnl_dollar": pnl_dollar,
                    "bankroll_after": bankroll,
                })
                all_bets.append(bet)

        equity_trace.append({"date": game_date, "bankroll": bankroll})

    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()
    equity_df = pd.DataFrame(equity_trace)
    # Collapse equity to end-of-day
    equity_df = equity_df.groupby("date").last().reset_index()

    return bets_df, equity_df, bankroll


# ═══════════════════════════════════════════════════════════════════════
# 5. Performance Metrics
# ═══════════════════════════════════════════════════════════════════════
def compute_metrics(bets_df, equity_df, starting_bankroll=10_000):
    """Compute comprehensive performance metrics."""
    if bets_df.empty:
        return {}

    total_pnl = bets_df["pnl_dollar"].sum()
    total_return = total_pnl / starting_bankroll
    n_bets = len(bets_df)
    win_rate = bets_df["won"].mean()

    # Daily returns
    equity_df = equity_df.copy()
    equity_df["daily_return"] = equity_df["bankroll"].pct_change().fillna(0)

    daily_pnl = bets_df.groupby("game_date")["pnl_dollar"].sum()
    weekly_pnl = daily_pnl.groupby(pd.to_datetime(daily_pnl.index).to_period("W")).sum()
    monthly_pnl = daily_pnl.groupby(pd.to_datetime(daily_pnl.index).to_period("M")).sum()

    # Sharpe (annualized, ~162 game days)
    daily_rets = equity_df["daily_return"].values[1:]
    if len(daily_rets) > 1 and daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(162)
    else:
        sharpe = 0.0

    # Max drawdown
    eq = equity_df["bankroll"].values
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = dd.max()

    # Worst periods
    worst_day = daily_pnl.min() if len(daily_pnl) > 0 else 0
    worst_week = weekly_pnl.min() if len(weekly_pnl) > 0 else 0
    worst_month = monthly_pnl.min() if len(monthly_pnl) > 0 else 0
    best_day = daily_pnl.max() if len(daily_pnl) > 0 else 0
    best_month = monthly_pnl.max() if len(monthly_pnl) > 0 else 0

    return {
        "n_bets": n_bets,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "worst_day": worst_day,
        "worst_week": worst_week,
        "worst_month": worst_month,
        "best_day": best_day,
        "best_month": best_month,
        "final_bankroll": equity_df["bankroll"].iloc[-1],
    }


def per_strategy_breakdown(bets_df, starting_bankroll=10_000):
    """P&L contribution by strategy."""
    if bets_df.empty:
        return pd.DataFrame()

    rows = []
    for strat, group in bets_df.groupby("strategy"):
        n = len(group)
        wins = group["won"].sum()
        pnl = group["pnl_dollar"].sum()
        avg_edge = group["edge"].mean()
        avg_kelly = group["kelly_f"].mean()

        # Flat-bet ROI (using pnl_flat)
        flat_roi = group["pnl_flat"].mean() if "pnl_flat" in group.columns else np.nan

        rows.append({
            "strategy": strat,
            "n_bets": n,
            "wins": int(wins),
            "win_rate": wins / n,
            "total_pnl": pnl,
            "pct_of_bankroll": pnl / starting_bankroll,
            "flat_roi": flat_roi,
            "avg_edge": avg_edge,
            "avg_kelly_f": avg_kelly,
        })

    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def monthly_breakdown(bets_df):
    """Monthly P&L breakdown."""
    if bets_df.empty:
        return pd.DataFrame()

    bets_df = bets_df.copy()
    bets_df["month"] = bets_df["game_date"].str[:7]

    rows = []
    for month, group in bets_df.groupby("month"):
        rows.append({
            "month": month,
            "n_bets": len(group),
            "wins": int(group["won"].sum()),
            "win_rate": group["won"].mean(),
            "pnl": group["pnl_dollar"].sum(),
            "strategies": dict(group.groupby("strategy")["pnl_dollar"].sum()),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 6. Daily Picks Function
# ═══════════════════════════════════════════════════════════════════════
def daily_picks(date_str, df, bankroll=10_000):
    """
    Generate recommended bets for a given date.

    Parameters:
        date_str: "YYYY-MM-DD"
        df: merged DataFrame (same format as load_all_data output)
        bankroll: current bankroll

    Returns:
        DataFrame of recommended bets
    """
    day_games = df[df["game_date"] == date_str]
    if day_games.empty:
        print(f"No games found for {date_str}")
        return pd.DataFrame()

    picks = []
    for _, row in day_games.iterrows():
        is_sep = pd.to_datetime(date_str).month == 9

        for strat_name, strat_func in STRATEGY_FUNCS.items():
            bets = strat_func(row)
            for bet in bets:
                p = bet["our_prob"]
                b = bet["odds_b"]
                kelly_mult = KELLY_FRACTIONS[strat_name]
                f = kelly_fraction(p, b) * kelly_mult
                f = min(f, MAX_BET_FRACTION)
                if is_sep:
                    f *= SEPTEMBER_MULTIPLIER
                if f <= 0:
                    continue

                stake = bankroll * f
                matchup = f"{row['away_team']} @ {row['home_team']}"

                picks.append({
                    "matchup": matchup,
                    "strategy": strat_name,
                    "side": bet["side"],
                    "market": bet["market"],
                    "edge": bet["edge"],
                    "our_prob": bet["our_prob"],
                    "kelly_pct": f * 100,
                    "stake": round(stake, 2),
                })

    picks_df = pd.DataFrame(picks)
    if not picks_df.empty:
        picks_df = picks_df.sort_values("stake", ascending=False)
    return picks_df


# ═══════════════════════════════════════════════════════════════════════
# 7. Individual Strategy Backtests (for comparison)
# ═══════════════════════════════════════════════════════════════════════
def backtest_single_strategy(df, strat_name, strat_func, kelly_mult,
                              starting_bankroll=10_000):
    """Run a single strategy in isolation."""
    bets = []
    bankroll = starting_bankroll
    eq_trace = [bankroll]

    for _, row in df.iterrows():
        is_sep = row.get("is_september", 0) == 1
        results = strat_func(row)
        for bet in results:
            p = bet["our_prob"]
            b = bet["odds_b"]
            f = kelly_fraction(p, b) * kelly_mult
            f = min(f, MAX_BET_FRACTION)
            if is_sep:
                f *= SEPTEMBER_MULTIPLIER
            if f <= 0:
                continue
            stake = bankroll * f
            if bet["won"]:
                pnl = stake * b
            else:
                pnl = -stake
            bankroll += pnl
            bet.update({
                "game_date": row["game_date"],
                "pnl_dollar": pnl,
                "kelly_f": f,
                "stake": stake,
                "bankroll_after": bankroll,
            })
            bets.append(bet)
        eq_trace.append(bankroll)

    bets_df = pd.DataFrame(bets) if bets else pd.DataFrame()
    return bets_df, bankroll


# ═══════════════════════════════════════════════════════════════════════
# 8. Main Report
# ═══════════════════════════════════════════════════════════════════════
def fmt(x, pct=False, dollar=False, dec=2):
    if pd.isna(x):
        return "N/A"
    if dollar:
        return f"${x:,.{dec}f}"
    if pct:
        return f"{x*100:.{dec}f}%"
    return f"{x:.{dec}f}"


def print_separator(title="", width=78):
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)


def main():
    starting_bankroll = 10_000

    print_separator("UNIFIED DAILY PREDICTION & BETTING STRATEGY")
    print(f"  Starting Bankroll: ${starting_bankroll:,.0f}")
    print(f"  Kelly Fractions:   {KELLY_FRACTIONS}")
    print(f"  Max Bet Fraction:  {MAX_BET_FRACTION:.0%}")
    print(f"  September Factor:  {SEPTEMBER_MULTIPLIER:.0%}")
    print()

    # ── Load data ─────────────────────────────────────────────────
    df = load_all_data()

    # ── Train & merge O/U model predictions ───────────────────────
    ou_preds, ou_cal_date = train_ou_model()
    df = df.merge(
        ou_preds[["game_date", "home_team", "away_team", "p_over_cal", "p_under_cal"]],
        on=["game_date", "home_team", "away_team"],
        how="left",
    )
    print(f"\n  Games with O/U predictions: {df['p_under_cal'].notna().sum()}")

    # ── Use walk-forward test period ──────────────────────────────
    # Kalshi arb uses 30% calibration; O/U uses 40% calibration
    # Use the later boundary (O/U cal date) as the start of test period
    df_test = df[df["game_date"] >= ou_cal_date].copy().reset_index(drop=True)
    print(f"\n  Test period: {df_test['game_date'].min()} to {df_test['game_date'].max()}")
    print(f"  Test period games: {len(df_test)}")

    # ══════════════════════════════════════════════════════════════
    # COMBINED BACKTEST
    # ══════════════════════════════════════════════════════════════
    print_separator("COMBINED WALK-FORWARD BACKTEST")

    bets_df, equity_df, final_bankroll = run_backtest(df_test, starting_bankroll)

    if bets_df.empty:
        print("  No bets generated. Check data coverage.")
        return

    metrics = compute_metrics(bets_df, equity_df, starting_bankroll)

    print(f"\n  Total Bets:        {metrics['n_bets']}")
    print(f"  Win Rate:          {fmt(metrics['win_rate'], pct=True)}")
    print(f"  Total P&L:         {fmt(metrics['total_pnl'], dollar=True)}")
    print(f"  Total Return:      {fmt(metrics['total_return'], pct=True)}")
    print(f"  Final Bankroll:    {fmt(metrics['final_bankroll'], dollar=True)}")
    print(f"  Sharpe Ratio:      {fmt(metrics['sharpe'])}")
    print(f"  Max Drawdown:      {fmt(metrics['max_drawdown'], pct=True)}")
    print(f"\n  Worst Day:         {fmt(metrics['worst_day'], dollar=True)}")
    print(f"  Worst Week:        {fmt(metrics['worst_week'], dollar=True)}")
    print(f"  Worst Month:       {fmt(metrics['worst_month'], dollar=True)}")
    print(f"  Best Day:          {fmt(metrics['best_day'], dollar=True)}")
    print(f"  Best Month:        {fmt(metrics['best_month'], dollar=True)}")

    # ── Per-strategy breakdown ────────────────────────────────────
    print_separator("PER-STRATEGY CONTRIBUTION")

    strat_df = per_strategy_breakdown(bets_df, starting_bankroll)
    for _, row in strat_df.iterrows():
        print(f"\n  {row['strategy'].upper()}")
        print(f"    Bets: {row['n_bets']}  |  Win Rate: {fmt(row['win_rate'], pct=True)}  "
              f"|  Flat ROI: {fmt(row['flat_roi'], pct=True)}")
        print(f"    P&L: {fmt(row['total_pnl'], dollar=True)}  "
              f"|  % of Bankroll: {fmt(row['pct_of_bankroll'], pct=True)}  "
              f"|  Avg Edge: {fmt(row['avg_edge'], pct=True)}")

    # ── Monthly P&L ───────────────────────────────────────────────
    print_separator("MONTHLY P&L BREAKDOWN")

    monthly = monthly_breakdown(bets_df)
    if not monthly.empty:
        print(f"\n  {'Month':>8}  {'Bets':>5}  {'Wins':>5}  {'WR':>6}  {'P&L':>12}")
        print("  " + "-" * 44)
        for _, row in monthly.iterrows():
            print(f"  {row['month']:>8}  {row['n_bets']:>5}  {row['wins']:>5}  "
                  f"{row['win_rate']:.1%}  {fmt(row['pnl'], dollar=True):>12}")

        # Strategy-level monthly
        print(f"\n  Monthly P&L by Strategy:")
        print(f"  {'Month':>8}", end="")
        strat_names = sorted(bets_df["strategy"].unique())
        for s in strat_names:
            print(f"  {s:>14}", end="")
        print()
        print("  " + "-" * (8 + 16 * len(strat_names)))
        for _, row in monthly.iterrows():
            print(f"  {row['month']:>8}", end="")
            for s in strat_names:
                v = row["strategies"].get(s, 0)
                print(f"  {fmt(v, dollar=True):>14}", end="")
            print()

    # ══════════════════════════════════════════════════════════════
    # INDIVIDUAL STRATEGY COMPARISON
    # ══════════════════════════════════════════════════════════════
    print_separator("INDIVIDUAL STRATEGY COMPARISON (isolated backtests)")

    comparison_rows = []
    for strat_name, strat_func in STRATEGY_FUNCS.items():
        kelly_mult = KELLY_FRACTIONS[strat_name]
        ind_bets, ind_final = backtest_single_strategy(
            df_test, strat_name, strat_func, kelly_mult, starting_bankroll
        )
        if ind_bets.empty:
            comparison_rows.append({
                "strategy": strat_name, "n_bets": 0,
                "win_rate": np.nan, "total_return": np.nan,
                "final_bankroll": starting_bankroll,
            })
            continue

        ind_pnl = ind_bets["pnl_dollar"].sum()
        comparison_rows.append({
            "strategy": strat_name,
            "n_bets": len(ind_bets),
            "win_rate": ind_bets["won"].mean(),
            "flat_roi": ind_bets["pnl_flat"].mean() if "pnl_flat" in ind_bets.columns else np.nan,
            "total_pnl": ind_pnl,
            "total_return": ind_pnl / starting_bankroll,
            "final_bankroll": ind_final,
        })

    # Add combined
    comparison_rows.append({
        "strategy": "COMBINED",
        "n_bets": metrics["n_bets"],
        "win_rate": metrics["win_rate"],
        "flat_roi": bets_df["pnl_flat"].mean() if "pnl_flat" in bets_df.columns else np.nan,
        "total_pnl": metrics["total_pnl"],
        "total_return": metrics["total_return"],
        "final_bankroll": metrics["final_bankroll"],
    })

    comp_df = pd.DataFrame(comparison_rows)
    print(f"\n  {'Strategy':>14}  {'Bets':>5}  {'WR':>6}  {'Flat ROI':>9}  "
          f"{'Kelly P&L':>11}  {'Return':>8}  {'Final $':>10}")
    print("  " + "-" * 74)
    for _, row in comp_df.iterrows():
        strat = row["strategy"]
        if strat == "COMBINED":
            print("  " + "-" * 74)
        print(f"  {strat:>14}  {int(row['n_bets']):>5}  "
              f"{fmt(row.get('win_rate', np.nan), pct=True):>6}  "
              f"{fmt(row.get('flat_roi', np.nan), pct=True):>9}  "
              f"{fmt(row.get('total_pnl', 0), dollar=True):>11}  "
              f"{fmt(row.get('total_return', np.nan), pct=True):>8}  "
              f"{fmt(row.get('final_bankroll', starting_bankroll), dollar=True):>10}")

    # ── Equity curve summary ──────────────────────────────────────
    print_separator("EQUITY CURVE SUMMARY")
    if not equity_df.empty:
        print(f"\n  Start:    {equity_df['date'].iloc[0]}  ->  ${starting_bankroll:,.0f}")
        print(f"  End:      {equity_df['date'].iloc[-1]}  ->  ${equity_df['bankroll'].iloc[-1]:,.2f}")
        print(f"  Peak:     ${equity_df['bankroll'].max():,.2f}")
        print(f"  Trough:   ${equity_df['bankroll'].min():,.2f}")

    # ── Sample daily picks for last available date ────────────────
    last_date = df_test["game_date"].iloc[-1]
    print_separator(f"SAMPLE DAILY PICKS: {last_date}")
    picks = daily_picks(last_date, df_test, metrics["final_bankroll"])
    if not picks.empty:
        print(f"\n  {'Matchup':>22}  {'Strategy':>14}  {'Side':>6}  {'Market':>7}  "
              f"{'Edge':>6}  {'Kelly%':>7}  {'Stake':>9}")
        print("  " + "-" * 82)
        for _, p in picks.iterrows():
            print(f"  {p['matchup']:>22}  {p['strategy']:>14}  {p['side']:>6}  "
                  f"{p['market']:>7}  {p['edge']:.1%}  {p['kelly_pct']:>6.2f}%  "
                  f"${p['stake']:>8.2f}")
    else:
        print("  No picks for this date.")

    # ── Expected annual projections ──────────────────────────────
    print_separator("EXPECTED ANNUAL PROJECTIONS")
    n_test_days = pd.to_datetime(df_test["game_date"]).nunique()
    full_season_days = 183  # ~late March to late September
    if n_test_days > 0:
        daily_bet_rate = metrics["n_bets"] / n_test_days
        projected_annual_bets = daily_bet_rate * full_season_days
        # Annualize return: scale from test-period days to full season
        test_period_return = metrics["total_return"]
        annualized_roi = test_period_return * (full_season_days / n_test_days)
        print(f"\n  Test period length:     {n_test_days} game days")
        print(f"  Bets per day (avg):     {daily_bet_rate:.1f}")
        print(f"  Projected annual bets:  {projected_annual_bets:.0f}")
        print(f"  Test period ROI:        {test_period_return*100:.1f}%")
        print(f"  Annualized ROI:         {annualized_roi*100:.1f}%")
        print(f"  (assuming similar edge persistence over full season)")

    print_separator("DONE")

    return {
        "bets_df": bets_df,
        "equity_df": equity_df,
        "metrics": metrics,
        "strategy_comparison": comp_df,
        "monthly": monthly,
    }


if __name__ == "__main__":
    results = main()
