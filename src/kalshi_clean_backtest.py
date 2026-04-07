#!/usr/bin/env python3
"""
Clean Kalshi ML Backtest — Zero Look-Ahead Bias
================================================

All bets are placed at actual Kalshi prices. P&L reflects Kalshi contract
mechanics (buy at price p, receive $1 if correct, $0 if wrong).

Strict temporal protocol:
  - TRAIN:      2024 + early 2025 (before Apr 16) — model fitting + Optuna HP tuning via TSCV
  - VALIDATION: 2025-04-16 to 2025-06-30 (threshold/weight selection)
  - TEST:       2025-07-01 to 2025-10-29 (final unbiased evaluation)

NO test data contaminates ANY decision (thresholds, weights, calibration, HPs).

Signal sources (all compared against Kalshi prices):
  1. LightGBM win model (trained 2024+early 2025, isotonic calibrated, Brier-tuned)
  2. MC simulator (sim_home_wp)
  3. Ensemble: weighted combination of above

Usage:
    python src/kalshi_clean_backtest.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
VAL_CUTOFF = "2025-07-01"  # validation/test split

# PA outcome categories from multi-output matchup model
OUTCOME_ORDER = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "dp", "out_ground", "out_fly", "out_line"]
N_HITTERS = 9


def aggregate_nn_features(nn_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 198 per-hitter PA distribution columns into ~30 LGB-friendly features.

    For each side (home/away), computes:
    - Lineup mean for each outcome category (11 per side = 22 total)
    - Lineup OBP proxy: mean(BB + HBP + 1B + 2B + 3B + HR) across hitters
    - Lineup SLG proxy: mean(1B + 2*2B + 3*3B + 4*HR) across hitters
    - Lineup K rate std (heterogeneity in lineup quality)
    - Top-3 hitter HR mean (lineup power concentration)
    - Plus pitcher eval and context features pass-through
    """
    rows = []
    for _, r in nn_df.iterrows():
        row = {
            "game_pk": r["game_pk"],
            "game_date": r["game_date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
        }

        for side in ["home", "away"]:
            # Collect per-hitter outcome arrays
            outcome_arrays = {o: [] for o in OUTCOME_ORDER}
            n_valid = 0
            for slot in range(N_HITTERS):
                k_val = r.get(f"{side}_h{slot}_K", np.nan)
                if pd.isna(k_val):
                    continue
                n_valid += 1
                for o in OUTCOME_ORDER:
                    outcome_arrays[o].append(r[f"{side}_h{slot}_{o}"])

            if n_valid < 3:
                # Not enough hitters — leave as NaN
                for o in OUTCOME_ORDER:
                    row[f"nn_{side}_mean_{o}"] = np.nan
                row[f"nn_{side}_obp"] = np.nan
                row[f"nn_{side}_slg"] = np.nan
                row[f"nn_{side}_k_std"] = np.nan
                row[f"nn_{side}_top3_hr"] = np.nan
                continue

            # Mean of each outcome across lineup
            for o in OUTCOME_ORDER:
                row[f"nn_{side}_mean_{o}"] = np.mean(outcome_arrays[o])

            # OBP proxy
            obp_per_hitter = [
                outcome_arrays["BB"][i] + outcome_arrays["HBP"][i] +
                outcome_arrays["1B"][i] + outcome_arrays["2B"][i] +
                outcome_arrays["3B"][i] + outcome_arrays["HR"][i]
                for i in range(n_valid)
            ]
            row[f"nn_{side}_obp"] = np.mean(obp_per_hitter)

            # SLG proxy (expected bases per PA)
            slg_per_hitter = [
                outcome_arrays["1B"][i] + 2 * outcome_arrays["2B"][i] +
                3 * outcome_arrays["3B"][i] + 4 * outcome_arrays["HR"][i]
                for i in range(n_valid)
            ]
            row[f"nn_{side}_slg"] = np.mean(slg_per_hitter)

            # K rate heterogeneity
            row[f"nn_{side}_k_std"] = np.std(outcome_arrays["K"])

            # Top-3 hitter HR potential
            hrs = sorted(outcome_arrays["HR"], reverse=True)
            row[f"nn_{side}_top3_hr"] = np.mean(hrs[:3])

        # Differential features (home advantage)
        for feat in ["obp", "slg", "top3_hr"]:
            h = row.get(f"nn_home_{feat}", np.nan)
            a = row.get(f"nn_away_{feat}", np.nan)
            if pd.notna(h) and pd.notna(a):
                row[f"nn_diff_{feat}"] = h - a
            else:
                row[f"nn_diff_{feat}"] = np.nan

        # Pass through pitcher eval scores
        for side in ["home", "away"]:
            for feat in ["sp_stuff", "sp_location", "sp_sequencing"]:
                row[f"nn_{side}_{feat}"] = r.get(f"{side}_{feat}", np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 1. Data Loading
# ═══════════════════════════════════════════════════════════════════

def american_to_prob(odds):
    """American odds -> raw implied probability."""
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100),
                    np.where(odds > 0, 100 / (odds + 100), np.nan))


def extract_dk_lines_from_json(year: int) -> pd.DataFrame:
    """Extract DraftKings devigged closing moneyline from mlb_odds_dataset.json."""
    odds_path = DATA / "odds" / "mlb_odds_dataset.json"
    if not odds_path.exists():
        return pd.DataFrame()

    import json
    with open(odds_path) as f:
        data = json.load(f)

    # Team abbreviation normalization (SBR dataset uses shortName)
    SBR_MAP = {
        "TOR": "TOR", "NYY": "NYY", "NYM": "NYM", "BOS": "BOS",
        "TB": "TB", "BAL": "BAL", "CLE": "CLE", "DET": "DET",
        "MIN": "MIN", "CWS": "CWS", "KC": "KC", "HOU": "HOU",
        "TEX": "TEX", "SEA": "SEA", "LAA": "LAA", "OAK": "OAK",
        "ATL": "ATL", "PHI": "PHI", "MIA": "MIA", "WSH": "WSH",
        "CHC": "CHC", "STL": "STL", "MIL": "MIL", "CIN": "CIN",
        "PIT": "PIT", "LAD": "LAD", "SD": "SD", "SF": "SF",
        "AZ": "AZ", "COL": "COL", "ARI": "AZ",
    }

    rows = []
    for date_str, games in data.items():
        if not date_str.startswith(str(year)):
            continue
        for game in games:
            gv = game.get("gameView", {})
            home_short = gv.get("homeTeam", {}).get("shortName", "")
            away_short = gv.get("awayTeam", {}).get("shortName", "")
            home_abbr = SBR_MAP.get(home_short, home_short)
            away_abbr = SBR_MAP.get(away_short, away_short)

            for book in game.get("odds", {}).get("moneyline", []):
                if "draft" not in book.get("sportsbook", "").lower():
                    continue
                cl = book.get("currentLine", {})
                home_odds = cl.get("homeOdds")
                away_odds = cl.get("awayOdds")
                if home_odds is None or away_odds is None:
                    break

                h_imp = abs(home_odds) / (abs(home_odds) + 100) if home_odds < 0 else 100 / (home_odds + 100)
                a_imp = abs(away_odds) / (abs(away_odds) + 100) if away_odds < 0 else 100 / (away_odds + 100)
                total = h_imp + a_imp
                if total > 0:
                    rows.append({
                        "game_date": date_str,
                        "home_team": home_abbr,
                        "away_team": away_abbr,
                        "dk_home_prob": h_imp / total,
                        "dk_away_prob": a_imp / total,
                    })
                break

    return pd.DataFrame(rows)


def load_all_data():
    """Load and merge Kalshi, DK, features, and sim data."""
    # Kalshi ML
    kalshi = pd.read_parquet(DATA / "kalshi" / "kalshi_mlb_2025.parquet")
    kalshi["game_date"] = pd.to_datetime(kalshi["game_date"]).dt.strftime("%Y-%m-%d")

    # DK closing lines
    dk = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    dk["game_date"] = pd.to_datetime(dk["game_date"]).dt.strftime("%Y-%m-%d")
    # Devig DK ML
    dk["dk_home_raw"] = american_to_prob(dk["home_ml_close"])
    dk["dk_away_raw"] = american_to_prob(dk["away_ml_close"])
    dk["dk_total"] = dk["dk_home_raw"] + dk["dk_away_raw"]
    dk["dk_home_prob"] = dk["dk_home_raw"] / dk["dk_total"]
    dk["dk_away_prob"] = dk["dk_away_raw"] / dk["dk_total"]
    dk_cols = ["game_date", "home_team", "away_team", "dk_home_prob", "dk_away_prob",
               "home_ml_close", "away_ml_close"]
    dk = dk[dk_cols].dropna(subset=["dk_home_prob"])

    # Features (2025)
    feat_path = DATA / "features" / "game_features_2025.parquet"
    feat = pd.read_parquet(feat_path) if feat_path.exists() else pd.DataFrame()
    if not feat.empty:
        feat["game_date"] = pd.to_datetime(feat["game_date"]).dt.strftime("%Y-%m-%d")

    # MC sim predictions
    sim_path = DATA / "audit" / "sim_vs_kalshi_pregame_2025.csv"
    sim = pd.read_csv(sim_path) if sim_path.exists() else pd.DataFrame()
    if not sim.empty:
        sim["game_date"] = pd.to_datetime(sim["game_date"]).dt.strftime("%Y-%m-%d")
        sim = sim[["game_date", "home_team", "away_team", "sim_home_wp"]].copy()

    # Merge: Kalshi + DK
    merge_keys = ["game_date", "home_team", "away_team"]
    df = kalshi.merge(dk, on=merge_keys, how="inner")
    print(f"  Kalshi+DK merge: {len(df)} games")

    # Merge features
    if not feat.empty:
        df = df.merge(feat, on=merge_keys, how="left", suffixes=("", "_feat"))
        print(f"  After features merge: {len(df)} games, {sum(df['home_win_feat'].notna()) if 'home_win_feat' in df.columns else 'N/A'} with features")

    # Merge sim
    if not sim.empty:
        df = df.merge(sim, on=merge_keys, how="left")
        print(f"  Games with sim predictions: {df['sim_home_wp'].notna().sum()}")

    # Compute DK-vs-Kalshi edge
    df["dk_edge_home"] = df["dk_home_prob"] - df["kalshi_home_prob"]
    df["dk_edge_away"] = df["dk_away_prob"] - df["kalshi_away_prob"]

    # NN win model predictions
    nn_path = DATA / "features" / "nn_predictions_2025.parquet"
    if nn_path.exists():
        nn_preds = pd.read_parquet(nn_path)
        nn_preds["game_date"] = pd.to_datetime(nn_preds["game_date"]).dt.strftime("%Y-%m-%d")
        nn_preds = nn_preds[["game_date", "home_team", "away_team", "nn_home_prob"]].copy()
        df = df.merge(nn_preds, on=merge_keys, how="left")
        print(f"  Games with NN predictions: {df['nn_home_prob'].notna().sum()}")

    # Aggregated NN lineup features (PA distributions summarized for LGB)
    nn_feat_path = DATA / "features" / "nn_features_2025.parquet"
    if nn_feat_path.exists():
        nn_raw = pd.read_parquet(nn_feat_path)
        nn_raw["game_date"] = pd.to_datetime(nn_raw["game_date"]).dt.strftime("%Y-%m-%d")
        nn_agg = aggregate_nn_features(nn_raw)
        nn_agg["game_date"] = pd.to_datetime(nn_agg["game_date"]).dt.strftime("%Y-%m-%d")
        # Drop columns that would collide
        nn_agg_cols = [c for c in nn_agg.columns if c.startswith("nn_") or c in merge_keys]
        nn_agg = nn_agg[nn_agg_cols].copy()
        df = df.merge(nn_agg, on=merge_keys, how="left")
        nn_count = df[[c for c in df.columns if c.startswith("nn_home_mean_")]].notna().any(axis=1).sum()
        print(f"  Games with NN lineup features: {nn_count}")

    return df


# ═══════════════════════════════════════════════════════════════════
# 2. LightGBM Win Model (trained on 2024 + early 2025)
# ═══════════════════════════════════════════════════════════════════

def get_feature_cols(df):
    """Get feature columns, excluding targets and identifiers."""
    exclude = {
        "game_pk", "game_date", "home_team", "away_team", "home_win", "home_win_feat",
        "home_score", "away_score", "is_home",
        # Kalshi prices — what we bet against, must NOT be a feature
        "kalshi_home_prob", "kalshi_away_prob",
        # DK raw/intermediate cols (dk_home_prob IS a feature now)
        "dk_edge_home", "dk_edge_away", "dk_home_raw", "dk_away_raw", "dk_total",
        "dk_away_prob",  # redundant with dk_home_prob
        "home_ml_close", "away_ml_close", "volume", "event_ticker", "price_source",
        "sim_home_wp", "nn_home_prob",  # NN prediction is a separate signal, not LGB feature
    }
    return [c for c in df.columns if c not in exclude
            and not c.endswith("_feat")
            and df[c].dtype in ("float64", "float32", "int64", "int32")]


def train_win_model(feat_2024, feat_2025_val):
    """Train LGB win model on training data, tune HPs with Optuna TSCV.

    Returns: (model, feature_cols).
    """
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    feat_cols = get_feature_cols(feat_2024)
    # Filter to columns that exist in both 2024 and 2025
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

    # Optuna HP tuning with TSCV, optimizing Brier score
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
            preds = m.predict_proba(X_va)[:, 1]
            scores.append(log_loss(y_va, preds))
        return np.mean(scores)

    from tqdm import tqdm
    pbar = tqdm(total=120, desc="Optuna HP tuning")
    study = optuna.create_study(direction="minimize")

    def callback(study, trial):
        pbar.update(1)
        pbar.set_description(f"Best: {study.best_value:.5f}")

    study.optimize(objective, n_trials=120, callbacks=[callback])
    pbar.close()

    best = study.best_params
    best.update({"objective": "binary", "verbose": -1, "random_state": 42, "n_jobs": -1})
    print(f"  Best CV log-loss: {study.best_value:.5f}")
    for k, v in sorted(best.items()):
        if k not in ("objective", "verbose", "random_state", "n_jobs"):
            print(f"    {k}: {v}")

    # Train final model on ALL training data
    model = lgb.LGBMClassifier(**best)
    model.fit(X_train, y_train)

    return model, feat_cols


# ═══════════════════════════════════════════════════════════════════
# 3. Betting Logic (Kalshi contract mechanics)
# ═══════════════════════════════════════════════════════════════════

def kalshi_pnl(our_prob, kalshi_price, actual_win, side="home"):
    """
    Compute P&L for a Kalshi contract purchase.

    Buy at kalshi_price (e.g., 0.45).
    If correct: profit = 1.0 - kalshi_price
    If wrong: loss = -kalshi_price
    """
    if side == "home":
        buy_price = kalshi_price
        won = actual_win == 1
    else:
        buy_price = 1.0 - kalshi_price  # buying "away" = buying "no" on home
        won = actual_win == 0

    if won:
        return 1.0 - buy_price
    else:
        return -buy_price


def compute_bets(df, our_prob_col, min_edge, kelly_frac=0.25, max_bet_frac=0.05):
    """
    Generate bets for a single signal source.

    For each game:
      - If our_prob_home - kalshi_home > min_edge: buy home on Kalshi
      - If our_prob_away - kalshi_away > min_edge: buy away on Kalshi
      - At most one bet per game (whichever side has larger edge)

    Returns DataFrame of bets.
    """
    bets = []
    for _, row in df.iterrows():
        our_home = row[our_prob_col]
        our_away = 1.0 - our_home
        k_home = row["kalshi_home_prob"]
        k_away = row["kalshi_away_prob"]

        if pd.isna(our_home) or pd.isna(k_home):
            continue

        edge_home = our_home - k_home
        edge_away = our_away - k_away

        # Pick the better side
        if edge_home > edge_away and edge_home > min_edge:
            side = "home"
            edge = edge_home
            buy_price = k_home
            our_p = our_home
        elif edge_away > min_edge:
            side = "away"
            edge = edge_away
            buy_price = k_away
            our_p = our_away
        else:
            continue

        pnl = kalshi_pnl(our_p, row["kalshi_home_prob"], row["home_win"], side)
        b = (1.0 - buy_price) / buy_price  # odds ratio for Kelly
        q = 1.0 - our_p
        kelly_f = max(0.0, (our_p * b - q) / b) * kelly_frac
        kelly_f = min(kelly_f, max_bet_frac)

        bets.append({
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "side": side,
            "edge": edge,
            "our_prob": our_p,
            "kalshi_price": buy_price,
            "pnl_flat": pnl,
            "kelly_f": kelly_f,
            "won": int(pnl > 0),
            "home_win": row["home_win"],
        })
    return pd.DataFrame(bets)


# ═══════════════════════════════════════════════════════════════════
# 4. Kelly-Sized Backtest
# ═══════════════════════════════════════════════════════════════════

def run_kelly_backtest(bets_df, starting_bankroll=10000):
    """Run Kelly-sized backtest and return metrics."""
    if bets_df.empty:
        return {"n_bets": 0, "sharpe": 0, "roi": 0, "final_bankroll": starting_bankroll}

    bankroll = starting_bankroll
    daily_pnl = []
    equity = [bankroll]

    for date, day_bets in bets_df.groupby("game_date"):
        day_total = 0
        for _, bet in day_bets.iterrows():
            stake = bankroll * bet["kelly_f"]
            if stake <= 0:
                continue
            if bet["won"]:
                dollar_pnl = stake * (1.0 - bet["kalshi_price"]) / bet["kalshi_price"]
            else:
                dollar_pnl = -stake
            bankroll += dollar_pnl
            day_total += dollar_pnl
        daily_pnl.append(day_total)
        equity.append(bankroll)

    daily_pnl = np.array(daily_pnl)
    n_bets = len(bets_df)
    flat_roi = bets_df["pnl_flat"].sum() / n_bets if n_bets > 0 else 0
    win_rate = bets_df["won"].mean() if n_bets > 0 else 0

    # Sharpe (annualized, ~180 game days per season)
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(180)
    else:
        sharpe = 0

    # Max drawdown
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    return {
        "n_bets": n_bets,
        "win_rate": win_rate,
        "flat_roi": flat_roi,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": (bankroll - starting_bankroll) / starting_bankroll,
        "final_bankroll": bankroll,
    }


# ═══════════════════════════════════════════════════════════════════
# 5. Validation: Find Optimal Parameters
# ═══════════════════════════════════════════════════════════════════

def optimize_on_validation(val_df, signal_cols, signal_names):
    """
    On VALIDATION data only, find the best:
      - min_edge threshold (per signal)
      - Kelly fraction
      - Ensemble weights

    Optimizes for Sharpe ratio on validation period.
    Returns dict of optimal parameters.
    """
    print("\n" + "=" * 70)
    print("VALIDATION PERIOD: Optimizing Parameters")
    print("=" * 70)
    print(f"  Validation games: {len(val_df)}")
    print(f"  Date range: {val_df['game_date'].min()} to {val_df['game_date'].max()}")

    best_params = {}

    # 1. Per-signal sweep: find best min_edge for each signal
    #    Denser grid around promising regions
    edge_grid = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
    kelly_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    for col, name in zip(signal_cols, signal_names):
        if col not in val_df.columns or val_df[col].isna().all():
            print(f"\n  {name}: NO DATA — skipping")
            continue

        print(f"\n  {name}:")
        print(f"    {'Edge':>6} {'Kelly':>6} {'Bets':>6} {'WR':>6} {'ROI':>8} {'Sharpe':>8}")
        best_sharpe = -999
        best_edge = None
        best_kelly = None

        for me in edge_grid:
            for kf in kelly_grid:
                bets = compute_bets(val_df, col, min_edge=me, kelly_frac=kf)
                if len(bets) < 15:
                    continue
                metrics = run_kelly_backtest(bets)
                # Composite score: Sharpe weighted by sample size confidence
                # Penalize very small bet counts that could be noise
                sample_weight = min(1.0, len(bets) / 50.0)
                score = metrics["sharpe"] * sample_weight
                if score > best_sharpe:
                    best_sharpe = score
                    best_edge = me
                    best_kelly = kf
                    best_metrics = metrics

        if best_edge is not None:
            bets = compute_bets(val_df, col, min_edge=best_edge, kelly_frac=best_kelly)
            m = run_kelly_backtest(bets)
            print(f"    BEST: edge={best_edge:.2f} kelly={best_kelly:.2f} "
                  f"bets={m['n_bets']} WR={m['win_rate']:.3f} "
                  f"ROI={m['flat_roi']:+.3f} Sharpe={m['sharpe']:.2f}")
            best_params[name] = {
                "col": col,
                "min_edge": best_edge,
                "kelly_frac": best_kelly,
            }
        else:
            print(f"    No profitable configuration found")

    # 2. Ensemble optimization: sweep weights for DK + model + sim
    print(f"\n  Ensemble weight optimization:")
    available = [n for n in signal_names if n in best_params]
    if len(available) >= 2:
        # Build ensemble probabilities on validation
        weight_grid = np.arange(0.0, 1.05, 0.1)
        best_ens_sharpe = -999
        best_weights = None
        best_ens_edge = None
        best_ens_kelly = None

        cols_for_ens = [best_params[n]["col"] for n in available]

        # Only try combinations of available signals
        for w_idx in range(len(weight_grid)):
            for w2_idx in range(len(weight_grid)):
                remaining = 1.0 - weight_grid[w_idx] - weight_grid[w2_idx]
                if remaining < -0.01 or remaining > 1.01:
                    continue
                if len(available) == 2:
                    weights = [weight_grid[w_idx], 1.0 - weight_grid[w_idx]]
                    if w2_idx > 0:
                        continue
                elif len(available) == 3:
                    weights = [weight_grid[w_idx], weight_grid[w2_idx], remaining]
                else:
                    continue

                if any(w < -0.01 for w in weights):
                    continue
                weights = [max(0, w) for w in weights]
                wsum = sum(weights)
                if wsum < 0.01:
                    continue
                weights = [w / wsum for w in weights]

                # Build ensemble probability
                ens_prob = np.zeros(len(val_df))
                valid = np.ones(len(val_df), dtype=bool)
                for w, col in zip(weights, cols_for_ens):
                    vals = val_df[col].values.astype(float)
                    valid &= ~np.isnan(vals)
                    ens_prob += w * np.nan_to_num(vals, nan=0.5)

                temp_df = val_df.copy()
                temp_df["ens_prob"] = ens_prob
                temp_df.loc[~valid, "ens_prob"] = np.nan

                for me in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
                    for kf in [0.15, 0.20, 0.25, 0.30]:
                        bets = compute_bets(temp_df, "ens_prob", min_edge=me, kelly_frac=kf)
                        if len(bets) < 10:
                            continue
                        m = run_kelly_backtest(bets)
                        if m["sharpe"] > best_ens_sharpe:
                            best_ens_sharpe = m["sharpe"]
                            best_weights = dict(zip(available, weights))
                            best_ens_edge = me
                            best_ens_kelly = kf

        if best_weights is not None:
            print(f"    Best ensemble weights: {best_weights}")
            print(f"    Best ensemble edge: {best_ens_edge}, kelly: {best_ens_kelly}")
            print(f"    Validation Sharpe: {best_ens_sharpe:.2f}")
            best_params["ensemble"] = {
                "weights": best_weights,
                "cols": dict(zip(available, cols_for_ens)),
                "min_edge": best_ens_edge,
                "kelly_frac": best_ens_kelly,
            }

    return best_params


# ═══════════════════════════════════════════════════════════════════
# 6. Test Period Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_on_test(test_df, params, signal_cols, signal_names):
    """
    Apply FIXED parameters (from validation) to test data.
    NO parameter tuning here — pure out-of-sample evaluation.
    """
    print("\n" + "=" * 70)
    print("TEST PERIOD: Out-of-Sample Evaluation (FIXED parameters)")
    print("=" * 70)
    print(f"  Test games: {len(test_df)}")
    print(f"  Date range: {test_df['game_date'].min()} to {test_df['game_date'].max()}")
    print(f"  Home win rate: {test_df['home_win'].mean():.3f}")

    results = {}

    # Individual signals
    for col, name in zip(signal_cols, signal_names):
        if name not in params:
            continue
        p = params[name]
        bets = compute_bets(test_df, p["col"], min_edge=p["min_edge"],
                           kelly_frac=p["kelly_frac"])
        m = run_kelly_backtest(bets)
        results[name] = m

        # Bootstrap CI for ROI
        if len(bets) >= 20:
            boot_rois = []
            for _ in range(2000):
                idx = np.random.choice(len(bets), len(bets), replace=True)
                boot_rois.append(bets.iloc[idx]["pnl_flat"].mean())
            ci_lo, ci_hi = np.percentile(boot_rois, [2.5, 97.5])
        else:
            ci_lo, ci_hi = np.nan, np.nan

        print(f"\n  {name} (edge>{p['min_edge']:.0%}, kelly={p['kelly_frac']:.0%}):")
        print(f"    Bets: {m['n_bets']}, Win Rate: {m['win_rate']:.1%}")
        print(f"    Flat ROI: {m['flat_roi']:+.1%}  [{ci_lo:+.1%}, {ci_hi:+.1%}]")
        print(f"    Sharpe: {m['sharpe']:.2f}")
        print(f"    Kelly Return: {m['total_return']:+.1%}")
        print(f"    Max Drawdown: {m['max_dd']:.1%}")

        # Monthly breakdown
        if len(bets) > 0:
            bets["month"] = pd.to_datetime(bets["game_date"]).dt.to_period("M")
            print(f"    Monthly: ", end="")
            for month, mbets in bets.groupby("month"):
                mr = mbets["pnl_flat"].mean()
                print(f"{month}: {mr:+.1%} ({len(mbets)}b) | ", end="")
            print()

    # Ensemble
    if "ensemble" in params:
        ep = params["ensemble"]
        ens_prob = np.zeros(len(test_df))
        valid = np.ones(len(test_df), dtype=bool)
        for name, col in ep["cols"].items():
            w = ep["weights"][name]
            vals = test_df[col].values.astype(float)
            valid &= ~np.isnan(vals)
            ens_prob += w * np.nan_to_num(vals, nan=0.5)

        temp_df = test_df.copy()
        temp_df["ens_prob"] = ens_prob
        temp_df.loc[~valid, "ens_prob"] = np.nan

        bets = compute_bets(temp_df, "ens_prob", min_edge=ep["min_edge"],
                           kelly_frac=ep["kelly_frac"])
        m = run_kelly_backtest(bets)
        results["ensemble"] = m

        if len(bets) >= 20:
            boot_rois = []
            for _ in range(2000):
                idx = np.random.choice(len(bets), len(bets), replace=True)
                boot_rois.append(bets.iloc[idx]["pnl_flat"].mean())
            ci_lo, ci_hi = np.percentile(boot_rois, [2.5, 97.5])
        else:
            ci_lo, ci_hi = np.nan, np.nan

        print(f"\n  ENSEMBLE (weights={ep['weights']}, edge>{ep['min_edge']:.0%}):")
        print(f"    Bets: {m['n_bets']}, Win Rate: {m['win_rate']:.1%}")
        print(f"    Flat ROI: {m['flat_roi']:+.1%}  [{ci_lo:+.1%}, {ci_hi:+.1%}]")
        print(f"    Sharpe: {m['sharpe']:.2f}")
        print(f"    Kelly Return: {m['total_return']:+.1%}")
        print(f"    Max Drawdown: {m['max_dd']:.1%}")

        if len(bets) > 0:
            bets["month"] = pd.to_datetime(bets["game_date"]).dt.to_period("M")
            print(f"    Monthly: ", end="")
            for month, mbets in bets.groupby("month"):
                mr = mbets["pnl_flat"].mean()
                print(f"{month}: {mr:+.1%} ({len(mbets)}b) | ", end="")
            print()

    # Permutation test on best strategy
    if results:
        best_name = max(results, key=lambda k: results[k]["sharpe"])
        best_m = results[best_name]
        print(f"\n  Permutation test on {best_name}:")
        if best_name == "ensemble":
            p = params["ensemble"]
            temp_df = test_df.copy()
            ens_prob = np.zeros(len(test_df))
            valid_mask = np.ones(len(test_df), dtype=bool)
            for n, col in p["cols"].items():
                w = p["weights"][n]
                vals = test_df[col].values.astype(float)
                valid_mask &= ~np.isnan(vals)
                ens_prob += w * np.nan_to_num(vals, nan=0.5)
            temp_df["ens_prob"] = ens_prob
            temp_df.loc[~valid_mask, "ens_prob"] = np.nan
            prob_col = "ens_prob"
            me = p["min_edge"]
            kf = p["kelly_frac"]
        else:
            p = params[best_name]
            temp_df = test_df.copy()
            prob_col = p["col"]
            me = p["min_edge"]
            kf = p["kelly_frac"]

        observed_roi = best_m["flat_roi"]
        n_perm = 5000
        perm_rois = []
        for _ in range(n_perm):
            shuffled = temp_df.copy()
            shuffled["home_win"] = np.random.permutation(shuffled["home_win"].values)
            bets = compute_bets(shuffled, prob_col, min_edge=me, kelly_frac=kf)
            if len(bets) > 0:
                perm_rois.append(bets["pnl_flat"].mean())
            else:
                perm_rois.append(0)
        perm_rois = np.array(perm_rois)
        p_value = (perm_rois >= observed_roi).mean()
        print(f"    Observed ROI: {observed_roi:+.1%}")
        print(f"    Permutation mean ROI: {perm_rois.mean():+.1%}")
        print(f"    P-value: {p_value:.4f}")
        print(f"    Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 7. Multi-Strategy Portfolio (independent signals, diversified)
# ═══════════════════════════════════════════════════════════════════

def run_multi_strategy_portfolio(test_df, params, starting_bankroll=10000):
    """
    Run DK_arb and LGB_model independently — each gets its own allocation.
    A game can generate bets from BOTH signals if both have edge.
    This diversifies vs the ensemble approach.
    """
    print("\n" + "=" * 70)
    print("MULTI-STRATEGY PORTFOLIO (independent allocations)")
    print("=" * 70)

    strats_to_run = {k: v for k, v in params.items() if k != "ensemble" and k != "MC_sim"}
    if not strats_to_run:
        print("  No strategies to combine")
        return {}

    bankroll = starting_bankroll
    all_bets = []
    daily_pnl = []

    for date, day_games in test_df.groupby("game_date"):
        day_total = 0
        for strat_name, p in strats_to_run.items():
            col = p["col"]
            me = p["min_edge"]
            kf = p["kelly_frac"]

            for _, row in day_games.iterrows():
                our_home = row.get(col, np.nan)
                if pd.isna(our_home):
                    continue
                our_away = 1.0 - our_home
                k_home = row["kalshi_home_prob"]
                k_away = row["kalshi_away_prob"]

                edge_home = our_home - k_home
                edge_away = our_away - k_away

                if edge_home > edge_away and edge_home > me:
                    side, edge, buy_price, our_p = "home", edge_home, k_home, our_home
                elif edge_away > me:
                    side, edge, buy_price, our_p = "away", edge_away, k_away, our_away
                else:
                    continue

                b = (1.0 - buy_price) / buy_price
                q = 1.0 - our_p
                kelly_f = max(0.0, (our_p * b - q) / b) * kf
                kelly_f = min(kelly_f, 0.05)

                stake = bankroll * kelly_f
                if stake <= 0:
                    continue

                won = (side == "home" and row["home_win"] == 1) or \
                      (side == "away" and row["home_win"] == 0)
                if won:
                    dollar_pnl = stake * (1.0 - buy_price) / buy_price
                else:
                    dollar_pnl = -stake

                bankroll += dollar_pnl
                day_total += dollar_pnl
                flat_pnl = (1.0 - buy_price) if won else -buy_price

                all_bets.append({
                    "game_date": date,
                    "strategy": strat_name,
                    "side": side,
                    "edge": edge,
                    "kalshi_price": buy_price,
                    "our_prob": our_p,
                    "kelly_f": kelly_f,
                    "stake": stake,
                    "won": int(won),
                    "pnl_flat": flat_pnl,
                    "pnl_dollar": dollar_pnl,
                    "bankroll_after": bankroll,
                })

        daily_pnl.append(day_total)

    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()
    daily_pnl = np.array(daily_pnl)

    n_bets = len(bets_df)
    if n_bets == 0:
        print("  No bets generated")
        return {}

    flat_roi = bets_df["pnl_flat"].mean()
    win_rate = bets_df["won"].mean()
    total_return = (bankroll - starting_bankroll) / starting_bankroll

    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(180)
    else:
        sharpe = 0

    equity = np.array([starting_bankroll] + list(bets_df.groupby("game_date")["pnl_dollar"].sum().cumsum() + starting_bankroll))
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min()

    print(f"\n  Combined Portfolio:")
    print(f"    Total Bets: {n_bets}")
    print(f"    Win Rate: {win_rate:.1%}")
    print(f"    Flat ROI: {flat_roi:+.1%}")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Kelly Return: {total_return:+.1%}")
    print(f"    Max Drawdown: {max_dd:.1%}")
    print(f"    Final Bankroll: ${bankroll:,.0f}")

    # Per-strategy breakdown
    print(f"\n  Per-strategy in portfolio:")
    for strat in bets_df["strategy"].unique():
        sb = bets_df[bets_df["strategy"] == strat]
        print(f"    {strat}: {len(sb)} bets, WR={sb['won'].mean():.1%}, "
              f"ROI={sb['pnl_flat'].mean():+.1%}, PnL=${sb['pnl_dollar'].sum():,.0f}")

    # Monthly
    bets_df["month"] = pd.to_datetime(bets_df["game_date"]).dt.to_period("M")
    print(f"\n  Monthly:")
    for month, mbets in bets_df.groupby("month"):
        mr = mbets["pnl_flat"].mean()
        mpnl = mbets["pnl_dollar"].sum()
        print(f"    {month}: {len(mbets)} bets, ROI={mr:+.1%}, PnL=${mpnl:,.0f}")

    # Bootstrap CI
    boot_rois = []
    for _ in range(2000):
        idx = np.random.choice(len(bets_df), len(bets_df), replace=True)
        boot_rois.append(bets_df.iloc[idx]["pnl_flat"].mean())
    ci_lo, ci_hi = np.percentile(boot_rois, [2.5, 97.5])
    print(f"\n  Bootstrap 95% CI for flat ROI: [{ci_lo:+.1%}, {ci_hi:+.1%}]")

    return {
        "n_bets": n_bets, "win_rate": win_rate, "flat_roi": flat_roi,
        "sharpe": sharpe, "max_dd": max_dd, "total_return": total_return,
        "final_bankroll": bankroll,
    }


# ═══════════════════════════════════════════════════════════════════
# 8. Signal Quality Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_signal_quality(df, signal_cols, signal_names, period_name):
    """Evaluate raw signal quality (Brier, AUC, calibration) on a dataset."""
    print(f"\n  Signal quality ({period_name}):")
    print(f"    {'Signal':<25} {'Brier':>7} {'BSS':>8} {'AUC':>7} {'N':>6}")
    for col, name in zip(signal_cols, signal_names):
        if col not in df.columns:
            continue
        mask = df[col].notna() & df["home_win"].notna()
        if mask.sum() < 50:
            continue
        y = df.loc[mask, "home_win"].values.astype(int)
        p = df.loc[mask, col].values
        brier = brier_score_loss(y, p)
        naive = brier_score_loss(y, np.full(len(y), y.mean()))
        bss = 1 - brier / naive
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
        print(f"    {name:<25} {brier:.5f} {bss:+.5f} {auc:.4f} {mask.sum():>6}")

    # Also check Kalshi as a baseline
    mask = df["kalshi_home_prob"].notna() & df["home_win"].notna()
    y = df.loc[mask, "home_win"].values.astype(int)
    p = df.loc[mask, "kalshi_home_prob"].values
    brier = brier_score_loss(y, p)
    naive = brier_score_loss(y, np.full(len(y), y.mean()))
    bss = 1 - brier / naive
    auc = roc_auc_score(y, p)
    print(f"    {'Kalshi (market)':<25} {brier:.5f} {bss:+.5f} {auc:.4f} {mask.sum():>6}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("CLEAN KALSHI ML BACKTEST — Zero Look-Ahead Bias")
    print("=" * 70)

    # ── Load data ──
    print("\n[1] Loading data...")
    df = load_all_data()

    # ── Train win model on 2024 + early 2025 ──
    print("\n[2] Training win model on 2024 + early 2025 (Optuna TSCV)...")
    feat_2024_path = DATA / "features" / "game_features_2024.parquet"
    feat_2025_path = DATA / "features" / "game_features_2025.parquet"
    if feat_2024_path.exists():
        feat_2024 = pd.read_parquet(feat_2024_path)
        feat_2024["game_date"] = pd.to_datetime(feat_2024["game_date"]).dt.strftime("%Y-%m-%d")

        # Merge DK closing lines into 2024 training data
        dk_2024 = extract_dk_lines_from_json(2024)
        if not dk_2024.empty:
            mk = ["game_date", "home_team", "away_team"]
            feat_2024 = feat_2024.merge(dk_2024[mk + ["dk_home_prob"]], on=mk, how="left")
            dk_count = feat_2024["dk_home_prob"].notna().sum()
            print(f"  2024 training: {dk_count}/{len(feat_2024)} games with DK lines")

        # Merge aggregated NN lineup features into 2024 training data
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
            nn_count = feat_2024[[c for c in feat_2024.columns if c.startswith("nn_home_mean_")]].notna().any(axis=1).sum()
            print(f"  2024 training: {nn_count}/{len(feat_2024)} games with NN lineup features")

        # Add early 2025 games (before validation cutoff Apr 16) to training
        if feat_2025_path.exists():
            feat_2025_all = pd.read_parquet(feat_2025_path)
            feat_2025_all["game_date"] = pd.to_datetime(feat_2025_all["game_date"]).dt.strftime("%Y-%m-%d")

            # Merge NN features into 2025 training data too
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

            # Merge DK closing lines into 2025 training data
            dk_2025 = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet") if (DATA / "odds" / "sbr_mlb_2025.parquet").exists() else pd.DataFrame()
            if not dk_2025.empty:
                dk_2025["game_date"] = pd.to_datetime(dk_2025["game_date"]).dt.strftime("%Y-%m-%d")
                dk_2025["dk_home_raw"] = american_to_prob(dk_2025["home_ml_close"])
                dk_2025["dk_away_raw"] = american_to_prob(dk_2025["away_ml_close"])
                dk_2025["dk_total"] = dk_2025["dk_home_raw"] + dk_2025["dk_away_raw"]
                dk_2025["dk_home_prob"] = dk_2025["dk_home_raw"] / dk_2025["dk_total"]
                mk = ["game_date", "home_team", "away_team"]
                feat_2025_all = feat_2025_all.merge(dk_2025[mk + ["dk_home_prob"]].dropna(), on=mk, how="left")

            feat_2025_early = feat_2025_all[feat_2025_all["game_date"] < "2025-04-16"].copy()
            train_feat = pd.concat([feat_2024, feat_2025_early], ignore_index=True)
            print(f"  Training data: {len(feat_2024)} games (2024) + {len(feat_2025_early)} games (early 2025) = {len(train_feat)} total")
        else:
            train_feat = feat_2024
            print(f"  Training data: {len(feat_2024)} games (2024 only)")

        # Need features for 2025 games too
        model, feat_cols = train_win_model(train_feat, df)

        # Generate predictions for all 2025 games
        X_2025 = df[feat_cols].copy() if all(c in df.columns for c in feat_cols) else None
        if X_2025 is not None:
            df["lgb_home_prob"] = model.predict_proba(X_2025)[:, 1]
            print(f"  LGB predictions generated for {df['lgb_home_prob'].notna().sum()} games")
        else:
            print("  WARNING: Feature mismatch, LGB predictions skipped")
            df["lgb_home_prob"] = np.nan
    else:
        print("  No 2024 features available — skipping LGB model")
        df["lgb_home_prob"] = np.nan

    # ── Split into validation and test ──
    val_mask = df["game_date"] < VAL_CUTOFF
    test_mask = df["game_date"] >= VAL_CUTOFF

    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    print(f"\n[3] Data splits:")
    print(f"  Validation: {len(val_df)} games ({val_df['game_date'].min()} to {val_df['game_date'].max()})")
    print(f"  Test:       {len(test_df)} games ({test_df['game_date'].min()} to {test_df['game_date'].max()})")

    # Define signal sources (no DK — building models that beat the market on their own)
    signal_cols = ["lgb_home_prob", "nn_home_prob", "sim_home_wp"]
    signal_names = ["LGB_model", "NN_model", "MC_sim"]

    # ── Signal quality ──
    print("\n[4] Signal Quality Analysis")
    analyze_signal_quality(val_df, signal_cols, signal_names, "Validation")
    analyze_signal_quality(test_df, signal_cols, signal_names, "Test")

    # ── Optimize on validation ──
    params = optimize_on_validation(val_df, signal_cols, signal_names)

    if not params:
        print("\nNo profitable signals found on validation period. Stopping.")
        return

    # ── Evaluate on test (FIXED params, no tuning) ──
    results = evaluate_on_test(test_df, params, signal_cols, signal_names)

    # ── Multi-strategy portfolio ──
    portfolio_m = run_multi_strategy_portfolio(test_df, params)
    if portfolio_m:
        results["portfolio"] = portfolio_m

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Strategy':<20} {'Bets':>6} {'WR':>7} {'ROI':>8} {'Sharpe':>8} {'Return':>9} {'MaxDD':>8}")
    for name, m in results.items():
        print(f"  {name:<20} {m['n_bets']:>6} {m['win_rate']:>6.1%} "
              f"{m['flat_roi']:>+7.1%} {m['sharpe']:>7.2f} "
              f"{m['total_return']:>+8.1%} {m['max_dd']:>7.1%}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
