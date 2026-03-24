#!/usr/bin/env python3
"""
Generate pregame win probabilities for today's (or any date's) games.

Usage:
    python src/predict.py                     # today's games
    python src/predict.py --date 2024-09-15   # specific date
    python src/predict.py --backtest 2024     # full season backtest with rolling training
"""

import argparse
import pickle
from datetime import date, timedelta
from pathlib import Path

import httpx
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import (
    _preindex_xrv, _get_before, _sp_features_fast, _bp_features_fast,
    _hit_features_fast, _def_features_fast, _precompute_pitcher_bases,
    compute_matchup_xrv, compute_bullpen_matchup_xrv, compute_park_factors,
    _recent_winpct, _load_hand_models, _compute_hand_matchup,
    _compute_hand_arsenal_matchup, compute_arsenal_matchup_xrv,
    compute_bullpen_arsenal_matchup_xrv, _compute_pitcher_arsenal_live,
    _standardize_arsenal, _filter_competitive,
    _load_team_oaa, _compute_team_priors,
)

from utils import DATA_DIR, XRV_DIR, GAMES_DIR, FEATURES_DIR, MODEL_DIR, OAA_DIR
MLB_API = "https://statsapi.mlb.com/api/v1"

# Same feature set as win_model.py
DIFF_FEATURES = [
    "diff_sp_xrv_mean",
    "diff_sp_k_rate",
    "diff_sp_bb_rate",
    "diff_sp_avg_velo",
    "diff_sp_rest_days",
    "diff_sp_overperf",
    "diff_sp_overperf_recent",
    "diff_bp_xrv_mean",
    "diff_bp_fatigue_score",
    "diff_bp_matchup_xrv_mean",
    "diff_hit_xrv_mean",
    "diff_hit_xrv_contact",
    "diff_hit_k_rate",
    "diff_def_xrv_delta",
    "diff_matchup_xrv_mean",
    "diff_matchup_xrv_sum",
    "diff_arsenal_matchup_xrv_mean",
    "diff_arsenal_matchup_xrv_sum",
    "diff_bp_arsenal_matchup_xrv_mean",
    "diff_platoon_pct",
    "diff_recent_form",
    "diff_sp_xrv_vs_lineup",
    # SP trend features
    "diff_sp_velo_trend",
    "diff_sp_spin_trend",
    "diff_sp_xrv_trend",
    # OAA defense
    "diff_oaa_rate",
    # Team strength prior
    "diff_team_prior",
    # Context-aware SP xRV
    "diff_sp_context_xrv",
]

RAW_FEATURES = [
    "park_factor",
    "home_sp_pitch_mix_entropy",
    "away_sp_pitch_mix_entropy",
    "home_sp_n_pitches",
    "away_sp_n_pitches",
    "home_matchup_n_known",
    "away_matchup_n_known",
    "home_arsenal_matchup_n_known",
    "away_arsenal_matchup_n_known",
    "temperature",
    "wind_speed",
    "is_dome",
    "wind_out",
    "wind_in",
]

ALL_FEATURES = DIFF_FEATURES + RAW_FEATURES


def load_training_data(exclude_year: int = None) -> pd.DataFrame:
    """Load all available feature files, optionally excluding a year."""
    frames = []
    for path in sorted(FEATURES_DIR.glob("game_features_*.parquet")):
        year = int(path.stem.split("_")[-1])
        if exclude_year and year >= exclude_year:
            continue
        df = pd.read_parquet(path)
        df["season"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No feature files found for training")
    return pd.concat(frames, ignore_index=True)


def train_ensemble(train_df: pd.DataFrame):
    """Train LR + XGB ensemble on training data."""
    available = [f for f in ALL_FEATURES if f in train_df.columns]
    X = train_df[available].copy()
    y = train_df["home_win"].values

    # LR
    scaler = StandardScaler()
    X_lr = scaler.fit_transform(X.fillna(0))
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(X_lr, y)

    # XGB
    xgb_model = None
    if HAS_XGB:
        dtrain = xgb.DMatrix(X, label=y)
        params = {
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
        xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    return lr, scaler, xgb_model, available


def predict_games(lr, scaler, xgb_model, features, games_df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for a set of games."""
    available = [f for f in features if f in games_df.columns]
    X = games_df[available].copy()

    # LR predictions
    X_lr = scaler.transform(X.fillna(0))
    lr_probs = lr.predict_proba(X_lr)[:, 1]

    # XGB predictions
    if xgb_model and HAS_XGB:
        dtest = xgb.DMatrix(X)
        xgb_probs = xgb_model.predict(dtest)
        ensemble_probs = 0.5 * lr_probs + 0.5 * xgb_probs
    else:
        xgb_probs = lr_probs
        ensemble_probs = lr_probs

    results = games_df[["game_pk", "game_date", "home_team", "away_team"]].copy()
    if "home_win" in games_df.columns:
        results["home_win"] = games_df["home_win"]
    if "home_score" in games_df.columns:
        results["home_score"] = games_df["home_score"]
    if "away_score" in games_df.columns:
        results["away_score"] = games_df["away_score"]

    results["home_win_prob_lr"] = lr_probs
    results["home_win_prob_xgb"] = xgb_probs
    results["home_win_prob"] = ensemble_probs
    results["away_win_prob"] = 1 - ensemble_probs

    return results


# ──────────────────────────────────────────────────────────────
# Live prediction: fetch today's games and build features on the fly
# ──────────────────────────────────────────────────────────────

def fetch_todays_games(client: httpx.Client, target_date: str) -> list[dict]:
    """Fetch scheduled games for a date from MLB Stats API."""
    resp = client.get(
        f"{MLB_API}/schedule",
        params={
            "sportId": 1,
            "date": target_date,
            "hydrate": "probablePitcher,venue,team,linescore,weather",
            "gameType": "R",
        },
    )
    resp.raise_for_status()
    data = resp.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            home_team = home.get("team", {})
            away_team = away.get("team", {})
            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})
            venue = g.get("venue", {})
            weather = g.get("weather", {})

            game = {
                "game_pk": g.get("gamePk"),
                "game_date": g.get("officialDate", target_date),
                "status": status,
                "home_team": home_team.get("abbreviation", ""),
                "away_team": away_team.get("abbreviation", ""),
                "home_team_name": home_team.get("name", ""),
                "away_team_name": away_team.get("name", ""),
                "home_sp_id": home_pitcher.get("id"),
                "home_sp_name": home_pitcher.get("fullName", ""),
                "away_sp_id": away_pitcher.get("id"),
                "away_sp_name": away_pitcher.get("fullName", ""),
                "venue_name": venue.get("name", ""),
                "game_time": g.get("gameDate", ""),
                "weather": weather,
            }

            # If game is final, include score
            if status == "Final":
                game["home_score"] = home.get("score", 0)
                game["away_score"] = away.get("score", 0)
                game["home_win"] = int(home.get("score", 0) > away.get("score", 0))

            games.append(game)

    return games


def fetch_lineup(client: httpx.Client, game_pk: int) -> dict:
    """Fetch starting lineup from boxscore endpoint."""
    try:
        resp = client.get(f"{MLB_API}/game/{game_pk}/boxscore")
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {"home": [], "away": []}

    result = {}
    for side in ("home", "away"):
        team_data = data.get("teams", {}).get(side, {})
        batting_order = team_data.get("battingOrder", [])
        players = team_data.get("players", {})

        lineup = []
        for pid in batting_order:
            key = f"ID{pid}"
            p = players.get(key, {})
            bat_side = p.get("batSide", {}).get("code", "R")
            lineup.append((int(pid), bat_side))

        result[side] = lineup

    return result


def build_live_features(
    games: list[dict],
    target_date: str,
    client: httpx.Client,
) -> pd.DataFrame:
    """Build feature vectors for today's games using available data."""
    # Determine which season we're in
    year = int(target_date[:4])

    # Load xRV data: current year + prior year
    xrv_frames = []
    for yr in [year - 1, year]:
        xrv_path = XRV_DIR / f"statcast_xrv_{yr}.parquet"
        if xrv_path.exists():
            xrv_frames.append(pd.read_parquet(xrv_path))
            print(f"  Loaded xRV for {yr}: {len(xrv_frames[-1]):,} pitches")
    if not xrv_frames:
        raise FileNotFoundError("No xRV data available")
    xrv = pd.concat(xrv_frames, ignore_index=True)
    xrv["game_date"] = xrv["game_date"].astype(str)

    print("  Pre-indexing xRV data...")
    idx = _preindex_xrv(xrv)

    # Load hand-specific matchup models (from prior season)
    matchup_models = _load_hand_models("matchup_model", year - 1)
    if matchup_models:
        print(f"  Matchup models loaded for hands: {list(matchup_models.keys())}")

    # Load hand-specific arsenal matchup models (from prior season)
    arsenal_models = _load_hand_models("arsenal_matchup", year - 1)
    if arsenal_models:
        print(f"  Arsenal models loaded for hands: {list(arsenal_models.keys())}")

    # Park factors from prior season
    prior_xrv_path = XRV_DIR / f"statcast_xrv_{year - 1}.parquet"
    if prior_xrv_path.exists():
        prior_xrv = pd.read_parquet(prior_xrv_path)
        park_factors = compute_park_factors(prior_xrv, year - 1)
    else:
        park_factors = {}

    # Load team OAA from prior season
    team_oaa = _load_team_oaa(year - 1)
    if team_oaa:
        print(f"  Team OAA from {year - 1}: {len(team_oaa)} teams")

    # Compute team priors from prior season win%
    team_priors = _compute_team_priors(year - 1)
    if team_priors:
        print(f"  Team priors from {year - 1}: {len(team_priors)} teams")

    # Compute recent form from game results
    games_path = GAMES_DIR / f"games_{year}.parquet"
    team_history = {}
    if games_path.exists():
        season_games = pd.read_parquet(games_path)
        season_games = season_games[
            (season_games["game_type"] == "R") &
            (season_games["game_date"] < target_date)
        ].sort_values("game_date")
        for _, g in season_games.iterrows():
            ht = g["home_team_abbr"]
            at = g["away_team_abbr"]
            hw = g["home_win"]
            team_history.setdefault(ht, []).append(hw)
            team_history.setdefault(at, []).append(1 - hw)

    nan_sp = {f"sp_{k}": np.nan for k in
              ["xrv_mean", "xrv_std", "n_pitches", "k_rate", "bb_rate", "avg_velo",
               "pitch_mix_entropy", "xrv_vs_L", "xrv_vs_R", "rest_days",
               "overperf", "overperf_recent"]}
    nan_matchup = {"matchup_xrv_mean": np.nan, "matchup_xrv_sum": np.nan,
                   "matchup_n_hitters": 0, "matchup_n_known": 0}

    feature_rows = []
    for game in games:
        gdate = game["game_date"]
        home_team = game["home_team"]
        away_team = game["away_team"]
        home_sp = game.get("home_sp_id")
        away_sp = game.get("away_sp_id")
        game_pk = game["game_pk"]

        row = {
            "game_pk": game_pk,
            "game_date": gdate,
            "home_team": home_team,
            "away_team": away_team,
        }
        # Copy over any result data
        for k in ("home_win", "home_score", "away_score"):
            if k in game:
                row[k] = game[k]

        # --- SP features ---
        for side, sp_id in [("home", home_sp), ("away", away_sp)]:
            if sp_id and int(sp_id) in idx["pitcher"]:
                stats = _sp_features_fast(idx["pitcher"][int(sp_id)], gdate, 2000)
                for k, v in stats.items():
                    row[f"{side}_{k}"] = v
            else:
                for k, v in nan_sp.items():
                    row[f"{side}_{k}"] = v

        # --- Fetch lineup for matchup features ---
        lineup = fetch_lineup(client, game_pk)

        # --- SP matchup: lineup vs opposing SP (hand-specific) ---
        nan_arsenal = {"arsenal_matchup_xrv_mean": np.nan, "arsenal_matchup_xrv_sum": np.nan,
                       "arsenal_matchup_n_hitters": 0, "arsenal_matchup_n_known": 0}

        for atk_side, def_sp, def_side in [("home", away_sp, "away"), ("away", home_sp, "home")]:
            lu = lineup.get(atk_side, [])
            if matchup_models and lu and def_sp and int(def_sp) in idx["pitcher"]:
                hitter_ids = [h[0] for h in lu]
                hitter_hands = [h[1] for h in lu]
                matchup = _compute_hand_matchup(
                    matchup_models, idx["pitcher"][int(def_sp)], int(def_sp),
                    gdate, 2000, hitter_ids, hitter_hands, compute_matchup_xrv)
                for k, v in matchup.items():
                    row[f"{atk_side}_{k}"] = v
            else:
                for k, v in nan_matchup.items():
                    row[f"{atk_side}_{k}"] = v

            # Arsenal matchup
            if arsenal_models and lu and def_sp and int(def_sp) in idx["pitcher"]:
                hitter_ids = [h[0] for h in lu]
                hitter_hands = [h[1] for h in lu]
                a_matchup = _compute_hand_arsenal_matchup(
                    arsenal_models, idx["pitcher"][int(def_sp)], int(def_sp),
                    gdate, 2000, hitter_ids, hitter_hands)
                for k, v in a_matchup.items():
                    row[f"{atk_side}_{k}"] = v
            else:
                for k, v in nan_arsenal.items():
                    row[f"{atk_side}_{k}"] = v

        # --- Bullpen features ---
        for side in ["home", "away"]:
            team = home_team if side == "home" else away_team
            if team in idx["pitching_team"]:
                bp = _bp_features_fast(idx["pitching_team"][team], gdate)
                for k, v in bp.items():
                    row[f"{side}_{k}"] = v
            else:
                for k in ["bp_xrv_mean", "bp_xrv_std", "bp_recent_ip", "bp_fatigue_score"]:
                    row[f"{side}_{k}"] = np.nan

        # --- Bullpen matchup (hand-specific) ---
        bp_matchup_artifacts = matchup_models.get("L") or matchup_models.get("R") if matchup_models else None
        bp_arsenal_artifacts = arsenal_models.get("L") or arsenal_models.get("R") if arsenal_models else None
        if bp_matchup_artifacts:
            for def_side, atk_side in [("home", "away"), ("away", "home")]:
                def_team = home_team if def_side == "home" else away_team
                lu = lineup.get(atk_side, [])
                if lu:
                    hids = [h[0] for h in lu]
                    hhands = [h[1] for h in lu]
                    bp_m = compute_bullpen_matchup_xrv(
                        bp_matchup_artifacts, idx, def_team, gdate, hids, hhands)
                    for k, v in bp_m.items():
                        row[f"{def_side}_{k}"] = v
                else:
                    row[f"{def_side}_bp_matchup_xrv_mean"] = np.nan
                    row[f"{def_side}_bp_matchup_n_relievers"] = 0
        else:
            for side in ["home", "away"]:
                row[f"{side}_bp_matchup_xrv_mean"] = np.nan
                row[f"{side}_bp_matchup_n_relievers"] = 0

        # --- Bullpen arsenal matchup ---
        if bp_arsenal_artifacts:
            for def_side, atk_side in [("home", "away"), ("away", "home")]:
                def_team = home_team if def_side == "home" else away_team
                lu = lineup.get(atk_side, [])
                if lu:
                    bp_a = compute_bullpen_arsenal_matchup_xrv(
                        bp_arsenal_artifacts, idx, def_team, gdate,
                        [h[0] for h in lu], [h[1] for h in lu])
                    for k, v in bp_a.items():
                        row[f"{def_side}_{k}"] = v
                else:
                    row[f"{def_side}_bp_arsenal_matchup_xrv_mean"] = np.nan
                    row[f"{def_side}_bp_arsenal_matchup_n_relievers"] = 0
        else:
            for side in ["home", "away"]:
                row[f"{side}_bp_arsenal_matchup_xrv_mean"] = np.nan
                row[f"{side}_bp_arsenal_matchup_n_relievers"] = 0

        # --- Team hitting ---
        for side in ["home", "away"]:
            team = home_team if side == "home" else away_team
            if team in idx["batting_team"]:
                hit = _hit_features_fast(idx["batting_team"][team], gdate)
                for k, v in hit.items():
                    row[f"{side}_{k}"] = v
            else:
                for k in ["hit_xrv_mean", "hit_xrv_contact", "hit_k_rate"]:
                    row[f"{side}_{k}"] = np.nan

        # --- Defense ---
        for side in ["home", "away"]:
            team = home_team if side == "home" else away_team
            if team in idx["pitching_team"]:
                d = _def_features_fast(idx["pitching_team"][team], gdate)
                for k, v in d.items():
                    row[f"{side}_{k}"] = v
            else:
                row[f"{side}_def_xrv_delta"] = np.nan

        # --- OAA defense ---
        row["home_oaa_rate"] = team_oaa.get(home_team, 0.0)
        row["away_oaa_rate"] = team_oaa.get(away_team, 0.0)

        # --- Team strength prior ---
        row["home_team_prior"] = team_priors.get(home_team, 0.5)
        row["away_team_prior"] = team_priors.get(away_team, 0.5)

        # --- Park factor ---
        row["park_factor"] = park_factors.get(home_team, 1.0)

        # --- Platoon advantage ---
        for atk_side, def_sp in [("home", away_sp), ("away", home_sp)]:
            lu = lineup.get(atk_side, [])
            if lu and def_sp and int(def_sp) in idx["pitcher"]:
                sp_df = idx["pitcher"][int(def_sp)]
                sp_throws = sp_df["p_throws"].iloc[-1] if "p_throws" in sp_df.columns and len(sp_df) > 0 else None
                if sp_throws:
                    platoon_count = sum(1 for _, hand in lu
                                       if (hand == "L" and sp_throws == "R")
                                       or (hand == "R" and sp_throws == "L"))
                    row[f"{atk_side}_platoon_pct"] = platoon_count / len(lu)

                    # Handedness-weighted SP xRV
                    def_side = "away" if atk_side == "home" else "home"
                    xrv_vs_L = row.get(f"{def_side}_sp_xrv_vs_L", np.nan)
                    xrv_vs_R = row.get(f"{def_side}_sp_xrv_vs_R", np.nan)
                    if not np.isnan(xrv_vs_L):
                        n_L = sum(1 for _, h in lu if h == "L")
                        n_R = len(lu) - n_L
                        row[f"{def_side}_sp_xrv_vs_lineup"] = (
                            n_L * xrv_vs_L + n_R * xrv_vs_R
                        ) / len(lu) if len(lu) > 0 else np.nan
                    else:
                        row[f"{def_side}_sp_xrv_vs_lineup"] = np.nan
                else:
                    row[f"{atk_side}_platoon_pct"] = np.nan
                    def_side = "away" if atk_side == "home" else "home"
                    row[f"{def_side}_sp_xrv_vs_lineup"] = np.nan
            else:
                row[f"{atk_side}_platoon_pct"] = np.nan
                def_side = "away" if atk_side == "home" else "home"
                row[f"{def_side}_sp_xrv_vs_lineup"] = np.nan

        # --- Recent form ---
        row["home_recent_form"] = _recent_winpct(team_history.get(home_team, []), 10)
        row["away_recent_form"] = _recent_winpct(team_history.get(away_team, []), 10)

        # --- Weather ---
        weather = game.get("weather", {})
        if weather:
            row["temperature"] = float(weather.get("temp", 0)) if weather.get("temp") else np.nan
            wind_str = weather.get("wind", "")
            if wind_str:
                parts = wind_str.split(",")
                try:
                    row["wind_speed"] = float(parts[0].strip().lower().replace("mph", "").strip())
                except (ValueError, IndexError):
                    row["wind_speed"] = 0.0
                wind_dir = parts[1].strip().lower() if len(parts) > 1 else ""
                row["wind_out"] = int("out" in wind_dir)
                row["wind_in"] = int("in" in wind_dir)
            else:
                row["wind_speed"] = 0.0
                row["wind_out"] = 0
                row["wind_in"] = 0
            venue = game.get("venue_name", "")
            row["is_dome"] = int("dome" in venue.lower() or "roof" in weather.get("condition", "").lower())
        else:
            row["temperature"] = np.nan
            row["wind_speed"] = np.nan
            row["is_dome"] = np.nan
            row["wind_out"] = np.nan
            row["wind_in"] = np.nan

        row["is_home"] = 1

        # Store SP names for display
        row["home_sp_name"] = game.get("home_sp_name", "TBD")
        row["away_sp_name"] = game.get("away_sp_name", "TBD")
        row["game_time"] = game.get("game_time", "")
        row["status"] = game.get("status", "")

        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)

    # Compute differentials
    diff_cols = [
        ("sp_xrv_mean", -1), ("sp_k_rate", 1), ("sp_bb_rate", -1),
        ("sp_avg_velo", -1), ("sp_rest_days", 1),
        ("sp_overperf", -1), ("sp_overperf_recent", -1),
        ("bp_xrv_mean", -1), ("bp_fatigue_score", -1),
        ("bp_matchup_xrv_mean", -1),
        ("hit_xrv_mean", 1), ("hit_xrv_contact", 1), ("hit_k_rate", -1),
        ("def_xrv_delta", -1),
        ("matchup_xrv_mean", 1), ("matchup_xrv_sum", 1),
        ("arsenal_matchup_xrv_mean", 1), ("arsenal_matchup_xrv_sum", 1),
        ("bp_arsenal_matchup_xrv_mean", -1),
        ("platoon_pct", 1), ("recent_form", 1),
        # New features
        ("sp_velo_trend", -1), ("sp_spin_trend", -1), ("sp_xrv_trend", -1),
        ("oaa_rate", 1), ("team_prior", 1),
    ]
    for col, sign in diff_cols:
        hc = f"home_{col}"
        ac = f"away_{col}"
        if hc in features_df.columns and ac in features_df.columns:
            features_df[f"diff_{col}"] = sign * (features_df[hc] - features_df[ac])

    if "home_sp_xrv_vs_lineup" in features_df.columns and "away_sp_xrv_vs_lineup" in features_df.columns:
        features_df["diff_sp_xrv_vs_lineup"] = -(
            features_df["home_sp_xrv_vs_lineup"] - features_df["away_sp_xrv_vs_lineup"]
        )

    # Context-aware SP xRV: home SP uses home split, away SP uses away split
    if "home_sp_home_xrv" in features_df.columns and "away_sp_away_xrv" in features_df.columns:
        features_df["diff_sp_context_xrv"] = -(
            features_df["home_sp_home_xrv"] - features_df["away_sp_away_xrv"]
        )

    return features_df


def predict_date(target_date: str):
    """Predict games for a specific date."""
    print(f"\n{'='*60}")
    print(f"MLB Predictions for {target_date}")
    print(f"{'='*60}")

    # Train on all available data
    print("\nTraining model on historical data...")
    train_df = load_training_data()
    print(f"  {len(train_df):,} training games")
    lr, scaler, xgb_model, features = train_ensemble(train_df)

    # Fetch today's games
    print(f"\nFetching games for {target_date}...")
    with httpx.Client(timeout=15.0) as client:
        games = fetch_todays_games(client, target_date)
        if not games:
            print("  No games found!")
            return

        print(f"  Found {len(games)} games")

        # Build features
        print("\nBuilding features...")
        features_df = build_live_features(games, target_date, client)

    # Generate predictions
    results = predict_games(lr, scaler, xgb_model, features, features_df)

    # Merge SP names back in
    if "home_sp_name" in features_df.columns:
        results["home_sp_name"] = features_df["home_sp_name"].values
        results["away_sp_name"] = features_df["away_sp_name"].values
    if "game_time" in features_df.columns:
        results["game_time"] = features_df["game_time"].values
    if "status" in features_df.columns:
        results["status"] = features_df["status"].values

    # Display
    print(f"\n{'='*60}")
    print(f"  {'Matchup':<35s} {'Home%':>6s} {'Away%':>6s} {'Pick':>6s}")
    print(f"  {'-'*55}")

    for _, r in results.iterrows():
        away_sp = r.get("away_sp_name", "TBD")
        home_sp = r.get("home_sp_name", "TBD")
        matchup = f"{r['away_team']} @ {r['home_team']}"
        pitchers = f"  {away_sp} vs {home_sp}"

        pick_team = r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"]
        pick_prob = max(r["home_win_prob"], r["away_win_prob"])

        result_str = ""
        if "home_win" in r and pd.notna(r.get("home_win")):
            actual = r["home_team"] if r["home_win"] == 1 else r["away_team"]
            correct = pick_team == actual
            result_str = f"  {'W' if correct else 'L'} ({actual} won)"

        print(f"  {matchup:<35s} {r['home_win_prob']:>5.1%} {r['away_win_prob']:>5.1%} {pick_team:>5s}{result_str}")
        print(f"  {pitchers}")
        print()

    # If we have results, show accuracy
    if "home_win" in results.columns and results["home_win"].notna().all():
        picks_correct = results.apply(
            lambda r: (r["home_win_prob"] >= 0.5) == (r["home_win"] == 1), axis=1)
        print(f"  Record: {picks_correct.sum()}/{len(picks_correct)} "
              f"({picks_correct.mean():.1%})")

    return results


def backtest_season(year: int):
    """Full season backtest with predictions for every game."""
    print(f"\n{'='*60}")
    print(f"Backtesting {year}")
    print(f"{'='*60}")

    train_df = load_training_data(exclude_year=year)
    print(f"  Training on {len(train_df):,} games from prior seasons")

    lr, scaler, xgb_model, features = train_ensemble(train_df)

    test_path = FEATURES_DIR / f"game_features_{year}.parquet"
    if not test_path.exists():
        print(f"  No features for {year}")
        return None
    test_df = pd.read_parquet(test_path)
    test_df["season"] = year
    print(f"  Predicting {len(test_df)} games")

    results = predict_games(lr, scaler, xgb_model, features, test_df)

    # Evaluate
    y_true = results["home_win"].values
    for model_name, col in [("LR", "home_win_prob_lr"), ("XGB", "home_win_prob_xgb"), ("Ensemble", "home_win_prob")]:
        probs = results[col].values
        ll = log_loss(y_true, probs)
        bs = brier_score_loss(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        print(f"  {model_name}: log_loss={ll:.4f}, brier={bs:.4f}, AUC={auc:.4f}")

    # Sample predictions
    print(f"\n  Sample predictions (first 10 games):")
    for _, r in results.head(10).iterrows():
        marker = ""
        if "home_win" in r:
            actual = "W" if r["home_win"] == 1 else "L"
            correct = (r["home_win_prob"] > 0.5) == (r["home_win"] == 1)
            marker = f" [{actual}] {'Y' if correct else 'X'}"
        print(f"    {r['game_date']} {r['away_team']:3s} @ {r['home_team']:3s}: "
              f"Home {r['home_win_prob']:.1%} / Away {r['away_win_prob']:.1%}{marker}")

    # Betting analysis
    print(f"\n  Betting Analysis (vs implied 50/50 baseline):")
    for threshold in [0.52, 0.55, 0.58, 0.60]:
        strong = results[results["home_win_prob"].apply(lambda p: max(p, 1-p)) >= threshold]
        if len(strong) == 0:
            continue
        picks = strong.apply(
            lambda r: r["home_win"] == 1 if r["home_win_prob"] >= 0.5 else r["home_win"] == 0,
            axis=1
        )
        win_rate = picks.mean()
        print(f"    Edge >= {threshold:.0%}: {len(strong)} games, "
              f"win rate {win_rate:.1%} ({picks.sum()}/{len(strong)})")

    return results


def main():
    parser = argparse.ArgumentParser(description="MLB Win Probability Predictions")
    parser.add_argument("--date", type=str,
                        help="Predict games on this date (YYYY-MM-DD). Default: today")
    parser.add_argument("--backtest", type=int, nargs="+",
                        help="Backtest season(s)")
    args = parser.parse_args()

    if args.backtest:
        all_results = []
        for year in args.backtest:
            r = backtest_season(year)
            if r is not None:
                all_results.append(r)

        if len(all_results) > 1:
            combined = pd.concat(all_results, ignore_index=True)
            y_true = combined["home_win"].values
            print(f"\n{'='*60}")
            print(f"Combined Backtest ({len(combined)} games)")
            print(f"{'='*60}")
            for col, name in [("home_win_prob_lr", "LR"), ("home_win_prob_xgb", "XGB"), ("home_win_prob", "Ensemble")]:
                probs = combined[col].values
                ll = log_loss(y_true, probs)
                auc = roc_auc_score(y_true, probs)
                bs = brier_score_loss(y_true, probs)
                print(f"  {name}: log_loss={ll:.4f}, AUC={auc:.4f}, brier={bs:.4f}")
    else:
        # Live prediction mode
        target_date = args.date or date.today().strftime("%Y-%m-%d")
        predict_date(target_date)


if __name__ == "__main__":
    main()
