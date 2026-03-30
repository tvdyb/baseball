#!/usr/bin/env python3
"""
Generate pregame win probabilities for today's (or any date's) games.

Usage:
    python src/predict.py                     # today's games
    python src/predict.py --date 2024-09-15   # specific date
    python src/predict.py --backtest 2024     # full season backtest with rolling training
"""

import argparse
import os
from datetime import date, datetime, timezone
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
    _preindex_xrv, compute_single_game_features, compute_park_factors,
    _load_hand_models, _load_team_oaa, _compute_team_priors,
    _load_trade_deadline_acquisitions, _load_pitcher_projections,
    _load_team_projections, _compute_adjusted_team_priors,
    _recent_winpct, DIFF_COLS, OPENING_DAY,
)
from win_model import (
    ALL_FEATURES, XGB_PARAMS,
    _smart_fillna, add_nonlinear_features,
)
from utils import XRV_DIR, GAMES_DIR, FEATURES_DIR

MLB_API = "https://statsapi.mlb.com/api/v1"

# Maximum age (in hours) for parquet files used in live prediction
MAX_DATA_AGE_HOURS = 48


class StaleDataError(Exception):
    """Raised when required data files are too old for live prediction."""
    pass


def _check_file_freshness(path: Path, max_age_hours: float = MAX_DATA_AGE_HOURS,
                           label: str = "") -> float:
    """Check modification time of a file. Returns age in hours.

    Raises StaleDataError if older than max_age_hours.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    mtime = path.stat().st_mtime
    age_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600
    name = label or path.name
    if age_hours > max_age_hours:
        raise StaleDataError(
            f"{name} is {age_hours:.1f}h old (max {max_age_hours}h). "
            f"Run the upstream pipeline to refresh: {path}"
        )
    return age_hours


def validate_live_data_freshness(target_date: str, max_age_hours: float = MAX_DATA_AGE_HOURS):
    """Validate that all data files needed for live prediction are fresh.

    Two-level check:
      1. File mtime: parquet files must have been written within max_age_hours
      2. Content currency: games file must contain data through yesterday at minimum

    Raises StaleDataError with actionable message if any check fails.
    """
    year = int(target_date[:4])
    issues = []

    # ── mtime checks ──
    for yr in [year - 1, year]:
        xrv_path = XRV_DIR / f"statcast_xrv_{yr}.parquet"
        if xrv_path.exists():
            try:
                age = _check_file_freshness(xrv_path, max_age_hours, f"xRV {yr}")
                print(f"  xRV {yr}: {age:.1f}h old ✓")
            except StaleDataError as e:
                issues.append(str(e))
        elif yr == year:
            print(f"  xRV {yr}: not found (OK if early in season)")
        else:
            issues.append(f"Prior-year xRV ({yr}) not found: {xrv_path}")

    games_path = GAMES_DIR / f"games_{year}.parquet"
    if games_path.exists():
        try:
            age = _check_file_freshness(games_path, max_age_hours, f"Games {year}")
            print(f"  Games {year}: {age:.1f}h old ✓")
        except StaleDataError as e:
            issues.append(str(e))

        # ── Content currency check: games file should cover through yesterday ──
        try:
            games_df = pd.read_parquet(games_path, columns=["game_date"])
            if len(games_df) > 0:
                latest = str(games_df["game_date"].max())
                yesterday = (date.today() - __import__("datetime").timedelta(days=2)).isoformat()
                if latest < yesterday:
                    issues.append(
                        f"Games {year} data ends at {latest}, expected through at least "
                        f"{yesterday}. Run: make scrape-games"
                    )
                else:
                    print(f"  Games {year}: data through {latest} ✓")
        except Exception as e:
            issues.append(f"Could not read games file for content check: {e}")
    else:
        print(f"  Games {year}: not found (OK for opening day)")

    if issues:
        raise StaleDataError(
            "Stale or missing data files for live prediction:\n  " +
            "\n  ".join(issues) +
            "\n\nRun: make scrape-statcast scrape-games build-xrv"
        )
    print("  All data files fresh ✓")


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


def _prepare_lr_features(df: pd.DataFrame):
    """Prepare LR feature matrix: linear features only + _smart_fillna."""
    available = [f for f in ALL_FEATURES if f in df.columns]
    return df[available].copy(), available


def _prepare_xgb_features(df: pd.DataFrame):
    """Prepare XGB feature matrix: ALL_FEATURES + nonlinear features (NaN handled by XGB)."""
    available = [f for f in ALL_FEATURES if f in df.columns]
    X = df[available].copy()
    # Add raw columns needed for nonlinear feature engineering
    for col in ["home_sp_rest_days", "away_sp_rest_days",
                "home_bp_fatigue_score", "away_bp_fatigue_score",
                "days_into_season"]:
        if col in df.columns and col not in X.columns:
            X[col] = df[col]
    X = add_nonlinear_features(X)
    return X, list(X.columns)


def train_ensemble(train_df: pd.DataFrame, lr_only: bool = False) -> tuple:
    """Train LR + XGB ensemble, matching win_model.py pipeline exactly.

    LR: linear features → _smart_fillna → StandardScaler
    XGB: nonlinear features → DMatrix (NaN handled natively) → early stopping with val split

    If lr_only=True, skip XGB training and set w_lr=1.0.
    """
    y = train_df["home_win"].values

    # LR path: linear features only
    X_lr_raw, lr_features = _prepare_lr_features(train_df)
    X_lr_filled, train_medians = _smart_fillna(X_lr_raw)
    scaler = StandardScaler()
    X_lr_scaled = scaler.fit_transform(X_lr_filled)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(X_lr_scaled, y)

    if lr_only:
        print("  LR-only mode: skipping XGB training")
        return lr, scaler, None, lr_features, [], 1.0, train_medians

    # XGB path: nonlinear features
    X_xgb, xgb_features = _prepare_xgb_features(train_df)

    xgb_model = None
    w_lr = 0.5
    if HAS_XGB:
        # Chronological validation split for early stopping (never use test data)
        n = len(X_xgb)
        val_size = int(n * 0.2)
        dtrain = xgb.DMatrix(X_xgb.iloc[:n - val_size], label=y[:n - val_size])
        dval = xgb.DMatrix(X_xgb.iloc[n - val_size:], label=y[n - val_size:])
        xgb_model = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50, verbose_eval=50,
        )

        # Learn blend weight via OOF predictions (not in-sample)
        from scipy.optimize import minimize_scalar
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=5)  # forward-only time-series folds
        lr_oof = np.zeros(len(y))
        xgb_oof = np.zeros(len(y))

        for fold_train, fold_val in tscv.split(X_lr_raw):
            Xf_filled, fm = _smart_fillna(X_lr_raw.iloc[fold_train])
            Xv_filled, _ = _smart_fillna(X_lr_raw.iloc[fold_val], fm)
            sc_f = StandardScaler()
            lr_f = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            lr_f.fit(sc_f.fit_transform(Xf_filled), y[fold_train])
            lr_oof[fold_val] = lr_f.predict_proba(sc_f.transform(Xv_filled))[:, 1]

            df_fold = xgb.DMatrix(X_xgb.iloc[fold_train], label=y[fold_train])
            dv_fold = xgb.DMatrix(X_xgb.iloc[fold_val], label=y[fold_val])
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
        print(f"  Learned ensemble weight (OOF): LR={w_lr:.2f}, XGB={1-w_lr:.2f}")

    return lr, scaler, xgb_model, lr_features, xgb_features, w_lr, train_medians


def predict_games(lr, scaler, xgb_model, lr_features, xgb_features,
                  games_df: pd.DataFrame, w_lr: float = 0.5,
                  train_medians: dict = None) -> pd.DataFrame:
    """Generate predictions matching the training pipeline exactly.

    LR: linear features → _smart_fillna(train_medians) → scaler.transform
    XGB: nonlinear features → DMatrix (NaN native)
    """
    # LR path
    available_lr = [f for f in lr_features if f in games_df.columns]
    X_lr_raw = games_df[available_lr].copy()
    X_lr_filled, _ = _smart_fillna(X_lr_raw, train_medians)
    # Align columns to training (add missing as 0)
    for col in lr_features:
        if col not in X_lr_filled.columns:
            X_lr_filled[col] = 0
    X_lr_filled = X_lr_filled[lr_features]
    X_lr_scaled = scaler.transform(X_lr_filled)
    lr_probs = lr.predict_proba(X_lr_scaled)[:, 1]

    # XGB path
    if xgb_model and HAS_XGB:
        X_xgb, _ = _prepare_xgb_features(games_df)
        # Align columns to training
        for col in xgb_features:
            if col not in X_xgb.columns:
                X_xgb[col] = np.nan  # XGB handles NaN natively
        X_xgb = X_xgb[xgb_features]
        dtest = xgb.DMatrix(X_xgb)
        xgb_probs = xgb_model.predict(dtest)
        ensemble_probs = w_lr * lr_probs + (1 - w_lr) * xgb_probs
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

            if status == "Final":
                game["home_score"] = home.get("score", 0)
                game["away_score"] = away.get("score", 0)
                game["home_win"] = int(home.get("score", 0) > away.get("score", 0))

            games.append(game)

    return games


class LineupStatus:
    """Distinguishes lineup fetch outcomes for downstream decision-making."""
    AVAILABLE = "available"       # lineup posted, parsed successfully
    NOT_POSTED = "not_posted"     # API reachable, lineups not yet posted
    API_ERROR = "api_error"       # network/HTTP error (transient)
    PARSE_ERROR = "parse_error"   # API returned unexpected schema


def fetch_lineup(client: httpx.Client, game_pk: int) -> dict:
    """Fetch starting lineup from boxscore endpoint.

    Returns dict with:
      - home, away: list of (player_id, bat_side) tuples
      - home_status, away_status: LineupStatus values
      - error: str or None (details for API_ERROR/PARSE_ERROR)
    """
    try:
        resp = client.get(f"{MLB_API}/game/{game_pk}/boxscore")
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        print(f"  WARNING: MLB API returned {e.response.status_code} for game {game_pk}")
        return {
            "home": [], "away": [],
            "home_status": LineupStatus.API_ERROR,
            "away_status": LineupStatus.API_ERROR,
            "error": f"HTTP {e.response.status_code}",
        }
    except (httpx.RequestError, httpx.TimeoutException) as e:
        print(f"  WARNING: MLB API unreachable for game {game_pk}: {type(e).__name__}")
        return {
            "home": [], "away": [],
            "home_status": LineupStatus.API_ERROR,
            "away_status": LineupStatus.API_ERROR,
            "error": str(e),
        }
    except Exception as e:
        print(f"  WARNING: Unexpected error fetching lineup for game {game_pk}: {e}")
        return {
            "home": [], "away": [],
            "home_status": LineupStatus.PARSE_ERROR,
            "away_status": LineupStatus.PARSE_ERROR,
            "error": str(e),
        }

    # Validate expected schema
    teams = data.get("teams")
    if not isinstance(teams, dict):
        print(f"  WARNING: Unexpected boxscore schema for game {game_pk}: "
              f"'teams' is {type(teams).__name__}")
        return {
            "home": [], "away": [],
            "home_status": LineupStatus.PARSE_ERROR,
            "away_status": LineupStatus.PARSE_ERROR,
            "error": "missing or invalid 'teams' key",
        }

    result = {"error": None}
    for side in ("home", "away"):
        team_data = teams.get(side, {})
        if not isinstance(team_data, dict):
            result[side] = []
            result[f"{side}_status"] = LineupStatus.PARSE_ERROR
            continue

        batting_order = team_data.get("battingOrder", [])
        players = team_data.get("players", {})

        if not batting_order:
            # API reachable, schema valid, lineups just not posted yet
            result[side] = []
            result[f"{side}_status"] = LineupStatus.NOT_POSTED
            continue

        lineup = []
        for pid in batting_order:
            key = f"ID{pid}"
            p = players.get(key, {})
            bat_side = p.get("batSide", {}).get("code", "R")
            lineup.append((int(pid), bat_side))

        result[side] = lineup
        result[f"{side}_status"] = (
            LineupStatus.AVAILABLE if len(lineup) >= 9
            else LineupStatus.NOT_POSTED
        )

    return result


def build_live_features(
    games: list[dict],
    target_date: str,
    client: httpx.Client,
    skip_freshness_check: bool = False,
) -> pd.DataFrame:
    """Build feature vectors for today's games using compute_single_game_features.

    Validates data freshness before loading (raises StaleDataError if stale).
    Set skip_freshness_check=True for backtesting or offline use.
    """
    year = int(target_date[:4])

    # Validate data freshness for live prediction
    if not skip_freshness_check:
        print("  Checking data freshness...")
        validate_live_data_freshness(target_date)

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

    # Load all context data (same as batch path)
    matchup_models = _load_hand_models("matchup_model", year - 1)
    if matchup_models:
        print(f"  Matchup models loaded for hands: {list(matchup_models.keys())}")

    arsenal_models = _load_hand_models("arsenal_matchup", year - 1)
    if arsenal_models:
        print(f"  Arsenal models loaded for hands: {list(arsenal_models.keys())}")

    prior_xrv_path = XRV_DIR / f"statcast_xrv_{year - 1}.parquet"
    if prior_xrv_path.exists():
        prior_xrv = pd.read_parquet(prior_xrv_path)
        park_factors = compute_park_factors(prior_xrv, year - 1)
    else:
        park_factors = {}

    team_oaa = _load_team_oaa(year - 1)
    if team_oaa:
        print(f"  Team OAA from {year - 1}: {len(team_oaa)} teams")

    team_priors = _compute_team_priors(year - 1)
    if team_priors:
        print(f"  Team priors from {year - 1}: {len(team_priors)} teams")

    team_projections = _load_team_projections(year)
    pitcher_projections = _load_pitcher_projections(year)
    adjusted_priors = _compute_adjusted_team_priors(year - 1, team_priors, idx)

    trade_stats = _load_trade_deadline_acquisitions(year, idx)
    if trade_stats:
        print(f"  Trade deadline stats: {len(trade_stats)} teams")

    opening_day = OPENING_DAY.get(year, target_date)
    prior_year_cutoff = f"{year}-01-01"

    # Compute recent form from completed games this season
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

    # Fetch lineups for each game and build features using shared function
    feature_rows = []
    display_meta = []  # SP names, game_time, status for display
    api_errors = 0
    not_posted = 0
    parse_errors = 0
    for game in games:
        game_pk = game["game_pk"]

        # Fetch live lineup with status tracking
        lineup = fetch_lineup(client, game_pk)
        lineups = {game_pk: lineup}

        # Track lineup fetch outcomes
        for side in ("home", "away"):
            st = lineup.get(f"{side}_status", LineupStatus.NOT_POSTED)
            if st == LineupStatus.API_ERROR:
                api_errors += 1
            elif st == LineupStatus.PARSE_ERROR:
                parse_errors += 1
            elif st == LineupStatus.NOT_POSTED:
                not_posted += 1

        # weather_map is empty — live weather comes from game["weather"] dict
        row, meta = compute_single_game_features(
            game, idx, lineups, matchup_models, arsenal_models,
            park_factors, team_oaa, team_priors, adjusted_priors,
            team_projections, pitcher_projections, trade_stats,
            {},  # weather_map empty — live weather in game["weather"]
            year, opening_day, prior_year_cutoff,
        )

        # Add recent form and games played (not in compute_single_game_features)
        home_team = game["home_team"]
        away_team = game["away_team"]
        row["home_recent_form"] = _recent_winpct(team_history.get(home_team, []), 10)
        row["away_recent_form"] = _recent_winpct(team_history.get(away_team, []), 10)
        row["home_team_games_played"] = len(team_history.get(home_team, []))
        row["away_team_games_played"] = len(team_history.get(away_team, []))

        # Tag lineup status so downstream can suppress trades on degraded games
        home_st = lineup.get("home_status", LineupStatus.NOT_POSTED)
        away_st = lineup.get("away_status", LineupStatus.NOT_POSTED)
        row["_lineup_home_status"] = home_st
        row["_lineup_away_status"] = away_st

        # Suppress game if either side had an API or parse error
        # NOT_POSTED is fine (normal pre-game), but errors mean we can't trust features
        is_error = home_st in (LineupStatus.API_ERROR, LineupStatus.PARSE_ERROR) or \
                   away_st in (LineupStatus.API_ERROR, LineupStatus.PARSE_ERROR)
        row["_lineup_degraded"] = is_error

        feature_rows.append(row)
        display_meta.append({
            "home_sp_name": game.get("home_sp_name", "TBD"),
            "away_sp_name": game.get("away_sp_name", "TBD"),
            "game_time": game.get("game_time", ""),
            "status": game.get("status", ""),
        })

    # Report lineup fetch outcomes
    total_sides = len(games) * 2
    if api_errors > 0:
        print(f"  WARNING: {api_errors}/{total_sides} lineup fetches failed (API error). "
              "Matchup features will be NaN for those games.")
    if parse_errors > 0:
        print(f"  ERROR: {parse_errors}/{total_sides} lineup fetches returned unexpected schema. "
              "MLB API may have changed.")
    if not_posted > 0:
        print(f"  INFO: {not_posted}/{total_sides} lineups not yet posted.")

    features_df = pd.DataFrame(feature_rows)

    # Compute differentials (same as batch path)
    for col, sign in DIFF_COLS:
        hc = f"home_{col}"
        ac = f"away_{col}"
        if hc in features_df.columns and ac in features_df.columns:
            features_df[f"diff_{col}"] = sign * (features_df[hc] - features_df[ac])

    if "home_sp_xrv_vs_lineup" in features_df.columns and "away_sp_xrv_vs_lineup" in features_df.columns:
        features_df["diff_sp_xrv_vs_lineup"] = -(
            features_df["home_sp_xrv_vs_lineup"] - features_df["away_sp_xrv_vs_lineup"]
        )

    if "home_sp_home_xrv" in features_df.columns and "away_sp_away_xrv" in features_df.columns:
        features_df["diff_sp_context_xrv"] = -(
            features_df["home_sp_home_xrv"] - features_df["away_sp_away_xrv"]
        )

    # Attach display metadata
    for i, dm in enumerate(display_meta):
        for k, v in dm.items():
            features_df.loc[i, k] = v

    return features_df


def predict_date(target_date: str, lr_only: bool = False):
    """Predict games for a specific date."""
    print(f"\n{'='*60}")
    print(f"MLB Predictions for {target_date}")
    print(f"{'='*60}")

    print("\nTraining model on historical data...")
    train_df = load_training_data()
    train_df["game_date"] = train_df["game_date"].astype(str)
    n_before = len(train_df)
    train_df = train_df[train_df["game_date"] < target_date]
    if len(train_df) < n_before:
        print(f"  Filtered out {n_before - len(train_df)} games on/after {target_date}")
    print(f"  {len(train_df):,} training games (through {train_df['game_date'].max()})")
    lr, scaler, xgb_model, lr_features, xgb_features, w_lr, train_medians = train_ensemble(train_df, lr_only=lr_only)

    print(f"\nFetching games for {target_date}...")
    with httpx.Client(timeout=15.0) as client:
        games = fetch_todays_games(client, target_date)
        if not games:
            print("  No games found!")
            return

        print(f"  Found {len(games)} games")

        print("\nBuilding features...")
        features_df = build_live_features(games, target_date, client)

    results = predict_games(lr, scaler, xgb_model, lr_features, xgb_features, features_df, w_lr, train_medians)

    # Merge display columns back
    for col in ["home_sp_name", "away_sp_name", "game_time", "status"]:
        if col in features_df.columns:
            results[col] = features_df[col].values

    # Add confidence from SP info
    if "home_sp_info_confidence" in features_df.columns:
        results["sp_confidence"] = (
            features_df["home_sp_info_confidence"].fillna(0)
            + features_df["away_sp_info_confidence"].fillna(0)
        ) / 2

    # Early-season warning
    if "days_into_season" in features_df.columns:
        days = features_df["days_into_season"].iloc[0] if len(features_df) > 0 else 999
        if days < 21:
            print("\n  \u26a0 EARLY SEASON: Model has limited current-season data."
                  " Consider reduced position sizing.")

    # Display
    print(f"\n{'='*60}")
    print(f"  {'Matchup':<35s} {'Home%':>6s} {'Away%':>6s} {'Pick':>6s} {'Conf':>5s}")
    print(f"  {'-'*60}")

    for _, r in results.iterrows():
        away_sp = r.get("away_sp_name", "TBD")
        home_sp = r.get("home_sp_name", "TBD")
        matchup = f"{r['away_team']} @ {r['home_team']}"
        pitchers = f"  {away_sp} vs {home_sp}"

        pick_team = r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"]

        conf_val = r.get("sp_confidence", 1.0)
        conf_str = f"{conf_val:.2f}" if pd.notna(conf_val) else "N/A"
        low_conf = " \u26a0" if pd.notna(conf_val) and conf_val < 0.3 else ""

        result_str = ""
        if "home_win" in r and pd.notna(r.get("home_win")):
            actual = r["home_team"] if r["home_win"] == 1 else r["away_team"]
            correct = pick_team == actual
            result_str = f"  {'W' if correct else 'L'} ({actual} won)"

        print(f"  {matchup:<35s} {r['home_win_prob']:>5.1%} {r['away_win_prob']:>5.1%}"
              f" {pick_team:>5s} {conf_str:>5s}{low_conf}{result_str}")
        print(f"  {pitchers}")
        print()

    if "home_win" in results.columns and results["home_win"].notna().all():
        picks_correct = results.apply(
            lambda r: (r["home_win_prob"] >= 0.5) == (r["home_win"] == 1), axis=1)
        print(f"  Record: {picks_correct.sum()}/{len(picks_correct)} "
              f"({picks_correct.mean():.1%})")

    return results


def backtest_season(year: int, lr_only: bool = False):
    """Full season backtest with predictions for every game."""
    print(f"\n{'='*60}")
    print(f"Backtesting {year}")
    print(f"{'='*60}")

    train_df = load_training_data(exclude_year=year)
    print(f"  Training on {len(train_df):,} games from prior seasons")

    lr, scaler, xgb_model, lr_features, xgb_features, w_lr, train_medians = train_ensemble(train_df, lr_only=lr_only)

    test_path = FEATURES_DIR / f"game_features_{year}.parquet"
    if not test_path.exists():
        print(f"  No features for {year}")
        return None
    test_df = pd.read_parquet(test_path)
    test_df["season"] = year
    print(f"  Predicting {len(test_df)} games")

    results = predict_games(lr, scaler, xgb_model, lr_features, xgb_features, test_df, w_lr, train_medians)

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
    parser.add_argument("--lr-only", action="store_true",
                        help="Use logistic regression only (skip XGBoost)")
    args = parser.parse_args()

    if args.backtest:
        all_results = []
        for year in args.backtest:
            r = backtest_season(year, lr_only=args.lr_only)
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
        target_date = args.date or date.today().strftime("%Y-%m-%d")
        predict_date(target_date, lr_only=args.lr_only)


if __name__ == "__main__":
    main()
