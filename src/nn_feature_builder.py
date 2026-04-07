#!/usr/bin/env python3
"""
Neural Net Feature Builder

Extracts per-game features for the NN win probability model:
- Per-hitter PA outcome distributions (11 categories) vs opposing SP
- Pitcher eval scores (stuff/location/sequencing)
- SP arsenal features and pitch mix
- Bullpen quality
- Park/weather context

Outputs: data/features/nn_features_{year}.parquet

Usage:
    python src/nn_feature_builder.py --season 2024
    python src/nn_feature_builder.py --season 2025
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    DATA_DIR, XRV_DIR, MODEL_DIR, FEATURES_DIR, GAMES_DIR,
    HARD_TYPES, BREAK_TYPES, OFFSPEED_TYPES,
)
from multi_output_matchup_model import (
    load_multi_output_models, predict_matchup_distribution,
    OUTCOME_ORDER, PITCH_TYPES, PTYPE_MAP, N_PTYPES,
)
from arsenal_matchup_model import ARSENAL_FEATURES
from feature_engineering import (
    _compute_pitcher_arsenal_live,
    _standardize_arsenal,
    _get_before,
    _extract_lineups,
    filter_competitive,
)

N_OUTCOMES = len(OUTCOME_ORDER)  # 11
N_HITTERS = 9  # lineup slots


def _load_xrv_indexed(year: int) -> tuple[pd.DataFrame, dict]:
    """Load xRV data and build pitcher index for fast lookup."""
    path = XRV_DIR / f"statcast_xrv_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"xRV data not found: {path}")
    xrv = pd.read_parquet(path)
    if "game_type" in xrv.columns:
        xrv = xrv[xrv["game_type"] == "R"]

    # Sort by game_date for _get_before binary search
    xrv = xrv.sort_values("game_date").reset_index(drop=True)

    # Build pitcher index
    pitcher_idx = {}
    for pid, grp in xrv.groupby("pitcher"):
        pitcher_idx[int(pid)] = grp.sort_values("game_date").reset_index(drop=True)

    return xrv, pitcher_idx


def _compute_pitch_mix(pitcher_df: pd.DataFrame, game_date: str,
                       n_pitches: int = 2000) -> np.ndarray | None:
    """Compute pitch-type mix vector for multi-output model."""
    before = _get_before(pitcher_df, game_date)
    recent = before.iloc[-n_pitches:] if len(before) > n_pitches else before
    recent = filter_competitive(recent)

    if len(recent) < 100:
        return None

    n = len(recent)
    pt_counts = recent["pitch_type"].value_counts()
    mix = np.zeros(N_PTYPES, dtype=np.float32)
    for pt, count in pt_counts.items():
        if pt in PTYPE_MAP:
            mix[PTYPE_MAP[pt]] = count / n
    return mix


def _load_pitcher_eval_scores(year: int) -> dict:
    """Load stuff, location, and sequencing scores for pitchers.

    Returns dict: {pitcher_id: {"stuff": float, "location": float, "sequencing": float}}
    """
    scores = {}

    # Load stuff scores (no-location model gives raw stuff)
    noloc_path = MODEL_DIR / f"stuff_noloc_{year}.pkl"
    withloc_path = MODEL_DIR / f"stuff_withloc_{year}.pkl"
    loc_path = MODEL_DIR / f"location_scores_{year}.pkl"
    seq_path = MODEL_DIR / f"sequencing_{year}.pkl"

    noloc_data = {}
    if noloc_path.exists():
        with open(noloc_path, "rb") as f:
            noloc_data = pickle.load(f)

    loc_data = {}
    if loc_path.exists():
        with open(loc_path, "rb") as f:
            loc_data = pickle.load(f)

    seq_data = {}
    if seq_path.exists():
        with open(seq_path, "rb") as f:
            seq_data = pickle.load(f)

    # Merge across hands — use weighted average of vsL and vsR
    all_pids = set()
    for hand in ["L", "R"]:
        for pid in noloc_data.get(f"pitcher_overall_vs{hand}", {}):
            all_pids.add(pid)

    for pid in all_pids:
        stuff_l = noloc_data.get("pitcher_overall_vsL", {}).get(pid)
        stuff_r = noloc_data.get("pitcher_overall_vsR", {}).get(pid)
        # Average across platoon sides (weight equally for now)
        stuff_vals = [v for v in [stuff_l, stuff_r] if v is not None]
        stuff = np.mean(stuff_vals) if stuff_vals else None

        loc_l = loc_data.get("vsL", {}).get(pid, {}).get("overall", None)
        loc_r = loc_data.get("vsR", {}).get(pid, {}).get("overall", None)
        loc_vals = [v for v in [loc_l, loc_r] if v is not None]
        loc = np.mean(loc_vals) if loc_vals else None

        seq_l = seq_data.get(f"sequencing_vsL", {}).get(pid, {}).get("sequencing_score", None)
        seq_r = seq_data.get(f"sequencing_vsR", {}).get(pid, {}).get("sequencing_score", None)
        seq_vals = [v for v in [seq_l, seq_r] if v is not None]
        seq = np.mean(seq_vals) if seq_vals else None

        scores[pid] = {
            "stuff": stuff,
            "location": loc,
            "sequencing": seq,
        }

    return scores


def build_nn_features(year: int, n_pitcher_pitches: int = 2000) -> pd.DataFrame:
    """Build NN features for all games in a season.

    For each game, produces a flat feature vector containing:
    - 9 x 11 = 99 PA outcome distribution values per side (home lineup vs away SP, etc.)
    - Pitcher eval scores per side
    - SP rest days per side
    - Bullpen quality per side
    - Park/weather context
    """
    print(f"\n{'='*60}")
    print(f"  Building NN features for {year}")
    print(f"{'='*60}")

    # Load game features (has labels, context features)
    gf_path = FEATURES_DIR / f"game_features_{year}.parquet"
    if not gf_path.exists():
        raise FileNotFoundError(f"Game features not found: {gf_path}")
    game_features = pd.read_parquet(gf_path)

    # Load games (has SP IDs)
    games_path = GAMES_DIR / f"games_{year}.parquet"
    if not games_path.exists():
        raise FileNotFoundError(f"Games data not found: {games_path}")
    games = pd.read_parquet(games_path)
    games_map = {int(r["game_pk"]): r.to_dict() for _, r in games.iterrows()}

    # Load xRV data and build indices
    print("  Loading xRV data...")
    xrv, pitcher_idx = _load_xrv_indexed(year)

    # Extract lineups from xRV
    print("  Extracting lineups...")
    lineups = _extract_lineups(xrv)

    # Load multi-output matchup models
    # Try current year first, then prior year (for early-season games)
    mo_models = None
    arsenal_stats = None
    for model_year in [year, year - 1]:
        try:
            mo_models = load_multi_output_models(model_year)
            if mo_models:
                # Get arsenal_stats from one of the models
                for hand in ["L", "R"]:
                    if hand in mo_models:
                        arsenal_stats = mo_models[hand]["arsenal_stats"]
                        break
                print(f"  Loaded multi-output models from {model_year}")
                break
        except Exception as e:
            print(f"  Could not load multi-output models for {model_year}: {e}")
    if not mo_models:
        print("  WARNING: No multi-output models available. PA distributions will be NaN.")

    # Load pitcher eval scores
    print("  Loading pitcher eval scores...")
    pitcher_evals = _load_pitcher_eval_scores(year)
    if not pitcher_evals:
        # Try prior year
        pitcher_evals = _load_pitcher_eval_scores(year - 1)
    print(f"  Pitcher evals loaded for {len(pitcher_evals)} pitchers")

    # Process each game
    rows = []
    n_games = len(game_features)
    n_with_lineups = 0
    n_with_distributions = 0

    for game_i, (_, gf_row) in enumerate(game_features.iterrows()):
        game_pk = int(gf_row["game_pk"])
        game_date = str(gf_row["game_date"])

        if (game_i + 1) % 200 == 0:
            print(f"  Processing game {game_i + 1}/{n_games}...")

        row = {
            "game_pk": game_pk,
            "game_date": game_date,
            "home_team": gf_row.get("home_team"),
            "away_team": gf_row.get("away_team"),
            "home_win": gf_row.get("home_win", np.nan),
        }

        # Get SP IDs from games data
        game_info = games_map.get(game_pk, {})
        home_sp_id = game_info.get("home_sp_id")
        away_sp_id = game_info.get("away_sp_id")
        if pd.notna(home_sp_id):
            home_sp_id = int(home_sp_id)
        else:
            home_sp_id = None
        if pd.notna(away_sp_id):
            away_sp_id = int(away_sp_id)
        else:
            away_sp_id = None

        # Get lineups
        game_lineup = lineups.get(game_pk)
        has_lineup = (game_lineup is not None
                      and len(game_lineup.get("home", [])) >= 5
                      and len(game_lineup.get("away", [])) >= 5)
        if has_lineup:
            n_with_lineups += 1

        # --- PA outcome distributions: home lineup vs away SP ---
        has_dist = False
        for side, lineup_key, opp_sp_id in [
            ("home", "home", away_sp_id),
            ("away", "away", home_sp_id),
        ]:
            # Initialize NaN distributions for all 9 slots
            for slot in range(N_HITTERS):
                for oi, outcome in enumerate(OUTCOME_ORDER):
                    row[f"{side}_h{slot}_{outcome}"] = np.nan

            if not (has_lineup and opp_sp_id and mo_models):
                continue

            opp_pitcher_df = pitcher_idx.get(opp_sp_id)
            if opp_pitcher_df is None or len(opp_pitcher_df) < 100:
                continue

            # Compute pitcher arsenal profile and pitch mix
            arsenal_raw = _compute_pitcher_arsenal_live(
                opp_pitcher_df, game_date, n_pitcher_pitches
            )
            if arsenal_raw is None:
                continue

            pitch_mix = _compute_pitch_mix(opp_pitcher_df, game_date, n_pitcher_pitches)
            if pitch_mix is None:
                continue

            arsenal_z = _standardize_arsenal(arsenal_raw, arsenal_stats)

            # Get lineup hitters
            lu = game_lineup[lineup_key][:N_HITTERS]
            slot_filled = False

            for slot, (hitter_id, hitter_hand) in enumerate(lu):
                # Pick the right hand-specific model
                model_key = hitter_hand  # "L" or "R"
                if model_key not in mo_models:
                    continue

                dist = predict_matchup_distribution(
                    mo_models[model_key], arsenal_z, pitch_mix, hitter_id
                )

                for oi, outcome in enumerate(OUTCOME_ORDER):
                    row[f"{side}_h{slot}_{outcome}"] = float(dist[oi])
                slot_filled = True

            if slot_filled:
                has_dist = True

        if has_dist:
            n_with_distributions += 1

        # --- Pitcher eval scores ---
        for side, sp_id in [("home", home_sp_id), ("away", away_sp_id)]:
            evals = pitcher_evals.get(sp_id, {}) if sp_id else {}
            row[f"{side}_sp_stuff"] = evals.get("stuff", np.nan)
            row[f"{side}_sp_location"] = evals.get("location", np.nan)
            row[f"{side}_sp_sequencing"] = evals.get("sequencing", np.nan)

        # --- SP rest days (from game_features) ---
        row["home_sp_rest_days"] = gf_row.get("home_sp_rest_days", np.nan)
        row["away_sp_rest_days"] = gf_row.get("away_sp_rest_days", np.nan)

        # --- SP expected innings proxy: n_pitches as confidence indicator ---
        row["home_sp_n_pitches"] = gf_row.get("home_sp_n_pitches", np.nan)
        row["away_sp_n_pitches"] = gf_row.get("away_sp_n_pitches", np.nan)

        # --- Bullpen quality ---
        row["home_bp_xrv"] = gf_row.get("home_bp_xrv_mean", np.nan)
        row["home_bp_fatigue"] = gf_row.get("home_bp_fatigue_score", np.nan)
        row["away_bp_xrv"] = gf_row.get("away_bp_xrv_mean", np.nan)
        row["away_bp_fatigue"] = gf_row.get("away_bp_fatigue_score", np.nan)

        # --- Context features ---
        row["park_factor"] = gf_row.get("park_factor", 1.0)
        row["temperature"] = gf_row.get("temperature", np.nan)
        row["wind_speed"] = gf_row.get("wind_speed", np.nan)
        row["is_dome"] = gf_row.get("is_dome", 0)
        row["is_night"] = gf_row.get("is_night", 0)

        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"\n  Results:")
    print(f"    Total games: {n_games}")
    print(f"    Games with lineups: {n_with_lineups}")
    print(f"    Games with PA distributions: {n_with_distributions}")
    print(f"    Output columns: {len(df.columns)}")

    # Save
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / f"nn_features_{year}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"    Saved to {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build NN features for win model")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--n-pitches", type=int, default=2000)
    args = parser.parse_args()

    build_nn_features(args.season, args.n_pitches)


if __name__ == "__main__":
    main()
