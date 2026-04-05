#!/usr/bin/env python3
"""
Pitcher stuff and location models.

P1 (no-location): Pitch specs → xRV without plate_x/plate_z.
    Converges fast (~200 pitches). Measures raw "stuff quality."

P2 (with-location): Same but with plate_x/plate_z.
    P2 - P1 = pure location score. Converges slower.

Both are LightGBM regressors trained on the population, then per-pitcher
scores are extracted as mean predictions on that pitcher's recent pitches.

Separate models for vs-RHH and vs-LHH.

Usage:
    python src/pitcher_stuff_model.py --season 2024
    python src/pitcher_stuff_model.py --season 2024 --prior-season  # for location priors
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import MODEL_DIR, XRV_DIR

# Features shared by both models (no location)
STUFF_FEATURES = [
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
    "release_extension", "spin_axis", "effective_speed",
    "release_pos_x", "release_pos_z",
    "balls", "strikes",
]

# Additional features for the with-location model
LOCATION_FEATURES = ["plate_x", "plate_z"]

# Pitch types with enough data to model
COMMON_PITCH_TYPES = {"FF", "SI", "FC", "SL", "CU", "CH", "ST", "SV", "SW", "KC", "FS"}


def _load_xrv(season: int) -> pd.DataFrame:
    """Load xRV parquet for a season."""
    path = XRV_DIR / f"statcast_xrv_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No xRV data for {season}: {path}")
    df = pd.read_parquet(path)
    # Filter to regular season, competitive pitches
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    return df


def _prepare_data(xrv: pd.DataFrame, hand: str) -> pd.DataFrame:
    """Filter and prepare data for a specific batter hand."""
    df = xrv[xrv["stand"] == hand].copy()
    # Only pitches with xRV values and known pitch types
    df = df.dropna(subset=["xrv"])
    df = df[df["pitch_type"].isin(COMMON_PITCH_TYPES)]
    # Label-encode pitch type
    df["pitch_type_code"] = df["pitch_type"].astype("category").cat.codes
    return df


def train_stuff_model(season: int, include_location: bool = False) -> dict:
    """Train population-level stuff model for a season.

    Returns dict with keys:
        model_vsL, model_vsR: trained LightGBM models
        pitcher_scores_vsL, pitcher_scores_vsR: {pitcher_id: {pitch_type: mean_pred}}
        features: list of feature names used
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm required: pip install lightgbm")

    # Load current + prior season data for training
    frames = []
    for yr in [season - 1, season]:
        try:
            frames.append(_load_xrv(yr))
        except FileNotFoundError:
            pass
    if not frames:
        raise FileNotFoundError(f"No xRV data available for {season} or {season-1}")
    xrv = pd.concat(frames, ignore_index=True)

    features = ["pitch_type_code"] + STUFF_FEATURES
    if include_location:
        features = features + LOCATION_FEATURES

    mode = "withloc" if include_location else "noloc"
    print(f"\n{'='*60}")
    print(f"  Pitcher Stuff Model ({mode}) — {season}")
    print(f"{'='*60}")
    print(f"  Total pitches: {len(xrv):,}")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 7,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    result = {"features": features, "mode": mode, "season": season}

    for hand in ["L", "R"]:
        df = _prepare_data(xrv, hand)
        print(f"\n  vs {hand}HH: {len(df):,} pitches")

        # Drop rows with NaN in features
        valid_mask = df[features + ["xrv"]].notna().all(axis=1)
        df = df[valid_mask].reset_index(drop=True)
        print(f"  After dropping NaN: {len(df):,}")

        X = df[features].to_numpy(dtype=np.float64, na_value=np.nan)
        y = df["xrv"].to_numpy(dtype=np.float64, na_value=np.nan)

        # Train LightGBM
        cat_idx = [0]  # pitch_type_code is categorical
        train_data = lgb.Dataset(
            X, label=y,
            feature_name=features,
            categorical_feature=[features[i] for i in cat_idx],
        )
        model = lgb.train(params, train_data, num_boost_round=600)

        # Evaluate
        preds = model.predict(X)
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  RMSE: {rmse:.5f}, R²: {r2:.4f}")

        # Extract per-pitcher, per-pitch-type scores from CURRENT season only
        current_mask = df["season"] == season if "season" in df.columns else np.ones(len(df), dtype=bool)
        current = df[current_mask].copy()
        current_preds = model.predict(current[features].to_numpy(dtype=np.float64, na_value=np.nan))
        current["pred"] = current_preds

        pitcher_scores = {}
        for (pid, pt), grp in current.groupby(["pitcher", "pitch_type"]):
            pid = int(pid)
            if pid not in pitcher_scores:
                pitcher_scores[pid] = {}
            pitcher_scores[pid][pt] = {
                "mean_pred": float(grp["pred"].mean()),
                "n_pitches": len(grp),
                "se": float(grp["pred"].std() / np.sqrt(len(grp))) if len(grp) > 1 else float("inf"),
            }

        # Overall pitcher score (weighted mean across pitch types)
        pitcher_overall = {}
        for pid, pt_dict in pitcher_scores.items():
            total_n = sum(v["n_pitches"] for v in pt_dict.values())
            if total_n >= 50:
                weighted = sum(v["mean_pred"] * v["n_pitches"] for v in pt_dict.values()) / total_n
                pitcher_overall[pid] = float(weighted)

        result[f"model_vs{hand}"] = model
        result[f"pitcher_scores_vs{hand}"] = pitcher_scores
        result[f"pitcher_overall_vs{hand}"] = pitcher_overall

        # Top/bottom pitchers
        sorted_pitchers = sorted(pitcher_overall.items(), key=lambda x: x[1])
        print(f"  Pitcher scores extracted: {len(pitcher_overall)}")
        if sorted_pitchers:
            print(f"  Best stuff (lowest xRV): {sorted_pitchers[0][1]:.5f} (pid={sorted_pitchers[0][0]})")
            print(f"  Worst stuff: {sorted_pitchers[-1][1]:.5f} (pid={sorted_pitchers[-1][0]})")
            scores = [v for _, v in sorted_pitchers]
            print(f"  Score range: [{min(scores):.5f}, {max(scores):.5f}], "
                  f"std={np.std(scores):.5f}")

    return result


def extract_location_scores(noloc_result: dict, withloc_result: dict) -> dict:
    """Compute location scores as P2 - P1 per pitcher per pitch type.

    Negative location score = pitcher places pitches better than stuff alone suggests.
    """
    location_scores = {}

    for hand in ["L", "R"]:
        noloc_scores = noloc_result[f"pitcher_scores_vs{hand}"]
        withloc_scores = withloc_result[f"pitcher_scores_vs{hand}"]

        loc = {}
        for pid in set(noloc_scores) & set(withloc_scores):
            pt_scores = {}
            for pt in set(noloc_scores[pid]) & set(withloc_scores[pid]):
                nl = noloc_scores[pid][pt]
                wl = withloc_scores[pid][pt]
                # Only if both have enough data
                if nl["n_pitches"] >= 50 and wl["n_pitches"] >= 50:
                    pt_scores[pt] = float(wl["mean_pred"] - nl["mean_pred"])
            if pt_scores:
                # Weighted average across pitch types
                noloc_pts = noloc_scores[pid]
                total_n = sum(noloc_pts[pt]["n_pitches"] for pt in pt_scores)
                weighted = sum(pt_scores[pt] * noloc_pts[pt]["n_pitches"] for pt in pt_scores) / total_n
                loc[pid] = {"per_type": pt_scores, "overall": float(weighted)}

        location_scores[f"vs{hand}"] = loc

    return location_scores


def save_models(result: dict, location_scores: dict = None):
    """Save trained models and scores."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    season = result["season"]
    mode = result["mode"]

    path = MODEL_DIR / f"stuff_{mode}_{season}.pkl"
    with open(path, "wb") as f:
        # Don't pickle LightGBM models directly — save as string
        save_dict = {k: v for k, v in result.items() if not k.startswith("model_")}
        for hand in ["L", "R"]:
            key = f"model_vs{hand}"
            if key in result:
                save_dict[f"model_str_vs{hand}"] = result[key].model_to_string()
        pickle.dump(save_dict, f)
    print(f"\n  Saved to {path}")

    if location_scores:
        loc_path = MODEL_DIR / f"location_scores_{season}.pkl"
        with open(loc_path, "wb") as f:
            pickle.dump(location_scores, f)
        print(f"  Location scores saved to {loc_path}")


def load_stuff_model(season: int, mode: str = "noloc") -> dict:
    """Load a trained stuff model."""
    import lightgbm as lgb

    path = MODEL_DIR / f"stuff_{mode}_{season}.pkl"
    with open(path, "rb") as f:
        result = pickle.load(f)

    # Reconstruct LightGBM models from strings
    for hand in ["L", "R"]:
        str_key = f"model_str_vs{hand}"
        if str_key in result:
            booster = lgb.Booster(model_str=result[str_key])
            result[f"model_vs{hand}"] = booster

    return result


def load_location_scores(season: int) -> dict:
    """Load pre-computed location scores for prior-season bridging."""
    path = MODEL_DIR / f"location_scores_{season}.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def get_pitcher_eval(
    pitcher_id: int,
    hand: str,
    noloc_result: dict,
    withloc_result: dict,
    location_scores: dict,
    prior_location: dict = None,
    games_into_season: int = 0,
) -> dict:
    """Get composite pitcher evaluation for a single pitcher.

    Returns dict with:
        stuff_score: mean P1 prediction (lower = better stuff for pitcher)
        location_score: P2 - P1 (negative = better location)
        composite: weighted combination
        converged: bool
    """
    hand_key = f"vs{hand}"
    stuff_overall = noloc_result.get(f"pitcher_overall_{hand_key}", {})
    stuff = stuff_overall.get(pitcher_id, None)

    # Location score with prior-season bridging
    loc_scores = location_scores.get(hand_key, {})
    current_loc = loc_scores.get(pitcher_id, {}).get("overall", 0.0)

    if prior_location and games_into_season < 60:
        prior_loc = prior_location.get(hand_key, {}).get(pitcher_id, {}).get("overall", 0.0)
        w = max(0.0, 1.0 - games_into_season / 60)
        effective_loc = w * prior_loc + (1 - w) * current_loc
    else:
        effective_loc = current_loc

    if stuff is None:
        return {"stuff_score": None, "location_score": None, "composite": None, "converged": False}

    return {
        "stuff_score": stuff,
        "location_score": effective_loc,
        "composite": stuff + effective_loc,  # lower = better pitcher
        "converged": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Pitcher Stuff Model")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--prior-season", action="store_true",
                        help="Also build location priors from prior season")
    args = parser.parse_args()

    # Train both models
    print("Training no-location model...")
    noloc = train_stuff_model(args.season, include_location=False)
    save_models(noloc)

    print("\nTraining with-location model...")
    withloc = train_stuff_model(args.season, include_location=True)
    save_models(withloc)

    # Extract location scores
    print("\nExtracting location scores...")
    loc_scores = extract_location_scores(noloc, withloc)
    for hand in ["L", "R"]:
        n = len(loc_scores[f"vs{hand}"])
        vals = [v["overall"] for v in loc_scores[f"vs{hand}"].values()]
        if vals:
            print(f"  vs {hand}HH: {n} pitchers, "
                  f"range [{min(vals):.5f}, {max(vals):.5f}], "
                  f"std={np.std(vals):.5f}")
    # Save location scores separately (don't overwrite withloc model)
    loc_path = MODEL_DIR / f"location_scores_{args.season}.pkl"
    with open(loc_path, "wb") as f:
        pickle.dump(loc_scores, f)
    print(f"  Location scores saved to {loc_path}")

    # Optionally build prior-season location scores
    if args.prior_season and args.season > 2024:
        prior = args.season - 1
        print(f"\nBuilding location priors from {prior}...")
        noloc_prior = train_stuff_model(prior, include_location=False)
        withloc_prior = train_stuff_model(prior, include_location=True)
        prior_loc = extract_location_scores(noloc_prior, withloc_prior)

        path = MODEL_DIR / f"location_prior_{args.season}.pkl"
        with open(path, "wb") as f:
            pickle.dump(prior_loc, f)
        print(f"  Prior location scores saved to {path}")


if __name__ == "__main__":
    main()
