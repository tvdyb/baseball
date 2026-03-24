#!/usr/bin/env python3
"""
Bayesian Hierarchical Pitcher-Hitter Matchup Model

Predicts expected xRV for a hitter facing a pitcher's recent pitch mix.
Uses hierarchical pooling across pitcher, hitter, pitch type, and matchup levels.

Inference: ADVI (default), --map (fast debug), --sample (full MCMC).
Models are split by batter handedness (vs LHH and vs RHH separately).

Usage:
    python src/matchup_model.py --season 2017
    python src/matchup_model.py --season 2017 --map      # fast debug
    python src/matchup_model.py --season 2017 --sample   # full MCMC
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

from utils import DATA_DIR, XRV_DIR, MODEL_DIR, filter_competitive as _filter_competitive


def load_season_xrv(year: int) -> pd.DataFrame:
    path = XRV_DIR / f"statcast_xrv_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"xRV data not found for {year}. Run build_xrv.py first.")
    df = pd.read_parquet(path)
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    return df


def prepare_matchup_data(
    df: pd.DataFrame,
    min_pitcher_pitches: int = 500,
    min_hitter_pa: int = 100,
    stand: str = None,
) -> pd.DataFrame:
    """Prepare data for the matchup model."""
    df = df.dropna(subset=["xrv"]).copy()

    if stand:
        df = df[df["stand"] == stand]

    # Filter competitive counts
    df = _filter_competitive(df)

    common_types = df["pitch_type"].value_counts()
    common_types = common_types[common_types > 1000].index.tolist()
    df = df[df["pitch_type"].isin(common_types)]

    pitcher_counts = df["pitcher"].value_counts()
    valid_pitchers = pitcher_counts[pitcher_counts >= min_pitcher_pitches].index
    df = df[df["pitcher"].isin(valid_pitchers)]

    hitter_pa = df.groupby("batter")["events"].apply(lambda x: x.notna().sum())
    valid_hitters = hitter_pa[hitter_pa >= min_hitter_pa].index
    df = df[df["batter"].isin(valid_hitters)]

    pitcher_ids = df["pitcher"].unique()
    hitter_ids = df["batter"].unique()
    pitch_types = df["pitch_type"].unique()

    pitcher_map = {pid: i for i, pid in enumerate(pitcher_ids)}
    hitter_map = {hid: i for i, hid in enumerate(hitter_ids)}
    ptype_map = {pt: i for i, pt in enumerate(pitch_types)}

    df["pitcher_idx"] = df["pitcher"].map(pitcher_map)
    df["hitter_idx"] = df["batter"].map(hitter_map)
    df["ptype_idx"] = df["pitch_type"].map(ptype_map)

    print(f"  Pitchers: {len(pitcher_ids)}")
    print(f"  Hitters: {len(hitter_ids)}")
    print(f"  Pitch types: {len(pitch_types)} {list(pitch_types)}")
    print(f"  Total pitches: {len(df):,}")

    df["matchup_key"] = df["pitcher"].astype(str) + "_" + df["batter"].astype(str)
    matchup_ids = df["matchup_key"].unique()
    matchup_map = {mk: i for i, mk in enumerate(matchup_ids)}
    df["matchup_idx"] = df["matchup_key"].map(matchup_map)
    print(f"  Unique matchups: {len(matchup_ids):,}")

    df.attrs["pitcher_map"] = pitcher_map
    df.attrs["hitter_map"] = hitter_map
    df.attrs["ptype_map"] = ptype_map
    df.attrs["matchup_map"] = matchup_map
    df.attrs["pitcher_ids"] = pitcher_ids
    df.attrs["hitter_ids"] = hitter_ids

    return df


def standardize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize continuous pitch features."""
    feature_cols = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z",
                    "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z"]
    stats = {}
    for col in feature_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if pd.notna(std) and std > 0:
                df[f"{col}_z"] = (df[col] - mean) / std
            else:
                df[f"{col}_z"] = 0.0
            stats[col] = {"mean": float(mean) if pd.notna(mean) else 0.0,
                          "std": float(std) if pd.notna(std) else 1.0}
    return df, stats


def build_model(df: pd.DataFrame, sample: bool = False, use_map: bool = False):
    """Build the Bayesian hierarchical matchup model."""
    n_pitchers = df["pitcher_idx"].nunique()
    n_hitters = df["hitter_idx"].nunique()
    n_ptypes = df["ptype_idx"].nunique()
    n_matchups = df["matchup_idx"].nunique()

    print(f"\n  Building PyMC model:")
    print(f"    {n_pitchers} pitchers, {n_hitters} hitters, {n_ptypes} pitch types")
    print(f"    {n_matchups:,} matchups, {len(df):,} observations")

    xrv = df["xrv"].values.astype(np.float64)
    pitcher_idx = df["pitcher_idx"].values.astype(int)
    hitter_idx = df["hitter_idx"].values.astype(int)
    ptype_idx = df["ptype_idx"].values.astype(int)
    matchup_idx = df["matchup_idx"].values.astype(int)

    def _get_z(col_name):
        z_col = f"{col_name}_z"
        if z_col in df.columns:
            arr = df[z_col].values.astype(np.float64)
        else:
            arr = np.zeros(len(df))
        arr[np.isnan(arr)] = 0.0
        return arr

    speed_z = _get_z("release_speed")
    hmov_z = _get_z("pfx_x")
    vmov_z = _get_z("pfx_z")
    locx_z = _get_z("plate_x")
    locz_z = _get_z("plate_z")
    spin_z = _get_z("release_spin_rate")
    ext_z = _get_z("release_extension")
    rel_x_z = _get_z("release_pos_x")
    rel_z_z = _get_z("release_pos_z")
    count_diff = (df["balls"].fillna(0) - df["strikes"].fillna(0)).values.astype(np.float64)

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=0.1)
        beta_speed = pm.Normal("beta_speed", mu=0, sigma=0.05)
        beta_hmov = pm.Normal("beta_hmov", mu=0, sigma=0.05)
        beta_vmov = pm.Normal("beta_vmov", mu=0, sigma=0.05)
        beta_locx = pm.Normal("beta_locx", mu=0, sigma=0.05)
        beta_locz = pm.Normal("beta_locz", mu=0, sigma=0.05)
        beta_count = pm.Normal("beta_count", mu=0, sigma=0.05)
        beta_spin = pm.Normal("beta_spin", mu=0, sigma=0.05)
        beta_ext = pm.Normal("beta_ext", mu=0, sigma=0.05)
        beta_rel_x = pm.Normal("beta_rel_x", mu=0, sigma=0.05)
        beta_rel_z = pm.Normal("beta_rel_z", mu=0, sigma=0.05)

        sigma_ptype = pm.HalfNormal("sigma_ptype", sigma=0.05)
        ptype_effect = pm.Normal("ptype_effect", mu=0, sigma=sigma_ptype, shape=n_ptypes)

        sigma_pitcher = pm.HalfNormal("sigma_pitcher", sigma=0.05)
        pitcher_effect = pm.Normal("pitcher_effect", mu=0, sigma=sigma_pitcher, shape=n_pitchers)

        sigma_hitter = pm.HalfNormal("sigma_hitter", sigma=0.05)
        hitter_effect = pm.Normal("hitter_effect", mu=0, sigma=sigma_hitter, shape=n_hitters)

        sigma_hitter_ptype = pm.HalfNormal("sigma_hitter_ptype", sigma=0.03)
        hitter_ptype_effect = pm.Normal(
            "hitter_ptype_effect", mu=0, sigma=sigma_hitter_ptype,
            shape=(n_hitters, n_ptypes),
        )

        sigma_matchup = pm.HalfNormal("sigma_matchup", sigma=0.02)
        matchup_effect = pm.Normal("matchup_effect", mu=0, sigma=sigma_matchup, shape=n_matchups)

        mu = (
            intercept
            + beta_speed * speed_z
            + beta_hmov * hmov_z
            + beta_vmov * vmov_z
            + beta_locx * locx_z
            + beta_locz * locz_z
            + beta_count * count_diff
            + beta_spin * spin_z
            + beta_ext * ext_z
            + beta_rel_x * rel_x_z
            + beta_rel_z * rel_z_z
            + ptype_effect[ptype_idx]
            + pitcher_effect[pitcher_idx]
            + hitter_effect[hitter_idx]
            + hitter_ptype_effect[hitter_idx, ptype_idx]
            + matchup_effect[matchup_idx]
        )

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.2)
        obs = pm.Normal("obs", mu=mu, sigma=sigma_obs, observed=xrv)

    print(f"    Parameters: ~{n_pitchers + n_hitters + n_hitters*n_ptypes + n_matchups + 14:,}")

    if sample:
        print("\n  Sampling (MCMC)...")
        with model:
            trace = pm.sample(
                draws=1000, tune=1000, chains=4, cores=4,
                target_accept=0.9, return_inferencedata=True,
            )
        return model, trace
    elif use_map:
        print("\n  Finding MAP estimate (fast debug mode)...")
        with model:
            map_estimate = pm.find_MAP(maxeval=10000)
        return model, map_estimate
    else:
        print("\n  Running ADVI (variational inference)...")
        with model:
            approx = pm.fit(
                n=30000,
                method="advi",
                callbacks=[pm.callbacks.CheckParametersConvergence(diff="absolute", tolerance=1e-4)],
            )
            trace = approx.sample(1000)
            map_estimate = {}
            for var_name in trace.posterior.data_vars:
                map_estimate[var_name] = trace.posterior[var_name].mean(dim=["chain", "draw"]).values
        return model, map_estimate


def predict_matchup_xrv(
    map_estimate: dict,
    pitcher_recent_pitches: pd.DataFrame,
    hitter_idx: int,
    feature_stats: dict,
    ptype_map: dict,
) -> float:
    """Predict expected xRV for a hitter against a pitcher's recent pitch mix."""
    intercept = map_estimate["intercept"]
    beta_speed = map_estimate["beta_speed"]
    beta_hmov = map_estimate["beta_hmov"]
    beta_vmov = map_estimate["beta_vmov"]
    beta_locx = map_estimate["beta_locx"]
    beta_locz = map_estimate["beta_locz"]
    beta_count = map_estimate["beta_count"]
    # New betas with backward compat
    beta_spin = map_estimate.get("beta_spin", 0.0)
    beta_ext = map_estimate.get("beta_ext", 0.0)
    beta_rel_x = map_estimate.get("beta_rel_x", 0.0)
    beta_rel_z = map_estimate.get("beta_rel_z", 0.0)

    ptype_effects = map_estimate["ptype_effect"]
    pitcher_effects = map_estimate["pitcher_effect"]
    hitter_effects = map_estimate["hitter_effect"]
    hitter_ptype_effects = map_estimate["hitter_ptype_effect"]

    xrvs = []
    pitcher_idx = pitcher_recent_pitches["pitcher_idx"].iloc[0]

    for _, pitch in pitcher_recent_pitches.iterrows():
        ptype = pitch.get("pitch_type", "")
        if ptype not in ptype_map:
            continue

        pt_idx = ptype_map[ptype]

        def z(col):
            if col in feature_stats and col in pitch.index:
                val = pitch[col]
                if pd.isna(val):
                    return 0.0
                s = feature_stats[col]
                return (val - s["mean"]) / s["std"] if s["std"] > 0 else 0.0
            return 0.0

        mu = (
            intercept
            + beta_speed * z("release_speed")
            + beta_hmov * z("pfx_x")
            + beta_vmov * z("pfx_z")
            + beta_locx * z("plate_x")
            + beta_locz * z("plate_z")
            + beta_count * 0
            + beta_spin * z("release_spin_rate")
            + beta_ext * z("release_extension")
            + beta_rel_x * z("release_pos_x")
            + beta_rel_z * z("release_pos_z")
            + ptype_effects[pt_idx]
            + pitcher_effects[pitcher_idx]
            + hitter_effects[hitter_idx]
            + hitter_ptype_effects[hitter_idx, pt_idx]
        )
        xrvs.append(mu)

    if not xrvs:
        return 0.0
    return float(np.mean(xrvs))


def save_model_artifacts(
    year: int,
    model_result,
    feature_stats: dict,
    df: pd.DataFrame,
    is_map: bool = True,
    hand: str = None,
):
    """Save model artifacts for later use in prediction."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "year": year,
        "feature_stats": feature_stats,
        "hand": hand,
        "pitcher_map": df.attrs.get("pitcher_map", {}),
        "hitter_map": df.attrs.get("hitter_map", {}),
        "ptype_map": df.attrs.get("ptype_map", {}),
        "matchup_map": df.attrs.get("matchup_map", {}),
        "pitcher_ids": df.attrs.get("pitcher_ids", []),
        "hitter_ids": df.attrs.get("hitter_ids", []),
    }

    if is_map:
        map_clean = {}
        for k, v in model_result.items():
            if hasattr(v, "eval"):
                map_clean[k] = np.array(v.eval())
            else:
                map_clean[k] = np.array(v)
        artifacts["map_estimate"] = map_clean
    else:
        artifacts["trace"] = model_result

    suffix = f"_vs{hand}" if hand else ""
    out_path = MODEL_DIR / f"matchup_model_{year}{suffix}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"  Saved model artifacts to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Bayesian Hierarchical Matchup Model")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--sample", action="store_true", help="Full MCMC sampling (slow)")
    parser.add_argument("--map", action="store_true", help="Use MAP instead of ADVI (faster but worse)")
    parser.add_argument("--min-pitcher-pitches", type=int, default=500)
    parser.add_argument("--min-hitter-pa", type=int, default=100)
    parser.add_argument("--subsample", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading xRV data for {args.season}...")
    df = load_season_xrv(args.season)
    print(f"  {len(df):,} pitches loaded")

    for hand in ["L", "R"]:
        print(f"\n{'='*60}")
        print(f"Training matchup model vs {hand}HH")
        print(f"{'='*60}")

        df_hand = df[df["stand"] == hand]
        print(f"  {len(df_hand):,} pitches vs {hand}HH")

        print(f"\nPreparing matchup data...")
        df_prep = prepare_matchup_data(
            df_hand,
            min_pitcher_pitches=args.min_pitcher_pitches,
            min_hitter_pa=args.min_hitter_pa,
            stand=hand,
        )

        print(f"\nStandardizing features...")
        df_prep, feature_stats = standardize_features(df_prep)

        if args.subsample > 0 and args.subsample < len(df_prep):
            print(f"\n  Subsampling to {args.subsample:,} pitches")
            df_prep = df_prep.sample(n=args.subsample, random_state=42)

        print(f"\nBuilding model...")
        model, result = build_model(df_prep, sample=args.sample, use_map=args.map)

        is_map = not args.sample
        save_model_artifacts(args.season, result, feature_stats, df_prep,
                           is_map=is_map, hand=hand)

        # Print diagnostics
        print(f"\n  Key estimates:")
        for param in ["intercept", "beta_speed", "beta_hmov", "beta_vmov",
                       "beta_locx", "beta_locz", "beta_count",
                       "beta_spin", "beta_ext", "beta_rel_x", "beta_rel_z"]:
            if param in result:
                print(f"    {param}: {float(np.array(result[param])):.6f}")

        for param in ["sigma_pitcher", "sigma_hitter", "sigma_hitter_ptype",
                       "sigma_matchup", "sigma_obs"]:
            if param in result:
                print(f"    {param}: {float(np.array(result[param])):.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
