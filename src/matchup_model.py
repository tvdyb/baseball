#!/usr/bin/env python3
"""
Bayesian Hierarchical Pitcher-Hitter Matchup Model

Predicts expected xRV for a hitter facing a pitcher's recent pitch mix.

The core idea:
  For each pitch a pitcher throws, we know its characteristics (type, velo,
  movement, location). We model how a specific hitter would perform against
  that pitch in terms of xRV. But most pitcher-hitter pairs have few or no
  historical matchup pitches, so we use a hierarchical model that pools
  information:

  Hierarchy:
    1. Population level: average xRV response to pitch characteristics
       (all hitters vs all pitchers)
    2. Pitcher level: how this pitcher's stuff deviates from average
       (his fastball is harder/softer, his slider breaks more/less)
    3. Hitter level: how this hitter responds to pitch types generally
       (good fastball hitter, bad against breaking balls)
    4. Matchup level: specific pitcher-hitter interaction
       (heavily shrunk toward pitcher + hitter effects due to small samples)

  Features per pitch:
    - Pitch type (FF, SL, CH, CU, etc.)
    - Velocity (release_speed)
    - Horizontal movement (pfx_x)
    - Vertical movement (pfx_z)
    - Location (plate_x, plate_z)
    - Count (balls, strikes)

  We model xRV as:
    xrv_ij ~ Normal(μ_ij, σ²)
    μ_ij = β_0 + β_pitch_features + α_pitcher[i] + γ_hitter[j] + δ_matchup[i,j]

    α_pitcher[i] ~ Normal(0, σ_pitcher²)
    γ_hitter[j] ~ Normal(0, σ_hitter²)
    δ_matchup[i,j] ~ Normal(0, σ_matchup²)

  For prediction: given pitcher P's last N pitches (against same handedness),
  compute expected xRV for hitter H against each of those pitches, then
  average. This gives us "how would hitter H do against pitcher P's recent
  stuff?"

Usage:
    python src/matchup_model.py --season 2017 --n-pitches 2000
    python src/matchup_model.py --season 2017 --sample  # full MCMC
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
XRV_DIR = DATA_DIR / "xrv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_season_xrv(year: int) -> pd.DataFrame:
    """Load xRV-augmented Statcast data for a season."""
    path = XRV_DIR / f"statcast_xrv_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"xRV data not found for {year}. Run build_xrv.py first.")
    df = pd.read_parquet(path)
    # Filter to regular season
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    return df


def prepare_matchup_data(
    df: pd.DataFrame,
    min_pitcher_pitches: int = 500,
    min_hitter_pa: int = 100,
) -> pd.DataFrame:
    """
    Prepare data for the matchup model.
    Filter to pitchers and hitters with enough data for stable estimates.
    Encode categoricals as integer indices.
    """
    # Filter to pitches with xRV
    df = df.dropna(subset=["xrv"]).copy()

    # Pitcher handedness-specific: split data by batter stand
    # This is critical — a pitcher's arsenal vs LHH is different from vs RHH

    # Filter to common pitch types
    common_types = df["pitch_type"].value_counts()
    common_types = common_types[common_types > 1000].index.tolist()
    df = df[df["pitch_type"].isin(common_types)]

    # Filter to pitchers with enough pitches
    pitcher_counts = df["pitcher"].value_counts()
    valid_pitchers = pitcher_counts[pitcher_counts >= min_pitcher_pitches].index
    df = df[df["pitcher"].isin(valid_pitchers)]

    # Filter to hitters with enough PA (approx: events not null = end of PA)
    hitter_pa = df.groupby("batter")["events"].apply(lambda x: x.notna().sum())
    valid_hitters = hitter_pa[hitter_pa >= min_hitter_pa].index
    df = df[df["batter"].isin(valid_hitters)]

    # Create integer indices
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

    # Create matchup index (pitcher, hitter pair)
    df["matchup_key"] = df["pitcher"].astype(str) + "_" + df["batter"].astype(str)
    matchup_ids = df["matchup_key"].unique()
    matchup_map = {mk: i for i, mk in enumerate(matchup_ids)}
    df["matchup_idx"] = df["matchup_key"].map(matchup_map)
    print(f"  Unique matchups: {len(matchup_ids):,}")

    # Store mappings
    df.attrs["pitcher_map"] = pitcher_map
    df.attrs["hitter_map"] = hitter_map
    df.attrs["ptype_map"] = ptype_map
    df.attrs["matchup_map"] = matchup_map
    df.attrs["pitcher_ids"] = pitcher_ids
    df.attrs["hitter_ids"] = hitter_ids

    return df


def standardize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize continuous pitch features."""
    feature_cols = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z"]
    stats = {}
    for col in feature_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f"{col}_z"] = (df[col] - mean) / std
            else:
                df[f"{col}_z"] = 0.0
            stats[col] = {"mean": mean, "std": std}
    return df, stats


def build_model(df: pd.DataFrame, sample: bool = False):
    """
    Build the Bayesian hierarchical matchup model.

    Model structure:
      xrv ~ Normal(μ, σ)
      μ = intercept
        + β_speed * release_speed
        + β_hmov * pfx_x
        + β_vmov * pfx_z
        + β_locx * plate_x
        + β_locz * plate_z
        + β_count * (balls - strikes)
        + pitch_type_effect[pitch_type]
        + pitcher_effect[pitcher]
        + hitter_effect[hitter]
        + hitter_ptype_effect[hitter, pitch_type]  # how this hitter handles this pitch type
        + matchup_effect[matchup]                   # specific matchup residual (heavy shrinkage)
    """
    n_pitchers = df["pitcher_idx"].nunique()
    n_hitters = df["hitter_idx"].nunique()
    n_ptypes = df["ptype_idx"].nunique()
    n_matchups = df["matchup_idx"].nunique()

    print(f"\n  Building PyMC model:")
    print(f"    {n_pitchers} pitchers, {n_hitters} hitters, {n_ptypes} pitch types")
    print(f"    {n_matchups:,} matchups, {len(df):,} observations")

    # Prepare arrays
    xrv = df["xrv"].values.astype(np.float64)
    pitcher_idx = df["pitcher_idx"].values.astype(int)
    hitter_idx = df["hitter_idx"].values.astype(int)
    ptype_idx = df["ptype_idx"].values.astype(int)
    matchup_idx = df["matchup_idx"].values.astype(int)

    # Standardized features
    speed_z = df["release_speed_z"].values.astype(np.float64) if "release_speed_z" in df.columns else np.zeros(len(df))
    hmov_z = df["pfx_x_z"].values.astype(np.float64) if "pfx_x_z" in df.columns else np.zeros(len(df))
    vmov_z = df["pfx_z_z"].values.astype(np.float64) if "pfx_z_z" in df.columns else np.zeros(len(df))
    locx_z = df["plate_x_z"].values.astype(np.float64) if "plate_x_z" in df.columns else np.zeros(len(df))
    locz_z = df["plate_z_z"].values.astype(np.float64) if "plate_z_z" in df.columns else np.zeros(len(df))
    count_diff = (df["balls"].fillna(0) - df["strikes"].fillna(0)).values.astype(np.float64)

    # Fill NaN features with 0 (already standardized, so 0 = mean)
    for arr in [speed_z, hmov_z, vmov_z, locx_z, locz_z]:
        arr[np.isnan(arr)] = 0.0

    coords = {
        "pitcher": np.arange(n_pitchers),
        "hitter": np.arange(n_hitters),
        "pitch_type": np.arange(n_ptypes),
    }

    with pm.Model(coords=coords) as model:
        # === Priors ===

        # Global intercept
        intercept = pm.Normal("intercept", mu=0, sigma=0.1)

        # Pitch characteristic effects (population level)
        beta_speed = pm.Normal("beta_speed", mu=0, sigma=0.05)
        beta_hmov = pm.Normal("beta_hmov", mu=0, sigma=0.05)
        beta_vmov = pm.Normal("beta_vmov", mu=0, sigma=0.05)
        beta_locx = pm.Normal("beta_locx", mu=0, sigma=0.05)
        beta_locz = pm.Normal("beta_locz", mu=0, sigma=0.05)
        beta_count = pm.Normal("beta_count", mu=0, sigma=0.05)

        # Pitch type effects (population level)
        sigma_ptype = pm.HalfNormal("sigma_ptype", sigma=0.05)
        ptype_effect = pm.Normal("ptype_effect", mu=0, sigma=sigma_ptype, dims="pitch_type")

        # Pitcher random effects
        sigma_pitcher = pm.HalfNormal("sigma_pitcher", sigma=0.05)
        pitcher_effect = pm.Normal("pitcher_effect", mu=0, sigma=sigma_pitcher, dims="pitcher")

        # Hitter random effects
        sigma_hitter = pm.HalfNormal("sigma_hitter", sigma=0.05)
        hitter_effect = pm.Normal("hitter_effect", mu=0, sigma=sigma_hitter, dims="hitter")

        # Hitter x pitch type interaction (key for the model!)
        # How does this hitter handle fastballs vs sliders vs curves?
        sigma_hitter_ptype = pm.HalfNormal("sigma_hitter_ptype", sigma=0.03)
        hitter_ptype_effect = pm.Normal(
            "hitter_ptype_effect",
            mu=0,
            sigma=sigma_hitter_ptype,
            shape=(n_hitters, n_ptypes),
        )

        # Matchup random effects (heavily shrunk — this is the pooling)
        sigma_matchup = pm.HalfNormal("sigma_matchup", sigma=0.02)
        matchup_effect = pm.Normal("matchup_effect", mu=0, sigma=sigma_matchup, shape=n_matchups)

        # === Linear predictor ===
        mu = (
            intercept
            + beta_speed * speed_z
            + beta_hmov * hmov_z
            + beta_vmov * vmov_z
            + beta_locx * locx_z
            + beta_locz * locz_z
            + beta_count * count_diff
            + ptype_effect[ptype_idx]
            + pitcher_effect[pitcher_idx]
            + hitter_effect[hitter_idx]
            + hitter_ptype_effect[hitter_idx, ptype_idx]
            + matchup_effect[matchup_idx]
        )

        # === Likelihood ===
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.2)
        obs = pm.Normal("obs", mu=mu, sigma=sigma_obs, observed=xrv)

    print(f"    Model built. Parameters: ~{n_pitchers + n_hitters + n_hitters*n_ptypes + n_matchups + 10:,}")

    if sample:
        print("\n  Sampling (MCMC)...")
        with model:
            trace = pm.sample(
                draws=1000,
                tune=1000,
                chains=4,
                cores=4,
                target_accept=0.9,
                return_inferencedata=True,
            )
        return model, trace
    else:
        # MAP estimate (fast, for iteration)
        print("\n  Finding MAP estimate (fast mode)...")
        with model:
            map_estimate = pm.find_MAP(maxeval=10000)
        return model, map_estimate


def predict_matchup_xrv(
    map_estimate: dict,
    pitcher_recent_pitches: pd.DataFrame,
    hitter_idx: int,
    feature_stats: dict,
    ptype_map: dict,
) -> float:
    """
    Predict expected xRV for a hitter against a pitcher's recent pitch mix.

    Args:
        map_estimate: MAP parameter estimates from the model
        pitcher_recent_pitches: DataFrame of the pitcher's last N pitches
            (against same handedness as hitter)
        hitter_idx: integer index of the hitter in the model
        feature_stats: standardization parameters
        ptype_map: pitch type -> index mapping

    Returns:
        Expected xRV per pitch (average across the pitcher's recent mix)
    """
    intercept = map_estimate["intercept"]
    beta_speed = map_estimate["beta_speed"]
    beta_hmov = map_estimate["beta_hmov"]
    beta_vmov = map_estimate["beta_vmov"]
    beta_locx = map_estimate["beta_locx"]
    beta_locz = map_estimate["beta_locz"]
    beta_count = map_estimate["beta_count"]
    ptype_effects = map_estimate["ptype_effect"]
    pitcher_effects = map_estimate["pitcher_effect"]
    hitter_effects = map_estimate["hitter_effect"]
    hitter_ptype_effects = map_estimate["hitter_ptype_effect"]
    matchup_effects = map_estimate["matchup_effect"]

    # For each of the pitcher's recent pitches, compute expected xRV for this hitter
    xrvs = []
    pitcher_idx = pitcher_recent_pitches["pitcher_idx"].iloc[0]

    for _, pitch in pitcher_recent_pitches.iterrows():
        ptype = pitch.get("pitch_type", "")
        if ptype not in ptype_map:
            continue

        pt_idx = ptype_map[ptype]

        # Standardize features
        def z(col):
            if col in feature_stats and col in pitch.index:
                val = pitch[col]
                if pd.isna(val):
                    return 0.0
                return (val - feature_stats[col]["mean"]) / feature_stats[col]["std"]
            return 0.0

        mu = (
            intercept
            + beta_speed * z("release_speed")
            + beta_hmov * z("pfx_x")
            + beta_vmov * z("pfx_z")
            + beta_locx * z("plate_x")
            + beta_locz * z("plate_z")
            + beta_count * 0  # use neutral count for pregame prediction
            + ptype_effects[pt_idx]
            + pitcher_effects[pitcher_idx]
            + hitter_effects[hitter_idx]
            + hitter_ptype_effects[hitter_idx, pt_idx]
            # Skip matchup effect for prediction (or use 0 if unseen)
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
):
    """Save model artifacts for later use in prediction."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "year": year,
        "feature_stats": feature_stats,
        "pitcher_map": df.attrs.get("pitcher_map", {}),
        "hitter_map": df.attrs.get("hitter_map", {}),
        "ptype_map": df.attrs.get("ptype_map", {}),
        "matchup_map": df.attrs.get("matchup_map", {}),
        "pitcher_ids": df.attrs.get("pitcher_ids", []),
        "hitter_ids": df.attrs.get("hitter_ids", []),
    }

    if is_map:
        # Convert MAP estimate arrays to regular numpy
        map_clean = {}
        for k, v in model_result.items():
            if hasattr(v, "eval"):
                map_clean[k] = np.array(v.eval())
            else:
                map_clean[k] = np.array(v)
        artifacts["map_estimate"] = map_clean
    else:
        artifacts["trace"] = model_result

    out_path = MODEL_DIR / f"matchup_model_{year}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"  Saved model artifacts to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Bayesian Hierarchical Matchup Model")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--sample", action="store_true",
                        help="Run full MCMC sampling (slow). Default: MAP estimate.")
    parser.add_argument("--min-pitcher-pitches", type=int, default=500)
    parser.add_argument("--min-hitter-pa", type=int, default=100)
    parser.add_argument("--subsample", type=int, default=0,
                        help="Subsample N pitches for faster iteration (0=all)")
    args = parser.parse_args()

    print(f"Loading xRV data for {args.season}...")
    df = load_season_xrv(args.season)
    print(f"  {len(df):,} pitches loaded")

    print(f"\nPreparing matchup data...")
    df = prepare_matchup_data(
        df,
        min_pitcher_pitches=args.min_pitcher_pitches,
        min_hitter_pa=args.min_hitter_pa,
    )

    print(f"\nStandardizing features...")
    df, feature_stats = standardize_features(df)

    if args.subsample > 0 and args.subsample < len(df):
        print(f"\n  Subsampling to {args.subsample:,} pitches for faster iteration")
        df = df.sample(n=args.subsample, random_state=42)

    print(f"\nBuilding model...")
    model, result = build_model(df, sample=args.sample)

    is_map = not args.sample
    save_model_artifacts(args.season, result, feature_stats, df, is_map=is_map)

    if is_map:
        # Print some interesting estimates
        print(f"\n  Key MAP estimates:")
        for param in ["intercept", "beta_speed", "beta_hmov", "beta_vmov",
                       "beta_locx", "beta_locz", "beta_count"]:
            if param in result:
                val = float(np.array(result[param]))
                print(f"    {param}: {val:.6f}")

        print(f"    sigma_pitcher: {float(np.array(result['sigma_pitcher'])):.6f}")
        print(f"    sigma_hitter: {float(np.array(result['sigma_hitter'])):.6f}")
        print(f"    sigma_hitter_ptype: {float(np.array(result['sigma_hitter_ptype'])):.6f}")
        print(f"    sigma_matchup: {float(np.array(result['sigma_matchup'])):.6f}")
        print(f"    sigma_obs: {float(np.array(result['sigma_obs'])):.6f}")

        # Show top/bottom pitchers and hitters
        pitcher_effects = np.array(result["pitcher_effect"])
        hitter_effects = np.array(result["hitter_effect"])
        pitcher_ids = df.attrs.get("pitcher_ids", [])
        hitter_ids = df.attrs.get("hitter_ids", [])

        if len(pitcher_ids) > 0:
            top_p = np.argsort(pitcher_effects)[:5]  # lowest xRV = best pitcher
            bot_p = np.argsort(pitcher_effects)[-5:]
            print(f"\n  Best pitchers (lowest xRV effect):")
            for i in top_p:
                print(f"    Pitcher {pitcher_ids[i]}: {pitcher_effects[i]:.6f}")
            print(f"  Worst pitchers (highest xRV effect):")
            for i in bot_p:
                print(f"    Pitcher {pitcher_ids[i]}: {pitcher_effects[i]:.6f}")

        if len(hitter_ids) > 0:
            top_h = np.argsort(hitter_effects)[-5:]  # highest xRV = best hitter
            bot_h = np.argsort(hitter_effects)[:5]
            print(f"\n  Best hitters (highest xRV effect):")
            for i in top_h:
                print(f"    Hitter {hitter_ids[i]}: {hitter_effects[i]:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
