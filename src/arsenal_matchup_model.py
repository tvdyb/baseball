#!/usr/bin/env python3
"""
Arsenal-based Bayesian Matchup Model

Key insight: A hitter has never faced this specific pitcher, but they've
faced hundreds of pitchers with similar arsenals. Instead of estimating
sparse pitcher-hitter matchup effects (most with <20 pitches), we learn
how each hitter responds to pitcher ARSENAL PROFILES.

We characterize each pitcher's arsenal along ~6 dimensions (fastball velo,
breaking ball usage, movement profile, etc.), then model each hitter's
sensitivity to these dimensions with hierarchical shrinkage.

Model:
  xrv ~ Normal(μ, σ²)
  μ = intercept
    + β · pitch_features                        # population pitch response
    + ptype_effect[pitch_type]                   # pitch type baseline
    + pitcher_effect[pitcher]                    # pitcher overall stuff quality
    + hitter_effect[hitter]                      # hitter overall quality
    + hitter_ptype_effect[hitter, pitch_type]    # hitter × pitch type
    + Σ_k hitter_arsenal_β[hitter, k] · arsenal[pitcher, k]  # hitter × arsenal

  hitter_arsenal_β[h, k] ~ Normal(μ_arsenal_k, σ_arsenal_k)

The arsenal interaction replaces the old matchup_effect[pitcher, hitter] term.
Instead of ~100K sparse matchup parameters, we have n_hitters × n_arsenal_features
(~4K × 6 = 24K) parameters that generalize to unseen matchups.

Usage:
    python src/arsenal_matchup_model.py --season 2024
    python src/arsenal_matchup_model.py --season 2024 --sample  # full MCMC
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
XRV_DIR = DATA_DIR / "xrv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

HARD_TYPES = {"FF", "SI", "FC"}
BREAK_TYPES = {"SL", "CU", "KC", "SV", "ST", "SW"}
OFFSPEED_TYPES = {"CH", "FS"}


def load_season_xrv(year: int) -> pd.DataFrame:
    path = XRV_DIR / f"statcast_xrv_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"xRV data not found for {year}. Run build_xrv.py first.")
    df = pd.read_parquet(path)
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    return df


# ──────────────────────────────────────────────────────────────
# 1. Pitcher Arsenal Profiles
# ──────────────────────────────────────────────────────────────

def compute_arsenal_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute arsenal profile for each pitcher from their season data.

    Features:
      hard_velo      - avg fastball velocity
      hard_pct       - % fastballs (FF/SI/FC)
      break_pct      - % breaking balls (SL/CU/KC/SV/ST/SW)
      offspeed_pct   - % offspeed (CH/FS)
      velo_spread    - velocity range (max pitch type avg - min pitch type avg)
      hmov_range     - horizontal movement range across pitch types
      entropy        - pitch mix entropy (Shannon)
    """
    profiles = []

    for pitcher_id, grp in df.groupby("pitcher"):
        n = len(grp)
        if n < 200:
            continue

        # Pitch type distribution
        pt_counts = grp["pitch_type"].value_counts()
        pt_pcts = pt_counts / n

        hard_pct = sum(pt_pcts.get(t, 0) for t in HARD_TYPES)
        break_pct = sum(pt_pcts.get(t, 0) for t in BREAK_TYPES)
        offspeed_pct = sum(pt_pcts.get(t, 0) for t in OFFSPEED_TYPES)

        # Entropy
        pcts = pt_pcts.values
        pcts = pcts[pcts > 0]
        entropy = -np.sum(pcts * np.log2(pcts))

        # Avg fastball velocity
        hard_pitches = grp[grp["pitch_type"].isin(HARD_TYPES)]
        hard_velo = hard_pitches["release_speed"].mean() if len(hard_pitches) > 0 else np.nan

        # Velocity spread: avg velo per pitch type, then max - min
        pt_velos = grp.groupby("pitch_type")["release_speed"].mean()
        velo_spread = pt_velos.max() - pt_velos.min() if len(pt_velos) > 1 else 0.0

        # Horizontal movement range
        pt_hmov = grp.groupby("pitch_type")["pfx_x"].mean()
        hmov_range = pt_hmov.max() - pt_hmov.min() if len(pt_hmov) > 1 else 0.0

        profiles.append({
            "pitcher": pitcher_id,
            "hard_velo": hard_velo,
            "hard_pct": hard_pct,
            "break_pct": break_pct,
            "offspeed_pct": offspeed_pct,
            "velo_spread": velo_spread,
            "hmov_range": hmov_range,
            "entropy": entropy,
            "n_pitches": n,
        })

    return pd.DataFrame(profiles)


# ──────────────────────────────────────────────────────────────
# 2. Data Preparation
# ──────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    arsenal_profiles: pd.DataFrame,
    min_pitcher_pitches: int = 500,
    min_hitter_pa: int = 100,
) -> tuple[pd.DataFrame, dict]:
    """Prepare data with arsenal features attached to each pitch."""
    df = df.dropna(subset=["xrv"]).copy()

    # Filter to common pitch types
    common_types = df["pitch_type"].value_counts()
    common_types = common_types[common_types > 1000].index.tolist()
    df = df[df["pitch_type"].isin(common_types)]

    # Filter pitchers/hitters by volume
    pitcher_counts = df["pitcher"].value_counts()
    valid_pitchers = pitcher_counts[pitcher_counts >= min_pitcher_pitches].index
    df = df[df["pitcher"].isin(valid_pitchers)]

    hitter_pa = df.groupby("batter")["events"].apply(lambda x: x.notna().sum())
    valid_hitters = hitter_pa[hitter_pa >= min_hitter_pa].index
    df = df[df["batter"].isin(valid_hitters)]

    # Merge arsenal profiles onto pitch data
    arsenal_cols = ["pitcher", "hard_velo", "hard_pct", "break_pct",
                    "offspeed_pct", "velo_spread", "hmov_range", "entropy"]
    df = df.merge(
        arsenal_profiles[arsenal_cols],
        on="pitcher",
        how="inner",
    )

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

    # Standardize continuous features
    feature_cols = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z"]
    arsenal_feature_cols = ["hard_velo", "hard_pct", "break_pct",
                           "offspeed_pct", "velo_spread", "hmov_range", "entropy"]
    all_std_cols = feature_cols + arsenal_feature_cols

    feature_stats = {}
    for col in all_std_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f"{col}_z"] = (df[col] - mean) / std
            else:
                df[f"{col}_z"] = 0.0
            feature_stats[col] = {"mean": float(mean), "std": float(std)}

    mappings = {
        "pitcher_map": pitcher_map,
        "hitter_map": hitter_map,
        "ptype_map": ptype_map,
        "pitcher_ids": pitcher_ids,
        "hitter_ids": hitter_ids,
        "feature_stats": feature_stats,
    }

    return df, mappings


# ──────────────────────────────────────────────────────────────
# 3. Bayesian Model
# ──────────────────────────────────────────────────────────────

ARSENAL_FEATURES = ["hard_velo", "hard_pct", "break_pct",
                    "offspeed_pct", "velo_spread", "hmov_range", "entropy"]
N_ARSENAL = len(ARSENAL_FEATURES)


def build_model(df: pd.DataFrame, sample: bool = False):
    """
    Build hierarchical model with arsenal interactions.

    Key difference from matchup_model.py: instead of
        matchup_effect[pitcher_i, hitter_j]  (sparse, ~100K params)
    we have:
        Σ_k hitter_arsenal_β[hitter_j, k] × arsenal_z[pitcher_i, k]  (dense, ~4K×7 params)

    Each hitter learns how they respond to each arsenal dimension,
    shrunk toward the population average response.
    """
    n_pitchers = df["pitcher_idx"].nunique()
    n_hitters = df["hitter_idx"].nunique()
    n_ptypes = df["ptype_idx"].nunique()

    print(f"\n  Building arsenal matchup model:")
    print(f"    {n_pitchers} pitchers, {n_hitters} hitters, {n_ptypes} pitch types")
    print(f"    {N_ARSENAL} arsenal features, {len(df):,} observations")
    print(f"    Arsenal interaction params: {n_hitters * N_ARSENAL:,}")

    # Prepare arrays
    xrv = df["xrv"].values.astype(np.float64)
    pitcher_idx = df["pitcher_idx"].values.astype(int)
    hitter_idx = df["hitter_idx"].values.astype(int)
    ptype_idx = df["ptype_idx"].values.astype(int)

    # Pitch-level features (standardized)
    speed_z = df["release_speed_z"].values.astype(np.float64) if "release_speed_z" in df.columns else np.zeros(len(df))
    hmov_z = df["pfx_x_z"].values.astype(np.float64) if "pfx_x_z" in df.columns else np.zeros(len(df))
    vmov_z = df["pfx_z_z"].values.astype(np.float64) if "pfx_z_z" in df.columns else np.zeros(len(df))
    locx_z = df["plate_x_z"].values.astype(np.float64) if "plate_x_z" in df.columns else np.zeros(len(df))
    locz_z = df["plate_z_z"].values.astype(np.float64) if "plate_z_z" in df.columns else np.zeros(len(df))
    count_diff = (df["balls"].fillna(0) - df["strikes"].fillna(0)).values.astype(np.float64)

    # Arsenal features (standardized, pitcher-level but repeated per pitch)
    arsenal_matrix = np.column_stack([
        df[f"{feat}_z"].values.astype(np.float64) for feat in ARSENAL_FEATURES
    ])

    # Fill NaN
    for arr in [speed_z, hmov_z, vmov_z, locx_z, locz_z]:
        arr[np.isnan(arr)] = 0.0
    arsenal_matrix[np.isnan(arsenal_matrix)] = 0.0

    with pm.Model() as model:
        # === Population-level pitch effects ===
        intercept = pm.Normal("intercept", mu=0, sigma=0.1)
        beta_speed = pm.Normal("beta_speed", mu=0, sigma=0.05)
        beta_hmov = pm.Normal("beta_hmov", mu=0, sigma=0.05)
        beta_vmov = pm.Normal("beta_vmov", mu=0, sigma=0.05)
        beta_locx = pm.Normal("beta_locx", mu=0, sigma=0.05)
        beta_locz = pm.Normal("beta_locz", mu=0, sigma=0.05)
        beta_count = pm.Normal("beta_count", mu=0, sigma=0.05)

        # Pitch type effects
        sigma_ptype = pm.HalfNormal("sigma_ptype", sigma=0.05)
        ptype_effect = pm.Normal("ptype_effect", mu=0, sigma=sigma_ptype,
                                 shape=n_ptypes)

        # === Pitcher stuff quality ===
        sigma_pitcher = pm.HalfNormal("sigma_pitcher", sigma=0.05)
        pitcher_effect = pm.Normal("pitcher_effect", mu=0, sigma=sigma_pitcher,
                                   shape=n_pitchers)

        # === Hitter baseline ===
        sigma_hitter = pm.HalfNormal("sigma_hitter", sigma=0.05)
        hitter_effect = pm.Normal("hitter_effect", mu=0, sigma=sigma_hitter,
                                  shape=n_hitters)

        # Hitter × pitch type interaction
        sigma_hitter_ptype = pm.HalfNormal("sigma_hitter_ptype", sigma=0.03)
        hitter_ptype_effect = pm.Normal(
            "hitter_ptype_effect", mu=0, sigma=sigma_hitter_ptype,
            shape=(n_hitters, n_ptypes),
        )

        # === NEW: Hitter × arsenal interaction ===
        # Population-level arsenal response (how does avg hitter respond to arsenal features)
        mu_arsenal = pm.Normal("mu_arsenal", mu=0, sigma=0.03, shape=N_ARSENAL)
        sigma_arsenal = pm.HalfNormal("sigma_arsenal", sigma=0.02, shape=N_ARSENAL)

        # Hitter-specific arsenal sensitivities (shrunk toward population)
        hitter_arsenal_beta = pm.Normal(
            "hitter_arsenal_beta",
            mu=mu_arsenal,          # each hitter shrunk toward population response
            sigma=sigma_arsenal,
            shape=(n_hitters, N_ARSENAL),
        )

        # === Linear predictor ===
        # Pitch-level effects
        mu_pitch = (
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
        )

        # Arsenal interaction: for each pitch, dot product of
        # this hitter's arsenal sensitivities with this pitcher's arsenal profile
        arsenal_interaction = pm.math.sum(
            hitter_arsenal_beta[hitter_idx] * arsenal_matrix,
            axis=1,
        )

        mu_total = mu_pitch + arsenal_interaction

        # === Likelihood ===
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.2)
        obs = pm.Normal("obs", mu=mu_total, sigma=sigma_obs, observed=xrv)

    total_params = (n_pitchers + n_hitters + n_hitters * n_ptypes
                    + n_hitters * N_ARSENAL + N_ARSENAL * 2 + 10)
    print(f"    Total parameters: ~{total_params:,}")

    if sample:
        print("\n  Sampling (MCMC)...")
        with model:
            trace = pm.sample(
                draws=1000, tune=1000, chains=4, cores=4,
                target_accept=0.9, return_inferencedata=True,
            )
        return model, trace
    else:
        print("\n  Finding MAP estimate...")
        with model:
            map_estimate = pm.find_MAP(maxeval=15000)
        return model, map_estimate


# ──────────────────────────────────────────────────────────────
# 4. Prediction
# ──────────────────────────────────────────────────────────────

def predict_hitter_vs_arsenal(
    map_estimate: dict,
    pitcher_bases: dict,
    pitcher_arsenal_z: np.ndarray,
    hitter_id: int,
    hitter_map: dict,
) -> float:
    """
    Predict xRV for a hitter against a pitcher characterized by
    their pitch bases and arsenal profile.

    pitcher_bases: precomputed from _precompute_pitcher_bases_arsenal()
    pitcher_arsenal_z: standardized arsenal feature vector (length N_ARSENAL)
    """
    hitter_effects = np.array(map_estimate["hitter_effect"])
    hitter_ptype_effects = np.array(map_estimate["hitter_ptype_effect"])
    hitter_arsenal_betas = np.array(map_estimate["hitter_arsenal_beta"])
    mu_arsenal = np.array(map_estimate["mu_arsenal"])

    if hitter_id in hitter_map:
        h_idx = hitter_map[hitter_id]
        h_effect = hitter_effects[h_idx]
        h_ptype = hitter_ptype_effects[h_idx]
        h_arsenal = hitter_arsenal_betas[h_idx]
    else:
        # Unknown hitter: use population average
        h_effect = 0.0
        h_ptype = np.zeros(hitter_ptype_effects.shape[1])
        h_arsenal = mu_arsenal

    # Arsenal interaction: dot product of hitter's sensitivities × pitcher's arsenal
    arsenal_bonus = np.dot(h_arsenal, pitcher_arsenal_z)

    # Average across pitcher's recent pitches (pitch_base already includes
    # intercept + pitch features + pitcher_effect + ptype_effect)
    hand = list(pitcher_bases.keys())[0] if pitcher_bases else None
    if hand is None or pitcher_bases.get(hand) is None:
        return 0.0

    pitch_base, ptype_idx = pitcher_bases[hand]
    hitter_preds = pitch_base + h_effect + h_ptype[ptype_idx] + arsenal_bonus

    return float(np.mean(hitter_preds))


# ──────────────────────────────────────────────────────────────
# 5. Save / Load
# ──────────────────────────────────────────────────────────────

def save_artifacts(year, map_estimate, mappings, is_map=True):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "year": year,
        "model_type": "arsenal",
        "arsenal_features": ARSENAL_FEATURES,
        **mappings,
    }

    if is_map:
        map_clean = {}
        for k, v in map_estimate.items():
            if hasattr(v, "eval"):
                map_clean[k] = np.array(v.eval())
            else:
                map_clean[k] = np.array(v)
        artifacts["map_estimate"] = map_clean

    out_path = MODEL_DIR / f"arsenal_matchup_{year}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"  Saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Arsenal-based Matchup Model")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--min-pitcher-pitches", type=int, default=500)
    parser.add_argument("--min-hitter-pa", type=int, default=100)
    parser.add_argument("--subsample", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading xRV data for {args.season}...")
    df = load_season_xrv(args.season)
    print(f"  {len(df):,} pitches loaded")

    print(f"\nComputing pitcher arsenal profiles...")
    arsenal_profiles = compute_arsenal_profiles(df)
    print(f"  {len(arsenal_profiles)} pitcher profiles computed")

    # Show arsenal stats
    for col in ARSENAL_FEATURES:
        if col in arsenal_profiles.columns:
            print(f"    {col}: mean={arsenal_profiles[col].mean():.2f}, "
                  f"std={arsenal_profiles[col].std():.2f}")

    print(f"\nPreparing data...")
    df_prep, mappings = prepare_data(
        df, arsenal_profiles,
        min_pitcher_pitches=args.min_pitcher_pitches,
        min_hitter_pa=args.min_hitter_pa,
    )

    if args.subsample > 0 and args.subsample < len(df_prep):
        print(f"\n  Subsampling to {args.subsample:,} pitches")
        df_prep = df_prep.sample(n=args.subsample, random_state=42)

    print(f"\nFitting model...")
    model, result = build_model(df_prep, sample=args.sample)

    is_map = not args.sample
    save_artifacts(args.season, result, mappings, is_map=is_map)

    if is_map:
        print(f"\n  Key MAP estimates:")
        for param in ["intercept", "beta_speed", "beta_hmov", "beta_vmov",
                       "beta_locx", "beta_locz", "beta_count"]:
            if param in result:
                print(f"    {param}: {float(np.array(result[param])):.6f}")

        print(f"\n  Variance components:")
        for param in ["sigma_pitcher", "sigma_hitter", "sigma_hitter_ptype", "sigma_obs"]:
            if param in result:
                print(f"    {param}: {float(np.array(result[param])):.6f}")

        # Arsenal interaction parameters
        mu_arsenal = np.array(result["mu_arsenal"])
        sigma_arsenal = np.array(result["sigma_arsenal"])
        print(f"\n  Arsenal population effects (μ_arsenal):")
        for feat, mu, sig in zip(ARSENAL_FEATURES, mu_arsenal, sigma_arsenal):
            print(f"    {feat:>15}: μ={mu:+.6f}  σ={sig:.6f}")

        # Hitter arsenal variance — how much do hitters differ in sensitivity?
        hitter_arsenal = np.array(result["hitter_arsenal_beta"])
        print(f"\n  Hitter arsenal sensitivity spread (std across hitters):")
        for i, feat in enumerate(ARSENAL_FEATURES):
            std = hitter_arsenal[:, i].std()
            print(f"    {feat:>15}: {std:.6f}")

        # Top/bottom pitchers
        pitcher_effects = np.array(result["pitcher_effect"])
        pitcher_ids = mappings["pitcher_ids"]
        if len(pitcher_ids) > 0:
            top_p = np.argsort(pitcher_effects)[:5]
            bot_p = np.argsort(pitcher_effects)[-5:]
            print(f"\n  Best pitchers (lowest xRV):")
            for i in top_p:
                print(f"    {pitcher_ids[i]}: {pitcher_effects[i]:.6f}")
            print(f"  Worst pitchers:")
            for i in bot_p:
                print(f"    {pitcher_ids[i]}: {pitcher_effects[i]:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
