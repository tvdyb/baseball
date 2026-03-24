#!/usr/bin/env python3
"""
Arsenal-based Bayesian Matchup Model

Key insight: A hitter has never faced this specific pitcher, but they've
faced hundreds of pitchers with similar arsenals. Instead of estimating
sparse pitcher-hitter matchup effects (most with <20 pitches), we learn
how each hitter responds to pitcher ARSENAL PROFILES.

We characterize each pitcher's arsenal along 13 dimensions (fastball velo,
breaking ball usage, movement profile, spin, extension, etc.), then model
each hitter's sensitivity to these dimensions with hierarchical shrinkage.

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

Inference: ADVI (default), --map (fast debug), --sample (full MCMC).
Models are split by batter handedness (vs LHH and vs RHH separately).

Usage:
    python src/arsenal_matchup_model.py --season 2024
    python src/arsenal_matchup_model.py --season 2024 --map      # fast debug
    python src/arsenal_matchup_model.py --season 2024 --sample   # full MCMC
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

from utils import (
    DATA_DIR, XRV_DIR, MODEL_DIR,
    HARD_TYPES, BREAK_TYPES, OFFSPEED_TYPES,
    filter_competitive as _filter_competitive,
)

ARSENAL_FEATURES = [
    "hard_velo", "hard_pct", "break_pct", "offspeed_pct",
    "velo_spread", "hmov_range", "entropy",
    # Enriched:
    "hard_spin",      # avg spin rate on fastballs
    "break_spin",     # avg spin rate on breaking balls
    "hard_ivb",       # avg induced vertical break on fastballs (pfx_z)
    "hard_ext",       # avg extension on fastballs
    "rel_x_spread",   # std dev of release_pos_x across pitch types (arm slot consistency)
    "vmov_range",     # range of avg pfx_z across pitch types (vertical separation)
]
N_ARSENAL = len(ARSENAL_FEATURES)


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
    13 features capturing velocity, pitch mix, movement, spin, extension, and deception.
    """
    profiles = []

    for pitcher_id, grp in df.groupby("pitcher"):
        n = len(grp)
        if n < 200:
            continue

        pt_counts = grp["pitch_type"].value_counts()
        pt_pcts = pt_counts / n

        hard_pct = sum(pt_pcts.get(t, 0) for t in HARD_TYPES)
        break_pct = sum(pt_pcts.get(t, 0) for t in BREAK_TYPES)
        offspeed_pct = sum(pt_pcts.get(t, 0) for t in OFFSPEED_TYPES)

        pcts = pt_pcts.values
        pcts = pcts[pcts > 0]
        entropy = -np.sum(pcts * np.log2(pcts))

        hard_pitches = grp[grp["pitch_type"].isin(HARD_TYPES)]
        break_pitches = grp[grp["pitch_type"].isin(BREAK_TYPES)]

        hard_velo = hard_pitches["release_speed"].mean() if len(hard_pitches) > 0 else np.nan

        pt_velos = grp.groupby("pitch_type")["release_speed"].mean()
        velo_spread = pt_velos.max() - pt_velos.min() if len(pt_velos) > 1 else 0.0

        pt_hmov = grp.groupby("pitch_type")["pfx_x"].mean()
        hmov_range = pt_hmov.max() - pt_hmov.min() if len(pt_hmov) > 1 else 0.0

        # New features
        hard_spin = hard_pitches["release_spin_rate"].mean() if (len(hard_pitches) > 0 and "release_spin_rate" in grp.columns) else np.nan
        break_spin = break_pitches["release_spin_rate"].mean() if (len(break_pitches) > 0 and "release_spin_rate" in grp.columns) else np.nan
        hard_ivb = hard_pitches["pfx_z"].mean() if len(hard_pitches) > 0 else np.nan
        hard_ext = hard_pitches["release_extension"].mean() if (len(hard_pitches) > 0 and "release_extension" in grp.columns) else np.nan

        if "release_pos_x" in grp.columns:
            pt_rel_x = grp.groupby("pitch_type")["release_pos_x"].mean()
            rel_x_spread = pt_rel_x.std() if len(pt_rel_x) > 1 else 0.0
        else:
            rel_x_spread = 0.0

        pt_vmov = grp.groupby("pitch_type")["pfx_z"].mean()
        vmov_range = pt_vmov.max() - pt_vmov.min() if len(pt_vmov) > 1 else 0.0

        profiles.append({
            "pitcher": pitcher_id,
            "hard_velo": hard_velo,
            "hard_pct": hard_pct,
            "break_pct": break_pct,
            "offspeed_pct": offspeed_pct,
            "velo_spread": velo_spread,
            "hmov_range": hmov_range,
            "entropy": entropy,
            "hard_spin": hard_spin,
            "break_spin": break_spin,
            "hard_ivb": hard_ivb,
            "hard_ext": hard_ext,
            "rel_x_spread": rel_x_spread,
            "vmov_range": vmov_range,
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
    stand: str = None,
) -> tuple[pd.DataFrame, dict]:
    """Prepare data with arsenal features attached to each pitch."""
    df = df.dropna(subset=["xrv"]).copy()

    if stand:
        df = df[df["stand"] == stand]

    # Filter competitive counts
    df = _filter_competitive(df)

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
    arsenal_cols = ["pitcher"] + [f for f in ARSENAL_FEATURES if f in arsenal_profiles.columns]
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
    feature_cols = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z",
                    "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z"]
    arsenal_feature_cols = [f for f in ARSENAL_FEATURES]
    all_std_cols = feature_cols + arsenal_feature_cols

    feature_stats = {}
    for col in all_std_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if pd.notna(std) and std > 0:
                df[f"{col}_z"] = (df[col] - mean) / std
            else:
                df[f"{col}_z"] = 0.0
            feature_stats[col] = {"mean": float(mean) if pd.notna(mean) else 0.0,
                                  "std": float(std) if pd.notna(std) else 1.0}

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

def build_model(df: pd.DataFrame, sample: bool = False, use_map: bool = False):
    """
    Build hierarchical model with arsenal interactions.

    Inference methods:
      - ADVI (default): variational inference, proper posterior means
      - MAP (--map): fast but crushes variance components
      - MCMC (--sample): gold standard, slowest
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

    # Arsenal features (standardized, pitcher-level but repeated per pitch)
    arsenal_arrays = []
    for feat in ARSENAL_FEATURES:
        z_col = f"{feat}_z"
        if z_col in df.columns:
            arr = df[z_col].values.astype(np.float64)
        else:
            arr = np.zeros(len(df))
        arr[np.isnan(arr)] = 0.0
        arsenal_arrays.append(arr)
    arsenal_matrix = np.column_stack(arsenal_arrays)

    with pm.Model() as model:
        # === Population-level pitch effects ===
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

        # === Hitter × arsenal interaction ===
        mu_arsenal = pm.Normal("mu_arsenal", mu=0, sigma=0.03, shape=N_ARSENAL)
        sigma_arsenal = pm.HalfNormal("sigma_arsenal", sigma=0.02, shape=N_ARSENAL)
        hitter_arsenal_beta = pm.Normal(
            "hitter_arsenal_beta",
            mu=mu_arsenal,
            sigma=sigma_arsenal,
            shape=(n_hitters, N_ARSENAL),
        )

        # === Linear predictor ===
        mu_pitch = (
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
        )

        arsenal_interaction = pm.math.sum(
            hitter_arsenal_beta[hitter_idx] * arsenal_matrix,
            axis=1,
        )

        mu_total = mu_pitch + arsenal_interaction

        # === Likelihood ===
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.2)
        obs = pm.Normal("obs", mu=mu_total, sigma=sigma_obs, observed=xrv)

    total_params = (n_pitchers + n_hitters + n_hitters * n_ptypes
                    + n_hitters * N_ARSENAL + N_ARSENAL * 2 + 14)
    print(f"    Total parameters: ~{total_params:,}")

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
            map_estimate = pm.find_MAP(maxeval=15000)
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
        h_effect = 0.0
        h_ptype = np.zeros(hitter_ptype_effects.shape[1])
        h_arsenal = mu_arsenal

    arsenal_bonus = np.dot(h_arsenal, pitcher_arsenal_z)

    hand = list(pitcher_bases.keys())[0] if pitcher_bases else None
    if hand is None or pitcher_bases.get(hand) is None:
        return 0.0

    pitch_base, ptype_idx = pitcher_bases[hand]
    hitter_preds = pitch_base + h_effect + h_ptype[ptype_idx] + arsenal_bonus

    return float(np.mean(hitter_preds))


# ──────────────────────────────────────────────────────────────
# 5. Save / Load
# ──────────────────────────────────────────────────────────────

def save_artifacts(year, map_estimate, mappings, is_map=True, hand=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "year": year,
        "model_type": "arsenal",
        "arsenal_features": ARSENAL_FEATURES,
        "hand": hand,
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

    suffix = f"_vs{hand}" if hand else ""
    out_path = MODEL_DIR / f"arsenal_matchup_{year}{suffix}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"  Saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Arsenal-based Matchup Model")
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
        print(f"Training arsenal model vs {hand}HH")
        print(f"{'='*60}")

        df_hand = df[df["stand"] == hand]
        print(f"  {len(df_hand):,} pitches vs {hand}HH")

        print(f"\nComputing pitcher arsenal profiles (vs {hand}HH)...")
        arsenal_profiles = compute_arsenal_profiles(df_hand)
        print(f"  {len(arsenal_profiles)} pitcher profiles computed")

        for col in ARSENAL_FEATURES:
            if col in arsenal_profiles.columns:
                print(f"    {col}: mean={arsenal_profiles[col].mean():.2f}, "
                      f"std={arsenal_profiles[col].std():.2f}")

        print(f"\nPreparing data...")
        df_prep, mappings = prepare_data(
            df_hand, arsenal_profiles,
            min_pitcher_pitches=args.min_pitcher_pitches,
            min_hitter_pa=args.min_hitter_pa,
            stand=hand,
        )

        if args.subsample > 0 and args.subsample < len(df_prep):
            print(f"\n  Subsampling to {args.subsample:,} pitches")
            df_prep = df_prep.sample(n=args.subsample, random_state=42)

        print(f"\nFitting model...")
        model, result = build_model(df_prep, sample=args.sample, use_map=args.map)

        is_map = not args.sample  # both MAP and ADVI produce dict format
        save_artifacts(args.season, result, mappings, is_map=is_map, hand=hand)

        # Print diagnostics
        print(f"\n  Key estimates:")
        for param in ["intercept", "beta_speed", "beta_hmov", "beta_vmov",
                       "beta_locx", "beta_locz", "beta_count",
                       "beta_spin", "beta_ext", "beta_rel_x", "beta_rel_z"]:
            if param in result:
                print(f"    {param}: {float(np.array(result[param])):.6f}")

        print(f"\n  Variance components:")
        for param in ["sigma_pitcher", "sigma_hitter", "sigma_hitter_ptype", "sigma_obs"]:
            if param in result:
                print(f"    {param}: {float(np.array(result[param])):.6f}")

        if "mu_arsenal" in result:
            mu_arsenal = np.array(result["mu_arsenal"])
            sigma_arsenal = np.array(result["sigma_arsenal"])
            print(f"\n  Arsenal population effects (μ_arsenal):")
            for feat, mu, sig in zip(ARSENAL_FEATURES, mu_arsenal, sigma_arsenal):
                print(f"    {feat:>15}: μ={mu:+.6f}  σ={sig:.6f}")

        if "hitter_arsenal_beta" in result:
            hitter_arsenal = np.array(result["hitter_arsenal_beta"])
            print(f"\n  Hitter arsenal sensitivity spread (std across hitters):")
            for i, feat in enumerate(ARSENAL_FEATURES):
                std = hitter_arsenal[:, i].std()
                print(f"    {feat:>15}: {std:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
