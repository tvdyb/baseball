#!/usr/bin/env python3
"""
Pitcher sequencing/deception model (P3).

Captures what pitch specs + location can't: tunneling, sequencing, deception.
Uses Bayesian mixed-effects model on (actual delta_run_exp - P2 predicted xRV) residuals.

Per-pitcher effects are shrunk toward zero via hierarchical prior.
Only pitchers with 1500+ pitches get meaningful non-zero effects.

AR(1) component: exponentially decayed mean of pitcher's last 500 pitch residuals
is computed as a feature, not a model parameter.

Separate models for vs-RHH and vs-LHH.

Usage:
    python src/pitcher_sequencing_model.py --season 2024
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pitcher_stuff_model import STUFF_FEATURES, LOCATION_FEATURES, COMMON_PITCH_TYPES, _load_xrv
from utils import MODEL_DIR

MIN_PITCHES = 200  # Minimum for any non-zero effect (shrinkage handles the rest)
ACTIVATION_PITCHES = 1500  # Full activation threshold
RECENT_WINDOW = 500  # Pitches for AR(1) component
DECAY_HALFLIFE = 100  # Pitches for exponential decay


def _compute_p2_residuals(season: int, hand: str) -> pd.DataFrame:
    """Compute residuals: delta_run_exp - P2_predicted for a season and hand."""
    from pitcher_stuff_model import load_stuff_model

    # Load P2 model
    withloc = load_stuff_model(season, mode="withloc")
    model = withloc[f"model_vs{hand}"]
    features = withloc["features"]

    # Load xRV data
    xrv = _load_xrv(season)
    df = xrv[xrv["stand"] == hand].copy()
    df = df[df["pitch_type"].isin(COMMON_PITCH_TYPES)]
    df = df.dropna(subset=["xrv", "delta_run_exp"])

    # Label-encode pitch type (must match training encoding)
    df["pitch_type_code"] = df["pitch_type"].astype("category").cat.codes

    # Compute P2 predictions
    valid_mask = df[features].notna().all(axis=1)
    df = df[valid_mask].reset_index(drop=True)

    X = df[features].to_numpy(dtype=np.float64, na_value=np.nan)
    df["p2_pred"] = model.predict(X)
    df["residual"] = df["delta_run_exp"].to_numpy(dtype=np.float64) - df["p2_pred"].to_numpy(dtype=np.float64)

    # Sort by game_date, at_bat_number, pitch_number for temporal ordering
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    return df


def _compute_recent_residual_mean(df: pd.DataFrame) -> np.ndarray:
    """Compute exponentially decayed mean of recent residuals per pitcher.

    For each pitch, looks at that pitcher's previous pitches and computes
    an exponentially decayed mean of residuals.
    """
    alpha = 1 - np.exp(-np.log(2) / DECAY_HALFLIFE)
    recent_means = np.zeros(len(df))

    # Group by pitcher, compute rolling EMA of residuals
    for pid, grp in df.groupby("pitcher"):
        idx = grp.index.values
        resids = grp["residual"].values

        # Compute EMA
        ema = 0.0
        for i, (ix, r) in enumerate(zip(idx, resids)):
            if i == 0:
                recent_means[ix] = 0.0  # No history for first pitch
                ema = r
            else:
                recent_means[ix] = ema
                ema = alpha * r + (1 - alpha) * ema

    return recent_means


def train_sequencing_model(season: int) -> dict:
    """Train sequencing model for a season.

    Uses a simpler empirical Bayes approach instead of full PyMC MCMC
    for speed: estimate per-pitcher effects as shrunk group means.

    Returns dict with per-pitcher sequencing scores.
    """
    print(f"\n{'='*60}")
    print(f"  Pitcher Sequencing Model — {season}")
    print(f"{'='*60}")

    result = {"season": season}

    for hand in ["L", "R"]:
        print(f"\n  vs {hand}HH:")
        df = _compute_p2_residuals(season, hand)
        print(f"    Pitches with residuals: {len(df):,}")

        # Compute AR(1) feature
        df["recent_resid_mean"] = _compute_recent_residual_mean(df)

        # Count-based features
        df["count_diff"] = df["balls"].to_numpy(dtype=np.float64) - df["strikes"].to_numpy(dtype=np.float64)

        # Global stats for shrinkage
        global_mean = df["residual"].mean()
        global_var = df["residual"].var()
        print(f"    Global residual mean: {global_mean:.6f}, var: {global_var:.6f}")

        # Per-pitcher group stats
        pitcher_stats = df.groupby("pitcher").agg(
            n=("residual", "size"),
            mean_resid=("residual", "mean"),
            var_resid=("residual", "var"),
            mean_recent=("recent_resid_mean", "mean"),
        ).reset_index()

        # Empirical Bayes shrinkage
        # Estimate between-pitcher variance (tau^2)
        # Using method of moments: var(group_means) = tau^2 + sigma^2/n_avg
        group_means = pitcher_stats[pitcher_stats["n"] >= MIN_PITCHES]["mean_resid"]
        if len(group_means) < 10:
            print(f"    Too few pitchers with {MIN_PITCHES}+ pitches, skipping")
            result[f"sequencing_vs{hand}"] = {}
            continue

        n_avg = pitcher_stats[pitcher_stats["n"] >= MIN_PITCHES]["n"].mean()
        sigma2 = global_var  # Within-pitcher variance
        var_group_means = group_means.var()
        tau2 = max(0, var_group_means - sigma2 / n_avg)

        print(f"    Between-pitcher var (tau²): {tau2:.8f}")
        print(f"    Within-pitcher var (sigma²): {sigma2:.6f}")

        # Shrinkage estimator for each pitcher
        # theta_i = (n_i * tau2) / (n_i * tau2 + sigma2) * x_bar_i + (sigma2 / (n_i * tau2 + sigma2)) * mu
        pitcher_scores = {}
        for _, row in pitcher_stats.iterrows():
            pid = int(row["pitcher"])
            n = row["n"]
            x_bar = row["mean_resid"]

            if n < MIN_PITCHES:
                continue

            # Shrinkage weight (0 = fully shrunk to global, 1 = fully trust data)
            w = (n * tau2) / (n * tau2 + sigma2) if (n * tau2 + sigma2) > 0 else 0
            theta = w * x_bar + (1 - w) * global_mean

            # Activation ramp: full effect only at ACTIVATION_PITCHES
            activation = min(1.0, n / ACTIVATION_PITCHES)
            effective_theta = theta * activation

            pitcher_scores[pid] = {
                "sequencing_score": float(effective_theta),
                "raw_mean": float(x_bar),
                "shrinkage_weight": float(w),
                "n_pitches": int(n),
                "activation": float(activation),
                "ar1_mean": float(row["mean_recent"]),
            }

        # Summary
        scores = [v["sequencing_score"] for v in pitcher_scores.values()]
        active = [v for v in pitcher_scores.values() if v["activation"] >= 1.0]
        print(f"    Pitchers scored: {len(pitcher_scores)}")
        print(f"    Fully activated (>={ACTIVATION_PITCHES} pitches): {len(active)}")
        if scores:
            print(f"    Score range: [{min(scores):.6f}, {max(scores):.6f}]")
            print(f"    Score std: {np.std(scores):.6f}")

        # Best/worst
        sorted_scores = sorted(pitcher_scores.items(), key=lambda x: x[1]["sequencing_score"])
        if sorted_scores:
            best = sorted_scores[0]
            worst = sorted_scores[-1]
            print(f"    Best sequencing (lowest residual): {best[1]['sequencing_score']:.6f} "
                  f"(pid={best[0]}, n={best[1]['n_pitches']})")
            print(f"    Worst sequencing: {worst[1]['sequencing_score']:.6f} "
                  f"(pid={worst[0]}, n={worst[1]['n_pitches']})")

        result[f"sequencing_vs{hand}"] = pitcher_scores

    return result


def save_sequencing_model(result: dict):
    """Save sequencing model results."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"sequencing_{result['season']}.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n  Saved to {path}")


def load_sequencing_model(season: int) -> dict:
    """Load sequencing model."""
    path = MODEL_DIR / f"sequencing_{season}.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Pitcher Sequencing Model")
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    result = train_sequencing_model(args.season)
    save_sequencing_model(result)


if __name__ == "__main__":
    main()
