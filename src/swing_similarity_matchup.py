#!/usr/bin/env python3
"""
Swing-similarity matchup model (M1).

For novel batter-pitcher pairs, find similar batters × similar pitchers
using embedding vectors and aggregate their historical PA outcomes.

Batter embedding (5D): swing_decision_z, iz_contact_skill, chase_contact_skill,
                       foul_fight_score, bip_quality_iz
Pitcher embedding (13D): ARSENAL_FEATURES from arsenal_matchup_model.py

Algorithm:
1. Standardize both embeddings
2. For (batter, pitcher): find K_BAT nearest batters, K_PIT nearest pitchers
3. Collect all PAs between any similar-batter and similar-pitcher
4. Weight by batter_sim × pitcher_sim
5. Compute weighted outcome distribution (11 categories)

Usage:
    python src/swing_similarity_matchup.py --season 2024
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arsenal_matchup_model import ARSENAL_FEATURES
from build_transition_matrix import OUTCOME_ORDER, EVENT_MAP, BBTYPE_MAP
from utils import MODEL_DIR, XRV_DIR

K_BAT = 20   # Number of similar batters
K_PIT = 15   # Number of similar pitchers
MIN_PA_POOL = 50  # Minimum weighted PAs for a valid prediction

BATTER_EMBEDDING_KEYS = [
    "swing_decision_z", "iz_contact_skill", "chase_contact_skill",
    "foul_fight_score", "bip_quality_iz",
]


def _load_hitter_eval(season: int) -> dict:
    """Load hitter evaluation embeddings."""
    path = MODEL_DIR / f"hitter_eval_{season}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_pitcher_arsenals(season: int) -> dict:
    """Compute pitcher arsenal embeddings from xRV data."""
    from feature_engineering import _compute_pitcher_arsenal_live, _preindex_xrv

    xrv_path = XRV_DIR / f"statcast_xrv_{season}.parquet"
    xrv = pd.read_parquet(xrv_path)
    if "game_type" in xrv.columns:
        xrv = xrv[xrv["game_type"] == "R"]

    idx = _preindex_xrv(xrv)
    # Use a date after the season ends to include all data
    end_date = pd.Timestamp(f"{season}-12-31")

    arsenals = {}
    for pid, pitcher_df in idx["pitcher"].items():
        pid = int(pid)
        arsenal = _compute_pitcher_arsenal_live(pitcher_df, end_date)
        if arsenal and len(arsenal) >= len(ARSENAL_FEATURES):
            vec = []
            for feat in ARSENAL_FEATURES:
                val = arsenal.get(feat, 0.0)
                if pd.isna(val):
                    val = 0.0
                vec.append(val)
            arsenals[pid] = np.array(vec, dtype=np.float64)

    return arsenals


def _classify_pa(events_val, bb_type_val) -> str:
    """Classify a PA outcome into one of 11 categories."""
    if pd.isna(events_val):
        return None
    ev = str(events_val).lower().replace(" ", "_")
    # Check EVENT_MAP first
    if ev in EVENT_MAP:
        return EVENT_MAP[ev]
    # Check bb_type for batted ball outcomes
    if not pd.isna(bb_type_val):
        bt = str(bb_type_val).lower()
        if bt in BBTYPE_MAP:
            return BBTYPE_MAP[bt]
    return None


def _build_pa_outcomes(season: int) -> pd.DataFrame:
    """Build PA-level outcome data for a season."""
    xrv_path = XRV_DIR / f"statcast_xrv_{season}.parquet"
    xrv = pd.read_parquet(xrv_path)
    if "game_type" in xrv.columns:
        xrv = xrv[xrv["game_type"] == "R"]

    # Filter to PA-ending pitches (those with events)
    pa = xrv[xrv["events"].notna()].copy()

    # Classify outcomes
    pa["outcome"] = pa.apply(lambda r: _classify_pa(r["events"], r.get("bb_type")), axis=1)
    pa = pa[pa["outcome"].notna() & pa["outcome"].isin(OUTCOME_ORDER)]

    return pa[["pitcher", "batter", "stand", "outcome"]].reset_index(drop=True)


def build_similarity_model(season: int) -> dict:
    """Build the similarity matchup model.

    Returns dict with:
        batter_embeddings: {batter_id: np.array(5)}
        pitcher_embeddings: {pitcher_id: np.array(13)}
        batter_scaler: StandardScaler
        pitcher_scaler: StandardScaler
        pa_outcomes: DataFrame of (pitcher, batter, stand, outcome)
        season: int
    """
    print(f"\n{'='*60}")
    print(f"  Swing Similarity Matchup — {season}")
    print(f"{'='*60}")

    # Load batter embeddings from composite dict
    hitter_eval = _load_hitter_eval(season)
    composite = hitter_eval.get("composite", {})
    batter_embs = {}
    for bid, entry in composite.items():
        vec = entry.get("embedding")
        if vec is not None and len(vec) == len(BATTER_EMBEDDING_KEYS):
            arr = np.array(vec, dtype=np.float64)
            if not np.any(np.isnan(arr)):
                batter_embs[int(bid)] = arr
    print(f"  Batter embeddings: {len(batter_embs)}")

    # Load pitcher arsenals
    print("  Computing pitcher arsenal embeddings...")
    pitcher_embs = _load_pitcher_arsenals(season)
    print(f"  Pitcher embeddings: {len(pitcher_embs)}")

    # Standardize
    bat_ids = sorted(batter_embs.keys())
    bat_matrix = np.array([batter_embs[b] for b in bat_ids])
    bat_scaler = StandardScaler().fit(bat_matrix)
    bat_matrix_scaled = bat_scaler.transform(bat_matrix)

    pit_ids = sorted(pitcher_embs.keys())
    pit_matrix = np.array([pitcher_embs[p] for p in pit_ids])
    pit_scaler = StandardScaler().fit(pit_matrix)
    pit_matrix_scaled = pit_scaler.transform(pit_matrix)

    # Build PA outcomes
    print("  Building PA outcomes...")
    pa = _build_pa_outcomes(season)
    print(f"  Total PAs: {len(pa):,}")

    # Pre-compute similarity matrices
    print("  Computing similarity matrices...")
    bat_sim = cosine_similarity(bat_matrix_scaled)
    pit_sim = cosine_similarity(pit_matrix_scaled)

    # Build lookup dicts for fast indexing
    bat_id_to_idx = {bid: i for i, bid in enumerate(bat_ids)}
    pit_id_to_idx = {pid: i for i, pid in enumerate(pit_ids)}

    result = {
        "batter_ids": bat_ids,
        "pitcher_ids": pit_ids,
        "batter_matrix_scaled": bat_matrix_scaled,
        "pitcher_matrix_scaled": pit_matrix_scaled,
        "batter_scaler": bat_scaler,
        "pitcher_scaler": pit_scaler,
        "bat_sim": bat_sim,
        "pit_sim": pit_sim,
        "bat_id_to_idx": bat_id_to_idx,
        "pit_id_to_idx": pit_id_to_idx,
        "pa_outcomes": pa,
        "season": season,
    }

    # Validation: sample some matchups
    print("\n  Sample matchup predictions:")
    n_valid = 0
    n_tried = 0
    rng = np.random.RandomState(42)
    sample_bats = rng.choice(bat_ids, min(50, len(bat_ids)), replace=False)
    sample_pits = rng.choice(pit_ids, min(50, len(pit_ids)), replace=False)

    for bid in sample_bats[:10]:
        for pid in sample_pits[:10]:
            n_tried += 1
            dist = predict_matchup(result, int(bid), int(pid), "R")
            if dist is not None:
                n_valid += 1

    print(f"    {n_valid}/{n_tried} sample matchups had sufficient PA pool")

    return result


def predict_matchup(model: dict, batter_id: int, pitcher_id: int, hand: str) -> np.ndarray:
    """Predict outcome distribution for a batter-pitcher matchup.

    Returns numpy array of shape (11,) with probabilities for each outcome
    in OUTCOME_ORDER, or None if insufficient data.
    """
    bat_idx_map = model["bat_id_to_idx"]
    pit_idx_map = model["pit_id_to_idx"]

    if batter_id not in bat_idx_map or pitcher_id not in pit_idx_map:
        return None

    bat_idx = bat_idx_map[batter_id]
    pit_idx = pit_idx_map[pitcher_id]

    # Get K nearest batters and pitchers
    bat_sims = model["bat_sim"][bat_idx]
    pit_sims = model["pit_sim"][pit_idx]

    # Top K (excluding self)
    bat_top_idx = np.argsort(bat_sims)[::-1]
    pit_top_idx = np.argsort(pit_sims)[::-1]

    sim_batter_ids = []
    sim_batter_weights = []
    for i in bat_top_idx:
        if len(sim_batter_ids) >= K_BAT:
            break
        bid = model["batter_ids"][i]
        if bid != batter_id and bat_sims[i] > 0:
            sim_batter_ids.append(bid)
            sim_batter_weights.append(bat_sims[i])

    sim_pitcher_ids = []
    sim_pitcher_weights = []
    for i in pit_top_idx:
        if len(sim_pitcher_ids) >= K_PIT:
            break
        pid = model["pitcher_ids"][i]
        if pid != pitcher_id and pit_sims[i] > 0:
            sim_pitcher_ids.append(pid)
            sim_pitcher_weights.append(pit_sims[i])

    if not sim_batter_ids or not sim_pitcher_ids:
        return None

    # Look up PAs between similar batters and similar pitchers
    pa = model["pa_outcomes"]
    mask = (
        pa["batter"].isin(sim_batter_ids) &
        pa["pitcher"].isin(sim_pitcher_ids) &
        (pa["stand"] == hand)
    )
    relevant_pas = pa[mask]

    if len(relevant_pas) < MIN_PA_POOL:
        return None

    # Build weight lookup
    bat_weight = dict(zip(sim_batter_ids, sim_batter_weights))
    pit_weight = dict(zip(sim_pitcher_ids, sim_pitcher_weights))

    # Compute weighted outcome distribution
    outcome_to_idx = {o: i for i, o in enumerate(OUTCOME_ORDER)}
    counts = np.zeros(len(OUTCOME_ORDER))

    for _, row in relevant_pas.iterrows():
        bw = bat_weight.get(int(row["batter"]), 0)
        pw = pit_weight.get(int(row["pitcher"]), 0)
        w = bw * pw
        oidx = outcome_to_idx.get(row["outcome"])
        if oidx is not None:
            counts[oidx] += w

    total = counts.sum()
    if total < 1e-10:
        return None

    return counts / total


def save_similarity_model(model: dict):
    """Save similarity model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"swing_similarity_{model['season']}.pkl"
    # Don't save the full PA dataframe and similarity matrices — they're large
    # Save what we need for prediction
    save_dict = {
        "batter_ids": model["batter_ids"],
        "pitcher_ids": model["pitcher_ids"],
        "batter_matrix_scaled": model["batter_matrix_scaled"],
        "pitcher_matrix_scaled": model["pitcher_matrix_scaled"],
        "batter_scaler": model["batter_scaler"],
        "pitcher_scaler": model["pitcher_scaler"],
        "bat_id_to_idx": model["bat_id_to_idx"],
        "pit_id_to_idx": model["pit_id_to_idx"],
        "pa_outcomes": model["pa_outcomes"],
        "season": model["season"],
    }
    with open(path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"\n  Saved to {path}")


def load_similarity_model(season: int) -> dict:
    """Load similarity model and reconstruct similarity matrices."""
    path = MODEL_DIR / f"swing_similarity_{season}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)

    # Reconstruct similarity matrices
    model["bat_sim"] = cosine_similarity(model["batter_matrix_scaled"])
    model["pit_sim"] = cosine_similarity(model["pitcher_matrix_scaled"])

    return model


def main():
    parser = argparse.ArgumentParser(description="Swing Similarity Matchup Model")
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    model = build_similarity_model(args.season)
    save_similarity_model(model)


if __name__ == "__main__":
    main()
