#!/usr/bin/env python3
"""
Game-level feature engineering.

For each game, compute pregame features with NO lookahead bias:
  1. Starting pitcher quality (xRV from recent pitches)
  2. Pitcher vs lineup matchup (Bayesian hierarchical model)
  3. Bullpen quality + availability (recent usage / rest)
  4. Team hitting quality (rolling xRV)
  5. Defense (rolling OAA / fielding metrics)
  6. Park factor
  7. Home/away advantage
  8. Weather (if available)

Every feature uses only data available BEFORE the game starts.
"""

import argparse
import pickle
import re
import unicodedata
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

from arsenal_matchup_model import ARSENAL_FEATURES
from utils import (
    DATA_DIR, XRV_DIR, GAMES_DIR, WEATHER_DIR, MODEL_DIR, FEATURES_DIR, OAA_DIR,
    ROSTER_DIR, HARD_TYPES, BREAK_TYPES, OFFSPEED_TYPES, filter_competitive,
)

# Backward-compat alias for internal callers
_filter_competitive = filter_competitive

# Differential columns: (feature_suffix, sign) where sign controls direction
# positive sign = higher home value is better for home team
DIFF_COLS = [
    ("sp_xrv_mean", -1),        # negative = better pitcher, so flip sign
    ("sp_k_rate", 1),
    ("sp_bb_rate", -1),
    ("sp_avg_velo", -1),         # higher velo = better for pitcher
    ("sp_rest_days", 1),         # more rest = better
    ("sp_overperf", -1),         # negative = pitcher beats their stuff (good for pitcher)
    ("sp_overperf_recent", -1),  # recent trend
    ("bp_xrv_mean", -1),
    ("bp_fatigue_score", -1),    # more fatigue = worse
    ("bp_matchup_xrv_mean", -1), # lower = bullpen is tougher vs opposing lineup
    ("arsenal_matchup_xrv_mean", 1),  # higher = lineup does better (arsenal model)
    ("arsenal_matchup_xrv_sum", 1),
    ("bp_arsenal_matchup_xrv_mean", -1),
    ("hit_xrv_mean", 1),        # higher = better for hitters
    ("hit_xrv_contact", 1),
    ("hit_k_rate", -1),
    ("def_xrv_delta", -1),      # more negative = better defense
    ("matchup_xrv_mean", 1),    # higher = lineup does better against opposing SP
    ("matchup_xrv_sum", 1),
    ("platoon_pct", 1),         # more platoon advantage = better
    ("recent_form", 1),         # higher win% = better
    # SP trend features
    ("sp_velo_trend", -1),      # velo increasing = better for pitcher
    ("sp_spin_trend", -1),      # spin increasing = better for pitcher
    ("sp_xrv_trend", -1),       # xRV trending up = worse for pitcher
    ("sp_transition_entropy", -1),  # higher entropy = more unpredictable = better for pitcher
    # OAA defense
    ("oaa_rate", 1),            # higher OAA = better defense
    # Team strength prior
    ("team_prior", 1),          # higher prior win% = stronger team
    # Trade deadline features
    ("trade_net", 1),           # more acquisitions = stronger
    ("trade_pitcher_xrv", -1),  # negative xRV = better pitcher acquired
    # Preseason projections (Step 3)
    ("projected_wpct", 1),      # higher projected win% = stronger
    ("sp_projected_era", -1),   # lower ERA = better pitcher
    ("sp_projected_war", 1),    # higher WAR = better pitcher
    # Roster-adjusted team prior (Step 6)
    ("adjusted_team_prior", 1),
]


def _load_hand_models(prefix: str, year: int) -> dict:
    """Load hand-specific models with fallback to combined model."""
    models = {}
    for hand in ["L", "R"]:
        path = MODEL_DIR / f"{prefix}_{year}_vs{hand}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[hand] = pickle.load(f)
            print(f"  Loaded {prefix} vs {hand}HH from {year}")
    # Fallback to combined model
    if not models:
        old_path = MODEL_DIR / f"{prefix}_{year}.pkl"
        if old_path.exists():
            with open(old_path, "rb") as f:
                combined = pickle.load(f)
            models = {"L": combined, "R": combined}
            print(f"  Loaded combined {prefix} from {year}")
    return models


# ──────────────────────────────────────────────────────────────
# 1. Starting Pitcher Quality
# ──────────────────────────────────────────────────────────────

def compute_pitcher_rolling_xrv(
    xrv_df: pd.DataFrame,
    game_date: str,
    pitcher_id: int,
    n_pitches: int = 2000,
    handedness: str = None,
) -> dict:
    """
    Compute rolling xRV stats for a pitcher using their last N pitches
    BEFORE the game date. Optionally filter by batter handedness.

    Returns dict with mean xRV, pitch mix, velocity trends, etc.
    """
    mask = (
        (xrv_df["pitcher"] == pitcher_id)
        & (xrv_df["game_date"] < game_date)
    )
    if handedness:
        mask = mask & (xrv_df["stand"] == handedness)

    recent = xrv_df.loc[mask].sort_values("game_date", ascending=False).head(n_pitches)

    if len(recent) < 50:
        return {
            "sp_xrv_mean": np.nan,
            "sp_xrv_std": np.nan,
            "sp_n_pitches": len(recent),
            "sp_k_rate": np.nan,
            "sp_bb_rate": np.nan,
            "sp_avg_velo": np.nan,
            "sp_pitch_mix_entropy": np.nan,
        }

    # Mean xRV per pitch (lower = better for pitcher)
    xrv_mean = recent["xrv"].mean()
    xrv_std = recent["xrv"].std()

    # Strikeout and walk rates (from pitch outcomes)
    total_pa = recent["events"].notna().sum()
    if total_pa > 0:
        k_rate = (recent["events"] == "strikeout").sum() / total_pa
        bb_rate = (recent["events"] == "walk").sum() / total_pa
    else:
        k_rate = bb_rate = np.nan

    # Average fastball velocity
    fb_types = ["FF", "SI", "FC"]
    fastballs = recent[recent["pitch_type"].isin(fb_types)]
    avg_velo = fastballs["release_speed"].mean() if len(fastballs) > 0 else np.nan

    # Pitch mix entropy (diversity of arsenal)
    mix = recent["pitch_type"].value_counts(normalize=True)
    entropy = -(mix * np.log2(mix + 1e-10)).sum()

    return {
        "sp_xrv_mean": xrv_mean,
        "sp_xrv_std": xrv_std,
        "sp_n_pitches": len(recent),
        "sp_k_rate": k_rate,
        "sp_bb_rate": bb_rate,
        "sp_avg_velo": avg_velo,
        "sp_pitch_mix_entropy": entropy,
    }


# ──────────────────────────────────────────────────────────────
# 2. Pitcher vs Lineup Matchup (from Bayesian model)
# ──────────────────────────────────────────────────────────────

def _precompute_pitcher_bases(
    model_artifacts: dict,
    pitcher_df: pd.DataFrame,
    pitcher_id: int,
    game_date: str,
    n_pitches: int = 2000,
) -> dict[str, tuple]:
    """
    Precompute pitcher pitch-base vectors split by batter handedness.
    Returns dict: hand -> (pitch_base_array, ptype_idx_array, n_pitches)
    Vectorized — no row-by-row iteration.
    """
    map_est = model_artifacts["map_estimate"]
    ptype_map = model_artifacts["ptype_map"]
    pitcher_map = model_artifacts["pitcher_map"]
    feature_stats = model_artifacts["feature_stats"]

    if pitcher_id not in pitcher_map:
        return {}

    p_idx = pitcher_map[pitcher_id]
    intercept = float(np.array(map_est["intercept"]))
    beta_speed = float(np.array(map_est["beta_speed"]))
    beta_hmov = float(np.array(map_est["beta_hmov"]))
    beta_vmov = float(np.array(map_est["beta_vmov"]))
    beta_locx = float(np.array(map_est["beta_locx"]))
    beta_locz = float(np.array(map_est["beta_locz"]))
    # New betas with backward compat
    beta_spin = float(np.array(map_est.get("beta_spin", 0.0)))
    beta_ext = float(np.array(map_est.get("beta_ext", 0.0)))
    beta_rel_x = float(np.array(map_est.get("beta_rel_x", 0.0)))
    beta_rel_z = float(np.array(map_est.get("beta_rel_z", 0.0)))
    ptype_effects = np.array(map_est["ptype_effect"])
    pitcher_effects = np.array(map_est["pitcher_effect"])

    # Get pitcher's pitches before game date
    before = _get_before(pitcher_df, game_date)

    result = {}
    for hand in ["L", "R"]:
        hand_pitches = before[before["stand"] == hand]
        recent = hand_pitches.iloc[-n_pitches:] if len(hand_pitches) > n_pitches else hand_pitches

        # Apply competitive count filter
        recent = _filter_competitive(recent)

        if len(recent) < 50:
            result[hand] = None
            continue

        # Map pitch types to indices, filter unknowns
        pt_codes = recent["pitch_type"].map(ptype_map)
        valid = pt_codes.notna()
        recent = recent[valid]
        ptype_idx = pt_codes[valid].astype(int).values

        if len(recent) == 0:
            result[hand] = None
            continue

        # Vectorized standardization
        def z_vec(col):
            if col not in feature_stats or col not in recent.columns:
                return np.zeros(len(recent))
            vals = recent[col].values.astype(float)
            s = feature_stats[col]
            if s["std"] > 0:
                out = (vals - s["mean"]) / s["std"]
            else:
                out = np.zeros(len(recent))
            out[np.isnan(out)] = 0.0
            return out

        pitch_base = (
            intercept
            + beta_speed * z_vec("release_speed")
            + beta_hmov * z_vec("pfx_x")
            + beta_vmov * z_vec("pfx_z")
            + beta_locx * z_vec("plate_x")
            + beta_locz * z_vec("plate_z")
            + beta_spin * z_vec("release_spin_rate")
            + beta_ext * z_vec("release_extension")
            + beta_rel_x * z_vec("release_pos_x")
            + beta_rel_z * z_vec("release_pos_z")
            + ptype_effects[ptype_idx]
            + pitcher_effects[p_idx]
        )

        result[hand] = (pitch_base, ptype_idx)

    return result


def compute_matchup_xrv(
    model_artifacts: dict,
    pitcher_bases: dict[str, tuple],
    lineup_hitter_ids: list[int],
    lineup_hitter_hands: list[str],
) -> dict:
    """
    Use the Bayesian hierarchical model to predict how a lineup
    would fare against a pitcher's recent pitch mix.

    Each hitter is matched against the pitcher's pitches to same-handed
    batters (pitcher's arsenal differs vs L/R).

    Args:
        model_artifacts: trained Bayesian model
        pitcher_bases: precomputed from _precompute_pitcher_bases()
        lineup_hitter_ids: list of batter MLB IDs in batting order
        lineup_hitter_hands: list of bat side ('L', 'R') for each hitter
    """
    if not pitcher_bases:
        return {"matchup_xrv_mean": np.nan, "matchup_xrv_sum": np.nan,
                "matchup_n_hitters": 0, "matchup_n_known": 0}

    map_est = model_artifacts["map_estimate"]
    hitter_map = model_artifacts["hitter_map"]
    hitter_effects = np.array(map_est["hitter_effect"])
    hitter_ptype_effects = np.array(map_est["hitter_ptype_effect"])

    hitter_xrvs = []
    n_known = 0

    for hid, hand in zip(lineup_hitter_ids, lineup_hitter_hands):
        bases = pitcher_bases.get(hand)
        if bases is None:
            continue

        pitch_base, ptype_idx = bases

        if hid not in hitter_map:
            # Unknown hitter — use population average (hitter effect = 0)
            hitter_xrvs.append(float(np.mean(pitch_base)))
            continue

        h_idx = hitter_map[hid]
        hitter_preds = (
            pitch_base
            + hitter_effects[h_idx]
            + hitter_ptype_effects[h_idx, ptype_idx]
        )
        hitter_xrvs.append(float(np.mean(hitter_preds)))
        n_known += 1

    return {
        "matchup_xrv_mean": np.mean(hitter_xrvs) if hitter_xrvs else np.nan,
        "matchup_xrv_sum": np.sum(hitter_xrvs) if hitter_xrvs else np.nan,
        "matchup_n_hitters": len(hitter_xrvs),
        "matchup_n_known": n_known,
    }


# ──────────────────────────────────────────────────────────────
# 2b. Bullpen vs Lineup Matchup (from Bayesian model)
# ──────────────────────────────────────────────────────────────

def compute_bullpen_matchup_xrv(
    model_artifacts: dict,
    idx: dict,
    team: str,
    game_date: str,
    lineup_hitter_ids: list[int],
    lineup_hitter_hands: list[str],
    lookback_days: int = 30,
    top_n: int = 7,
    n_pitches: int = 1500,
) -> dict:
    """
    Use the Bayesian model to predict how the bullpen would fare vs a lineup.

    Identifies the team's top N most-used relievers (by pitch count in lookback),
    computes each reliever's matchup xRV vs the opposing lineup, then returns
    a usage-weighted average.
    """
    nan_result = {
        "bp_matchup_xrv_mean": np.nan,
        "bp_matchup_n_relievers": 0,
    }

    if not model_artifacts or not lineup_hitter_ids:
        return nan_result

    # Get team's pitching data
    team_pitches = idx["pitching_team"].get(team)
    if team_pitches is None:
        return nan_result

    before = _get_before(team_pitches, game_date)
    cutoff = (pd.Timestamp(game_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    recent = before[before["game_date"] >= cutoff]

    if len(recent) == 0:
        return nan_result

    # Identify starters vs relievers
    starters = recent.groupby("game_pk").first()["pitcher"].values
    starter_set = set(starters)
    reliever_pitches = recent[~recent["pitcher"].isin(starter_set)]

    if len(reliever_pitches) == 0:
        return nan_result

    # Top N relievers by pitch count (proxy for trust/usage)
    usage = reliever_pitches.groupby("pitcher").size().sort_values(ascending=False)
    top_relievers = usage.head(top_n)
    total_pitches = top_relievers.sum()

    map_est = model_artifacts["map_estimate"]
    hitter_map = model_artifacts["hitter_map"]
    hitter_effects = np.array(map_est["hitter_effect"])
    hitter_ptype_effects = np.array(map_est["hitter_ptype_effect"])

    reliever_xrvs = []
    reliever_weights = []

    for pitcher_id, pitch_count in top_relievers.items():
        pitcher_id = int(pitcher_id)
        if pitcher_id not in idx["pitcher"]:
            continue

        pitcher_df = idx["pitcher"][pitcher_id]
        pitcher_bases = _precompute_pitcher_bases(
            model_artifacts, pitcher_df, pitcher_id, game_date, n_pitches
        )

        if not pitcher_bases:
            continue

        # Compute matchup vs lineup
        matchup = compute_matchup_xrv(
            model_artifacts, pitcher_bases, lineup_hitter_ids, lineup_hitter_hands
        )

        if not np.isnan(matchup["matchup_xrv_mean"]):
            reliever_xrvs.append(matchup["matchup_xrv_mean"])
            reliever_weights.append(pitch_count / total_pitches)

    if not reliever_xrvs:
        return nan_result

    # Usage-weighted average
    weights = np.array(reliever_weights)
    weights = weights / weights.sum()
    bp_matchup_mean = np.average(reliever_xrvs, weights=weights)

    return {
        "bp_matchup_xrv_mean": bp_matchup_mean,
        "bp_matchup_n_relievers": len(reliever_xrvs),
    }


# ──────────────────────────────────────────────────────────────
# 2c. Arsenal-based Matchup (hitter response to pitcher arsenal type)
# ──────────────────────────────────────────────────────────────

def _compute_pitcher_arsenal_live(pitcher_df: pd.DataFrame, game_date: str,
                                   n_pitches: int = 2000) -> dict | None:
    """
    Compute a pitcher's arsenal profile from their recent pitches before game_date.
    Returns dict with raw arsenal feature values, or None if insufficient data.
    """
    before = _get_before(pitcher_df, game_date)
    recent = before.iloc[-n_pitches:] if len(before) > n_pitches else before

    # Apply competitive count filter
    recent = _filter_competitive(recent)

    if len(recent) < 100:
        return None

    n = len(recent)
    pt_counts = recent["pitch_type"].value_counts()
    pt_pcts = pt_counts / n

    hard_pct = sum(pt_pcts.get(t, 0) for t in HARD_TYPES)
    break_pct = sum(pt_pcts.get(t, 0) for t in BREAK_TYPES)
    offspeed_pct = sum(pt_pcts.get(t, 0) for t in OFFSPEED_TYPES)

    pcts = pt_pcts.values
    pcts = pcts[pcts > 0]
    entropy = -float(np.sum(pcts * np.log2(pcts)))

    hard_pitches = recent[recent["pitch_type"].isin(HARD_TYPES)]
    hard_velo = float(hard_pitches["release_speed"].mean()) if len(hard_pitches) > 0 else np.nan

    pt_velos = recent.groupby("pitch_type")["release_speed"].mean()
    velo_spread = float(pt_velos.max() - pt_velos.min()) if len(pt_velos) > 1 else 0.0

    pt_hmov = recent.groupby("pitch_type")["pfx_x"].mean()
    hmov_range = float(pt_hmov.max() - pt_hmov.min()) if len(pt_hmov) > 1 else 0.0

    # New enriched features
    break_pitches = recent[recent["pitch_type"].isin(BREAK_TYPES)]

    hard_spin = float(hard_pitches["release_spin_rate"].mean()) if (
        len(hard_pitches) > 0 and "release_spin_rate" in recent.columns
    ) else np.nan
    break_spin = float(break_pitches["release_spin_rate"].mean()) if (
        len(break_pitches) > 0 and "release_spin_rate" in recent.columns
    ) else np.nan
    hard_ivb = float(hard_pitches["pfx_z"].mean()) if len(hard_pitches) > 0 else np.nan
    hard_ext = float(hard_pitches["release_extension"].mean()) if (
        len(hard_pitches) > 0 and "release_extension" in recent.columns
    ) else np.nan

    # Release point spread (arm slot consistency across pitch types)
    if "release_pos_x" in recent.columns:
        pt_rel_x = recent.groupby("pitch_type")["release_pos_x"].mean().dropna()
        rel_x_spread = float(pt_rel_x.std()) if len(pt_rel_x) > 1 and pd.notna(pt_rel_x.std()) else 0.0
    else:
        rel_x_spread = 0.0

    # Vertical movement range across pitch types
    pt_vmov = recent.groupby("pitch_type")["pfx_z"].mean()
    vmov_range = float(pt_vmov.max() - pt_vmov.min()) if len(pt_vmov) > 1 else 0.0

    return {
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
    }


def _standardize_arsenal(arsenal_raw: dict, feature_stats: dict) -> np.ndarray:
    """Standardize arsenal features using training-time stats."""
    z = np.zeros(len(ARSENAL_FEATURES))
    for i, feat in enumerate(ARSENAL_FEATURES):
        val = arsenal_raw.get(feat, 0.0)
        if np.isnan(val):
            val = 0.0
        stats = feature_stats.get(feat)
        if stats and stats["std"] > 0:
            z[i] = (val - stats["mean"]) / stats["std"]
    return z


def compute_arsenal_matchup_xrv(
    arsenal_artifacts: dict,
    pitcher_bases: dict[str, tuple],
    pitcher_arsenal_z: np.ndarray,
    lineup_hitter_ids: list[int],
    lineup_hitter_hands: list[str],
) -> dict:
    """
    Predict lineup's expected xRV against a pitcher using the arsenal model.

    Instead of sparse matchup effects, uses each hitter's learned sensitivity
    to the pitcher's arsenal profile. Unknown hitters get the population
    average arsenal response.
    """
    if not pitcher_bases:
        return {"arsenal_matchup_xrv_mean": np.nan, "arsenal_matchup_xrv_sum": np.nan,
                "arsenal_matchup_n_hitters": 0, "arsenal_matchup_n_known": 0}

    map_est = arsenal_artifacts["map_estimate"]
    hitter_map = arsenal_artifacts["hitter_map"]
    hitter_effects = np.array(map_est["hitter_effect"])
    hitter_ptype_effects = np.array(map_est["hitter_ptype_effect"])
    hitter_arsenal_betas = np.array(map_est["hitter_arsenal_beta"])
    mu_arsenal = np.array(map_est["mu_arsenal"])

    hitter_xrvs = []
    n_known = 0

    for hid, hand in zip(lineup_hitter_ids, lineup_hitter_hands):
        bases = pitcher_bases.get(hand)
        if bases is None:
            continue

        pitch_base, ptype_idx = bases

        if hid in hitter_map:
            h_idx = hitter_map[hid]
            h_effect = hitter_effects[h_idx]
            h_ptype = hitter_ptype_effects[h_idx, ptype_idx]
            h_arsenal = hitter_arsenal_betas[h_idx]
            n_known += 1
        else:
            # Unknown hitter: population average
            h_effect = 0.0
            h_ptype = np.zeros(len(ptype_idx))
            h_arsenal = mu_arsenal

        # Arsenal interaction: how this hitter responds to this pitcher's arsenal type
        # Handle dimension mismatch between old (7-feature) and new (13-feature) models
        n_model = len(h_arsenal)
        if n_model != len(pitcher_arsenal_z):
            arsenal_z_trimmed = pitcher_arsenal_z[:n_model]
        else:
            arsenal_z_trimmed = pitcher_arsenal_z
        arsenal_bonus = np.dot(h_arsenal, arsenal_z_trimmed)

        hitter_preds = pitch_base + h_effect + h_ptype + arsenal_bonus
        hitter_xrvs.append(float(np.mean(hitter_preds)))

    return {
        "arsenal_matchup_xrv_mean": np.mean(hitter_xrvs) if hitter_xrvs else np.nan,
        "arsenal_matchup_xrv_sum": np.sum(hitter_xrvs) if hitter_xrvs else np.nan,
        "arsenal_matchup_n_hitters": len(hitter_xrvs),
        "arsenal_matchup_n_known": n_known,
    }


def compute_bullpen_arsenal_matchup_xrv(
    arsenal_artifacts: dict,
    idx: dict,
    team: str,
    game_date: str,
    lineup_hitter_ids: list[int],
    lineup_hitter_hands: list[str],
    lookback_days: int = 30,
    top_n: int = 7,
    n_pitches: int = 1500,
) -> dict:
    """Bullpen version of arsenal matchup — top N relievers vs opposing lineup."""
    nan_result = {"bp_arsenal_matchup_xrv_mean": np.nan, "bp_arsenal_matchup_n_relievers": 0}

    if not arsenal_artifacts or not lineup_hitter_ids:
        return nan_result

    team_pitches = idx["pitching_team"].get(team)
    if team_pitches is None:
        return nan_result

    before = _get_before(team_pitches, game_date)
    cutoff = (pd.Timestamp(game_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    recent = before[before["game_date"] >= cutoff]

    if len(recent) == 0:
        return nan_result

    starters = recent.groupby("game_pk").first()["pitcher"].values
    starter_set = set(starters)
    reliever_pitches = recent[~recent["pitcher"].isin(starter_set)]

    if len(reliever_pitches) == 0:
        return nan_result

    usage = reliever_pitches.groupby("pitcher").size().sort_values(ascending=False)
    top_relievers = usage.head(top_n)
    total_pitches = top_relievers.sum()

    reliever_xrvs = []
    reliever_weights = []

    for pitcher_id, pitch_count in top_relievers.items():
        pitcher_id = int(pitcher_id)
        if pitcher_id not in idx["pitcher"]:
            continue

        pitcher_df = idx["pitcher"][pitcher_id]

        # Compute pitcher bases using arsenal model's parameters
        pitcher_bases = _precompute_pitcher_bases(
            arsenal_artifacts, pitcher_df, pitcher_id, game_date, n_pitches
        )
        if not pitcher_bases:
            continue

        # Compute arsenal profile for this reliever
        arsenal_raw = _compute_pitcher_arsenal_live(pitcher_df, game_date, n_pitches)
        if arsenal_raw is None:
            continue
        arsenal_z = _standardize_arsenal(arsenal_raw, arsenal_artifacts["feature_stats"])

        matchup = compute_arsenal_matchup_xrv(
            arsenal_artifacts, pitcher_bases, arsenal_z,
            lineup_hitter_ids, lineup_hitter_hands
        )

        if not np.isnan(matchup["arsenal_matchup_xrv_mean"]):
            reliever_xrvs.append(matchup["arsenal_matchup_xrv_mean"])
            reliever_weights.append(pitch_count / total_pitches)

    if not reliever_xrvs:
        return nan_result

    weights = np.array(reliever_weights)
    weights = weights / weights.sum()
    bp_matchup_mean = float(np.average(reliever_xrvs, weights=weights))

    return {
        "bp_arsenal_matchup_xrv_mean": bp_matchup_mean,
        "bp_arsenal_matchup_n_relievers": len(reliever_xrvs),
    }


# ──────────────────────────────────────────────────────────────
# 3. Bullpen Quality + Availability
# ──────────────────────────────────────────────────────────────

def compute_bullpen_features(
    xrv_df: pd.DataFrame,
    game_date: str,
    team: str,
    lookback_days: int = 30,
    rest_lookback_days: int = 3,
) -> dict:
    """
    Compute bullpen quality and availability for a team.

    Quality: rolling xRV of all non-starter pitchers on the team.
    Availability: innings pitched in last 1-3 days (fatigue proxy).
    """
    cutoff = pd.Timestamp(game_date) - timedelta(days=lookback_days)
    rest_cutoff = pd.Timestamp(game_date) - timedelta(days=rest_lookback_days)

    # Identify team's pitches (when they're at home or away)
    team_mask = (
        ((xrv_df["home_team"] == team) & (xrv_df["inning_topbot"] == "Top"))
        | ((xrv_df["away_team"] == team) & (xrv_df["inning_topbot"] == "Bot"))
    )
    team_pitches = xrv_df.loc[team_mask & (xrv_df["game_date"] < game_date)].copy()

    if len(team_pitches) == 0:
        return {
            "bp_xrv_mean": np.nan,
            "bp_xrv_std": np.nan,
            "bp_recent_ip": np.nan,
            "bp_fatigue_score": np.nan,
        }

    # Identify relievers: pitchers who threw in relief (not first inning, or entered mid-game)
    # Simplified: pitchers who have appeared in innings > 1 for this team recently
    recent = team_pitches[team_pitches["game_date"] >= cutoff.strftime("%Y-%m-%d")]

    # Find per-game first pitcher (starter)
    starters = recent.groupby("game_pk").apply(
        lambda g: g.sort_values(["inning", "at_bat_number"]).iloc[0]["pitcher"],
        include_groups=False,
    )
    starter_set = set(starters.values)

    relievers = recent[~recent["pitcher"].isin(starter_set)]

    bp_xrv = relievers["xrv"].mean() if len(relievers) > 0 else np.nan
    bp_xrv_std = relievers["xrv"].std() if len(relievers) > 0 else np.nan

    # Recent usage (fatigue): pitches thrown in last N days
    very_recent = team_pitches[team_pitches["game_date"] >= rest_cutoff.strftime("%Y-%m-%d")]
    bp_recent = very_recent[~very_recent["pitcher"].isin(starter_set)]
    bp_recent_pitches = len(bp_recent)
    # Approximate IP from pitch count (~15 pitches per IP)
    bp_recent_ip = bp_recent_pitches / 15.0

    # Fatigue score: higher = more tired bullpen
    # Weighted by recency
    fatigue = 0.0
    for days_ago in range(1, rest_lookback_days + 1):
        day = (pd.Timestamp(game_date) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        day_pitches = bp_recent[bp_recent["game_date"] == day]
        weight = 1.0 / days_ago  # more recent = more weight
        fatigue += len(day_pitches) * weight / 15.0

    return {
        "bp_xrv_mean": bp_xrv,
        "bp_xrv_std": bp_xrv_std,
        "bp_recent_ip": bp_recent_ip,
        "bp_fatigue_score": fatigue,
    }


# ──────────────────────────────────────────────────────────────
# 4. Team Hitting Quality
# ──────────────────────────────────────────────────────────────

def compute_hitting_features(
    xrv_df: pd.DataFrame,
    game_date: str,
    team: str,
    lookback_days: int = 30,
) -> dict:
    """
    Rolling team hitting quality measured by xRV.
    """
    cutoff = pd.Timestamp(game_date) - timedelta(days=lookback_days)

    # Team's batting pitches
    team_mask = (
        ((xrv_df["home_team"] == team) & (xrv_df["inning_topbot"] == "Bot"))
        | ((xrv_df["away_team"] == team) & (xrv_df["inning_topbot"] == "Top"))
    )
    team_batting = xrv_df.loc[
        team_mask
        & (xrv_df["game_date"] < game_date)
        & (xrv_df["game_date"] >= cutoff.strftime("%Y-%m-%d"))
    ]

    if len(team_batting) == 0:
        return {"hit_xrv_mean": np.nan, "hit_xrv_contact": np.nan, "hit_k_rate": np.nan}

    hit_xrv = team_batting["xrv"].mean()

    # Contact quality only
    contact = team_batting[team_batting["type"] == "X"]
    hit_xrv_contact = contact["xrv"].mean() if len(contact) > 0 else np.nan

    # Team K rate
    pa = team_batting["events"].notna().sum()
    k_rate = (team_batting["events"] == "strikeout").sum() / pa if pa > 0 else np.nan

    return {
        "hit_xrv_mean": hit_xrv,
        "hit_xrv_contact": hit_xrv_contact,
        "hit_k_rate": k_rate,
    }


# ──────────────────────────────────────────────────────────────
# 5. Defense (simplified — OAA not in Statcast pitch data)
# ──────────────────────────────────────────────────────────────

def compute_defense_features(
    xrv_df: pd.DataFrame,
    game_date: str,
    team: str,
    lookback_days: int = 60,
) -> dict:
    """
    Proxy for defensive quality: difference between actual run value
    and expected run value on balls in play (xRV - actual RV on contact).
    Positive = defense is costing runs (bad). Negative = defense saving runs.
    """
    cutoff = pd.Timestamp(game_date) - timedelta(days=lookback_days)

    # Team's fielding: when they're on defense
    team_mask = (
        ((xrv_df["home_team"] == team) & (xrv_df["inning_topbot"] == "Top"))
        | ((xrv_df["away_team"] == team) & (xrv_df["inning_topbot"] == "Bot"))
    )
    defense = xrv_df.loc[
        team_mask
        & (xrv_df["game_date"] < game_date)
        & (xrv_df["game_date"] >= cutoff.strftime("%Y-%m-%d"))
        & (xrv_df["type"] == "X")
    ]

    if len(defense) < 100:
        return {"def_xrv_delta": np.nan}

    # xRV - actual delta_run_exp: how much better/worse than expected
    # Negative = defense saves runs (actual worse for hitters than expected)
    xrv_delta = (defense["delta_run_exp"] - defense["xrv"]).mean()

    return {"def_xrv_delta": xrv_delta}


# ──────────────────────────────────────────────────────────────
# 6. Park Factor (static per home_team / season)
# ──────────────────────────────────────────────────────────────

def compute_park_factors(xrv_df: pd.DataFrame, season: int,
                         shrinkage_n: int = 20000) -> dict[str, float]:
    """
    Compute park factor as ratio of xRV at this park vs league average.
    Uses Bayesian shrinkage: parks with fewer pitches are pulled toward 1.0.
    shrinkage_n = number of pitches needed for full weight on park estimate.
    """
    park_stats = xrv_df.groupby("home_team")["xrv"].agg(["mean", "count"])
    league_avg = xrv_df["xrv"].mean()

    factors = {}
    for team, row in park_stats.iterrows():
        n = row["count"]
        weight = n / (n + shrinkage_n)
        shrunk_mean = weight * row["mean"] + (1 - weight) * league_avg
        factors[team] = shrunk_mean / league_avg if league_avg != 0 else 1.0

    return factors


# ──────────────────────────────────────────────────────────────
# Main: build features for all games in a season
# ──────────────────────────────────────────────────────────────

def _recent_winpct(history: list, n: int = 10) -> float:
    """Win% over last n games. Returns NaN if fewer than 3 games."""
    if len(history) < 3:
        return np.nan
    recent = history[-n:]
    return sum(recent) / len(recent)


def _preindex_xrv(xrv: pd.DataFrame) -> dict:
    """
    Pre-index xRV data for fast lookup.
    Returns dicts mapping pitcher/team to their sorted pitch DataFrames.
    """
    xrv = xrv.sort_values("game_date").reset_index(drop=True)

    # Determine which team is pitching / batting for each row
    # Pitching team: home when inning_topbot == "Top", away when "Bot"
    # Batting team: home when inning_topbot == "Bot", away when "Top"
    xrv["pitching_team"] = np.where(
        xrv["inning_topbot"] == "Top", xrv["home_team"], xrv["away_team"]
    )
    xrv["batting_team"] = np.where(
        xrv["inning_topbot"] == "Bot", xrv["home_team"], xrv["away_team"]
    )

    idx = {
        "pitcher": {pid: grp for pid, grp in xrv.groupby("pitcher")},
        "pitching_team": {t: grp for t, grp in xrv.groupby("pitching_team")},
        "batting_team": {t: grp for t, grp in xrv.groupby("batting_team")},
        "xrv": xrv,
    }
    return idx


def _get_before(df: pd.DataFrame, game_date) -> pd.DataFrame:
    """Get rows before game_date using the sorted game_date column."""
    # Binary search for cutoff — ensure type matches the column
    dates = df["game_date"].values
    if dates.dtype.kind == "M" and not isinstance(game_date, np.datetime64):
        game_date = np.datetime64(game_date)
    idx = np.searchsorted(dates, game_date, side="left")
    return df.iloc[:idx]


def _sp_features_fast(pitcher_df: pd.DataFrame, game_date: str, n_pitches: int) -> dict:
    """Compute SP features from pre-indexed pitcher data."""
    before = _get_before(pitcher_df, game_date)
    recent = before.iloc[-n_pitches:] if len(before) > n_pitches else before

    nan_result = {
        "sp_xrv_mean": np.nan, "sp_xrv_std": np.nan, "sp_n_pitches": len(recent),
        "sp_k_rate": np.nan, "sp_bb_rate": np.nan, "sp_avg_velo": np.nan,
        "sp_pitch_mix_entropy": np.nan,
        "sp_xrv_vs_L": np.nan, "sp_xrv_vs_R": np.nan,
        "sp_rest_days": np.nan,
    }

    if len(recent) < 50:
        return nan_result

    xrv_mean = recent["xrv"].mean()
    xrv_std = recent["xrv"].std()
    total_pa = recent["events"].notna().sum()
    k_rate = (recent["events"] == "strikeout").sum() / total_pa if total_pa > 0 else np.nan
    bb_rate = (recent["events"] == "walk").sum() / total_pa if total_pa > 0 else np.nan

    fb_mask = recent["pitch_type"].isin(["FF", "SI", "FC"])
    avg_velo = recent.loc[fb_mask, "release_speed"].mean() if fb_mask.any() else np.nan

    mix = recent["pitch_type"].value_counts(normalize=True)
    entropy = -(mix * np.log2(mix + 1e-10)).sum()

    # xRV split by batter handedness — key for platoon advantage
    vs_L = recent[recent["stand"] == "L"]
    vs_R = recent[recent["stand"] == "R"]
    xrv_vs_L = vs_L["xrv"].mean() if len(vs_L) >= 20 else xrv_mean
    xrv_vs_R = vs_R["xrv"].mean() if len(vs_R) >= 20 else xrv_mean

    # Days since last appearance (rest), capped at 14
    game_dates = recent["game_date"].unique()
    if len(game_dates) >= 1:
        last_date = game_dates[-1]  # sorted ascending
        rest_days = (pd.Timestamp(game_date) - pd.Timestamp(last_date)).days
        rest_days = min(rest_days, 14)  # cap: anything over 14 = fully rested / season start
    else:
        rest_days = np.nan

    # --- Change 4: Home/away SP split ---
    # Home pitcher pitches when inning_topbot == "Top", away when "Bot"
    if "inning_topbot" in recent.columns:
        home_pitches = recent[recent["inning_topbot"] == "Top"]
        away_pitches = recent[recent["inning_topbot"] == "Bot"]
        sp_home_xrv = home_pitches["xrv"].mean() if len(home_pitches) >= 30 else xrv_mean
        sp_away_xrv = away_pitches["xrv"].mean() if len(away_pitches) >= 30 else xrv_mean
    else:
        sp_home_xrv = xrv_mean
        sp_away_xrv = xrv_mean

    # Overperformance residual: actual outcome vs expected (xRV)
    # Negative = pitcher consistently beats their stuff (deception, sequencing, tunneling)
    if "delta_run_exp" in recent.columns:
        residuals = recent["delta_run_exp"] - recent["xrv"]
        resid_valid = residuals.dropna()
        if len(resid_valid) >= 100:
            sp_overperf = resid_valid.mean()
            recent_500 = before.iloc[-500:] if len(before) > 500 else before
            if "delta_run_exp" in recent_500.columns:
                resid_recent = (recent_500["delta_run_exp"] - recent_500["xrv"]).dropna()
                sp_overperf_recent = resid_recent.mean() if len(resid_recent) >= 50 else sp_overperf
            else:
                sp_overperf_recent = sp_overperf
        else:
            sp_overperf = np.nan
            sp_overperf_recent = np.nan
    else:
        sp_overperf = np.nan
        sp_overperf_recent = np.nan

    # --- Change 7: Transition entropy (pitch sequencing unpredictability) ---
    if len(recent) >= 100 and "pitch_type" in recent.columns:
        pt_seq = recent["pitch_type"].values
        transitions = {}
        for a, b in zip(pt_seq[:-1], pt_seq[1:]):
            transitions.setdefault(a, []).append(b)
        entropies = []
        for from_pt, to_pts in transitions.items():
            counts = pd.Series(to_pts).value_counts(normalize=True).values
            counts = counts[counts > 0]
            entropies.append(-float(np.sum(counts * np.log2(counts))))
        sp_transition_entropy = float(np.mean(entropies)) if entropies else 0.0
    else:
        sp_transition_entropy = np.nan

    # --- Change 2: SP stuff trend (short vs long window) ---
    recent_short = before.iloc[-300:] if len(before) > 300 else before
    if len(recent_short) >= 100:
        short_xrv = recent_short["xrv"].mean()
        short_fb = recent_short[recent_short["pitch_type"].isin(["FF", "SI", "FC"])]
        short_velo = short_fb["release_speed"].mean() if len(short_fb) > 0 else np.nan
        long_spin = recent.loc[fb_mask, "release_spin_rate"].mean() if (
            fb_mask.any() and "release_spin_rate" in recent.columns
        ) else np.nan
        short_spin = short_fb["release_spin_rate"].mean() if (
            len(short_fb) > 0 and "release_spin_rate" in recent_short.columns
        ) else np.nan

        sp_velo_trend = short_velo - avg_velo if not (np.isnan(short_velo) or np.isnan(avg_velo)) else np.nan
        sp_spin_trend = short_spin - long_spin if not (np.isnan(short_spin) or np.isnan(long_spin)) else np.nan
        sp_xrv_trend = short_xrv - xrv_mean  # positive = getting worse recently
    else:
        sp_velo_trend = np.nan
        sp_spin_trend = np.nan
        sp_xrv_trend = np.nan

    return {
        "sp_xrv_mean": xrv_mean, "sp_xrv_std": xrv_std, "sp_n_pitches": len(recent),
        "sp_k_rate": k_rate, "sp_bb_rate": bb_rate, "sp_avg_velo": avg_velo,
        "sp_pitch_mix_entropy": entropy,
        "sp_xrv_vs_L": xrv_vs_L, "sp_xrv_vs_R": xrv_vs_R,
        "sp_rest_days": rest_days,
        "sp_overperf": sp_overperf,
        "sp_overperf_recent": sp_overperf_recent,
        "sp_velo_trend": sp_velo_trend,     # positive = velo increasing recently
        "sp_spin_trend": sp_spin_trend,     # positive = spin increasing recently
        "sp_xrv_trend": sp_xrv_trend,       # positive = getting worse recently
        "sp_home_xrv": sp_home_xrv,         # xRV when pitching at home
        "sp_away_xrv": sp_away_xrv,         # xRV when pitching on the road
        "sp_transition_entropy": sp_transition_entropy,  # pitch sequencing unpredictability
    }


def _bp_features_fast(team_pitches: pd.DataFrame, game_date: str,
                       lookback_days: int = 30, rest_days: int = 3,
                       fallback_games: int = 15) -> dict:
    """Compute bullpen features from pre-indexed team pitching data."""
    nan_result = {"bp_xrv_mean": np.nan, "bp_xrv_std": np.nan,
                  "bp_recent_ip": np.nan, "bp_fatigue_score": np.nan}
    before = _get_before(team_pitches, game_date)
    if len(before) == 0:
        return nan_result

    cutoff = (pd.Timestamp(game_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    rest_cutoff = (pd.Timestamp(game_date) - timedelta(days=rest_days)).strftime("%Y-%m-%d")
    recent = before[before["game_date"] >= cutoff]

    # Fallback: if calendar window is too thin, use last N games
    if len(recent) < 50:
        game_pks = before["game_pk"].unique()
        if len(game_pks) > fallback_games:
            recent_games = game_pks[-fallback_games:]
            recent = before[before["game_pk"].isin(set(recent_games))]
        else:
            recent = before

    if len(recent) == 0:
        return nan_result

    # Find starters (first pitcher in each game)
    starters = recent.groupby("game_pk").first()["pitcher"].values
    starter_set = set(starters)
    relievers = recent[~recent["pitcher"].isin(starter_set)]

    bp_xrv = relievers["xrv"].mean() if len(relievers) > 0 else np.nan
    bp_xrv_std = relievers["xrv"].std() if len(relievers) > 0 else np.nan

    # Fatigue from last N days
    very_recent = before[before["game_date"] >= rest_cutoff]
    bp_recent = very_recent[~very_recent["pitcher"].isin(starter_set)]
    bp_recent_ip = len(bp_recent) / 15.0

    fatigue = 0.0
    for days_ago in range(1, rest_days + 1):
        day = (pd.Timestamp(game_date) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        day_count = (bp_recent["game_date"] == day).sum()
        fatigue += day_count / days_ago / 15.0

    return {"bp_xrv_mean": bp_xrv, "bp_xrv_std": bp_xrv_std,
            "bp_recent_ip": bp_recent_ip, "bp_fatigue_score": fatigue}


def _hit_features_fast(team_batting: pd.DataFrame, game_date: str,
                        lookback_days: int = 30, fallback_games: int = 15) -> dict:
    """Compute team hitting features from pre-indexed data."""
    before = _get_before(team_batting, game_date)
    cutoff = (pd.Timestamp(game_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    recent = before[before["game_date"] >= cutoff]

    # Fallback: if calendar window is too thin, use last N games
    if len(recent) < 50:
        game_pks = before["game_pk"].unique()
        if len(game_pks) > fallback_games:
            recent_games = game_pks[-fallback_games:]
            recent = before[before["game_pk"].isin(set(recent_games))]
        else:
            recent = before

    if len(recent) == 0:
        return {"hit_xrv_mean": np.nan, "hit_xrv_contact": np.nan, "hit_k_rate": np.nan}

    hit_xrv = recent["xrv"].mean()
    contact = recent[recent["type"] == "X"]
    hit_xrv_contact = contact["xrv"].mean() if len(contact) > 0 else np.nan
    pa = recent["events"].notna().sum()
    k_rate = (recent["events"] == "strikeout").sum() / pa if pa > 0 else np.nan

    return {"hit_xrv_mean": hit_xrv, "hit_xrv_contact": hit_xrv_contact, "hit_k_rate": k_rate}


def _def_features_fast(team_pitches: pd.DataFrame, game_date: str,
                        lookback_days: int = 60, fallback_games: int = 25) -> dict:
    """Compute defense features from pre-indexed team pitching data."""
    before = _get_before(team_pitches, game_date)
    cutoff = (pd.Timestamp(game_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    recent = before[(before["game_date"] >= cutoff) & (before["type"] == "X")]

    # Fallback: if calendar window has too few BIP, use last N games
    if len(recent) < 100:
        contact = before[before["type"] == "X"]
        game_pks = contact["game_pk"].unique()
        if len(game_pks) > fallback_games:
            recent_games = game_pks[-fallback_games:]
            recent = contact[contact["game_pk"].isin(set(recent_games))]
        else:
            recent = contact

    if len(recent) < 50:
        return {"def_xrv_delta": np.nan}

    return {"def_xrv_delta": (recent["delta_run_exp"] - recent["xrv"]).mean()}


def _extract_lineups(xrv_season: pd.DataFrame) -> dict:
    """
    Extract starting lineups from pitch-level xRV data.
    For each game_pk, returns:
      {game_pk: {"home": [(batter_id, stand), ...], "away": [(batter_id, stand), ...]}}
    The first 9 unique batters per side approximate the starting lineup.
    """
    lineups = {}

    # Group by game for efficiency
    for game_pk, game_df in xrv_season.groupby("game_pk"):
        home_batting = game_df[game_df["inning_topbot"] == "Bot"]
        away_batting = game_df[game_df["inning_topbot"] == "Top"]

        home_lu = []
        seen = set()
        for _, r in home_batting.iterrows():
            bid = r["batter"]
            if bid not in seen and len(home_lu) < 9:
                home_lu.append((int(bid), r["stand"]))
                seen.add(bid)

        away_lu = []
        seen = set()
        for _, r in away_batting.iterrows():
            bid = r["batter"]
            if bid not in seen and len(away_lu) < 9:
                away_lu.append((int(bid), r["stand"]))
                seen.add(bid)

        lineups[game_pk] = {"home": home_lu, "away": away_lu}

    return lineups


def _compute_hand_matchup(
    hand_models: dict,
    pitcher_df: pd.DataFrame,
    pitcher_id: int,
    game_date: str,
    n_pitches: int,
    hitter_ids: list[int],
    hitter_hands: list[str],
    matchup_fn,
) -> dict:
    """
    Compute matchup using hand-specific models.
    Each hitter is scored against the model trained on their batting side.
    """
    nan_result = {"matchup_xrv_mean": np.nan, "matchup_xrv_sum": np.nan,
                  "matchup_n_hitters": 0, "matchup_n_known": 0}

    all_xrvs = []
    total_known = 0

    # Group hitters by hand
    for hand in ["L", "R"]:
        model = hand_models.get(hand)
        if model is None:
            continue
        hand_hids = [hid for hid, h in zip(hitter_ids, hitter_hands) if h == hand]
        hand_hands = [h for h in hitter_hands if h == hand]
        if not hand_hids:
            continue

        pitcher_bases = _precompute_pitcher_bases(
            model, pitcher_df, pitcher_id, game_date, n_pitches)
        result = matchup_fn(model, pitcher_bases, hand_hids, hand_hands)

        if not np.isnan(result.get("matchup_xrv_mean", np.nan)):
            # Weight by number of hitters
            n_h = result["matchup_n_hitters"]
            if n_h > 0:
                all_xrvs.extend([result["matchup_xrv_mean"]] * n_h)
                total_known += result["matchup_n_known"]

    if not all_xrvs:
        return nan_result

    return {
        "matchup_xrv_mean": np.mean(all_xrvs),
        "matchup_xrv_sum": np.sum(all_xrvs),
        "matchup_n_hitters": len(all_xrvs),
        "matchup_n_known": total_known,
    }


def _compute_hand_arsenal_matchup(
    hand_models: dict,
    pitcher_df: pd.DataFrame,
    pitcher_id: int,
    game_date: str,
    n_pitches: int,
    hitter_ids: list[int],
    hitter_hands: list[str],
) -> dict:
    """
    Compute arsenal matchup using hand-specific models.
    Each hitter is scored against the model trained on their batting side.
    """
    nan_result = {"arsenal_matchup_xrv_mean": np.nan, "arsenal_matchup_xrv_sum": np.nan,
                  "arsenal_matchup_n_hitters": 0, "arsenal_matchup_n_known": 0}

    all_xrvs = []
    total_known = 0

    arsenal_raw = _compute_pitcher_arsenal_live(pitcher_df, game_date, n_pitches)
    if arsenal_raw is None:
        return nan_result

    for hand in ["L", "R"]:
        model = hand_models.get(hand)
        if model is None:
            continue
        hand_hids = [hid for hid, h in zip(hitter_ids, hitter_hands) if h == hand]
        hand_hands = [h for h in hitter_hands if h == hand]
        if not hand_hids:
            continue

        pitcher_bases = _precompute_pitcher_bases(
            model, pitcher_df, pitcher_id, game_date, n_pitches)
        arsenal_z = _standardize_arsenal(arsenal_raw, model["feature_stats"])

        result = compute_arsenal_matchup_xrv(
            model, pitcher_bases, arsenal_z, hand_hids, hand_hands)

        if not np.isnan(result.get("arsenal_matchup_xrv_mean", np.nan)):
            n_h = result["arsenal_matchup_n_hitters"]
            if n_h > 0:
                all_xrvs.extend([result["arsenal_matchup_xrv_mean"]] * n_h)
                total_known += result["arsenal_matchup_n_known"]

    if not all_xrvs:
        return nan_result

    return {
        "arsenal_matchup_xrv_mean": np.mean(all_xrvs),
        "arsenal_matchup_xrv_sum": np.sum(all_xrvs),
        "arsenal_matchup_n_hitters": len(all_xrvs),
        "arsenal_matchup_n_known": total_known,
    }


def _load_team_oaa(season: int) -> dict[str, float]:
    """
    Load per-player OAA from scrape_oaa.py output and aggregate to team-level OAA rate.
    Uses fielder_3..fielder_9 from statcast to map players to teams.
    Returns {team_abbr: total_oaa_per_player} for the given season.
    """
    oaa_path = OAA_DIR / "oaa_all.parquet"
    if not oaa_path.exists():
        return {}
    oaa = pd.read_parquet(oaa_path)
    oaa = oaa[oaa["season"] == season]
    if len(oaa) == 0:
        return {}

    # Aggregate per-player OAA across positions
    player_oaa = oaa.groupby("player_id")["oaa"].sum()

    # Map players to teams via statcast fielder columns
    statcast_path = DATA_DIR / "statcast" / f"statcast_{season}.parquet"
    if not statcast_path.exists():
        # Try subdirectory format
        stat_dir = DATA_DIR / "statcast" / str(season)
        if stat_dir.exists():
            frames = []
            for f in sorted(stat_dir.glob("*.parquet")):
                cols = ["home_team", "away_team", "inning_topbot"] + [f"fielder_{i}" for i in range(3, 10)]
                try:
                    frames.append(pd.read_parquet(f, columns=cols))
                except Exception:
                    pass
            if frames:
                stat_df = pd.concat(frames, ignore_index=True)
            else:
                return {}
        else:
            return {}
    else:
        cols = ["home_team", "away_team", "inning_topbot"] + [f"fielder_{i}" for i in range(3, 10)]
        stat_df = pd.read_parquet(statcast_path, columns=cols)

    # Determine fielding team: home when top of inning, away when bottom
    stat_df["fielding_team"] = np.where(
        stat_df["inning_topbot"] == "Top", stat_df["home_team"], stat_df["away_team"])

    # For each fielder position, find most common team assignment
    player_team = {}
    for pos in range(3, 10):
        col = f"fielder_{pos}"
        if col in stat_df.columns:
            pt = stat_df.groupby(col)["fielding_team"].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
            for pid, team in pt.items():
                if pd.notna(pid) and team:
                    player_team[int(pid)] = team

    # Aggregate OAA by team
    team_oaa = {}
    team_counts = {}
    for pid, oaa_val in player_oaa.items():
        team = player_team.get(int(pid))
        if team:
            team_oaa[team] = team_oaa.get(team, 0) + oaa_val
            team_counts[team] = team_counts.get(team, 0) + 1

    # Return average OAA per player on roster
    result = {}
    for team in team_oaa:
        if team_counts[team] > 0:
            result[team] = team_oaa[team] / team_counts[team]
    return result


def _load_trade_deadline_acquisitions(year: int, idx: dict) -> dict[str, dict]:
    """
    Load trade deadline transactions and compute per-team talent acquisition signal.
    Uses xRV data to assess quality of acquired players.
    Returns {team_abbr: {"net_acquisitions": int, "acquired_pitcher_xrv": float, "acquired_hitter_xrv": float}}
    """
    tx_path = ROSTER_DIR / f"transactions_{year}.parquet"
    if not tx_path.exists():
        return {}

    tx = pd.read_parquet(tx_path)
    # Focus on trade deadline window: June 15 - Aug 1
    tx = tx[(tx["date"] >= f"{year}-06-15") & (tx["date"] <= f"{year}-08-01")]
    if len(tx) == 0:
        return {}

    team_stats = {}
    for _, row in tx.iterrows():
        to_team = row["to_team"]
        from_team = row["from_team"]
        pid = int(row["player_id"])

        if not to_team:
            continue

        # Initialize team entries
        for t in [to_team, from_team]:
            if t and t not in team_stats:
                team_stats[t] = {"net_acquisitions": 0, "acquired_pitcher_xrv": 0.0, "acquired_hitter_xrv": 0.0}

        # Count acquisitions/losses
        team_stats[to_team]["net_acquisitions"] += 1
        if from_team:
            team_stats[from_team]["net_acquisitions"] -= 1

        # Assess player quality from prior xRV data
        if pid in idx.get("pitcher", {}):
            pitcher_df = idx["pitcher"][pid]
            if len(pitcher_df) >= 100:
                xrv = pitcher_df["xrv"].mean()
                team_stats[to_team]["acquired_pitcher_xrv"] += xrv

    return team_stats


def _compute_team_priors(prior_year: int) -> dict[str, float]:
    """
    Compute preseason team strength prior from prior season win%.
    Returns {team_abbr: win_pct} centered at 0.5.
    """
    games_path = GAMES_DIR / f"games_{prior_year}.parquet"
    if not games_path.exists():
        return {}
    games = pd.read_parquet(games_path)
    games = games[games["game_type"] == "R"]

    team_wins = {}
    team_games = {}
    for _, g in games.iterrows():
        ht = g["home_team_abbr"]
        at = g["away_team_abbr"]
        hw = g["home_win"]
        team_wins[ht] = team_wins.get(ht, 0) + hw
        team_games[ht] = team_games.get(ht, 0) + 1
        team_wins[at] = team_wins.get(at, 0) + (1 - hw)
        team_games[at] = team_games.get(at, 0) + 1

    priors = {}
    for team in team_wins:
        if team_games[team] > 0:
            priors[team] = team_wins[team] / team_games[team]
    return priors


PROJECTIONS_DIR = DATA_DIR / "projections"

# Hardcoded Opening Day dates to avoid relying on games parquet min date
OPENING_DAY = {
    2017: "2017-04-02", 2018: "2018-03-29", 2019: "2019-03-20",
    2020: "2020-07-23", 2021: "2021-04-01", 2022: "2022-04-07",
    2023: "2023-03-30", 2024: "2024-03-20", 2025: "2025-03-18",
    2026: "2026-03-26",
}

_SUFFIX_RE = re.compile(r'\s+(jr\.?|sr\.?|ii|iii|iv)\s*$', re.IGNORECASE)


def _normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching.

    Lowercases, strips accents/diacritics, removes suffixes (Jr., Sr., II, etc.),
    and collapses whitespace. E.g. "José Ramírez Jr." -> "jose ramirez".
    """
    if not name:
        return ""
    # Lowercase
    name = name.strip().lower()
    # Strip accents: NFD decompose, then remove combining characters
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Remove suffixes
    name = _SUFFIX_RE.sub("", name)
    # Remove remaining punctuation except spaces/hyphens
    name = re.sub(r"[^\w\s-]", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _load_team_projections(year: int) -> dict[str, float]:
    """Load preseason projected win% for each team.

    Returns {team_abbr: projected_wpct}. Returns empty dict if unavailable.
    """
    path = PROJECTIONS_DIR / f"team_projections_{year}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["team_abbr"], df["projected_wpct"]))


def _load_pitcher_projections(year: int) -> dict:
    """Load preseason pitcher projections keyed by MLBAM ID and normalized name.

    Returns a dict where keys are either:
      - int (MLBAM ID from pybaseball crosswalk)
      - str (normalized pitcher name via _normalize_name)
      - str prefixed with "last:" for last-name-only fallback (only if unambiguous)
    """
    path = PROJECTIONS_DIR / f"pitcher_projections_{year}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)

    # Build FanGraphs ID -> MLBAM ID crosswalk
    fg_to_mlbam: dict[int, int] = {}
    try:
        from pybaseball import playerid_reverse_lookup
        fg_ids_int = [int(x) for x in df["fg_id"].dropna().unique() if str(x).isdigit()]
        if fg_ids_int:
            crosswalk = playerid_reverse_lookup(fg_ids_int, key_type="fangraphs")
            fg_to_mlbam = {
                int(row["key_fangraphs"]): int(row["key_mlbam"])
                for _, row in crosswalk.iterrows()
                if pd.notna(row.get("key_fangraphs")) and pd.notna(row.get("key_mlbam"))
            }
            print(f"    Crosswalk: mapped {len(fg_to_mlbam)}/{len(fg_ids_int)} FG IDs to MLBAM IDs")
    except Exception as e:
        print(f"    Crosswalk unavailable ({e}); using name matching only")

    result = {}
    last_name_counts: dict[str, list[str]] = {}  # last_name -> [normalized full names]

    for _, row in df.iterrows():
        raw_name = str(row.get("name", "")).strip()
        if not raw_name:
            continue

        proj = {
            "projected_era": row.get("projected_era"),
            "projected_war": row.get("projected_war"),
        }

        # Key by normalized name
        norm = _normalize_name(raw_name)
        if norm:
            result[norm] = proj

        # Key by MLBAM ID if crosswalk is available
        fg_id = row.get("fg_id")
        if pd.notna(fg_id) and str(fg_id).isdigit():
            mlbam_id = fg_to_mlbam.get(int(fg_id))
            if mlbam_id:
                result[mlbam_id] = proj

        # Track last names for fallback
        parts = norm.split() if norm else []
        if parts:
            last = parts[-1]
            last_name_counts.setdefault(last, []).append(norm)

    # Add unambiguous last-name fallback entries
    for last, full_names in last_name_counts.items():
        if len(full_names) == 1:
            result[f"last:{last}"] = result[full_names[0]]

    return result


def _compute_adjusted_team_priors(
    prior_year: int, team_priors: dict[str, float], idx: dict,
) -> dict[str, float]:
    """Compute roster-adjusted team priors using offseason transactions.

    Adjusts raw prior-season win% based on pitching talent gained/lost
    during the offseason (Oct prior_year through Mar target_year).
    """
    target_year = prior_year + 1
    tx_path = ROSTER_DIR / f"transactions_{target_year}.parquet"
    if not tx_path.exists():
        # Try prior year's offseason transactions
        tx_path = ROSTER_DIR / f"transactions_{prior_year}.parquet"
        if not tx_path.exists():
            return team_priors.copy()

    tx = pd.read_parquet(tx_path)

    # Offseason window: Oct of prior year through March of target year
    offseason_start = f"{prior_year}-10-01"
    offseason_end = f"{target_year}-03-31"
    tx = tx[(tx["date"] >= offseason_start) & (tx["date"] <= offseason_end)]
    if len(tx) == 0:
        return team_priors.copy()

    # Compute net pitcher xRV change per team
    team_xrv_change = {}
    for _, row in tx.iterrows():
        to_team = row.get("to_team")
        from_team = row.get("from_team")
        pid = int(row["player_id"]) if pd.notna(row.get("player_id")) else None

        if not pid or not to_team:
            continue

        # Check if this player is a pitcher in our xRV data
        if pid in idx.get("pitcher", {}):
            pitcher_df = idx["pitcher"][pid]
            if len(pitcher_df) >= 100:
                # Total xRV impact = per-pitch xRV * number of pitches
                # Negative xRV = good pitcher, so -total = positive talent
                total_xrv = pitcher_df["xrv"].mean() * len(pitcher_df)
                talent = -total_xrv
                team_xrv_change[to_team] = team_xrv_change.get(to_team, 0) + talent
                if from_team:
                    team_xrv_change[from_team] = team_xrv_change.get(from_team, 0) - talent

    if not team_xrv_change:
        return team_priors.copy()

    # Convert total xRV talent to win% adjustment:
    # ~10 runs = 1 win in MLB, 1 win over 162 games ≈ 0.00617 win%
    adjusted = {}
    for team, base_prior in team_priors.items():
        total_talent = team_xrv_change.get(team, 0)
        talent_in_wins = total_talent / 10.0
        wpct_adj = talent_in_wins / 162.0
        adjusted[team] = np.clip(base_prior + wpct_adj, 0.3, 0.7)
    return adjusted


def _sp_prior_season_features(idx: dict, pitcher_id: int, prior_year_cutoff: str, game_date: str = None) -> dict:
    """Compute SP features from prior season data as fallback for early season.

    Uses all pitches from prior season for the given pitcher.
    Returns the same dict format as _sp_features_fast, or None if insufficient data.
    If game_date is provided, computes actual rest days from last prior-season appearance.
    """
    if pitcher_id not in idx.get("pitcher", {}):
        return None

    pitcher_df = idx["pitcher"][pitcher_id]
    # Get only prior-season pitches (before the current season started)
    before = _get_before(pitcher_df, prior_year_cutoff)
    if len(before) < 100:
        return None

    # Use last 2000 pitches from prior season
    recent = before.iloc[-2000:] if len(before) > 2000 else before

    xrv_mean = recent["xrv"].mean()
    xrv_std = recent["xrv"].std()
    total_pa = recent["events"].notna().sum()
    k_rate = (recent["events"] == "strikeout").sum() / total_pa if total_pa > 0 else np.nan
    bb_rate = (recent["events"] == "walk").sum() / total_pa if total_pa > 0 else np.nan

    fb_mask = recent["pitch_type"].isin(["FF", "SI", "FC"])
    avg_velo = recent.loc[fb_mask, "release_speed"].mean() if fb_mask.any() else np.nan

    mix = recent["pitch_type"].value_counts(normalize=True)
    entropy = -(mix * np.log2(mix + 1e-10)).sum()

    vs_L = recent[recent["stand"] == "L"]
    vs_R = recent[recent["stand"] == "R"]
    xrv_vs_L = vs_L["xrv"].mean() if len(vs_L) >= 20 else xrv_mean
    xrv_vs_R = vs_R["xrv"].mean() if len(vs_R) >= 20 else xrv_mean

    # Home/away splits
    if "inning_topbot" in recent.columns:
        home_pitches = recent[recent["inning_topbot"] == "Top"]
        away_pitches = recent[recent["inning_topbot"] == "Bot"]
        sp_home_xrv = home_pitches["xrv"].mean() if len(home_pitches) >= 30 else xrv_mean
        sp_away_xrv = away_pitches["xrv"].mean() if len(away_pitches) >= 30 else xrv_mean
    else:
        sp_home_xrv = xrv_mean
        sp_away_xrv = xrv_mean

    # Overperformance
    sp_overperf = np.nan
    sp_overperf_recent = np.nan
    if "delta_run_exp" in recent.columns:
        resid = (recent["delta_run_exp"] - recent["xrv"]).dropna()
        if len(resid) >= 100:
            sp_overperf = resid.mean()
            sp_overperf_recent = sp_overperf  # no "recent" distinction for prior season

    # Transition entropy
    sp_transition_entropy = np.nan
    if len(recent) >= 100 and "pitch_type" in recent.columns:
        pt_seq = recent["pitch_type"].values
        transitions = {}
        for a, b in zip(pt_seq[:-1], pt_seq[1:]):
            transitions.setdefault(a, []).append(b)
        entropies = []
        for from_pt, to_pts in transitions.items():
            counts = pd.Series(to_pts).value_counts(normalize=True).values
            counts = counts[counts > 0]
            entropies.append(-float(np.sum(counts * np.log2(counts))))
        sp_transition_entropy = float(np.mean(entropies)) if entropies else 0.0

    # Compute actual rest days from last prior-season appearance
    rest_days = 14  # default fallback
    if game_date and "game_date" in before.columns:
        last_appearance = str(before["game_date"].max())
        try:
            rest_days = min((pd.Timestamp(game_date) - pd.Timestamp(last_appearance)).days, 180)
        except Exception:
            pass

    return {
        "sp_xrv_mean": xrv_mean, "sp_xrv_std": xrv_std,
        "sp_n_pitches": len(recent),
        "sp_k_rate": k_rate, "sp_bb_rate": bb_rate, "sp_avg_velo": avg_velo,
        "sp_pitch_mix_entropy": entropy,
        "sp_xrv_vs_L": xrv_vs_L, "sp_xrv_vs_R": xrv_vs_R,
        "sp_rest_days": rest_days,
        "sp_overperf": sp_overperf, "sp_overperf_recent": sp_overperf_recent,
        "sp_velo_trend": 0.0, "sp_spin_trend": 0.0, "sp_xrv_trend": 0.0,
        "sp_home_xrv": sp_home_xrv, "sp_away_xrv": sp_away_xrv,
        "sp_transition_entropy": sp_transition_entropy,
        "sp_from_prior_season": 1,
    }


def compute_single_game_features(
    game: dict,
    idx: dict,
    lineups: dict,
    matchup_models: dict,
    arsenal_models: dict,
    park_factors: dict,
    team_oaa: dict,
    team_priors: dict,
    adjusted_priors: dict,
    team_projections: dict,
    pitcher_projections: dict,
    trade_stats: dict,
    weather_map: dict,
    target_year: int,
    opening_day: str,
    prior_year_cutoff: str,
    n_pitcher_pitches: int = 2000,
) -> tuple[dict, dict]:
    """Compute all pregame features for a single game.

    Parameters
    ----------
    game : dict
        Must have keys: game_pk, game_date, home_team, away_team,
        home_sp_id, away_sp_id. Optional: home_sp_name, away_sp_name,
        home_win, home_score, away_score, weather (dict).
    lineups : dict
        {game_pk: {"home": [(pid, hand), ...], "away": [...]}}
    weather_map : dict
        {game_pk: {"temperature": ..., "wind_speed": ..., ...}} OR empty.
        If game has a "weather" key (live API format), that takes precedence.

    Returns
    -------
    (row, meta) where row is the feature dict and meta has counters:
        {"sp_prior_fallback": bool, "matchup_computed": bool}
    """
    gdate = str(game["game_date"])
    home_team = game["home_team"]
    away_team = game["away_team"]
    home_sp = game.get("home_sp_id")
    away_sp = game.get("away_sp_id")
    home_sp_name = game.get("home_sp_name", "")
    away_sp_name = game.get("away_sp_name", "")
    game_pk = game["game_pk"]

    meta = {"sp_prior_fallback": False, "matchup_computed": False}

    row = {
        "game_pk": game_pk,
        "game_date": gdate,
        "home_team": home_team,
        "away_team": away_team,
    }
    # Copy optional result fields
    for k in ("home_win", "home_score", "away_score"):
        if k in game and game[k] is not None:
            row[k] = game[k]

    nan_sp = {f"sp_{k}": np.nan for k in
              ["xrv_mean", "xrv_std", "n_pitches", "k_rate", "bb_rate", "avg_velo",
               "pitch_mix_entropy", "xrv_vs_L", "xrv_vs_R", "rest_days",
               "overperf", "overperf_recent"]}
    nan_matchup = {"matchup_xrv_mean": np.nan, "matchup_xrv_sum": np.nan,
                   "matchup_n_hitters": 0, "matchup_n_known": 0}

    # --- Starting Pitcher features (with prior-season fallback) ---
    for side, sp_id in [("home", home_sp), ("away", away_sp)]:
        got_features = False
        if pd.notna(sp_id) and int(sp_id) in idx["pitcher"]:
            stats = _sp_features_fast(idx["pitcher"][int(sp_id)], gdate, n_pitcher_pitches)
            if not np.isnan(stats.get("sp_xrv_mean", np.nan)):
                for k, v in stats.items():
                    row[f"{side}_{k}"] = v
                row[f"{side}_sp_from_prior_season"] = 0
                got_features = True
            else:
                fallback = _sp_prior_season_features(idx, int(sp_id), prior_year_cutoff, gdate)
                if fallback:
                    for k, v in fallback.items():
                        row[f"{side}_{k}"] = v
                    meta["sp_prior_fallback"] = True
                    got_features = True
        elif pd.notna(sp_id):
            fallback = _sp_prior_season_features(idx, int(sp_id), prior_year_cutoff, gdate)
            if fallback:
                for k, v in fallback.items():
                    row[f"{side}_{k}"] = v
                meta["sp_prior_fallback"] = True
                got_features = True

        if not got_features:
            for k, v in nan_sp.items():
                row[f"{side}_{k}"] = v
            row[f"{side}_sp_from_prior_season"] = np.nan

    # --- Matchup: lineup vs opposing SP (Bayesian model, hand-specific) ---
    game_lineup = lineups.get(game_pk)

    if matchup_models and game_lineup and pd.notna(away_sp):
        away_sp_int = int(away_sp)
        home_lu = game_lineup.get("home", [])
        if home_lu and away_sp_int in idx["pitcher"]:
            hitter_ids = [h[0] for h in home_lu]
            hitter_hands = [h[1] for h in home_lu]
            matchup = _compute_hand_matchup(
                matchup_models, idx["pitcher"][away_sp_int], away_sp_int,
                gdate, n_pitcher_pitches, hitter_ids, hitter_hands, compute_matchup_xrv)
            for k, v in matchup.items():
                row[f"home_{k}"] = v
            if not np.isnan(matchup["matchup_xrv_mean"]):
                meta["matchup_computed"] = True
        else:
            for k, v in nan_matchup.items():
                row[f"home_{k}"] = v
    else:
        for k, v in nan_matchup.items():
            row[f"home_{k}"] = v

    if matchup_models and game_lineup and pd.notna(home_sp):
        home_sp_int = int(home_sp)
        away_lu = game_lineup.get("away", [])
        if away_lu and home_sp_int in idx["pitcher"]:
            hitter_ids = [h[0] for h in away_lu]
            hitter_hands = [h[1] for h in away_lu]
            matchup = _compute_hand_matchup(
                matchup_models, idx["pitcher"][home_sp_int], home_sp_int,
                gdate, n_pitcher_pitches, hitter_ids, hitter_hands, compute_matchup_xrv)
            for k, v in matchup.items():
                row[f"away_{k}"] = v
        else:
            for k, v in nan_matchup.items():
                row[f"away_{k}"] = v
    else:
        for k, v in nan_matchup.items():
            row[f"away_{k}"] = v

    # --- Arsenal Matchup ---
    nan_arsenal = {"arsenal_matchup_xrv_mean": np.nan, "arsenal_matchup_xrv_sum": np.nan,
                   "arsenal_matchup_n_hitters": 0, "arsenal_matchup_n_known": 0}

    if arsenal_models and game_lineup and pd.notna(away_sp):
        away_sp_int = int(away_sp)
        home_lu = game_lineup.get("home", [])
        if home_lu and away_sp_int in idx["pitcher"]:
            hitter_ids = [h[0] for h in home_lu]
            hitter_hands = [h[1] for h in home_lu]
            a_matchup = _compute_hand_arsenal_matchup(
                arsenal_models, idx["pitcher"][away_sp_int], away_sp_int,
                gdate, n_pitcher_pitches, hitter_ids, hitter_hands)
            for k, v in a_matchup.items():
                row[f"home_{k}"] = v
        else:
            for k, v in nan_arsenal.items():
                row[f"home_{k}"] = v
    else:
        for k, v in nan_arsenal.items():
            row[f"home_{k}"] = v

    if arsenal_models and game_lineup and pd.notna(home_sp):
        home_sp_int = int(home_sp)
        away_lu = game_lineup.get("away", [])
        if away_lu and home_sp_int in idx["pitcher"]:
            hitter_ids = [h[0] for h in away_lu]
            hitter_hands = [h[1] for h in away_lu]
            a_matchup = _compute_hand_arsenal_matchup(
                arsenal_models, idx["pitcher"][home_sp_int], home_sp_int,
                gdate, n_pitcher_pitches, hitter_ids, hitter_hands)
            for k, v in a_matchup.items():
                row[f"away_{k}"] = v
        else:
            for k, v in nan_arsenal.items():
                row[f"away_{k}"] = v
    else:
        for k, v in nan_arsenal.items():
            row[f"away_{k}"] = v

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

    # --- Roster-adjusted team prior ---
    row["home_adjusted_team_prior"] = adjusted_priors.get(home_team, 0.5)
    row["away_adjusted_team_prior"] = adjusted_priors.get(away_team, 0.5)

    # --- Preseason projections ---
    row["home_projected_wpct"] = team_projections.get(home_team, np.nan)
    row["away_projected_wpct"] = team_projections.get(away_team, np.nan)

    # Pitcher projections: try MLBAM ID, then normalized name, then last-name fallback
    for side_label, sp_id_val, sp_name_val in [("home", home_sp, home_sp_name), ("away", away_sp, away_sp_name)]:
        sp_proj = None
        if pd.notna(sp_id_val):
            sp_proj = pitcher_projections.get(int(sp_id_val))
        if not sp_proj and pd.notna(sp_name_val) and sp_name_val:
            sp_proj = pitcher_projections.get(_normalize_name(str(sp_name_val)))
        if not sp_proj and pd.notna(sp_name_val) and sp_name_val:
            parts = _normalize_name(str(sp_name_val)).split()
            if parts:
                sp_proj = pitcher_projections.get(f"last:{parts[-1]}")
        if sp_proj:
            row[f"{side_label}_sp_projected_era"] = sp_proj.get("projected_era", np.nan)
            row[f"{side_label}_sp_projected_war"] = sp_proj.get("projected_war", np.nan)
        else:
            row[f"{side_label}_sp_projected_era"] = np.nan
            row[f"{side_label}_sp_projected_war"] = np.nan

    # --- Season context features ---
    row["days_into_season"] = (pd.Timestamp(gdate) - pd.Timestamp(opening_day)).days

    for side_label in ["home", "away"]:
        n_p = row.get(f"{side_label}_sp_n_pitches", 0)
        if pd.notna(n_p):
            row[f"{side_label}_sp_info_confidence"] = np.log1p(n_p) / np.log1p(2000)
        else:
            row[f"{side_label}_sp_info_confidence"] = 0.0

    # --- Trade deadline features ---
    home_trade = trade_stats.get(home_team, {})
    away_trade = trade_stats.get(away_team, {})
    if gdate >= f"{target_year}-08-01" and trade_stats:
        row["home_trade_net"] = home_trade.get("net_acquisitions", 0)
        row["away_trade_net"] = away_trade.get("net_acquisitions", 0)
        row["home_trade_pitcher_xrv"] = home_trade.get("acquired_pitcher_xrv", 0.0)
        row["away_trade_pitcher_xrv"] = away_trade.get("acquired_pitcher_xrv", 0.0)
    else:
        row["home_trade_net"] = 0
        row["away_trade_net"] = 0
        row["home_trade_pitcher_xrv"] = 0.0
        row["away_trade_pitcher_xrv"] = 0.0

    # --- Park factor ---
    row["park_factor"] = park_factors.get(home_team, 1.0)

    # --- Platoon advantage ---
    if game_lineup:
        home_lu = game_lineup.get("home", [])
        away_lu = game_lineup.get("away", [])

        if pd.notna(away_sp) and home_lu:
            away_sp_throws = None
            if int(away_sp) in idx["pitcher"]:
                sp_df = idx["pitcher"][int(away_sp)]
                if "p_throws" in sp_df.columns and len(sp_df) > 0:
                    away_sp_throws = sp_df["p_throws"].iloc[-1]
            if away_sp_throws:
                platoon_count = sum(1 for _, hand in home_lu
                                   if (hand == "L" and away_sp_throws == "R")
                                   or (hand == "R" and away_sp_throws == "L"))
                row["home_platoon_pct"] = platoon_count / len(home_lu)
                if not np.isnan(row.get("away_sp_xrv_vs_L", np.nan)):
                    n_L = sum(1 for _, h in home_lu if h == "L")
                    n_R = len(home_lu) - n_L
                    row["away_sp_xrv_vs_lineup"] = (
                        n_L * row.get("away_sp_xrv_vs_L", 0)
                        + n_R * row.get("away_sp_xrv_vs_R", 0)
                    ) / len(home_lu) if len(home_lu) > 0 else np.nan
                else:
                    row["away_sp_xrv_vs_lineup"] = np.nan
            else:
                row["home_platoon_pct"] = np.nan
                row["away_sp_xrv_vs_lineup"] = np.nan
        else:
            row["home_platoon_pct"] = np.nan
            row["away_sp_xrv_vs_lineup"] = np.nan

        if pd.notna(home_sp) and away_lu:
            home_sp_throws = None
            if int(home_sp) in idx["pitcher"]:
                sp_df = idx["pitcher"][int(home_sp)]
                if "p_throws" in sp_df.columns and len(sp_df) > 0:
                    home_sp_throws = sp_df["p_throws"].iloc[-1]
            if home_sp_throws:
                platoon_count = sum(1 for _, hand in away_lu
                                   if (hand == "L" and home_sp_throws == "R")
                                   or (hand == "R" and home_sp_throws == "L"))
                row["away_platoon_pct"] = platoon_count / len(away_lu)
                if not np.isnan(row.get("home_sp_xrv_vs_L", np.nan)):
                    n_L = sum(1 for _, h in away_lu if h == "L")
                    n_R = len(away_lu) - n_L
                    row["home_sp_xrv_vs_lineup"] = (
                        n_L * row.get("home_sp_xrv_vs_L", 0)
                        + n_R * row.get("home_sp_xrv_vs_R", 0)
                    ) / len(away_lu) if len(away_lu) > 0 else np.nan
                else:
                    row["away_platoon_pct"] = np.nan
                    row["home_sp_xrv_vs_lineup"] = np.nan
            else:
                row["away_platoon_pct"] = np.nan
                row["home_sp_xrv_vs_lineup"] = np.nan
        else:
            row["away_platoon_pct"] = np.nan
            row["home_sp_xrv_vs_lineup"] = np.nan
    else:
        row["home_platoon_pct"] = np.nan
        row["away_platoon_pct"] = np.nan
        row["away_sp_xrv_vs_lineup"] = np.nan
        row["home_sp_xrv_vs_lineup"] = np.nan

    # --- Bullpen matchup (Bayesian model, hand-specific) ---
    bp_matchup_artifacts = matchup_models.get("L") or matchup_models.get("R") if matchup_models else None
    if bp_matchup_artifacts and game_lineup:
        away_lu = game_lineup.get("away", [])
        if away_lu:
            away_hitter_ids = [h[0] for h in away_lu]
            away_hitter_hands = [h[1] for h in away_lu]
            bp_m = compute_bullpen_matchup_xrv(
                bp_matchup_artifacts, idx, home_team, gdate,
                away_hitter_ids, away_hitter_hands)
            for k, v in bp_m.items():
                row[f"home_{k}"] = v
        else:
            row["home_bp_matchup_xrv_mean"] = np.nan
            row["home_bp_matchup_n_relievers"] = 0

        home_lu = game_lineup.get("home", [])
        if home_lu:
            home_hitter_ids = [h[0] for h in home_lu]
            home_hitter_hands = [h[1] for h in home_lu]
            bp_m = compute_bullpen_matchup_xrv(
                bp_matchup_artifacts, idx, away_team, gdate,
                home_hitter_ids, home_hitter_hands)
            for k, v in bp_m.items():
                row[f"away_{k}"] = v
        else:
            row["away_bp_matchup_xrv_mean"] = np.nan
            row["away_bp_matchup_n_relievers"] = 0
    else:
        row["home_bp_matchup_xrv_mean"] = np.nan
        row["home_bp_matchup_n_relievers"] = 0
        row["away_bp_matchup_xrv_mean"] = np.nan
        row["away_bp_matchup_n_relievers"] = 0

    # --- Bullpen arsenal matchup ---
    bp_arsenal_artifacts = arsenal_models.get("L") or arsenal_models.get("R") if arsenal_models else None
    if bp_arsenal_artifacts and game_lineup:
        away_lu = game_lineup.get("away", [])
        if away_lu:
            bp_a = compute_bullpen_arsenal_matchup_xrv(
                bp_arsenal_artifacts, idx, home_team, gdate,
                [h[0] for h in away_lu], [h[1] for h in away_lu])
            for k, v in bp_a.items():
                row[f"home_{k}"] = v
        else:
            row["home_bp_arsenal_matchup_xrv_mean"] = np.nan
            row["home_bp_arsenal_matchup_n_relievers"] = 0

        home_lu = game_lineup.get("home", [])
        if home_lu:
            bp_a = compute_bullpen_arsenal_matchup_xrv(
                bp_arsenal_artifacts, idx, away_team, gdate,
                [h[0] for h in home_lu], [h[1] for h in home_lu])
            for k, v in bp_a.items():
                row[f"away_{k}"] = v
        else:
            row["away_bp_arsenal_matchup_xrv_mean"] = np.nan
            row["away_bp_arsenal_matchup_n_relievers"] = 0
    else:
        row["home_bp_arsenal_matchup_xrv_mean"] = np.nan
        row["home_bp_arsenal_matchup_n_relievers"] = 0
        row["away_bp_arsenal_matchup_xrv_mean"] = np.nan
        row["away_bp_arsenal_matchup_n_relievers"] = 0

    # --- Weather ---
    # Live path: game["weather"] dict from MLB API
    # Batch path: weather_map[game_pk] from weather parquet
    live_weather = game.get("weather")
    if live_weather and isinstance(live_weather, dict) and live_weather.get("temp"):
        row["temperature"] = float(live_weather.get("temp", 0)) if live_weather.get("temp") else np.nan
        wind_str = live_weather.get("wind", "")
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
        row["is_dome"] = int("dome" in venue.lower() or "roof" in live_weather.get("condition", "").lower())
    elif weather_map and game_pk in weather_map:
        w = weather_map[game_pk]
        row["temperature"] = w.get("temperature")
        row["wind_speed"] = w.get("wind_speed", 0.0)
        row["is_dome"] = w.get("is_dome", 0)
        wind_dir = str(w.get("wind_dir", "none")).lower()
        row["wind_out"] = int("out" in wind_dir)
        row["wind_in"] = int("in" in wind_dir)
    else:
        row["temperature"] = np.nan
        row["wind_speed"] = np.nan
        row["is_dome"] = np.nan
        row["wind_out"] = np.nan
        row["wind_in"] = np.nan

    row["is_home"] = 1

    return row, meta


def build_game_features(
    target_year: int,
    n_pitcher_pitches: int = 2000,
) -> pd.DataFrame:
    """
    Build pregame feature matrix for all games in target_year.

    Uses xRV data from CURRENT season (rolling, no lookahead) plus
    prior season for park factors and early-season priors.
    Integrates Bayesian matchup model predictions using actual lineups.
    """
    print(f"\n{'='*60}")
    print(f"Building game features for {target_year}")
    print(f"{'='*60}")

    # Load game results
    games_path = GAMES_DIR / f"games_{target_year}.parquet"
    if not games_path.exists():
        raise FileNotFoundError(f"Game data not found for {target_year}")
    games = pd.read_parquet(games_path)
    games = games[games["game_type"] == "R"].copy()  # Regular season only
    games = games.sort_values("game_date").reset_index(drop=True)
    print(f"  {len(games)} regular season games")

    # Load xRV data — current year + prior year for early-season features
    xrv_frames = []
    for yr in [target_year - 1, target_year]:
        xrv_path = XRV_DIR / f"statcast_xrv_{yr}.parquet"
        if xrv_path.exists():
            xrv_frames.append(pd.read_parquet(xrv_path))
            print(f"  Loaded xRV for {yr}: {len(xrv_frames[-1]):,} pitches")
    if not xrv_frames:
        raise FileNotFoundError("No xRV data available")
    xrv = pd.concat(xrv_frames, ignore_index=True)
    xrv["game_date"] = xrv["game_date"].astype(str)

    # Pre-index for fast lookups
    print("  Pre-indexing xRV data...")
    idx = _preindex_xrv(xrv)

    # Extract lineups from current season xRV data
    # (lineups are known pregame — published hours before first pitch)
    current_xrv_path = XRV_DIR / f"statcast_xrv_{target_year}.parquet"
    lineups = {}
    if current_xrv_path.exists():
        current_xrv = pd.read_parquet(current_xrv_path)
        print("  Extracting lineups from pitch data...")
        lineups = _extract_lineups(current_xrv)
        print(f"  Extracted lineups for {len(lineups):,} games")

    # Load hand-specific matchup models (trained on prior season to avoid lookahead)
    matchup_models = _load_hand_models("matchup_model", target_year - 1)
    if matchup_models:
        print(f"  Matchup models loaded for hands: {list(matchup_models.keys())}")
    else:
        print(f"  WARNING: No matchup model for {target_year - 1}, skipping matchup features")

    # Load hand-specific arsenal matchup models (trained on prior season)
    arsenal_models = _load_hand_models("arsenal_matchup", target_year - 1)
    if arsenal_models:
        print(f"  Arsenal models loaded for hands: {list(arsenal_models.keys())}")
    else:
        print(f"  WARNING: No arsenal model for {target_year - 1}, skipping arsenal matchup features")

    # Compute park factors from prior season
    prior_xrv_path = XRV_DIR / f"statcast_xrv_{target_year - 1}.parquet"
    if prior_xrv_path.exists():
        prior_xrv = pd.read_parquet(prior_xrv_path)
        park_factors = compute_park_factors(prior_xrv, target_year - 1)
        print(f"  Park factors from {target_year - 1}: {len(park_factors)} parks")
    else:
        park_factors = {}

    # Load weather data
    weather_path = WEATHER_DIR / f"weather_{target_year}.parquet"
    weather_map = {}
    if weather_path.exists():
        weather_df = pd.read_parquet(weather_path)
        for _, w in weather_df.iterrows():
            weather_map[w["game_pk"]] = w.to_dict()
        print(f"  Weather data for {len(weather_map)} games")
    else:
        print(f"  WARNING: No weather data for {target_year}")

    # Load team OAA from prior season (no lookahead)
    team_oaa = _load_team_oaa(target_year - 1)
    if team_oaa:
        print(f"  Team OAA from {target_year - 1}: {len(team_oaa)} teams")
    else:
        print(f"  WARNING: No OAA data for {target_year - 1}")

    # Compute team priors from prior season win%
    team_priors = _compute_team_priors(target_year - 1)
    if team_priors:
        print(f"  Team priors from {target_year - 1}: {len(team_priors)} teams")

    # Load preseason projections (Step 3)
    team_projections = _load_team_projections(target_year)
    if team_projections:
        print(f"  Team projections for {target_year}: {len(team_projections)} teams")
    else:
        print(f"  WARNING: No team projections for {target_year}")

    pitcher_projections = _load_pitcher_projections(target_year)
    if pitcher_projections:
        print(f"  Pitcher projections for {target_year}: {len(pitcher_projections)} pitchers")
    else:
        print(f"  WARNING: No pitcher projections for {target_year}")

    # Compute roster-adjusted team priors (Step 6)
    adjusted_priors = _compute_adjusted_team_priors(target_year - 1, team_priors, idx)
    if adjusted_priors != team_priors:
        print(f"  Roster-adjusted priors computed for {len(adjusted_priors)} teams")

    # Determine Opening Day for days_into_season (Step 4)
    opening_day = OPENING_DAY.get(target_year, str(games["game_date"].min()))
    print(f"  Opening Day: {opening_day}")

    # Load trade deadline acquisitions (current season, only affects games after deadline)
    trade_stats = _load_trade_deadline_acquisitions(target_year, idx)
    if trade_stats:
        print(f"  Trade deadline stats: {len(trade_stats)} teams with transactions")

    # Prior-season cutoff for SP fallback (Step 2)
    prior_year_cutoff = f"{target_year}-01-01"
    sp_prior_fallback_count = 0

    # Build features for each game
    feature_rows = []
    total = len(games)
    matchup_computed = 0

    for i, game in games.iterrows():
        game_dict = {
            "game_pk": game["game_pk"],
            "game_date": str(game["game_date"]),
            "home_team": game["home_team_abbr"],
            "away_team": game["away_team_abbr"],
            "home_sp_id": game.get("home_sp_id"),
            "away_sp_id": game.get("away_sp_id"),
            "home_sp_name": game.get("home_sp_name", ""),
            "away_sp_name": game.get("away_sp_name", ""),
            "home_win": game["home_win"],
            "home_score": game["home_score"],
            "away_score": game["away_score"],
        }
        row, meta = compute_single_game_features(
            game_dict, idx, lineups, matchup_models, arsenal_models,
            park_factors, team_oaa, team_priors, adjusted_priors,
            team_projections, pitcher_projections, trade_stats,
            weather_map, target_year, opening_day, prior_year_cutoff,
            n_pitcher_pitches,
        )
        if meta["sp_prior_fallback"]:
            sp_prior_fallback_count += 1
        if meta["matchup_computed"]:
            matchup_computed += 1

        feature_rows.append(row)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total} games processed ({matchup_computed} matchups computed)")

    features_df = pd.DataFrame(feature_rows)
    print(f"  Matchup features computed for {matchup_computed}/{total} games")
    if sp_prior_fallback_count:
        print(f"  SP prior-season fallback used for {sp_prior_fallback_count} pitcher-games")

    # --- Team recent form + games played (computed vectorized over the games df) ---
    # Win% in last 10 games for each team, computed without lookahead
    games_sorted = games.sort_values("game_date").reset_index(drop=True)
    home_form = {}
    away_form = {}
    home_games_played = {}
    away_games_played = {}
    team_history = {}  # team -> list of (date, win_flag)
    for _, g in games_sorted.iterrows():
        ht = g["home_team_abbr"]
        at = g["away_team_abbr"]
        hw = g["home_win"]

        # Look up recent form and games played before this game
        home_form[g["game_pk"]] = _recent_winpct(team_history.get(ht, []), 10)
        away_form[g["game_pk"]] = _recent_winpct(team_history.get(at, []), 10)
        home_games_played[g["game_pk"]] = len(team_history.get(ht, []))
        away_games_played[g["game_pk"]] = len(team_history.get(at, []))

        # Update history
        team_history.setdefault(ht, []).append(hw)
        team_history.setdefault(at, []).append(1 - hw)

    features_df["home_recent_form"] = features_df["game_pk"].map(home_form)
    features_df["away_recent_form"] = features_df["game_pk"].map(away_form)
    features_df["home_team_games_played"] = features_df["game_pk"].map(home_games_played)
    features_df["away_team_games_played"] = features_df["game_pk"].map(away_games_played)

    # Compute differentials (home - away) for cleaner modeling
    for col, sign in DIFF_COLS:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in features_df.columns and away_col in features_df.columns:
            features_df[f"diff_{col}"] = sign * (
                features_df[home_col] - features_df[away_col]
            )

    # SP xRV vs actual lineup (handedness-weighted) — the platoon-aware SP quality
    if "home_sp_xrv_vs_lineup" in features_df.columns and "away_sp_xrv_vs_lineup" in features_df.columns:
        features_df["diff_sp_xrv_vs_lineup"] = -(
            features_df["home_sp_xrv_vs_lineup"] - features_df["away_sp_xrv_vs_lineup"]
        )

    # Context-aware SP xRV: home SP uses their home split, away SP uses road split
    if "home_sp_home_xrv" in features_df.columns and "away_sp_away_xrv" in features_df.columns:
        features_df["diff_sp_context_xrv"] = -(
            features_df["home_sp_home_xrv"] - features_df["away_sp_away_xrv"]
        )

    # Save
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / f"game_features_{target_year}.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"\n  Saved {len(features_df)} game feature vectors to {out_path}")
    print(f"  Feature columns: {len(features_df.columns)}")

    # Summary stats
    non_null_pct = features_df.notna().mean()
    sparse_cols = non_null_pct[non_null_pct < 0.5].index.tolist()
    if sparse_cols:
        print(f"  WARNING: Sparse columns (<50% non-null): {sparse_cols}")

    return features_df


def main():
    parser = argparse.ArgumentParser(description="Build game-level features")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--n-pitches", type=int, default=2000,
                        help="Pitcher rolling window size")
    args = parser.parse_args()

    build_game_features(args.season, n_pitcher_pitches=args.n_pitches)


if __name__ == "__main__":
    main()
