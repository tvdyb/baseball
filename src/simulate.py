#!/usr/bin/env python3
"""
Monte Carlo MLB Game Simulator.

Simulates games at-bat-by-at-bat using the Bayesian matchup model to derive
per-hitter outcome distributions, then runs thousands of simulations to
estimate win probability.

Works both pregame (once SPs and lineups are announced) and mid-game
(given current game state from the MLB live feed API).

Usage:
    python src/simulate.py                              # today's games
    python src/simulate.py --date 2025-04-15            # specific date
    python src/simulate.py --game-pk 745123             # specific game (mid-game if live)
    python src/simulate.py --backtest 2025              # validate against full season
    python src/simulate.py --home NYY --away BOS \\
        --home-sp 543243 --away-sp 605141 --n-sims 50000
"""

import argparse
import functools
import pickle
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_transition_matrix import (
    OUTCOME_ORDER, _deterministic_transition, _classify_event, EVENT_MAP, BBTYPE_MAP,
)
from feature_engineering import (
    _preindex_xrv,
    _compute_pitcher_arsenal_live,
    _standardize_arsenal,
    compute_matchup_xrv,
    _load_hand_models,
    _sp_features_fast,
    _bp_features_fast,
    _get_before,
)
from multi_output_matchup_model import (
    load_multi_output_models,
    predict_matchup_distribution,
    PITCH_TYPES as MO_PITCH_TYPES,
    PTYPE_MAP as MO_PTYPE_MAP,
    N_PTYPES as MO_N_PTYPES,
)
from predict import fetch_todays_games, fetch_lineup, MLB_API
from utils import DATA_DIR, XRV_DIR, MODEL_DIR

SIM_DIR = DATA_DIR / "sim"


# ── Eval Score Loading ────────────────────────────────────

@functools.lru_cache(maxsize=4)
def _load_eval_scores(season: int) -> dict:
    """Load pitcher (P1/location/P3) and hitter (H1/H2/H4) eval scores.

    Returns dict with keys: 'pitcher_stuff', 'pitcher_location',
    'pitcher_sequencing', 'hitter_eval', each mapping pitcher/batter IDs
    to their scores.  Missing files are silently skipped.
    """
    result = {
        "pitcher_stuff_vsL": {}, "pitcher_stuff_vsR": {},
        "pitcher_loc_vsL": {}, "pitcher_loc_vsR": {},
        "pitcher_seq_vsL": {}, "pitcher_seq_vsR": {},
        "hitter_swing": {},     # H1
        "hitter_contact": {},   # H2
        "hitter_bip": {},       # H4
    }

    # P1 — stuff (no location)
    p1_path = MODEL_DIR / f"stuff_noloc_{season}.pkl"
    if p1_path.exists():
        with open(p1_path, "rb") as f:
            p1 = pickle.load(f)
        result["pitcher_stuff_vsL"] = p1.get("pitcher_overall_vsL", {})
        result["pitcher_stuff_vsR"] = p1.get("pitcher_overall_vsR", {})

    # Location scores (P2 - P1)
    loc_path = MODEL_DIR / f"location_scores_{season}.pkl"
    if loc_path.exists():
        with open(loc_path, "rb") as f:
            loc = pickle.load(f)
        for hand in ("L", "R"):
            hand_data = loc.get(f"vs{hand}", {})
            for pid, v in hand_data.items():
                if isinstance(v, dict) and "overall" in v:
                    result[f"pitcher_loc_vs{hand}"][pid] = v["overall"]

    # P3 — sequencing
    seq_path = MODEL_DIR / f"sequencing_{season}.pkl"
    if seq_path.exists():
        with open(seq_path, "rb") as f:
            seq = pickle.load(f)
        for hand in ("L", "R"):
            hand_data = seq.get(f"sequencing_vs{hand}", {})
            for pid, v in hand_data.items():
                if isinstance(v, dict) and "sequencing_score" in v:
                    result[f"pitcher_seq_vs{hand}"][pid] = v["sequencing_score"]

    # Hitter eval (H1, H2, H4)
    h_path = MODEL_DIR / f"hitter_eval_{season}.pkl"
    if h_path.exists():
        with open(h_path, "rb") as f:
            h = pickle.load(f)
        # H1: swing decision z-score
        for bid, v in h.get("swing_decision", {}).items():
            if isinstance(v, dict) and v.get("z_score") is not None:
                result["hitter_swing"][bid] = v["z_score"]
        # H2: in-zone contact skill
        for bid, v in h.get("contact_rates", {}).items():
            if isinstance(v, dict) and v.get("iz_contact_skill") is not None:
                result["hitter_contact"][bid] = v["iz_contact_skill"]
        # H4: ball-in-play quality
        for bid, v in h.get("bip_quality", {}).items():
            if isinstance(v, dict) and v.get("bip_quality_iz") is not None:
                result["hitter_bip"][bid] = v["bip_quality_iz"]

    return result


# Score distribution constants (from empirical analysis of 2024 models)
_P1_STD = 0.006   # pitcher stuff score std
_LOC_STD = 0.003  # pitcher location score std
_SEQ_STD = 0.001  # pitcher sequencing score std
_H1_STD = 1.0     # hitter swing decision (already z-scored)
_H2_STD = 0.052   # hitter in-zone contact residual std
_H4_STD = 0.053   # hitter BIP quality std
_H4_MEAN = 0.060  # hitter BIP quality mean


def _compute_eval_adjustment(
    pitcher_id: int,
    batter_id: int,
    bat_hand: str,
    eval_scores: dict,
    sensitivity: float = 0.012,
) -> float:
    """Compute net eval-based distribution adjustment for a matchup.

    Returns a shift value: positive = boost offense (better hitter or worse pitcher),
    negative = suppress offense (better pitcher or worse batter).
    The shift is applied the same way as calibration_shift.
    """
    hand = bat_hand if bat_hand in ("L", "R") else "R"

    # Pitcher quality (positive = worse pitcher = allows more runs)
    p_stuff = eval_scores[f"pitcher_stuff_vs{hand}"].get(pitcher_id, 0.0)
    p_loc = eval_scores[f"pitcher_loc_vs{hand}"].get(pitcher_id, 0.0)
    p_seq = eval_scores[f"pitcher_seq_vs{hand}"].get(pitcher_id, 0.0)

    # Z-score each component, then weighted combine
    p_z = (0.50 * (p_stuff / _P1_STD) +
           0.30 * (p_loc / _LOC_STD) +
           0.20 * (p_seq / _SEQ_STD))

    # Hitter quality (positive = better hitter)
    h_swing = eval_scores["hitter_swing"].get(batter_id, 0.0)
    h_contact = eval_scores["hitter_contact"].get(batter_id, 0.0)
    h_bip = eval_scores["hitter_bip"].get(batter_id, 0.0)

    h_z = (0.40 * (h_swing / _H1_STD) +
           0.30 * (h_contact / _H2_STD) +
           0.30 * ((h_bip - _H4_MEAN) / _H4_STD))

    # Net: positive pitcher_z = worse pitcher (more offense)
    # positive hitter_z = better hitter (more offense)
    net_z = p_z + h_z

    # Convert to probability shift, capped at ±0.04
    shift = net_z * sensitivity
    return max(-0.04, min(0.04, shift))


def _apply_eval_shift(dist: np.ndarray, shift: float) -> np.ndarray:
    """Apply eval-based shift to an outcome distribution.

    Positive shift = boost offense (move mass from outs to hits/walks).
    Negative shift = suppress offense (move mass from hits/walks to outs).
    Same mechanics as _calibrate_distribution but bidirectional.
    """
    if abs(shift) < 1e-6:
        return dist
    adjusted = dist.copy()
    offense_total = sum(adjusted[i] for i in _OFFENSE_IDX)
    out_total = sum(adjusted[i] for i in _OUT_IDX)
    if offense_total <= 0 or out_total <= 0:
        return dist

    if shift > 0:
        # Boost offense: move mass from outs to offense
        for i in _OUT_IDX:
            adjusted[i] -= shift * (adjusted[i] / out_total)
        for i in _OFFENSE_IDX:
            adjusted[i] += shift * (adjusted[i] / offense_total)
    else:
        # Suppress offense: move mass from offense to outs
        abs_shift = -shift
        for i in _OFFENSE_IDX:
            adjusted[i] -= abs_shift * (adjusted[i] / offense_total)
        for i in _OUT_IDX:
            adjusted[i] += abs_shift * (adjusted[i] / out_total)

    adjusted = np.maximum(adjusted, 1e-8)
    adjusted /= adjusted.sum()
    return adjusted


# Pre-computed indices for hot-path performance
_DP_IDX = OUTCOME_ORDER.index("dp")
_OG_IDX = OUTCOME_ORDER.index("out_ground")
_OF_IDX = OUTCOME_ORDER.index("out_fly")
_OL_IDX = OUTCOME_ORDER.index("out_line")
_K_IDX = OUTCOME_ORDER.index("K")
_BB_IDX = OUTCOME_ORDER.index("BB")
_HBP_IDX = OUTCOME_ORDER.index("HBP")
_1B_IDX = OUTCOME_ORDER.index("1B")
_2B_IDX = OUTCOME_ORDER.index("2B")
_3B_IDX = OUTCOME_ORDER.index("3B")
_HR_IDX = OUTCOME_ORDER.index("HR")
_N_OUTCOMES = len(OUTCOME_ORDER)

# Outcome category index sets for calibration
_OFFENSE_IDX = [_BB_IDX, _HBP_IDX, _1B_IDX, _2B_IDX, _3B_IDX, _HR_IDX]
_OUT_IDX = [_K_IDX, _OG_IDX, _OF_IDX, _OL_IDX]

# First-inning adjustment factors: ratio of inning-1 outcome rates to overall rates.
# Empirically derived from 2024 Statcast (21,716 first-inning PAs vs 191,819 total).
# Applied to SP distributions in inning 1 to improve NRFI predictions.
# First-inning adjustment disabled — ablation showed it hurts NRFI predictions.
# The empirical ratios are preserved for reference but set to 1.0 (no-op).
# Original values: K=1.026, BB=1.046, HBP=1.070, 1B=0.957, 2B=1.004,
#   3B=0.964, HR=1.189, dp=1.005, out_ground=0.926, out_fly=1.026, out_line=1.004
_FIRST_INNING_ADJ = {o: 1.0 for o in [
    "K", "BB", "HBP", "1B", "2B", "3B", "HR", "dp", "out_ground", "out_fly", "out_line",
]}

# Outs added per outcome type (pre-computed for hot path)
_OUTS_ADDED = {
    "K": 1, "BB": 0, "HBP": 0, "1B": 0, "2B": 0, "3B": 0, "HR": 0,
    "dp": 2, "out_ground": 1, "out_fly": 1, "out_line": 1,
}


# ── Data Structures ────────────────────────────────────────

@dataclass
class GameState:
    """Complete state of a game at any point."""
    inning: int = 1
    top_bottom: str = "Top"       # "Top" or "Bot"
    outs: int = 0
    bases: tuple[bool, bool, bool] = (False, False, False)  # 1B, 2B, 3B
    home_score: int = 0
    away_score: int = 0
    home_lineup_pos: int = 0      # 0-8 index into batting order
    away_lineup_pos: int = 0
    home_pitcher: str = "sp"      # "sp" or "bp"
    away_pitcher: str = "sp"
    home_sp_pitches: int = 0
    away_sp_pitches: int = 0
    home_rp_pitches: float = 0.0  # current reliever pitch count
    away_rp_pitches: float = 0.0
    home_reliever_idx: int = 0    # index into bullpen reliever order
    away_reliever_idx: int = 0


@dataclass
class RelieverInfo:
    """A single reliever's identity and outcome distributions vs each hitter."""
    pitcher_id: int
    outcome_dists: list[np.ndarray]  # per-hitter, 9 arrays of shape (N_OUTCOMES,)
    pitches_per_pa: float = 3.9      # K-heavy pitchers throw more per PA


@dataclass
class TeamContext:
    """A team's offensive and pitching identity for simulation.

    Outcome distributions are per-hitter, per-pitcher-state numpy arrays
    in OUTCOME_ORDER, derived from batter/pitcher empirical rates combined
    via log5 and adjusted by the multi-output matchup model.
    """
    team: str                                    # abbreviation
    lineup: list[tuple[int, str]]                # [(player_id, bat_side), ...] x9
    # Per-hitter outcome distributions: list of 9 numpy arrays, one per lineup slot
    sp_outcome_dists: list[np.ndarray] = field(default_factory=list)   # vs opposing SP
    sp_pitches_per_pa: float = 3.9               # SP-specific pitches/PA estimate
    # Individual relievers (ordered by expected usage: high-leverage first)
    relievers: list[RelieverInfo] = field(default_factory=list)
    # Fallback composite bullpen dists (used when no reliever data available)
    bp_outcome_dists: list[np.ndarray] = field(default_factory=list)   # vs opposing bullpen


@dataclass
class SimConfig:
    """Simulation parameters.

    Defaults are based on 2017-2024 MLB averages:
    - SP pitch limit: 92 ± 10 (MLB avg ~91 pitches/start, Gaussian for variance)
    - Max PA per half-inning: 25 (safety limit; MLB record is 23 batters)
    - Reliever pitch limit: 20 ± 5 (typical reliever outing ~15-25 pitches)
    """
    n_sims: int = 10_000
    sp_pitch_limit_mean: int = 92
    sp_pitch_limit_std: int = 10
    reliever_pitch_limit_mean: int = 20
    reliever_pitch_limit_std: int = 5
    max_pa_per_half_inning: int = 25
    random_seed: int | None = None
    model_blend_weight: float = 0.6  # weight for multi-output model vs log5
    home_advantage: float = 0.012    # calibrated: identical teams → ~53.5% home WP
    calibration_shift: float = 0.014  # compensate for Markov independence + model blend inflation
    log5_amplify: float = 1.3        # amplify batter/pitcher deviations from league avg
    eval_sensitivity: float = 0.012  # probability shift per z-score from pitcher/hitter eval models
    wp_temperature: float = 1.0      # scale WP deviations from 0.50 (1.0 = no change)


# ── Outcome Distribution ──────────────────────────────────
# Per-batter, per-pitcher outcome distributions derived from:
#   1. Batter's empirical PA outcome rates (recent history)
#   2. Pitcher's empirical PA outcome rates (recent history)
#   3. Log5 odds-ratio combination (standard baseball method)
#   4. Bayesian matchup model interaction adjustment

# Minimum PA thresholds before trusting individual rates over league avg.
# Below these thresholds, we try MiLB stats, then fall back to league averages.
# 50 PAs ≈ 2 weeks of games for a regular starter.
_MIN_BATTER_PA = 50
_MIN_PITCHER_PA = 100

# Estimated pitches per PA by outcome (high-K pitchers throw more per PA)
_PITCHES_PER_PA_BY_OUTCOME = {
    "K": 4.8, "BB": 5.6, "HBP": 2.5, "1B": 3.5, "2B": 3.4, "3B": 3.3,
    "HR": 3.6, "dp": 3.0, "out_ground": 3.4, "out_fly": 3.6, "out_line": 3.3,
}

# MiLB level adjustment factors (scale MiLB stats to approximate MLB equivalents)
# sportId: 11=AAA, 12=AA, 13=A+, 14=A
_MILB_LEVEL_ADJUSTMENTS = {
    11: {"K": 1.05, "BB": 0.97, "HR": 0.82, "1B": 0.98, "2B": 0.95, "3B": 0.90,
         "HBP": 1.0, "dp": 1.0, "out_ground": 1.02, "out_fly": 1.02, "out_line": 1.02},
    12: {"K": 1.10, "BB": 0.95, "HR": 0.72, "1B": 0.95, "2B": 0.90, "3B": 0.85,
         "HBP": 1.0, "dp": 1.0, "out_ground": 1.05, "out_fly": 1.05, "out_line": 1.05},
    13: {"K": 1.15, "BB": 0.92, "HR": 0.62, "1B": 0.92, "2B": 0.85, "3B": 0.80,
         "HBP": 1.0, "dp": 1.0, "out_ground": 1.08, "out_fly": 1.08, "out_line": 1.08},
    14: {"K": 1.20, "BB": 0.90, "HR": 0.55, "1B": 0.90, "2B": 0.80, "3B": 0.75,
         "HBP": 1.0, "dp": 1.0, "out_ground": 1.10, "out_fly": 1.10, "out_line": 1.10},
}


def load_sim_artifacts() -> tuple[dict, dict]:
    """Load pre-computed simulation artifacts (base rates + transition matrix)."""
    with open(SIM_DIR / "league_base_rates.pkl", "rb") as f:
        base_rates = pickle.load(f)
    with open(SIM_DIR / "transition_matrix.pkl", "rb") as f:
        transition_matrix = pickle.load(f)
    return base_rates, transition_matrix


def _classify_pa_events(df: pd.DataFrame) -> pd.Series:
    """Vectorized classification of Statcast events to sim outcome categories.

    Much faster than row-by-row _classify_event for large DataFrames.
    """
    pa = df[df["events"].notna()].copy()
    outcome = pa["events"].map(EVENT_MAP)

    # Handle field_out using bb_type
    field_out_mask = pa["events"] == "field_out"
    if field_out_mask.any():
        bb_mapped = pa.loc[field_out_mask, "bb_type"].map(BBTYPE_MAP).fillna("out_ground")
        outcome.loc[field_out_mask] = bb_mapped

    # Catch-all for unmapped events
    still_null = outcome.isna()
    if still_null.any():
        has_dp = pa.loc[still_null, "events"].str.contains("double_play", na=False)
        outcome.loc[still_null & has_dp] = "dp"
        has_out = pa.loc[still_null & ~has_dp, "events"].str.contains("out", na=False)
        outcome.loc[still_null & ~has_dp & has_out] = "out_ground"

    return outcome


def compute_outcome_rates(
    df: pd.DataFrame,
    game_date: str,
    n_pa: int = 500,
    min_pa: int | None = None,
) -> tuple[dict[str, float], int] | None:
    """Compute empirical PA outcome rates from a player's recent PAs.

    Works for both batters (pass batter-filtered df) and pitchers
    (pass pitcher-filtered df).

    Returns (rates_dict, n_pa_used) or None if insufficient data.
    """
    if min_pa is None:
        min_pa = _MIN_BATTER_PA
    before = _get_before(df, game_date)
    # Get PA rows (where events is not null)
    pa = before[before["events"].notna()]
    pa = pa.iloc[-n_pa:] if len(pa) > n_pa else pa

    if len(pa) < min_pa:
        return None

    outcomes = _classify_pa_events(pa)
    outcomes = outcomes.dropna()
    total = len(outcomes)
    if total == 0:
        return None

    counts = outcomes.value_counts()
    rates = {o: counts.get(o, 0) / total for o in OUTCOME_ORDER}
    return rates, total


def _shrink_rates(
    rates: dict[str, float],
    n_pa: int,
    league_rates: dict[str, float],
    shrinkage_pa: int = 120,
) -> dict[str, float]:
    """Apply Bayesian shrinkage toward league average.

    With shrinkage_pa=120:
      50 PAs  → 29% individual signal
      120 PAs → 50% individual signal
      300 PAs → 71% individual signal
      500 PAs → 81% individual signal
    """
    weight = n_pa / (n_pa + shrinkage_pa)
    shrunk = {}
    for o in OUTCOME_ORDER:
        shrunk[o] = weight * rates.get(o, 0.0) + (1 - weight) * league_rates.get(o, 0.0)
    return shrunk


def log5_combine(
    batter_rates: dict[str, float],
    pitcher_rates: dict[str, float],
    league_rates: dict[str, float],
    amplify: float = 1.0,
) -> dict[str, float]:
    """Combine batter and pitcher outcome rates using the log5 / odds-ratio method.

    P(outcome|B,P) ∝ L * ((B * P) / (L * L)) ^ amplify

    With amplify=1.0 this is standard log5: (B * P) / L.
    With amplify>1.0 (e.g. 1.3), deviations from league average are
    exaggerated — a pitcher 20% better than average is treated as 26% better.
    This increases game-to-game variance without changing the overall mean.
    """
    combined = {}
    for o in OUTCOME_ORDER:
        b = max(batter_rates.get(o, 0.0), 1e-6)
        p = max(pitcher_rates.get(o, 0.0), 1e-6)
        lg = max(league_rates.get(o, 0.0), 1e-6)
        if amplify == 1.0:
            combined[o] = (b * p) / lg
        else:
            ratio = (b * p) / (lg * lg)
            combined[o] = lg * (ratio ** amplify)

    # Normalize
    total = sum(combined.values())
    if total > 0:
        combined = {o: v / total for o, v in combined.items()}
    return combined


def rates_to_probs(rates: dict[str, float]) -> np.ndarray:
    """Convert outcome rate dict to numpy probability array."""
    total = sum(rates.values())
    if total <= 0:
        return np.full(_N_OUTCOMES, 1.0 / _N_OUTCOMES)
    return np.array([rates.get(o, 0.0) / total for o in OUTCOME_ORDER])


def build_matchup_distribution(
    batter_rates: dict[str, float] | None,
    pitcher_rates: dict[str, float] | None,
    league_rates: dict[str, float],
    model_probs: np.ndarray | None = None,
    similarity_probs: np.ndarray | None = None,
    model_weight: float = 0.45,
    similarity_weight: float = 0.20,
    log5_weight: float = 0.35,
    amplify: float = 1.0,
) -> np.ndarray:
    """Build a full per-batter-pitcher outcome distribution.

    Combines:
      1. Batter's empirical outcome rates (or league avg if unknown)
      2. Pitcher's empirical outcome rates (or league avg if unknown)
      3. Log5 odds-ratio combination (with optional amplification)
      4. Blend with multi-output matchup model predictions (if available)
      5. Blend with similarity-based matchup predictions (if available)

    Returns numpy probability array in OUTCOME_ORDER.
    """
    b_rates = batter_rates if batter_rates is not None else league_rates
    p_rates = pitcher_rates if pitcher_rates is not None else league_rates

    log5_probs = rates_to_probs(log5_combine(b_rates, p_rates, league_rates,
                                              amplify=amplify))

    if model_probs is None and similarity_probs is None:
        return log5_probs

    # Compute effective weights based on available components
    w_model = model_weight if model_probs is not None else 0.0
    w_sim = similarity_weight if similarity_probs is not None else 0.0
    w_log5 = 1.0 - w_model - w_sim  # Remainder goes to log5

    blended = w_log5 * log5_probs
    if model_probs is not None:
        blended += w_model * model_probs
    if similarity_probs is not None:
        blended += w_sim * similarity_probs

    blended /= blended.sum()
    return blended


def _make_no_dp_dist(probs: np.ndarray) -> np.ndarray:
    """Create a dp-adjusted distribution (redistribute dp prob to out_ground)."""
    adjusted = probs.copy()
    adjusted[_OG_IDX] += adjusted[_DP_IDX]
    adjusted[_DP_IDX] = 0.0
    total = adjusted.sum()
    if total > 0:
        adjusted /= total
    return adjusted


def _calibrate_distribution(dist: np.ndarray, shift: float) -> np.ndarray:
    """Apply global dampener to compensate for Markov independence assumption.

    Shifts probability mass from offensive outcomes (BB, HBP, 1B, 2B, 3B, HR)
    to out outcomes (K, out_ground, out_fly, out_line), proportionally.

    A shift of 0.010 reduces expected runs from ~9.3 to ~8.65 per game.
    """
    if shift <= 0:
        return dist
    adjusted = dist.copy()
    offense_total = sum(adjusted[i] for i in _OFFENSE_IDX)
    out_total = sum(adjusted[i] for i in _OUT_IDX)
    if offense_total <= 0 or out_total <= 0:
        return dist
    for i in _OFFENSE_IDX:
        adjusted[i] -= shift * (adjusted[i] / offense_total)
    for i in _OUT_IDX:
        adjusted[i] += shift * (adjusted[i] / out_total)
    adjusted = np.maximum(adjusted, 1e-8)
    adjusted /= adjusted.sum()
    return adjusted


def _apply_park_weather(dist: np.ndarray, park_factor: float,
                        weather: dict | None = None) -> np.ndarray:
    """Scale outcome distribution by park factor and weather conditions.

    park_factor: 1.0 = average park, >1.0 = hitter-friendly (Coors ~1.15),
                 <1.0 = pitcher-friendly (Oracle ~0.85).
    """
    if park_factor == 1.0 and weather is None:
        return dist

    adjusted = dist.copy()
    # Park: HR is most park-sensitive, singles least
    hr_mult = park_factor ** 1.5
    hit_mult = park_factor ** 0.7

    # Weather adjustments
    if weather:
        temp = weather.get("temperature", 72)
        if temp and temp > 0:
            hr_mult *= 1.0 + (temp - 72) * 0.003
        if weather.get("wind_out"):
            hr_mult *= 1.08
        elif weather.get("wind_in"):
            hr_mult *= 0.92

    adjusted[_HR_IDX] *= hr_mult
    adjusted[_1B_IDX] *= hit_mult
    adjusted[_2B_IDX] *= hit_mult * park_factor ** 0.3
    adjusted[_3B_IDX] *= hit_mult * park_factor ** 0.5
    # BB/HBP unaffected by park
    adjusted = np.maximum(adjusted, 1e-8)
    adjusted /= adjusted.sum()
    return adjusted


def _apply_first_inning_adj(dist: np.ndarray) -> np.ndarray:
    """Apply first-inning adjustment factors to an outcome distribution.

    Multiplies each outcome probability by its empirical first-inning ratio,
    then renormalizes. This shifts the distribution to reflect the higher K,
    BB, HR rates observed in the first inning.
    """
    adjusted = dist.copy()
    for i, outcome in enumerate(OUTCOME_ORDER):
        adjusted[i] *= _FIRST_INNING_ADJ.get(outcome, 1.0)
    adjusted = np.maximum(adjusted, 1e-8)
    adjusted /= adjusted.sum()
    return adjusted


def precompute_all_distributions(
    home_ctx: TeamContext,
    away_ctx: TeamContext,
    calibration_shift: float = 0.0,
    park_factor: float = 1.0,
    weather: dict | None = None,
) -> dict:
    """
    Pre-build outcome distributions for all hitter/pitcher combinations.

    Uses the per-hitter distributions already computed in TeamContext.
    Supports individual relievers: (side, pos, "rp", reliever_idx) keys.

    Returns dict with keys:
      (side, lineup_pos, pitcher_state) -> prob_array (normal)
      (side, lineup_pos, pitcher_state, "no_dp") -> prob_array (dp redistributed)
      For relievers: pitcher_state = "rp0", "rp1", etc.
    """
    dists = {}
    uniform = np.full(_N_OUTCOMES, 1.0 / _N_OUTCOMES)

    def _adjust(probs):
        """Apply calibration + park/weather adjustments."""
        p = probs
        if calibration_shift > 0:
            p = _calibrate_distribution(p, calibration_shift)
        if park_factor != 1.0 or weather is not None:
            p = _apply_park_weather(p, park_factor, weather)
        return p

    for side, ctx in [("home", home_ctx), ("away", away_ctx)]:
        for pos in range(len(ctx.lineup)):
            # SP distributions
            sp_probs = ctx.sp_outcome_dists[pos] if pos < len(ctx.sp_outcome_dists) else uniform
            sp_probs = _adjust(sp_probs)
            dists[(side, pos, "sp")] = sp_probs
            dists[(side, pos, "sp", "no_dp")] = _make_no_dp_dist(sp_probs)

            # First-inning variant of SP distributions
            inn1_probs = _apply_first_inning_adj(sp_probs)
            dists[(side, pos, "sp", "inn1")] = inn1_probs
            dists[(side, pos, "sp", "inn1", "no_dp")] = _make_no_dp_dist(inn1_probs)

            # Individual reliever distributions
            for ri, reliever in enumerate(ctx.relievers):
                rp_key = f"rp{ri}"
                rp_probs = reliever.outcome_dists[pos] if pos < len(reliever.outcome_dists) else uniform
                rp_probs = _adjust(rp_probs)
                dists[(side, pos, rp_key)] = rp_probs
                dists[(side, pos, rp_key, "no_dp")] = _make_no_dp_dist(rp_probs)

            # Fallback composite bullpen
            bp_probs = ctx.bp_outcome_dists[pos] if pos < len(ctx.bp_outcome_dists) else uniform
            bp_probs = _adjust(bp_probs)
            dists[(side, pos, "bp")] = bp_probs
            dists[(side, pos, "bp", "no_dp")] = _make_no_dp_dist(bp_probs)
    return dists


# ── Base Running ──────────────────────────────────────────

def apply_transition(
    outcome: str,
    bases: tuple[bool, bool, bool],
    outs: int,
    transition_matrix: dict,
    rng: np.random.Generator,
) -> tuple[tuple[bool, bool, bool], int, int]:
    """
    Apply base-running transition for a PA outcome.

    Returns (new_bases, runs_scored, outs_added).
    """
    outs_added = _OUTS_ADDED.get(outcome, 0)

    key = (outcome, bases, outs)
    dist = transition_matrix.get(key)
    if dist is None:
        dist = _deterministic_transition(outcome, bases, outs)

    if len(dist) == 1:
        new_bases, runs, _ = dist[0]
    else:
        # Use uniform random for weighted selection (avoids numpy overhead for small lists)
        r = rng.random()
        cumsum = 0.0
        new_bases, runs, _ = dist[-1]  # fallback to last
        for nb, rn, p in dist:
            cumsum += p
            if r < cumsum:
                new_bases, runs = nb, rn
                break

    return new_bases, runs, outs_added


# ── Simulation Core ───────────────────────────────────────

# Pre-computed pitch count estimates per outcome for real pitch tracking
_PITCHES_PER_OUTCOME = np.array(
    [_PITCHES_PER_PA_BY_OUTCOME[o] for o in OUTCOME_ORDER], dtype=np.float64,
)


def _sample_outcome_python(probs, r):
    """Sample an outcome index from probability array using cumulative sum."""
    cumsum = 0.0
    n = probs.shape[0]
    for i in range(n):
        cumsum += probs[i]
        if r < cumsum:
            return i
    return n - 1


# Lazy numba JIT -- compiled on first use to avoid torch+numba import conflict
_sample_outcome_jit = None


def _sample_outcome(probs, r):
    global _sample_outcome_jit
    if _sample_outcome_jit is None:
        try:
            import numba as nb
            _sample_outcome_jit = nb.njit(_sample_outcome_python)
        except ImportError:
            _sample_outcome_jit = _sample_outcome_python
    return _sample_outcome_jit(probs, r)


def simulate_plate_appearance(
    state: GameState,
    batting_side: str,
    pitching_side: str,
    dists: dict,
    transition_matrix: dict,
    rng: np.random.Generator,
) -> tuple[str, int, tuple[bool, bool, bool], int, float]:
    """
    Simulate a single plate appearance.

    Returns (outcome, runs_scored, new_bases, outs_added, est_pitches).
    """
    if batting_side == "home":
        pos = state.home_lineup_pos
        pitcher_state = state.away_pitcher
    else:
        pos = state.away_lineup_pos
        pitcher_state = state.home_pitcher

    # Use first-inning adjusted distributions in inning 1 (SP only)
    use_inn1 = state.inning == 1 and pitcher_state == "sp"

    # Use pre-computed dp-adjusted distribution when dp is impossible
    if not state.bases[0] or state.outs >= 2:
        if use_inn1:
            probs = dists.get((batting_side, pos, pitcher_state, "inn1", "no_dp"))
        else:
            probs = dists.get((batting_side, pos, pitcher_state, "no_dp"))
    else:
        if use_inn1:
            probs = dists.get((batting_side, pos, pitcher_state, "inn1"))
        else:
            probs = dists.get((batting_side, pos, pitcher_state))

    if probs is None:
        probs = dists.get((batting_side, pos, pitcher_state))
    if probs is None:
        probs = np.full(_N_OUTCOMES, 1.0 / _N_OUTCOMES)

    outcome_idx = _sample_outcome(probs, rng.random())
    outcome = OUTCOME_ORDER[outcome_idx]

    # Per-outcome pitch count estimate (K~4.8, BB~5.6, contact~3.3-3.6)
    est_pitches = _PITCHES_PER_OUTCOME[outcome_idx]

    new_bases, runs, outs_added = apply_transition(
        outcome, state.bases, state.outs, transition_matrix, rng,
    )

    return outcome, runs, new_bases, outs_added, est_pitches


def simulate_half_inning(
    state: GameState,
    batting_side: str,
    pitching_side: str,
    dists: dict,
    transition_matrix: dict,
    config: SimConfig,
    rng: np.random.Generator,
) -> int:
    """
    Simulate a half-inning. Modifies state in-place.
    Returns runs scored in this half-inning.
    """
    runs_total = 0

    for _ in range(config.max_pa_per_half_inning):
        if state.outs >= 3:
            break

        outcome, runs, new_bases, outs_added, est_pitches = simulate_plate_appearance(
            state, batting_side, pitching_side, dists, transition_matrix, rng,
        )

        state.bases = new_bases
        state.outs = min(state.outs + outs_added, 3)
        runs_total += runs

        # Update score
        if batting_side == "home":
            state.home_score += runs
        else:
            state.away_score += runs

        # Advance lineup position
        if batting_side == "home":
            state.home_lineup_pos = (state.home_lineup_pos + 1) % 9
        else:
            state.away_lineup_pos = (state.away_lineup_pos + 1) % 9

        # Track pitcher pitch count using per-outcome estimates
        if pitching_side == "home":
            if state.home_pitcher == "sp":
                state.home_sp_pitches += est_pitches
            else:
                state.home_rp_pitches += est_pitches
        else:
            if state.away_pitcher == "sp":
                state.away_sp_pitches += est_pitches
            else:
                state.away_rp_pitches += est_pitches

        # Walk-off check: bottom of 9th+ and home team takes the lead
        if (batting_side == "home" and state.inning >= 9
                and state.home_score > state.away_score):
            break

    return runs_total


def _advance_reliever(state: GameState, side: str, ctx: TeamContext,
                      rng: np.random.Generator, config: SimConfig) -> None:
    """Advance to the next reliever for a team. Modifies state in-place."""
    if side == "home":
        ri = state.home_reliever_idx
        if ri < len(ctx.relievers):
            state.home_pitcher = f"rp{ri}"
            state.home_reliever_idx = ri + 1
        else:
            state.home_pitcher = "bp"  # exhausted relievers, fall back to composite
    else:
        ri = state.away_reliever_idx
        if ri < len(ctx.relievers):
            state.away_pitcher = f"rp{ri}"
            state.away_reliever_idx = ri + 1
        else:
            state.away_pitcher = "bp"


def simulate_game(
    home_ctx: TeamContext,
    away_ctx: TeamContext,
    initial_state: GameState,
    dists: dict,
    transition_matrix: dict,
    config: SimConfig,
    rng: np.random.Generator,
    track_first_inning: bool = False,
) -> tuple[int, int] | tuple[int, int, int, int]:
    """
    Simulate a complete game from the given state.

    Returns (home_runs, away_runs) by default.
    If track_first_inning=True, returns (home_runs, away_runs, away_1st_inn_runs, home_1st_inn_runs).
    """
    state = GameState(
        inning=initial_state.inning,
        top_bottom=initial_state.top_bottom,
        outs=initial_state.outs,
        bases=initial_state.bases,
        home_score=initial_state.home_score,
        away_score=initial_state.away_score,
        home_lineup_pos=initial_state.home_lineup_pos,
        away_lineup_pos=initial_state.away_lineup_pos,
        home_pitcher=initial_state.home_pitcher,
        away_pitcher=initial_state.away_pitcher,
        home_sp_pitches=initial_state.home_sp_pitches,
        away_sp_pitches=initial_state.away_sp_pitches,
        home_rp_pitches=initial_state.home_rp_pitches,
        away_rp_pitches=initial_state.away_rp_pitches,
        home_reliever_idx=initial_state.home_reliever_idx,
        away_reliever_idx=initial_state.away_reliever_idx,
    )

    max_innings = 15  # safety limit

    # First-inning tracking (initialized to 0; set properly during inning 1)
    away_1st = 0
    home_1st = 0

    # SP pull thresholds for this game (sampled once per game)
    home_sp_limit = rng.normal(config.sp_pitch_limit_mean, config.sp_pitch_limit_std)
    away_sp_limit = rng.normal(config.sp_pitch_limit_mean, config.sp_pitch_limit_std)

    # Reliever pitch limits (pitch counts are tracked in state.home_rp_pitches / away_rp_pitches)
    home_rp_limit = rng.normal(config.reliever_pitch_limit_mean, config.reliever_pitch_limit_std)
    away_rp_limit = rng.normal(config.reliever_pitch_limit_mean, config.reliever_pitch_limit_std)

    # If resuming mid-inning, finish the current half-inning first
    if state.outs > 0 or any(state.bases):
        batting = "away" if state.top_bottom == "Top" else "home"
        pitching = "home" if state.top_bottom == "Top" else "away"
        simulate_half_inning(
            state, batting, pitching, dists, transition_matrix, config, rng,
        )
        # Check walk-off
        if batting == "home" and state.inning >= 9 and state.home_score > state.away_score:
            return state.home_score, state.away_score
        # Move to next half
        if state.top_bottom == "Top":
            state.top_bottom = "Bot"
        else:
            state.top_bottom = "Top"
            state.inning += 1
        state.outs = 0
        state.bases = (False, False, False)

    while state.inning <= max_innings:
        # SP→reliever transition at inning boundaries
        if state.away_pitcher == "sp" and state.away_sp_pitches >= away_sp_limit:
            _advance_reliever(state, "away", away_ctx, rng, config)
            away_rp_limit = rng.normal(config.reliever_pitch_limit_mean,
                                       config.reliever_pitch_limit_std)
            state.away_rp_pitches = 0.0
        if state.home_pitcher == "sp" and state.home_sp_pitches >= home_sp_limit:
            _advance_reliever(state, "home", home_ctx, rng, config)
            home_rp_limit = rng.normal(config.reliever_pitch_limit_mean,
                                       config.reliever_pitch_limit_std)
            state.home_rp_pitches = 0.0

        # Reliever→reliever transition (check if current reliever is gassed)
        if state.away_pitcher.startswith("rp") and state.away_rp_pitches >= away_rp_limit:
            _advance_reliever(state, "away", away_ctx, rng, config)
            away_rp_limit = rng.normal(config.reliever_pitch_limit_mean,
                                       config.reliever_pitch_limit_std)
            state.away_rp_pitches = 0.0
        if state.home_pitcher.startswith("rp") and state.home_rp_pitches >= home_rp_limit:
            _advance_reliever(state, "home", home_ctx, rng, config)
            home_rp_limit = rng.normal(config.reliever_pitch_limit_mean,
                                       config.reliever_pitch_limit_std)
            state.home_rp_pitches = 0.0

        # Track first-inning scoring
        score_before_top1 = state.away_score if (track_first_inning and state.inning == 1) else None

        # Top of inning: away team bats
        if state.top_bottom == "Top":
            if state.inning >= 10:
                state.bases = (False, True, False)

            simulate_half_inning(
                state, "away", "home", dists, transition_matrix, config, rng,
            )

            if track_first_inning and state.inning == 1 and score_before_top1 is not None:
                away_1st = state.away_score - score_before_top1

            state.outs = 0
            state.bases = (False, False, False)
            state.top_bottom = "Bot"

            if state.inning >= 9 and state.home_score > state.away_score:
                if track_first_inning:
                    return state.home_score, state.away_score, away_1st, home_1st
                return state.home_score, state.away_score

        score_before_bot1 = state.home_score if (track_first_inning and state.inning == 1) else None

        # Bottom of inning: home team bats
        if state.top_bottom == "Bot":
            if state.inning >= 10:
                state.bases = (False, True, False)

            simulate_half_inning(
                state, "home", "away", dists, transition_matrix, config, rng,
            )

            if track_first_inning and state.inning == 1 and score_before_bot1 is not None:
                home_1st = state.home_score - score_before_bot1

            if state.inning >= 9 and state.home_score > state.away_score:
                if track_first_inning:
                    return state.home_score, state.away_score, away_1st, home_1st
                return state.home_score, state.away_score

            state.outs = 0
            state.bases = (False, False, False)
            state.top_bottom = "Top"
            state.inning += 1

        if state.inning > 9 and state.top_bottom == "Top" and state.home_score != state.away_score:
            if track_first_inning:
                return state.home_score, state.away_score, away_1st, home_1st
            return state.home_score, state.away_score

    if track_first_inning:
        return state.home_score, state.away_score, away_1st, home_1st
    return state.home_score, state.away_score


def monte_carlo_win_prob(
    home_ctx: TeamContext,
    away_ctx: TeamContext,
    initial_state: GameState,
    transition_matrix: dict,
    config: SimConfig = None,
    park_factor: float = 1.0,
    weather: dict | None = None,
) -> dict:
    """
    Run N Monte Carlo simulations to estimate win probability.

    Returns dict with home_wp, away_wp, runs distributions, etc.
    """
    if config is None:
        config = SimConfig()

    rng = np.random.default_rng(config.random_seed)

    # Pre-compute all outcome distributions with calibration + park/weather
    dists = precompute_all_distributions(
        home_ctx, away_ctx,
        calibration_shift=config.calibration_shift,
        park_factor=park_factor,
        weather=weather,
    )

    home_wins = 0
    away_wins = 0
    ties = 0
    home_runs_dist = Counter()
    away_runs_dist = Counter()
    total_runs_dist = Counter()
    run_diff_dist = Counter()
    n_extras = 0
    nrfi_count = 0

    # Track first inning only for pregame sims (starting from top of 1st)
    track_1st = (initial_state.inning == 1 and initial_state.top_bottom == "Top"
                 and initial_state.outs == 0 and not any(initial_state.bases))

    for _ in range(config.n_sims):
        if track_1st:
            hr, ar, a1, h1 = simulate_game(
                home_ctx, away_ctx, initial_state,
                dists, transition_matrix, config, rng,
                track_first_inning=True,
            )
            if a1 == 0 and h1 == 0:
                nrfi_count += 1
        else:
            hr, ar = simulate_game(
                home_ctx, away_ctx, initial_state,
                dists, transition_matrix, config, rng,
            )
        home_runs_dist[hr] += 1
        away_runs_dist[ar] += 1
        total_runs_dist[hr + ar] += 1
        run_diff_dist[hr - ar] += 1

        if hr > ar:
            home_wins += 1
        elif ar > hr:
            away_wins += 1
        else:
            ties += 1

    n = config.n_sims
    raw_home_wp = home_wins / n
    raw_away_wp = away_wins / n

    # Temperature scaling: amplify WP deviations from 0.50
    temp = config.wp_temperature
    if temp != 1.0 and raw_home_wp > 0 and raw_away_wp > 0:
        # Convert to log-odds, scale, convert back
        import math
        log_odds = math.log(raw_home_wp / raw_away_wp)
        scaled_odds = log_odds * temp
        home_wp = 1.0 / (1.0 + math.exp(-scaled_odds))
        away_wp = 1.0 - home_wp
    else:
        home_wp = raw_home_wp
        away_wp = raw_away_wp

    # Runs stats
    all_home = []
    all_away = []
    for runs, count in home_runs_dist.items():
        all_home.extend([runs] * count)
    for runs, count in away_runs_dist.items():
        all_away.extend([runs] * count)

    result = {
        "home_wp": home_wp,
        "away_wp": away_wp,
        "tie_pct": ties / n,
        "home_runs_mean": np.mean(all_home) if all_home else 0,
        "away_runs_mean": np.mean(all_away) if all_away else 0,
        "home_runs_std": np.std(all_home) if all_home else 0,
        "away_runs_std": np.std(all_away) if all_away else 0,
        "total_runs_mean": np.mean(all_home) + np.mean(all_away) if all_home else 0,
        "home_runs_dist": dict(home_runs_dist),
        "away_runs_dist": dict(away_runs_dist),
        "total_runs_dist": dict(total_runs_dist),
        "run_diff_dist": dict(run_diff_dist),
        "n_sims": n,
        "home_wins": home_wins,
        "away_wins": away_wins,
        "ties": ties,
    }
    if track_1st:
        result["nrfi_prob"] = nrfi_count / n
    return result


def compute_total_prob(mc_result: dict, line: float) -> float:
    """Compute P(over line) from MC simulation total runs distribution.

    Uses the joint total_runs_dist (not marginals) to preserve
    intra-game correlation between home and away runs.
    """
    total_dist = mc_result.get("total_runs_dist", {})
    n_sims = mc_result.get("n_sims", 1)
    over = sum(count for runs, count in total_dist.items() if runs > line)
    return over / n_sims


def compute_spread_prob(mc_result: dict, spread: float) -> float:
    """Compute P(home team covers spread) from MC simulation.

    spread is from home team's perspective:
      spread = -1.5 means home favored by 1.5 runs (home covers if home wins by 2+)
      spread = +1.5 means home is underdog (home covers if loss is by 1 or less, or wins)
    Home covers when: (home_runs - away_runs) + spread > 0
    """
    diff_dist = mc_result.get("run_diff_dist", {})
    n_sims = mc_result.get("n_sims", 1)
    covers = sum(count for diff, count in diff_dist.items() if diff + spread > 0)
    return covers / n_sims


# ── Data Loading ──────────────────────────────────────────

def ensure_batter_index(idx: dict) -> None:
    """Add a batter-level index to the pre-indexed xRV data if not already present."""
    if "batter" in idx:
        return
    xrv_df = idx.get("xrv")
    if xrv_df is None:
        idx["batter"] = {}
        return
    idx["batter"] = {bid: grp for bid, grp in xrv_df.groupby("batter")}


@functools.lru_cache(maxsize=512)
def _fetch_milb_outcome_rates(
    player_id: int,
    season: int = 2025,
) -> dict[str, float] | None:
    """Fetch minor league stats from MLB Stats API and convert to outcome rates.

    Tries AAA first, then AA, A+, A. Returns None if no MiLB data found.
    """
    for sport_id in [11, 12, 13, 14]:  # AAA, AA, A+, A
        try:
            url = (f"{MLB_API}/people/{player_id}/stats"
                   f"?stats=season&group=hitting&gameType=R&season={season}&sportId={sport_id}")
            resp = httpx.get(url, timeout=10.0)
            if resp.status_code != 200:
                continue
            data = resp.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            if not splits:
                continue
            stat = splits[0].get("stat", {})
            pa = stat.get("plateAppearances", 0)
            if pa < _MIN_BATTER_PA:
                continue

            # Extract counting stats
            k = stat.get("strikeOuts", 0)
            bb = stat.get("baseOnBalls", 0)
            hbp = stat.get("hitByPitch", 0)
            hr = stat.get("homeRuns", 0)
            doubles = stat.get("doubles", 0)
            triples = stat.get("triples", 0)
            hits = stat.get("hits", 0)
            singles = hits - doubles - triples - hr
            gidp = stat.get("groundIntoDoublePlay", 0)
            go = stat.get("groundOuts", 0)
            ao = stat.get("airOuts", 0)

            # Estimate out types (API gives groundOuts + airOuts)
            total_outs = pa - k - bb - hbp - hits - gidp
            out_ground = max(go - gidp, 0)
            out_fly = int(ao * 0.75)  # ~75% fly outs, 25% line outs
            out_line = max(ao - out_fly, 0)

            raw_rates = {
                "K": k / pa, "BB": bb / pa, "HBP": hbp / pa,
                "1B": max(singles, 0) / pa, "2B": doubles / pa, "3B": triples / pa,
                "HR": hr / pa, "dp": gidp / pa,
                "out_ground": out_ground / pa, "out_fly": out_fly / pa,
                "out_line": out_line / pa,
            }

            # Apply level adjustment
            adj = _MILB_LEVEL_ADJUSTMENTS.get(sport_id, {})
            adjusted = {o: raw_rates[o] * adj.get(o, 1.0) for o in OUTCOME_ORDER}

            # Normalize
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {o: v / total for o, v in adjusted.items()}
            return adjusted

        except Exception:
            continue
    return None


def _compute_batter_outcome_rates(
    idx: dict,
    batter_id: int,
    game_date: str,
    n_pa: int = 500,
    season: int | None = None,
    league_rates: dict | None = None,
    vs_hand: str | None = None,
) -> dict[str, float] | None:
    """Compute a batter's empirical PA outcome rates with Bayesian shrinkage.

    If vs_hand is provided ("L" or "R"), filters to PAs against that pitcher hand.
    Falls back to unfiltered rates with extra shrinkage if filtered sample < 15 PAs.

    Fallback chain: MLB stats (>= 15 PAs, shrunk) -> MiLB stats -> None (league avg).
    """
    if batter_id == 0:
        return None
    batter_df = idx.get("batter", {}).get(batter_id)
    if batter_df is not None:
        # Try platoon-filtered rates first
        if vs_hand is not None:
            filtered_df = batter_df[batter_df["p_throws"] == vs_hand]
            result = compute_outcome_rates(filtered_df, game_date, n_pa, min_pa=15)
            if result is not None:
                rates, actual_pa = result
                if league_rates:
                    return _shrink_rates(rates, actual_pa, league_rates, shrinkage_pa=120)
                return rates
            # Filtered sample too small — fall back to unfiltered with extra shrinkage
            result = compute_outcome_rates(batter_df, game_date, n_pa, min_pa=15)
            if result is not None:
                rates, actual_pa = result
                if league_rates:
                    return _shrink_rates(rates, actual_pa, league_rates, shrinkage_pa=240)
                return rates
        else:
            result = compute_outcome_rates(batter_df, game_date, n_pa, min_pa=15)
            if result is not None:
                rates, actual_pa = result
                if league_rates:
                    return _shrink_rates(rates, actual_pa, league_rates, shrinkage_pa=120)
                return rates

    # Try MiLB stats
    if season is not None:
        milb_rates = _fetch_milb_outcome_rates(batter_id, season)
        if milb_rates is not None:
            return milb_rates

    return None


def _compute_pitcher_outcome_rates(
    idx: dict,
    pitcher_id: int,
    game_date: str,
    n_pa: int = 800,
    league_rates: dict | None = None,
    vs_hand: str | None = None,
) -> dict[str, float] | None:
    """Compute a pitcher's empirical PA outcome rates with Bayesian shrinkage.

    If vs_hand is provided ("L" or "R"), filters to PAs against that batter hand.
    Falls back to unfiltered rates with extra shrinkage if filtered sample < 30 PAs.
    """
    pitcher_df = idx["pitcher"].get(pitcher_id)
    if pitcher_df is None:
        return None
    # Try platoon-filtered rates first
    if vs_hand is not None:
        filtered_df = pitcher_df[pitcher_df["stand"] == vs_hand]
        result = compute_outcome_rates(filtered_df, game_date, n_pa, min_pa=30)
        if result is not None:
            rates, actual_pa = result
            if league_rates:
                return _shrink_rates(rates, actual_pa, league_rates, shrinkage_pa=200)
            return rates
        # Filtered sample too small — fall back to unfiltered with extra shrinkage
        result = compute_outcome_rates(pitcher_df, game_date, n_pa, min_pa=30)
        if result is None:
            return None
        rates, actual_pa = result
        if league_rates:
            return _shrink_rates(rates, actual_pa, league_rates, shrinkage_pa=400)
        return rates
    else:
        result = compute_outcome_rates(pitcher_df, game_date, n_pa, min_pa=30)
        if result is None:
            return None
        rates, actual_pa = result
        if league_rates:
            return _shrink_rates(rates, actual_pa, league_rates, shrinkage_pa=200)
        return rates


def _compute_pitcher_recent_form(
    idx: dict,
    pitcher_id: int,
    game_date: str,
    season_rates: dict[str, float] | None,
    n_recent_pa: int = 80,
    form_weight: float = 0.20,
) -> dict[str, float] | None:
    """Blend pitcher's recent form into their season-long rates.

    Computes outcome rates from the pitcher's most recent ~80 PAs (roughly
    last 3 starts) and blends toward/away from season rates.

    Returns adjusted rates, or None if insufficient data.
    """
    if season_rates is None:
        return None
    pitcher_df = idx["pitcher"].get(pitcher_id)
    if pitcher_df is None:
        return season_rates

    gd = pd.Timestamp(game_date)
    before = pitcher_df[pitcher_df["game_date"] < gd]
    # Filter to completed plate appearances (rows with an event)
    before_pa = before[before["events"].notna()]
    if len(before_pa) < 30:
        return season_rates

    # Most recent n_recent_pa completed plate appearances
    recent = before_pa.tail(n_recent_pa)
    if len(recent) < 20:
        return season_rates

    # Compute recent rates
    total = len(recent)
    recent_rates = {}
    for o in OUTCOME_ORDER:
        if o == "dp":
            recent_rates[o] = ((recent["events"] == "double_play") |
                               (recent["events"] == "grounded_into_double_play")).sum() / total
        elif o == "K":
            recent_rates[o] = (recent["events"] == "strikeout").sum() / total
        elif o == "BB":
            recent_rates[o] = (recent["events"] == "walk").sum() / total
        elif o == "HBP":
            recent_rates[o] = (recent["events"] == "hit_by_pitch").sum() / total
        elif o == "1B":
            recent_rates[o] = (recent["events"] == "single").sum() / total
        elif o == "2B":
            recent_rates[o] = (recent["events"] == "double").sum() / total
        elif o == "3B":
            recent_rates[o] = (recent["events"] == "triple").sum() / total
        elif o == "HR":
            recent_rates[o] = (recent["events"] == "home_run").sum() / total
        else:
            # out_ground, out_fly, out_line — skip, will be remainder
            recent_rates[o] = 0.0

    # Ground/fly/line outs = 1 - sum of specific outcomes
    specific_sum = sum(recent_rates[o] for o in ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "dp"])
    remaining = max(0, 1.0 - specific_sum)
    # Distribute remaining across out types using season proportions
    out_season = sum(season_rates.get(o, 0) for o in ["out_ground", "out_fly", "out_line"])
    if out_season > 0:
        for o in ["out_ground", "out_fly", "out_line"]:
            recent_rates[o] = remaining * (season_rates.get(o, 0) / out_season)
    else:
        for o in ["out_ground", "out_fly", "out_line"]:
            recent_rates[o] = remaining / 3

    # Blend: season + form_weight * (recent - season)
    # Scale form_weight by sample size: more PAs → more trust
    effective_weight = form_weight * min(1.0, len(recent) / n_recent_pa)
    adjusted = {}
    for o in OUTCOME_ORDER:
        s = season_rates.get(o, 0.0)
        r = recent_rates.get(o, 0.0)
        adjusted[o] = s + effective_weight * (r - s)

    # Normalize
    total_adj = sum(adjusted.values())
    if total_adj > 0:
        adjusted = {o: v / total_adj for o, v in adjusted.items()}
    return adjusted


def _compute_bp_outcome_rates(
    idx: dict,
    team: str,
    game_date: str,
    lookback_days: int = 30,
) -> dict[str, float] | None:
    """Compute a team's bullpen composite outcome rates."""
    team_pitches = idx["pitching_team"].get(team)
    if team_pitches is None:
        return None

    before = _get_before(team_pitches, game_date)
    cutoff = np.datetime64(pd.Timestamp(game_date) - pd.Timedelta(days=lookback_days))
    recent = before[before["game_date"] >= cutoff]

    if len(recent) == 0:
        return None

    # Exclude starters
    starters = recent.groupby("game_pk").first()["pitcher"].values
    reliever_pitches = recent[~recent["pitcher"].isin(set(starters))]

    if len(reliever_pitches) == 0:
        return None

    result = compute_outcome_rates(reliever_pitches, game_date, n_pa=2000, min_pa=30)
    if result is None:
        return None
    rates, _ = result
    return rates


def _identify_top_relievers(
    idx: dict,
    team: str,
    game_date: str,
    lookback_days: int = 30,
    max_relievers: int = 7,
) -> list[int]:
    """Identify top relievers by recent usage (pitch count), ordered by usage.

    Returns list of pitcher IDs, most-used first.
    """
    team_pitches = idx.get("pitching_team", {}).get(team)
    if team_pitches is None:
        return []

    before = _get_before(team_pitches, game_date)
    cutoff = np.datetime64(pd.Timestamp(game_date) - pd.Timedelta(days=lookback_days))
    recent = before[before["game_date"] >= cutoff]

    if len(recent) == 0:
        return []

    # Exclude starters
    starters = set(recent.groupby("game_pk").first()["pitcher"].values)
    reliever_pitches = recent[~recent["pitcher"].isin(starters)]

    if len(reliever_pitches) == 0:
        return []

    # Top relievers by pitch count
    usage = reliever_pitches.groupby("pitcher").size().sort_values(ascending=False)
    return usage.head(max_relievers).index.tolist()


def _estimate_pitcher_pitches_per_pa(
    idx: dict,
    pitcher_id: int,
    game_date: str,
    n_pa: int = 200,
) -> float:
    """Estimate a pitcher's average pitches per PA from recent data."""
    pitcher_df = idx.get("pitcher", {}).get(pitcher_id)
    if pitcher_df is None:
        return 3.9

    before = _get_before(pitcher_df, game_date)
    if len(before) == 0:
        return 3.9

    recent = before.iloc[-2000:]  # last 2000 pitches max
    pa_count = recent["events"].notna().sum()
    if pa_count < 20:
        return 3.9

    return len(recent) / pa_count


def _compute_pitcher_arsenal_and_mix(
    idx: dict,
    pitcher_id: int,
    game_date: str,
    arsenal_stats: dict | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute standardized arsenal features and pitch-type mix for a pitcher."""
    pitcher_df = idx.get("pitcher", {}).get(pitcher_id)
    if pitcher_df is None:
        return None, None

    arsenal_raw = _compute_pitcher_arsenal_live(pitcher_df, game_date)
    if arsenal_raw is None:
        return None, None

    arsenal_z = _standardize_arsenal(arsenal_raw, arsenal_stats) if arsenal_stats else None

    # Compute pitch-type mix
    before = _get_before(pitcher_df, game_date)
    recent = before.iloc[-2000:] if len(before) > 2000 else before
    pt_counts = recent["pitch_type"].value_counts()
    n = len(recent)
    mix = np.zeros(MO_N_PTYPES)
    for pt, count in pt_counts.items():
        if pt in MO_PTYPE_MAP:
            mix[MO_PTYPE_MAP[pt]] = count / n

    return arsenal_z, mix


def _predict_matchup_probs_for_lineup(
    mo_models: dict,
    arsenal_z: np.ndarray,
    pitch_mix: np.ndarray,
    lineup: list[tuple[int, str]],
) -> list[np.ndarray | None]:
    """Get multi-output model predictions for each hitter in a lineup vs a pitcher."""
    results = []
    for batter_id, bat_side in lineup[:9]:
        if batter_id == 0 or arsenal_z is None:
            results.append(None)
            continue
        hand = bat_side if bat_side in ("L", "R") else "R"
        model_art = mo_models.get(hand)
        if model_art is None:
            results.append(None)
            continue
        probs = predict_matchup_distribution(model_art, arsenal_z, pitch_mix, batter_id)
        results.append(probs)
    return results


def load_simulation_context(
    game: dict,
    target_date: str,
    lineup_data: dict,
    idx: dict,
    matchup_models: dict,
    base_rates: dict = None,
    mo_models: dict | None = None,
    similarity_model: dict | None = None,
    config: SimConfig | None = None,
) -> tuple[TeamContext, TeamContext]:
    """
    Build TeamContext objects for home and away teams.

    For each batter-pitcher matchup, builds a full outcome distribution from:
      1. Batter's empirical outcome rates (MLB or MiLB fallback)
      2. Pitcher's empirical outcome rates
      3. Log5 odds-ratio combination
      4. Multi-output matchup model direct predictions (blended with log5)
      5. Similarity-based matchup predictions (if available)
    """
    if config is None:
        config = SimConfig()

    home_team = game["home_team"]
    away_team = game["away_team"]
    home_sp = game.get("home_sp_id")
    away_sp = game.get("away_sp_id")

    home_lineup = lineup_data.get("home", [])
    away_lineup = lineup_data.get("away", [])

    while len(home_lineup) < 9:
        home_lineup.append((0, "R"))
    while len(away_lineup) < 9:
        away_lineup.append((0, "R"))

    gdate = str(target_date)
    season = int(gdate[:4])

    ensure_batter_index(idx)

    if base_rates is None:
        base_rates = {o: 1.0 / _N_OUTCOMES for o in OUTCOME_ORDER}

    # ── Compute platoon-specific league base rates ──
    xrv_df = idx.get("xrv")
    base_rates_vsL = base_rates
    base_rates_vsR = base_rates
    if xrv_df is not None and "p_throws" in xrv_df.columns:
        for hand, label in [("L", "vsL"), ("R", "vsR")]:
            hand_pa = xrv_df[
                (xrv_df["p_throws"] == hand) & xrv_df["events"].notna()
            ]
            if len(hand_pa) >= 100:
                outcomes = _classify_pa_events(hand_pa)
                outcomes = outcomes.dropna()
                total = len(outcomes)
                if total > 0:
                    counts = outcomes.value_counts()
                    hand_rates = {o: counts.get(o, 0) / total for o in OUTCOME_ORDER}
                    if label == "vsL":
                        base_rates_vsL = hand_rates
                    else:
                        base_rates_vsR = hand_rates

    def _get_pitcher_hand(pitcher_id) -> str:
        """Look up pitcher throwing hand from xRV data. Default 'R'."""
        if pitcher_id is None or (isinstance(pitcher_id, float) and pd.isna(pitcher_id)):
            return "R"
        p_df = idx.get("pitcher", {}).get(int(pitcher_id))
        if p_df is not None and "p_throws" in p_df.columns and len(p_df) > 0:
            return p_df["p_throws"].iloc[0]
        return "R"

    def _resolve_bat_hand(bat_hand: str, pitcher_hand: str) -> str:
        """Resolve switch hitters to the opposite of the pitcher hand."""
        if bat_hand == "S":
            return "L" if pitcher_hand == "R" else "R"
        return bat_hand

    def _base_rates_for_hand(pitcher_hand: str) -> dict:
        """Get platoon-specific base rates for a given pitcher hand."""
        return base_rates_vsL if pitcher_hand == "L" else base_rates_vsR

    # ── Load eval scores (P1/P2/P3 pitcher + H1/H2/H4 hitter) ──
    eval_scores = None
    eval_sens = config.eval_sensitivity
    if eval_sens > 0:
        # Try current season, fall back to prior
        eval_scores = _load_eval_scores(season)
        has_data = any(len(v) > 0 for v in eval_scores.values())
        if not has_data and season > 2017:
            eval_scores = _load_eval_scores(season - 1)
            has_data = any(len(v) > 0 for v in eval_scores.values())
        if not has_data:
            eval_scores = None

    # Get arsenal stats from multi-output model for standardization
    arsenal_stats = None
    if mo_models:
        for hand_art in mo_models.values():
            arsenal_stats = hand_art.get("arsenal_stats")
            if arsenal_stats:
                break

    # ── Compute pitcher outcome rates (with shrinkage, platoon-split) ──
    away_sp_hand = _get_pitcher_hand(away_sp)
    home_sp_hand = _get_pitcher_hand(home_sp)

    # Compute SP rates split by batter hand (vsL / vsR)
    away_sp_rates_by_hand = {"L": None, "R": None}
    home_sp_rates_by_hand = {"L": None, "R": None}
    # Also keep overall rates for backward compat / fallback
    away_sp_rates = None
    home_sp_rates = None
    if pd.notna(away_sp):
        for bh in ("L", "R"):
            away_sp_rates_by_hand[bh] = _compute_pitcher_outcome_rates(
                idx, int(away_sp), gdate,
                league_rates=_base_rates_for_hand(away_sp_hand), vs_hand=bh)
        away_sp_rates = _compute_pitcher_outcome_rates(idx, int(away_sp), gdate,
                                                        league_rates=base_rates)
    if pd.notna(home_sp):
        for bh in ("L", "R"):
            home_sp_rates_by_hand[bh] = _compute_pitcher_outcome_rates(
                idx, int(home_sp), gdate,
                league_rates=_base_rates_for_hand(home_sp_hand), vs_hand=bh)
        home_sp_rates = _compute_pitcher_outcome_rates(idx, int(home_sp), gdate,
                                                        league_rates=base_rates)

    # Pitcher recent form disabled — ablation showed it hurts O/U correlation.

    # Bullpen composite rates (fallback)
    home_bp_rates = _compute_bp_outcome_rates(idx, home_team, gdate)
    away_bp_rates = _compute_bp_outcome_rates(idx, away_team, gdate)

    # ── Multi-output model predictions for SP matchups ──
    home_vs_away_sp_probs = [None] * 9
    away_vs_home_sp_probs = [None] * 9

    away_sp_arsenal_z, away_sp_mix = None, None
    home_sp_arsenal_z, home_sp_mix = None, None

    if mo_models and pd.notna(away_sp):
        away_sp_arsenal_z, away_sp_mix = _compute_pitcher_arsenal_and_mix(
            idx, int(away_sp), gdate, arsenal_stats)
        if away_sp_arsenal_z is not None:
            home_vs_away_sp_probs = _predict_matchup_probs_for_lineup(
                mo_models, away_sp_arsenal_z, away_sp_mix, home_lineup)

    if mo_models and pd.notna(home_sp):
        home_sp_arsenal_z, home_sp_mix = _compute_pitcher_arsenal_and_mix(
            idx, int(home_sp), gdate, arsenal_stats)
        if home_sp_arsenal_z is not None:
            away_vs_home_sp_probs = _predict_matchup_probs_for_lineup(
                mo_models, home_sp_arsenal_z, home_sp_mix, away_lineup)

    # ── Similarity model predictions for SP matchups ──
    home_vs_away_sp_sim = [None] * 9
    away_vs_home_sp_sim = [None] * 9

    if similarity_model:
        from swing_similarity_matchup import predict_matchup as predict_sim_matchup
        if pd.notna(away_sp):
            for pos, (batter_id, bat_side) in enumerate(home_lineup[:9]):
                if batter_id != 0:
                    hand = bat_side if bat_side in ("L", "R") else "R"
                    home_vs_away_sp_sim[pos] = predict_sim_matchup(
                        similarity_model, batter_id, int(away_sp), hand)
        if pd.notna(home_sp):
            for pos, (batter_id, bat_side) in enumerate(away_lineup[:9]):
                if batter_id != 0:
                    hand = bat_side if bat_side in ("L", "R") else "R"
                    away_vs_home_sp_sim[pos] = predict_sim_matchup(
                        similarity_model, batter_id, int(home_sp), hand)

    # ── SP pitches-per-PA estimates ──
    home_sp_ppa = 3.9
    away_sp_ppa = 3.9
    if pd.notna(home_sp):
        home_sp_ppa = _estimate_pitcher_pitches_per_pa(idx, int(home_sp), gdate)
    if pd.notna(away_sp):
        away_sp_ppa = _estimate_pitcher_pitches_per_pa(idx, int(away_sp), gdate)

    # ── Build per-hitter outcome distributions ──
    model_weight = config.model_blend_weight
    hfa = config.home_advantage  # home-field advantage shift
    amplify = config.log5_amplify

    def _apply_hfa_shift(rates: dict | None, is_home_batting: bool) -> dict | None:
        """Apply home-field advantage to batter outcome rates.

        Home batters get a small boost to contact rates (1B, 2B, 3B, HR)
        at the expense of K/out rates. Away batters get the reverse.
        The magnitude is calibrated so two identical teams produce ~53.5%
        home win probability. Factor ≈ 1 + hfa for home, 1 - hfa for away.
        """
        if rates is None or hfa == 0:
            return rates
        factor = 1.0 + hfa if is_home_batting else 1.0 - hfa
        contact = {"1B", "2B", "3B", "HR", "BB", "HBP"}
        adjusted = {}
        for o, r in rates.items():
            if o in contact:
                adjusted[o] = r * factor
            else:
                adjusted[o] = r / factor
        # Re-normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {o: v / total for o, v in adjusted.items()}
        return adjusted

    def _build_dists_for_lineup(lineup, opposing_sp_id, opposing_sp_rates,
                                opposing_sp_rates_by_hand,
                                opposing_sp_hand,
                                opposing_bp_rates,
                                model_probs_vs_sp, sim_probs_vs_sp,
                                is_home_batting: bool):
        sp_dists = []
        bp_dists = []
        for pos, (batter_id, _hand) in enumerate(lineup[:9]):
            # Platoon splits disabled — ablation showed they add noise.
            # Use overall rates with overall base rates.
            batter_rates = _compute_batter_outcome_rates(
                idx, batter_id, gdate, season=season,
                league_rates=base_rates)
            batter_rates = _apply_hfa_shift(batter_rates, is_home_batting)

            sp_model_probs = model_probs_vs_sp[pos] if pos < len(model_probs_vs_sp) else None
            sp_sim_probs = sim_probs_vs_sp[pos] if pos < len(sim_probs_vs_sp) else None
            sp_dist = build_matchup_distribution(
                batter_rates, opposing_sp_rates, base_rates,
                model_probs=sp_model_probs,
                similarity_probs=sp_sim_probs,
                model_weight=model_weight,
                amplify=amplify,
            )

            # Apply eval-based adjustments (pitcher stuff + hitter quality)
            if eval_scores and eval_sens > 0 and opposing_sp_id:
                ev_shift = _compute_eval_adjustment(
                    int(opposing_sp_id), batter_id,
                    _hand, eval_scores, eval_sens)
                sp_dist = _apply_eval_shift(sp_dist, ev_shift)

            sp_dists.append(sp_dist)

            bp_dist = build_matchup_distribution(
                batter_rates, opposing_bp_rates, base_rates,
                amplify=amplify,
            )

            # Hitter-side eval adjustment for bullpen matchups (no specific pitcher)
            if eval_scores and eval_sens > 0:
                h_swing = eval_scores["hitter_swing"].get(batter_id, 0.0)
                h_contact = eval_scores["hitter_contact"].get(batter_id, 0.0)
                h_bip = eval_scores["hitter_bip"].get(batter_id, 0.0)
                h_z = (0.40 * (h_swing / _H1_STD) +
                       0.30 * (h_contact / _H2_STD) +
                       0.30 * ((h_bip - _H4_MEAN) / _H4_STD))
                bp_shift = max(-0.03, min(0.03, h_z * eval_sens))
                bp_dist = _apply_eval_shift(bp_dist, bp_shift)

            bp_dists.append(bp_dist)

        return sp_dists, bp_dists

    home_sp_dists, home_bp_dists = _build_dists_for_lineup(
        home_lineup, away_sp, away_sp_rates,
        away_sp_rates_by_hand, away_sp_hand,
        away_bp_rates,
        home_vs_away_sp_probs, home_vs_away_sp_sim,
        is_home_batting=True,
    )
    away_sp_dists, away_bp_dists = _build_dists_for_lineup(
        away_lineup, home_sp, home_sp_rates,
        home_sp_rates_by_hand, home_sp_hand,
        home_bp_rates,
        away_vs_home_sp_probs, away_vs_home_sp_sim,
        is_home_batting=False,
    )

    # ── Build individual reliever distributions ──
    def _build_relievers(team, opposing_lineup, opposing_bp_rates,
                         is_opposing_home: bool):
        reliever_ids = _identify_top_relievers(idx, team, gdate)
        relievers = []
        for rp_id in reliever_ids:
            rp_ppa = _estimate_pitcher_pitches_per_pa(idx, rp_id, gdate)

            # Get model predictions for this reliever vs lineup
            rp_arsenal_z, rp_mix = None, None
            if mo_models and arsenal_stats:
                rp_arsenal_z, rp_mix = _compute_pitcher_arsenal_and_mix(
                    idx, rp_id, gdate, arsenal_stats)

            rp_rates = _compute_pitcher_outcome_rates(
                idx, rp_id, gdate, league_rates=base_rates)

            rp_dists = []
            for pos, (batter_id, _hand) in enumerate(opposing_lineup[:9]):
                batter_rates = _compute_batter_outcome_rates(
                    idx, batter_id, gdate, season=season,
                    league_rates=base_rates)
                batter_rates = _apply_hfa_shift(batter_rates, is_opposing_home)

                # Model predictions for this reliever
                rp_model_probs = None
                if mo_models and rp_arsenal_z is not None:
                    hand = _hand if _hand in ("L", "R") else "R"
                    model_art = mo_models.get(hand)
                    if model_art:
                        rp_model_probs = predict_matchup_distribution(
                            model_art, rp_arsenal_z, rp_mix, batter_id)

                dist = build_matchup_distribution(
                    batter_rates, rp_rates if rp_rates else opposing_bp_rates,
                    base_rates, model_probs=rp_model_probs, model_weight=model_weight,
                    amplify=amplify,
                )

                # Apply eval-based adjustments for reliever matchups
                if eval_scores and eval_sens > 0:
                    ev_shift = _compute_eval_adjustment(
                        rp_id, batter_id, _hand, eval_scores, eval_sens)
                    dist = _apply_eval_shift(dist, ev_shift)

                rp_dists.append(dist)

            relievers.append(RelieverInfo(
                pitcher_id=rp_id,
                outcome_dists=rp_dists,
                pitches_per_pa=rp_ppa,
            ))
        return relievers

    # Home team's relievers face away lineup (away batters → not home)
    home_relievers = _build_relievers(home_team, away_lineup, home_bp_rates,
                                       is_opposing_home=False)
    # Away team's relievers face home lineup (home batters → home)
    away_relievers = _build_relievers(away_team, home_lineup, away_bp_rates,
                                       is_opposing_home=True)

    home_ctx = TeamContext(
        team=home_team,
        lineup=home_lineup[:9],
        sp_outcome_dists=home_sp_dists,
        sp_pitches_per_pa=home_sp_ppa,
        relievers=home_relievers,
        bp_outcome_dists=home_bp_dists,
    )
    away_ctx = TeamContext(
        team=away_team,
        lineup=away_lineup[:9],
        sp_outcome_dists=away_sp_dists,
        sp_pitches_per_pa=away_sp_ppa,
        relievers=away_relievers,
        bp_outcome_dists=away_bp_dists,
    )

    return home_ctx, away_ctx


# ── Live Game State ───────────────────────────────────────

def fetch_live_game_state(client: httpx.Client, game_pk: int) -> GameState:
    """
    Fetch current game state from MLB live feed API.

    Returns a GameState populated with current inning, outs, bases, score,
    and pitcher state.
    """
    resp = client.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")
    resp.raise_for_status()
    data = resp.json()

    live = data.get("liveData", {})
    linescore = live.get("linescore", {})

    inning = linescore.get("currentInning", 1)
    is_top = linescore.get("isTopInning", True)
    outs = linescore.get("outs", 0)

    home_runs = linescore.get("teams", {}).get("home", {}).get("runs", 0)
    away_runs = linescore.get("teams", {}).get("away", {}).get("runs", 0)

    offense = linescore.get("offense", {})
    on_1b = offense.get("first") is not None
    on_2b = offense.get("second") is not None
    on_3b = offense.get("third") is not None

    # Determine current pitcher state (SP vs BP)
    boxscore = live.get("boxscore", {})
    home_team_box = boxscore.get("teams", {}).get("home", {})
    away_team_box = boxscore.get("teams", {}).get("away", {})

    # Check if SP is still pitching, and map to actual reliever index
    home_pitcher = "sp"
    away_pitcher = "sp"
    home_reliever_idx = 0
    away_reliever_idx = 0
    home_sp_pitches = 0
    away_sp_pitches = 0
    home_rp_pitches = 0.0
    away_rp_pitches = 0.0

    for side, team_box in [("home", home_team_box), ("away", away_team_box)]:
        pitchers = team_box.get("pitchers", [])
        n_relievers_used = max(0, len(pitchers) - 1)  # first pitcher is SP
        if n_relievers_used > 0:
            # Map to reliever index: rp0, rp1, rp2, then bp
            ri = n_relievers_used - 1  # 0-indexed current reliever
            if ri <= 2:
                pitcher_state = f"rp{ri}"
                reliever_idx = ri + 1  # next reliever to advance to
            else:
                pitcher_state = "bp"
                reliever_idx = ri + 1
            if side == "home":
                home_pitcher = pitcher_state
                home_reliever_idx = reliever_idx
            else:
                away_pitcher = pitcher_state
                away_reliever_idx = reliever_idx

        # Estimate pitch counts from boxscore player stats
        players = team_box.get("players", {})
        if pitchers:
            sp_id = f"ID{pitchers[0]}"
            sp_stats = players.get(sp_id, {}).get("stats", {})
            pitching_stats = sp_stats.get("pitching", {})
            sp_pc = int(pitching_stats.get("numberOfPitches", 0))
            if side == "home":
                home_sp_pitches = sp_pc
            else:
                away_sp_pitches = sp_pc

            # Current reliever pitch count
            if n_relievers_used > 0:
                current_rp_id = f"ID{pitchers[-1]}"
                rp_stats = players.get(current_rp_id, {}).get("stats", {})
                rp_pitching = rp_stats.get("pitching", {})
                rp_pc = float(rp_pitching.get("numberOfPitches", 0))
                if side == "home":
                    home_rp_pitches = rp_pc
                else:
                    away_rp_pitches = rp_pc

    # Determine lineup positions from batting order
    home_lineup_pos = 0
    away_lineup_pos = 0
    current_play = live.get("plays", {}).get("currentPlay", {})
    if current_play:
        matchup = current_play.get("matchup", {})
        batter_id = matchup.get("batter", {}).get("id")
        if batter_id:
            # Try to find batter in lineup
            for side, team_box in [("home", home_team_box), ("away", away_team_box)]:
                batting_order = team_box.get("battingOrder", [])
                for i, pid in enumerate(batting_order):
                    if pid == batter_id:
                        if side == "home":
                            home_lineup_pos = i % 9
                        else:
                            away_lineup_pos = i % 9

    return GameState(
        inning=inning,
        top_bottom="Top" if is_top else "Bot",
        outs=outs,
        bases=(on_1b, on_2b, on_3b),
        home_score=home_runs,
        away_score=away_runs,
        home_lineup_pos=home_lineup_pos,
        away_lineup_pos=away_lineup_pos,
        home_pitcher=home_pitcher,
        away_pitcher=away_pitcher,
        home_sp_pitches=home_sp_pitches,
        away_sp_pitches=away_sp_pitches,
        home_rp_pitches=home_rp_pitches,
        away_rp_pitches=away_rp_pitches,
        home_reliever_idx=home_reliever_idx,
        away_reliever_idx=away_reliever_idx,
    )


# ── Backtest ──────────────────────────────────────────────

def run_backtest(
    season: int,
    base_rates: dict,
    transition_matrix: dict,
    config: SimConfig,
):
    """
    Run the simulator on all games in a season and compare to actual outcomes.
    """
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    from feature_engineering import OPENING_DAY

    games_path = DATA_DIR / "games" / f"games_{season}.parquet"
    if not games_path.exists():
        print(f"Games file not found: {games_path}")
        return

    games_df = pd.read_parquet(games_path)
    games_df = games_df[
        (games_df["home_win"].notna()) &
        (games_df["home_sp_id"].notna()) &
        (games_df["away_sp_id"].notna())
    ].copy()

    print(f"  {len(games_df)} games for backtest")

    # Load xRV data and models
    xrv_frames = []
    for yr in range(season - 1, season + 1):
        path = XRV_DIR / f"statcast_xrv_{yr}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "game_type" in df.columns:
                df = df[df["game_type"] == "R"]
            xrv_frames.append(df)
    if not xrv_frames:
        print("No xRV data found")
        return
    xrv = pd.concat(xrv_frames, ignore_index=True)
    idx = _preindex_xrv(xrv)

    model_year = season - 1
    matchup_models = _load_hand_models("matchup_model", model_year)
    if not matchup_models:
        matchup_models = _load_hand_models("matchup_model", season)

    # Load multi-output models
    mo_models = load_multi_output_models(model_year)
    if not mo_models:
        mo_models = load_multi_output_models(season)
    if not mo_models:
        print("  No multi-output models found; using log5 only.")

    # Load lineups
    lineups_path = DATA_DIR / "games" / f"lineups_{season}.json"
    lineups = {}
    if lineups_path.exists():
        import json
        with open(lineups_path) as f:
            raw = json.load(f)
        for gpk, lu in raw.items():
            home_lu = [(p["player_id"], p.get("bat_side", "R")) for p in lu.get("home", [])]
            away_lu = [(p["player_id"], p.get("bat_side", "R")) for p in lu.get("away", [])]
            lineups[int(gpk)] = {"home": home_lu, "away": away_lu}

    actuals = []
    preds = []
    n_skip = 0

    for _, row in games_df.iterrows():
        game = row.to_dict()
        gpk = game["game_pk"]
        lu = lineups.get(gpk)
        if lu is None or not lu.get("home") or not lu.get("away"):
            n_skip += 1
            continue

        try:
            home_ctx, away_ctx = load_simulation_context(
                game, str(game["game_date"]), lu, idx, matchup_models,
                base_rates, mo_models=mo_models, config=config,
            )
            result = monte_carlo_win_prob(
                home_ctx, away_ctx, GameState(),
                transition_matrix, config,
            )
            preds.append(result["home_wp"])
            actuals.append(int(game["home_win"]))
        except Exception as e:
            n_skip += 1
            print(f"  SKIP game {gpk}: {type(e).__name__}: {e}")
            continue

    if not preds:
        print("No predictions generated")
        return

    preds = np.array(preds)
    actuals = np.array(actuals)

    preds_clipped = np.clip(preds, 0.01, 0.99)

    ll = log_loss(actuals, preds_clipped)
    bs = brier_score_loss(actuals, preds_clipped)
    auc = roc_auc_score(actuals, preds)
    acc = ((preds > 0.5) == actuals).mean()

    print(f"\n{'='*60}")
    print(f"Backtest Results: {season}")
    print(f"{'='*60}")
    print(f"  Games: {len(preds)} ({n_skip} skipped)")
    print(f"  Log loss:    {ll:.4f}")
    print(f"  Brier score: {bs:.4f}")
    print(f"  AUC:         {auc:.4f}")
    print(f"  Accuracy:    {acc:.1%}")
    print(f"  Mean pred:   {preds.mean():.4f}")
    print(f"  Home win%:   {actuals.mean():.4f}")

    print(f"\n  Calibration:")
    for lo in np.arange(0.3, 0.75, 0.05):
        hi = lo + 0.05
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() >= 10:
            print(f"    [{lo:.2f}, {hi:.2f}): pred={preds[mask].mean():.3f}  "
                  f"actual={actuals[mask].mean():.3f}  n={mask.sum()}")


# ── CLI ───────────────────────────────────────────────────

def predict_date(
    target_date: str,
    base_rates: dict,
    transition_matrix: dict,
    config: SimConfig,
    game_pk_filter: int = None,
):
    """Run MC simulation for all games on a date."""
    client = httpx.Client(timeout=30.0)

    print(f"\nFetching games for {target_date}...")
    games = fetch_todays_games(client, target_date)
    if not games:
        print("No games found.")
        return

    if game_pk_filter:
        games = [g for g in games if g["game_pk"] == game_pk_filter]
        if not games:
            print(f"Game {game_pk_filter} not found.")
            return

    print(f"  {len(games)} games found")

    year = int(target_date[:4])
    xrv_frames = []
    for yr in range(year - 1, year + 1):
        path = XRV_DIR / f"statcast_xrv_{yr}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "game_type" in df.columns:
                df = df[df["game_type"] == "R"]
            xrv_frames.append(df)
            print(f"  Loaded xRV {yr}: {len(df):,} pitches")
    if not xrv_frames:
        print("No xRV data available.")
        return
    xrv = pd.concat(xrv_frames, ignore_index=True)
    idx = _preindex_xrv(xrv)

    model_year = year - 1
    matchup_models = _load_hand_models("matchup_model", model_year)
    if not matchup_models:
        matchup_models = _load_hand_models("matchup_model", year)

    mo_models = load_multi_output_models(model_year)
    if not mo_models:
        mo_models = load_multi_output_models(year)

    print(f"\n{'='*70}")
    print(f"{'Game':<35} {'Home WP':>8} {'Away WP':>8} {'Runs':>8} {'Status'}")
    print(f"{'='*70}")

    for game in games:
        status = game.get("status", "")
        is_live = status in ("Live", "In Progress")
        is_final = status == "Final"

        lu = fetch_lineup(client, game["game_pk"])
        if not lu.get("home") or not lu.get("away"):
            label = f"{game['away_team']:>3} @ {game['home_team']:<3}"
            print(f"  {label:<35} {'---':>8} {'---':>8} {'---':>8} No lineups")
            continue

        if is_live:
            try:
                initial_state = fetch_live_game_state(client, game["game_pk"])
            except Exception as e:
                print(f"  Could not fetch live state for {game['game_pk']}: {e}")
                initial_state = GameState()
        else:
            initial_state = GameState()

        try:
            home_ctx, away_ctx = load_simulation_context(
                game, target_date, lu, idx, matchup_models or {},
                base_rates, mo_models=mo_models, config=config,
            )
            result = monte_carlo_win_prob(
                home_ctx, away_ctx, initial_state,
                transition_matrix, config,
            )

            label = f"{game['away_team']:>3} @ {game['home_team']:<3}"
            sp_label = f"({game.get('away_sp_name', '?')} vs {game.get('home_sp_name', '?')})"

            status_str = ""
            if is_live:
                inn = initial_state.inning
                tb = "T" if initial_state.top_bottom == "Top" else "B"
                status_str = f"LIVE {tb}{inn} {initial_state.away_score}-{initial_state.home_score}"
            elif is_final:
                status_str = f"FINAL {game.get('away_score', '?')}-{game.get('home_score', '?')}"

            total_runs = result["home_runs_mean"] + result["away_runs_mean"]
            print(f"  {label} {sp_label:<25} "
                  f"{result['home_wp']:>7.1%} {result['away_wp']:>7.1%} "
                  f"{total_runs:>6.1f}  {status_str}")

        except Exception as e:
            label = f"{game['away_team']:>3} @ {game['home_team']:<3}"
            print(f"  {label:<35} {'ERR':>8} {'ERR':>8} {'---':>8} {e}")

    client.close()


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo MLB Game Simulator")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date (YYYY-MM-DD). Default: today")
    parser.add_argument("--game-pk", type=int, default=None,
                        help="Specific game ID")
    parser.add_argument("--backtest", type=int, default=None,
                        help="Run backtest for a season")
    parser.add_argument("--n-sims", type=int, default=10_000,
                        help="Number of simulations (default: 10000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--home", type=str, default=None,
                        help="Home team abbreviation (manual matchup)")
    parser.add_argument("--away", type=str, default=None,
                        help="Away team abbreviation (manual matchup)")
    parser.add_argument("--home-sp", type=int, default=None,
                        help="Home SP player ID (manual matchup)")
    parser.add_argument("--away-sp", type=int, default=None,
                        help="Away SP player ID (manual matchup)")
    args = parser.parse_args()

    config = SimConfig(
        n_sims=args.n_sims,
        random_seed=args.seed,
    )

    # Load simulation artifacts
    print("Loading simulation artifacts...")
    try:
        base_rates, transition_matrix = load_sim_artifacts()
    except FileNotFoundError as e:
        print(f"Simulation data not found: {e}")
        print("Run `python src/build_transition_matrix.py` first.")
        return

    if args.backtest:
        config.n_sims = min(config.n_sims, 1000)  # faster for backtest
        run_backtest(args.backtest, base_rates, transition_matrix, config)
        return

    target_date = args.date or date.today().isoformat()
    predict_date(target_date, base_rates, transition_matrix,
                 config, game_pk_filter=args.game_pk)


if __name__ == "__main__":
    main()
