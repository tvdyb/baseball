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
    _precompute_pitcher_bases,
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

# Pre-computed indices for hot-path performance
_DP_IDX = OUTCOME_ORDER.index("dp")
_OG_IDX = OUTCOME_ORDER.index("out_ground")
_N_OUTCOMES = len(OUTCOME_ORDER)

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
) -> dict[str, float] | None:
    """Compute empirical PA outcome rates from a player's recent PAs.

    Works for both batters (pass batter-filtered df) and pitchers
    (pass pitcher-filtered df).

    Returns dict of {outcome: rate} or None if insufficient data.
    """
    before = _get_before(df, game_date)
    # Get PA rows (where events is not null)
    pa = before[before["events"].notna()]
    pa = pa.iloc[-n_pa:] if len(pa) > n_pa else pa

    if len(pa) < _MIN_BATTER_PA:
        return None

    outcomes = _classify_pa_events(pa)
    outcomes = outcomes.dropna()
    total = len(outcomes)
    if total == 0:
        return None

    counts = outcomes.value_counts()
    rates = {o: counts.get(o, 0) / total for o in OUTCOME_ORDER}
    return rates


def log5_combine(
    batter_rates: dict[str, float],
    pitcher_rates: dict[str, float],
    league_rates: dict[str, float],
) -> dict[str, float]:
    """Combine batter and pitcher outcome rates using the log5 / odds-ratio method.

    P(outcome|B,P) ∝ (B_rate * P_rate) / L_rate

    This is the standard baseball approach (Bill James' log5 method):
    each player's rate is expressed as a deviation from league average,
    and the deviations combine multiplicatively.
    """
    combined = {}
    for o in OUTCOME_ORDER:
        b = max(batter_rates.get(o, 0.0), 1e-6)
        p = max(pitcher_rates.get(o, 0.0), 1e-6)
        lg = max(league_rates.get(o, 0.0), 1e-6)
        combined[o] = (b * p) / lg

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
    model_weight: float = 0.6,
) -> np.ndarray:
    """Build a full per-batter-pitcher outcome distribution.

    Combines:
      1. Batter's empirical outcome rates (or league avg if unknown)
      2. Pitcher's empirical outcome rates (or league avg if unknown)
      3. Log5 odds-ratio combination
      4. Blend with multi-output matchup model predictions (if available)

    Returns numpy probability array in OUTCOME_ORDER.
    """
    b_rates = batter_rates if batter_rates is not None else league_rates
    p_rates = pitcher_rates if pitcher_rates is not None else league_rates

    log5_probs = rates_to_probs(log5_combine(b_rates, p_rates, league_rates))

    if model_probs is None:
        return log5_probs

    # Blend model prediction with log5 baseline
    blended = model_weight * model_probs + (1.0 - model_weight) * log5_probs
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


def precompute_all_distributions(
    home_ctx: TeamContext,
    away_ctx: TeamContext,
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

    for side, ctx in [("home", home_ctx), ("away", away_ctx)]:
        for pos in range(len(ctx.lineup)):
            # SP distributions
            sp_probs = ctx.sp_outcome_dists[pos] if pos < len(ctx.sp_outcome_dists) else uniform
            dists[(side, pos, "sp")] = sp_probs
            dists[(side, pos, "sp", "no_dp")] = _make_no_dp_dist(sp_probs)

            # Individual reliever distributions
            for ri, reliever in enumerate(ctx.relievers):
                rp_key = f"rp{ri}"
                rp_probs = reliever.outcome_dists[pos] if pos < len(reliever.outcome_dists) else uniform
                dists[(side, pos, rp_key)] = rp_probs
                dists[(side, pos, rp_key, "no_dp")] = _make_no_dp_dist(rp_probs)

            # Fallback composite bullpen
            bp_probs = ctx.bp_outcome_dists[pos] if pos < len(ctx.bp_outcome_dists) else uniform
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

    # Use pre-computed dp-adjusted distribution when dp is impossible
    if not state.bases[0] or state.outs >= 2:
        probs = dists.get((batting_side, pos, pitcher_state, "no_dp"))
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
) -> tuple[int, int]:
    """
    Simulate a complete game from the given state.

    Returns (home_runs, away_runs).
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

        # Top of inning: away team bats
        if state.top_bottom == "Top":
            if state.inning >= 10:
                state.bases = (False, True, False)

            simulate_half_inning(
                state, "away", "home", dists, transition_matrix, config, rng,
            )

            state.outs = 0
            state.bases = (False, False, False)
            state.top_bottom = "Bot"

            if state.inning >= 9 and state.home_score > state.away_score:
                return state.home_score, state.away_score

        # Bottom of inning: home team bats
        if state.top_bottom == "Bot":
            if state.inning >= 10:
                state.bases = (False, True, False)

            simulate_half_inning(
                state, "home", "away", dists, transition_matrix, config, rng,
            )
            if state.inning >= 9 and state.home_score > state.away_score:
                return state.home_score, state.away_score

            state.outs = 0
            state.bases = (False, False, False)
            state.top_bottom = "Top"
            state.inning += 1

        if state.inning > 9 and state.top_bottom == "Top" and state.home_score != state.away_score:
            return state.home_score, state.away_score

    return state.home_score, state.away_score


def monte_carlo_win_prob(
    home_ctx: TeamContext,
    away_ctx: TeamContext,
    initial_state: GameState,
    transition_matrix: dict,
    config: SimConfig = None,
) -> dict:
    """
    Run N Monte Carlo simulations to estimate win probability.

    Returns dict with home_wp, away_wp, runs distributions, etc.
    """
    if config is None:
        config = SimConfig()

    rng = np.random.default_rng(config.random_seed)

    # Pre-compute all outcome distributions (already built per-hitter in TeamContext)
    dists = precompute_all_distributions(home_ctx, away_ctx)

    home_wins = 0
    away_wins = 0
    ties = 0
    home_runs_dist = Counter()
    away_runs_dist = Counter()
    n_extras = 0

    for _ in range(config.n_sims):
        hr, ar = simulate_game(
            home_ctx, away_ctx, initial_state,
            dists, transition_matrix, config, rng,
        )
        home_runs_dist[hr] += 1
        away_runs_dist[ar] += 1

        if hr > ar:
            home_wins += 1
        elif ar > hr:
            away_wins += 1
        else:
            ties += 1

    n = config.n_sims
    home_wp = home_wins / n
    away_wp = away_wins / n

    # Runs stats
    all_home = []
    all_away = []
    for runs, count in home_runs_dist.items():
        all_home.extend([runs] * count)
    for runs, count in away_runs_dist.items():
        all_away.extend([runs] * count)

    return {
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
        "n_sims": n,
        "home_wins": home_wins,
        "away_wins": away_wins,
        "ties": ties,
    }


# ── Data Loading ──────────────────────────────────────────

def _compute_per_hitter_matchup_xrv(
    matchup_models: dict,
    pitcher_df: pd.DataFrame,
    pitcher_id: int,
    game_date: str,
    lineup: list[tuple[int, str]],
    n_pitches: int = 2000,
) -> list[float]:
    """
    Compute per-hitter matchup xRV for a lineup vs a pitcher using
    the Bayesian hierarchical model.

    This captures the pitcher-hitter interaction that log5 can't:
    how well this hitter handles this pitcher's specific arsenal,
    pitch types, and movement profile.

    Returns list of floats (one per lineup slot). Zero for unknown matchups.
    """
    hitter_ids = [h[0] for h in lineup]
    hitter_hands = [h[1] for h in lineup]
    xrvs = [0.0] * len(lineup)

    for hand in ["L", "R"]:
        model = matchup_models.get(hand)
        if model is None:
            continue

        hand_indices = [i for i, (_, h) in enumerate(lineup) if h == hand]
        if not hand_indices:
            continue

        hand_hids = [hitter_ids[i] for i in hand_indices]
        hand_hands = [hitter_hands[i] for i in hand_indices]

        pitcher_bases = _precompute_pitcher_bases(
            model, pitcher_df, pitcher_id, game_date, n_pitches,
        )
        if not pitcher_bases:
            continue

        # Compute individual hitter xRVs from model
        map_est = model["map_estimate"]
        hitter_map = model["hitter_map"]
        hitter_effects = np.array(map_est["hitter_effect"])
        hitter_ptype_effects = np.array(map_est["hitter_ptype_effect"])

        # Population-average prediction (no hitter effect) for this pitcher
        for j, (hid, hhand) in enumerate(zip(hand_hids, hand_hands)):
            bases = pitcher_bases.get(hhand)
            if bases is None:
                continue
            pitch_base, ptype_idx = bases

            pop_avg = float(np.mean(pitch_base))

            if hid in hitter_map:
                h_idx = hitter_map[hid]
                preds = (
                    pitch_base
                    + hitter_effects[h_idx]
                    + hitter_ptype_effects[h_idx, ptype_idx]
                )
                full_pred = float(np.mean(preds))
                # The interaction is the deviation from population average
                # This isolates what the matchup model adds beyond individual rates
                xrvs[hand_indices[j]] = full_pred - pop_avg
            else:
                xrvs[hand_indices[j]] = 0.0

    return xrvs


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
) -> dict[str, float] | None:
    """Compute a batter's empirical PA outcome rates.

    Fallback chain: MLB stats (>= 50 PAs) -> MiLB stats -> None (league avg).
    """
    if batter_id == 0:
        return None
    batter_df = idx.get("batter", {}).get(batter_id)
    if batter_df is not None:
        rates = compute_outcome_rates(batter_df, game_date, n_pa)
        if rates is not None:
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
) -> dict[str, float] | None:
    """Compute a pitcher's empirical PA outcome rates (as pitcher) from recent history."""
    pitcher_df = idx["pitcher"].get(pitcher_id)
    if pitcher_df is None:
        return None
    return compute_outcome_rates(pitcher_df, game_date, n_pa)


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

    return compute_outcome_rates(reliever_pitches, game_date, n_pa=2000)


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
    config: SimConfig | None = None,
) -> tuple[TeamContext, TeamContext]:
    """
    Build TeamContext objects for home and away teams.

    For each batter-pitcher matchup, builds a full outcome distribution from:
      1. Batter's empirical outcome rates (MLB or MiLB fallback)
      2. Pitcher's empirical outcome rates
      3. Log5 odds-ratio combination
      4. Multi-output matchup model direct predictions (blended with log5)
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

    # Get arsenal stats from multi-output model for standardization
    arsenal_stats = None
    if mo_models:
        for hand_art in mo_models.values():
            arsenal_stats = hand_art.get("arsenal_stats")
            if arsenal_stats:
                break

    # ── Compute pitcher outcome rates ──
    away_sp_rates = None
    home_sp_rates = None
    if pd.notna(away_sp):
        away_sp_rates = _compute_pitcher_outcome_rates(idx, int(away_sp), gdate)
    if pd.notna(home_sp):
        home_sp_rates = _compute_pitcher_outcome_rates(idx, int(home_sp), gdate)

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

    # ── SP pitches-per-PA estimates ──
    home_sp_ppa = 3.9
    away_sp_ppa = 3.9
    if pd.notna(home_sp):
        home_sp_ppa = _estimate_pitcher_pitches_per_pa(idx, int(home_sp), gdate)
    if pd.notna(away_sp):
        away_sp_ppa = _estimate_pitcher_pitches_per_pa(idx, int(away_sp), gdate)

    # ── Build per-hitter outcome distributions ──
    model_weight = config.model_blend_weight

    def _build_dists_for_lineup(lineup, opposing_sp_rates, opposing_bp_rates,
                                model_probs_vs_sp):
        sp_dists = []
        bp_dists = []
        for pos, (batter_id, _hand) in enumerate(lineup[:9]):
            batter_rates = _compute_batter_outcome_rates(idx, batter_id, gdate,
                                                         season=season)

            sp_model_probs = model_probs_vs_sp[pos] if pos < len(model_probs_vs_sp) else None
            sp_dist = build_matchup_distribution(
                batter_rates, opposing_sp_rates, base_rates,
                model_probs=sp_model_probs, model_weight=model_weight,
            )
            sp_dists.append(sp_dist)

            bp_dist = build_matchup_distribution(
                batter_rates, opposing_bp_rates, base_rates,
            )
            bp_dists.append(bp_dist)

        return sp_dists, bp_dists

    home_sp_dists, home_bp_dists = _build_dists_for_lineup(
        home_lineup, away_sp_rates, away_bp_rates, home_vs_away_sp_probs,
    )
    away_sp_dists, away_bp_dists = _build_dists_for_lineup(
        away_lineup, home_sp_rates, home_bp_rates, away_vs_home_sp_probs,
    )

    # ── Build individual reliever distributions ──
    def _build_relievers(team, opposing_lineup, opposing_bp_rates):
        reliever_ids = _identify_top_relievers(idx, team, gdate)
        relievers = []
        for rp_id in reliever_ids:
            rp_rates = _compute_pitcher_outcome_rates(idx, rp_id, gdate)
            rp_ppa = _estimate_pitcher_pitches_per_pa(idx, rp_id, gdate)

            # Get model predictions for this reliever vs lineup
            rp_arsenal_z, rp_mix = None, None
            if mo_models and arsenal_stats:
                rp_arsenal_z, rp_mix = _compute_pitcher_arsenal_and_mix(
                    idx, rp_id, gdate, arsenal_stats)

            rp_dists = []
            for pos, (batter_id, _hand) in enumerate(opposing_lineup[:9]):
                batter_rates = _compute_batter_outcome_rates(idx, batter_id, gdate,
                                                             season=season)

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
                )
                rp_dists.append(dist)

            relievers.append(RelieverInfo(
                pitcher_id=rp_id,
                outcome_dists=rp_dists,
                pitches_per_pa=rp_ppa,
            ))
        return relievers

    # Home team's relievers face away lineup
    home_relievers = _build_relievers(home_team, away_lineup, home_bp_rates)
    # Away team's relievers face home lineup
    away_relievers = _build_relievers(away_team, home_lineup, away_bp_rates)

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
