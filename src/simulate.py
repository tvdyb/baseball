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
    compute_matchup_xrv,
    _load_hand_models,
    _sp_features_fast,
    _bp_features_fast,
    _get_before,
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


@dataclass
class TeamContext:
    """A team's offensive and pitching identity for simulation.

    Outcome distributions are per-hitter, per-pitcher-state numpy arrays
    in OUTCOME_ORDER, derived from batter/pitcher empirical rates combined
    via log5 and adjusted by the Bayesian matchup model.
    """
    team: str                                    # abbreviation
    lineup: list[tuple[int, str]]                # [(player_id, bat_side), ...] x9
    # Per-hitter outcome distributions: list of 9 numpy arrays, one per lineup slot
    sp_outcome_dists: list[np.ndarray] = field(default_factory=list)   # vs opposing SP
    bp_outcome_dists: list[np.ndarray] = field(default_factory=list)   # vs opposing bullpen


@dataclass
class SimConfig:
    """Simulation parameters."""
    n_sims: int = 10_000
    sp_pitch_limit_mean: int = 92
    sp_pitch_limit_std: int = 10
    pitches_per_pa: float = 3.9    # league avg pitches per PA
    random_seed: int | None = None


# ── Outcome Distribution ──────────────────────────────────
# Per-batter, per-pitcher outcome distributions derived from:
#   1. Batter's empirical PA outcome rates (recent history)
#   2. Pitcher's empirical PA outcome rates (recent history)
#   3. Log5 odds-ratio combination (standard baseball method)
#   4. Bayesian matchup model interaction adjustment

# Minimum PA thresholds before trusting individual rates
_MIN_BATTER_PA = 50
_MIN_PITCHER_PA = 100

# Clamps to prevent extreme outcome rates
_OUTCOME_CLAMPS = {"K": 0.50, "HR": 0.10, "BB": 0.25, "HBP": 0.05,
                   "3B": 0.03, "dp": 0.08}


def load_sim_artifacts() -> tuple[dict, dict, dict]:
    """Load pre-computed simulation artifacts."""
    with open(SIM_DIR / "league_base_rates.pkl", "rb") as f:
        base_rates = pickle.load(f)
    with open(SIM_DIR / "xrv_calibration.pkl", "rb") as f:
        calibration = pickle.load(f)
    with open(SIM_DIR / "transition_matrix.pkl", "rb") as f:
        transition_matrix = pickle.load(f)
    return base_rates, calibration, transition_matrix


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


def apply_matchup_adjustment(
    combined_rates: dict[str, float],
    matchup_xrv: float,
    calibration: dict,
) -> dict[str, float]:
    """Apply Bayesian matchup model interaction as a final adjustment.

    The matchup xRV captures pitcher-hitter specifics that log5 misses:
    how a particular hitter handles this pitcher's specific arsenal,
    pitch sequencing, and movement profile.

    Uses the fitted log-linear calibration:
      P(outcome) *= exp(alpha_outcome * xrv * scale)
    then renormalizes.
    """
    if np.isnan(matchup_xrv) or matchup_xrv == 0.0:
        return combined_rates

    alphas = calibration["alphas"]
    scale = calibration["scale"]

    adjusted = {}
    for o in OUTCOME_ORDER:
        alpha = alphas.get(o, 0.0)
        adjusted[o] = combined_rates[o] * np.exp(alpha * matchup_xrv * scale)

    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {o: v / total for o, v in adjusted.items()}
    return adjusted


def rates_to_probs(rates: dict[str, float]) -> np.ndarray:
    """Convert outcome rate dict to numpy prob array, applying clamps."""
    for o, cap in _OUTCOME_CLAMPS.items():
        if rates.get(o, 0) > cap:
            rates[o] = cap
    total = sum(rates.values())
    if total <= 0:
        return np.full(_N_OUTCOMES, 1.0 / _N_OUTCOMES)
    return np.array([rates.get(o, 0.0) / total for o in OUTCOME_ORDER])


def build_matchup_distribution(
    batter_rates: dict[str, float] | None,
    pitcher_rates: dict[str, float] | None,
    league_rates: dict[str, float],
    matchup_xrv: float,
    calibration: dict,
) -> np.ndarray:
    """Build a full per-batter-pitcher outcome distribution.

    Combines:
      1. Batter's empirical outcome rates (or league avg if unknown)
      2. Pitcher's empirical outcome rates (or league avg if unknown)
      3. Log5 odds-ratio combination
      4. Bayesian matchup model interaction adjustment

    Returns numpy probability array in OUTCOME_ORDER.
    """
    b_rates = batter_rates if batter_rates is not None else league_rates
    p_rates = pitcher_rates if pitcher_rates is not None else league_rates

    combined = log5_combine(b_rates, p_rates, league_rates)
    adjusted = apply_matchup_adjustment(combined, matchup_xrv, calibration)
    return rates_to_probs(adjusted)


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

    Returns dict with two keys per combo:
      (side, lineup_pos, pitcher_state) -> prob_array (normal)
      (side, lineup_pos, pitcher_state, "no_dp") -> prob_array (dp redistributed)
    """
    dists = {}
    for side, ctx in [("home", home_ctx), ("away", away_ctx)]:
        for pos in range(len(ctx.lineup)):
            for ps in ["sp", "bp"]:
                if ps == "sp":
                    dist_list = ctx.sp_outcome_dists
                else:
                    dist_list = ctx.bp_outcome_dists
                if pos < len(dist_list):
                    probs = dist_list[pos]
                else:
                    probs = np.full(_N_OUTCOMES, 1.0 / _N_OUTCOMES)
                dists[(side, pos, ps)] = probs
                dists[(side, pos, ps, "no_dp")] = _make_no_dp_dist(probs)
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

def simulate_plate_appearance(
    state: GameState,
    batting_side: str,
    pitching_side: str,
    dists: dict,
    transition_matrix: dict,
    rng: np.random.Generator,
) -> tuple[str, int, tuple[bool, bool, bool], int]:
    """
    Simulate a single plate appearance.

    Returns (outcome, runs_scored, new_bases, outs_added).
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

    # Weighted random selection using cumulative sum (faster than rng.choice for small arrays)
    r = rng.random()
    cumsum = 0.0
    outcome_idx = _N_OUTCOMES - 1  # fallback
    for i in range(_N_OUTCOMES):
        cumsum += probs[i]
        if r < cumsum:
            outcome_idx = i
            break
    outcome = OUTCOME_ORDER[outcome_idx]

    new_bases, runs, outs_added = apply_transition(
        outcome, state.bases, state.outs, transition_matrix, rng,
    )

    return outcome, runs, new_bases, outs_added


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
    max_pa = 25  # safety limit per half-inning

    for _ in range(max_pa):
        if state.outs >= 3:
            break

        outcome, runs, new_bases, outs_added = simulate_plate_appearance(
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

        # Track pitcher pitch count (estimate)
        if pitching_side == "home":
            if state.home_pitcher == "sp":
                state.home_sp_pitches += config.pitches_per_pa
        else:
            if state.away_pitcher == "sp":
                state.away_sp_pitches += config.pitches_per_pa

        # Walk-off check: bottom of 9th+ and home team takes the lead
        if (batting_side == "home" and state.inning >= 9
                and state.home_score > state.away_score):
            break

    return runs_total


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
    )

    max_innings = 15  # safety limit

    # SP pull thresholds for this game (sampled once per game)
    home_sp_limit = rng.normal(config.sp_pitch_limit_mean, config.sp_pitch_limit_std)
    away_sp_limit = rng.normal(config.sp_pitch_limit_mean, config.sp_pitch_limit_std)

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
        # SP pull check at inning boundaries
        if state.away_pitcher == "sp" and state.away_sp_pitches >= away_sp_limit:
            state.away_pitcher = "bp"
        if state.home_pitcher == "sp" and state.home_sp_pitches >= home_sp_limit:
            state.home_pitcher = "bp"

        # Extra innings: Manfred runner on 2B from 10th inning on
        if state.inning >= 10:
            state.bases = (False, True, False)

        # Top of inning: away team bats
        if state.top_bottom == "Top":
            simulate_half_inning(
                state, "away", "home", dists, transition_matrix, config, rng,
            )
            state.outs = 0
            state.bases = (False, False, False)
            state.top_bottom = "Bot"

            # If home team already leads after top of 9th+, game over
            # (This is wrong — the bottom still needs to happen unless home leads GOING INTO bottom)
            # Actually: bottom of 9th is skipped only if home leads after top 9
            if state.inning >= 9 and state.home_score > state.away_score:
                return state.home_score, state.away_score

            # Extra innings: Manfred runner
            if state.inning >= 10:
                state.bases = (False, True, False)

        # Bottom of inning: home team bats
        if state.top_bottom == "Bot":
            simulate_half_inning(
                state, "home", "away", dists, transition_matrix, config, rng,
            )
            # Walk-off check
            if state.inning >= 9 and state.home_score > state.away_score:
                return state.home_score, state.away_score

            state.outs = 0
            state.bases = (False, False, False)
            state.top_bottom = "Top"
            state.inning += 1

        # End of regulation check
        if state.inning > 9 and state.top_bottom == "Top" and state.home_score != state.away_score:
            return state.home_score, state.away_score

    # Tie after max innings — rare; call it a tie (split result)
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

        if hr + ar > 0:  # approximate extras detection
            # Check if game went to extras by looking at run totals
            pass  # Could track this in simulate_game if needed

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


def _compute_batter_outcome_rates(
    idx: dict,
    batter_id: int,
    game_date: str,
    n_pa: int = 500,
) -> dict[str, float] | None:
    """Compute a batter's empirical PA outcome rates from recent history."""
    if batter_id == 0:
        return None
    batter_df = idx.get("batter", {}).get(batter_id)
    if batter_df is None:
        return None
    return compute_outcome_rates(batter_df, game_date, n_pa)


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


def load_simulation_context(
    game: dict,
    target_date: str,
    lineup_data: dict,
    idx: dict,
    matchup_models: dict,
    base_rates: dict = None,
    calibration: dict = None,
) -> tuple[TeamContext, TeamContext]:
    """
    Build TeamContext objects for home and away teams.

    For each batter-pitcher matchup, builds a full outcome distribution from:
      1. Batter's empirical outcome rates (K%, BB%, HR%, etc.)
      2. Pitcher's empirical outcome rates
      3. Log5 odds-ratio combination
      4. Bayesian matchup model interaction adjustment
    """
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

    # Ensure batter index exists for fast per-batter lookups
    ensure_batter_index(idx)

    # Use league average as fallback
    if base_rates is None:
        base_rates = {o: 1.0 / _N_OUTCOMES for o in OUTCOME_ORDER}
    if calibration is None:
        calibration = {"alphas": {o: 0.0 for o in OUTCOME_ORDER}, "scale": 8.0}

    # ── Compute pitcher outcome rates ──
    away_sp_rates = None
    home_sp_rates = None
    if pd.notna(away_sp):
        away_sp_rates = _compute_pitcher_outcome_rates(idx, int(away_sp), gdate)
    if pd.notna(home_sp):
        home_sp_rates = _compute_pitcher_outcome_rates(idx, int(home_sp), gdate)

    # Bullpen outcome rates
    home_bp_rates = _compute_bp_outcome_rates(idx, home_team, gdate)
    away_bp_rates = _compute_bp_outcome_rates(idx, away_team, gdate)

    # ── Compute Bayesian matchup interaction xRVs ──
    home_vs_away_sp_xrv = [0.0] * 9  # home hitters vs away SP
    away_vs_home_sp_xrv = [0.0] * 9  # away hitters vs home SP

    if matchup_models and pd.notna(away_sp) and int(away_sp) in idx["pitcher"]:
        home_vs_away_sp_xrv = _compute_per_hitter_matchup_xrv(
            matchup_models, idx["pitcher"][int(away_sp)],
            int(away_sp), gdate, home_lineup,
        )
    if matchup_models and pd.notna(home_sp) and int(home_sp) in idx["pitcher"]:
        away_vs_home_sp_xrv = _compute_per_hitter_matchup_xrv(
            matchup_models, idx["pitcher"][int(home_sp)],
            int(home_sp), gdate, away_lineup,
        )

    # ── Build per-hitter outcome distributions ──
    def _build_dists_for_lineup(
        lineup, opposing_sp_rates, opposing_bp_rates,
        matchup_xrvs,
    ):
        sp_dists = []
        bp_dists = []
        for pos, (batter_id, _hand) in enumerate(lineup[:9]):
            batter_rates = _compute_batter_outcome_rates(idx, batter_id, gdate)

            # vs SP: log5(batter, pitcher) + matchup interaction
            sp_dist = build_matchup_distribution(
                batter_rates, opposing_sp_rates, base_rates,
                matchup_xrvs[pos] if pos < len(matchup_xrvs) else 0.0,
                calibration,
            )
            sp_dists.append(sp_dist)

            # vs BP: log5(batter, bp_composite), no matchup interaction
            bp_dist = build_matchup_distribution(
                batter_rates, opposing_bp_rates, base_rates,
                0.0,  # no specific matchup model for bullpen
                calibration,
            )
            bp_dists.append(bp_dist)

        return sp_dists, bp_dists

    home_sp_dists, home_bp_dists = _build_dists_for_lineup(
        home_lineup, away_sp_rates, away_bp_rates, home_vs_away_sp_xrv,
    )
    away_sp_dists, away_bp_dists = _build_dists_for_lineup(
        away_lineup, home_sp_rates, home_bp_rates, away_vs_home_sp_xrv,
    )

    home_ctx = TeamContext(
        team=home_team,
        lineup=home_lineup[:9],
        sp_outcome_dists=home_sp_dists,
        bp_outcome_dists=home_bp_dists,
    )
    away_ctx = TeamContext(
        team=away_team,
        lineup=away_lineup[:9],
        sp_outcome_dists=away_sp_dists,
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
    resp = client.get(f"{MLB_API}/game/{game_pk}/feed/live")
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

    # Check if SP is still pitching
    home_pitcher = "sp"
    away_pitcher = "sp"
    for side, team_box, pitcher_var in [
        ("home", home_team_box, "home"),
        ("away", away_team_box, "away"),
    ]:
        pitchers = team_box.get("pitchers", [])
        if len(pitchers) > 1:
            # More than one pitcher has been used → bullpen is active
            if side == "home":
                home_pitcher = "bp"
            else:
                away_pitcher = "bp"

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
    )


# ── Backtest ──────────────────────────────────────────────

def run_backtest(
    season: int,
    base_rates: dict,
    calibration: dict,
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
    # Filter to completed regular-season games with known SPs
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
                base_rates, calibration,
            )
            result = monte_carlo_win_prob(
                home_ctx, away_ctx, GameState(),
                transition_matrix, config,
            )
            preds.append(result["home_wp"])
            actuals.append(int(game["home_win"]))
        except Exception as e:
            n_skip += 1
            continue

    if not preds:
        print("No predictions generated")
        return

    preds = np.array(preds)
    actuals = np.array(actuals)

    # Clip to avoid log(0)
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

    # Calibration bins
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
    calibration: dict,
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

    # Load data
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
    if not matchup_models:
        print("No matchup models found. Using league-average xRV.")

    print(f"\n{'='*70}")
    print(f"{'Game':<35} {'Home WP':>8} {'Away WP':>8} {'Runs':>8} {'Status'}")
    print(f"{'='*70}")

    for game in games:
        status = game.get("status", "")
        is_live = status in ("Live", "In Progress")
        is_final = status == "Final"

        # Fetch lineup
        lu = fetch_lineup(client, game["game_pk"])
        if not lu.get("home") or not lu.get("away"):
            label = f"{game['away_team']:>3} @ {game['home_team']:<3}"
            print(f"  {label:<35} {'---':>8} {'---':>8} {'---':>8} No lineups")
            continue

        # Determine initial state
        if is_live:
            try:
                initial_state = fetch_live_game_state(client, game["game_pk"])
            except Exception:
                initial_state = GameState()
        else:
            initial_state = GameState()

        try:
            home_ctx, away_ctx = load_simulation_context(
                game, target_date, lu, idx, matchup_models or {},
                base_rates, calibration,
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
        base_rates, calibration, transition_matrix = load_sim_artifacts()
    except FileNotFoundError as e:
        print(f"Simulation data not found: {e}")
        print("Run `python src/build_transition_matrix.py` first.")
        return

    if args.backtest:
        config.n_sims = min(config.n_sims, 1000)  # faster for backtest
        run_backtest(args.backtest, base_rates, calibration, transition_matrix, config)
        return

    target_date = args.date or date.today().isoformat()
    predict_date(target_date, base_rates, calibration, transition_matrix,
                 config, game_pk_filter=args.game_pk)


if __name__ == "__main__":
    main()
