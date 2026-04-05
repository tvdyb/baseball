#!/usr/bin/env python3
"""
Diagnostic script: Why does the MC simulator over-predict runs by ~0.8/game?

Compares:
  1. Sim's average matchup distributions vs actual 2025 league-wide outcome rates
  2. Per-game sim distributions vs actual per-game outcomes (for backtest games)
  3. Transition matrix expected runs vs empirical expected runs
  4. Component-level attribution: log5, model blend, HFA, transition matrix
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_transition_matrix import (
    OUTCOME_ORDER, EVENT_MAP, BBTYPE_MAP,
    _classify_events_vectorized, _deterministic_transition,
)
from utils import DATA_DIR

SIM_DIR = DATA_DIR / "sim"
N_OUTCOMES = len(OUTCOME_ORDER)

# Runs value for each outcome (approximate, for quick attribution)
# These are rough linear weights from run expectancy
_OUTCOME_RUN_VALUES = {
    "K": -0.27, "BB": 0.32, "HBP": 0.35, "1B": 0.47, "2B": 0.77,
    "3B": 1.07, "HR": 1.40, "dp": -0.85, "out_ground": -0.25,
    "out_fly": -0.21, "out_line": -0.26,
}


def load_league_base_rates():
    with open(SIM_DIR / "league_base_rates.pkl", "rb") as f:
        return pickle.load(f)


def load_transition_matrix():
    with open(SIM_DIR / "transition_matrix.pkl", "rb") as f:
        return pickle.load(f)


def compute_actual_2025_rates():
    """Compute actual 2025 PA outcome rates from statcast."""
    sc = pd.read_parquet(
        DATA_DIR / "statcast" / "statcast_2025.parquet",
        columns=["game_pk", "events", "bb_type", "game_type"],
    )
    sc = sc[sc["game_type"] == "R"]
    pa = sc[sc["events"].notna()].copy()
    pa["outcome"] = _classify_events_vectorized(pa)
    pa = pa[pa["outcome"].notna()]
    total = len(pa)
    counts = pa["outcome"].value_counts()
    rates = {o: counts.get(o, 0) / total for o in OUTCOME_ORDER}
    return rates, total


def compute_per_game_actual_rates(game_pks: set):
    """Compute per-game outcome distributions for specific games."""
    sc = pd.read_parquet(
        DATA_DIR / "statcast" / "statcast_2025.parquet",
        columns=["game_pk", "events", "bb_type", "game_type",
                 "on_1b", "on_2b", "on_3b", "outs_when_up",
                 "inning", "inning_topbot",
                 "home_score", "away_score", "post_home_score", "post_away_score"],
    )
    sc = sc[sc["game_type"] == "R"]
    sc = sc[sc["game_pk"].isin(game_pks)]
    pa = sc[sc["events"].notna()].copy()
    pa["outcome"] = _classify_events_vectorized(pa)
    pa = pa[pa["outcome"].notna()]

    per_game = {}
    for gpk, grp in pa.groupby("game_pk"):
        total = len(grp)
        counts = grp["outcome"].value_counts()
        rates = np.array([counts.get(o, 0) / total for o in OUTCOME_ORDER])
        per_game[gpk] = rates
    return per_game, pa


def compute_expected_runs_from_transition(tm):
    """For key base states, compute expected runs per PA from transition matrix."""
    results = {}
    base_states = {
        "empty": (False, False, False),
        "1B": (True, False, False),
        "2B": (False, True, False),
        "3B": (False, False, True),
        "1B_2B": (True, True, False),
        "1B_3B": (True, False, True),
        "2B_3B": (False, True, True),
        "loaded": (True, True, True),
    }
    for name, bases in base_states.items():
        for outs in range(3):
            total_exp_runs = 0.0
            for oi, outcome in enumerate(OUTCOME_ORDER):
                key = (outcome, bases, outs)
                dist = tm.get(key, _deterministic_transition(outcome, bases, outs))
                exp_runs = sum(runs * prob for (_, runs, prob) in dist)
                total_exp_runs += exp_runs  # unweighted; we weight later
            results[(name, outs)] = total_exp_runs
    return results


def compute_empirical_expected_runs(pa_df):
    """Compute empirical expected runs per base state from actual PA data."""

    def _base_state(row):
        b1 = pd.notna(row.get("on_1b"))
        b2 = pd.notna(row.get("on_2b"))
        b3 = pd.notna(row.get("on_3b"))
        return (b1, b2, b3)

    base_state_names = {
        (False, False, False): "empty",
        (True, False, False): "1B",
        (False, True, False): "2B",
        (False, False, True): "3B",
        (True, True, False): "1B_2B",
        (True, False, True): "1B_3B",
        (False, True, True): "2B_3B",
        (True, True, True): "loaded",
    }

    pa_df = pa_df.copy()
    pa_df["bases"] = pa_df.apply(_base_state, axis=1)
    pa_df["outs"] = pa_df["outs_when_up"].fillna(0).astype(int)

    # Compute runs scored per PA using post_score - pre_score
    has_post = "post_home_score" in pa_df.columns and "post_away_score" in pa_df.columns
    if has_post:
        pa_df["runs_scored"] = np.where(
            pa_df["inning_topbot"] == "Top",
            pa_df["post_away_score"] - pa_df["away_score"],
            pa_df["post_home_score"] - pa_df["home_score"],
        )
        pa_df["runs_scored"] = pa_df["runs_scored"].clip(0, 4)
    else:
        pa_df["runs_scored"] = 0

    results = {}
    for (bases, outs_val), grp in pa_df.groupby(["bases", "outs"]):
        name = base_state_names.get(bases, str(bases))
        if isinstance(outs_val, (int, np.integer)) and outs_val < 3:
            results[(name, int(outs_val))] = grp["runs_scored"].mean()
    return results


def main():
    print("=" * 70)
    print("MC SIMULATOR DIAGNOSTIC REPORT")
    print("=" * 70)

    # Load backtest results
    bt = pd.read_parquet(DATA_DIR / "backtest" / "nrfi_ou_backtest_2025.parquet")
    print(f"\nBacktest: {len(bt)} games")
    print(f"  Sim mean total:    {bt['sim_total_mean'].mean():.2f}")
    print(f"  Actual mean total: {bt['actual_total'].mean():.2f}")
    print(f"  Bias (sim-actual): {bt['sim_total_mean'].mean() - bt['actual_total'].mean():+.3f}")
    print(f"  Correlation:       {bt['sim_total_mean'].corr(bt['actual_total']):.4f}")

    # ──────────────────────────────────────────────────────────
    # Section 1: League base rates (sim artifact) vs actual 2025
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SECTION 1: Sim league base rates vs actual 2025 outcome rates")
    print("=" * 70)

    sim_base_rates = load_league_base_rates()
    actual_rates, n_actual_pa = compute_actual_2025_rates()

    print(f"\n  {'Outcome':>10}  {'Sim Base':>10}  {'Actual 2025':>12}  {'Diff':>8}  {'Direction':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*10}")

    total_run_impact = 0.0
    for o in OUTCOME_ORDER:
        s = sim_base_rates[o]
        a = actual_rates[o]
        diff = s - a
        rv = _OUTCOME_RUN_VALUES[o]
        impact = diff * rv * 38  # ~38 PA per team per game, 2 teams
        total_run_impact += impact
        direction = "OVER" if diff > 0.002 else ("UNDER" if diff < -0.002 else "~match")
        print(f"  {o:>10}  {s:>10.4f}  {a:>12.4f}  {diff:>+8.4f}  {direction:>10}")

    print(f"\n  Approx run impact from base rate mismatch: {total_run_impact:+.2f} runs/game")
    print(f"  (Uses ~38 PA/team/game x 2 teams x linear weights)")

    # ──────────────────────────────────────────────────────────
    # Section 2: Per-game sim distributions vs actual
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SECTION 2: Per-game outcome rate comparison (backtest games)")
    print("=" * 70)

    game_pks = set(bt["game_pk"].values)
    per_game_actual, pa_df = compute_per_game_actual_rates(game_pks)
    n_matched = len(per_game_actual)
    print(f"\n  Games with actual PA data: {n_matched}/{len(game_pks)}")

    if n_matched > 0:
        # Compute the mean actual rates across matched games
        all_actual = np.array(list(per_game_actual.values()))
        mean_actual = all_actual.mean(axis=0)

        print(f"\n  Mean actual rates across {n_matched} backtest games:")
        print(f"  {'Outcome':>10}  {'Sim Base':>10}  {'Actual (games)':>14}")
        for i, o in enumerate(OUTCOME_ORDER):
            print(f"  {o:>10}  {sim_base_rates[o]:>10.4f}  {mean_actual[i]:>14.4f}")

    # ──────────────────────────────────────────────────────────
    # Section 3: Sim distribution quality check
    # ──────────────────────────────────────────────────────────
    # We can't re-run sim contexts without loading all models.
    # Instead, analyze how outcome rates translate to runs using the transition matrix.
    print(f"\n{'='*70}")
    print("SECTION 3: Transition matrix expected runs analysis")
    print("=" * 70)

    tm = load_transition_matrix()

    # For each outcome, compute the average runs scored across all base states
    # weighted by how often each state occurs in practice
    print("\n  3a: Runs scored per outcome (averaged over all base states, 0 outs)")
    print(f"  {'Outcome':>10}  {'TM exp runs':>12}  {'Det. exp runs':>14}  {'Diff':>8}")
    for o in OUTCOME_ORDER:
        tm_runs = 0.0
        det_runs = 0.0
        base_states = [(b1, b2, b3) for b1 in [False, True]
                       for b2 in [False, True] for b3 in [False, True]]
        for bases in base_states:
            # Transition matrix version
            key = (o, bases, 0)
            dist = tm.get(key, _deterministic_transition(o, bases, 0))
            tm_r = sum(runs * prob for (_, runs, prob) in dist)
            tm_runs += tm_r
            # Deterministic version
            det_dist = _deterministic_transition(o, bases, 0)
            det_r = sum(runs * prob for (_, runs, prob) in det_dist)
            det_runs += det_r
        # Average over 8 base states
        tm_runs /= 8
        det_runs /= 8
        print(f"  {o:>10}  {tm_runs:>12.4f}  {det_runs:>14.4f}  {tm_runs-det_runs:>+8.4f}")

    # 3b: Compare empirical expected runs to transition matrix for key states
    print(f"\n  3b: Expected runs per PA by base state (empirical vs transition matrix)")
    print(f"      Uses sim base rates as outcome weights for TM calculation")

    sim_rates_arr = np.array([sim_base_rates[o] for o in OUTCOME_ORDER])
    empirical_er = compute_empirical_expected_runs(pa_df)

    base_states_named = [
        ("empty", (False, False, False)),
        ("1B", (True, False, False)),
        ("2B", (False, True, False)),
        ("3B", (False, False, True)),
        ("1B_2B", (True, True, False)),
        ("loaded", (True, True, True)),
    ]

    print(f"  {'State':>8} {'Outs':>4}  {'TM (sim rates)':>14}  {'TM (actual rates)':>17}  {'Empirical':>10}  {'TM-Emp':>8}")
    actual_rates_arr = np.array([actual_rates[o] for o in OUTCOME_ORDER])

    for name, bases in base_states_named:
        for outs in range(3):
            # Weighted expected runs from transition matrix using sim base rates
            tm_er_sim = 0.0
            tm_er_actual = 0.0
            for oi, outcome in enumerate(OUTCOME_ORDER):
                key = (outcome, bases, outs)
                dist = tm.get(key, _deterministic_transition(outcome, bases, outs))
                exp_runs = sum(runs * prob for (_, runs, prob) in dist)
                tm_er_sim += sim_rates_arr[oi] * exp_runs
                tm_er_actual += actual_rates_arr[oi] * exp_runs

            emp = empirical_er.get((name, outs), float('nan'))
            print(f"  {name:>8} {outs:>4}  {tm_er_sim:>14.4f}  {tm_er_actual:>17.4f}  {emp:>10.4f}  {tm_er_sim-emp:>+8.4f}")

    # ──────────────────────────────────────────────────────────
    # Section 4: Full-game expected runs decomposition
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SECTION 4: Full-game expected runs decomposition")
    print("=" * 70)

    # A rough approximation of runs per game from outcome distributions:
    # Each half-inning starts with bases empty, 0 outs.
    # Expected runs per half-inning ≈ sum over outcomes of P(outcome) * E[runs|outcome,empty,0outs]
    # adjusted for the fact that 3 outs end the inning.
    # More precisely: use a Markov chain approach through the 24 base-out states.

    print("\n  4a: Markov chain expected runs per half-inning")

    def compute_half_inning_expected_runs(outcome_rates, tm):
        """Compute expected runs per half-inning using Markov chain through 24 states."""
        # States: (bases, outs) where bases is (b1,b2,b3) and outs is 0,1,2
        # Terminal: outs >= 3
        base_states = [(b1, b2, b3) for b1 in [False, True]
                       for b2 in [False, True] for b3 in [False, True]]
        state_idx = {}
        for i, bases in enumerate(base_states):
            for outs in range(3):
                state_idx[(bases, outs)] = i * 3 + outs
        n_states = 24

        # Build transition matrix and reward vector
        T = np.zeros((n_states, n_states))
        R = np.zeros(n_states)  # expected immediate reward (runs) from each state

        for bases in base_states:
            for outs in range(3):
                si = state_idx[(bases, outs)]
                for oi, outcome in enumerate(OUTCOME_ORDER):
                    p_outcome = outcome_rates[oi]
                    key = (outcome, bases, outs)
                    dist = tm.get(key, _deterministic_transition(outcome, bases, outs))

                    # Outs added by this outcome
                    outs_added_map = {
                        "K": 1, "BB": 0, "HBP": 0, "1B": 0, "2B": 0, "3B": 0, "HR": 0,
                        "dp": 2, "out_ground": 1, "out_fly": 1, "out_line": 1,
                    }
                    outs_added = outs_added_map[outcome]
                    new_outs = outs + outs_added

                    for (new_bases, runs, prob) in dist:
                        R[si] += p_outcome * prob * runs
                        if new_outs < 3:
                            sj = state_idx[(new_bases, new_outs)]
                            T[si, sj] += p_outcome * prob
                        # If new_outs >= 3, transition to terminal (absorbed)

        # Solve: E[runs | state i] = R[i] + sum_j T[i,j] * E[runs | state j]
        # (I - T) * E = R
        # E = (I - T)^{-1} * R
        try:
            E = np.linalg.solve(np.eye(n_states) - T, R)
        except np.linalg.LinAlgError:
            E = np.zeros(n_states)

        # Starting state: bases empty, 0 outs
        start = state_idx[((False, False, False), 0)]
        return E[start], E

    sim_er, sim_E = compute_half_inning_expected_runs(sim_rates_arr, tm)
    actual_er, actual_E = compute_half_inning_expected_runs(actual_rates_arr, tm)

    print(f"  Expected runs/half-inning (sim base rates + TM):    {sim_er:.4f}")
    print(f"  Expected runs/half-inning (actual 2025 rates + TM): {actual_er:.4f}")
    print(f"  Difference per half-inning:                         {sim_er - actual_er:+.4f}")
    print(f"  Difference per game (x18 half-innings):             {(sim_er - actual_er)*18:+.3f}")
    print()
    print(f"  Predicted game total (sim rates):    {sim_er * 18:.2f}")
    print(f"  Predicted game total (actual rates):  {actual_er * 18:.2f}")
    print(f"  Actual 2025 mean total:              {bt['actual_total'].mean():.2f}")
    print(f"  Sim backtest mean total:             {bt['sim_total_mean'].mean():.2f}")

    # ──────────────────────────────────────────────────────────
    # Section 5: Isolate the contribution of each component
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SECTION 5: Component-level attribution")
    print("=" * 70)

    # What if we used the DETERMINISTIC transition rules instead of empirical TM?
    det_tm = {}
    base_states_all = [(b1, b2, b3) for b1 in [False, True]
                       for b2 in [False, True] for b3 in [False, True]]
    for outcome in OUTCOME_ORDER:
        for bases in base_states_all:
            for outs in range(3):
                det_tm[(outcome, bases, outs)] = _deterministic_transition(outcome, bases, outs)

    sim_er_det, _ = compute_half_inning_expected_runs(sim_rates_arr, det_tm)
    actual_er_det, _ = compute_half_inning_expected_runs(actual_rates_arr, det_tm)

    print(f"\n  5a: Transition matrix (empirical vs deterministic)")
    print(f"  Empirical TM + sim rates:       {sim_er:.4f} runs/half-inning ({sim_er*18:.2f}/game)")
    print(f"  Deterministic TM + sim rates:   {sim_er_det:.4f} runs/half-inning ({sim_er_det*18:.2f}/game)")
    print(f"  TM contribution to bias:        {(sim_er - sim_er_det)*18:+.3f} runs/game")

    print(f"\n  5b: Outcome distribution contribution")
    print(f"  Sim rates + empirical TM:       {sim_er:.4f} runs/half-inning ({sim_er*18:.2f}/game)")
    print(f"  Actual rates + empirical TM:    {actual_er:.4f} runs/half-inning ({actual_er*18:.2f}/game)")
    print(f"  Distribution contribution:      {(sim_er - actual_er)*18:+.3f} runs/game")

    print(f"\n  5c: Per-outcome contribution to run inflation")
    print(f"  {'Outcome':>10}  {'Sim rate':>10}  {'Actual rate':>12}  {'Diff':>8}  {'RunVal':>8}  {'Contribution':>13}")
    total_contrib = 0.0
    for i, o in enumerate(OUTCOME_ORDER):
        diff = sim_rates_arr[i] - actual_rates_arr[i]
        rv = _OUTCOME_RUN_VALUES[o]
        # ~76 PA per game (both teams) = 38 * 2
        contrib = diff * rv * 76
        total_contrib += contrib
        print(f"  {o:>10}  {sim_rates_arr[i]:>10.4f}  {actual_rates_arr[i]:>12.4f}  {diff:>+8.4f}  {rv:>+8.2f}  {contrib:>+13.3f}")
    print(f"  {'TOTAL':>10}  {'':>10}  {'':>12}  {'':>8}  {'':>8}  {total_contrib:>+13.3f}")

    # ──────────────────────────────────────────────────────────
    # Section 6: Transition matrix bug hunt
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SECTION 6: Transition matrix spot checks")
    print("=" * 70)

    # Check some key transitions for anomalies
    suspect_keys = [
        ("1B", (False, False, False), 0),  # Single, bases empty
        ("2B", (True, False, False), 0),   # Double, runner on 1st
        ("out_fly", (False, False, True), 0),  # Sac fly situation
        ("out_fly", (False, False, True), 1),  # Sac fly, 1 out
        ("out_ground", (True, False, False), 0), # Ground out, runner on 1st
        ("HR", (True, True, True), 0),     # Grand slam
        ("BB", (True, True, True), 0),     # Bases loaded walk
        ("out_ground", (False, False, True), 0), # Ground out, runner on 3rd
        ("1B", (False, True, False), 0),   # Single with runner on 2nd
        ("1B", (True, True, False), 0),    # Single with 1st and 2nd
    ]

    print(f"\n  Transition matrix entries for key situations:")
    for key in suspect_keys:
        outcome, bases, outs = key
        dist = tm.get(key, _deterministic_transition(outcome, bases, outs))
        det = _deterministic_transition(outcome, bases, outs)
        exp_tm = sum(r * p for (_, r, p) in dist)
        exp_det = sum(r * p for (_, r, p) in det)

        b_str = "".join(["1" if b else "_" for b in bases])
        print(f"\n  {outcome}, bases=[{b_str}], {outs} outs:")
        print(f"    TM exp runs: {exp_tm:.3f}, Det exp runs: {exp_det:.3f}, diff: {exp_tm-exp_det:+.3f}")
        if len(dist) <= 8:
            for (nb, r, p) in sorted(dist, key=lambda x: -x[2]):
                nb_str = "".join(["1" if b else "_" for b in nb])
                print(f"      -> [{nb_str}] +{r}R  p={p:.3f}")

    # ──────────────────────────────────────────────────────────
    # Section 7: Double play rate analysis
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SECTION 7: Double play handling diagnostic")
    print("=" * 70)

    # The sim redistributes DP probability when no runner on 1st or 2 outs.
    # Check if the no_dp redistribution is causing issues.
    dp_rate_sim = sim_base_rates.get("dp", 0)
    dp_rate_actual = actual_rates.get("dp", 0)
    og_rate_sim = sim_base_rates.get("out_ground", 0)
    og_rate_actual = actual_rates.get("out_ground", 0)

    print(f"\n  DP rate:         sim={dp_rate_sim:.4f}, actual={dp_rate_actual:.4f}")
    print(f"  out_ground rate: sim={og_rate_sim:.4f}, actual={og_rate_actual:.4f}")
    print(f"  Combined:        sim={dp_rate_sim+og_rate_sim:.4f}, actual={dp_rate_actual+og_rate_actual:.4f}")
    print(f"\n  When DP is impossible (no runner on 1B or 2 outs), sim adds DP prob to out_ground.")
    print(f"  This should reduce runs (more outs). If DP rate is too LOW, fewer double plays")
    print(f"  means fewer outs per PA = more baserunners = more runs.")

    # ──────────────────────────────────────────────────────────
    # Section 8: Quick check on actual vs predicted per-game runs
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SECTION 8: Sim total distribution analysis")
    print("=" * 70)

    print(f"\n  Sim total distribution (percentiles):")
    for p in [10, 25, 50, 75, 90]:
        print(f"    {p}th: sim={bt['sim_total_mean'].quantile(p/100):.1f}, actual={bt['actual_total'].quantile(p/100):.0f}")

    print(f"\n  Sim total variance: {bt['sim_total_mean'].var():.2f}")
    print(f"  Actual total variance: {bt['actual_total'].var():.2f}")
    print(f"  Ratio (sim/actual): {bt['sim_total_mean'].var() / bt['actual_total'].var():.3f}")

    # How many games does the sim predict > 10 runs total?
    for thresh in [6, 8, 10, 12, 14]:
        sim_pct = (bt["sim_total_mean"] > thresh).mean()
        act_pct = (bt["actual_total"] > thresh).mean()
        print(f"  P(total > {thresh}): sim={sim_pct:.3f}, actual={act_pct:.3f}")

    # ──────────────────────────────────────────────────────────
    # Section 9: Sensitivity - how much do sim rates NEED to shift?
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SECTION 9: What rate shifts would fix the bias?")
    print("=" * 70)

    target_er = bt["actual_total"].mean() / 18  # target runs per half-inning
    print(f"\n  Target runs/half-inning: {target_er:.4f}")
    print(f"  Current (sim rates + TM): {sim_er:.4f}")
    print(f"  Gap: {sim_er - target_er:+.4f} runs/half-inning")

    # What if we scale K rate up and HR/hit rates down?
    for k_boost in [0.00, 0.01, 0.02, 0.03, 0.05]:
        adj_rates = sim_rates_arr.copy()
        k_idx = OUTCOME_ORDER.index("K")
        hr_idx = OUTCOME_ORDER.index("HR")
        bb_idx = OUTCOME_ORDER.index("BB")
        s_idx = OUTCOME_ORDER.index("1B")
        d_idx = OUTCOME_ORDER.index("2B")
        # Boost K, reduce hits proportionally
        adj_rates[k_idx] += k_boost
        hit_indices = [s_idx, d_idx, hr_idx, bb_idx]
        for hi in hit_indices:
            adj_rates[hi] -= k_boost / len(hit_indices)
        adj_rates = np.maximum(adj_rates, 0.001)
        adj_rates /= adj_rates.sum()
        adj_er, _ = compute_half_inning_expected_runs(adj_rates, tm)
        print(f"  K+{k_boost:.2f}: {adj_er:.4f} runs/half-inning ({adj_er*18:.2f}/game)")

    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Total bias: +{bt['sim_total_mean'].mean() - bt['actual_total'].mean():.2f} runs/game (sim predicts too many runs)
  Correlation: {bt['sim_total_mean'].corr(bt['actual_total']):.4f} (near zero = sim doesn't track game-to-game variation)

  BREAKDOWN:
  1. Base rate mismatch (sim artifact rates vs 2025):
     Expected contribution: {(sim_er - actual_er)*18:+.3f} runs/game
     (The league_base_rates.pkl was trained on 2021-2025 data,
      but 2025 may differ from the multi-year average)

  2. Transition matrix vs deterministic rules:
     Expected contribution: {(sim_er - sim_er_det)*18:+.3f} runs/game

  3. Matchup distribution blending (log5 + model + similarity):
     The build_matchup_distribution() blends log5 with model predictions.
     If models systematically predict higher hit/HR rates, this inflates runs.
     Cannot decompose without re-running sim context (would need ~10min).

  4. Low correlation suggests the sim's per-game differentiation is weak:
     The matchup-specific adjustments (log5, model, similarity) may not
     be creating enough spread between games, OR the spread doesn't
     correspond to actual run-scoring variation.

  KEY SUSPECTS:
  - If base rates are close to actual but sim still over-predicts by ~0.8,
    the matchup model blend is likely inflating offensive outcomes.
  - Near-zero correlation means the sim is basically predicting the same
    total for every game (low variance in sim_total_mean = {bt['sim_total_mean'].std():.2f}
    vs actual variance = {bt['actual_total'].std():.2f}).
""")


if __name__ == "__main__":
    main()
