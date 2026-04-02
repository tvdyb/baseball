#!/usr/bin/env python3
"""Unit tests for the Monte Carlo game simulator."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_transition_matrix import OUTCOME_ORDER, _deterministic_transition
from simulate import (
    GameState,
    TeamContext,
    SimConfig,
    build_matchup_distribution,
    log5_combine,
    apply_matchup_adjustment,
    rates_to_probs,
    precompute_all_distributions,
    apply_transition,
    simulate_half_inning,
    simulate_game,
    monte_carlo_win_prob,
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def base_rates():
    return {
        "K": 0.224, "BB": 0.082, "HBP": 0.012,
        "1B": 0.146, "2B": 0.043, "3B": 0.004, "HR": 0.031,
        "dp": 0.025, "out_ground": 0.240, "out_fly": 0.160, "out_line": 0.033,
    }


@pytest.fixture
def calibration():
    return {
        "alphas": {o: 0.0 for o in OUTCOME_ORDER},  # neutral calibration
        "scale": 8.0,
    }


@pytest.fixture
def active_calibration():
    """Calibration with non-zero alphas for testing matchup effects."""
    return {
        "alphas": {
            "K": -2.0, "BB": 0.5, "HBP": 0.1,
            "1B": 1.0, "2B": 1.5, "3B": 1.5, "HR": 2.0,
            "dp": -0.8, "out_ground": -0.5, "out_fly": -0.3, "out_line": 0.0,
        },
        "scale": 8.0,
    }


@pytest.fixture
def transition_matrix():
    """Build a deterministic-only transition matrix for testing."""
    all_bases = [
        (b1, b2, b3)
        for b1 in [False, True]
        for b2 in [False, True]
        for b3 in [False, True]
    ]
    matrix = {}
    for outcome in OUTCOME_ORDER:
        for bases in all_bases:
            for outs in range(3):
                key = (outcome, bases, outs)
                matrix[key] = _deterministic_transition(outcome, bases, outs)
    return matrix


@pytest.fixture
def dummy_lineup():
    return [(100000 + i, "R") for i in range(9)]


def _make_league_avg_dists(base_rates):
    """Build outcome distributions using league-average rates for all hitters."""
    probs = rates_to_probs(dict(base_rates))
    return [probs.copy() for _ in range(9)]


@pytest.fixture
def home_ctx(dummy_lineup, base_rates):
    dists = _make_league_avg_dists(base_rates)
    return TeamContext(
        team="NYY",
        lineup=dummy_lineup,
        sp_outcome_dists=dists,
        bp_outcome_dists=[d.copy() for d in dists],
    )


@pytest.fixture
def away_ctx(dummy_lineup, base_rates):
    dists = _make_league_avg_dists(base_rates)
    return TeamContext(
        team="BOS",
        lineup=dummy_lineup,
        sp_outcome_dists=dists,
        bp_outcome_dists=[d.copy() for d in dists],
    )


# ── Outcome Distribution Tests ───────────────────────────

class TestOutcomeDistribution:

    def test_log5_preserves_league_avg(self, base_rates):
        """Log5 of league avg with itself should return league avg."""
        combined = log5_combine(base_rates, base_rates, base_rates)
        for o in OUTCOME_ORDER:
            assert abs(combined[o] - base_rates[o]) < 1e-4, f"{o}: {combined[o]} != {base_rates[o]}"

    def test_log5_high_k_pitcher(self, base_rates):
        """High-K pitcher should increase K rate in combined distribution."""
        high_k = dict(base_rates)
        high_k["K"] = 0.35  # much higher than league
        # Reduce others to compensate
        total = sum(high_k.values())
        for o in OUTCOME_ORDER:
            high_k[o] /= total
        combined = log5_combine(base_rates, high_k, base_rates)
        assert combined["K"] > base_rates["K"]

    def test_log5_high_hr_batter(self, base_rates):
        """High-HR batter should increase HR rate."""
        slugger = dict(base_rates)
        slugger["HR"] = 0.06
        total = sum(slugger.values())
        for o in OUTCOME_ORDER:
            slugger[o] /= total
        combined = log5_combine(slugger, base_rates, base_rates)
        assert combined["HR"] > base_rates["HR"]

    def test_matchup_distribution_sums_to_one(self, base_rates, calibration):
        probs = build_matchup_distribution(
            base_rates, base_rates, base_rates, 0.0, calibration,
        )
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_matchup_distribution_with_shift(self, base_rates, active_calibration):
        for xrv in [-0.01, -0.005, 0.005, 0.01]:
            probs = build_matchup_distribution(
                base_rates, base_rates, base_rates, xrv, active_calibration,
            )
            assert abs(probs.sum() - 1.0) < 1e-9
            assert (probs >= 0).all()

    def test_matchup_distribution_nan_xrv(self, base_rates, calibration):
        """NaN matchup xRV should use base log5 rates without adjustment."""
        probs_nan = build_matchup_distribution(
            base_rates, base_rates, base_rates, float("nan"), calibration,
        )
        probs_zero = build_matchup_distribution(
            base_rates, base_rates, base_rates, 0.0, calibration,
        )
        np.testing.assert_array_almost_equal(probs_nan, probs_zero)

    def test_matchup_adjustment_positive_xrv(self, base_rates, active_calibration):
        """Positive xRV (hitter-favorable) should increase hit rates."""
        combined = log5_combine(base_rates, base_rates, base_rates)
        adjusted = apply_matchup_adjustment(combined, 0.01, active_calibration)
        hr_idx = OUTCOME_ORDER.index("HR")
        k_idx = OUTCOME_ORDER.index("K")
        probs_neutral = rates_to_probs(dict(combined))
        probs_positive = rates_to_probs(adjusted)
        assert probs_positive[hr_idx] > probs_neutral[hr_idx]
        assert probs_positive[k_idx] < probs_neutral[k_idx]

    def test_none_batter_rates_uses_league_avg(self, base_rates, calibration):
        """None batter rates should fall back to league average."""
        probs_none = build_matchup_distribution(
            None, base_rates, base_rates, 0.0, calibration,
        )
        probs_league = build_matchup_distribution(
            base_rates, base_rates, base_rates, 0.0, calibration,
        )
        np.testing.assert_array_almost_equal(probs_none, probs_league)

    def test_rates_to_probs_clamping(self):
        """Extreme rates should be clamped."""
        rates = {o: 0.0 for o in OUTCOME_ORDER}
        rates["K"] = 0.90  # way too high
        rates["out_ground"] = 0.10
        probs = rates_to_probs(rates)
        k_idx = OUTCOME_ORDER.index("K")
        # After clamping K to 0.50 and renormalizing
        assert probs[k_idx] < 0.90


# ── Deterministic Transition Tests ────────────────────────

class TestDeterministicTransitions:

    def test_hr_clears_bases(self):
        for bases in [(True, True, True), (True, False, False), (False, False, False)]:
            results = _deterministic_transition("HR", bases, 0)
            total_runs = sum(r * p for _, r, p in results)
            runners = sum(bases) + 1
            assert abs(total_runs - runners) < 1e-9
            for nb, _, _ in results:
                assert nb == (False, False, False)

    def test_walk_bases_loaded(self):
        results = _deterministic_transition("BB", (True, True, True), 0)
        assert len(results) == 1
        nb, runs, prob = results[0]
        assert runs == 1
        assert nb == (True, True, True)

    def test_walk_empty_bases(self):
        results = _deterministic_transition("BB", (False, False, False), 0)
        assert len(results) == 1
        nb, runs, _ = results[0]
        assert runs == 0
        assert nb == (True, False, False)

    def test_strikeout_no_change(self):
        bases = (True, False, True)
        results = _deterministic_transition("K", bases, 1)
        assert len(results) == 1
        nb, runs, _ = results[0]
        assert nb == bases
        assert runs == 0

    def test_sac_fly(self):
        """Fly out with runner on 3B and < 2 outs scores the runner."""
        results = _deterministic_transition("out_fly", (False, False, True), 1)
        assert len(results) == 1
        nb, runs, _ = results[0]
        assert runs == 1
        assert nb == (False, False, False)

    def test_sac_fly_two_outs_no_score(self):
        """Fly out with 2 outs does NOT score runner (inning over)."""
        results = _deterministic_transition("out_fly", (False, False, True), 2)
        assert len(results) == 1
        nb, runs, _ = results[0]
        assert runs == 0

    def test_triple_clears_runners(self):
        results = _deterministic_transition("3B", (True, True, True), 0)
        assert len(results) == 1
        nb, runs, _ = results[0]
        assert runs == 3
        assert nb == (False, False, True)


# ── Simulation Tests ──────────────────────────────────────

class TestSimulation:

    def test_game_completes(self, home_ctx, away_ctx, transition_matrix):
        config = SimConfig(n_sims=1, random_seed=42)
        rng = np.random.default_rng(42)
        dists = precompute_all_distributions(home_ctx, away_ctx)
        hr, ar = simulate_game(
            home_ctx, away_ctx, GameState(),
            dists, transition_matrix, config, rng,
        )
        assert hr >= 0
        assert ar >= 0

    def test_monte_carlo_probabilities_sum_to_one(
        self, home_ctx, away_ctx, transition_matrix
    ):
        config = SimConfig(n_sims=100, random_seed=42)
        result = monte_carlo_win_prob(
            home_ctx, away_ctx, GameState(),
            transition_matrix, config,
        )
        assert abs(result["home_wp"] + result["away_wp"] + result["tie_pct"] - 1.0) < 1e-9

    def test_deterministic_with_seed(
        self, home_ctx, away_ctx, transition_matrix
    ):
        """Same seed should produce same results."""
        config = SimConfig(n_sims=100, random_seed=123)
        r1 = monte_carlo_win_prob(
            home_ctx, away_ctx, GameState(),
            transition_matrix, config,
        )
        r2 = monte_carlo_win_prob(
            home_ctx, away_ctx, GameState(),
            transition_matrix, config,
        )
        assert r1["home_wp"] == r2["home_wp"]

    def test_midgame_state(
        self, home_ctx, away_ctx, transition_matrix
    ):
        """Simulation from mid-game state should work."""
        state = GameState(
            inning=7, top_bottom="Bot", outs=1,
            bases=(True, False, True),
            home_score=3, away_score=5,
            home_lineup_pos=4, away_lineup_pos=2,
            home_pitcher="bp", away_pitcher="bp",
        )
        config = SimConfig(n_sims=100, random_seed=42)
        result = monte_carlo_win_prob(
            home_ctx, away_ctx, state,
            transition_matrix, config,
        )
        # Away team is winning 5-3 in bottom 7th, should have higher WP
        assert result["away_wp"] > result["home_wp"]

    def test_walkoff_bottom_9(
        self, home_ctx, away_ctx, transition_matrix
    ):
        """Home team winning after top 9 should end game."""
        state = GameState(
            inning=9, top_bottom="Top", outs=0,
            home_score=10, away_score=0,
        )
        config = SimConfig(n_sims=50, random_seed=42)
        result = monte_carlo_win_prob(
            home_ctx, away_ctx, state,
            transition_matrix, config,
        )
        # Home team should win almost every sim
        assert result["home_wp"] > 0.95

    def test_strong_hitter_advantage(self, base_rates, active_calibration, transition_matrix):
        """Team with better hitters (higher hit rates) should win more often."""
        lineup = [(100000 + i, "R") for i in range(9)]

        # Slugger lineup: higher hit rates
        slugger_rates = dict(base_rates)
        slugger_rates["HR"] = 0.05
        slugger_rates["1B"] = 0.18
        slugger_rates["K"] = 0.18
        total = sum(slugger_rates.values())
        slugger_rates = {o: v / total for o, v in slugger_rates.items()}

        # Weak lineup: higher K, lower hits
        weak_rates = dict(base_rates)
        weak_rates["K"] = 0.30
        weak_rates["HR"] = 0.02
        weak_rates["1B"] = 0.12
        total = sum(weak_rates.values())
        weak_rates = {o: v / total for o, v in weak_rates.items()}

        strong_dists = [rates_to_probs(dict(slugger_rates)) for _ in range(9)]
        weak_dists = [rates_to_probs(dict(weak_rates)) for _ in range(9)]

        strong = TeamContext(
            team="STR", lineup=lineup,
            sp_outcome_dists=strong_dists,
            bp_outcome_dists=[d.copy() for d in strong_dists],
        )
        weak = TeamContext(
            team="WEK", lineup=lineup,
            sp_outcome_dists=weak_dists,
            bp_outcome_dists=[d.copy() for d in weak_dists],
        )

        config = SimConfig(n_sims=2000, random_seed=42)
        result = monte_carlo_win_prob(
            strong, weak, GameState(),
            transition_matrix, config,
        )
        # Strong team (home) should win more than 50%
        assert result["home_wp"] > 0.50

    def test_per_hitter_distributions_matter(self, base_rates, transition_matrix):
        """Different per-hitter distributions should produce different results than uniform."""
        lineup = [(100000 + i, "R") for i in range(9)]

        # Lineup with three power hitters (slots 2-4), rest league avg
        league_dist = rates_to_probs(dict(base_rates))
        power_rates = dict(base_rates)
        power_rates["HR"] = 0.07   # well above league avg 0.04
        power_rates["2B"] = 0.06   # above league avg 0.043
        power_rates["K"] = 0.22    # only slightly above league avg 0.20
        total = sum(power_rates.values())
        power_rates = {o: v / total for o, v in power_rates.items()}
        power_dist = rates_to_probs(power_rates)

        mixed_dists = [league_dist.copy() for _ in range(9)]
        for slot in [2, 3, 4]:
            mixed_dists[slot] = power_dist  # heart of lineup is sluggers

        uniform_dists = [league_dist.copy() for _ in range(9)]

        mixed_ctx = TeamContext(
            team="MIX", lineup=lineup,
            sp_outcome_dists=mixed_dists,
            bp_outcome_dists=[d.copy() for d in mixed_dists],
        )
        uniform_ctx = TeamContext(
            team="UNI", lineup=lineup,
            sp_outcome_dists=uniform_dists,
            bp_outcome_dists=[d.copy() for d in uniform_dists],
        )

        config = SimConfig(n_sims=5000, random_seed=42)
        r_mixed = monte_carlo_win_prob(
            mixed_ctx, uniform_ctx, GameState(),
            transition_matrix, config,
        )
        # Mixed team has one power hitter — should score slightly more on average
        assert r_mixed["home_runs_mean"] > r_mixed["away_runs_mean"]


# ── Apply Transition Tests ────────────────────────────────

class TestApplyTransition:

    def test_hr_scores_all(self, transition_matrix):
        rng = np.random.default_rng(42)
        bases = (True, True, True)
        new_bases, runs, outs_added = apply_transition(
            "HR", bases, 0, transition_matrix, rng,
        )
        assert new_bases == (False, False, False)
        assert runs == 4
        assert outs_added == 0

    def test_strikeout_adds_out(self, transition_matrix):
        rng = np.random.default_rng(42)
        _, runs, outs_added = apply_transition(
            "K", (False, False, False), 0, transition_matrix, rng,
        )
        assert outs_added == 1
        assert runs == 0

    def test_dp_adds_two_outs(self, transition_matrix):
        rng = np.random.default_rng(42)
        _, _, outs_added = apply_transition(
            "dp", (True, False, False), 0, transition_matrix, rng,
        )
        assert outs_added == 2
