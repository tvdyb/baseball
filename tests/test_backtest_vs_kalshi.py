"""Tests for backtest_vs_kalshi.py — metrics, matching, and state extraction."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest_vs_kalshi import (
    StateSnapshot,
    GameState,
    compute_metrics,
    compute_roi,
    compute_calibration,
    match_candle_to_timestamp,
    extract_half_inning_states,
)


# ── Candle Matching ──────────────────────────────────────────────────────────


class TestCandleMatching:
    def test_exact_match(self):
        candles = [(100.0, 0.55), (200.0, 0.60), (300.0, 0.65)]
        assert match_candle_to_timestamp(candles, 200.0) == 0.60

    def test_closest_before(self):
        candles = [(100.0, 0.55), (200.0, 0.60), (300.0, 0.65)]
        # Target at 250 — closest is 200 (gap=50) or 300 (gap=-50), both within window
        price = match_candle_to_timestamp(candles, 250.0)
        assert price in (0.60, 0.65)  # either is acceptable within gap

    def test_no_match_outside_window(self):
        candles = [(100.0, 0.55)]
        assert match_candle_to_timestamp(candles, 2000.0, max_gap_seconds=600) is None

    def test_empty_candles(self):
        assert match_candle_to_timestamp([], 100.0) is None

    def test_prefers_closest(self):
        candles = [(100.0, 0.40), (190.0, 0.55), (210.0, 0.65)]
        # Target 200: closest is 210 (gap=10) vs 190 (gap=10) — tie, either ok
        price = match_candle_to_timestamp(candles, 200.0)
        assert price in (0.55, 0.65)


# ── Metrics ──────────────────────────────────────────────────────────────────


class TestMetrics:
    @pytest.fixture
    def sample_df(self):
        """100-game sample with known properties."""
        rng = np.random.default_rng(42)
        n = 200
        home_win = rng.binomial(1, 0.54, size=n)

        # Simulator: decent predictions with some noise
        sim = np.clip(0.54 + 0.15 * (home_win - 0.5) + rng.normal(0, 0.08, n), 0.05, 0.95)
        # Market: slightly better predictions
        market = np.clip(0.54 + 0.18 * (home_win - 0.5) + rng.normal(0, 0.06, n), 0.05, 0.95)

        return pd.DataFrame({
            "game_pk": range(n),
            "game_date": pd.date_range("2025-04-15", periods=n, freq="D").strftime("%Y-%m-%d"),
            "home_team": ["NYY"] * n,
            "away_team": ["BOS"] * n,
            "home_win": home_win,
            "sim_home_wp": sim,
            "kalshi_home_prob": market,
            "kalshi_volume": rng.uniform(1000, 50000, n),
        })

    def test_metrics_keys(self, sample_df):
        m = compute_metrics(sample_df, label="test")
        assert "sim_log_loss" in m
        assert "market_log_loss" in m
        assert "sim_auc" in m
        assert m["n"] == 200
        assert m["label"] == "test"

    def test_metrics_sane_values(self, sample_df):
        m = compute_metrics(sample_df, label="test")
        # Log loss should be between 0 and 1 for reasonable predictions
        assert 0 < m["sim_log_loss"] < 1.5
        assert 0 < m["market_log_loss"] < 1.5
        # AUC should be above 0.5 (better than random)
        assert m["sim_auc"] > 0.5
        assert m["market_auc"] > 0.5
        # Accuracy should be reasonable
        assert 0.4 < m["sim_accuracy"] < 0.8

    def test_metrics_empty_df(self):
        m = compute_metrics(pd.DataFrame(), label="empty")
        assert m == {}

    def test_perfect_predictor(self):
        df = pd.DataFrame({
            "home_win": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "sim_home_wp": [0.95, 0.05, 0.95, 0.05, 0.95, 0.05, 0.95, 0.05, 0.95, 0.05],
            "kalshi_home_prob": [0.50] * 10,
        })
        m = compute_metrics(df, label="perfect")
        # Simulator should have much better log loss than uninformative market
        assert m["sim_log_loss"] < m["market_log_loss"]
        assert m["sim_auc"] > m["market_auc"]


class TestROI:
    def test_roi_structure(self):
        df = pd.DataFrame({
            "home_win": [1, 0, 1, 0, 1] * 20,
            "sim_home_wp": [0.65, 0.35, 0.70, 0.30, 0.60] * 20,
            "kalshi_home_prob": [0.50] * 100,
            "game_date": pd.date_range("2025-04-01", periods=100, freq="D").strftime("%Y-%m-%d"),
        })
        results = compute_roi(df, edge_thresholds=[0.05, 0.10, 0.15])
        assert len(results) == 3
        assert all("threshold" in r for r in results)
        assert all("n_bets" in r for r in results)

    def test_no_bets_at_high_threshold(self):
        df = pd.DataFrame({
            "home_win": [1, 0, 1],
            "sim_home_wp": [0.52, 0.48, 0.51],
            "kalshi_home_prob": [0.50, 0.50, 0.50],
        })
        results = compute_roi(df, edge_thresholds=[0.10])
        assert results[0]["n_bets"] == 0


class TestCalibration:
    def test_calibration_bins(self):
        n = 500
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.2, 0.8, n)
        wins = (rng.random(n) < probs).astype(int)

        df = pd.DataFrame({
            "home_win": wins,
            "sim_home_wp": probs,
        })
        cal = compute_calibration(df, "sim_home_wp")
        assert len(cal) > 0
        for row in cal:
            assert "bin" in row
            assert "n" in row
            assert 0 <= row["actual_rate"] <= 1

    def test_well_calibrated(self):
        """A well-calibrated predictor should have small gaps."""
        n = 2000
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.3, 0.7, n)
        wins = (rng.random(n) < probs).astype(int)

        df = pd.DataFrame({"home_win": wins, "sim_home_wp": probs})
        cal = compute_calibration(df, "sim_home_wp")
        for row in cal:
            assert abs(row["gap"]) < 0.06, f"Calibration gap too large: {row}"


# ── Half-Inning State Extraction ─────────────────────────────────────────────


class TestStateExtraction:
    def _make_play(self, inning, is_top, start_time, runs_scored=0, is_complete=True):
        runners = []
        for _ in range(runs_scored):
            runners.append({"movement": {"end": "score"}})
        return {
            "about": {
                "inning": inning,
                "isTopInning": is_top,
                "halfInning": "top" if is_top else "bottom",
                "startTime": start_time,
                "isComplete": is_complete,
            },
            "result": {"type": "atBat", "rbi": runs_scored},
            "runners": runners,
            "matchup": {"pitcher": {"id": 100 if is_top else 200}},
        }

    def test_basic_extraction(self):
        plays = [
            self._make_play(1, True, "2025-07-15T23:00:00Z"),
            self._make_play(1, True, "2025-07-15T23:05:00Z"),
            self._make_play(1, False, "2025-07-15T23:10:00Z"),
            self._make_play(1, False, "2025-07-15T23:15:00Z"),
            self._make_play(2, True, "2025-07-15T23:20:00Z"),
        ]
        pbp = {
            "liveData": {
                "plays": {"allPlays": plays},
                "boxscore": {"teams": {"home": {}, "away": {}}},
            }
        }
        snapshots = extract_half_inning_states(pbp)
        assert len(snapshots) == 3  # Top 1, Bot 1, Top 2
        assert snapshots[0].game_state.inning == 1
        assert snapshots[0].game_state.top_bottom == "Top"
        assert snapshots[1].game_state.top_bottom == "Bot"
        assert snapshots[2].game_state.inning == 2

    def test_scores_accumulate(self):
        plays = [
            self._make_play(1, True, "2025-07-15T23:00:00Z", runs_scored=2),
            self._make_play(1, False, "2025-07-15T23:10:00Z", runs_scored=1),
            self._make_play(2, True, "2025-07-15T23:20:00Z"),
        ]
        pbp = {
            "liveData": {
                "plays": {"allPlays": plays},
                "boxscore": {"teams": {"home": {}, "away": {}}},
            }
        }
        snapshots = extract_half_inning_states(pbp)
        # By Top 2, score should be away=2, home=1
        top2 = [s for s in snapshots if s.inning_half == "Top 2"]
        assert len(top2) == 1
        assert top2[0].game_state.away_score == 2
        assert top2[0].game_state.home_score == 1

    def test_empty_plays(self):
        pbp = {
            "liveData": {
                "plays": {"allPlays": []},
                "boxscore": {"teams": {"home": {}, "away": {}}},
            }
        }
        assert extract_half_inning_states(pbp) == []
