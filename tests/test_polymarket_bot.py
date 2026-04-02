"""Tests for Polymarket bot order lifecycle, risk management, and state reconciliation."""
import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polymarket_bot import (
    BotConfig,
    GameState,
    MLBMarketMaker,
    PolyMarket,
    check_lineup_status,
    compute_half_kelly,
    fetch_mlb_game_status,
    round_to_tick,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return BotConfig(
        target_date="2026-03-30",
        bankroll=5000.0,
        min_edge=0.02,
        dry_run=True,
        max_loss=500.0,
    )


@pytest.fixture
def bot(config):
    return MLBMarketMaker(config)


@pytest.fixture
def game_state():
    """A game state with a matched Polymarket market."""
    gs = GameState(
        home_team="STL",
        away_team="NYM",
        model_fair=0.55,
        game_time_utc="2026-03-30T23:45:00+00:00",
    )
    gs.poly_market = PolyMarket(
        home_team="STL",
        away_team="NYM",
        home_token_id="token_stl_home",
        away_token_id="token_stl_away",
        game_start_time="2026-03-30T23:45:00+00:00",
        condition_id="cond_123",
        question="Will STL beat NYM?",
        volume=5000.0,
        neg_risk=False,
    )
    gs.poly_home_mid = 0.425
    gs.poly_home_bid = 0.42
    gs.poly_home_ask = 0.43
    gs.poly_spread = 0.01
    gs.edge = gs.model_fair - gs.poly_home_mid
    gs.half_kelly, gs.side = compute_half_kelly(gs.model_fair, gs.poly_home_mid)
    return gs


# ── Kelly Sizing ──────────────────────────────────────────────────────────────

class TestKellySizing:
    def test_buy_home_positive_edge(self):
        hk, side = compute_half_kelly(0.55, 0.45)
        assert side == "BUY_HOME"
        assert hk > 0
        # Kelly = (0.55 - 0.45) / (1 - 0.45) = 0.1818, half = 0.0909
        assert abs(hk - 0.0909) < 0.01

    def test_buy_away_negative_edge(self):
        hk, side = compute_half_kelly(0.40, 0.50)
        assert side == "BUY_AWAY"
        assert hk > 0
        # Kelly = (0.50 - 0.40) / 0.50 = 0.20, half = 0.10
        assert abs(hk - 0.10) < 0.01

    def test_no_edge(self):
        hk, side = compute_half_kelly(0.50, 0.50)
        assert hk == 0.0
        assert side == ""

    def test_invalid_price(self):
        assert compute_half_kelly(0.50, 0.0) == (0.0, "")
        assert compute_half_kelly(0.50, 1.0) == (0.0, "")
        assert compute_half_kelly(0.50, None) == (0.0, "")

    def test_round_to_tick(self):
        assert round_to_tick(0.555) == 0.56
        assert round_to_tick(0.554) == 0.55
        assert round_to_tick(0.001) == 0.0
        assert round_to_tick(0.995) == 1.0


# ── Quote Computation ─────────────────────────────────────────────────────────

class TestQuoteComputation:
    def test_compute_quotes_normal(self, bot, game_state):
        quotes = bot.compute_quotes(game_state)
        assert quotes is not None
        bid_price, bid_size, ask_price, ask_size = quotes
        # Quotes are centered around blended fair = w*model + (1-w)*market
        w = bot.config.model_weight
        blended = w * game_state.model_fair + (1 - w) * game_state.poly_home_mid
        assert bid_price < blended < ask_price
        assert bid_size > 0
        assert ask_size > 0

    def test_compute_quotes_halted(self, bot, game_state):
        game_state.quote_halted = True
        assert bot.compute_quotes(game_state) is None

    def test_compute_quotes_vol_spike(self, bot, game_state):
        game_state.vol_spike_until = time.time() + 60
        assert bot.compute_quotes(game_state) is None

    def test_compute_quotes_adverse_halted(self, bot, game_state):
        game_state.adverse_halted = True
        game_state.adverse_halt_until = time.time() + 30
        assert bot.compute_quotes(game_state) is None

    def test_compute_quotes_pre_pitch_pulled(self, bot, game_state):
        game_state.pre_pitch_pulled = True
        assert bot.compute_quotes(game_state) is None

    def test_compute_quotes_no_market(self, bot, game_state):
        game_state.poly_market = None
        assert bot.compute_quotes(game_state) is None

    def test_compute_quotes_thin_edge(self, bot, game_state):
        game_state.model_fair = 0.43  # edge ~0.5% < min_edge 2%
        game_state.edge = 0.43 - 0.425
        assert bot.compute_quotes(game_state) is None

    def test_compute_quotes_widened_spread(self, bot, game_state):
        """Widened spread should produce wider bid-ask than normal."""
        normal = bot.compute_quotes(game_state)
        game_state.spread_widened = True
        widened = bot.compute_quotes(game_state)
        assert normal is not None and widened is not None
        normal_width = normal[2] - normal[0]  # ask - bid
        widened_width = widened[2] - widened[0]
        assert widened_width > normal_width

    def test_adverse_halt_expires(self, bot, game_state):
        """Adverse halt should expire after timeout."""
        game_state.adverse_halted = True
        game_state.adverse_halt_until = time.time() - 1  # expired
        assert not game_state.is_adverse_halted
        # Should be able to quote again
        assert bot.compute_quotes(game_state) is not None


# ── Order Placement (Dry Run) ────────────────────────────────────────────────

class TestOrderPlacement:
    @pytest.mark.asyncio
    async def test_place_order_dry_run(self, bot):
        result = await bot._place_order(
            "token_123", "BUY", 0.50, 100.0, "NYM@STL", "BID"
        )
        assert result is not None
        assert "orderID" in result
        assert result["orderID"].startswith("dry_")

    @pytest.mark.asyncio
    async def test_place_order_no_client(self, bot):
        bot.config.dry_run = False
        bot._clob_client = None
        result = await bot._place_order(
            "token_123", "BUY", 0.50, 100.0, "NYM@STL", "BID"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_game_orders_dry_run(self, bot, game_state):
        game_state.bid_order_id = "dry_NYM@STL_BID_123"
        game_state.ask_order_id = "dry_NYM@STL_ASK_456"
        game_state.bid_price = 0.50
        game_state.ask_price = 0.60

        await bot._cancel_game_orders(game_state)
        assert game_state.bid_order_id is None
        assert game_state.ask_order_id is None
        assert game_state.bid_price is None
        assert game_state.ask_price is None


# ── Fill Detection & Adverse Selection ────────────────────────────────────────

class TestFillDetection:
    @pytest.mark.asyncio
    async def test_poll_order_status_dry_run_noop(self, bot):
        """In dry-run mode, polling should be a no-op."""
        bot.games["NYM@STL"] = GameState(
            home_team="STL", away_team="NYM", model_fair=0.55,
        )
        bot.games["NYM@STL"].bid_order_id = "dry_bid_123"
        await bot._poll_order_status()
        # Should still have the order
        assert bot.games["NYM@STL"].bid_order_id == "dry_bid_123"

    @pytest.mark.asyncio
    async def test_poll_detects_fill(self, bot, game_state):
        """Live mode: detect a partial fill via get_order."""
        bot.config.dry_run = False
        bot.games["NYM@STL"] = game_state
        game_state.bid_order_id = "order_abc"
        game_state.bid_price = 0.50
        game_state.bid_size = 100.0
        game_state.bid_filled = 0.0

        mock_client = MagicMock()
        mock_client.get_order.return_value = {
            "status": "LIVE",
            "size_matched": 50.0,
            "original_size": 100.0,
            "price": 0.50,
        }
        bot._clob_client = mock_client

        await bot._poll_order_status()

        assert game_state.bid_filled == 50.0
        assert game_state.net_position == 50.0
        assert game_state.total_bought == 50.0
        assert game_state.cost_basis == 25.0  # 50 * 0.50
        assert game_state.consecutive_bid_fills == 1
        assert len(game_state.fill_history) == 1

    @pytest.mark.asyncio
    async def test_poll_detects_fully_filled(self, bot, game_state):
        """Fully filled order should clear the order ID."""
        bot.config.dry_run = False
        bot.games["NYM@STL"] = game_state
        game_state.bid_order_id = "order_abc"
        game_state.bid_price = 0.50
        game_state.bid_size = 100.0
        game_state.bid_filled = 0.0

        mock_client = MagicMock()
        mock_client.get_order.return_value = {
            "status": "MATCHED",
            "size_matched": 100.0,
            "original_size": 100.0,
            "price": 0.50,
        }
        bot._clob_client = mock_client

        await bot._poll_order_status()

        assert game_state.bid_order_id is None  # cleared
        assert game_state.bid_filled == 0.0  # reset
        assert game_state.net_position == 100.0  # position recorded
        assert game_state.total_bought == 100.0

    @pytest.mark.asyncio
    async def test_adverse_selection_triggers(self, bot, game_state):
        """3 consecutive same-side fills should trigger adverse halt."""
        bot.config.dry_run = False
        bot.games["NYM@STL"] = game_state
        game_state.consecutive_bid_fills = 2  # already at 2
        game_state.bid_order_id = "order_abc"
        game_state.bid_price = 0.50
        game_state.bid_size = 10.0
        game_state.bid_filled = 0.0

        mock_client = MagicMock()
        mock_client.get_order.return_value = {
            "status": "LIVE",
            "size_matched": 10.0,
            "original_size": 10.0,
            "price": 0.50,
        }
        bot._clob_client = mock_client

        await bot._poll_order_status()

        # 3rd fill should trigger adverse halt
        assert game_state.adverse_halted is True
        assert game_state.adverse_halt_until > time.time()


# ── Kill Switch ───────────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_kill_switch_not_triggered(self, bot, game_state):
        bot.games["NYM@STL"] = game_state
        game_state.realized_pnl = -100.0
        assert bot._check_kill_switch() is False
        assert not bot._killed

    def test_kill_switch_triggered(self, bot, game_state):
        bot.games["NYM@STL"] = game_state
        game_state.realized_pnl = -600.0  # > max_loss of 500
        assert bot._check_kill_switch() is True
        assert bot._killed
        assert game_state.quote_halted
        assert game_state.halt_reason == "KILL SWITCH"

    def test_kill_switch_includes_unrealized(self, bot, game_state):
        bot.games["NYM@STL"] = game_state
        game_state.realized_pnl = -200.0
        game_state.net_position = 1000.0
        game_state.cost_basis = 600.0
        game_state.poly_home_mid = 0.05  # mark-to-market: 1000*0.05 = 50, unrealized = 50-600 = -550
        # total: -200 + (-550) = -750 > 500
        assert bot._check_kill_switch() is True


# ── State Persistence & Reconciliation ────────────────────────────────────────

class TestStatePersistence:
    def test_save_state_dry_run_noop(self, bot, game_state):
        """Dry run should not save state."""
        bot.games["NYM@STL"] = game_state
        bot.save_state()
        assert not bot._state_path().exists()

    def test_save_and_load_state(self, bot, game_state, tmp_path):
        """State should round-trip through JSON."""
        bot.config.dry_run = False

        # Point state dir to tmp
        import polymarket_bot as pb
        original_state_dir = pb.STATE_DIR
        pb.STATE_DIR = tmp_path
        try:
            bot.games["NYM@STL"] = game_state
            game_state.net_position = 50.0
            game_state.realized_pnl = 12.50
            game_state.cost_basis = 25.0
            game_state.total_bought = 50.0
            game_state.bid_order_id = "order_xyz"
            game_state.bid_price = 0.50
            game_state.fill_history = [{"side": "BID", "price": 0.50, "size": 50, "ts": 123}]

            bot.save_state()

            state_file = tmp_path / f"bot_state_{bot.config.target_date}.json"
            assert state_file.exists()

            with open(state_file) as f:
                state = json.load(f)

            assert state["target_date"] == "2026-03-30"
            game = state["games"]["NYM@STL"]
            assert game["net_position"] == 50.0
            assert game["realized_pnl"] == 12.50
            assert game["bid_order_id"] == "order_xyz"
            assert len(game["fill_history"]) == 1
        finally:
            pb.STATE_DIR = original_state_dir

    @pytest.mark.asyncio
    async def test_reconcile_live_order(self, bot, game_state, tmp_path):
        """Reconciliation should restore live orders and positions."""
        bot.config.dry_run = False

        import polymarket_bot as pb
        original_state_dir = pb.STATE_DIR
        pb.STATE_DIR = tmp_path
        try:
            # Write saved state
            state = {
                "saved_at": "2026-03-30T12:00:00Z",
                "target_date": "2026-03-30",
                "cycle": 5,
                "games": {
                    "NYM@STL": {
                        "home_team": "STL",
                        "away_team": "NYM",
                        "model_fair": 0.55,
                        "game_time_utc": "",
                        "bid_order_id": "order_live_1",
                        "ask_order_id": "order_dead_2",
                        "bid_price": 0.50,
                        "ask_price": 0.60,
                        "bid_size": 100.0,
                        "ask_size": 50.0,
                        "bid_filled": 30.0,
                        "ask_filled": 0.0,
                        "net_position": 30.0,
                        "realized_pnl": 5.0,
                        "cost_basis": 15.0,
                        "total_bought": 30.0,
                        "total_sold": 0.0,
                        "fill_history": [{"side": "BID", "price": 0.50, "size": 30, "ts": 100}],
                        "consecutive_bid_fills": 1,
                        "consecutive_ask_fills": 0,
                        "quote_halted": False,
                        "halt_reason": "",
                    }
                },
            }
            state_file = tmp_path / "bot_state_2026-03-30.json"
            with open(state_file, "w") as f:
                json.dump(state, f)

            # Setup bot with the game
            bot.games["NYM@STL"] = game_state

            # Mock CLOB client
            mock_client = MagicMock()

            def mock_get_order(order_id):
                if order_id == "order_live_1":
                    return {"status": "LIVE", "size_matched": 40.0, "original_size": 100.0}
                elif order_id == "order_dead_2":
                    return {"status": "CANCELLED", "size_matched": 0.0, "original_size": 50.0}
                return None

            mock_client.get_order.side_effect = mock_get_order
            bot._clob_client = mock_client

            await bot.load_and_reconcile_state()

            gs = bot.games["NYM@STL"]
            # Position and P&L restored
            assert gs.net_position == 30.0
            assert gs.realized_pnl == 5.0
            assert gs.cost_basis == 15.0
            assert len(gs.fill_history) == 1

            # Live order restored with updated fill
            assert gs.bid_order_id == "order_live_1"
            assert gs.bid_filled == 40.0

            # Dead order NOT restored
            assert gs.ask_order_id is None
        finally:
            pb.STATE_DIR = original_state_dir

    @pytest.mark.asyncio
    async def test_reconcile_wrong_date_ignored(self, bot, game_state, tmp_path):
        """State file for a different date should be ignored."""
        bot.config.dry_run = False

        import polymarket_bot as pb
        original_state_dir = pb.STATE_DIR
        pb.STATE_DIR = tmp_path
        try:
            state = {"saved_at": "now", "target_date": "2026-04-01", "games": {}}
            with open(tmp_path / "bot_state_2026-03-30.json", "w") as f:
                json.dump(state, f)

            bot.games["NYM@STL"] = game_state
            await bot.load_and_reconcile_state()

            # Nothing should be restored
            assert game_state.net_position == 0.0
        finally:
            pb.STATE_DIR = original_state_dir


# ── Lineup Degradation Suppression ────────────────────────────────────────────

class TestLineupDegradation:
    def test_degraded_game_halted_on_load(self, bot, tmp_path):
        """Games with lineup_degraded=True should be halted on load."""
        picks = {
            "date": "2026-03-30",
            "generated_utc": "2026-03-30T00:00:00Z",
            "picks": [
                {
                    "home_team": "STL",
                    "away_team": "NYM",
                    "model_fair": 0.55,
                    "home_sp": "Kyle Leahy",
                    "away_sp": "Clay Holmes",
                    "lineup_degraded": True,
                },
                {
                    "home_team": "PHI",
                    "away_team": "WSH",
                    "model_fair": 0.65,
                    "home_sp": "Taijuan Walker",
                    "away_sp": "Foster Griffin",
                    "lineup_degraded": False,
                },
            ],
        }

        import polymarket_bot as pb
        original_picks_dir = pb.PICKS_DIR
        pb.PICKS_DIR = tmp_path
        try:
            with open(tmp_path / "picks_2026-03-30.json", "w") as f:
                json.dump(picks, f)

            bot.load_model_picks()

            stl_game = bot.games["NYM@STL"]
            assert stl_game.quote_halted is True
            assert "lineup" in stl_game.halt_reason

            phi_game = bot.games["WSH@PHI"]
            assert phi_game.quote_halted is False
        finally:
            pb.PICKS_DIR = original_picks_dir


# ── Lineup Fetch Error Classification ─────────────────────────────────────────

class TestLineupFetch:
    @patch("polymarket_bot.httpx.Client")
    def test_check_lineup_api_error(self, mock_client_cls):
        """HTTP errors should return api_error status."""
        import httpx
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", request=MagicMock(), response=mock_resp
        )
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = check_lineup_status(12345)
        assert result["status"] == "api_error"
        assert result["home_lineup"] is False

    @patch("polymarket_bot.httpx.Client")
    def test_check_lineup_parse_error(self, mock_client_cls):
        """Unexpected schema should return parse_error status."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"unexpected": "schema"}  # no 'teams' key
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = check_lineup_status(12345)
        assert result["status"] == "parse_error"

    @patch("polymarket_bot.httpx.Client")
    def test_check_lineup_not_posted(self, mock_client_cls):
        """Valid response with empty batting order = not posted."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "teams": {
                "home": {"battingOrder": []},
                "away": {"battingOrder": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
            }
        }
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = check_lineup_status(12345)
        assert result["status"] == "ok"
        assert result["home_lineup"] is False  # empty batting order
        assert result["away_lineup"] is True   # 9 batters


# ── GameState Properties ──────────────────────────────────────────────────────

class TestGameStateProperties:
    def test_minutes_to_first_pitch(self):
        gs = GameState(
            home_team="STL", away_team="NYM", model_fair=0.55,
            game_time_utc="2099-12-31T23:59:00+00:00",
        )
        mins = gs.minutes_to_first_pitch
        assert mins is not None
        assert mins > 0

    def test_minutes_to_first_pitch_no_time(self):
        gs = GameState(home_team="STL", away_team="NYM", model_fair=0.55)
        assert gs.minutes_to_first_pitch is None

    def test_vol_spike_active(self):
        gs = GameState(home_team="STL", away_team="NYM", model_fair=0.55)
        gs.vol_spike_until = time.time() + 60
        assert gs.is_vol_spiking is True

    def test_vol_spike_expired(self):
        gs = GameState(home_team="STL", away_team="NYM", model_fair=0.55)
        gs.vol_spike_until = time.time() - 1
        assert gs.is_vol_spiking is False

    def test_half_spread_pre_lineup(self):
        gs = GameState(home_team="STL", away_team="NYM", model_fair=0.55)
        assert gs.half_spread == 0.03  # pre-lineup spread

    def test_half_spread_post_lineup(self):
        gs = GameState(home_team="STL", away_team="NYM", model_fair=0.55)
        gs.home_lineup_confirmed = True
        gs.away_lineup_confirmed = True
        assert gs.half_spread == 0.015  # post-lineup spread

    def test_half_spread_widened(self):
        gs = GameState(home_team="STL", away_team="NYM", model_fair=0.55)
        gs.spread_widened = True
        assert gs.half_spread == 0.045  # 3¢ * 1.5


# ── MLB API Error Handling ────────────────────────────────────────────────────

class TestMLBAPIErrorHandling:
    @patch("polymarket_bot.httpx.Client")
    def test_fetch_game_status_http_error(self, mock_client_cls):
        """HTTP error should return empty list, not crash."""
        import httpx
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_resp
        )
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_mlb_game_status("2026-03-30")
        assert result == []

    @patch("polymarket_bot.httpx.Client")
    def test_fetch_game_status_timeout(self, mock_client_cls):
        """Timeout should return empty list, not crash."""
        import httpx
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ReadTimeout("timeout")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_mlb_game_status("2026-03-30")
        assert result == []
