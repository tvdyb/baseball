# MLB Monte Carlo Game Simulator

At-bat-by-at-bat Monte Carlo game simulator using Bayesian matchup models, with in-game rebalancing strategy backtested against Kalshi prediction market prices.

## Quick Start

```bash
pip install -r requirements.txt

# Build simulation lookup tables (transition matrix, xRV calibration)
make build-sim-data

# Simulate today's games
make simulate

# Simulate a specific game (mid-game if live)
make simulate-game GAME_PK=777250

# Backtest simulator vs Kalshi (pregame + in-game)
make sim-vs-kalshi-full SEASON=2025 MAX_GAMES=100
```

## Key Ideas

- **Monte Carlo at-bat simulation**: Simulates games pitch-by-pitch using base-running transition matrices derived from historical Statcast data. Each PA outcome is sampled from a probability distribution shaped by the pitcher-hitter matchup.
- **Bayesian matchup xRV**: Per-hitter expected run values against the opposing pitcher, computed via hierarchical shrinkage models (PyMC/ADVI) trained in arsenal feature space. Avoids the sparse pitcher×hitter matrix problem.
- **Mid-game state injection**: Feed current inning, score, outs, baserunners, and lineup position from the MLB live API — simulator runs forward from any game state.
- **Kelly rebalancing**: Rebalance position every half-inning using fractional Kelly sizing based on edge between simulator win probability and Kalshi market price.

## Backtest Results (2025 season, 492 games)

| Strategy | Games Traded | Total PnL | ROI | Sharpe | Max DD |
|----------|-------------|-----------|-----|--------|--------|
| 25% Kelly, 5% edge | 472 | $4,012 | 0.8% | 3.66 | -$282 |
| 50% Kelly, 5% edge | 472 | $8,024 | 1.6% | 3.66 | -$564 |
| 100% Kelly, 5% edge | 472 | $16,049 | 3.3% | 3.66 | -$1,128 |

Best config (quarter Kelly, 5% min edge): ROI 0.82%, 95% CI [0.51%, 1.14%], P(ROI>0) = 100%.

## Pipeline

```
scrape_statcast.py ──────────────────────┐
scrape_games.py ─────────────────────────┤
scrape_weather.py ───────────────────────┤
scrape_oaa.py ───────────────────────────┤  data/
scrape_sprint_speed.py ──────────────────┤
scrape_transactions.py ──────────────────┤
scrape_kalshi.py ────────────────────────┤
scrape_projections.py ───────────────────┘
        │
        ▼
build_xrv.py ─────────────────────────→ data/xrv/         (pitch-level xRV)
        │
        ▼
matchup_model.py ─────────────────────→ models/            (Bayesian matchup effects)
arsenal_matchup_model.py ─────────────→ models/            (arsenal matchup effects)
        │
        ▼
build_transition_matrix.py ───────────→ data/sim/          (transition matrix, calibration)
        │
        ▼
simulate.py ──────────────────────────→ MC win probability (pregame or mid-game)
        │
        ▼
backtest_vs_kalshi.py ────────────────→ sim vs market backtest + Kelly rebalancing
```

## Project Structure

```
src/
├── simulate.py                MC game simulator (core engine)
├── build_transition_matrix.py Base-running transitions + xRV calibration from Statcast
├── backtest_vs_kalshi.py      Backtest simulator vs Kalshi (pregame + in-game)
├── build_xrv.py               Expected run value model (LightGBM + count-based)
├── matchup_model.py           Bayesian pitcher-hitter matchup model (PyMC/ADVI)
├── arsenal_matchup_model.py   Arsenal-dimension matchup model
├── feature_engineering.py     Matchup xRV computation + pregame features
├── win_model.py               Pregame win probability model (LR + XGBoost)
├── predict.py                 Daily predictions + lineup/schedule API
├── scrape_*.py                Data scrapers (Statcast, games, weather, etc.)
└── utils.py                   Shared constants and utilities

tests/
├── test_simulate.py           MC simulator unit tests
└── test_backtest_vs_kalshi.py Backtest metrics and state extraction tests
```

## Requirements

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

`data/` and `models/` are gitignored — the pipeline generates everything from scratch via the scraping scripts.
