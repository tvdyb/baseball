# MLB Win Probability Model

End-to-end pipeline from pitch-level Statcast data to pregame win probabilities, benchmarked against Kalshi prediction market prices.

## Key Ideas

- **xRV (expected run value) model**: Count-based run expectancy for non-contact outcomes, LightGBM for contact outcomes using exit velocity, launch angle, spray angle, sprint speed, and park. Trained on prior seasons only to avoid lookahead bias.
- **Arsenal-based Bayesian matchup model**: Characterizes pitchers along 13 arsenal dimensions (velo, movement, mix, spin, extension, release point spread), learns per-hitter sensitivities via hierarchical shrinkage (PyMC/ADVI). Avoids the sparse pitcher×hitter estimation problem by working in arsenal feature space.
- **Game-level win model**: Logistic regression + XGBoost ensemble with learned blend weights. Walk-forward evaluation across seasons with no lookahead bias. 45 features including SP quality, bullpen fatigue, matchup effects, platoon splits, defense (OAA), team priors, trade deadline acquisitions, and weather.
- **Market comparison**: Model vs Kalshi closing prices — log loss, Brier score, AUC, calibration by bin, edge analysis, and simulated flat-bet ROI at various edge thresholds.

## Pipeline

```
scrape_statcast.py ──────────────────────────┐
scrape_games.py ─────────────────────────────┤
scrape_weather.py ───────────────────────────┤
scrape_oaa.py ───────────────────────────────┤  data/
scrape_sprint_speed.py ──────────────────────┤
scrape_transactions.py ──────────────────────┤
scrape_kalshi.py ────────────────────────────┘
        │
        ▼
build_xrv.py ──────────────────────────────→ data/xrv/       (pitch-level xRV)
        │
        ▼
matchup_model.py ──────────────────────────→ models/          (Bayesian matchup effects)
arsenal_matchup_model.py ──────────────────→ models/          (arsenal matchup effects)
        │
        ▼
feature_engineering.py ────────────────────→ data/features/   (pregame feature matrix)
        │
        ▼
win_model.py ──────────────────────────────→ models/          (trained win model)
predict.py ────────────────────────────────→ daily predictions
        │
        ▼
compare_vs_market.py ──────────────────────→ model vs Kalshi benchmark
ablation_matchup.py ───────────────────────→ feature group ablation study
```

## Usage

Run the full pipeline from scratch:

```bash
# 1. Scrape all data (Statcast, games, weather, defense, sprint speed, transactions)
make scrape

# 2. Build pitch-level expected run values (prior-season training, no lookahead)
make build-xrv

# 3. Train Bayesian matchup models (per-season, pooled multi-year)
make matchup-models

# 4. Build pregame feature matrix for all seasons
make features

# 5. Walk-forward model evaluation
make train

# 6. Predict today's games
make predict

# 7. Benchmark against Kalshi
make compare

# Or run everything:
make all
```

Override the default season range:

```bash
make scrape SEASONS="2022 2023 2024 2025"
make features SEASONS="2025"
```

## Requirements

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
├── scrape_statcast.py          Pitch-level Statcast data via pybaseball
├── scrape_games.py             Game results and lineups from MLB Stats API
├── scrape_weather.py           Game-day weather from MLB Stats API
├── scrape_oaa.py               Outs Above Average (fielding) via pybaseball
├── scrape_sprint_speed.py      Sprint speed leaderboards via pybaseball
├── scrape_transactions.py      Roster transactions from MLB Stats API
├── scrape_kalshi.py            Kalshi prediction market closing prices
├── build_xrv.py                Expected run value model (LightGBM contact + count-based non-contact)
├── matchup_model.py            Bayesian pitcher-hitter matchup model (PyMC/ADVI)
├── arsenal_matchup_model.py    Arsenal-dimension matchup model (PyMC/ADVI)
├── feature_engineering.py      Pregame feature matrix construction
├── win_model.py                Win probability model (LR + XGBoost ensemble)
├── predict.py                  Generate predictions for upcoming games
├── compare_vs_market.py        Benchmark model vs prediction market prices
├── ablation_matchup.py         Feature group ablation study
└── utils.py                    Shared constants and utilities
```

`data/` and `models/` are gitignored — the pipeline generates everything from scratch via the scraping scripts.
