SEASONS ?= 2017 2018 2019 2020 2021 2022 2023 2024 2025
PYTHON  ?= python3

# Run full pipeline end-to-end
all: scrape build-xrv matchup-models features train predict compare

# ── Scraping ────────────────────────────────────────────────

# Scrape all raw data: Statcast pitches, game results, weather, OAA, sprint speed, transactions, Kalshi
scrape: scrape-statcast scrape-games scrape-weather scrape-oaa scrape-sprint-speed scrape-transactions scrape-kalshi projections

scrape-statcast:
	$(PYTHON) src/scrape_statcast.py --seasons $(SEASONS)

scrape-games:
	$(PYTHON) src/scrape_games.py --seasons $(SEASONS)

scrape-weather:
	$(PYTHON) src/scrape_weather.py --seasons $(SEASONS)

scrape-oaa:
	$(PYTHON) src/scrape_oaa.py --seasons $(SEASONS)

# No CLI args — hardcoded season range
scrape-sprint-speed:
	$(PYTHON) src/scrape_sprint_speed.py

scrape-transactions:
	$(PYTHON) src/scrape_transactions.py --seasons $(SEASONS)

# Kalshi only has 2025 data
scrape-kalshi:
	$(PYTHON) src/scrape_kalshi.py --year 2025

# Sportsbook O/U lines (requires ODDS_API_KEY env var)
scrape-odds:
	$(PYTHON) src/scrape_odds.py --year 2025

# Scrape DraftKings O/U + ML lines from SBR data
sbr-odds:
	$(PYTHON) src/scrape_sbr_odds.py

# Scrape Kalshi total runs markets
kalshi-total:
	$(PYTHON) src/scrape_kalshi_total.py

# Preseason projections from FanGraphs
projections:
	$(PYTHON) src/scrape_projections.py --seasons $(SEASONS)

# ── Model Building ──────────────────────────────────────────

# Build pitch-level expected run values (prior-season training to avoid lookahead)
build-xrv:
	$(PYTHON) src/build_xrv.py --seasons $(SEASONS) --prior

# Train Bayesian matchup + arsenal models per season (uses prior season's xRV)
# Each season Y trains on data from year Y, producing models used for Y+1 features
matchup-models:
	@for yr in $(SEASONS); do \
		echo "Training matchup models for $$yr..."; \
		$(PYTHON) src/matchup_model.py --season $$yr; \
		$(PYTHON) src/arsenal_matchup_model.py --season $$yr; \
	done

# Train matchup models with pooled multi-season data (richer estimates)
matchup-models-pooled:
	$(PYTHON) src/matchup_model.py --season 2024 --pool-seasons 2022 2023 2024
	$(PYTHON) src/arsenal_matchup_model.py --season 2024 --pool-seasons 2022 2023 2024

# Train pitcher stuff + location models (P1 + P2)
pitcher-stuff:
	@for yr in $(SEASONS); do \
		echo "Training pitcher stuff models for $$yr..."; \
		$(PYTHON) src/pitcher_stuff_model.py --season $$yr; \
	done

# Train hitter evaluation models (H1-H4)
hitter-eval:
	@for yr in $(SEASONS); do \
		echo "Training hitter eval for $$yr..."; \
		$(PYTHON) src/hitter_eval.py --season $$yr; \
	done

# Train pitcher sequencing model (P3)
sequencing:
	@for yr in $(SEASONS); do \
		echo "Training sequencing model for $$yr..."; \
		$(PYTHON) src/pitcher_sequencing_model.py --season $$yr; \
	done

# Train swing similarity matchup model (M1)
similarity:
	@for yr in $(SEASONS); do \
		echo "Training similarity model for $$yr..."; \
		$(PYTHON) src/swing_similarity_matchup.py --season $$yr; \
	done

# Train all component models (pitcher + hitter + similarity)
component-models: pitcher-stuff hitter-eval sequencing similarity

# Train NRFI classification model
nrfi-model:
	$(PYTHON) src/nrfi_model.py

# Train total runs regression model
total-runs-model:
	$(PYTHON) src/total_runs_model.py

# Train stacking O/U model
ou-model:
	$(PYTHON) src/total_runs_stacker.py

# Train O/U vs line classifier
ou-vs-line:
	$(PYTHON) src/ou_vs_line_model.py

# Train all O/U and total-runs models
train-ou-models: nrfi-model total-runs-model ou-model ou-vs-line

# ── Features & Training ────────────────────────────────────

# Build pregame feature matrix for each season
features:
	@for yr in $(SEASONS); do \
		echo "Building features for $$yr..."; \
		$(PYTHON) src/feature_engineering.py --season $$yr; \
	done

# Walk-forward model evaluation across seasons
train:
	$(PYTHON) src/win_model.py --walk-forward

# ── Prediction & Evaluation ─────────────────────────────────

# Generate predictions for today's games
predict:
	$(PYTHON) src/predict.py

# Benchmark model vs Kalshi prediction market prices
compare:
	$(PYTHON) src/compare_vs_market.py

# Feature group ablation study
ablation:
	$(PYTHON) src/ablation_matchup.py

# Full model audit (walk-forward, bootstrap, calibration, ablation, market comparison)
audit:
	$(PYTHON) src/audit.py

# Audit without the expensive ablation step
audit-quick:
	$(PYTHON) src/audit.py --skip-ablation

# Remove audit outputs
clean-audit:
	rm -rf data/audit/

# Predict today's games
predict-today:
	$(PYTHON) src/predict.py

# Predict today's games (LR-only mode)
predict-today-lr:
	$(PYTHON) src/predict.py --lr-only

# Generate market-making picks for tomorrow
picks:
	$(PYTHON) src/picks.py

# Picks for a specific date
picks-date:
	$(PYTHON) src/picks.py --date $(DATE)

# Polymarket market-making bot (dry run)
poly-dry:
	$(PYTHON) src/polymarket_bot.py --date $(DATE) --dry-run --bankroll $(BANKROLL)

# Polymarket market-making bot (live)
poly-live:
	$(PYTHON) src/polymarket_bot.py --date $(DATE) --bankroll $(BANKROLL)

# Polymarket sizing table only (no trading)
poly-sizing:
	$(PYTHON) src/polymarket_bot.py --date $(DATE) --bankroll $(BANKROLL) --sizing-only

# ── Monte Carlo Simulation ────────────────────────────────

# Build simulation lookup tables (base-running transitions, xRV calibration)
build-sim-data:
	$(PYTHON) src/build_transition_matrix.py --seasons $(SEASONS)

# Run MC simulation for today's games
simulate:
	$(PYTHON) src/simulate.py

# Simulate a specific date
simulate-date:
	$(PYTHON) src/simulate.py --date $(DATE)

# Simulate a specific game (mid-game if live)
simulate-game:
	$(PYTHON) src/simulate.py --game-pk $(GAME_PK)

# Backtest simulator against historical results
simulate-backtest:
	$(PYTHON) src/simulate.py --backtest $(SEASON)

# Backtest all models vs DK market
backtest-vs-market:
	$(PYTHON) src/backtest_vs_market.py

# Stress-test individual strategies
strategy-backtest:
	$(PYTHON) src/strategy_backtest.py

# Ensemble model comparison
ensemble:
	$(PYTHON) src/ensemble_predictions.py

# Calibrate O/U predictions
calibrate-ou:
	$(PYTHON) src/calibrate_ou.py

# ML edge analysis
ml-edge:
	$(PYTHON) src/ml_edge_analysis.py

# O/U edge analysis
ou-edge:
	$(PYTHON) src/ou_edge_analysis.py

# Kalshi arbitrage backtest
kalshi-arb:
	$(PYTHON) src/kalshi_arb.py

# Backtest MC simulator vs Kalshi (pregame only)
sim-vs-kalshi:
	$(PYTHON) src/backtest_vs_kalshi.py --season $(SEASON)

# Backtest MC simulator vs Kalshi (pregame + in-game)
sim-vs-kalshi-full:
	$(PYTHON) src/backtest_vs_kalshi.py --season $(SEASON) --ingame --max-games $(or $(MAX_GAMES),100)

# Walk-forward Kalshi backtest
kalshi-backtest:
	$(PYTHON) src/kalshi_backtest.py --season $(or $(SEASON),2025) --n-sims $(or $(N_SIMS),5000)

# Full rebuild: all component models + features + train
rebuild-all: build-xrv matchup-models component-models features train

# Live in-game trader (dry run)
live-dry:
	$(PYTHON) src/live_trader.py --bankroll $(BANKROLL) --dry-run

# Live in-game trader
live:
	$(PYTHON) src/live_trader.py --bankroll $(BANKROLL)

# ── Live Trading & Daily Picks ──────────────────────────────

# Unified multi-strategy backtest and daily picks
unified-picks:
	$(PYTHON) src/unified_strategy.py \
	  $(if $(DATE),--date $(DATE),) \
	  $(if $(BANKROLL),--bankroll $(BANKROLL),)

# ── DK vs Kalshi Arbitrage Scanner ──────────────────────────
# Requires ODDS_API_KEY env var for live DK odds.
# Optional overrides: BANKROLL, EDGE, DATE
# Example: make arb-scan BANKROLL=5000 EDGE=0.07

# Live arb scan for today's games
arb-scan:
	$(PYTHON) src/kalshi_arb_live.py \
	  --bankroll $(or $(BANKROLL),1000) \
	  --edge-threshold $(or $(EDGE),0.07) \
	  $(if $(DATE),--date $(DATE),)

# Dry-run arb scan with sample data (no API calls needed)
arb-scan-dry:
	$(PYTHON) src/kalshi_arb_live.py --dry-run \
	  --bankroll $(or $(BANKROLL),1000) \
	  --edge-threshold $(or $(EDGE),0.07) \
	  --verbose

.PHONY: all scrape scrape-statcast scrape-games scrape-weather scrape-oaa \
        scrape-sprint-speed scrape-transactions scrape-kalshi scrape-odds \
        sbr-odds kalshi-total projections \
        build-xrv matchup-models matchup-models-pooled \
        pitcher-stuff hitter-eval sequencing similarity component-models \
        nrfi-model total-runs-model ou-model ou-vs-line train-ou-models \
        features train predict compare ablation \
        audit audit-quick clean-audit predict-today predict-today-lr \
        picks picks-date poly-dry poly-live poly-sizing \
        backtest-vs-market strategy-backtest ensemble calibrate-ou \
        ml-edge ou-edge kalshi-arb \
        build-sim-data simulate simulate-date simulate-game simulate-backtest \
        sim-vs-kalshi sim-vs-kalshi-full kalshi-backtest rebuild-all \
        unified-picks live-dry live arb-scan arb-scan-dry
