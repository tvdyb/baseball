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

.PHONY: all scrape scrape-statcast scrape-games scrape-weather scrape-oaa \
        scrape-sprint-speed scrape-transactions scrape-kalshi projections \
        build-xrv matchup-models matchup-models-pooled \
        features train predict compare ablation \
        audit audit-quick clean-audit predict-today predict-today-lr
