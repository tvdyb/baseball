# MLB Pregame Win Probability Model — Strategy Report

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Data Pipeline](#2-data-pipeline)
3. [Expected Run Value (xRV) Model](#3-expected-run-value-xrv-model)
4. [Bayesian Matchup Models](#4-bayesian-matchup-models)
5. [Feature Engineering](#5-feature-engineering)
6. [Win Probability Ensemble](#6-win-probability-ensemble)
7. [Walk-Forward Evaluation](#7-walk-forward-evaluation)
8. [Market Comparison & Betting Simulation](#8-market-comparison--betting-simulation)
9. [2025 Out-of-Sample Results](#9-2025-out-of-sample-results)
10. [Design Principles](#10-design-principles)

---

## 1. Executive Summary

This is a pregame MLB win probability model that combines pitch-level sabermetric features with an LR/XGB ensemble to predict game outcomes. The model is benchmarked against Kalshi prediction market closing prices to assess whether it captures information the market does not.

**2025 season results (out-of-sample, trained on 2018–2024):**

| Metric | Model | Kalshi |
|--------|-------|--------|
| Log loss | 0.6771 | 0.6813 |
| Brier score | 0.2422 | 0.2436 |
| AUC | 0.5867 | 0.5831 |

On 2,231 games with both model and Kalshi prices, the model achieves lower log loss and higher AUC than the market. A simulated flat-bet strategy at a 3% edge threshold produces **+5.3% ROI on 1,129 bets ($5,972 profit on $100 flat bets)** after 2% Kalshi fees.

---

## 2. Data Pipeline

Nine scrapers collect raw data into `data/` subdirectories:

| Scraper | Source | Output | Description |
|---------|--------|--------|-------------|
| `scrape_games.py` | MLB Stats API | `data/games/` | Game results, lineups, scores, venues |
| `scrape_statcast.py` | Baseball Savant | `data/statcast/` | Pitch-level data (velocity, movement, outcomes) |
| `scrape_sprint_speed.py` | Baseball Savant | `data/sprint_speed/` | Baserunner sprint speeds |
| `scrape_oaa.py` | Baseball Savant | `data/oaa/` | Outs Above Average (fielding) |
| `scrape_weather.py` | Weather APIs | `data/weather/` | Temperature, wind speed/direction per game |
| `scrape_transactions.py` | MLB API | `data/rosters/` | Trades, signings, DFA moves |
| `scrape_projections.py` | Projection systems | `data/projections/` | Preseason projected WAR, ERA, win% |
| `scrape_kalshi.py` | Kalshi API | `data/kalshi/` | Pregame closing prices (home win probability) |
| `scrape_polymarket.py` | Polymarket API | `data/polymarket/` | Alternative market prices |

All data is stored as Parquet files indexed by season.

---

## 3. Expected Run Value (xRV) Model

### 3.1 What xRV Measures

Expected Run Value quantifies the run-scoring impact of every pitch. Each pitch changes the game state (count, baserunners, outs) and therefore has a deterministic effect on expected runs — except when contact occurs, where the outcome depends on batted ball physics.

### 3.2 Non-Contact Pitches (Deterministic)

For balls (B) and strikes (S), xRV equals the actual `delta_run_exp` from the Statcast data. A called strike on a 3-1 count that changes the count to 3-2 has a known run expectancy change based on RE24 matrices. No modeling needed.

### 3.3 Contact Pitches (LightGBM Model)

For contact events (type = "X"), the outcome depends on batted ball quality. A LightGBM regression model predicts the expected run value:

**Features:**
- `launch_speed` — exit velocity (mph)
- `launch_angle` — vertical launch angle (degrees)
- `spray_angle` — horizontal angle (degrees)
- `sprint_speed` — batter's sprint speed (27.0 mph fallback for league avg)
- `park` — home team (categorical, captures park dimensions)

**Hyperparameters:**
```
objective: regression    metric: rmse
learning_rate: 0.05      num_leaves: 64
max_depth: 7             min_child_samples: 100
subsample: 0.8           colsample_bytree: 0.8
reg_alpha: 0.1           reg_lambda: 1.0
num_boost_round: 800
```

**Target:** `delta_run_exp` (actual run value change on that pitch)

### 3.4 Why xRV Instead of Traditional Stats

Traditional stats (ERA, FIP, wOBA) are aggregated over plate appearances and confound pitcher skill with sequencing luck, defense, and park effects. xRV operates at the pitch level and decomposes:
- **Pitcher skill**: How much run value does a pitch give up based on its physical characteristics?
- **Contact quality**: What is the expected outcome given exit velocity, launch angle, and park?
- **Batted ball luck**: The gap between expected and actual outcomes (removed by using xRV instead of actual runs)

### 3.5 Training Protocol (No Lookahead)

For walk-forward evaluation, the xRV model is trained using the prior-season method: for target year Y, train on all pitches from seasons [2017..Y-1], then apply to year Y. This ensures no data from the prediction year leaks into the xRV estimates.

---

## 4. Bayesian Matchup Models

Two hierarchical Bayesian models capture pitcher-hitter interactions that aggregate team-level features miss.

### 4.1 Sparse Matchup Model (`matchup_model.py`)

A PyMC hierarchical model that estimates expected xRV for specific pitcher-hitter pairs.

**Hierarchical Structure:**
```
intercept           ~ Normal(0, 0.1)
beta_features       ~ Normal(0, 0.05)          [speed, hmov, vmov, locx, locz, count, spin, ext, rel_x, rel_z]
sigma_ptype         ~ HalfNormal(0.05)
  ptype_effect      ~ Normal(0, sigma_ptype)    [per pitch type]
sigma_pitcher       ~ HalfNormal(0.05)
  pitcher_effect    ~ Normal(0, sigma_pitcher)  [per pitcher]
sigma_hitter        ~ HalfNormal(0.05)
  hitter_effect     ~ Normal(0, sigma_hitter)   [per hitter]
sigma_hitter_ptype  ~ HalfNormal(0.03)
  hitter_ptype_eff  ~ Normal(0, sigma_hitter_ptype)  [per hitter × pitch type]
sigma_matchup       ~ HalfNormal(0.02)
  matchup_effect    ~ Normal(0, sigma_matchup)  [per pitcher-hitter pair]
sigma_obs           ~ HalfNormal(0.2)
```

**Linear predictor:**
```
mu = intercept + beta · pitch_features + ptype_effect + pitcher_effect
     + hitter_effect + hitter_ptype_effect + matchup_effect
```

The tight priors (especially sigma_matchup = 0.02) provide heavy shrinkage — with sparse pitcher-hitter samples, the model pulls estimates toward population means rather than overfitting to small-sample noise.

**Inference:** ADVI (variational inference), 30,000 iterations with convergence callback. Separate models fit per batter handedness (L/R).

**Output:** `predict_matchup_xrv(hitter_id, pitcher_id)` → expected xRV per pitch for that matchup.

### 4.2 Arsenal Matchup Model (`arsenal_matchup_model.py`)

Extends the sparse model with a 13-dimensional pitcher arsenal profile, capturing hitter sensitivity to pitcher *types* rather than specific pitchers.

**Arsenal Feature Vector (13 dimensions):**

| Feature | Description |
|---------|-------------|
| `hard_velo` | Avg fastball velocity |
| `hard_pct` | Fastball usage % |
| `break_pct` | Breaking ball usage % |
| `offspeed_pct` | Offspeed usage % |
| `velo_spread` | Max-min velocity across pitch types |
| `hmov_range` | Max-min horizontal movement |
| `entropy` | Pitch mix diversity (Shannon entropy) |
| `hard_spin` | Avg fastball spin rate |
| `break_spin` | Avg breaking ball spin rate |
| `hard_ivb` | Avg fastball induced vertical break |
| `hard_ext` | Avg fastball extension |
| `rel_x_spread` | Std dev of release point x-position |
| `vmov_range` | Range of vertical movement across types |

**Additional priors:**
```
mu_arsenal    ~ Normal(0, 0.03)     [shape = 13]
sigma_arsenal ~ HalfNormal(0.02)    [shape = 13]
hitter_arsenal_beta ~ Normal(mu_arsenal, sigma_arsenal)  [shape = n_hitters × 13]
```

**Linear predictor adds:** `sum_k(hitter_arsenal_beta[h,k] * arsenal[pitcher,k])`

This captures patterns like "this hitter struggles against high-spin breaking ball pitchers" without needing direct matchup observations.

### 4.3 How Matchup Features Enter the Win Model

For each game, matchup predictions are generated for each hitter in the lineup vs. the opposing starting pitcher and top relievers:
- `matchup_xrv_mean` / `matchup_xrv_sum` — sparse model predictions
- `arsenal_matchup_xrv_mean` / `arsenal_matchup_xrv_sum` — arsenal model predictions
- `bp_matchup_xrv_mean` — bullpen matchup (sparse model vs. likely relievers)
- `bp_arsenal_matchup_xrv_mean` — bullpen matchup (arsenal model)
- `matchup_n_known` / `arsenal_matchup_n_known` — number of lineup hitters with model predictions (confidence signal)

---

## 5. Feature Engineering

### 5.1 Architecture

All feature computation flows through a single function: `compute_single_game_features()` in `feature_engineering.py`. This function is called by both:
- **Batch pipeline** (`build_game_features()`) — generates historical features for all games in a season
- **Live prediction** (`predict.py` → `build_live_features()`) — generates features for today's games

This shared-function design eliminates train/serve skew.

### 5.2 Pre-Indexing

Before computing features, raw data is pre-indexed for O(1) lookups:
- `_preindex_xrv(xrv_df)` — indexes xRV data by pitcher ID, team, and date for fast rolling-window queries
- Matchup models, park factors, OAA, projections, and weather are all loaded into dictionaries keyed by team/player/date

### 5.3 Complete Feature Inventory

#### Starting Pitcher Features (per side: `home_` / `away_` prefix)

| Feature | Description | Source |
|---------|-------------|--------|
| `sp_xrv_mean` | Mean xRV per pitch (rolling window, last N pitches) | xRV model |
| `sp_xrv_std` | Std dev of xRV | xRV model |
| `sp_n_pitches` | Number of pitches in rolling window | Statcast |
| `sp_k_rate` | Strikeout rate | Statcast |
| `sp_bb_rate` | Walk rate | Statcast |
| `sp_avg_velo` | Average fastball velocity | Statcast |
| `sp_pitch_mix_entropy` | Shannon entropy of pitch type distribution | Statcast |
| `sp_xrv_vs_L` / `sp_xrv_vs_R` | xRV split by batter handedness | xRV model |
| `sp_rest_days` | Days since last appearance | Games data |
| `sp_overperf` | Season-long actual ERA minus xRV-implied ERA | xRV + Games |
| `sp_overperf_recent` | Recent-window overperformance | xRV + Games |
| `sp_from_prior_season` | Boolean: using prior-season fallback (< 50 pitches) | xRV model |
| `sp_velo_trend` | Recent velocity trend (slope) | Statcast |
| `sp_spin_trend` | Recent spin rate trend (slope) | Statcast |
| `sp_xrv_trend` | Recent xRV trend (slope) | xRV model |
| `sp_home_xrv` / `sp_away_xrv` | xRV at home vs. away splits | xRV model |
| `sp_transition_entropy` | Pitch sequence transition entropy (unpredictability) | Statcast |
| `sp_info_confidence` | Confidence score based on sample size | Derived |
| `sp_context_xrv` | Context-adjusted xRV (high-leverage situations) | xRV model |

**Prior-Season Fallback:** When a pitcher has fewer than 50 pitches in the current season rolling window (early season, injury returns, callups), the model falls back to prior-season xRV values. The `sp_from_prior_season` flag lets the model learn to discount these estimates.

#### Bullpen Features (per side)

| Feature | Description |
|---------|-------------|
| `bp_xrv_mean` | Mean xRV across recent bullpen appearances |
| `bp_xrv_std` | Bullpen xRV variability |
| `bp_recent_ip` | Recent innings pitched (workload) |
| `bp_fatigue_score` | Composite fatigue metric |
| `bp_matchup_xrv_mean` | Bayesian matchup prediction vs. opposing lineup |
| `bp_matchup_n_relievers` | Number of relievers with matchup predictions |
| `bp_arsenal_matchup_xrv_mean` | Arsenal matchup vs. opposing lineup |
| `bp_arsenal_matchup_n_relievers` | Count with arsenal predictions |

#### Hitting Features (per side)

| Feature | Description |
|---------|-------------|
| `hit_xrv_mean` | Mean xRV per pitch for lineup hitters |
| `hit_xrv_contact` | Mean xRV on contact events (batted ball quality) |
| `hit_k_rate` | Lineup strikeout rate |

#### Defense Features (per side)

| Feature | Description |
|---------|-------------|
| `def_xrv_delta` | Defensive xRV adjustment (more negative = better defense) |
| `oaa_rate` | Outs Above Average rate from Statcast |

#### Matchup Features (per side)

| Feature | Description |
|---------|-------------|
| `matchup_xrv_mean` | Sparse Bayesian matchup — mean predicted xRV |
| `matchup_xrv_sum` | Sparse Bayesian matchup — total predicted xRV |
| `matchup_n_known` | Hitters with sparse matchup predictions |
| `arsenal_matchup_xrv_mean` | Arsenal Bayesian matchup — mean predicted xRV |
| `arsenal_matchup_xrv_sum` | Arsenal Bayesian matchup — total predicted xRV |
| `arsenal_matchup_n_known` | Hitters with arsenal matchup predictions |
| `sp_xrv_vs_lineup` | SP's xRV vs. lineup handedness composition |

#### Team Strength Features (per side)

| Feature | Description |
|---------|-------------|
| `team_prior` | Prior-season win percentage |
| `adjusted_team_prior` | Win% adjusted for offseason transactions |
| `projected_wpct` | Preseason projected win percentage |
| `sp_projected_era` | SP's projected ERA |
| `sp_projected_war` | SP's projected WAR |
| `recent_form` | Rolling recent win percentage |
| `team_games_played` | Games played so far (sample size signal) |

**Adjusted Team Priors:** The raw prior-season win% is adjusted for offseason roster changes:
1. Identify pitchers acquired/lost in transactions from Oct(Y-1) to Mar(Y)
2. Sum their xRV impact: `total_xrv = per_pitch_xrv × num_pitches`
3. Convert to talent: `talent = -total_xrv` (negative xRV = good pitcher)
4. Convert to wins: `talent_in_wins = total_talent / 10.0` (10 runs ≈ 1 win)
5. Adjust win%: `wpct_adj = talent_in_wins / 162`
6. Final: clipped to [0.30, 0.70]

#### Trade Deadline Features (per side, post-deadline only)

| Feature | Description |
|---------|-------------|
| `trade_net` | Net talent acquired at trade deadline |
| `trade_pitcher_xrv` | Pitcher xRV talent acquired |

#### Context Features

| Feature | Description |
|---------|-------------|
| `park_factor` | Venue park factor (shrinkage-adjusted) |
| `days_into_season` | Calendar days since Opening Day |
| `temperature` | Game-time temperature |
| `wind_speed` | Wind speed |
| `is_dome` | Dome/retractable roof indicator |
| `wind_out` | Wind blowing out (increases offense) |
| `wind_in` | Wind blowing in (decreases offense) |
| `platoon_pct` | Lineup platoon advantage percentage |

**Park Factor Shrinkage:**
```
weight = n_pitches / (n_pitches + 20000)
shrunk_mean = weight × park_mean + (1 - weight) × league_avg
park_factor = shrunk_mean / league_avg
```
Parks with fewer observations are pulled toward 1.0 (neutral).

### 5.4 Differencing

33 features are computed as home minus away differentials with sign conventions:

```python
DIFF_COLS = [
    ("sp_xrv_mean", -1),        # negative xRV = better pitcher → sign=-1
    ("sp_k_rate", 1),           # higher K rate = better → sign=+1
    ("sp_bb_rate", -1),         # higher BB rate = worse → sign=-1
    ("hit_xrv_mean", 1),        # higher = better for hitters → sign=+1
    ("matchup_xrv_mean", 1),
    ("team_prior", 1),
    ...
]
```

For each `(suffix, sign)` tuple:
```
diff_{suffix} = sign × (home_{suffix} - away_{suffix})
```

This produces 35 `diff_*` features. Combined with 19 raw features, the model uses **54 total features**.

### 5.5 Missing Value Strategy

**For Logistic Regression (`_smart_fillna`):**
- `diff_*` columns → fill with 0 (assume equal when unknown)
- Count columns (`sp_n_pitches`, `matchup_n_known`, etc.) → fill with training set median
- Everything else → fill with 0
- Medians from training set are saved and reused on test data

**For XGBoost:**
- No fill needed — XGBoost handles NaN natively via its split-finding algorithm
- Missing values are routed to the optimal child node during tree construction

---

## 6. Win Probability Ensemble

### 6.1 Architecture

The final win probability is a weighted blend of two models:

```
P(home_win) = w_LR × P_LR + (1 - w_LR) × P_XGB
```

Each model sees different feature representations of the same underlying data.

### 6.2 Logistic Regression Path

1. Select available features from the 54 `ALL_FEATURES` list
2. Apply `_smart_fillna` (diff→0, counts→median, else→0)
3. StandardScaler (mean=0, std=1)
4. Logistic Regression: `C=1.0, max_iter=1000, solver=lbfgs`

LR provides a calibrated baseline that captures linear relationships and is resistant to overfitting.

### 6.3 XGBoost Path

1. Select available features from `ALL_FEATURES`
2. Add auxiliary columns needed for nonlinear features: `sp_rest_days`, `bp_fatigue_score`, `days_into_season`
3. Apply `add_nonlinear_features()` to create additional engineered features:
   - **SP rest penalties:** `short_rest = max(0, 4 - rest)`, `long_rest = max(0, rest - 7)`
   - **Log pitch counts:** `log1p(sp_n_pitches)` (diminishing information returns)
   - **Squared fatigue:** `bp_fatigue_score²` (exponential cost of overuse)
   - **Signed-square differentials:** `sign(x) × x²` for `diff_sp_xrv_mean`, `diff_hit_xrv_mean`, `diff_matchup_xrv_mean` (larger edges matter disproportionately)
   - **Early-season mask:** `early_mask = clip(1 - days/60, 0, 1)` — weight that decays from 1.0 on Opening Day to 0.0 after 60 days
   - **Early-season interactions:** `early_mask × diff_projected_wpct`, `early_mask × diff_team_prior`, etc. (projections matter more early, current form matters more later)
   - **Prior dominance blend:** `z_proj × early_mask + z_xrv × (1 - early_mask)` — smoothly transitions from preseason projections to in-season performance
4. Build `xgb.DMatrix` (NaN values handled natively)
5. XGBoost gradient-boosted trees:

```
objective: binary:logistic    eval_metric: logloss
max_depth: 4                  learning_rate: 0.05
subsample: 0.8                colsample_bytree: 0.8
min_child_weight: 50          reg_alpha: 0.1
reg_lambda: 1.0               num_boost_round: 500
early_stopping_rounds: 50
```

**Early stopping validation split:** Last 20% of training data (chronological). The model trains on the first 80% and stops when validation log loss hasn't improved for 50 rounds. This prevents overfitting without leaking test data.

### 6.4 Blend Weight Optimization (OOF)

The blend weight `w_LR` is learned via 5-fold out-of-fold (OOF) cross-validation on the training set:

1. Split training data into 5 chronological folds (no shuffle)
2. For each fold: train both LR and XGB on 4 folds, predict on the held-out fold
3. Concatenate OOF predictions from both models
4. Optimize: `w_LR = argmin log_loss(y_oof, w × LR_oof + (1-w) × XGB_oof)` via `scipy.optimize.minimize_scalar`

This prevents the blend weight from being optimized on in-sample predictions, which would bias toward the more flexible (XGB) model.

---

## 7. Walk-Forward Evaluation

### 7.1 Protocol

The model is evaluated using strict walk-forward (expanding window) methodology:

```
Year 2021: Train on 2018-2020, predict 2021
Year 2022: Train on 2018-2021, predict 2022
Year 2023: Train on 2018-2022, predict 2023
Year 2024: Train on 2018-2023, predict 2024
Year 2025: Train on 2018-2024, predict 2025
```

No data from the prediction year is used in training — not for feature engineering, not for xRV models, not for matchup models, not for blend weights.

### 7.2 What This Validates

- The model generalizes across seasons (different rosters, rule changes, etc.)
- Feature engineering doesn't encode future information
- XGBoost early stopping doesn't leak via test-set evaluation
- The blend weight is stable across years

---

## 8. Market Comparison & Betting Simulation

### 8.1 Kalshi Integration

Kalshi pregame closing prices provide the market's implied win probability for each game. These are scraped daily and stored in `data/kalshi/`.

The **model edge** for each game is:
```
edge = model_home_prob - kalshi_home_prob
```

A positive edge means the model thinks the home team is underpriced; negative means the away team is underpriced.

### 8.2 Fee-Adjusted PnL

Kalshi charges fees on each contract purchase. The simulation uses a default 2% fee rate:

```python
def compute_bet_pnl(edge, market_prob, outcome, fee_pct=0.02):
    if edge > 0:  # Bet home
        cost = market_prob × 100
        fee = cost × fee_pct
        return (100 - cost - fee) if home_wins else (-cost - fee)
    else:  # Bet away
        cost = (1 - market_prob) × 100
        fee = cost × fee_pct
        return (100 - cost - fee) if away_wins else (-cost - fee)
```

All PnL figures in this report include fees.

### 8.3 Edge Thresholds

The simulation evaluates performance at multiple edge thresholds: **3%, 5%, 7%, 10%**. Higher thresholds mean fewer but more confident bets.

### 8.4 Sharpe Ratio

Annualized using actual calendar days in sample:
```
bets_per_year = n_bets × (365 / days_in_sample)
Sharpe = (mean_pnl / std_pnl) × sqrt(bets_per_year)
```

---

## 9. 2025 Out-of-Sample Results

### 9.1 Model Accuracy (All 2,501 Games)

| Metric | Value |
|--------|-------|
| Log loss | 0.6783 |
| Brier score | 0.2428 |
| AUC | 0.5802 |
| Probability range | 0.265 – 0.784 |
| Mean predicted home prob | 0.528 |

### 9.2 Model vs. Kalshi (2,231 Games with Market Prices)

| Metric | Model | Kalshi | Delta |
|--------|-------|--------|-------|
| Log loss | 0.6771 | 0.6813 | -0.0042 (model better) |
| Brier score | 0.2422 | 0.2436 | -0.0014 |
| AUC | 0.5867 | 0.5831 | +0.0036 |

Model-Kalshi probability correlation: **0.7265**

The model beats the market on all three calibration/discrimination metrics. The 0.73 correlation indicates the model captures most of the same information as the market, plus additional signal.

### 9.3 Betting Simulation (3% Edge Threshold, $100 Flat Bets)

| Metric | Value |
|--------|-------|
| Total bets | 1,129 |
| Total PnL | +$5,972 |
| ROI | +5.3% |
| Win rate | 43.2% |
| Date range | Apr 16 – Sep 28, 2025 |

### 9.4 Monthly Breakdown

| Month | Bets | PnL | ROI | Win Rate |
|-------|------|-----|-----|----------|
| April | 107 | -$1,015 | -9.5% | 31.8% |
| May | 217 | +$1,636 | +7.5% | 41.0% |
| June | 199 | +$892 | +4.5% | 43.7% |
| July | 193 | +$1,512 | +7.8% | 45.6% |
| August | 230 | +$2,140 | +9.3% | 46.5% |
| September | 183 | +$807 | +4.4% | 45.4% |

April underperformance is expected: the model relies on in-season rolling features that are sparse in the first month. By May, current-season data accumulates and the model stabilizes.

### 9.5 Edge Distribution

| Threshold | Bets Above |
|-----------|-----------|
| |edge| > 3% | 1,129 |
| |edge| > 5% | 755 |
| |edge| > 7% | 500 |
| |edge| > 10% | 274 |

Mean absolute edge: **9.1%**. Median: **6.5%**.

### 9.6 Directional Breakdown

| Side | Bets | ROI |
|------|------|-----|
| Home | 462 | +0.4% |
| Away | 667 | +8.7% |

The model's edge is concentrated in away bets — it is better at identifying when the market overprices the home team.

---

## 10. Design Principles

### 10.1 No Lookahead Bias

Every component is trained only on data available before the prediction date:
- xRV models: trained on prior seasons only
- Matchup models: fitted on prior-year pitch data
- Features: rolling windows using only past games
- Ensemble weights: OOF on training years only
- XGB early stopping: chronological train/val split within training data

### 10.2 Single Source of Truth

`compute_single_game_features()` is the sole function for feature computation, used by both batch and live pipelines. This eliminates feature drift between historical evaluation and live predictions.

### 10.3 Separate Model Pipelines

LR and XGB have intentionally different preprocessing:
- LR gets imputed, scaled linear features — appropriate for a model that assumes linearity
- XGB gets raw features with NaN plus nonlinear engineered features — leveraging its native missing-value handling and ability to capture interactions

### 10.4 Conservative Regularization

Every model layer includes regularization against overfitting:
- xRV: LightGBM with `min_child_samples=100`
- Matchup models: tight Bayesian priors (sigma ≤ 0.05) with heavy shrinkage
- Park factors: Bayesian shrinkage toward 1.0 (`shrinkage_n=20000`)
- LR: `C=1.0` (moderate L2 penalty)
- XGB: `min_child_weight=50`, `max_depth=4`, `reg_alpha=0.1`, `reg_lambda=1.0`
- Ensemble: OOF blend weight prevents overfitting to in-sample predictions

### 10.5 Graceful Degradation

When data is missing, the model degrades gracefully rather than failing:
- Missing pitcher data → prior-season fallback with confidence flag
- Missing matchup data → `n_known` count signals to the model that estimates are uncertain
- Missing weather → filled to neutral values
- Missing Kalshi prices → model predictions still generated, just no edge comparison

---

*Report generated from codebase analysis. All metrics are out-of-sample (2025 season, model trained on 2018–2024).*
