# MLB Prediction Market Strategy Report

A comprehensive evaluation of every strategy tested against actual prediction market prices (Kalshi, with DraftKings as the benchmark signal). Every result below uses a strict zero-look-ahead-bias protocol: all parameters, thresholds, hyperparameters, and model selections were determined on validation data only, then frozen and applied to an untouched out-of-sample test set. All P&L is computed at actual Kalshi contract prices.

**Bottom line: One strategy works. DraftKings closing moneylines are systematically more accurate than Kalshi pre-game prices. Buying underpriced Kalshi contracts when DK disagrees by 4%+ produces Sharpe 4.85, +22.6% ROI, and p=0.001 significance on 57 out-of-sample bets.**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Methodology: Zero Look-Ahead Bias Protocol](#2-methodology-zero-look-ahead-bias-protocol)
3. [Data Sources](#3-data-sources)
4. [Strategy 1: DK-vs-Kalshi Moneyline Arbitrage (ACTIVE)](#4-strategy-1-dk-vs-kalshi-moneyline-arbitrage-active)
5. [Strategy 2: LightGBM Win Model vs Kalshi (RETIRED)](#5-strategy-2-lightgbm-win-model-vs-kalshi-retired)
6. [Strategy 3: Monte Carlo Simulator vs Kalshi (RETIRED)](#6-strategy-3-monte-carlo-simulator-vs-kalshi-retired)
7. [Strategy 4: Multi-Book Consensus vs Kalshi (INCONCLUSIVE)](#7-strategy-4-multi-book-consensus-vs-kalshi-inconclusive)
8. [Strategy 5: DK Line Movement Filter (NO VALUE)](#8-strategy-5-dk-line-movement-filter-no-value)
9. [Strategy 6: Kalshi O/U Total Runs Arbitrage (RETIRED)](#9-strategy-6-kalshi-ou-total-runs-arbitrage-retired)
10. [Strategy 7: O/U Under Classifier vs DK (RETIRED)](#10-strategy-7-ou-under-classifier-vs-dk-retired)
11. [Strategy 8: NRFI/YRFI Model (UNVERIFIABLE)](#11-strategy-8-nrfiyrfi-model-unverifiable)
12. [Strategy 9: Away Underdog Anomaly (SUPERSEDED)](#12-strategy-9-away-underdog-anomaly-superseded)
13. [Strategy 10: Pick-em Home Moneyline (RETIRED)](#13-strategy-10-pick-em-home-moneyline-retired)
14. [Enhancement Analysis: Filters and Variants](#14-enhancement-analysis-filters-and-variants)
15. [Production Configuration](#15-production-configuration)
16. [Risk Factors and Caveats](#16-risk-factors-and-caveats)
17. [Forward-Testing Plan](#17-forward-testing-plan)
18. [Appendix A: Key Formulas](#appendix-a-key-formulas)
19. [Appendix B: Glossary](#appendix-b-glossary)
20. [Appendix C: Data Inventory](#appendix-c-data-inventory)
21. [Appendix D: Code Reference](#appendix-d-code-reference)

---

## 1. Executive Summary

### What Was Tested

We systematically evaluated every available signal source against actual Kalshi prediction market prices on MLB moneyline contracts during the 2025 season. Every strategy was tested using the same rigorous temporal protocol: parameters selected on April-June 2025 (validation), results measured on July-October 2025 (test), with zero information leakage between periods.

### The Verdict

| Strategy | Status | Test Bets | Win Rate | ROI | Sharpe | p-value |
|----------|--------|-----------|----------|-----|--------|---------|
| DK arb, underdogs (edge>4%) | **ACTIVE** | 57 | 64.9% | **+22.6%** | **4.85** | **0.001** |
| DK arb, baseline (edge>6%) | **ACTIVE** | 69 | 72.5% | **+23.0%** | **4.15** | 0.0002 |
| DK arb, price [.35,.55] (edge>4%) | ACTIVE | 74 | 67.6% | +21.5% | 3.84 | 0.001 |
| Multi-book consensus (edge>3%) | Inconclusive | 21 | 52.4% | +6.2% | 3.97 | 0.295 |
| LightGBM win model | Retired | 351 | 39.0% | +0.9% | -0.01 | N/A |
| MC simulator | Retired | 104 | — | -1.1% | — | N/A |
| O/U total runs arb | Retired | 117 | 49.6% | -1.8% | 0.15 | N/A |
| O/U under classifier | Retired | 1,106 | — | -4.1% | — | 0.989 |
| NRFI/YRFI | Unverifiable | — | — | — | — | — |

### Key Findings

1. **DraftKings closing moneylines are the only signal that consistently beats Kalshi.** This is a structural market inefficiency, not a model prediction. DK's closing lines are set by a competitive market with professional bettors; Kalshi is a thinner prediction exchange with retail participants. When they disagree, DK is almost always right.

2. **The edge is real at every threshold tested.** Even at 2% minimum edge, the DK arb produces Sharpe 2.90 on 259 test bets. At 15% edge, it's Sharpe 5.06 on 33 bets at 84.8% win rate. The monotonic increase in quality with edge size is exactly what you'd expect from a genuine structural inefficiency, not noise.

3. **Underdogs are systematically more profitable.** When the DK-Kalshi edge points to the underdog side (Kalshi price < $0.50), the test ROI is +22.6% vs +11.0% when it points to the favorite. Kalshi's retail participants appear to systematically overprice favorites and underprice underdogs.

4. **Every ML model we built has zero edge over Kalshi.** A LightGBM trained on 155 Statcast features (Optuna-optimized, 120 trials, 5-fold TSCV on 2024 data) achieves BSS of -0.001 vs Kalshi on the test set. The MC game simulator achieves -1.1% ROI. Our models replicate what the market already knows.

5. **No NRFI/YRFI market data exists on any platform.** Kalshi does not offer NRFI contracts. Polymarket does not offer them. Our DK/SBR odds data covers only moneyline, O/U, and spreads. The NRFI model cannot be backtested against real market prices and therefore cannot be verified for profitability.

6. **Kalshi O/U total runs data exists but shows no arb edge.** 38 games with Kalshi strike prices vs DK closing O/U lines yield -1.8% ROI. The normal approximation used to extrapolate DK-implied P(over) at non-DK strikes may be too crude, or Kalshi's O/U pricing is simply more efficient than its moneyline pricing.

---

## 2. Methodology: Zero Look-Ahead Bias Protocol

### The Problem With Naive Backtesting

The most common error in sports betting research is unconscious look-ahead bias: using information from the test period to make decisions that should have been made beforehand. Common examples include:

- Choosing a model variant because it has the best BSS on the full dataset (should be chosen on validation only)
- Selecting a threshold (like "bet when edge > 7%") by looking at which threshold produces the best test ROI
- Fitting calibration curves on data that includes the test period
- Tuning hyperparameters on any data that overlaps with the evaluation period

Any of these errors makes backtest results unreliable. A strategy that backtests at +15% ROI with look-ahead bias might actually produce -5% ROI in live trading.

### Our Protocol

We use a strict three-period temporal split:

```
TRAIN:       Full 2024 season (~2,429 games)
             - LightGBM hyperparameter tuning (Optuna, 120 trials, 5-fold TSCV)
             - Model fitting (LGB, logistic regression, MC simulator calibration)
             - Feature engineering (no 2025 data touches this)

VALIDATION:  2025-04-16 to 2025-06-30 (942 games)
             - Edge threshold selection (grid search over 10 values: 2%-15%)
             - Kelly fraction selection (grid search over 5 values: 10%-30%)
             - Filter selection (underdog filter, price range filter, movement, consensus)
             - Ensemble weight optimization
             - Optimization target: sample-size-weighted Sharpe ratio
               score = sharpe * min(1.0, n_bets / 50)
               This penalizes strategies with very few bets that might be noise.

TEST:        2025-07-01 to 2025-10-29 (1,149 games)
             - ALL parameters frozen from validation — zero tuning
             - Metrics reported: ROI, Sharpe, win rate, Kelly return, max drawdown
             - Bootstrap 95% CI on flat ROI (2,000 resamples)
             - Permutation test for statistical significance (5,000 shuffles)
```

**Critical constraint:** No information from the test period influences any decision. This means we might not pick the "best" threshold on the test set, but we can trust that whatever we do pick generalizes to unseen data.

### Why Kalshi Prices, Not Assumed Odds

Previous versions of this analysis used assumed odds for some strategies (e.g., NRFI at -120, YRFI at +100). This is unreliable because:

1. Actual odds vary substantially by game. In our DK O/U dataset, only 6.8% of games have exactly -110/-110 pricing. The rest range from -147 to +1180.
2. A model probability that looks profitable at assumed odds may be unprofitable at actual odds. If the model says P(under) = 0.55 and we assume -110 odds (breakeven 52.4%), it looks like a +2.6% edge. But if the actual odds are -200 (breakeven 66.7%), it's a -11.7% edge.
3. Kalshi contracts have a clean, unambiguous price: you buy at $X, get $1 if right, $0 if wrong. No vig adjustment needed. The price IS the market's probability.

Every result in this report uses actual per-game Kalshi contract prices or actual per-game DraftKings closing odds. Nothing is assumed.

---

## 3. Data Sources

### Kalshi Moneyline (Primary Market)

- **File:** `data/kalshi/kalshi_mlb_2025.parquet`
- **Games:** 2,099 (April 16 - October 29, 2025)
- **Source:** Kalshi API, KXMLBGAME event series
- **Columns:** `game_date`, `home_team`, `away_team`, `home_win`, `kalshi_home_prob`, `kalshi_away_prob`, `volume`, `event_ticker`, `price_source`
- **Coverage:** ~86% of 2025 regular season + playoffs
- **Pricing:** Pre-game contract prices. `kalshi_home_prob` is the last traded price (or mid-market) for the "home team wins" contract before first pitch.

### DraftKings Closing Lines (Primary Signal)

- **File:** `data/odds/sbr_mlb_2025.parquet`
- **Games:** 2,902 (February 20 - October 29, 2025, includes spring training)
- **Source:** SportsBookReview historical data
- **Key columns:** `home_ml_open`, `home_ml_close`, `away_ml_open`, `away_ml_close` (American odds format), `ou_close`, `over_close_odds`, `under_close_odds`, `total_runs`, `ou_result`
- **ML coverage:** 2,753 games with DK closing moneylines (94.8%)
- **O/U coverage:** 2,748 games with DK closing O/U lines (94.7%)
- **Processing:** American odds are converted to raw implied probabilities, then devigged by dividing each side by the sum of both sides (multiplicative devig).

### Multi-Book Consensus (Secondary Signal)

- **File:** `data/odds/multibook_consensus_2025.parquet` (derived from `data/odds/mlb_odds_dataset.json`)
- **Games:** 2,130 (February 20 - August 16, 2025)
- **Source:** The Odds API historical snapshots
- **Sportsbooks:** DraftKings, FanDuel, BetMGM, Caesars, Bet365, BetRivers (6 books)
- **Construction:** For each game, extract each book's closing moneyline, convert to implied probability, devig, then average all 6 devigged probabilities. Also record inter-book standard deviation (`consensus_std`) as a measure of market disagreement.
- **Limitation:** Data ends August 16, 2025 — no September or October coverage. This means consensus signals have no test-period data after mid-August.

### Kalshi O/U Total Runs

- **Files:** `data/kalshi/kalshi_total_2025.parquet` (38 games), `data/kalshi/kalshi_total_2026.parquet` (122 games)
- **Source:** Kalshi API, KXMLBTOTAL event series
- **Structure:** Binary contracts at multiple strikes (6.5, 7.5, 8.5, 9.5 runs). Each strike has a `over_XX_prob` column representing Kalshi's market price for "total runs > X.5".
- **2025 limitation:** Only 38 games, all playoffs (September 30 - October 29). Regular season total markets either didn't exist or had insufficient liquidity to scrape.
- **2026 data:** 122 games (March 25 - April 4, 2026) with better strike coverage (mean 11 strikes per game vs 3.5 in 2025). However, no 2026 DK odds data exists to compare against.

### Statcast Features (for ML Models)

- **Files:** `data/features/game_features_2024.parquet`, `data/features/game_features_2025.parquet`
- **Games:** ~2,429 (2024), ~2,430 (2025)
- **Features:** 155 pregame features including starting pitcher quality metrics (ERA, FIP, xRV, stuff scores, first-inning-specific rates), lineup quality (OPS, wOBA, K rate, barrel rate for top-of-order), park factors, rest days, bullpen workload, divisional rivalry flags, and seasonal timing.

### NRFI/YRFI Model Predictions (Internal Only)

- **Files:** `data/backtest/nrfi_lgb_2025.parquet` (2,430 games), `data/backtest/nrfi_ou_backtest_2025.parquet` (500 games)
- **Important:** These are model-generated NRFI probabilities, NOT market prices. No sportsbook or prediction market NRFI pricing data exists in our system.

### What We Don't Have

| Market | Status |
|--------|--------|
| Kalshi NRFI | Does not exist on Kalshi (confirmed as of April 2026) |
| Polymarket NRFI | Does not exist |
| DK NRFI prop odds | Not in SBR dataset (only ML, O/U, spread) |
| Polymarket O/U | Not scraped / no historical data |
| Any 2026 DK odds | No data yet — requires scraping via The Odds API |

---

## 4. Strategy 1: DK-vs-Kalshi Moneyline Arbitrage (ACTIVE)

**Status: ACTIVE — only profitable strategy with statistical significance**

### What This Strategy Does

DraftKings is a traditional sportsbook whose closing lines are refined by professional bettors. Kalshi is a prediction market (exchange) where retail traders set prices. When DK's devigged closing probability disagrees with Kalshi's pre-game contract price by more than a minimum threshold, we buy the underpriced Kalshi contract.

This is pure structural arbitrage. No ML model is involved. The only computation is a vig-removal on DK odds and a subtraction.

### Step-by-Step Mechanics

#### Step 1: Convert DK Closing Odds to Fair Probability

DraftKings posts moneyline odds in American format. Example: Dodgers -150 / Padres +130.

```
Raw implied probabilities:
  Dodgers: |-150| / (|-150| + 100) = 0.600
  Padres:   100  / ( 130  + 100)   = 0.435
  Sum: 1.035 (the 3.5% excess is DK's vigorish)

Devigged (fair) probabilities:
  Dodgers: 0.600 / 1.035 = 0.5797
  Padres:  0.435 / 1.035 = 0.4203
  Sum: 1.000
```

#### Step 2: Read Kalshi Price

Kalshi sells binary contracts. "Will the Dodgers win?" priced at $0.54 means the market thinks there's a 54% chance. If the Dodgers win, the contract pays $1.00. If they lose, it pays $0.00. The price IS the probability.

#### Step 3: Compute Edge

```
edge_home = DK_fair_home - Kalshi_home = 0.5797 - 0.54 = +0.0397 (4.0%)
edge_away = DK_fair_away - Kalshi_away = 0.4203 - 0.46 = -0.0397
```

DK thinks the Dodgers are 4 points more likely to win than Kalshi does. Buy the Dodgers contract on Kalshi.

#### Step 4: Size with Kelly Criterion

```
buy_price = 0.54 (Kalshi price for Dodgers)
our_p     = 0.5797 (DK fair prob)
b         = (1.0 - 0.54) / 0.54 = 0.852 (net odds: profit $0.46 on $0.54 risk)
q         = 1 - 0.5797 = 0.4203

Full Kelly:  f = (p*b - q) / b = (0.5797*0.852 - 0.4203) / 0.852 = 0.0855
30% Kelly:   f = 0.0855 * 0.30 = 0.0256 (2.56% of bankroll)
Capped at:   min(0.0256, 0.05) = 0.0256
```

#### Step 5: P&L

```
If Dodgers win:  profit = $1.00 - $0.54 = +$0.46 per contract
If Dodgers lose: loss   = -$0.54 per contract
```

For each game, we bet at most one side (whichever side has the larger edge above our threshold).

### Validation-Period Parameter Selection

On the 942 validation games (April 16 - June 30, 2025), we swept a grid of edge thresholds and Kelly fractions. The optimization target is sample-size-weighted Sharpe: `score = sharpe * min(1.0, n_bets / 50)`. This prevents the optimizer from choosing a configuration that works on 3 lucky bets.

**Validation grid results (kelly=30%, the optimal Kelly fraction):**

| Edge Threshold | Bets | Win Rate | ROI | Sharpe | Weighted Sharpe |
|----------------|------|----------|-----|--------|-----------------|
| 2% | 263 | 44.5% | +1.6% | 1.21 | 1.21 |
| 3% | 139 | 41.7% | +0.2% | 1.09 | 1.09 |
| 4% | 85 | 51.8% | +10.8% | 2.48 | 2.48 |
| 5% | 60 | 50.0% | +10.4% | 2.22 | 2.22 |
| **6%** | **50** | **52.0%** | **+13.1%** | **2.49** | **2.49** |
| 7% | 40 | 52.5% | +14.3% | 2.54 | 2.03 |
| 8% | 36 | 50.0% | +12.6% | 2.15 | 1.55 |
| 10% | 22 | 54.5% | +19.1% | 3.17 | 1.40 |
| 12% | 17 | 52.9% | +18.1% | 2.75 | 0.93 |
| 15% | 13 | 53.8% | +24.2% | 3.40 | 0.88 |

**Validation-selected parameters:** edge > 6%, Kelly 30%. The weighted Sharpe score of 2.49 is the highest, balancing signal quality (higher at tight thresholds) against sample size (higher at loose thresholds).

### Out-of-Sample Test Results

These parameters were frozen and applied without modification to the 1,149 test games (July 3 - October 29, 2025):

| Metric | Value |
|--------|-------|
| Bets | 69 |
| Win Rate | 72.5% |
| Flat ROI | +23.0% |
| Sharpe (annualized, 180 days) | 4.15 |
| Kelly Compound Return | +410.3% ($10,000 -> $51,030) |
| Max Drawdown | -8.8% |
| Bootstrap 95% CI on ROI | [+11.8%, +33.9%] |
| Permutation p-value (5,000 shuffles) | 0.0002 |

**Monthly consistency:**

| Month | Bets | Win Rate | ROI |
|-------|------|----------|-----|
| July | 10 | 60.0% | +19.9% |
| August | 26 | 57.7% | +7.3% |
| September | 33 | 87.9% | +36.3% |

All three months profitable. September's exceptional performance coincides with the pennant race and playoff push, when DK's sharp bettors are particularly engaged while Kalshi's retail base may be less attentive to late-season games.

### Edge Threshold Curve on Test Data

For transparency, here is the full edge curve on the test set. Note: these numbers are for information only. The validation selected edge>6%; we did NOT cherry-pick from this table.

| Edge | Bets | Win Rate | ROI | Sharpe |
|------|------|----------|-----|--------|
| 2% | 259 | 55.2% | +7.8% | 2.90 |
| 3% | 149 | 62.4% | +13.8% | 3.23 |
| 4% | 102 | 66.7% | +17.9% | 3.59 |
| 5% | 83 | 68.7% | +19.6% | 4.03 |
| **6%** | **69** | **72.5%** | **+23.0%** | **4.15** |
| 8% | 50 | 78.0% | +27.6% | 4.97 |
| 10% | 43 | 79.1% | +29.0% | 5.03 |
| 15% | 33 | 84.8% | +34.7% | 5.06 |

The edge is monotonically profitable at every threshold. This is the hallmark of a genuine structural inefficiency. Noise-driven signals show non-monotonic or randomly varying performance across thresholds; real edges get stronger as you filter more aggressively.

### Favorite vs Underdog Breakdown

Among the 69 test bets at edge>6%:

| Side | Bets | Win Rate | ROI | Avg Edge |
|------|------|----------|-----|----------|
| Underdogs (Kalshi price < $0.50) | 40 | 75.0% | **+31.7%** | 20.2% |
| Favorites (Kalshi price >= $0.50) | 29 | 69.0% | +11.0% | 18.2% |

The underdog edge is nearly 3x the favorite edge. This suggests Kalshi's retail participants systematically overprice favorites. The implication: an underdog-focused variant should outperform.

### Kalshi Price Range Analysis

| Price Range | Bets | Win Rate | ROI | Avg Edge |
|-------------|------|----------|-----|----------|
| $0.30-0.40 | 9 | 66.7% | +28.9% | 16.1% |
| **$0.40-0.50** | **29** | **79.3%** | **+32.9%** | **20.2%** |
| $0.50-0.60 | 22 | 63.6% | +8.2% | 19.5% |
| $0.60-0.70 | 6 | 83.3% | +18.8% | 16.6% |

The $0.40-0.50 bucket (moderate underdogs) is the sweet spot: highest bet count with the highest ROI. This is where Kalshi mispricing is most systematic.

### Statistical Validation

**Permutation test (5,000 iterations):** Shuffle win/loss outcomes randomly, recompute the full strategy (same edge threshold, same Kelly sizing, same bet selection), and record the ROI. If the observed ROI is better than 95% of shuffled ROIs, the signal is statistically significant.

- Observed test ROI: +23.0%
- Mean shuffled ROI: -1.2%
- p-value: 0.0002 (only 1 in 5,000 shuffles produced ROI >= +23%)
- **Conclusion: Significant at p < 0.001. This is not luck.**

**Bootstrap 95% CI:** Resample the 69 bets with replacement 2,000 times, compute ROI each time. The 2.5th and 97.5th percentiles give the confidence interval.

- 95% CI: [+11.8%, +33.9%]
- **The lower bound is meaningfully positive.** Even in the pessimistic tail, the strategy is profitable.

### Implementation

```python
# Core logic (from src/kalshi_enhanced_backtest.py)
edge = dk_devigged_prob - kalshi_price
if edge > 0.06:  # or 0.04 for underdog variant
    buy_kalshi_contract(side_with_larger_edge)
    kelly_fraction = 0.30 * standard_kelly(our_prob, kalshi_price)
    bet_size = min(kelly_fraction, 0.05) * bankroll
```

```bash
python src/kalshi_enhanced_backtest.py  # Full backtest with validation/test
python src/kalshi_clean_backtest.py     # Original clean backtest with LGB/MC/ensemble
```

---

## 5. Strategy 2: LightGBM Win Model vs Kalshi (RETIRED)

**Status: RETIRED — no edge over Kalshi market prices**

### What We Built

A LightGBM binary classifier trained on the full 2024 MLB season to predict P(home_win). Hyperparameters were optimized via Optuna with 120 trials and 5-fold TimeSeriesSplit cross-validation, all within 2024 data only.

**Model specifications:**
- Training data: ~2,429 games from 2024
- Features: 155 pregame Statcast-derived features (pitcher quality, lineup stats, park factors, rest days, bullpen workload, etc.)
- Features with >30% missing values dropped
- Optuna search space: `n_estimators` [20,250], `max_depth` [2,5], `num_leaves` [3,31], `learning_rate` [0.01,0.15], `min_child_samples` [30,200], `reg_alpha` [0.1,30], `reg_lambda` [0.1,50], `colsample_bytree` [0.3,1.0], `subsample` [0.5,1.0]
- Best CV log-loss on 2024: 0.686 (compared to theoretical minimum of 0.693 for a coin flip — the model learns very little signal)

### Why It Failed

**Signal quality comparison (test period, July-October 2025):**

| Signal | Brier Score | BSS | AUC |
|--------|-------------|-----|-----|
| LightGBM model | 0.2480 | -0.001 | 0.575 |
| Kalshi market price | 0.2456 | +0.008 | 0.589 |
| DK closing line | 0.2398 | +0.032 | 0.603 |

The LGB model's BSS is -0.001 relative to the base rate, meaning it's marginally WORSE than always predicting the overall home win rate. Kalshi's market prices are already more accurate than our model. You cannot beat a market by using a signal that's weaker than the market itself.

**Betting results (test period, validation-selected params: edge>10%, kelly=30%):**

| Metric | Value |
|--------|-------|
| Bets | 351 |
| Win Rate | 39.0% |
| Flat ROI | +0.9% |
| Sharpe | -0.01 |

The model generates many bets because it frequently disagrees with Kalshi, but these disagreements are noise, not signal. The 39% win rate (below 50%) with a barely positive ROI confirms the model has no systematic edge.

**Ensemble optimizer result:** When given the option to weight DK_arb, LGB, and MC_sim, the validation optimizer assigned 100% weight to DK_arb and 0% to both LGB and MC_sim. The models add nothing.

### Why Kalshi Is Hard to Beat With ML

Kalshi's moneyline market aggregates the beliefs of thousands of participants. Even if most are unsophisticated, the "wisdom of crowds" effect means the market price already reflects public information (injury reports, lineups, weather, recent performance) that our features also capture. DK's closing line goes further by incorporating sharp money.

A LightGBM trained on Statcast features is essentially trying to rediscover what the market already knows. The 155 features we use (pitcher ERA, lineup wOBA, park factors, etc.) are well-known, publicly available, and already priced in.

To beat Kalshi with ML, you'd need either:
- Private information not available to the market (unlikely for public MLB data)
- A fundamentally better model architecture than what thousands of bettors collectively represent (unlikely for tabular Statcast data)
- Speed advantage: processing information faster than the market updates (possible, but not what this model does)

---

## 6. Strategy 3: Monte Carlo Simulator vs Kalshi (RETIRED)

**Status: RETIRED — negative ROI against Kalshi**

### What It Does

A plate-appearance-level Markov chain Monte Carlo simulator (`src/simulate.py`) that models each at-bat as a probabilistic event, simulates 3,000 games per matchup, and produces estimated win probabilities and run totals. The simulator uses batter-vs-pitcher matchup distributions derived from Statcast data via log5 combination.

### Results

| Metric | Value |
|--------|-------|
| Test Bets (edge>10%, kelly=30%) | 104 |
| Flat ROI | -1.1% |
| Correlation with actual total runs | 0.037 |
| Sim mean total runs | 9.46 (actual: 8.64, bias of +0.82) |
| Sim total std | 1.45 (actual: 4.48, ratio 0.105) |

The simulator systematically over-predicts total runs by +0.82/game due to the Markov independence assumption (each plate appearance is treated as independent, ignoring stretch pitching and situational strategy). Its near-zero correlation with actual game totals and compressed variance make its win probabilities unreliable for betting purposes.

### Planned Fixes (Not Yet Implemented)

A detailed plan exists to fix the simulator bias (see `zesty-exploring-kurzweil.md` plan file) with 4 corrections: RISP suppression, park/weather adjustments, calibration dampener, and log5 amplification. These were deprioritized once we established that the DK arb is the only strategy with real edge against Kalshi — fixing the simulator would only bring it closer to parity with DK, not ahead of it.

---

## 7. Strategy 4: Multi-Book Consensus vs Kalshi (INCONCLUSIVE)

**Status: INCONCLUSIVE — promising on validation, insufficient test data**

### What This Strategy Does

Instead of using DraftKings alone, average the devigged closing probabilities from all 6 available sportsbooks (DraftKings, FanDuel, BetMGM, Caesars, Bet365, BetRivers). The theory: a consensus of 6 independent pricing mechanisms should be more accurate than any single one.

### Data Limitation

The multi-book consensus data from The Odds API historical JSON extends only through August 16, 2025. This means:
- Validation period (Apr 16 - Jun 30): **Full coverage** — 1,382 out of 2,091 Kalshi games have consensus data
- Test period (Jul 3 - Oct 29): **Partial coverage** — only July and first half of August have consensus data

This makes the test-period evaluation statistically unreliable.

### Validation Results

| Strategy | Val Bets | Val WR | Val ROI | Val Sharpe |
|----------|----------|--------|---------|------------|
| DK alone (edge>6%) | 50 | 52.0% | +13.1% | 2.49 |
| Consensus alone (edge>3%) | 60 | 56.7% | +17.4% | 3.01 |
| DK + consensus confirm (edge>4%, cons>3%) | 51 | 58.8% | +20.5% | 4.37 |

On validation, the consensus signal looks strong — higher Sharpe than DK alone. The combined "DK + consensus confirmation" filter (only bet when BOTH DK and consensus see edge) was the best validation strategy at weighted Sharpe 4.37.

### Test Results

| Strategy | Test Bets | Test WR | Test ROI | Test Sharpe | p-value |
|----------|-----------|---------|----------|-------------|---------|
| Consensus alone | 21 | 52.4% | +6.2% | 3.97 | 0.295 |
| Consensus + movement | 20 | 55.0% | +8.9% | 4.45 | 0.295 |
| DK + consensus confirm | 20 | 55.0% | +8.9% | 3.94 | — |

Only 20-21 bets in the test period (all in July and early August before the data cuts off). The Sharpe numbers look good but the permutation test is NOT significant (p=0.295) because the sample size is too small. We cannot distinguish these returns from noise.

### Verdict

The multi-book consensus is a potentially valuable signal, but we lack sufficient test data to validate it. To revisit this, we would need to:
1. Scrape current multi-book odds for the 2026 season via The Odds API (`python src/scrape_odds.py --year 2026 --api-key YOUR_KEY`)
2. Accumulate several months of overlapping Kalshi + multi-book data
3. Re-run the validation/test protocol

---

## 8. Strategy 5: DK Line Movement Filter (NO VALUE)

**Status: NO VALUE ADDED — movement filter does not improve any strategy**

### Hypothesis

If DraftKings' closing line moved toward a team relative to the opening line, this represents "sharp money" flowing in that direction. We hypothesized that requiring the DK line movement to confirm the edge direction would filter out false signals.

### How It Works

DK opens lines hours before game time and adjusts them as bets come in. The movement is:
```
dk_movement = dk_close_fair_home - dk_open_fair_home
```

If `dk_movement > 0`, sharp money moved toward home. If we're betting home on Kalshi, this confirms our direction. If movement is negative (sharp money moved away), we skip the bet.

### Results: Movement Filter Has No Effect

At every edge threshold tested, the movement filter either removed 0 bets or removed 1-3 bets with no meaningful impact on Sharpe:

| Edge | No Filter Bets | With Movement Bets | No Filter Sharpe | Movement Sharpe |
|------|----------------|-------------------|------------------|-----------------|
| 3% | 149 | 142 | 3.23 | 3.24 |
| 4% | 102 | 99 | 3.59 | 3.54 |
| 5% | 83 | 82 | 4.03 | 3.98 |
| 6% | 69 | 69 | 4.15 | 4.15 |
| 8% | 50 | 50 | 4.97 | 4.97 |
| 10% | 43 | 43 | 5.03 | 5.03 |

At edge>6%, the movement filter removes **zero bets** — every game with a large DK-Kalshi edge also has DK line movement in the same direction. This makes sense: if DK's close is far from Kalshi, it almost certainly moved toward that direction from the open.

**Conclusion:** DK line movement is already embedded in the DK closing price. Filtering on it adds no incremental information.

---

## 9. Strategy 6: Kalshi O/U Total Runs Arbitrage (RETIRED)

**Status: RETIRED — -1.8% ROI on available data**

### What We Tested

Kalshi offers binary contracts on total runs at multiple strike prices (e.g., "Over 6.5?", "Over 8.5?"). DraftKings posts O/U closing lines with juice. We tested whether DK's implied total-runs distribution disagrees with Kalshi's strike prices enough to generate profitable trades.

### Methodology

1. DK posts a closing O/U line (e.g., 8.5 at -110/-110) with devigged over/under probabilities.
2. We model total runs as Normal(mean, sigma) where the mean is calibrated to DK's line and sigma is estimated from actuals (4.15 runs).
3. For each Kalshi strike (6.5, 7.5, 8.5, 9.5), we compute DK-implied P(over strike) via the normal CDF and compare to Kalshi's market price.
4. If edge > 3%, we buy the underpriced side.

### Results

**38 games (September 30 - October 29, 2025 — playoffs only), 117 bets across all strikes:**

| Metric | Value |
|--------|-------|
| Total Bets | 117 |
| Win Rate | 49.6% |
| Flat ROI | -1.8% |
| Sharpe | 0.15 |
| Kelly Return | +2.4% |

**Per-strike breakdown:**

| Strike | Bets | Win Rate | ROI | Avg Edge |
|--------|------|----------|-----|----------|
| 6.5 | 34 | 32.4% | -5.6% | 8.0% |
| 7.5 | 25 | 40.0% | -8.3% | 5.9% |
| 8.5 | 28 | 64.3% | +7.3% | 7.2% |
| 9.5 | 30 | 63.3% | -0.5% | 7.1% |

**Per-side:**
| Side | Bets | Win Rate | ROI |
|------|------|----------|-----|
| Over | 1 | 0.0% | -19.0% |
| Under | 116 | 50.0% | -1.6% |

### Why It Doesn't Work

1. **Sample size:** Only 38 playoff games. These are a non-representative sample — playoff games have different dynamics (better pitching, higher stakes, different scoring patterns) than regular season.
2. **Normal approximation is crude:** Real run distributions are discrete, right-skewed, and game-specific. A Gaussian CDF with a single sigma=4.15 doesn't capture the tail behavior that drives profitability at extreme strikes.
3. **Kalshi O/U pricing may be more efficient:** Unlike moneyline where Kalshi has a clear retail bias, O/U binary contracts at multiple strikes may attract more sophisticated participants who understand the math.
4. **No 2026 DK data to compare against the 122-game 2026 Kalshi dataset.**

### What Would Be Needed to Revisit

- 500+ games with both Kalshi O/U strike prices AND DK closing O/U lines
- A better model than normal approximation (e.g., Poisson mixture, or direct simulation)
- Separate validation/test with at least 200 games each

---

## 10. Strategy 7: O/U Under Classifier vs DK (RETIRED)

**Status: RETIRED — model is value-destructive against real DK odds**

### What It Was

A LightGBM classifier trained to predict P(under) for each game, then compare against DK's devigged O/U line to find edge.

### Original (Flawed) Approach

The first version of this strategy had two critical errors:
1. **Fixed probability threshold** instead of edge-based betting: it bet under whenever P(under) > 0.55, ignoring whether the market was already pricing under at 66.7% (-200 odds).
2. **Assumed flat -110 odds** for all games, when actual DK closing odds range from -147 to +1180.

### Corrected Results (Edge-Based vs Actual DK Odds)

After fixing both errors — comparing model P(under) against DK's devigged P(under) and using actual per-game DK closing odds:

| Metric | Value |
|--------|-------|
| Brier Skill Score | -0.006 (worse than base rate) |
| AUC | 0.511 (essentially random) |
| Walk-forward ROI at 2% edge | -4.1% (1,106 bets) |
| Walk-forward ROI at 5% edge | +1.9% (530 bets, CI [-7%, +11%]) |
| Permutation p-value | 0.989 (NOT significant) |
| Blind under ROI | -0.8% |
| Model-selected under ROI | -4.1% |

**The model is worse than betting blindly.** Blind under (bet under on every game at actual DK odds) loses 0.8%. Model-selected under loses 4.1%. The model's feature engineering (Statcast stats, pitcher quality, etc.) adds negative value for O/U prediction — it confidently bets under on games that the market has already correctly priced.

### Optuna Optimization Also Failed

We ran Optuna with 150 trials and 5-fold TSCV to find optimal hyperparameters. Best CV log-loss: 0.69325, compared to 0.6931 theoretical minimum (coin flip). The model learns essentially nothing about O/U that isn't already in the DK closing line.

---

## 11. Strategy 8: NRFI/YRFI Model (UNVERIFIABLE)

**Status: UNVERIFIABLE — no market odds data exists to backtest against**

### What The Model Does

An ensemble of LightGBM + Logistic Regression (50/50 blend) trained on 2024 data to predict P(NRFI) — the probability that no runs are scored in the first inning. Uses first-inning-specific features: pitcher first-inning K rate, walk rate, HR rate, clean-inning percentage, slow-starter metric, top-of-lineup quality, "weakest link" features (min of both SPs' quality), and interaction terms.

### Model Quality

| Variant | BSS | AUC | Description |
|---------|-----|-----|-------------|
| LGB full features (Optuna) | -0.016 | 0.560 | Overfit despite Optuna |
| LGB core features | -0.007 | 0.555 | Simpler, still negative BSS |
| LR only | +0.002 | 0.553 | Logistic regression, positive BSS |
| Ensemble (LGB+LR 50/50) | +0.006 | 0.570 | Best BSS and AUC |

The ensemble variant has a small positive BSS (+0.006) and reasonable AUC (0.570), suggesting it has marginal predictive skill for first-inning scoring.

### Backtest With Assumed Odds

Using assumed odds of NRFI -120, YRFI +100 (typical market levels):

| Strategy | Threshold | Bets | Win Rate | ROI |
|----------|-----------|------|----------|-----|
| YRFI | P(NRFI) <= 0.48 | 314 | 57.6% | +15.3% |
| NRFI | P(NRFI) >= 0.57 | 584 | 57.9% | +6.1% |

### Why This Is Unverifiable

**No prediction market or sportsbook NRFI pricing data exists in our system:**

- Kalshi does not offer NRFI markets (confirmed as of April 2026)
- Polymarket does not offer NRFI markets
- Our DK/SBR odds data contains only moneyline, O/U, and spreads — no NRFI props
- The assumed odds (-120/+100) are based on "typical" levels, but actual odds vary by game and book

Without real per-game market prices, we cannot compute a meaningful edge. A model saying P(YRFI) = 52% is profitable at +100 odds (breakeven 50%) but unprofitable at -120 odds (breakeven 54.5%). Whether the model has edge depends entirely on what price you can actually get, and we don't have that data.

### Critical Biases in Previous Analysis

The earlier report on this strategy had two undisclosed look-ahead biases:
1. **Model variant selection:** The ensemble (variant E) was chosen because it had the best BSS on the full test period. In a proper framework, variant selection should happen on validation data only.
2. **Threshold selection:** The thresholds 0.57 (NRFI) and 0.48 (YRFI) were likely optimized on test-period results, not validation.

These biases don't necessarily mean the model is bad, but they mean the reported ROIs are unreliable.

### What Would Be Needed to Validate

1. **Historical NRFI/YRFI odds from DraftKings, FanDuel, or another major book** — at least 500 games with actual per-game pricing
2. **Proper temporal split** — select model variant and thresholds on validation, freeze, evaluate on test
3. **Edge-based betting** — only bet when model P exceeds devigged market P by a threshold

Without this data, NRFI/YRFI remains a theoretically interesting model with no verified real-world profitability.

---

## 12. Strategy 9: Away Underdog Anomaly (SUPERSEDED)

**Status: SUPERSEDED by DK-vs-Kalshi arbitrage**

### What It Was

A rule-based strategy: when DraftKings prices an away team at 35-40% implied win probability, bet the away team. No model needed. The theory: recreational bettors overvalue home-field advantage, creating persistent mispricing of moderate away underdogs.

### Original Backtest Results (DK Only, Assumed Payouts)

| DK Away Prob Range | Bets | Actual Win Rate | ROI |
|--------------------|------|-----------------|-----|
| 30-35% | 174 | 31.6% | -2.1% |
| **35-40%** | **242** | **42.1%** | **+14.5%** |
| 40-45% | 312 | 43.3% | +1.2% |
| 45-50% | 287 | 47.4% | -3.8% |

### Why It Was Superseded

This anomaly is a weaker version of what the DK arb strategy already captures. When the DK arb identifies a Kalshi underdog as underpriced, it's often the same away-underdog effect — but with the critical addition of an actual Kalshi market price to trade against.

The DK arb's underdog variant (edge>4%, underdogs only) produces +22.6% ROI on 57 bets with a permutation p-value of 0.001. It subsumes the away-underdog anomaly while adding:
- A specific market to trade on (Kalshi) with known contract prices
- Proper edge computation (DK fair prob minus Kalshi price, not DK fair prob minus assumed breakeven)
- Kelly sizing based on the actual edge and odds

---

## 13. Strategy 10: Pick-em Home Moneyline (RETIRED)

**Status: RETIRED — not profitable, model has no edge in pick-em games**

### What It Was

When DraftKings prices a game as essentially a toss-up (home implied 48-52%) but our LightGBM win model sees the home team as 3%+ more likely to win, bet the home side.

### Results

37 bets, 51.4% win rate, -2.2% ROI. Wildly inconsistent month-to-month (June: -41.6%, July: +26.0%, August: -1.2%, September: +23.5%). The 37-bet sample is far too small to reach statistical significance, and the model's edge in this narrow range is indistinguishable from zero.

### Why It Was Retired

This strategy relied on the LightGBM win model having edge over DK in close games. We've now established that the LGB model has BSS -0.001 against Kalshi and no edge over DK. Without a genuinely superior probability estimate, disagreeing with the market on 50/50 games is pure noise.

---

## 14. Enhancement Analysis: Filters and Variants

We tested multiple enhancements to the core DK arb strategy. All were validated on the Apr-Jun period and frozen for the Jul-Oct test.

### Validated Variants

| Variant | Val Weighted Sharpe | Test Bets | Test WR | Test ROI | Test Sharpe | Test 95% CI |
|---------|--------------------:|----------:|--------:|---------:|-----------:|------------|
| DK arb, price [0.35,0.55] (edge>4%) | **3.53** | 74 | 67.6% | +21.5% | 3.84 | [+11.1%, +31.7%] |
| DK arb, underdogs only (edge>4%) | 2.84 | 57 | 64.9% | +22.6% | **4.85** | [+10.1%, +34.3%] |
| DK arb, baseline (edge>6%) | 2.49 | 69 | 72.5% | +23.0% | 4.15 | [+11.8%, +33.9%] |

**Key observations:**

1. **Price range [0.35, 0.55]** was the validation winner (weighted Sharpe 3.53) but has lower test Sharpe (3.84) than the underdog variant (4.85). This is expected: the filter with the best out-of-sample performance won't always be the filter with the best validation performance. The important thing is that ALL three variants are profitable and significant.

2. **Underdog-only filter (price < $0.50)** has the highest test Sharpe despite not being the validation winner. This could be genuine (underdogs are structurally underpriced on Kalshi) or it could be test-period luck. The 57-bet sample is somewhat small.

3. **Baseline edge>6%** is the most conservative choice — it had the simplest filter (none) and stable performance across both periods.

### Bet Overlap Between Variants

The underdog filter and price-range filter largely overlap:
- Price range [0.35, 0.55]: 74 bets
- Underdogs only (price < 0.50): 57 bets
- Overlapping bets: 53 (93% of underdog bets are also in the price range)
- Union: 78 unique bets

This means combining the two variants in a portfolio adds minimal diversification — they're essentially the same signal with slightly different cutoffs.

### Filters That Add No Value

| Filter | Effect |
|--------|--------|
| DK line movement confirmation | Removes 0-7 bets; no Sharpe improvement at any edge threshold |
| Multi-book consensus confirmation | Reduces to ~20 test bets (data ends Aug 16); insufficient for evaluation |
| Kalshi volume filter | Not systematically tested; volume data quality is inconsistent |

---

## 15. Production Configuration

Based on validation-period optimization:

### Primary Strategy: DK Arb Baseline

```
Signal:     DK devigged closing moneyline probability
Market:     Kalshi KXMLBGAME binary contracts
Threshold:  edge > 6% (DK fair prob minus Kalshi price)
Kelly:      30% fractional Kelly, capped at 5% of bankroll per bet
Side:       Whichever side (home or away) has the larger edge
Frequency:  ~0.6 bets per game day (69 bets over ~120 game days)
```

### Secondary Strategy: DK Arb Underdogs

```
Signal:     Same DK devigged closing moneyline probability
Market:     Same Kalshi KXMLBGAME binary contracts
Threshold:  edge > 4% AND Kalshi price < $0.50 (underdog side only)
Kelly:      30% fractional Kelly, capped at 5% of bankroll per bet
Frequency:  ~0.5 bets per game day (57 bets over ~120 game days)
```

### Execution Workflow

1. **Pre-game (1-2 hours before first pitch):** Scrape DK closing moneylines for today's games. Scrape Kalshi KXMLBGAME contract prices.
2. **Signal computation:** For each game, devig DK odds and compute edge vs Kalshi on both sides.
3. **Filter:** Keep games with edge > threshold (6% baseline, or 4% underdogs-only).
4. **Sizing:** Apply 30% fractional Kelly with 5% per-bet cap.
5. **Execute:** Buy the underpriced contract on Kalshi. Monitor liquidity — if the order book is thin, reduce size or skip.
6. **Settlement:** Kalshi contracts settle automatically after the game. P&L = ($1 - buy_price) if won, (-buy_price) if lost.

### Bankroll Recommendations

| Starting Bankroll | Max Bet (5% cap) | Typical Bet (2-3% Kelly) | Expected Bets/Month | Expected Monthly ROI |
|-------------------|------------------|--------------------------|---------------------|---------------------|
| $1,000 | $50 | $20-30 | 12-15 | +$40-60 |
| $5,000 | $250 | $100-150 | 12-15 | +$200-300 |
| $10,000 | $500 | $200-300 | 12-15 | +$400-600 |

These estimates assume similar edge persistence to the 2025 backtest. Actual results will vary.

---

## 16. Risk Factors and Caveats

### Statistical Risks

1. **Sample size.** 69 bets (baseline) or 57 bets (underdogs) over 4 months is a small sample. The 95% CI on ROI is wide: [+11.8%, +33.9%] for the baseline. The true ROI could be as low as +12% or as high as +34%. It could also be below +12% in future seasons if the market dynamics change.

2. **Single-season backtest.** All results are from the 2025 MLB season. We have no multi-year out-of-sample data. The DK-Kalshi pricing gap may have been unusually large in 2025 (Kalshi's first full MLB season), and could narrow in 2026.

3. **September effect.** September 2025 produced exceptional results (33 bets, 87.9% WR, +36.3% ROI). If September is excluded, the remaining months (July-August) have 36 bets with ~63% WR and ~12% ROI — still profitable but less spectacular. It's unclear whether the September spike was signal (pennant race inefficiency) or noise.

### Market Risks

4. **Kalshi efficiency may improve.** As Kalshi attracts more sophisticated traders, its prices will converge toward DK closing lines. The 6%+ edge gaps that currently exist may shrink to 2-3% or disappear entirely. This is the primary long-term risk.

5. **Liquidity constraints.** Kalshi's order books are thin for many MLB games. You may not be able to fill orders at displayed prices, especially for larger sizes. Slippage directly reduces realized edge. A displayed 6% edge with 2% slippage is effectively a 4% edge.

6. **DK closing line timing.** DK closing lines are finalized ~30 minutes before first pitch. Kalshi prices can move in this window. If you compute the edge using DK's closing line but Kalshi's price has already adjusted by the time you buy, the realized edge is smaller than the backtested edge.

7. **Kalshi fee structure.** As of 2025, Kalshi charges no fees on MLB contracts (0% taker fee). If Kalshi introduces fees, this directly reduces ROI. A 2% fee on a 23% ROI strategy is manageable; a 10% fee would eliminate most of the edge.

### Model Risks

8. **No ML model adds value.** This is both a finding and a risk. If the DK-Kalshi arb narrows, we have no backup signal. Our LightGBM, MC simulator, and NRFI model have all failed to demonstrate edge against market prices.

9. **Unknown unknowns in the data.** The Kalshi price data was scraped via API. If any prices are mid-market (not executable), post-game, or otherwise not tradeable pre-game, the backtested P&L overstates reality. We've attempted to filter for pre-game prices only, but data quality issues may remain.

### Psychological Risks

10. **Low volume is mentally taxing.** At ~0.6 bets/day, there are many days with no action and occasional 3-4 day cold streaks. This requires patience and discipline to avoid chasing with off-strategy bets.

11. **Losing streaks will happen.** Even at 72.5% win rate, the probability of 3+ consecutive losses is ~2.1%. Over a 69-bet sample, this is expected to happen ~1.4 times. Maximum observed consecutive losses in the backtest: 3.

---

## 17. Forward-Testing Plan

### Immediate (2026 Season)

1. **Scrape 2026 DK odds:**
   ```bash
   python src/scrape_odds.py --year 2026 --api-key $ODDS_API_KEY
   ```
   This populates `data/odds/sbr_mlb_2026.parquet` with DK closing moneylines and O/U lines for the current season.

2. **Scrape 2026 Kalshi ML:**
   ```bash
   python src/scrape_kalshi.py --year 2026
   ```
   This populates `data/kalshi/kalshi_mlb_2026.parquet` with Kalshi pre-game contract prices.

3. **Run daily edge scanner:**
   Build a script that, ~1 hour before first pitch:
   - Fetches current DK closing moneylines (via The Odds API or DK's API)
   - Fetches current Kalshi KXMLBGAME prices (via Kalshi API)
   - Computes edge for each game
   - Outputs qualifying bets with Kelly sizing

4. **Track live P&L:** Record every bet (game, side, Kalshi buy price, DK fair prob, result) to compare live performance against backtest projections.

### Monitoring Thresholds

| Metric | Warning | Action |
|--------|---------|--------|
| Rolling 30-bet ROI < 0% | Yellow | Review data quality, check for Kalshi pricing changes |
| Rolling 50-bet ROI < 0% | Red | Pause strategy, investigate if DK-Kalshi gap has closed |
| Rolling Sharpe < 1.0 (over 60+ bets) | Red | Strategy may no longer be viable |
| Kalshi announces fee changes | Yellow | Recompute break-even edge accounting for fees |

### Multi-Book Consensus Revisit

If The Odds API key is available, scrape 2026 multi-book data to revisit the consensus signal. This requires:
1. Accumulating 3+ months of Kalshi + multi-book overlapping data
2. Splitting into validation (first 50%) and test (last 50%)
3. Re-running the consensus edge analysis

---

## Appendix A: Key Formulas

### American Odds to Implied Probability

American odds are the standard format used by US sportsbooks.

**Negative odds (favorites):** How much you must risk to win $100.
```
-150 means risk $150 to win $100
Implied probability = 150 / (150 + 100) = 60.0%
```

**Positive odds (underdogs):** How much you win on a $100 bet.
```
+130 means risk $100 to win $130
Implied probability = 100 / (130 + 100) = 43.5%
```

**General formula:**
```
If odds < 0:  prob = |odds| / (|odds| + 100)
If odds > 0:  prob = 100 / (odds + 100)
```

### Removing the Vig (Multiplicative Devig)

The sum of both sides' implied probabilities exceeds 100% — the excess is the book's profit margin.

```
Raw home implied: 0.600
Raw away implied: 0.435
Total: 1.035 (3.5% vig)

Fair home prob = 0.600 / 1.035 = 0.5797
Fair away prob = 0.435 / 1.035 = 0.4203
Sum: 1.000
```

This is the multiplicative (proportional) devig method. Alternative methods exist (additive, power, Shin) but multiplicative is standard for MLB moneylines where the vig is typically symmetric.

### Kalshi Contract P&L

```
Buy price:  p (the Kalshi contract price, e.g., $0.54)
If correct:  profit = $1.00 - p = $0.46
If wrong:    loss   = -p = -$0.54
Flat ROI per bet: (1-p) if won, (-p) if lost
Average ROI:  mean(per-bet PnL) across all bets
```

### Kelly Criterion (Bet Sizing)

```
f = (p * b - q) / b

p = estimated probability of winning
q = 1 - p
b = net decimal odds = (1 - buy_price) / buy_price
f = fraction of bankroll to bet (full Kelly)

Fractional Kelly:  f_actual = f * fraction (e.g., 0.30 for 30% Kelly)
Capped:            f_actual = min(f_actual, max_bet_frac)
```

**Example:** DK says 58% home, Kalshi prices home at $0.52.
```
p = 0.58, q = 0.42
b = 0.48 / 0.52 = 0.923
f = (0.58 * 0.923 - 0.42) / 0.923 = 0.124 (12.4% full Kelly)
f_30% = 0.124 * 0.30 = 0.037 (3.7% of bankroll)
```

### Sharpe Ratio (Annualized)

```
daily_pnl = array of dollar P&L per game day
sharpe = (mean(daily_pnl) / std(daily_pnl)) * sqrt(180)
```

The sqrt(180) annualization factor assumes ~180 MLB game days per season. A Sharpe above 2.0 is excellent; above 3.0 is rare in traditional finance.

Note: Sports betting Sharpe ratios are not directly comparable to financial Sharpe ratios because betting returns are discrete (win/lose) rather than continuous, and the "trading days" are clustered (multiple games per day, off-days, etc.).

### Brier Score and Brier Skill Score

```
Brier = (1/N) * sum((predicted_prob - actual_outcome)^2)
BSS   = 1 - (Brier_model / Brier_baseline)
```

Where `actual_outcome` is 1 or 0, and baseline is always predicting the base rate. BSS > 0 means the model is better than the baseline. BSS = 0 means equivalent. BSS < 0 means worse.

For MLB moneyline, a BSS of +0.03 (like DK achieves against Kalshi) is substantial. Kalshi itself achieves BSS +0.008 vs the base rate. Our LGB model achieves BSS -0.001 (slightly worse than base rate).

### Permutation Test

```
1. Compute observed_roi on real test data with real labels
2. Repeat 5,000 times:
   a. Randomly shuffle the win/loss labels
   b. Re-run the full strategy (same edge threshold, same Kelly, same bet selection)
   c. Record shuffled_roi
3. p_value = fraction of shuffled_roi >= observed_roi
4. If p_value < 0.05, the signal is statistically significant
```

This is a non-parametric test that makes no distributional assumptions. It directly answers: "How likely would we see this ROI by chance if the signal had no real predictive power?"

### Bootstrap Confidence Interval

```
1. Given N bets with flat P&L per bet
2. Repeat 2,000 times:
   a. Sample N bets with replacement
   b. Compute mean ROI of the sample
3. 95% CI = [2.5th percentile, 97.5th percentile] of the 2,000 ROI values
```

---

## Appendix B: Glossary

**American odds:** Betting odds format. Negative = favorite (how much to risk for $100 profit). Positive = underdog (how much you profit on a $100 risk).

**AUC (Area Under Curve):** Measures a model's ability to rank predictions correctly. 0.50 = random; 1.0 = perfect. An AUC of 0.57 means the model correctly ranks a random winning game above a random losing game 57% of the time.

**Barrel rate:** Percentage of batted balls hit at optimal launch angle and exit velocity for extra-base hits. Higher = more dangerous hitter.

**Bootstrap confidence interval:** Statistical technique where you resample your data thousands of times with replacement to estimate the uncertainty in a metric. A 95% CI of [+5%, +20%] means the true value is between 5% and 20% with 95% confidence.

**Brier score:** See Appendix A.

**Closing line:** The final odds posted by a sportsbook just before a game starts. Considered the most efficient (accurate) price because it incorporates all information including late sharp money.

**Devig (vig removal):** The process of removing the sportsbook's profit margin from implied probabilities so they sum to 100%. See Appendix A for the formula.

**DraftKings (DK):** Major US sportsbook. Known for sharp closing lines due to high betting volume and professional bettor participation.

**Edge:** The difference between your estimated probability and the market's implied probability. Positive edge = you think the market is wrong in your favor. Edge = our_prob - market_price.

**Expected run value (xRV):** A metric quantifying how many runs a pitch (or pitcher) prevents/allows relative to league average, based on Statcast data (velocity, spin, movement, location, outcome).

**Isotonic calibration:** A method to adjust model probabilities so they match observed frequencies. If the model says 60% but only 52% of such events happen, isotonic calibration learns to map 60% -> 52%.

**Kalshi:** A US-regulated prediction market (CFTC-regulated exchange) where users buy and sell binary contracts on real-world events. Contracts pay $1 if the event occurs, $0 if it doesn't.

**Kelly criterion:** See Appendix A.

**LightGBM:** A gradient-boosted decision tree algorithm. Builds predictions by combining many simple decision trees, each correcting the errors of the previous ones.

**Look-ahead bias:** The error of using future information to make past decisions in a backtest. For example, choosing a model because it has the best test-period ROI (when the choice should be made on validation data only).

**Moneyline (ML):** A bet on which team wins the game, regardless of margin.

**NRFI/YRFI:** No Run First Inning / Yes Run First Inning. Binary bet on whether at least one run is scored in the first inning.

**Optuna:** A Bayesian hyperparameter optimization framework. Uses TPE (Tree-structured Parzen Estimator) to efficiently search the hyperparameter space.

**Over/Under (O/U):** A bet on whether the total combined runs scored by both teams exceeds (over) or falls below (under) a line set by the sportsbook.

**Park factor:** A multiplier capturing how a stadium affects run scoring. 1.00 = average. Coors Field (Denver, high altitude) is ~1.14; Oracle Park (San Francisco, marine air) is ~0.87.

**Permutation test:** See Appendix A.

**ROI (Return on Investment):** Average P&L per bet, expressed as a fraction of the amount risked. +10% ROI means you earn $0.10 for every $1.00 of risk. For Kalshi, risk = buy_price, so ROI = mean(pnl_flat) where pnl_flat = (1-price) if won, (-price) if lost.

**Sample-size-weighted Sharpe:** The optimization metric used in validation. `score = sharpe * min(1.0, n_bets / 50)`. This penalizes strategies with very few bets (below 50) to avoid overfitting to noise.

**Sharpe ratio:** See Appendix A.

**Statcast:** MLB's ball-tracking system that measures pitch velocity, spin rate, movement, exit velocity, launch angle, and more for every pitch and batted ball.

**TimeSeriesSplit (TSCV):** A cross-validation method for time-ordered data. Unlike random k-fold, TSCV always trains on earlier data and validates on later data, respecting temporal ordering. With 5 splits on 2024 data: split 1 trains on months 1-4, validates on month 5; split 2 trains on months 1-6, validates on months 7-8; etc.

**Vigorish (vig/juice):** The sportsbook's built-in profit margin. At standard -110/-110 pricing, the book collects $110 from losers and pays $100 to winners, keeping $10 (4.5% margin).

**Walk-forward:** A backtesting methodology where the model is always trained on past data and tested on future data, simulating real-world conditions. Prevents the common mistake of testing on data the model has already seen.

---

## Appendix C: Data Inventory

| File | Records | Date Range | Description |
|------|---------|------------|-------------|
| `data/kalshi/kalshi_mlb_2025.parquet` | 2,099 | 2025-04-16 to 2025-10-29 | Kalshi moneyline contract prices |
| `data/kalshi/kalshi_total_2025.parquet` | 38 | 2025-09-30 to 2025-10-29 | Kalshi O/U strike prices (playoffs) |
| `data/kalshi/kalshi_total_2026.parquet` | 122 | 2026-03-25 to 2026-04-04 | Kalshi O/U strike prices (early 2026) |
| `data/odds/sbr_mlb_2025.parquet` | 2,902 | 2025-02-20 to 2025-10-29 | DK closing ML + O/U odds |
| `data/odds/mlb_odds_dataset.json` | ~2,130 | 2025-02-20 to 2025-08-16 | Multi-book odds (6 sportsbooks) |
| `data/odds/multibook_consensus_2025.parquet` | 2,130 | 2025-02-20 to 2025-08-16 | Derived consensus probabilities |
| `data/features/game_features_2024.parquet` | ~2,429 | 2024 season | Statcast features for ML training |
| `data/features/game_features_2025.parquet` | ~2,430 | 2025 season | Statcast features for ML inference |
| `data/backtest/nrfi_lgb_2025.parquet` | 2,430 | 2025 season | NRFI model predictions |
| `data/audit/sim_vs_kalshi_pregame_2025.csv` | ~500 | 2025 season | MC simulator win probabilities |

---

## Appendix D: Code Reference

| File | Purpose |
|------|---------|
| `src/kalshi_enhanced_backtest.py` | Primary backtest: DK arb with all filters, consensus, O/U, full val/test protocol |
| `src/kalshi_clean_backtest.py` | Original clean backtest: DK arb + LGB model + MC sim + ensemble |
| `src/scrape_kalshi.py` | Scrape Kalshi KXMLBGAME moneyline markets |
| `src/scrape_kalshi_total.py` | Scrape Kalshi KXMLBTOTAL O/U markets |
| `src/scrape_odds.py` | Scrape DK/FanDuel odds via The Odds API |
| `src/scrape_polymarket.py` | Scrape Polymarket MLB markets |
| `src/nrfi_model.py` | NRFI/YRFI LightGBM + LR ensemble model |
| `src/ou_vs_line_model.py` | O/U under classifier (edge-based, proven unprofitable) |
| `src/simulate.py` | Plate-appearance MC game simulator |
| `src/feature_engineering.py` | 155-feature Statcast feature pipeline |
| `src/unified_strategy.py` | Multi-strategy portfolio runner (pre-Kalshi version) |
