# Monte Carlo MLB Game Simulator — Technical Report

## 1. Overview

This is an at-bat-by-at-bat Monte Carlo game simulator that produces win probabilities for MLB games, both pregame and mid-game. It uses Bayesian hierarchical matchup models as its engine and historical Statcast base-running data for state transitions. The simulator is backtested against Kalshi prediction market prices to measure edge and profitability.

The core idea: instead of predicting a single pregame P(home_win) from features, we simulate the actual game thousands of times. Each simulation plays through every plate appearance, sampling outcomes from probability distributions shaped by the specific pitcher-hitter matchup. This gives us not just a win probability but full run distributions, and—critically—allows mid-game updates by injecting the current game state and simulating forward.

---

## 2. Architecture

```
                    ┌─────────────────────────────────────┐
                    │       Pre-computed Artifacts         │
                    │  (built once from historical data)   │
                    ├─────────────────────────────────────┤
                    │  1. League base rates (11 outcomes)  │
                    │  2. Transition matrix (880+ states)  │
                    │  3. Multi-output matchup model (PT)  │
                    └─────────────┬───────────────────────┘
                                  │
    ┌──────────────┐              │              ┌──────────────┐
    │  Lineups +   │              ▼              │  MLB Live    │
    │  Pitchers    │──▶  load_simulation_context  ◀──│  Feed API    │
    └──────────────┘     (per-hitter dists)       └──────────────┘
                                  │                      │
                                  ▼                      ▼
                    ┌─────────────────────────┐  ┌──────────────┐
                    │  monte_carlo_win_prob()  │  │  GameState   │
                    │  (10,000 simulations)    │◀─│  (mid-game)  │
                    └─────────────────────────┘  └──────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  P(home_win), run dists, │
                    │  extras probability      │
                    └─────────────────────────┘
```

---

## 3. Pre-computed Artifacts

Two lookup tables are built once from 5+ years of Statcast data (`build_transition_matrix.py`):

### 3.1 League-Average Base Rates

From every plate appearance in the Statcast dataset, we compute the unconditional probability of each of 11 outcome categories:

| Outcome | Description | Typical Rate |
|---------|-------------|-------------|
| K | Strikeout | ~22% |
| BB | Walk | ~8% |
| HBP | Hit by pitch | ~1% |
| 1B | Single | ~15% |
| 2B | Double | ~5% |
| 3B | Triple | ~0.5% |
| HR | Home run | ~3% |
| dp | Double play | ~2% |
| out_ground | Ground out | ~22% |
| out_fly | Fly out | ~16% |
| out_line | Line out | ~5% |

These serve as the "prior" — the starting point before we incorporate batter-specific, pitcher-specific, and matchup-specific information.

### 3.2 Base-Running Transition Matrix

For every combination of `(PA outcome, base state, outs)` — 11 × 8 × 3 = 264 possible situations — we compute the empirical distribution over `(new base state, runs scored)` from historical data.

**How it's built:**
1. Group all PAs by half-inning (game_pk + inning + top/bot)
2. For each PA, record the outcome, current base state, and outs
3. Look at the *next* PA in the same half-inning to observe the new base state
4. Compute runs scored from `post_home_score - home_score` (or away equivalent)
5. Aggregate into probability distributions

For example, the entry `(1B, runner_on_2B, 1_out)` might have:
- 60% → runner scores, batter on 1B (1 run)
- 30% → runner to 3B, batter on 1B (0 runs)
- 10% → runner out at home, batter on 1B (0 runs, +1 out)

For combinations with fewer than 20 observations, we fall back to **deterministic rules** that encode standard baseball logic (e.g., HR clears bases, sac fly scores runner from 3B with <2 outs, etc.).

---

## 4. Per-Hitter Outcome Distributions

The heart of the simulator is building a custom 11-category probability distribution for every hitter-pitcher matchup in the game. This combines two independent sources:

### 4.1 Source A: Log5 Baseline

For each batter, we compute their recent PA outcome frequencies from the last ~500 PAs. For the pitcher, the last ~800 PAs. These are combined using the **log5 / odds-ratio method** (Bill James):

```
P(outcome | batter, pitcher) ∝ (batter_rate × pitcher_rate) / league_rate
```

This captures individual tendencies — a high-K pitcher facing a high-K batter produces even more strikeouts than either individually.

**Data fallback chain for batters:**
1. MLB empirical rates (>= 50 PAs)
2. MiLB stats from the MLB Stats API (AAA → AA → A+ → A, with level adjustment multipliers)
3. League average (last resort)

### 4.2 Source B: Multi-Output Matchup Model

The **multi-output matchup model** (`multi_output_matchup_model.py`, PyTorch) directly predicts the 11-category outcome distribution for each hitter-pitcher pair. Unlike the previous scalar xRV approach, this preserves matchup-specific outcome profiles.

**Architecture:**
```
logits[c] = intercept[c]
           + beta_arsenal[k,c] · arsenal_z[k]        # population arsenal effect
           + beta_ptmix[p,c] · pitch_mix[p]           # population pitch-type effect
           + hitter_base[h,c]                          # per-hitter baseline
           + hitter_arsenal[h,k,c] · arsenal_z[k]     # per-hitter arsenal sensitivity
           + hitter_ptype[h,p,c] · pitch_mix[p]       # per-hitter pitch-type response

probs = softmax(logits)  →  11-category distribution
```

The model works in **arsenal feature space** (13 dimensions: velo, movement, mix, spin, extension, release spread), so it generalizes to pitchers not seen in training. Per-hitter parameters are regularized via L2 penalties (equivalent to hierarchical Bayesian shrinkage), meaning unknown hitters get population-average predictions while well-observed hitters get personalized outcome distributions.

**Why this is better than scalar xRV:** A hitter who crushes fastballs but whiffs on sliders will have elevated HR/XBH probabilities against fastball-heavy pitchers but elevated K% against slider-heavy pitchers. The old approach collapsed this to a single number and applied uniform outcome shifts.

### 4.3 Blending

The final distribution is a weighted average:

```
P(outcome) = w × model_probs + (1 - w) × log5_probs
```

Where `w = 0.6` (tunable). The log5 baseline provides robustness when the model has limited data for a hitter. The softmax output guarantees valid probability distributions without any need for clamping.

### 4.5 Double Play Adjustment

When a double play is impossible (no runner on 1B or 2 outs), the DP probability is redistributed to ground outs. This is pre-computed as a separate distribution to avoid per-PA branching in the hot loop.

---

## 5. Game Simulation

### 5.1 Single Plate Appearance

```python
simulate_plate_appearance(state, batting_side, pitching_side, dists, transition_matrix, rng)
```

1. Look up the pre-computed outcome distribution for this (batter, pitcher state) pair
2. Sample an outcome using cumulative-sum weighted random selection
3. Look up the base-running transition for (outcome, current bases, current outs)
4. If the transition has multiple possible results (e.g., runner may or may not score on a single), sample from the empirical distribution
5. Return the outcome, runs scored, new base state, and outs added

### 5.2 Half-Inning

```python
simulate_half_inning(state, batting_side, pitching_side, dists, transition_matrix, config, rng)
```

Loop plate appearances until 3 outs (or walk-off in bottom of 9th+). Each PA:
- Updates bases, outs, score
- Advances the lineup position (cycles through 9 batters)
- Tracks pitch count using **per-outcome estimates** (K~4.8, BB~5.6, contact~3.3-3.6 pitches) rather than a flat average
- Checks for walk-off condition

Safety limit of 25 PAs per half-inning (MLB record is 23 batters in a half-inning). Outcome sampling uses a Numba JIT-compiled cumulative sum for performance.

### 5.3 Full Game

```python
simulate_game(home_ctx, away_ctx, initial_state, dists, transition_matrix, config, rng)
```

Alternates half-innings from the initial state:

1. **SP → reliever transition:** At each inning boundary, check if the starter's accumulated pitch count exceeds a Gaussian-sampled limit (mean=92, σ=10). Pitch counts accumulate per-outcome (a K costs ~4.8 pitches, a ground out ~3.4), so high-K starters reach their limit sooner.
2. **Individual relievers:** When the SP is pulled, the simulator cycles through the team's top relievers (up to 7, ordered by recent usage). Each reliever has their own per-hitter outcome distributions from the matchup model. Relievers are pulled after ~20 ± 5 pitches (one inning typical), then the next reliever enters. This captures late-game dynamics far better than the previous single bullpen composite.
3. **Manfred runner:** From the 10th inning on, place a ghost runner on 2B at the start of each half-inning.
4. **Walk-off:** If the home team takes the lead in the bottom of the 9th or later, the game ends immediately.
5. **Skip bottom:** If the home team leads after the top of the 9th or later, skip the bottom half.
6. **Extra innings:** Continue until one team leads after a complete inning (max 15 innings safety limit).

### 5.4 Monte Carlo Aggregation

```python
monte_carlo_win_prob(home_ctx, away_ctx, initial_state, transition_matrix, config)
```

Runs `n_sims` (default 10,000) independent simulations from the initial state. Aggregates:

- **Win probability:** home_wins / n_sims
- **Run distributions:** full histogram of home and away runs
- **Mean/median runs:** per side and total
- **Standard deviation:** of run totals

### 5.5 Mid-Game State Injection

The simulator can start from any game state, not just the pregame default. The `fetch_live_game_state()` function pulls from the MLB Stats API:

- Current inning, top/bottom, outs
- Score (home and away runs)
- Base occupancy (runners on 1st, 2nd, 3rd)
- Current lineup positions
- Pitcher state (SP or BP, based on how many pitchers have appeared)

For the backtest, `extract_half_inning_states()` reconstructs these states from historical play-by-play data at each half-inning boundary, giving ~18 state points per game.

---

## 6. Backtest Methodology

### 6.1 Data Sources

- **MC Simulator:** Runs from the game state at each half-inning boundary, producing P(home_win)
- **Kalshi:** Minute-by-minute candlestick data (last-traded price) from the Kalshi prediction market API
- **Ground truth:** Actual game outcomes from MLB Stats API

### 6.2 Candle Matching

For each half-inning state point, we match the closest Kalshi candle (within 30 minutes) to get the market's implied probability at that game moment. This uses `price.close` (last-traded price), which matches what traders actually see on the Kalshi platform.

### 6.3 Metrics

| Metric | What it measures |
|--------|-----------------|
| Log loss | Calibration quality — penalizes confidently wrong predictions |
| Brier score | Mean squared error of probability predictions |
| AUC | Discrimination — ability to rank games by win likelihood |
| Accuracy | Simple correct/incorrect at 50% threshold |
| Edge | Sim probability minus market probability (positive = we think home is underpriced) |
| ROI | Return on investment at various edge thresholds |
| Sharpe | Risk-adjusted return (annualized) |

### 6.4 Kelly Rebalancing Strategy

At each half-inning boundary, we compute the Kelly-optimal position:

**For YES bets** (model says home is underpriced, p_sim > p_market):
```
f* = (p_sim - p_market) / (1 - p_market)
```

**For NO bets** (model says home is overpriced, p_sim < p_market):
```
f* = (p_market - p_sim) / p_market
```

We scale by a Kelly fraction (25%, 50%, or 100%) and require a minimum edge before trading. At each half-inning, we rebalance to the new target position by buying or selling contracts at the current Kalshi price. At game end, contracts settle at $1 (home wins) or $0 (home loses).

This rebalancing strategy is powerful because:
1. **It compounds small edges:** The model has consistent 3-5% edges that accumulate over ~18 half-innings per game
2. **It reduces variance:** Frequent rebalancing means we're not making a single all-or-nothing bet
3. **It's self-correcting:** If the model is wrong early, the position adjusts as new information arrives

---

## 7. Backtest Results

*Results from the 2025 season backtest (2,000 pregame sims, 1,000 in-game sims):*

### 7.1 Pregame Performance

2,119 games from the 2025 MLB season. Each game simulated 2,000 times from pregame state.

| Metric | Simulator | Kalshi | Baseline (50%) |
|--------|-----------|--------|----------------|
| Log Loss | 0.7110 | 0.6802 | 0.6910 |
| Brier Score | 0.2578 | 0.2435 | — |
| AUC | 0.5369 | 0.5819 | — |
| Accuracy | 51.8% | 55.6% | — |
| Mean Prediction | 0.5014 | 0.5383 | — |

**Pregame takeaway:** The simulator underperforms Kalshi pregame. Log loss is +0.031 worse than the market. The simulator's mean prediction (0.501) is too centered vs Kalshi (0.538), suggesting it under-weights home-field advantage. Flat-bet ROI is negative at all edge thresholds.

**Calibration (pregame):** The simulator is systematically under-confident at low predictions and over-confident at high predictions. Below 50%, predicted probabilities undershoot actual win rates by 9-17 percentage points. Above 60%, predictions overshoot actuals by 8-12 points. Kalshi is well-calibrated by comparison (gaps < 7 points in all bins).

### 7.2 In-Game Performance

8,737 state points across 494 games. Each state simulated 1,000 times.

| Metric | Simulator | Kalshi | Baseline (50%) |
|--------|-----------|--------|----------------|
| Log Loss | 0.5039 | 0.4852 | 0.6913 |
| Brier Score | 0.1703 | 0.1631 | — |
| AUC | 0.8255 | 0.8371 | — |
| Accuracy | 73.3% | 74.8% | — |
| Mean Prediction | 0.5090 | 0.5260 | — |

**By game phase:**

| Phase | N | Sim LL | Mkt LL | Delta | Sim Acc |
|-------|---|--------|--------|-------|---------|
| Early (Inn 1-3) | 2,952 | 0.6660 | 0.6334 | +0.033 | 59.3% |
| Mid (Inn 4-6) | 2,952 | 0.5155 | 0.4916 | +0.024 | 75.2% |
| Late (Inn 7-9+) | 2,833 | 0.3230 | 0.3242 | **-0.001** | 85.8% |

**By inning (late game):**

| Inning | N | Sim LL | Kalshi LL | Delta | Sim Acc |
|--------|---|--------|-----------|-------|---------|
| 7 | 983 | 0.3766 | 0.3749 | +0.002 | 83.5% |
| 8 | 982 | 0.2978 | 0.2943 | +0.004 | 87.2% |
| 9 | 758 | **0.2426** | 0.2593 | **-0.017** | 90.4% |

**In-game takeaway:** The simulator tracks Kalshi closely, with a log loss gap of just +0.019 overall. In late innings (7th+), the simulator **matches or slightly beats** Kalshi (LL delta -0.001). In the 9th inning specifically, the simulator **beats Kalshi by 1.7 points** of log loss with 90.4% accuracy. Convergence analysis shows MAE drops from 0.079 (early) to 0.042 (late) and correlation rises from 0.863 to 0.980.

**Calibration (in-game):** Much better than pregame. The extremes (below 35% and above 65%) are well-calibrated (gaps of 3.0 and 1.4 points). The mid-range (40-60%) shows a modest underestimation bias of 4-8 points.

### 7.3 Kelly Rebalancing Strategy

The core value proposition: rebalance position at every half-inning boundary using Kelly-optimal sizing. This compounds small edges (~1.7% mean absolute edge) across ~18 opportunities per game.

| Kelly % | Min Edge | Games | Total PnL | ROI | Sharpe | 95% CI | P(ROI>0) |
|---------|----------|-------|-----------|-----|--------|--------|----------|
| 25% | 3% | 492 | $39,936 | +0.8% | 3.46 | [+0.5%, +1.1%] | 100% |
| 25% | 5% | 492 | $40,802 | +0.8% | 3.71 | [+0.5%, +1.2%] | 100% |
| 25% | 7% | 492 | $38,911 | +0.8% | 3.69 | [+0.5%, +1.1%] | 100% |
| 50% | 5% | 492 | $81,603 | +1.7% | 3.71 | [+1.0%, +2.3%] | 100% |
| 100% | 5% | 492 | $163,207 | +3.3% | 3.71 | [+2.1%, +4.6%] | 100% |

**Key findings:**
1. **All configurations profitable** with P(ROI > 0) = 100% across 10,000 bootstrap samples
2. **Sharpe ratios of 3.5-3.7** — the frequent rebalancing dramatically reduces variance
3. **Edge threshold barely matters** — the 3%, 5%, and 7% filters produce similar ROI, suggesting the model's edges are broadly distributed rather than concentrated in a few games
4. **PnL scales linearly with Kelly fraction** — no evidence of over-betting at 100% Kelly, though real-world execution would favor 25-50% for safety

**Why this works despite worse pregame predictions:** The simulator doesn't need to *beat* Kalshi at any single point in time. It needs to have *different* errors that cancel over many rebalancing opportunities. The late-game convergence (where the sim matches or beats Kalshi) means the position naturally drifts toward correct settlement as the game progresses.

**Important caveats:** This backtest assumes execution at last-traded price with no slippage. Real Kalshi execution involves:
- Bid-ask spreads of 2-5 cents per contract
- 7% fee on net profits
- Potential liquidity constraints on frequent rebalancing
- These frictions would reduce realized PnL by an estimated 30-50%

---

## 8. Upstream Model Details

### 8.1 xRV (Expected Run Value) Model

**File:** `build_xrv.py`

Assigns a run value to every pitch. Two components:
- **Non-contact outcomes** (K, BB, HBP): Count-based run expectancy from base-out state tables
- **Contact outcomes**: LightGBM model using exit velocity, launch angle, spray angle, sprint speed, and park factors

Trained on prior-season data only (no lookahead). The xRV values are the inputs to the matchup model.

### 8.2 Multi-Output Matchup Model

**File:** `multi_output_matchup_model.py`

PyTorch model that directly predicts 11-category outcome distributions per hitter-pitcher matchup. Replaces the previous scalar xRV model + log-linear calibration with a single end-to-end prediction.

Works in arsenal feature space (13 dimensions). Per-hitter parameters are L2-regularized (equivalent to hierarchical Bayesian shrinkage). Trained on ~160K PA-level observations pooled across 2024-2025 seasons (vsL: 65K PAs, 240 hitters; vsR: 95K PAs, 357 hitters). Intercepts initialized from empirical outcome rates and regularized toward that prior to prevent degenerate population averages. Softmax output guarantees valid probability distributions.

### 8.3 Feature Engineering

**File:** `feature_engineering.py`

Computes pitcher arsenal profiles (13 dimensions), pitch-type mixes, and per-hitter matchup predictions. Pre-indexes all pitch data by pitcher and batter for fast lookup.

---

## 9. File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `simulate.py` | ~1630 | Core MC simulator: data structures, outcome distributions, game simulation, CLI |
| `multi_output_matchup_model.py` | ~700 | PyTorch multi-output matchup model: direct outcome prediction |
| `build_transition_matrix.py` | ~500 | Build base rates and transition matrix from Statcast |
| `backtest_vs_kalshi.py` | ~1260 | Full backtest framework: pregame, in-game, metrics, Kelly analysis |
| `plot_game_trace.py` | ~270 | Live game win probability trace: model vs Kalshi visualization |
| `matchup_model.py` | ~420 | Legacy Bayesian matchup model (PyMC/ADVI, used for feature engineering) |
| `feature_engineering.py` | ~2375 | Feature computation including arsenal profiles and matchup predictions |
| `build_xrv.py` | ~400 | Pitch-level expected run value model |

---

## 10. Limitations & Future Work

1. **No defensive positioning:** The transition matrix doesn't account for fielding quality (OAA) or shift strategies. A team with elite outfield defense will turn more fly balls into outs than the league-average transition matrix assumes.

2. **No pinch hitting:** The simulator doesn't model in-game substitutions (pinch hitters, defensive replacements). This matters most in late-game NL situations.

3. **Market microstructure:** The Kelly backtest assumes we can trade at the last-traded price with no slippage or fees. Real execution on Kalshi involves bid-ask spreads (~2-5¢) and 7% fee on profits, which would reduce realized PnL.

4. **MiLB level adjustments are approximate:** The factors translating MiLB stats to MLB equivalents are rough estimates. A proper calibration using historical MiLB→MLB promotion data would improve accuracy for callups.
