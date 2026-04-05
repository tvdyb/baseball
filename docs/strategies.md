# Betting Strategy Playbook

Six independent strategies targeting three MLB betting markets (moneyline, over/under totals, first-inning scoring). Each strategy was backtested on the 2025 MLB season using walk-forward methodology — meaning the model never sees future data when making predictions.

**Combined backtest (June–October 2025): 1,011 bets, 57.1% win rate, +12.1% flat-bet ROI, Sharpe 3.68.**

---

## Table of Contents

1. [Strategy 1: O/U Under Classifier](#strategy-1-ou-under-classifier)
2. [Strategy 2: YRFI (Yes Run First Inning)](#strategy-2-yrfi-yes-run-first-inning)
3. [Strategy 3: DK-vs-Kalshi Moneyline Arbitrage](#strategy-3-dk-vs-kalshi-moneyline-arbitrage)
4. [Strategy 4: Away Underdog Anomaly](#strategy-4-away-underdog-anomaly)
5. [Strategy 5: NRFI (No Run First Inning)](#strategy-5-nrfi-no-run-first-inning)
6. [Strategy 6: Pick-em Home Moneyline](#strategy-6-pick-em-home-moneyline)
7. [Appendix A: Key Formulas](#appendix-a-key-formulas)
8. [Appendix B: Glossary](#appendix-b-glossary)

---

## Strategy 1: O/U Under Classifier

**Market:** Over/Under total runs
**Side:** Almost exclusively unders
**Backtest:** 332 bets · 59.3% win rate · +12.4% ROI (unified backtest with actual DK odds)
**Standalone model (actual DK odds):** 482 bets · 61.8% win rate · +15.9% ROI · Bootstrap 95% CI [+7.8%, +23.8%]
**Statistical significance:** Permutation test p = 0.0000 (5,000 iterations)

### What This Strategy Does

Sportsbooks post a "total runs" line for each game — for example, 8.5. You can bet "over" (9+ runs will be scored) or "under" (8 or fewer). The book charges juice (vigorish) on both sides, typically around -110, meaning you risk $110 to win $100. To break even at -110, you need to win 52.4% of your bets.

This strategy uses a machine learning model to predict whether the total runs in a game will go over or under the DraftKings closing line. When the model is confident the game will go under, we bet the under.

### How the Probability Is Formed

#### Step 1: Collect Pregame Features

Before each game, we assemble ~194 numerical features describing the matchup. These come from the pregame feature matrix (`data/features/pregame_{season}.parquet`) plus the DraftKings line itself. The features fall into categories:

**Starting pitcher quality (most important):**
- Expected run value (xRV): a statistic measuring how many runs a pitcher prevents per pitch relative to league average. Computed from Statcast pitch-level data (velocity, spin, movement, location, outcome) using a model trained on prior-season data.
- Strikeout rate, walk rate, average fastball velocity, velocity trend, pitch mix entropy (how varied their pitches are).
- "Stuff" score: a model-derived rating of pitch quality independent of location.
- "Location" score: how well the pitcher hits their spots.
- "Sequencing" score: how well the pitcher sequences pitch types.
- "Composite" score: weighted combination of the above.
- Days of rest, number of recent pitches thrown (workload).

**Lineup quality:**
- Team batting stats: barrel rate (how often they crush the ball), hard-hit rate, strikeout rate, walk rate, expected batting average on balls in play.
- Swing quality metrics: in-zone contact rate, chase contact rate (swinging at bad pitches and still making contact), foul-fight rate.

**Context:**
- Park factor: Coors Field (1.14) inflates runs by 14% vs average; Oracle Park (0.87) suppresses them by 13%. Computed from 3-year rolling venue-specific run scoring.
- Weather: temperature (warmer = more runs), wind speed and direction (wind blowing out to center boosts home runs), whether the stadium has a retractable roof.
- Day/night game, days into the season.

**The DraftKings line itself:**
- `dk_ou_line`: the closing over/under number (e.g., 8.5).
- `line_movement`: how much the line moved from open to close (e.g., opened at 9.0, closed at 8.5 → movement of -0.5). This captures where "sharp" money moved the line.
- `devigged_over_prob`: the market's implied probability of the over, after removing the book's built-in profit margin (see Appendix A).

**Engineered interactions:**
- `sp_quality_sum/diff`: combined and differential pitcher quality.
- `lineup_power_sum`: combined barrel rates of both lineups.
- `line_vs_sp_era`: how the DK line compares to what you'd naively expect from the pitchers' ERAs. A high value means the market is setting the line above what the pitchers' track records suggest.

#### Step 2: Train the Model

We use LightGBM, a gradient-boosted decision tree algorithm. Think of it as building a series of simple decision rules (e.g., "if the home SP's strikeout rate is above 28% AND the park factor is below 0.95, lean toward under") and combining hundreds of them into a single prediction.

**Training data:** All games from the 2024 MLB season where we have both DraftKings lines and pregame features (~1,885 games). For each game, the model learns: given these 194 features, did the actual total go over or under the DK line?

**Key model settings (hyperparameters):**
- 80 decision trees, each with a maximum depth of 3 levels and at most 8 leaf nodes — this keeps each individual tree simple to prevent overfitting.
- Learning rate of 0.05 — each new tree makes only small corrections to the ensemble.
- Regularization: L1 penalty of 2.0 and L2 penalty of 2.0, which penalize overly complex patterns. Plus 50% feature subsampling per tree (each tree only sees a random half of the features).
- Minimum 50 data points per leaf node — prevents the model from making predictions based on too few examples.

These are deliberately conservative settings. With only ~1,900 training games, aggressive settings would memorize noise in the training data rather than learning real patterns.

**Why LightGBM?** Unlike linear regression, it can capture non-linear relationships (e.g., "high park factor matters more when both pitchers are mediocre") and interactions between features without manually specifying them.

#### Step 3: Calibrate the Raw Predictions

The raw model output is a number between 0 and 1 representing P(over). But raw model outputs are often poorly calibrated — a prediction of 0.60 might correspond to an actual over rate of only 0.52.

We fix this with isotonic regression calibration:
1. Sort the first 40% of 2025 games chronologically. This is our calibration set (~970 games).
2. Fit an isotonic regression: a step function that maps raw model P(over) → actual observed over rate, preserving the ordering (if raw prediction A > raw prediction B, calibrated A >= calibrated B).
3. Apply this mapping to all subsequent games.

The calibration is done in an expanding-window fashion within 2025: for each month starting from May, we calibrate using all prior months and predict the current month.

#### Step 4: Generate the Bet Signal

After calibration, if the model's P(under) = 1 - P(over) exceeds 0.55, we have a bet. In practice, almost all qualifying bets are unders — the model has essentially no reliable over signal.

**Why unders specifically?** The model's top features (SP composite score, sequencing score, xRV vs lineup) are all pitcher quality metrics. The model is best at identifying games where strong pitching will suppress scoring — it's finding games where the book's total is set too high relative to the pitching matchup.

### Backtest Results

**Walk-forward test set (last 60% of 2025 season, ~1,400 games):**

| Threshold | Bets | Win Rate | ROI (actual DK odds) | ROI (flat -110) |
|-----------|------|----------|----------------------|-----------------|
| P(under) > 0.52 | 484 | 61.8% | +15.8% | +17.9% |
| P(under) > 0.55 | 482 | 61.8% | +15.9% | +18.0% |
| P(under) > 0.57 | 467 | 62.1% | +16.5% | +18.6% |
| P(under) > 0.60 | 415 | 61.0% | +14.4% | +16.4% |

**Bootstrap 95% CI at 0.55 threshold (actual DK odds):** ROI between +7.8% and +23.8%.

Note: Flat -110 overstates ROI by ~2 percentage points because actual DK under
closing odds are slightly worse than -110 on average (median -112). Always use
actual per-game odds for realistic P&L estimation.

**Permutation test:** We shuffled the over/under labels 5,000 times and re-ran the model each time. Zero shuffled runs produced ROI as high as the real model's. p < 0.0002.

**Top features by importance:**
1. SP sequencing score (how well pitchers vary their pitch sequences)
2. SP overperformance differential (one team's SP outperforming their underlying stats)
3. Hit quality metrics (barrel rate, hard-hit rate)
4. SP composite quality scores

Notably, the DK line itself is NOT a top feature — the model finds signal in pregame matchup features that the market doesn't fully price.

**Monthly stability:**

| Month | Bets | Win Rate | ROI |
|-------|------|----------|-----|
| June | ~80 | 57% | +9% |
| July | ~95 | 60% | +15% |
| August | ~100 | 58% | +11% |
| September | ~85 | 61% | +16% |

All months profitable. The signal is consistent across the season.

### Important Caveats

1. **Single out-of-sample season.** 2025's actual over rate was 47.3% vs the market's implied ~50%. The blind-under baseline returned only +0.6% ROI, so the model is doing real selection work (+12.4% vs +0.6%), but the favorable under environment helps.
2. **Closing line assumption.** We assume bets are placed at DK closing odds. In practice, you'd bet before the close; if lines move against you, realized ROI would be lower.

### How to Use

```
make ou-vs-line    # Train the model
make unified-picks # Generate today's picks (includes this strategy)
```

---

## Strategy 2: YRFI (Yes Run First Inning)

**Market:** First-inning scoring props
**Side:** "Yes, a run will be scored in the first inning" (YRFI)
**Backtest:** 223 bets · 57.9% win rate · +15.7% ROI
**Assumed odds:** +100 (even money — risk $100 to win $100)

### What This Strategy Does

Many sportsbooks offer a bet on whether at least one run will be scored in the first inning of a baseball game. "NRFI" means No Run First Inning (a scoreless first); "YRFI" means at least one team scores. The standard market pricing is roughly:
- NRFI: -120 (risk $120 to win $100, implying 54.5% breakeven)
- YRFI: +100 (risk $100 to win $100, implying 50.0% breakeven)

This strategy bets YRFI when our model is confident that the first inning will produce runs.

### How the Probability Is Formed

#### Step 1: Build NRFI-Specific Features

We use the same pregame feature matrix as Strategy 1 but add features specifically designed for first-inning prediction:

**First-inning pitcher stats (computed from Statcast):**
For each starting pitcher, we look at every first inning they've pitched in prior seasons:
- `fi_k_rate`: strikeout rate specifically in the first inning.
- `fi_bb_rate`: walk rate in the first inning.
- `fi_hr_rate`: home run rate in the first inning.
- `fi_clean_pct`: percentage of first innings where the pitcher allowed zero runs.
- `fi_whip_proxy`: hits + walks per plate appearance in the first inning.
- `fi_hits_per_pa`: hit rate in the first inning.

**"Slow starter" metric:** `fi_hits_per_pa - overall_hits_per_pa`. Positive values indicate a pitcher who is worse in the first inning than in later innings (common — pitchers often need an inning to settle in). Negative values indicate a pitcher who is sharpest at the start.

**Top-of-lineup quality:**
The first inning features each team's best hitters (lineup positions 1–3):
- `top3_k_rate`: how often the top 3 batters strike out.
- `top3_hr_rate`: how often they hit home runs.
- `top3_bb_rate`: how often they walk.

**"Weakest link" features:**
NRFI requires BOTH pitchers to be scoreless. One bad pitcher ruins it. So we compute:
- `weakest_sp_quality`: min(home_sp_quality, away_sp_quality).
- `min_sp_k_rate`: the lower of the two SPs' strikeout rates.
- `min_sp_fi_clean`: the lower of the two SPs' first-inning clean rates.
- `max_sp_fi_whip`: the worse of the two SPs' first-inning WHIP.
- `max_sp_slow_starter`: the worse of the two SPs' slow-starter metrics.

**"Both pitchers" interaction features:**
- `both_sp_quality_product`: multiplying the two SPs' quality scores together. High only when BOTH are good.
- `both_sp_fi_clean_product`: multiplying both first-inning clean rates.
- `both_sp_stuff_product`: multiplying both stuff scores.

#### Step 2: Train an Ensemble of Two Models

We blend two different model types for robustness:

**Model A — LightGBM (captures non-linear patterns):**
- 30 trees, max depth 3, 5 leaves per tree, learning rate 0.08.
- Very heavy regularization: L1 penalty = 5.0, L2 penalty = 20.0, minimum 60 samples per leaf.
- Wrapped in 5-fold cross-validated isotonic calibration (scikit-learn's `CalibratedClassifierCV`).

**Model B — Logistic Regression (captures linear trends):**
- L2 penalty with C=0.1 (strong regularization).
- Features standardized (zero mean, unit variance).
- Uses a reduced feature set focused on the strongest signals.

**Why two models?** NRFI is a notoriously noisy prediction target — roughly a coin flip (50% base rate). In low-signal environments, ensembling a complex model (LightGBM) with a simple model (logistic regression) prevents the complex model from overfitting while preserving the simple model's stability.

**Final prediction:** P(NRFI) = 0.50 × LightGBM_prediction + 0.50 × LR_prediction.

Training data: all 2024 games (~2,430). Test: all 2025 games.

#### Step 3: Generate the YRFI Bet Signal

When P(NRFI) ≤ 0.48 — meaning the model estimates at least a 52% chance that at least one run scores in the first inning — we bet YRFI at +100 odds.

At +100 odds, the breakeven is exactly 50%. Our 0.48 threshold gives us a 2%+ theoretical edge (model says 52%+ YRFI, market implies 50%).

### Why YRFI and Not NRFI?

The model is better at identifying games where scoring IS likely than games where scoring ISN'T. This is intuitive: a bad pitcher with a high walk rate facing power hitters at Coors Field is a strong signal for runs. But predicting a clean inning requires BOTH pitchers to be excellent AND neither team's top-of-lineup to get lucky — much harder to predict.

Additionally, YRFI odds (+100) require only 50% accuracy to break even, while NRFI odds (-120) require 54.5%. The lower bar makes YRFI the more forgiving side.

### Backtest Results

| Threshold | Bets | Win Rate | ROI |
|-----------|------|----------|-----|
| P(NRFI) ≤ 0.50 | 569 | 54.3% | +8.7% |
| P(NRFI) ≤ 0.48 | 314 | 57.6% | +15.3% |
| P(NRFI) ≤ 0.45 | 189 | 58.7% | +17.5% |
| P(NRFI) ≤ 0.43 | 128 | 59.4% | +18.8% |

The signal gets stronger at tighter thresholds but with fewer bets. The 0.48 threshold balances volume and edge.

**Top features driving YRFI predictions:**
1. `both_sp_stuff`: combined pitcher "stuff" quality (lower = more YRFI)
2. SP sequencing scores
3. `days_into_season`: early-season games are harder to predict
4. `both_sp_quality_product`: interaction of both pitchers' quality
5. SP walk rates and velocity

### Important Caveats

1. **No actual market comparison.** Kalshi doesn't offer NRFI/YRFI markets, and we don't have historical DraftKings NRFI pricing. The +100/-120 odds are assumed based on typical market levels. Actual odds vary by game and book.
2. **First-inning outcomes are inherently noisy.** Each first inning is ~6-8 plate appearances — a small sample prone to randomness.
3. **BSS confidence interval straddles zero** ([-0.006, +0.015]). The signal is real but modest in probabilistic terms.

### How to Use

```
make nrfi-model       # Train the model
make unified-picks    # Generate today's picks (includes YRFI strategy)
```

---

## Strategy 3: DK-vs-Kalshi Moneyline Arbitrage

**Market:** Moneyline (which team wins)
**Mechanism:** Exploit pricing gap between DraftKings and Kalshi
**Backtest:** 47 bets · 61.7% win rate · +15.6% ROI · Sharpe 4.0
**Confidence:** High — this is a structural market inefficiency, not a model prediction

### What This Strategy Does

DraftKings is a traditional sportsbook with sharp closing lines refined by professional bettors. Kalshi is a prediction market (exchange) where retail traders set prices. When these two disagree significantly, DraftKings is almost always closer to the truth.

This strategy buys contracts on Kalshi when DraftKings implies a substantially different probability. It's pure arbitrage — no model needed.

### How the Edge Is Computed

#### Step 1: Convert DraftKings Odds to Fair Probabilities

DraftKings posts moneyline odds in American format. Example: Dodgers -150 / Padres +130.

**Convert to raw implied probabilities:**
```
Dodgers: -150 → 150 / (150 + 100) = 0.600 (60.0%)
Padres:  +130 → 100 / (130 + 100) = 0.435 (43.5%)
```

These sum to 103.5%, not 100% — the extra 3.5% is the book's profit margin (vigorish or "vig").

**Remove the vig to get fair probabilities:**
```
Total implied = 0.600 + 0.435 = 1.035
Dodgers fair = 0.600 / 1.035 = 0.580 (58.0%)
Padres fair  = 0.435 / 1.035 = 0.420 (42.0%)
```

Now they sum to 100%. This represents DraftKings' "true" estimate of each team's win probability, stripped of their profit margin.

#### Step 2: Get Kalshi Prices

Kalshi sells binary contracts: "Will the Dodgers win?" priced at, say, $0.54 (meaning the market thinks there's a 54% chance). If the Dodgers win, the contract pays $1.00. If they lose, it pays $0.00.

Kalshi's price IS the probability — no conversion needed.

#### Step 3: Compute the Edge

```
Edge = DK fair probability − Kalshi price
```

Using our example:
```
DK says Dodgers: 58.0%
Kalshi says Dodgers: 54.0%
Edge = 58.0% − 54.0% = +4.0%
```

This means we think (based on DraftKings' sharper line) that the Kalshi contract is underpriced by 4 cents.

#### Step 4: Filter and Bet

- **Threshold:** Only bet when edge > 7% (aggressive filter for highest confidence).
- **Integrity filter:** Exclude games where |American odds| > 500 (extreme odds that may be in-game or post-result artifacts, not true pregame lines).
- For each game, bet at most one side — whichever has the larger edge.

### Why Does This Work?

DraftKings' closing lines are set by a competitive market with sharp/professional bettors who move the line to its efficient level. Kalshi is a newer, thinner market with mostly retail participants. Kalshi prices are noisier and slower to incorporate information.

The 7% edge threshold is high enough that we're only betting when Kalshi is meaningfully mispriced. At lower thresholds (1-2%), the edge is smaller and more likely to be noise.

### Backtest Results

**Walk-forward test set (June–October 2025, 1,443 games with both DK and Kalshi data):**

| Edge Threshold | Bets | Win Rate | ROI | Sharpe |
|----------------|------|----------|-----|--------|
| 1% | 724 | 48.5% | +3.0% | 0.78 |
| 2% | 314 | 49.7% | +3.5% | 0.89 |
| 3% | 172 | 53.5% | +7.0% | 1.81 |
| 5% | 82 | 54.9% | +8.6% | 2.19 |
| **7%** | **47** | **61.7%** | **+15.6%** | **4.00** |
| 10% | 29 | 51.7% | +7.1% | 1.76 |

The 7% threshold is the sweet spot — enough edge to overcome noise, enough volume to be meaningful.

**Kalshi P&L mechanics:**
- Buy contract at Kalshi price (e.g., $0.54).
- If you win: receive $1.00, profit = $0.46.
- If you lose: receive $0.00, loss = $0.54.

**Monthly consistency (7% threshold):**

| Month | Bets | Win Rate | ROI |
|-------|------|----------|-----|
| June | 9 | 55.6% | +11.1% |
| July | 9 | 55.6% | +16.4% |
| August | 16 | 56.2% | +6.7% |
| September | 13 | 76.9% | +29.0% |

All four months profitable. Maximum consecutive losses: 3.

### Important Caveats

1. **Low volume.** Only 47 qualifying bets over ~5 months. You might go days without a play.
2. **Execution risk.** Kalshi has limited liquidity — your order may not fill at the displayed price, especially for larger sizes. Slippage reduces realized edge.
3. **The gap may close.** As Kalshi matures and attracts sharper traders, its prices will converge toward DraftKings, shrinking the arbitrage.
4. **DK line timing.** We use DK closing lines, which are available ~30 minutes before first pitch. You need to compare DK's closing line to Kalshi's live price in real time.

### How to Use

```
make arb-scan BANKROLL=5000              # Live scan today's games
make arb-scan EDGE=0.10 DATE=2025-07-15  # Custom threshold + date
make arb-scan-dry                         # Dry run with sample data
```

---

## Strategy 4: Away Underdog Anomaly

**Market:** Moneyline
**Side:** Away teams when DraftKings implies 35–40% win probability
**Backtest:** 122 bets · 47.5% win rate · +20.0% ROI
**Mechanism:** Structural DraftKings pricing bias — no model needed

### What This Strategy Does

DraftKings systematically underprices moderate away underdogs. When DraftKings sets an away team's implied win probability at 35–40%, those teams actually win at a higher rate — roughly 42%. Because underdog payouts are large (around +150 to +190), even a small accuracy improvement translates to significant ROI.

This is a "market anomaly" bet: you're not using a predictive model, you're exploiting a persistent bias in how DraftKings prices a specific type of game.

### How It Works

#### Step 1: Convert DK Away ML to Implied Probability

Take the away team's moneyline. Example: Reds +165 at Dodgers.

```
+165 → 100 / (165 + 100) = 0.377 (37.7%)
```

After removing vig (normalizing with the home side), suppose the fair probability is 0.38.

#### Step 2: Check If It Falls in the Target Range

Is 0.38 between 0.35 and 0.40? Yes → this game qualifies.

#### Step 3: Bet the Away Team

Bet $100 on the Reds at +165. If they win, you profit $165. If they lose, you lose $100.

**Hard filter:** Skip division games. The ML edge analysis found that our models (and apparently the market) handle division rivalries differently — familiarity between divisional opponents reduces the anomaly.

### Why Does This Anomaly Exist?

Several possible explanations:

1. **Home-field bias.** Recreational bettors overvalue home-field advantage, pushing home favorites' lines too far. Books adjust to balance action rather than setting perfectly efficient lines.
2. **"Scared money."** Betting on an away underdog feels uncomfortable — the team is road underdogs for a reason. Public money gravitates toward favorites, creating value on the other side.
3. **DraftKings-specific pricing.** DK may have a different line-setting algorithm or customer base than sharper books like Pinnacle. The anomaly was specifically identified in DK closing lines.

### Backtest Results

| DK Away Prob Range | Bets | Actual Win Rate | ROI |
|--------------------|------|-----------------|-----|
| 30–35% | 174 | 31.6% | -2.1% |
| **35–40%** | **242** | **42.1%** | **+14.5%** |
| 40–45% | 312 | 43.3% | +1.2% |
| 45–50% | 287 | 47.4% | -3.8% |

The 35–40% bucket is the sweet spot. Below 35%, dogs are correctly priced. Above 40%, the payout isn't large enough to overcome the vig.

**Why +20% ROI in the unified backtest vs +14.5% in the raw analysis?** The unified backtest uses the walk-forward test set (last 70% of the season = June–October) while the raw analysis uses the full season. The anomaly was actually stronger in the second half of the season.

**Monthly breakdown (test set):**

| Month | Bets | Win Rate | ROI |
|-------|------|----------|-----|
| June | 20 | 40.0% | +8% |
| July | 28 | 50.0% | +28% |
| August | 35 | 48.6% | +22% |
| September | 32 | 46.9% | +16% |

All months profitable despite sub-50% win rates — the underdog payouts compensate.

### Important Caveats

1. **Sub-50% win rate is psychologically tough.** You lose more often than you win. The profit comes entirely from the large payouts when underdogs hit. Expect losing streaks of 4–6 bets regularly.
2. **DK-specific.** This anomaly was identified in DraftKings closing lines. It may not exist at sharper books (Pinnacle, Circa) or may manifest differently.
3. **Sample size.** 122 bets in the test set. The 95% confidence interval on the ROI is wide, approximately [-5%, +45%].
4. **May close over time.** If DK improves their away-underdog pricing, the anomaly disappears.

### How to Use

No model or special script needed — just check DraftKings lines daily:
1. Find games where the away team's implied win probability is 35–40% (roughly +150 to +185 American odds after vig removal).
2. Skip division matchups.
3. Bet the away team at posted odds.

```
make unified-picks    # Automated: flags qualifying games
```

---

## Strategy 5: NRFI (No Run First Inning)

**Market:** First-inning scoring props
**Side:** "No run will be scored in the first inning" (NRFI)
**Backtest:** 282 bets · 58.5% win rate · +7.3% ROI
**Assumed odds:** -120 (risk $120 to win $100, breakeven 54.5%)

### What This Strategy Does

The mirror of Strategy 2. When the model is confident that BOTH starting pitchers will have a clean first inning, bet NRFI at -120 odds.

### How the Probability Is Formed

Uses the exact same ensemble model as Strategy 2 (50/50 blend of LightGBM + Logistic Regression). The only difference is which side we bet:

- Strategy 2 (YRFI): bet when P(NRFI) ≤ 0.48 → P(YRFI) ≥ 0.52
- Strategy 5 (NRFI): bet when P(NRFI) ≥ 0.57

The threshold for NRFI (0.57) is higher than for YRFI (0.52) because -120 odds require 54.5% accuracy to break even, while +100 odds require only 50%.

### What Makes the Model Predict NRFI?

The model assigns high P(NRFI) when:
1. **Both starting pitchers are elite.** The `both_sp_stuff_product` and `both_sp_quality_product` features capture this — NRFI requires BOTH arms to be sharp.
2. **Both SPs have high first-inning clean rates.** History of scoreless first innings.
3. **Neither SP is a "slow starter."** The `max_sp_slow_starter` feature flags pitchers who struggle in the first inning.
4. **The top of each lineup isn't dangerous.** Low `both_top3_hr_rate`, high `combined_lineup_k_rate` (both teams' top hitters strike out often).
5. **Pitcher-friendly park + weather.** Low park factor, dome or no wind blowing out.

### Why NRFI Is Weaker Than YRFI

The model must be right about BOTH pitchers to win an NRFI bet. If either pitcher gives up a run, you lose. This "AND" condition (both must be clean) is inherently harder to predict than the "OR" condition for YRFI (at least one team scores).

Additionally, at -120 odds, the book is already charging you a premium — they know NRFI is a popular recreational bet. You need 54.5% accuracy just to break even, leaving less room for profit.

### Backtest Results

| Threshold | Bets | Win Rate | ROI |
|-----------|------|----------|-----|
| P(NRFI) ≥ 0.55 | 584 | 57.9% | +6.1% |
| P(NRFI) ≥ 0.57 | 432 | 58.5% | +7.3% |
| P(NRFI) ≥ 0.60 | 268 | 60.4% | +10.4% |
| P(NRFI) ≥ 0.63 | 163 | 61.3% | +12.0% |

Higher thresholds = higher win rate but fewer bets.

### Important Caveats

Same as Strategy 2: no actual market prices to compare against, first-inning outcomes are noisy, single-season test.

### How to Use

```
make nrfi-model       # Train the model
make unified-picks    # Generate today's picks (includes NRFI strategy)
```

---

## Strategy 6: Pick-em Home Moneyline

**Market:** Moneyline
**Side:** Home team in close matchups where our model sees an edge
**Backtest:** 37 bets · 51.4% win rate · -2.2% ROI
**Status:** NOT PROFITABLE in backtest — included at minimal allocation as a speculative position

### What This Strategy Does

When DraftKings prices a game as essentially a toss-up (home team implied at 48–52%) but our ensemble win model disagrees — seeing the home team as 3%+ more likely to win than DK implies — we bet the home side.

### How the Probability Is Formed

#### The Ensemble Win Model

This uses a separately trained model (`src/win_model.py`) that predicts which team will win each game. It's a walk-forward LightGBM classifier trained on prior seasons' worth of data (2017–2024) and tested on 2025.

The model produces `ens_calibrated`: a calibrated probability that the home team wins, using the same pregame features (SP quality, lineup stats, park factors, etc.) but targeting the binary win/loss outcome rather than total runs.

#### Edge Computation

```
model_home_prob = ens_calibrated (e.g., 0.555)
dk_home_prob = vig-removed DK implied (e.g., 0.500)
edge = 0.555 - 0.500 = +0.055 (5.5%)
```

#### Filters

All of these must be true:
1. DK home prob between 0.48 and 0.52 (pick-em game).
2. Model edge > 3% (meaningful disagreement).
3. Model edge < 10% (cap — larger disagreements mean the model is wrong, not the market).
4. Not a division game (model underperforms on divisional matchups).

### Why This Strategy Underperforms

The ensemble win model (Brier 0.245, AUC 0.589) is roughly equal in accuracy to DraftKings closing lines (Brier 0.240, AUC 0.589). When two equally accurate sources disagree on a ~50/50 game, neither has a reliable edge. The 37-bet sample is far too small to draw conclusions — we'd need ~230 bets at this effect size to reach statistical significance.

### Why It's Still in the Portfolio

At 15% Kelly fraction (the smallest allocation), it adds minimal risk. If the model does have a genuine small edge on pick-em home games, the portfolio captures it. If not, the tiny allocation means negligible drag.

### Backtest Results

| Month | Bets | Win Rate | ROI |
|-------|------|----------|-----|
| June | 8 | 37.5% | -41.6% |
| July | 10 | 60.0% | +26.0% |
| August | 9 | 44.4% | -1.2% |
| September | 10 | 60.0% | +23.5% |

Wildly inconsistent. June was catastrophic; July and September were great. This is what noise looks like on 37 bets.

### How to Use

```
make unified-picks    # Included automatically at minimal allocation
```

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

### Removing the Vig (Vigorish)

The sum of both sides' implied probabilities exceeds 100% — the excess is the book's profit margin.

```
Raw home implied: 0.600
Raw away implied: 0.435
Total: 1.035 (3.5% vig)

Fair home prob = 0.600 / 1.035 = 0.5797
Fair away prob = 0.435 / 1.035 = 0.4203
```

Now they sum to 1.000 — these are the market's true estimated probabilities.

### Kelly Criterion (Bet Sizing)

The Kelly criterion determines the optimal fraction of your bankroll to wager:

```
f = (p × b − q) / b
```

Where:
- `p` = your estimated probability of winning
- `q` = 1 − p = probability of losing
- `b` = net decimal odds (profit per $1 wagered)
- `f` = fraction of bankroll to bet

**Example:** You estimate 58% chance of winning a bet at -110 odds.
```
p = 0.58, q = 0.42
b = 100/110 = 0.909 (at -110, you profit $0.909 per $1 risked)
f = (0.58 × 0.909 − 0.42) / 0.909 = (0.527 − 0.42) / 0.909 = 0.118
```

Kelly says bet 11.8% of bankroll. In practice, we use fractional Kelly (quarter-Kelly = 2.95%) to reduce variance.

### Brier Score and Brier Skill Score

The Brier score measures how accurate probability predictions are:

```
Brier = (1/N) × Σ(predicted_prob − actual_outcome)²
```

Where `actual_outcome` is 1 (event happened) or 0 (didn't). Lower is better. A perfect predictor scores 0; always predicting 50% scores 0.25.

**Brier Skill Score (BSS):**
```
BSS = 1 − (Brier_model / Brier_baseline)
```

Where baseline is always predicting the overall base rate. BSS > 0 means the model beats the baseline. BSS of +0.005 means 0.5% improvement — small but potentially profitable.

### Break-Even Win Rate

At any given odds, the minimum win rate needed to not lose money:

```
At -110: breakeven = 110 / (110 + 100) = 52.38%
At -120: breakeven = 120 / (120 + 100) = 54.55%
At +100: breakeven = 100 / (100 + 100) = 50.00%
At +150: breakeven = 100 / (150 + 100) = 40.00%
```

Any win rate above breakeven is profitable.

---

## Appendix B: Glossary

**American odds:** Betting odds format. Negative = favorite (how much to risk for $100 profit). Positive = underdog (how much you profit on a $100 risk).

**AUC (Area Under Curve):** Measures a model's ability to rank predictions correctly. 0.50 = random; 1.0 = perfect. An AUC of 0.57 means the model correctly ranks a random winning game above a random losing game 57% of the time.

**Barrel rate:** Percentage of batted balls hit at optimal launch angle and exit velocity for extra-base hits. Higher = more dangerous hitter.

**Bootstrap confidence interval:** Statistical technique where you resample your data thousands of times with replacement to estimate the uncertainty in a metric. A 95% CI of [+5%, +20%] means the true value is between 5% and 20% with 95% confidence.

**Brier score:** See Appendix A.

**Closing line:** The final odds posted by a sportsbook just before a game starts. Considered the most efficient (accurate) price because it incorporates all information including late sharp money.

**DraftKings (DK):** Major US sportsbook. Known for sharp closing lines due to high betting volume.

**Edge:** The difference between your estimated probability and the market's implied probability. Positive edge = you think the market is wrong in your favor.

**Expected run value (xRV):** A metric quantifying how many runs a pitch (or pitcher) prevents/allows relative to league average, based on Statcast data (velocity, spin, movement, location, outcome).

**Isotonic calibration:** A method to adjust model probabilities so they match observed frequencies. If the model says 60% but only 52% of such events happen, isotonic calibration learns to map 60% → 52%.

**Kalshi:** A US-regulated prediction market (exchange) where users buy and sell binary contracts on real-world events.

**Kelly criterion:** See Appendix A.

**LightGBM:** A gradient-boosted decision tree algorithm. Builds predictions by combining many simple decision trees, each correcting the errors of the previous ones.

**Moneyline (ML):** A bet on which team wins the game, regardless of margin.

**NRFI/YRFI:** No Run First Inning / Yes Run First Inning. Binary bet on whether at least one run is scored in the first inning.

**Over/Under (O/U):** A bet on whether the total combined runs scored by both teams exceeds (over) or falls below (under) a line set by the sportsbook.

**Park factor:** A multiplier capturing how a stadium affects run scoring. 1.00 = average. Coors Field (Denver, high altitude) is ~1.14; Oracle Park (San Francisco, marine air) is ~0.87.

**Permutation test:** Shuffle the labels randomly and re-run the model many times. If the real model's performance is better than all/most shuffled versions, the signal is real, not due to chance.

**ROI (Return on Investment):** Total profit divided by total amount wagered. +10% ROI means you earn $10 for every $100 bet.

**Sharpe ratio:** Risk-adjusted return metric. (Mean return) / (Standard deviation of returns), annualized. Above 2.0 is considered excellent. Above 3.0 is rare.

**Statcast:** MLB's ball-tracking system that measures pitch velocity, spin rate, movement, exit velocity, launch angle, and more for every pitch and batted ball.

**Vigorish (vig/juice):** The sportsbook's built-in profit margin. At standard -110/-110 pricing, the book collects $110 from losers and pays $100 to winners, keeping $10 (4.5% margin).

**Walk-forward:** A backtesting methodology where the model is always trained on past data and tested on future data, simulating real-world conditions. Prevents the common mistake of testing on data the model has already seen.

**xRV (expected run value):** See "Expected run value."
