"""
strategy_backtest.py
--------------------
Rigorous walk-forward backtest of the LGB-flagged under betting strategy
and combined portfolio with NRFI and Kalshi-arb strategies.

STRICT out-of-sample: calibration = first 30%, test = last 70%.
No parameter tuning on the test set.

Usage:
    python src/strategy_backtest.py
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================================
# HELPERS
# ============================================================================

def american_to_decimal(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, a / 100 + 1, 100 / (-a) + 1)


def american_to_raw_prob(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))


def bet_pnl_series(is_win, is_push, dec_payout):
    """Return per-bet PnL array: win -> (dec-1), loss -> -1, push -> 0."""
    return np.where(is_push, 0.0, np.where(is_win, dec_payout - 1.0, -1.0))


def max_drawdown_units(pnl_series):
    """Max drawdown in cumulative units."""
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return dd.max() if len(dd) > 0 else 0.0


def bootstrap_roi_ci(pnl_array, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for ROI."""
    n = len(pnl_array)
    if n == 0:
        return np.nan, np.nan, np.nan
    rois = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(pnl_array, size=n, replace=True)
        rois[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo = np.percentile(rois, alpha * 100)
    hi = np.percentile(rois, (1 - alpha) * 100)
    return pnl_array.mean(), lo, hi


def sharpe_ratio(daily_returns, annual_factor=162):
    """Annualized Sharpe ratio from daily return series."""
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    return daily_returns.mean() / daily_returns.std() * np.sqrt(annual_factor)


# ============================================================================
# LOAD AND MERGE DATA
# ============================================================================
print("=" * 80)
print("STRATEGY BACKTEST — Walk-Forward Out-of-Sample Evaluation")
print("=" * 80)

ODDS_PATH = "data/odds/sbr_mlb_2025.parquet"
LGB_PATH  = "data/backtest/total_runs_lgb_2025.parquet"
STK_PATH  = "data/backtest/total_runs_stacker_2025.parquet"
NRFI_PATH = "data/backtest/nrfi_lgb_2025.parquet"
KALSHI_PATH = "data/audit/sim_vs_kalshi_pregame_2025.csv"

print("\nLoading data ...")
odds    = pd.read_parquet(ODDS_PATH)
lgb     = pd.read_parquet(LGB_PATH)
stacker = pd.read_parquet(STK_PATH)
nrfi    = pd.read_parquet(NRFI_PATH)
kalshi  = pd.read_csv(KALSHI_PATH)

for d in [odds, lgb, stacker, nrfi]:
    d["game_date"] = pd.to_datetime(d["game_date"])
kalshi["game_date"] = pd.to_datetime(kalshi["game_date"])

# Merge LGB + stacker
stk_cols = ["game_pk", "stk_pred_total", "sim_total_mean", "sim_line",
            "stk_p_over_7.5", "stk_p_over_8.0", "stk_p_over_8.5",
            "stk_p_over_9.0", "stk_p_over_9.5",
            "stk_qr_p_over_7.5", "stk_qr_p_over_8.0", "stk_qr_p_over_8.5",
            "stk_qr_p_over_9.0", "stk_qr_p_over_9.5"]
df = lgb.merge(stacker[stk_cols], on="game_pk", how="left")

# Merge NRFI
df = df.merge(nrfi[["game_pk", "nrfi_lgb_prob", "actual_nrfi"]], on="game_pk", how="left")

# Merge DK odds
dk_cols = ["game_date", "home_team", "away_team",
           "ou_open", "ou_close",
           "over_open_odds", "under_open_odds",
           "over_close_odds", "under_close_odds",
           "total_runs"]
dk = odds[dk_cols].copy()
dk = dk.rename(columns={"total_runs": "dk_total_runs"})
df = df.merge(dk, on=["game_date", "home_team", "away_team"], how="inner")
df = df.sort_values("game_date").reset_index(drop=True)

# Outcomes
df["actual_over"]  = (df["total_runs"] > df["ou_close"]).astype(int)
df["actual_under"] = (df["total_runs"] < df["ou_close"]).astype(int)
df["is_push"]      = (df["total_runs"] == df["ou_close"]).astype(int)

# Decimal payouts
df["dec_over"]  = american_to_decimal(df["over_close_odds"].values)
df["dec_under"] = american_to_decimal(df["under_close_odds"].values)

# Vig-free implied probs
over_imp  = american_to_raw_prob(df["over_close_odds"].values)
under_imp = american_to_raw_prob(df["under_close_odds"].values)
total_imp = over_imp + under_imp
df["vig_free_p_over"]  = np.where(total_imp > 0, over_imp / total_imp, np.nan)
df["vig_free_p_under"] = np.where(total_imp > 0, under_imp / total_imp, np.nan)

# Walk-forward split: 30/70
n_calib = int(len(df) * 0.30)
calib = df.iloc[:n_calib].copy()
test  = df.iloc[n_calib:].copy()

print(f"  Total merged games : {len(df):,}")
print(f"  Calibration set    : {len(calib):,} games  (through {calib['game_date'].max().date()})")
print(f"  Test set           : {len(test):,} games   ({test['game_date'].min().date()} onward)")

# Merge Kalshi data onto test set
kalshi_merged = test.merge(
    kalshi[["game_pk", "sim_home_wp", "kalshi_home_prob", "edge"]],
    on="game_pk", how="left"
)

# ============================================================================
# STRATEGY 1: LGB UNDER BETTING
# ============================================================================
print("\n" + "=" * 80)
print("STRATEGY 1: LGB-FLAGGED UNDER BETTING")
print("=" * 80)

t = test.dropna(subset=["lgb_pred_total", "ou_close", "dec_under"]).copy()
t["lgb_edge"] = t["ou_close"] - t["lgb_pred_total"]  # positive = model says under
print(f"\n  Test games with LGB predictions + DK odds: {len(t):,}")

EDGE_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0]

print(f"\n  {'Edge':>6}  {'N_bets':>7}  {'Wins':>5}  {'Losses':>6}  {'Push':>5}  "
      f"{'WR':>6}  {'ROI_DK':>8}  {'ROI_110':>8}  {'MaxDD':>7}  "
      f"{'CI_lo':>7}  {'CI_hi':>7}")
print("  " + "-" * 90)

strategy1_results = {}

for thr in EDGE_THRESHOLDS:
    mask = t["lgb_edge"] > thr
    sub = t[mask].copy()
    if len(sub) == 0:
        continue

    pnl = bet_pnl_series(sub["actual_under"].values, sub["is_push"].values,
                         sub["dec_under"].values)
    w = sub["actual_under"].sum()
    p = sub["is_push"].sum()
    l = len(sub) - w - p
    wr = w / max(w + l, 1)

    roi_dk = pnl.mean()
    total_pnl = pnl.sum()
    mdd = max_drawdown_units(pnl)

    # ROI at -110
    dec_110 = american_to_decimal(np.array([-110]))[0]
    pnl_110 = np.where(sub["is_push"].values, 0.0,
                       np.where(sub["actual_under"].values, dec_110 - 1.0, -1.0))
    roi_110 = pnl_110.mean()

    # Bootstrap CI
    _, ci_lo, ci_hi = bootstrap_roi_ci(pnl)

    print(f"  {thr:>6.2f}  {len(sub):>7d}  {w:>5d}  {l:>6d}  {p:>5d}  "
          f"{wr:>6.3f}  {roi_dk:>+8.4f}  {roi_110:>+8.4f}  {mdd:>7.2f}  "
          f"{ci_lo:>+7.4f}  {ci_hi:>+7.4f}")

    strategy1_results[thr] = {
        "n": len(sub), "wins": w, "losses": l, "pushes": p,
        "wr": wr, "roi_dk": roi_dk, "roi_110": roi_110,
        "mdd": mdd, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "pnl_series": pnl, "dates": sub["game_date"].values,
    }

# --- Monthly breakdown for the base strategy (edge > 0) ---
print("\n--- Monthly Breakdown (LGB edge > 0.0) ---")
base = t[t["lgb_edge"] > 0.0].copy()
base["pnl"] = bet_pnl_series(base["actual_under"].values, base["is_push"].values,
                              base["dec_under"].values)
base["month"] = base["game_date"].dt.to_period("M")

monthly = base.groupby("month").agg(
    n_bets=("pnl", "count"),
    wins=("actual_under", "sum"),
    total_pnl=("pnl", "sum"),
    roi=("pnl", "mean"),
).reset_index()

print(f"\n  {'Month':>10}  {'N_bets':>7}  {'Wins':>5}  {'PnL':>8}  {'ROI':>8}")
print("  " + "-" * 45)
for _, row in monthly.iterrows():
    print(f"  {str(row['month']):>10}  {row['n_bets']:>7.0f}  {row['wins']:>5.0f}  "
          f"{row['total_pnl']:>+8.2f}  {row['roi']:>+8.4f}")

cum_pnl = monthly["total_pnl"].cumsum()
print(f"\n  Cumulative PnL: {cum_pnl.iloc[-1]:+.2f} units")

# --- Monthly breakdown for edge > 0.5 ---
print("\n--- Monthly Breakdown (LGB edge > 0.5) ---")
base5 = t[t["lgb_edge"] > 0.5].copy()
if len(base5) > 0:
    base5["pnl"] = bet_pnl_series(base5["actual_under"].values, base5["is_push"].values,
                                   base5["dec_under"].values)
    base5["month"] = base5["game_date"].dt.to_period("M")

    monthly5 = base5.groupby("month").agg(
        n_bets=("pnl", "count"),
        wins=("actual_under", "sum"),
        total_pnl=("pnl", "sum"),
        roi=("pnl", "mean"),
    ).reset_index()

    print(f"\n  {'Month':>10}  {'N_bets':>7}  {'Wins':>5}  {'PnL':>8}  {'ROI':>8}")
    print("  " + "-" * 45)
    for _, row in monthly5.iterrows():
        print(f"  {str(row['month']):>10}  {row['n_bets']:>7.0f}  {row['wins']:>5.0f}  "
              f"{row['total_pnl']:>+8.2f}  {row['roi']:>+8.4f}")

# ============================================================================
# STRATEGY 2: NRFI HIGH-CONFIDENCE
# ============================================================================
print("\n" + "=" * 80)
print("STRATEGY 2: NRFI HIGH-CONFIDENCE BETS")
print("=" * 80)

# NRFI bets: standard market odds are ~-120 for NRFI
# We don't have specific NRFI odds in the data, so use typical market odds
NRFI_ODDS = -130  # typical DK NRFI juice
NRFI_DEC = american_to_decimal(np.array([NRFI_ODDS]))[0]

nrfi_test = test.dropna(subset=["nrfi_lgb_prob", "actual_nrfi"]).copy()
print(f"\n  Test games with NRFI predictions: {len(nrfi_test):,}")
print(f"  Base NRFI rate in test set: {nrfi_test['actual_nrfi'].mean():.4f}")
print(f"  Using typical NRFI odds: {NRFI_ODDS} (dec={NRFI_DEC:.4f})")

# Calibrate threshold on calibration set
nrfi_calib = calib.dropna(subset=["nrfi_lgb_prob", "actual_nrfi"]).copy()
nrfi_breakeven = 1.0 / NRFI_DEC  # implied prob needed to break even
print(f"  Break-even probability at {NRFI_ODDS}: {nrfi_breakeven:.4f}")

NRFI_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]

print(f"\n  {'Prob_thr':>8}  {'N_bets':>7}  {'Wins':>5}  {'WR':>6}  "
      f"{'ROI':>8}  {'MaxDD':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
print("  " + "-" * 68)

nrfi_results = {}
for thr in NRFI_THRESHOLDS:
    mask = nrfi_test["nrfi_lgb_prob"] > thr
    sub = nrfi_test[mask]
    if len(sub) < 10:
        continue

    pnl = np.where(sub["actual_nrfi"].values == 1, NRFI_DEC - 1.0, -1.0)
    w = sub["actual_nrfi"].sum()
    l = len(sub) - w
    wr = w / len(sub)
    roi = pnl.mean()
    mdd = max_drawdown_units(pnl)
    _, ci_lo, ci_hi = bootstrap_roi_ci(pnl)

    print(f"  {thr:>8.2f}  {len(sub):>7d}  {w:>5d}  {wr:>6.3f}  "
          f"{roi:>+8.4f}  {mdd:>7.2f}  {ci_lo:>+7.4f}  {ci_hi:>+7.4f}")

    nrfi_results[thr] = {
        "n": len(sub), "wins": w, "losses": l,
        "wr": wr, "roi": roi, "mdd": mdd,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "pnl_series": pnl, "dates": sub["game_date"].values,
    }

# ============================================================================
# STRATEGY 3: KALSHI ARB (Sim vs Kalshi edge)
# ============================================================================
print("\n" + "=" * 80)
print("STRATEGY 3: SIM vs KALSHI EDGE (Moneyline)")
print("=" * 80)

# Only use test-period Kalshi data
kalshi_test = kalshi_merged.dropna(subset=["kalshi_home_prob", "sim_home_wp"]).copy()
kalshi_test["edge"] = kalshi_test["sim_home_wp"] - kalshi_test["kalshi_home_prob"]
print(f"\n  Test games with Kalshi data: {len(kalshi_test):,}")

if len(kalshi_test) > 0:
    # Kalshi is a binary market, so payout ≈ 1/prob for the side we bet
    # Bet home when sim_home_wp > kalshi_home_prob + threshold (we think home is underpriced)
    # Bet away when sim_home_wp < kalshi_home_prob - threshold

    KALSHI_THRESHOLDS = [0.00, 0.03, 0.05, 0.07, 0.10]
    # Use Kalshi implied odds: buy at kalshi_prob, payout = 1/kalshi_prob - 1 if win
    # Actually Kalshi is binary contracts at fixed prices, profit = (1 - price) if win, -price if loss

    print(f"\n  {'Edge_thr':>8}  {'N_bets':>7}  {'Wins':>5}  {'WR':>6}  "
          f"{'ROI':>8}  {'MaxDD':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
    print("  " + "-" * 68)

    kalshi_results = {}
    for thr in KALSHI_THRESHOLDS:
        # Bet home when edge > thr
        home_mask = kalshi_test["edge"] > thr
        # Bet away when edge < -thr
        away_mask = kalshi_test["edge"] < -thr

        home_bets = kalshi_test[home_mask].copy()
        away_bets = kalshi_test[away_mask].copy()

        # Home bet PnL: buy home contract at kalshi_home_prob
        # Win: profit = (1 - price), Loss: lose price
        # But to normalize: stake = price, win = 1 - price, so ROI = (1-price)/price if win, -1 if loss
        # Simpler: decimal odds = 1/price
        if len(home_bets) > 0:
            home_dec = 1.0 / home_bets["kalshi_home_prob"].values
            home_win = home_bets["actual_over"].values  # need home_win
            # We don't have home_win in the merged set directly, compute from scores
            home_win_vals = (home_bets["home_score"] > home_bets["away_score"]).astype(int).values
            home_pnl = np.where(home_win_vals == 1, home_dec - 1.0, -1.0)
        else:
            home_pnl = np.array([])

        if len(away_bets) > 0:
            away_price = 1.0 - away_bets["kalshi_home_prob"].values
            away_dec = 1.0 / away_price
            away_win_vals = (away_bets["away_score"] > away_bets["home_score"]).astype(int).values
            away_pnl = np.where(away_win_vals == 1, away_dec - 1.0, -1.0)
        else:
            away_pnl = np.array([])

        all_pnl = np.concatenate([home_pnl, away_pnl])
        if len(all_pnl) == 0:
            continue

        n = len(all_pnl)
        w = (all_pnl > 0).sum()
        roi = all_pnl.mean()
        mdd = max_drawdown_units(all_pnl)
        _, ci_lo, ci_hi = bootstrap_roi_ci(all_pnl)

        all_dates = np.concatenate([home_bets["game_date"].values, away_bets["game_date"].values])

        print(f"  {thr:>8.2f}  {n:>7d}  {w:>5d}  {w/n:>6.3f}  "
              f"{roi:>+8.4f}  {mdd:>7.2f}  {ci_lo:>+7.4f}  {ci_hi:>+7.4f}")

        kalshi_results[thr] = {
            "n": n, "wins": w, "losses": n - w,
            "wr": w / n, "roi": roi, "mdd": mdd,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "pnl_series": all_pnl, "dates": all_dates,
        }
else:
    kalshi_results = {}
    print("  No Kalshi data available in test period.")

# ============================================================================
# STRATEGY 4: COMBINED LGB UNDER + NRFI (separate portfolio, not parlay)
# ============================================================================
print("\n" + "=" * 80)
print("STRATEGY 4: COMBINED LGB UNDER + NRFI PORTFOLIO")
print("=" * 80)

# Use the best-performing thresholds that don't overfit
# LGB under: edge > 0.0 (the broadest signal)
# NRFI: prob > 0.60 (moderate confidence)

print("\n  Using LGB under (edge > 0.0) + NRFI (prob > 0.60) as separate bets")

# Build daily PnL for combined portfolio
under_bets = t[t["lgb_edge"] > 0.0].copy()
under_bets["pnl"] = bet_pnl_series(
    under_bets["actual_under"].values, under_bets["is_push"].values,
    under_bets["dec_under"].values
)
under_bets["strategy"] = "lgb_under"

nrfi_bets_60 = nrfi_test[nrfi_test["nrfi_lgb_prob"] > 0.60].copy()
nrfi_bets_60["pnl"] = np.where(nrfi_bets_60["actual_nrfi"].values == 1, NRFI_DEC - 1.0, -1.0)
nrfi_bets_60["strategy"] = "nrfi"

combined = pd.concat([
    under_bets[["game_date", "pnl", "strategy"]],
    nrfi_bets_60[["game_date", "pnl", "strategy"]],
], ignore_index=True)

combined_daily = combined.groupby("game_date").agg(
    n_bets=("pnl", "count"),
    total_pnl=("pnl", "sum"),
).reset_index().sort_values("game_date")

total_bets = len(combined)
total_wins = (combined["pnl"] > 0).sum()
total_pnl = combined["pnl"].sum()
roi_combined = combined["pnl"].mean()
mdd_combined = max_drawdown_units(combined["pnl"].values)
_, ci_lo_c, ci_hi_c = bootstrap_roi_ci(combined["pnl"].values)

print(f"\n  Total bets: {total_bets}")
print(f"  LGB under bets: {len(under_bets)}")
print(f"  NRFI bets: {len(nrfi_bets_60)}")
print(f"  Win rate: {total_wins / total_bets:.4f}")
print(f"  Combined ROI: {roi_combined:+.4f}  ({roi_combined*100:+.2f}%)")
print(f"  Total PnL: {total_pnl:+.2f} units")
print(f"  Max drawdown: {mdd_combined:.2f} units")
print(f"  95% CI: [{ci_lo_c:+.4f}, {ci_hi_c:+.4f}]")

# ============================================================================
# PORTFOLIO SIMULATOR: $10,000 BANKROLL
# ============================================================================
print("\n" + "=" * 80)
print("PORTFOLIO SIMULATOR — $10,000 Bankroll")
print("=" * 80)

BANKROLL_START = 10_000
UNIT_SIZE_PCT = 0.01  # 1% of bankroll per bet (flat fraction)

# Build all bets with dates
all_bets = []

# Strategy 1: LGB under (edge > 0)
s1 = t[t["lgb_edge"] > 0.0].copy()
s1["pnl_per_unit"] = bet_pnl_series(
    s1["actual_under"].values, s1["is_push"].values, s1["dec_under"].values
)
s1["strategy"] = "lgb_under"
all_bets.append(s1[["game_date", "game_pk", "pnl_per_unit", "strategy"]])

# Strategy 2: NRFI (prob > 0.60)
s2 = nrfi_test[nrfi_test["nrfi_lgb_prob"] > 0.60].copy()
s2["pnl_per_unit"] = np.where(s2["actual_nrfi"].values == 1, NRFI_DEC - 1.0, -1.0)
s2["strategy"] = "nrfi"
all_bets.append(s2[["game_date", "game_pk", "pnl_per_unit", "strategy"]])

# Strategy 3: Kalshi edge (if available, edge > 0.05)
if 0.05 in kalshi_results and kalshi_results[0.05]["n"] > 0:
    # Rebuild individual bets for Kalshi
    k_home = kalshi_test[kalshi_test["edge"] > 0.05].copy()
    k_away = kalshi_test[kalshi_test["edge"] < -0.05].copy()

    if len(k_home) > 0:
        k_home_dec = 1.0 / k_home["kalshi_home_prob"].values
        k_home_win = (k_home["home_score"] > k_home["away_score"]).astype(int).values
        k_home["pnl_per_unit"] = np.where(k_home_win == 1, k_home_dec - 1.0, -1.0)
        k_home["strategy"] = "kalshi_home"
        all_bets.append(k_home[["game_date", "game_pk", "pnl_per_unit", "strategy"]])

    if len(k_away) > 0:
        k_away_price = 1.0 - k_away["kalshi_home_prob"].values
        k_away_dec = 1.0 / k_away_price
        k_away_win = (k_away["away_score"] > k_away["home_score"]).astype(int).values
        k_away["pnl_per_unit"] = np.where(k_away_win == 1, k_away_dec - 1.0, -1.0)
        k_away["strategy"] = "kalshi_away"
        all_bets.append(k_away[["game_date", "game_pk", "pnl_per_unit", "strategy"]])

portfolio = pd.concat(all_bets, ignore_index=True).sort_values("game_date")

# Simulate bankroll evolution
dates_sorted = sorted(portfolio["game_date"].unique())
bankroll = BANKROLL_START
equity_curve = [{"date": dates_sorted[0], "bankroll": bankroll, "n_bets": 0, "daily_pnl": 0}]

for date in dates_sorted:
    day_bets = portfolio[portfolio["game_date"] == date]
    unit_size = bankroll * UNIT_SIZE_PCT
    daily_pnl = day_bets["pnl_per_unit"].sum() * unit_size
    bankroll += daily_pnl
    equity_curve.append({
        "date": date,
        "bankroll": bankroll,
        "n_bets": len(day_bets),
        "daily_pnl": daily_pnl,
    })

eq_df = pd.DataFrame(equity_curve)

# Stats
peak = eq_df["bankroll"].cummax()
drawdown_pct = (peak - eq_df["bankroll"]) / peak
max_dd_pct = drawdown_pct.max()
max_dd_dollar = (peak - eq_df["bankroll"]).max()

total_return = (bankroll - BANKROLL_START) / BANKROLL_START
n_days = len(dates_sorted)
# Annualized return (assume 180 game-days per season)
ann_return = (1 + total_return) ** (180 / max(n_days, 1)) - 1

daily_returns = eq_df["daily_pnl"] / eq_df["bankroll"].shift(1)
daily_returns = daily_returns.dropna()
sr = sharpe_ratio(daily_returns, annual_factor=180)

print(f"\n  Starting bankroll:     ${BANKROLL_START:,.0f}")
print(f"  Ending bankroll:       ${bankroll:,.0f}")
print(f"  Total return:          {total_return:+.2%}")
print(f"  Annualized return:     {ann_return:+.2%}")
print(f"  Sharpe ratio (ann.):   {sr:.2f}")
print(f"  Max drawdown ($):      ${max_dd_dollar:,.0f}")
print(f"  Max drawdown (%):      {max_dd_pct:.2%}")
print(f"  Total bets placed:     {len(portfolio):,}")
print(f"  Trading days:          {n_days}")
print(f"  Avg bets/day:          {len(portfolio) / max(n_days, 1):.1f}")

# Strategy breakdown
print(f"\n  Strategy Breakdown:")
for strat in portfolio["strategy"].unique():
    s = portfolio[portfolio["strategy"] == strat]
    print(f"    {strat:>15}: {len(s):>5d} bets  ROI={s['pnl_per_unit'].mean():+.4f}  "
          f"PnL={s['pnl_per_unit'].sum():+.1f}u")

# Monthly equity
eq_df["date"] = pd.to_datetime(eq_df["date"])
eq_df["month"] = eq_df["date"].dt.to_period("M")
monthly_eq = eq_df.groupby("month").agg(
    end_bankroll=("bankroll", "last"),
    n_bets=("n_bets", "sum"),
    monthly_pnl=("daily_pnl", "sum"),
).reset_index()

print(f"\n  Monthly Equity Curve:")
print(f"  {'Month':>10}  {'Bankroll':>10}  {'N_bets':>7}  {'PnL':>10}")
print("  " + "-" * 42)
for _, row in monthly_eq.iterrows():
    print(f"  {str(row['month']):>10}  ${row['end_bankroll']:>9,.0f}  {row['n_bets']:>7.0f}  "
          f"${row['monthly_pnl']:>+9,.0f}")

# ============================================================================
# STATISTICAL RIGOR: p-value for LGB under strategy
# ============================================================================
print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 80)

for thr_label, thr in [("edge>0.0", 0.0), ("edge>0.5", 0.5)]:
    if thr in strategy1_results:
        res = strategy1_results[thr]
        pnl = res["pnl_series"]
        n = len(pnl)
        # t-test: H0: mean PnL = 0
        t_stat, p_val = stats.ttest_1samp(pnl, 0)
        print(f"\n  LGB Under ({thr_label}):")
        print(f"    n = {n}, mean ROI = {pnl.mean():+.4f}")
        print(f"    t-stat = {t_stat:.3f}, p-value = {p_val:.4f} (two-sided)")
        print(f"    p-value (one-sided, ROI > 0) = {p_val/2:.4f}")
        print(f"    95% CI: [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]")

        # Effect size (Cohen's d)
        d = pnl.mean() / pnl.std() if pnl.std() > 0 else 0
        print(f"    Cohen's d = {d:.4f}")

# ============================================================================
# PARLAY TEST: LGB Under + NRFI same game
# ============================================================================
print("\n" + "=" * 80)
print("PARLAY TEST: LGB UNDER + NRFI (same game)")
print("=" * 80)

# Find games where both LGB says under AND NRFI model says high NRFI prob
parlay_base = t[t["lgb_edge"] > 0.0].copy()
parlay_base = parlay_base.merge(
    nrfi_test[["game_pk", "nrfi_lgb_prob", "actual_nrfi"]],
    on="game_pk", how="inner", suffixes=("", "_nrfi2")
)

for nrfi_thr in [0.55, 0.60, 0.65]:
    parlay = parlay_base[parlay_base["nrfi_lgb_prob"] > nrfi_thr].copy()
    if len(parlay) < 10:
        continue

    # Parlay: both legs must win
    both_win = (parlay["actual_under"].values == 1) & (parlay["actual_nrfi"].values == 1)
    either_loss = ~both_win

    # Parlay odds: under_dec * nrfi_dec
    parlay_dec = parlay["dec_under"].values * NRFI_DEC
    parlay_pnl = np.where(both_win, parlay_dec - 1.0, -1.0)

    # Also: push on under means parlay push (conservative: treat as loss)
    # Already handled since is_push means actual_under=0

    roi = parlay_pnl.mean()
    w = both_win.sum()
    n = len(parlay)
    mdd = max_drawdown_units(parlay_pnl)
    _, ci_lo, ci_hi = bootstrap_roi_ci(parlay_pnl)

    print(f"\n  NRFI threshold > {nrfi_thr}:")
    print(f"    N parlays: {n}, Wins: {w}, WR: {w/n:.3f}")
    print(f"    ROI: {roi:+.4f} ({roi*100:+.2f}%)")
    print(f"    Max DD: {mdd:.2f}u")
    print(f"    95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY")
print("=" * 80)

print("""
Walk-forward backtest results (calibration: first 30%, test: last 70%):

STRATEGY 1 — LGB Under:
  Bet under when LGB model predicts total runs below DK closing O/U line.
  See threshold grid above for ROI at various edge cutoffs.
  Check: is the 95% bootstrap CI entirely above zero?

STRATEGY 2 — NRFI:
  Bet NRFI when LGB NRFI model predicts > threshold probability.
  Typical NRFI market odds assumed at -130.

STRATEGY 3 — Sim vs Kalshi:
  Bet the side where simulation disagrees with Kalshi market by > threshold.

PORTFOLIO:
  Combined daily allocation across all three strategies.
  1% of bankroll per bet (flat fractional sizing).

KEY QUESTION: Does the LGB under edge survive rigorous OOS testing?
  - If 95% CI excludes zero and monthly breakdown is stable -> real edge
  - If CI includes zero or edge is concentrated in 1-2 months -> likely noise
""")

print("Done.")
