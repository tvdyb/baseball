#!/usr/bin/env python3
"""
DraftKings vs Kalshi Moneyline Arbitrage Strategy
==================================================
Formalizes the observation that DK closing lines contain edge over Kalshi
prediction-market prices, and stress-tests it with walk-forward evaluation,
Kelly sizing, ensemble signals, and full risk analysis.

Usage:
    python src/kalshi_arb.py
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


# ---------------------------------------------------------------------------
# 1. DATA LOADING & MERGING
# ---------------------------------------------------------------------------

def american_to_prob(odds: float) -> float:
    """Convert American odds to raw implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def remove_vig(p_home: float, p_away: float) -> tuple[float, float]:
    """Remove vig from raw implied probabilities (normalize to sum to 1)."""
    total = p_home + p_away
    if total == 0:
        return 0.5, 0.5
    return p_home / total, p_away / total


def load_dk(path: Path, max_ml_abs: float = 500.0) -> pd.DataFrame:
    """
    Load DK closing ML lines.

    max_ml_abs: filter out games where |American odds| > this threshold.
    Lines beyond ±500 (i.e., implied prob > ~83%) indicate in-game or
    post-score-update "closing" lines rather than true pregame prices.
    These create spurious edge against Kalshi's pregame prices and are
    excluded to avoid lookahead contamination.
    """
    dk = pd.read_parquet(path)
    dk["game_date"] = dk["game_date"].astype(str)
    # Keep only completed games with both ML lines
    mask = (
        dk["status"].str.startswith("Final") &
        dk["home_ml_close"].notna() &
        dk["away_ml_close"].notna() &
        dk["home_score"].notna() &
        dk["away_score"].notna()
    )
    dk = dk[mask].copy()
    dk["home_win"] = (dk["home_score"] > dk["away_score"]).astype(int)

    # Convert ML to vig-removed probabilities
    dk["raw_p_home"] = dk["home_ml_close"].apply(american_to_prob)
    dk["raw_p_away"] = dk["away_ml_close"].apply(american_to_prob)
    dk[["dk_home_prob", "dk_away_prob"]] = dk.apply(
        lambda r: pd.Series(remove_vig(r["raw_p_home"], r["raw_p_away"])),
        axis=1,
    )

    # Compute vig (overround)
    dk["dk_vig"] = (dk["raw_p_home"] + dk["raw_p_away"]) - 1.0

    # --- Integrity filter: remove in-game / post-result "closing" lines ---
    # Lines with |American odds| > max_ml_abs correspond to implied probabilities
    # beyond ~83%, which only occur when a score is already lopsided mid-game.
    # Keeping these would create spurious edge vs Kalshi's pregame prices.
    extreme_mask = (
        (dk["home_ml_close"].abs() > max_ml_abs) |
        (dk["away_ml_close"].abs() > max_ml_abs)
    )
    n_extreme = extreme_mask.sum()
    if n_extreme > 0:
        print(f"  [filter] Removed {n_extreme} games with |ML| > {max_ml_abs:.0f} "
              f"(likely in-game lines): "
              + ", ".join(dk[extreme_mask]["game_date"].unique()[:5].tolist()))
    dk = dk[~extreme_mask].copy()

    return dk[["game_date", "home_team", "away_team", "home_win",
               "dk_home_prob", "dk_away_prob", "dk_vig",
               "home_ml_close", "away_ml_close"]]


def load_kalshi(path: Path) -> pd.DataFrame:
    kal = pd.read_parquet(path)
    kal["game_date"] = kal["game_date"].astype(str)
    return kal[["game_date", "home_team", "away_team", "home_win",
                "kalshi_home_prob", "kalshi_away_prob", "volume"]]


def load_sim_probs(pkl_path: Path) -> pd.DataFrame:
    """Load MC simulator win probs from the existing backtest artifact."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    df = obj["games"].copy()
    df["game_date"] = df["game_date"].astype(str)
    return df[["game_date", "home_team", "away_team",
               "sim_home_wp"]].rename(columns={"sim_home_wp": "model_home_prob"})


def merge_data(dk: pd.DataFrame, kal: pd.DataFrame,
               sim: pd.DataFrame | None = None) -> pd.DataFrame:
    keys = ["game_date", "home_team", "away_team"]

    # Deduplicate each source before merging: keep row with highest volume
    # (Kalshi can have multiple candle rows per game from different price snapshots)
    if "volume" in kal.columns:
        kal = kal.sort_values("volume", ascending=False).drop_duplicates(
            subset=keys, keep="first"
        )
    else:
        kal = kal.drop_duplicates(subset=keys, keep="first")
    dk = dk.drop_duplicates(subset=keys, keep="first")

    merged = pd.merge(dk, kal, on=keys, suffixes=("_dk", "_kal"))
    # home_win from DK is ground truth; Kalshi's should agree (drop duplicate)
    merged = merged.drop(columns=["home_win_kal"]).rename(
        columns={"home_win_dk": "home_win"}
    )
    if sim is not None:
        sim = sim.drop_duplicates(subset=keys, keep="first")
        merged = pd.merge(merged, sim, on=keys, how="left")
    merged = merged.sort_values("game_date").reset_index(drop=True)
    n_dups = merged.duplicated(subset=keys).sum()
    if n_dups > 0:
        print(f"  [warn] {n_dups} duplicate game keys remain after dedup; dropping")
        merged = merged.drop_duplicates(subset=keys, keep="first")
    return merged


# ---------------------------------------------------------------------------
# 2. EDGE COMPUTATION
# ---------------------------------------------------------------------------

def compute_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute edge on both sides:
      edge_home = dk_home_prob - kalshi_home_prob
      edge_away = dk_away_prob - kalshi_away_prob

    A positive edge means DK thinks this side is more likely than Kalshi does
    => Kalshi is underpricing this side => we buy it on Kalshi.
    """
    df = df.copy()
    df["edge_home"] = df["dk_home_prob"] - df["kalshi_home_prob"]
    df["edge_away"] = df["dk_away_prob"] - df["kalshi_away_prob"]

    # If ensemble sim model is available, blend 50/50
    if "model_home_prob" in df.columns:
        df["ensemble_home_prob"] = (
            0.5 * df["dk_home_prob"] + 0.5 * df["model_home_prob"]
        )
        df["ensemble_away_prob"] = 1.0 - df["ensemble_home_prob"]
        df["edge_ensemble_home"] = df["ensemble_home_prob"] - df["kalshi_home_prob"]
        df["edge_ensemble_away"] = df["ensemble_away_prob"] - df["kalshi_away_prob"]
    return df


# ---------------------------------------------------------------------------
# 3. BET SELECTION
# ---------------------------------------------------------------------------

def select_bets(df: pd.DataFrame, edge_threshold: float,
                use_ensemble: bool = False) -> pd.DataFrame:
    """
    For each game pick at most one side to bet (the one with highest edge
    above threshold). Returns a bet-level dataframe.
    """
    records = []
    for _, row in df.iterrows():
        if use_ensemble and "edge_ensemble_home" in row.index:
            e_home = row["edge_ensemble_home"]
            e_away = row["edge_ensemble_away"]
        else:
            e_home = row["edge_home"]
            e_away = row["edge_away"]

        best_side = None
        best_edge = edge_threshold  # must beat threshold

        if e_home > best_edge:
            best_side = "home"
            best_edge = e_home
        if e_away > best_edge:
            best_side = "away"
            best_edge = e_away

        if best_side is None:
            continue

        if best_side == "home":
            kalshi_prob = row["kalshi_home_prob"]
            won = int(row["home_win"] == 1)
        else:
            kalshi_prob = row["kalshi_away_prob"]
            won = int(row["home_win"] == 0)

        # Kalshi payout: $1 bet at price p wins (1/p - 1) if correct, else -1
        # Kalshi binary contracts: pay $1 on win, cost = prob (in cents/dollar)
        # So: net PnL per $1 wagered = (1/kalshi_prob - 1)*won - (1-won)
        # But more cleanly: pnl_per_unit = (1 - kalshi_prob) if won, else -kalshi_prob
        pnl = (1.0 - kalshi_prob) if won else -kalshi_prob

        records.append({
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "side": best_side,
            "edge": best_edge,
            "kalshi_prob": kalshi_prob,
            "dk_prob": row["dk_home_prob"] if best_side == "home" else row["dk_away_prob"],
            "won": won,
            "pnl_flat": pnl,   # flat $1 stake
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. KELLY CRITERION
# ---------------------------------------------------------------------------

def kelly_fraction(p: float, q: float, b: float) -> float:
    """
    Kelly fraction f = (p*b - q) / b
    where p = win prob (our edge-estimate), b = net odds (payout ratio),
    q = 1 - p.
    For Kalshi: b = (1 - kalshi_prob) / kalshi_prob
    """
    if b <= 0 or p <= 0:
        return 0.0
    q = 1.0 - p
    f = (p * b - q) / b
    return max(0.0, f)


def kelly_pnl(bet: pd.Series, fraction: float = 1.0, bankroll: float = 1.0) -> float:
    """Compute PnL given Kelly fraction and current bankroll."""
    b = (1.0 - bet["kalshi_prob"]) / bet["kalshi_prob"]
    p = bet["dk_prob"]   # our probability estimate
    f = kelly_fraction(p, 1 - p, b) * fraction
    stake = bankroll * f
    if bet["won"]:
        return stake * b
    else:
        return -stake


# ---------------------------------------------------------------------------
# 5. PERFORMANCE METRICS
# ---------------------------------------------------------------------------

def max_drawdown(equity: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown as fraction of peak equity.

    For flat sizing, equity starts at 1.0 (initial bankroll) and each bet
    adds/subtracts $1 worth of P&L.  The drawdown is relative to the running
    peak, so an early run of losses is properly measured against the starting
    bankroll.
    """
    if len(equity) == 0:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def sharpe_ratio(daily_returns: np.ndarray, periods_per_year: float = 162.0) -> float:
    """Annualized Sharpe ratio (assume 162 game-days/year, risk-free = 0)."""
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(periods_per_year)


def max_consecutive_losses(won_series: pd.Series) -> int:
    max_streak = 0
    streak = 0
    for w in won_series:
        if w == 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def flat_max_drawdown_units(pnls: np.ndarray) -> float:
    """
    Max peak-to-trough drawdown for flat-stake betting, in P&L units.

    Returns the maximum cumulative loss from a high-water mark in dollar terms
    (per $1 flat bet).  E.g. a value of 2.5 means you were at most $2.50 below
    your prior peak at some point.  Use total_bets as denominator to convert
    to a percentage-of-volume figure.
    """
    if len(pnls) == 0:
        return 0.0
    cum = np.cumsum(pnls)
    # Prepend 0 so the initial bankroll state is captured
    cum = np.concatenate([[0.0], cum])
    peak = cum[0]
    max_dd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd


def evaluate_bets(bets: pd.DataFrame, sizing: str = "flat",
                  kelly_mult: float = 1.0) -> dict:
    """
    Compute full performance statistics for a set of bets.

    For flat sizing, max_drawdown is reported in P&L units (dollars per $1 bet
    wagered), NOT as a fraction of an initial bankroll.  This avoids the
    nonsensical >100% drawdown that arises when cumulative losses exceed the
    starting bankroll.

    For Kelly sizing, max_drawdown is the standard fraction-of-peak-bankroll.
    """
    if bets.empty:
        return {
            "n_bets": 0, "win_rate": np.nan, "roi": np.nan,
            "max_drawdown": np.nan, "sharpe": np.nan,
            "max_consec_losses": np.nan, "total_pnl": np.nan,
        }

    n = len(bets)
    wins = bets["won"].sum()
    win_rate = wins / n

    if sizing == "flat":
        pnls = bets["pnl_flat"].values
        roi = pnls.mean()
        # Drawdown in absolute P&L units
        dd = flat_max_drawdown_units(pnls)
        cum_pnl = np.concatenate([[0.0], np.cumsum(pnls)])
        equity = cum_pnl  # cumulative P&L (starts at 0, not 1)
    else:
        # Kelly sizing: simulate bankroll
        bankroll = 1.0
        pnls = []
        equity_trace = [1.0]
        for _, bet in bets.iterrows():
            b = (1.0 - bet["kalshi_prob"]) / bet["kalshi_prob"]
            p = bet["dk_prob"]
            f = kelly_fraction(p, 1 - p, b) * kelly_mult
            f = min(f, 0.25)  # cap at 25% per bet for safety
            stake = bankroll * f
            if bet["won"]:
                pnl = stake * b
            else:
                pnl = -stake
            pnls.append(pnl)
            bankroll += pnl
            equity_trace.append(bankroll)
        pnls = np.array(pnls)
        equity = np.array(equity_trace)
        dd = max_drawdown(equity)  # fraction of bankroll peak

    return {
        "n_bets": n,
        "win_rate": win_rate,
        "roi": roi if sizing == "flat" else (equity[-1] - 1.0),
        "max_drawdown": dd,
        "sharpe": sharpe_ratio(pnls),
        "max_consec_losses": max_consecutive_losses(bets["won"]),
        "total_pnl": pnls.sum() if sizing == "flat" else equity[-1] - 1.0,
        "equity": equity,
        "pnls": pnls,
        "bets_df": bets,
    }


# ---------------------------------------------------------------------------
# 6. WALK-FORWARD EVALUATION
# ---------------------------------------------------------------------------

def walk_forward_eval(df: pd.DataFrame,
                      edge_thresholds: list[float],
                      calib_frac: float = 0.30,
                      use_ensemble: bool = False) -> pd.DataFrame:
    """
    Chronological split: first calib_frac is calibration, rest is test.
    Returns a summary table over edge thresholds.
    """
    split = int(len(df) * calib_frac)
    test_df = df.iloc[split:].copy()

    rows = []
    for thresh in edge_thresholds:
        bets = select_bets(test_df, thresh, use_ensemble=use_ensemble)
        stats = evaluate_bets(bets, sizing="flat")

        row = {
            "edge_threshold": thresh,
            "n_bets": stats["n_bets"],
            "win_rate": round(stats["win_rate"] * 100, 1) if not np.isnan(stats.get("win_rate", np.nan)) else np.nan,
            "roi_pct": round(stats["roi"] * 100, 2) if not np.isnan(stats.get("roi", np.nan)) else np.nan,
            "max_dd_units": round(stats["max_drawdown"], 2) if not np.isnan(stats.get("max_drawdown", np.nan)) else np.nan,
            "sharpe": round(stats["sharpe"], 2) if not np.isnan(stats.get("sharpe", np.nan)) else np.nan,
            "max_consec_losses": stats.get("max_consec_losses", np.nan),
            "total_pnl_flat": round(stats.get("total_pnl", np.nan), 2),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. KELLY COMPARISON
# ---------------------------------------------------------------------------

def kelly_comparison(bets: pd.DataFrame) -> pd.DataFrame:
    """Compare flat / half-Kelly / full-Kelly on same bet set."""
    rows = []
    for label, sizing, mult in [
        ("flat", "flat", 1.0),
        ("half_kelly", "kelly", 0.5),
        ("full_kelly", "kelly", 1.0),
    ]:
        stats = evaluate_bets(bets, sizing=sizing, kelly_mult=mult)
        if sizing == "flat":
            dd_str = f"{stats['max_drawdown']:.2f}u"  # in P&L units
            roi = stats["roi"] * 100
            final = 1.0 + stats["total_pnl"]
        else:
            dd_str = f"{stats['max_drawdown']*100:.1f}%"  # fraction of bankroll
            roi = stats["roi"] * 100
            final = stats["roi"] + 1.0
        rows.append({
            "sizing": label,
            "n_bets": stats["n_bets"],
            "final_bankroll": round(final, 4),
            "roi_pct": round(roi, 2),
            "max_drawdown": dd_str,
            "sharpe": round(stats["sharpe"], 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. MONTHLY BREAKDOWN
# ---------------------------------------------------------------------------

def monthly_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    bets = bets.copy()
    bets["month"] = bets["game_date"].str[:7]
    grp = bets.groupby("month")
    rows = []
    for month, g in grp:
        rows.append({
            "month": month,
            "n_bets": len(g),
            "wins": g["won"].sum(),
            "win_rate_pct": round(g["won"].mean() * 100, 1),
            "flat_pnl": round(g["pnl_flat"].sum(), 3),
            "flat_roi_pct": round(g["pnl_flat"].mean() * 100, 2),
            "avg_edge_pct": round(g["edge"].mean() * 100, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. WORST DRAWDOWN PERIOD
# ---------------------------------------------------------------------------

def worst_drawdown_period(bets: pd.DataFrame) -> dict:
    """Find start/end date of worst drawdown period (flat P&L units)."""
    if bets.empty:
        return {}
    pnls = bets["pnl_flat"].values
    cum = np.concatenate([[0.0], np.cumsum(pnls)])
    # dates: prepend None for the initial state
    dates = np.concatenate([["start"], bets["game_date"].values])

    peak_idx = 0
    worst_start = 0
    worst_end = 0
    worst_dd = 0.0
    peak_val = cum[0]

    for i in range(len(cum)):
        if cum[i] > peak_val:
            peak_val = cum[i]
            peak_idx = i
        dd = peak_val - cum[i]
        if dd > worst_dd:
            worst_dd = dd
            worst_start = peak_idx
            worst_end = i

    return {
        "worst_drawdown_units": round(worst_dd, 2),
        "start_date": dates[worst_start] if worst_start < len(dates) else "N/A",
        "end_date": dates[worst_end] if worst_end < len(dates) else "N/A",
        "duration_bets": worst_end - worst_start,
    }


# ---------------------------------------------------------------------------
# 10. MAIN REPORT
# ---------------------------------------------------------------------------

def print_separator(title: str = "", width: int = 72):
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)


def print_df(df: pd.DataFrame, title: str = ""):
    if title:
        print(f"\n{title}")
    print(df.to_string(index=False))
    print()


def main():
    print_separator("DK vs Kalshi Moneyline Arbitrage Strategy")
    print(f"  Root: {ROOT}")
    print()

    # --- Load data ---
    print("Loading data...")
    dk = load_dk(DATA / "odds" / "sbr_mlb_2025.parquet")
    kal = load_kalshi(DATA / "kalshi" / "kalshi_mlb_2025.parquet")
    sim = load_sim_probs(MODELS / "kalshi_backtest_2025.pkl")

    print(f"  DK records (completed games with ML):  {len(dk):,}")
    print(f"  Kalshi records:                        {len(kal):,}")
    print(f"  Simulator records:                     {len(sim):,}")

    # --- Merge ---
    df = merge_data(dk, kal, sim)
    print(f"  Merged (DK x Kalshi):                  {len(df):,} games")

    sim_coverage = df["model_home_prob"].notna().sum()
    print(f"  Sim model coverage:                    {sim_coverage:,} games")
    print()

    # --- Dataset overview ---
    print_separator("Dataset Overview")
    print(f"  Date range:    {df['game_date'].min()}  to  {df['game_date'].max()}")
    print(f"  Home win rate: {df['home_win'].mean():.3f}")
    print(f"  Avg DK vig:    {df['dk_vig'].mean()*100:.2f}%")
    print(f"  Avg Kalshi home_prob: {df['kalshi_home_prob'].mean():.3f}")
    print(f"  Avg DK home_prob:     {df['dk_home_prob'].mean():.3f}")

    # Edge distribution
    df = compute_edges(df)
    print(f"\n  Edge distribution (DK - Kalshi, home side):")
    for pct in [5, 25, 50, 75, 95]:
        v = df["edge_home"].quantile(pct / 100)
        print(f"    p{pct}: {v:+.4f}")
    print()

    # --- Walk-forward evaluation: DK vs Kalshi ---
    print_separator("Walk-Forward Evaluation: DK Closing Line vs Kalshi")
    split_n = int(len(df) * 0.30)
    print(f"  Calibration set: {split_n} games ({df.iloc[:split_n]['game_date'].min()} "
          f"to {df.iloc[:split_n-1]['game_date'].max()})")
    print(f"  Test set:        {len(df) - split_n} games ({df.iloc[split_n]['game_date']} "
          f"to {df.iloc[-1]['game_date']})")
    print()

    edge_thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    wf_dk = walk_forward_eval(df, edge_thresholds, use_ensemble=False)
    print_df(wf_dk, "  DK-vs-Kalshi (flat $1 per bet):")

    # --- Walk-forward: Ensemble ---
    if "edge_ensemble_home" in df.columns:
        print_separator("Walk-Forward Evaluation: Ensemble (50% DK + 50% Sim) vs Kalshi")
        wf_ens = walk_forward_eval(df, edge_thresholds, use_ensemble=True)
        print_df(wf_ens, "  Ensemble-vs-Kalshi (flat $1 per bet):")

        # Comparison at 7% threshold
        print_separator("DK-only vs Ensemble at 7% Edge Threshold")
        thresh_7 = 0.07
        test_df = df.iloc[split_n:].copy()
        bets_dk = select_bets(test_df, thresh_7, use_ensemble=False)
        bets_ens = select_bets(test_df, thresh_7, use_ensemble=True)
        stats_dk = evaluate_bets(bets_dk, sizing="flat")
        stats_ens = evaluate_bets(bets_ens, sizing="flat")
        comp = pd.DataFrame([
            {"strategy": "DK-only", **{k: v for k, v in stats_dk.items()
              if k not in ("equity", "pnls", "bets_df")}},
            {"strategy": "Ensemble", **{k: v for k, v in stats_ens.items()
              if k not in ("equity", "pnls", "bets_df")}},
        ])
        comp_display = comp[["strategy", "n_bets", "win_rate", "roi",
                             "max_drawdown", "sharpe", "max_consec_losses"]].copy()
        comp_display["win_rate"] = comp_display["win_rate"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
        comp_display["roi"] = comp_display["roi"].map(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A")
        comp_display["max_drawdown"] = comp_display["max_drawdown"].map(lambda x: f"{x:.2f}u" if not pd.isna(x) else "N/A")
        comp_display["sharpe"] = comp_display["sharpe"].map(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
        print_df(comp_display)
    else:
        # No sim model available — use just DK for further analysis
        thresh_7 = 0.07
        test_df = df.iloc[split_n:].copy()
        bets_dk = select_bets(test_df, thresh_7, use_ensemble=False)
        bets_ens = bets_dk  # fallback

    # --- Kelly sizing comparison at 7% threshold ---
    print_separator("Kelly Sizing Comparison at 7% Edge Threshold (Test Set)")
    kelly_table = kelly_comparison(bets_dk)
    print_df(kelly_table)

    # --- Monthly breakdown ---
    print_separator("Monthly P&L Breakdown (DK-only, 7% threshold, test set)")
    monthly = monthly_breakdown(bets_dk)
    if not monthly.empty:
        print_df(monthly)
    else:
        print("  No bets in test period.\n")

    # --- Worst drawdown period ---
    print_separator("Drawdown Risk Analysis (DK-only, 7% threshold, test set)")
    dd_period = worst_drawdown_period(bets_dk)
    if dd_period:
        for k, v in dd_period.items():
            print(f"  {k}: {v}")
    print()

    # --- Consecutive loss analysis across all thresholds ---
    print_separator("Risk Summary: Max Consecutive Losses by Threshold (DK-only, test set)")
    test_df_risk = df.iloc[split_n:].copy()
    risk_rows = []
    for thresh in edge_thresholds:
        b = select_bets(test_df_risk, thresh, use_ensemble=False)
        if b.empty:
            risk_rows.append({"edge_threshold": thresh, "n_bets": 0,
                              "max_consec_losses": "N/A", "win_rate": "N/A",
                              "worst_dd_pct": "N/A"})
            continue
        mc = max_consecutive_losses(b["won"])
        wr = b["won"].mean()
        dd_info = worst_drawdown_period(b)
        risk_rows.append({
            "edge_threshold": thresh,
            "n_bets": len(b),
            "max_consec_losses": mc,
            "win_rate_pct": f"{wr*100:.1f}%",
            "worst_dd_units": f"{dd_info.get('worst_drawdown_units', 0):.2f}u",
            "dd_start": dd_info.get("start_date", "N/A"),
            "dd_end": dd_info.get("end_date", "N/A"),
        })
    print_df(pd.DataFrame(risk_rows))

    # --- Calibration check ---
    print_separator("DK vs Kalshi Calibration Check (full merged set)")
    edges = df["edge_home"].values
    bins = [-0.30, -0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10, 0.30]
    labels = ["<-10%", "-10:-5%", "-5:-2%", "-2:0%", "0:2%", "2:5%", "5:10%", ">10%"]
    df["edge_bin"] = pd.cut(df["edge_home"], bins=bins, labels=labels)
    calib = df.groupby("edge_bin", observed=True).agg(
        n=("home_win", "count"),
        actual_win_rate=("home_win", "mean"),
        avg_dk_prob=("dk_home_prob", "mean"),
        avg_kalshi_prob=("kalshi_home_prob", "mean"),
    ).reset_index()
    calib["actual_win_rate"] = calib["actual_win_rate"].map(lambda x: f"{x:.3f}")
    calib["avg_dk_prob"] = calib["avg_dk_prob"].map(lambda x: f"{x:.3f}")
    calib["avg_kalshi_prob"] = calib["avg_kalshi_prob"].map(lambda x: f"{x:.3f}")
    print_df(calib)

    # --- Optimal Kelly fraction distribution ---
    print_separator("Optimal Kelly Fraction Distribution (7% threshold, test set)")
    if not bets_dk.empty:
        kellys = []
        for _, bet in bets_dk.iterrows():
            b = (1.0 - bet["kalshi_prob"]) / bet["kalshi_prob"]
            p = bet["dk_prob"]
            f = kelly_fraction(p, 1 - p, b)
            kellys.append(f)
        kellys = np.array(kellys)
        print(f"  Full Kelly fractions (capped at 25% per bet):")
        for pct in [10, 25, 50, 75, 90, 95]:
            print(f"    p{pct}: {np.percentile(kellys, pct)*100:.2f}%")
        print(f"    mean: {kellys.mean()*100:.2f}%")
        print(f"    max:  {kellys.max()*100:.2f}%")
    print()

    # --- Executive summary ---
    print_separator("EXECUTIVE SUMMARY")
    best_thresh_row = wf_dk.loc[wf_dk["roi_pct"].idxmax()]
    print(f"  Best edge threshold (flat ROI):  {best_thresh_row['edge_threshold']*100:.0f}%")
    print(f"  Bets at best threshold:          {int(best_thresh_row['n_bets'])}")
    print(f"  Win rate at best threshold:      {best_thresh_row['win_rate']:.1f}%")
    print(f"  Flat ROI at best threshold:      {best_thresh_row['roi_pct']:.2f}%")
    print(f"  Max drawdown at best threshold:  {best_thresh_row['max_dd_units']:.2f} units (flat P&L)")
    print(f"  Sharpe at best threshold:        {best_thresh_row['sharpe']:.2f}")
    print()

    if "edge_ensemble_home" in df.columns:
        best_ens_row = wf_ens.loc[wf_ens["roi_pct"].idxmax()]
        print(f"  Ensemble best threshold (flat ROI): {best_ens_row['edge_threshold']*100:.0f}%")
        print(f"  Ensemble best ROI:                  {best_ens_row['roi_pct']:.2f}%")
        print()

    # --- Return structured results ---
    return {
        "merged_df": df,
        "wf_dk": wf_dk,
        "bets_dk_7pct": bets_dk,
        "kelly_table": kelly_table,
        "monthly": monthly,
    }


if __name__ == "__main__":
    results = main()
