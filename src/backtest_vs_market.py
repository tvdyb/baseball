"""
Market comparison backtest: model predictions vs DraftKings closing lines.
Measures profitability (ROI) for O/U totals, ML, and Kalshi markets.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ── helpers ──────────────────────────────────────────────────────────────────

def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


def remove_vig(p1: float, p2: float):
    """Remove vig from a two-outcome market. Returns (fair_p1, fair_p2)."""
    total = p1 + p2
    if total == 0 or pd.isna(total):
        return np.nan, np.nan
    return p1 / total, p2 / total


def kelly_fraction(edge: float, odds_decimal: float) -> float:
    """Full Kelly fraction: f* = edge / (odds - 1).  Capped at 5% max."""
    if odds_decimal <= 1 or edge <= 0:
        return 0.0
    f = edge / (odds_decimal - 1.0)
    return min(f, 0.05)


def decimal_from_american(odds: float) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / (-odds)


# ── data loading ─────────────────────────────────────────────────────────────

def load_all():
    """Load and merge all data sources."""

    # 1) DraftKings odds
    odds = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    odds["game_date"] = pd.to_datetime(odds["game_date"])
    # Drop spring training / incomplete
    odds = odds[odds["status"] == "Final"].copy()
    odds = odds.dropna(subset=["ou_close", "over_close_odds", "under_close_odds"])

    # 2) MC sim backtest (500 games)
    sim = pd.read_parquet(DATA / "backtest" / "nrfi_ou_backtest_2025.parquet")
    sim["game_date"] = pd.to_datetime(sim["game_date"])

    # 3) LGB total runs
    lgb = pd.read_parquet(DATA / "backtest" / "total_runs_lgb_2025.parquet")
    lgb["game_date"] = pd.to_datetime(lgb["game_date"])

    # 4) Stacker total runs
    stk = pd.read_parquet(DATA / "backtest" / "total_runs_stacker_2025.parquet")
    stk["game_date"] = pd.to_datetime(stk["game_date"])

    # 5) NRFI LGB
    nrfi = pd.read_parquet(DATA / "backtest" / "nrfi_lgb_2025.parquet")
    nrfi["game_date"] = pd.to_datetime(nrfi["game_date"])

    # 6) Kalshi ML
    kalshi = pd.read_parquet(DATA / "kalshi" / "kalshi_mlb_2025.parquet")
    kalshi["game_date"] = pd.to_datetime(kalshi["game_date"])

    return odds, sim, lgb, stk, nrfi, kalshi


def merge_ou(odds, sim, lgb, stk):
    """Merge O/U data: odds + model predictions."""
    keys = ["game_date", "home_team", "away_team"]

    # Start from odds
    df = odds[keys + ["ou_close", "over_close_odds", "under_close_odds",
                       "home_ml_close", "away_ml_close",
                       "total_runs", "home_score", "away_score"]].copy()

    # Market implied probs for O/U
    df["mkt_p_over_raw"] = df["over_close_odds"].apply(american_to_prob)
    df["mkt_p_under_raw"] = df["under_close_odds"].apply(american_to_prob)
    df[["mkt_p_over", "mkt_p_under"]] = df.apply(
        lambda r: pd.Series(remove_vig(r["mkt_p_over_raw"], r["mkt_p_under_raw"])),
        axis=1)

    # Market implied probs for ML
    df["mkt_home_ml_raw"] = df["home_ml_close"].apply(american_to_prob)
    df["mkt_away_ml_raw"] = df["away_ml_close"].apply(american_to_prob)
    df[["mkt_home_ml", "mkt_away_ml"]] = df.apply(
        lambda r: pd.Series(remove_vig(r["mkt_home_ml_raw"], r["mkt_away_ml_raw"])),
        axis=1)

    # Actual outcomes
    df["actual_over"] = (df["total_runs"] > df["ou_close"]).astype(int)
    df["actual_under"] = (df["total_runs"] < df["ou_close"]).astype(int)
    df["push"] = (df["total_runs"] == df["ou_close"]).astype(int)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Decimal odds for payout calculations
    df["over_dec"] = df["over_close_odds"].apply(decimal_from_american)
    df["under_dec"] = df["under_close_odds"].apply(decimal_from_american)
    df["home_ml_dec"] = df["home_ml_close"].apply(decimal_from_american)
    df["away_ml_dec"] = df["away_ml_close"].apply(decimal_from_american)

    # Merge sim (left join — sim only covers 500 games)
    sim_cols = ["game_date", "home_team", "away_team",
                "sim_total_mean", "sim_home_wp", "sim_nrfi_prob"]
    df = df.merge(sim[sim_cols], on=keys, how="left")

    # Merge LGB
    lgb_cols = ["game_date", "home_team", "away_team",
                "lgb_pred_total", "lgb_residual_std"] + \
               [c for c in lgb.columns if c.startswith("lgb_p_over_")]
    df = df.merge(lgb[lgb_cols], on=keys, how="left")

    # Merge stacker
    stk_cols = ["game_date", "home_team", "away_team",
                "stk_pred_total"] + \
               [c for c in stk.columns if c.startswith("stk_p_over_") or c.startswith("stk_qr_p_over_")]
    df = df.merge(stk[stk_cols], on=keys, how="left")

    df = df.sort_values("game_date").reset_index(drop=True)
    return df


# ── P(over) computation ─────────────────────────────────────────────────────

def compute_p_over_sim(row):
    """P(over) from MC sim using normal approximation with lgb_residual_std."""
    mean = row.get("sim_total_mean")
    std = row.get("lgb_residual_std")  # use LGB residual std as proxy
    line = row["ou_close"]
    if pd.isna(mean) or pd.isna(std) or std <= 0:
        return np.nan
    # P(total > line) = 1 - Phi((line - mean) / std)
    # For half-point lines, exact; for whole-number lines, use continuity correction
    if line == int(line):
        return 1.0 - norm.cdf((line + 0.5 - mean) / std)
    else:
        return 1.0 - norm.cdf((line - mean) / std)


def compute_p_over_lgb(row):
    """P(over) from LGB using the pre-computed column matching ou_close."""
    line = row["ou_close"]
    col = f"lgb_p_over_{line}"
    val = row.get(col)
    if pd.notna(val):
        return val
    # Fallback: normal approx from lgb_pred_total and lgb_residual_std
    mean = row.get("lgb_pred_total")
    std = row.get("lgb_residual_std")
    if pd.isna(mean) or pd.isna(std) or std <= 0:
        return np.nan
    if line == int(line):
        return 1.0 - norm.cdf((line + 0.5 - mean) / std)
    else:
        return 1.0 - norm.cdf((line - mean) / std)


def compute_p_over_stacker(row):
    """P(over) from stacker using pre-computed column matching ou_close."""
    line = row["ou_close"]
    col = f"stk_p_over_{line}"
    val = row.get(col)
    if pd.notna(val):
        return val
    # Fallback: use QR version
    col_qr = f"stk_qr_p_over_{line}"
    val_qr = row.get(col_qr)
    if pd.notna(val_qr):
        return val_qr
    return np.nan


# ── betting simulation ───────────────────────────────────────────────────────

def simulate_flat_bets(df, model_p_over_col, thresholds, split_frac=0.30):
    """
    Flat $100 bet simulation.
    - Calibration: first split_frac of data (by date order)
    - Test: remaining data
    - Bet over when edge > threshold, under when edge < -threshold
    - Pushes refunded (no P&L)
    Returns dict of {threshold: metrics_dict}
    """
    df_valid = df.dropna(subset=[model_p_over_col, "mkt_p_over"]).copy()
    df_valid = df_valid[df_valid["push"] == 0].copy()  # exclude pushes for clarity

    n = len(df_valid)
    if n == 0:
        return {}

    split_idx = int(n * split_frac)
    test = df_valid.iloc[split_idx:].copy()

    if len(test) == 0:
        return {}

    test["edge"] = test[model_p_over_col] - test["mkt_p_over"]

    results = {}
    for thr in thresholds:
        # Over bets: edge > threshold
        over_mask = test["edge"] > thr
        under_mask = test["edge"] < -thr

        bets = []
        for _, row in test[over_mask].iterrows():
            pnl = (row["over_dec"] - 1) * 100 if row["actual_over"] == 1 else -100
            bets.append({"side": "over", "pnl": pnl, "edge": row["edge"],
                         "dec_odds": row["over_dec"]})

        for _, row in test[under_mask].iterrows():
            pnl = (row["under_dec"] - 1) * 100 if row["actual_under"] == 1 else -100
            bets.append({"side": "under", "pnl": pnl, "edge": -row["edge"],
                         "dec_odds": row["under_dec"]})

        if len(bets) == 0:
            results[thr] = {"n_bets": 0, "roi": np.nan, "win_rate": np.nan,
                            "pnl": 0, "kelly_roi": np.nan}
            continue

        bets_df = pd.DataFrame(bets)
        total_pnl = bets_df["pnl"].sum()
        n_bets = len(bets_df)
        wins = (bets_df["pnl"] > 0).sum()
        roi = total_pnl / (n_bets * 100)
        win_rate = wins / n_bets

        # Kelly sizing
        kelly_pnl = 0
        bankroll = 10000
        for _, b in bets_df.iterrows():
            frac = kelly_fraction(b["edge"], b["dec_odds"])
            wager = bankroll * frac
            if b["pnl"] > 0:
                kelly_pnl += wager * (b["dec_odds"] - 1)
            else:
                kelly_pnl -= wager
        kelly_roi = kelly_pnl / 10000

        results[thr] = {
            "n_bets": n_bets,
            "roi": roi,
            "win_rate": win_rate,
            "pnl": total_pnl,
            "kelly_roi": kelly_roi,
        }

    return results


def simulate_ml_bets(df, model_home_col, thresholds, split_frac=0.30):
    """ML flat bet simulation."""
    df_valid = df.dropna(subset=[model_home_col, "mkt_home_ml"]).copy()
    # Exclude ties (shouldn't happen in MLB, but safety)
    df_valid = df_valid[df_valid["home_score"] != df_valid["away_score"]].copy()

    n = len(df_valid)
    if n == 0:
        return {}

    split_idx = int(n * split_frac)
    test = df_valid.iloc[split_idx:].copy()

    if len(test) == 0:
        return {}

    test["home_edge"] = test[model_home_col] - test["mkt_home_ml"]
    test["away_edge"] = (1 - test[model_home_col]) - test["mkt_away_ml"]

    results = {}
    for thr in thresholds:
        bets = []
        for _, row in test.iterrows():
            if row["home_edge"] > thr:
                pnl = (row["home_ml_dec"] - 1) * 100 if row["home_win"] == 1 else -100
                bets.append({"side": "home", "pnl": pnl, "edge": row["home_edge"],
                             "dec_odds": row["home_ml_dec"]})
            elif row["away_edge"] > thr:
                pnl = (row["away_ml_dec"] - 1) * 100 if row["home_win"] == 0 else -100
                bets.append({"side": "away", "pnl": pnl, "edge": row["away_edge"],
                             "dec_odds": row["away_ml_dec"]})

        if len(bets) == 0:
            results[thr] = {"n_bets": 0, "roi": np.nan, "win_rate": np.nan,
                            "pnl": 0, "kelly_roi": np.nan}
            continue

        bets_df = pd.DataFrame(bets)
        total_pnl = bets_df["pnl"].sum()
        n_bets = len(bets_df)
        wins = (bets_df["pnl"] > 0).sum()
        roi = total_pnl / (n_bets * 100)
        win_rate = wins / n_bets

        kelly_pnl = 0
        bankroll = 10000
        for _, b in bets_df.iterrows():
            frac = kelly_fraction(b["edge"], b["dec_odds"])
            wager = bankroll * frac
            if b["pnl"] > 0:
                kelly_pnl += wager * (b["dec_odds"] - 1)
            else:
                kelly_pnl -= wager
        kelly_roi = kelly_pnl / 10000

        results[thr] = {
            "n_bets": n_bets,
            "roi": roi,
            "win_rate": win_rate,
            "pnl": total_pnl,
            "kelly_roi": kelly_roi,
        }

    return results


# ── reporting ────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_bet_table(results_by_model, thresholds):
    """Pretty-print betting results for multiple models across thresholds."""
    header = f"{'Threshold':>10} | {'Model':>20} | {'Bets':>6} | {'Win Rate':>9} | {'ROI':>9} | {'P&L ($)':>10} | {'Kelly ROI':>10}"
    print(header)
    print("-" * len(header))
    for thr in thresholds:
        for model_name, res_dict in results_by_model.items():
            r = res_dict.get(thr)
            if r is None:
                continue
            if r["n_bets"] == 0:
                print(f"{thr:>9.0%} | {model_name:>20} | {'0':>6} | {'N/A':>9} | {'N/A':>9} | {'$0':>10} | {'N/A':>10}")
            else:
                print(f"{thr:>9.0%} | {model_name:>20} | {r['n_bets']:>6} | {r['win_rate']:>8.1%} | {r['roi']:>+8.1%} | ${r['pnl']:>+9.0f} | {r['kelly_roi']:>+9.1%}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print_section("LOADING DATA")
    odds, sim, lgb, stk, nrfi, kalshi = load_all()
    print(f"  DraftKings odds:     {len(odds):>5} games")
    print(f"  MC sim backtest:     {len(sim):>5} games")
    print(f"  LGB total runs:      {len(lgb):>5} games")
    print(f"  Stacker total runs:  {len(stk):>5} games")
    print(f"  NRFI LGB:            {len(nrfi):>5} games")
    print(f"  Kalshi ML:           {len(kalshi):>5} games")

    df = merge_ou(odds, sim, lgb, stk)
    print(f"\n  Merged dataset:      {len(df):>5} games")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    # ── Compute model P(over) for each model ─────────────────────────────
    print_section("COMPUTING MODEL P(OVER)")

    df["sim_p_over"] = df.apply(compute_p_over_sim, axis=1)
    df["lgb_p_over"] = df.apply(compute_p_over_lgb, axis=1)
    df["stk_p_over"] = df.apply(compute_p_over_stacker, axis=1)

    # Ensemble: average of available models
    model_cols = ["sim_p_over", "lgb_p_over", "stk_p_over"]
    df["ens_p_over"] = df[model_cols].mean(axis=1, skipna=True)

    for col in model_cols + ["ens_p_over"]:
        n_valid = df[col].notna().sum()
        print(f"  {col:>15}: {n_valid:>5} games with predictions")

    # ── O/U PROFITABILITY ────────────────────────────────────────────────
    print_section("O/U PROFITABILITY vs DraftKings (test set = last 70%)")

    thresholds = [0.02, 0.03, 0.05, 0.07, 0.10]

    ou_models = {
        "MC Sim": "sim_p_over",
        "LGB": "lgb_p_over",
        "Stacker": "stk_p_over",
        "Ensemble (avg)": "ens_p_over",
    }

    ou_results = {}
    for name, col in ou_models.items():
        ou_results[name] = simulate_flat_bets(df, col, thresholds)

    print_bet_table(ou_results, thresholds)

    # ── Edge distribution ────────────────────────────────────────────────
    print_section("O/U EDGE DISTRIBUTION (test set)")
    for name, col in ou_models.items():
        valid = df.dropna(subset=[col, "mkt_p_over"])
        split_idx = int(len(valid) * 0.30)
        test = valid.iloc[split_idx:]
        edge = test[col] - test["mkt_p_over"]
        print(f"  {name:>20}: mean={edge.mean():+.4f}  std={edge.std():.4f}  "
              f"median={edge.median():+.4f}  [5%={edge.quantile(0.05):+.4f}, 95%={edge.quantile(0.95):+.4f}]")

    # ── ML PROFITABILITY vs DraftKings ───────────────────────────────────
    print_section("ML PROFITABILITY vs DraftKings (test set = last 70%)")

    ml_thresholds = [0.02, 0.03, 0.05, 0.07, 0.10]

    # sim_home_wp is our ML model prediction
    ml_models = {"MC Sim WP": "sim_home_wp"}

    ml_results = {}
    for name, col in ml_models.items():
        ml_results[name] = simulate_ml_bets(df, col, ml_thresholds)

    print_bet_table(ml_results, ml_thresholds)

    # ── ML vs Kalshi ─────────────────────────────────────────────────────
    print_section("ML PROFITABILITY vs KALSHI")

    # Merge Kalshi with odds to get actual outcomes
    kalshi_merge = kalshi.merge(
        odds[["game_date", "home_team", "away_team", "home_score", "away_score",
              "home_ml_close", "away_ml_close"]],
        on=["game_date", "home_team", "away_team"],
        how="inner"
    )
    kalshi_merge["home_win"] = (kalshi_merge["home_score"] > kalshi_merge["away_score"]).astype(int)

    # Also merge sim predictions
    sim_cols = ["game_date", "home_team", "away_team", "sim_home_wp"]
    kalshi_merge = kalshi_merge.merge(
        sim[sim_cols], on=["game_date", "home_team", "away_team"], how="left"
    )

    print(f"  Kalshi games with outcomes: {len(kalshi_merge)}")
    print(f"  Kalshi games with sim WP:   {kalshi_merge['sim_home_wp'].notna().sum()}")

    # Bet against Kalshi: model vs Kalshi implied prob
    kalshi_thresholds = [0.02, 0.03, 0.05, 0.07, 0.10]

    # Since Kalshi is a prediction market, assume ~no vig (or minimal)
    # Use kalshi probs directly as the "fair" line
    kalshi_results = {}

    for model_name, model_col, market_col in [
        ("Sim vs Kalshi", "sim_home_wp", "kalshi_home_prob"),
    ]:
        valid = kalshi_merge.dropna(subset=[model_col, market_col]).copy()
        valid = valid[valid["home_score"] != valid["away_score"]].copy()

        n = len(valid)
        split_idx = int(n * 0.30)
        test = valid.iloc[split_idx:].copy()

        test["home_edge"] = test[model_col] - test[market_col]
        test["away_edge"] = (1 - test[model_col]) - test[market_col.replace("home", "away")]

        res = {}
        for thr in kalshi_thresholds:
            bets = []
            for _, row in test.iterrows():
                if row["home_edge"] > thr:
                    # Kalshi payout is binary: risk price to win (1 - price)
                    price = row[market_col]
                    if row["home_win"] == 1:
                        pnl = (1 - price) / price * 100  # normalize to $100 risked
                    else:
                        pnl = -100
                    bets.append({"side": "home", "pnl": pnl, "edge": row["home_edge"],
                                 "dec_odds": 1 / price if price > 0 else np.nan})
                elif row["away_edge"] > thr:
                    price = row[market_col.replace("home", "away")]
                    if row["home_win"] == 0:
                        pnl = (1 - price) / price * 100
                    else:
                        pnl = -100
                    bets.append({"side": "away", "pnl": pnl, "edge": row["away_edge"],
                                 "dec_odds": 1 / price if price > 0 else np.nan})

            if len(bets) == 0:
                res[thr] = {"n_bets": 0, "roi": np.nan, "win_rate": np.nan,
                            "pnl": 0, "kelly_roi": np.nan}
                continue

            bets_df = pd.DataFrame(bets)
            total_pnl = bets_df["pnl"].sum()
            n_bets = len(bets_df)
            wins = (bets_df["pnl"] > 0).sum()
            roi = total_pnl / (n_bets * 100)
            win_rate = wins / n_bets

            kelly_pnl = 0
            bankroll = 10000
            for _, b in bets_df.iterrows():
                frac = kelly_fraction(b["edge"], b["dec_odds"])
                wager = bankroll * frac
                if b["pnl"] > 0:
                    kelly_pnl += wager * (b["dec_odds"] - 1)
                else:
                    kelly_pnl -= wager
            kelly_roi = kelly_pnl / 10000

            res[thr] = {
                "n_bets": n_bets, "roi": roi, "win_rate": win_rate,
                "pnl": total_pnl, "kelly_roi": kelly_roi,
            }

        kalshi_results[model_name] = res

    # Also: DK ML vs Kalshi (is DK a better line?)
    dk_kalshi_valid = kalshi_merge.dropna(subset=["home_ml_close", "kalshi_home_prob"]).copy()
    dk_kalshi_valid = dk_kalshi_valid[dk_kalshi_valid["home_score"] != dk_kalshi_valid["away_score"]].copy()
    dk_kalshi_valid["dk_home_prob_raw"] = dk_kalshi_valid["home_ml_close"].apply(american_to_prob)
    dk_kalshi_valid["dk_away_prob_raw"] = dk_kalshi_valid["away_ml_close"].apply(american_to_prob)
    dk_kalshi_valid[["dk_home_prob", "dk_away_prob"]] = dk_kalshi_valid.apply(
        lambda r: pd.Series(remove_vig(r["dk_home_prob_raw"], r["dk_away_prob_raw"])), axis=1)

    n = len(dk_kalshi_valid)
    split_idx = int(n * 0.30)
    dk_test = dk_kalshi_valid.iloc[split_idx:].copy()

    dk_test["home_edge"] = dk_test["dk_home_prob"] - dk_test["kalshi_home_prob"]
    dk_test["away_edge"] = dk_test["dk_away_prob"] - dk_test["kalshi_away_prob"]

    res = {}
    for thr in kalshi_thresholds:
        bets = []
        for _, row in dk_test.iterrows():
            if row["home_edge"] > thr:
                price = row["kalshi_home_prob"]
                if row["home_win"] == 1:
                    pnl = (1 - price) / price * 100
                else:
                    pnl = -100
                bets.append({"side": "home", "pnl": pnl, "edge": row["home_edge"],
                             "dec_odds": 1 / price if price > 0 else np.nan})
            elif row["away_edge"] > thr:
                price = row["kalshi_away_prob"]
                if row["home_win"] == 0:
                    pnl = (1 - price) / price * 100
                else:
                    pnl = -100
                bets.append({"side": "away", "pnl": pnl, "edge": row["away_edge"],
                             "dec_odds": 1 / price if price > 0 else np.nan})

        if len(bets) == 0:
            res[thr] = {"n_bets": 0, "roi": np.nan, "win_rate": np.nan,
                        "pnl": 0, "kelly_roi": np.nan}
            continue

        bets_df = pd.DataFrame(bets)
        total_pnl = bets_df["pnl"].sum()
        n_bets_val = len(bets_df)
        wins = (bets_df["pnl"] > 0).sum()

        kelly_pnl = 0
        bankroll = 10000
        for _, b in bets_df.iterrows():
            frac = kelly_fraction(b["edge"], b["dec_odds"])
            wager = bankroll * frac
            if b["pnl"] > 0:
                kelly_pnl += wager * (b["dec_odds"] - 1)
            else:
                kelly_pnl -= wager

        res[thr] = {
            "n_bets": n_bets_val,
            "roi": total_pnl / (n_bets_val * 100),
            "win_rate": wins / n_bets_val,
            "pnl": total_pnl,
            "kelly_roi": kelly_pnl / 10000,
        }

    kalshi_results["DK Line vs Kalshi"] = res

    print_bet_table(kalshi_results, kalshi_thresholds)

    # ── KELLY OPTIMAL SIZING ─────────────────────────────────────────────
    print_section("KELLY CRITERION: EXPECTED GROWTH per $10k BANKROLL (test set)")
    print(f"{'Threshold':>10} | {'Model':>20} | {'Market':>15} | {'Avg Kelly %':>12} | {'Kelly Growth':>13}")
    print("-" * 85)

    # O/U Kelly
    for name, col in ou_models.items():
        valid = df.dropna(subset=[col, "mkt_p_over"])
        valid = valid[valid["push"] == 0]
        split_idx = int(len(valid) * 0.30)
        test = valid.iloc[split_idx:].copy()
        test["edge"] = test[col] - test["mkt_p_over"]
        for thr in [0.03, 0.05, 0.07]:
            over_mask = test["edge"] > thr
            under_mask = test["edge"] < -thr
            edges = pd.concat([test.loc[over_mask, "edge"],
                               test.loc[under_mask, "edge"].abs()])
            decs = pd.concat([test.loc[over_mask, "over_dec"],
                              test.loc[under_mask, "under_dec"]])
            if len(edges) == 0:
                continue
            kelly_fracs = [kelly_fraction(e, d) for e, d in zip(edges, decs)]
            avg_kelly = np.mean(kelly_fracs)
            r = ou_results.get(name, {}).get(thr, {})
            kelly_growth = r.get("kelly_roi", np.nan)
            if pd.notna(kelly_growth):
                print(f"{thr:>9.0%} | {name:>20} | {'DK O/U':>15} | {avg_kelly:>11.2%} | ${kelly_growth * 10000:>+11.0f}")

    # ML Kelly
    for name, col in ml_models.items():
        valid = df.dropna(subset=[col, "mkt_home_ml"])
        valid = valid[valid["home_score"] != valid["away_score"]]
        split_idx = int(len(valid) * 0.30)
        test = valid.iloc[split_idx:].copy()
        test["home_edge"] = test[col] - test["mkt_home_ml"]
        test["away_edge"] = (1 - test[col]) - test["mkt_away_ml"]
        for thr in [0.03, 0.05, 0.07]:
            edges = pd.concat([
                test.loc[test["home_edge"] > thr, "home_edge"],
                test.loc[test["away_edge"] > thr, "away_edge"]])
            if len(edges) == 0:
                continue
            # approximate kelly frac
            avg_kelly = np.mean([kelly_fraction(e, 2.0) for e in edges])
            r = ml_results.get(name, {}).get(thr, {})
            kelly_growth = r.get("kelly_roi", np.nan)
            if pd.notna(kelly_growth):
                print(f"{thr:>9.0%} | {name:>20} | {'DK ML':>15} | {avg_kelly:>11.2%} | ${kelly_growth * 10000:>+11.0f}")

    # ── COMPREHENSIVE SUMMARY TABLE ──────────────────────────────────────
    print_section("COMPREHENSIVE SUMMARY (best threshold per model)")

    summary_rows = []

    # O/U models
    for name, res_dict in ou_results.items():
        best_thr = None
        best_roi = -999
        for thr, r in res_dict.items():
            if r["n_bets"] >= 20 and pd.notna(r["roi"]) and r["roi"] > best_roi:
                best_roi = r["roi"]
                best_thr = thr
        if best_thr is not None:
            r = res_dict[best_thr]
            summary_rows.append({
                "Model": name, "Market": "DK O/U", "Best Threshold": f"{best_thr:.0%}",
                "Bets": r["n_bets"], "Win Rate": f"{r['win_rate']:.1%}",
                "Flat ROI": f"{r['roi']:+.1%}", "P&L ($100/bet)": f"${r['pnl']:+.0f}",
                "Kelly ROI": f"{r['kelly_roi']:+.1%}",
            })

    # ML models
    for name, res_dict in ml_results.items():
        best_thr = None
        best_roi = -999
        for thr, r in res_dict.items():
            if r["n_bets"] >= 20 and pd.notna(r["roi"]) and r["roi"] > best_roi:
                best_roi = r["roi"]
                best_thr = thr
        if best_thr is not None:
            r = res_dict[best_thr]
            summary_rows.append({
                "Model": name, "Market": "DK ML", "Best Threshold": f"{best_thr:.0%}",
                "Bets": r["n_bets"], "Win Rate": f"{r['win_rate']:.1%}",
                "Flat ROI": f"{r['roi']:+.1%}", "P&L ($100/bet)": f"${r['pnl']:+.0f}",
                "Kelly ROI": f"{r['kelly_roi']:+.1%}",
            })

    # Kalshi models
    for name, res_dict in kalshi_results.items():
        best_thr = None
        best_roi = -999
        for thr, r in res_dict.items():
            if r["n_bets"] >= 20 and pd.notna(r["roi"]) and r["roi"] > best_roi:
                best_roi = r["roi"]
                best_thr = thr
        if best_thr is not None:
            r = res_dict[best_thr]
            summary_rows.append({
                "Model": name, "Market": "Kalshi ML", "Best Threshold": f"{best_thr:.0%}",
                "Bets": r["n_bets"], "Win Rate": f"{r['win_rate']:.1%}",
                "Flat ROI": f"{r['roi']:+.1%}", "P&L ($100/bet)": f"${r['pnl']:+.0f}",
                "Kelly ROI": f"{r['kelly_roi']:+.1%}",
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
    else:
        print("  No models had >= 20 bets at any threshold.")

    # ── CALIBRATION CHECK ────────────────────────────────────────────────
    print_section("CALIBRATION CHECK: predicted vs actual P(over)")
    for name, col in ou_models.items():
        valid = df.dropna(subset=[col]).copy()
        valid = valid[valid["push"] == 0]
        if len(valid) == 0:
            continue
        bins = pd.qcut(valid[col], q=5, duplicates="drop")
        cal = valid.groupby(bins, observed=True).agg(
            pred_mean=(col, "mean"),
            actual_mean=("actual_over", "mean"),
            count=("actual_over", "count"),
        )
        print(f"\n  {name}:")
        print(f"  {'Pred Range':>25} | {'Pred Mean':>10} | {'Actual Mean':>12} | {'Count':>6} | {'Gap':>8}")
        for idx, row in cal.iterrows():
            gap = row["pred_mean"] - row["actual_mean"]
            print(f"  {str(idx):>25} | {row['pred_mean']:>10.3f} | {row['actual_mean']:>12.3f} | {row['count']:>6.0f} | {gap:>+7.3f}")

    print(f"\n{'='*80}")
    print("  BACKTEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
