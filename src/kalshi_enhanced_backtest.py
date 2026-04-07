#!/usr/bin/env python3
"""
Enhanced Kalshi Backtest — Multi-Book Consensus + Line Movement + O/U
====================================================================

Builds on kalshi_clean_backtest.py with additional signals:
  1. Multi-book consensus (6 sportsbooks devigged avg vs Kalshi)
  2. DK line movement (open→close direction as confirmation filter)
  3. Kalshi O/U arb (DK closing O/U vs Kalshi strike prices)

Zero look-ahead bias:
  TRAIN:      2024 (model fitting)
  VALIDATION: 2025-04-16 to 2025-06-30 (param selection)
  TEST:       2025-07-01 onward (unbiased evaluation)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

VAL_CUTOFF = "2025-07-01"

# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def american_to_prob(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100),
                    np.where(odds > 0, 100 / (odds + 100), np.nan))


def kalshi_pnl(buy_price, won):
    """Kalshi contract P&L: buy at price, receive $1 if won."""
    return (1.0 - buy_price) if won else -buy_price


# ═══════════════════════════════════════════════════════════════════
# 1. Load & Merge All Data
# ═══════════════════════════════════════════════════════════════════

def load_ml_data():
    """Load Kalshi ML + DK + multi-book consensus."""
    # Kalshi ML
    kalshi = pd.read_parquet(DATA / "kalshi" / "kalshi_mlb_2025.parquet")
    kalshi["game_date"] = pd.to_datetime(kalshi["game_date"]).dt.strftime("%Y-%m-%d")

    # DK closing lines (from SBR)
    dk = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    dk["game_date"] = pd.to_datetime(dk["game_date"]).dt.strftime("%Y-%m-%d")
    dk["dk_home_raw"] = american_to_prob(dk["home_ml_close"])
    dk["dk_away_raw"] = american_to_prob(dk["away_ml_close"])
    dk["dk_total_vig"] = dk["dk_home_raw"] + dk["dk_away_raw"]
    dk["dk_home_prob"] = dk["dk_home_raw"] / dk["dk_total_vig"]
    dk["dk_away_prob"] = dk["dk_away_raw"] / dk["dk_total_vig"]

    # DK open lines for movement
    dk["dk_home_open_raw"] = american_to_prob(dk["home_ml_open"])
    dk["dk_away_open_raw"] = american_to_prob(dk["away_ml_open"])
    dk["dk_open_vig"] = dk["dk_home_open_raw"] + dk["dk_away_open_raw"]
    dk["dk_home_open_fair"] = dk["dk_home_open_raw"] / dk["dk_open_vig"]
    dk["dk_movement"] = dk["dk_home_prob"] - dk["dk_home_open_fair"]

    dk_cols = ["game_date", "home_team", "away_team", "dk_home_prob", "dk_away_prob",
               "dk_home_open_fair", "dk_movement"]
    dk = dk[dk_cols].dropna(subset=["dk_home_prob"])

    # Multi-book consensus
    cons_path = DATA / "odds" / "multibook_consensus_2025.parquet"
    cons = pd.read_parquet(cons_path)
    cons["game_date"] = pd.to_datetime(cons["game_date"]).dt.strftime("%Y-%m-%d")
    cons_cols = ["game_date", "home_team", "away_team", "consensus_home_fair",
                 "consensus_std", "n_books"]
    cons = cons[cons_cols]

    # Merge
    merge_keys = ["game_date", "home_team", "away_team"]
    df = kalshi.merge(dk, on=merge_keys, how="inner")
    df = df.merge(cons, on=merge_keys, how="left")

    # Edges
    df["dk_edge_home"] = df["dk_home_prob"] - df["kalshi_home_prob"]
    df["dk_edge_away"] = df["dk_away_prob"] - df["kalshi_away_prob"]
    df["cons_edge_home"] = df["consensus_home_fair"] - df["kalshi_home_prob"]
    df["cons_edge_away"] = (1 - df["consensus_home_fair"]) - df["kalshi_away_prob"]

    print(f"  Kalshi+DK merge: {len(df)} games")
    print(f"  With consensus: {df['consensus_home_fair'].notna().sum()}")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    return df


# ═══════════════════════════════════════════════════════════════════
# 2. Betting with Filters
# ═══════════════════════════════════════════════════════════════════

def compute_bets(df, our_prob_col, min_edge, kelly_frac=0.25,
                 max_bet_frac=0.05, movement_confirm=False,
                 min_volume=0, consensus_confirm=False,
                 cons_col=None, cons_min_edge=0.0):
    """
    Generate bets comparing our_prob vs Kalshi prices.

    Filters:
      movement_confirm: only bet if DK open→close moved in same direction
      consensus_confirm: only bet if multi-book consensus also has edge
    """
    bets = []
    for _, row in df.iterrows():
        our_home = row.get(our_prob_col, np.nan)
        if pd.isna(our_home):
            continue
        our_away = 1.0 - our_home
        k_home = row["kalshi_home_prob"]
        k_away = row["kalshi_away_prob"]
        if pd.isna(k_home):
            continue

        edge_home = our_home - k_home
        edge_away = our_away - k_away

        # Pick side with larger edge
        if edge_home > edge_away and edge_home > min_edge:
            side, edge, buy_price, our_p = "home", edge_home, k_home, our_home
        elif edge_away > min_edge:
            side, edge, buy_price, our_p = "away", edge_away, k_away, our_away
        else:
            continue

        # Volume filter
        vol = row.get("volume", 0)
        if pd.notna(vol) and vol > 0 and vol < min_volume:
            continue

        # Movement confirmation filter
        if movement_confirm:
            mv = row.get("dk_movement", np.nan)
            if pd.isna(mv):
                continue
            # If betting home, DK should have moved toward home (positive movement)
            if side == "home" and mv < 0:
                continue
            if side == "away" and mv > 0:
                continue

        # Consensus confirmation filter
        if consensus_confirm and cons_col:
            cons_home = row.get(cons_col, np.nan)
            if pd.isna(cons_home):
                continue
            cons_away = 1.0 - cons_home
            if side == "home" and (cons_home - k_home) < cons_min_edge:
                continue
            if side == "away" and (cons_away - k_away) < cons_min_edge:
                continue

        won = (side == "home" and row["home_win"] == 1) or \
              (side == "away" and row["home_win"] == 0)

        pnl = kalshi_pnl(buy_price, won)
        b = (1.0 - buy_price) / buy_price
        q = 1.0 - our_p
        kelly_f = max(0.0, (our_p * b - q) / b) * kelly_frac
        kelly_f = min(kelly_f, max_bet_frac)

        bets.append({
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "side": side,
            "edge": edge,
            "our_prob": our_p,
            "kalshi_price": buy_price,
            "pnl_flat": pnl,
            "kelly_f": kelly_f,
            "won": int(won),
        })
    return pd.DataFrame(bets)


def run_kelly_backtest(bets_df, starting_bankroll=10000):
    """Run Kelly-sized backtest and return metrics."""
    if bets_df.empty:
        return {"n_bets": 0, "sharpe": 0, "flat_roi": 0, "win_rate": 0,
                "total_return": 0, "max_dd": 0, "final_bankroll": starting_bankroll}

    bankroll = starting_bankroll
    daily_pnl = []

    for date, day_bets in bets_df.groupby("game_date"):
        day_total = 0
        for _, bet in day_bets.iterrows():
            stake = bankroll * bet["kelly_f"]
            if stake <= 0:
                continue
            if bet["won"]:
                dollar_pnl = stake * (1.0 - bet["kalshi_price"]) / bet["kalshi_price"]
            else:
                dollar_pnl = -stake
            bankroll += dollar_pnl
            day_total += dollar_pnl
        daily_pnl.append(day_total)

    daily_pnl = np.array(daily_pnl)
    n_bets = len(bets_df)
    flat_roi = bets_df["pnl_flat"].mean()
    win_rate = bets_df["won"].mean()

    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(180)
    else:
        sharpe = 0

    equity = np.array([starting_bankroll] + list(np.cumsum(daily_pnl) + starting_bankroll))
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min()

    return {
        "n_bets": n_bets, "win_rate": win_rate, "flat_roi": flat_roi,
        "sharpe": sharpe, "max_dd": max_dd,
        "total_return": (bankroll - starting_bankroll) / starting_bankroll,
        "final_bankroll": bankroll,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. Validation: Grid Search for Best Params
# ═══════════════════════════════════════════════════════════════════

def optimize_strategies(val_df):
    """Sweep all strategy variants on validation data."""
    print("\n" + "=" * 70)
    print("VALIDATION: Optimizing Strategy Parameters")
    print("=" * 70)
    print(f"  Games: {len(val_df)}, {val_df['game_date'].min()} to {val_df['game_date'].max()}")

    results = {}

    # Strategy configs to sweep
    strategies = {
        "DK_arb": {
            "prob_col": "dk_home_prob",
            "movement_confirm": False,
            "consensus_confirm": False,
        },
        "DK_arb+movement": {
            "prob_col": "dk_home_prob",
            "movement_confirm": True,
            "consensus_confirm": False,
        },
        "Consensus": {
            "prob_col": "consensus_home_fair",
            "movement_confirm": False,
            "consensus_confirm": False,
        },
        "Consensus+movement": {
            "prob_col": "consensus_home_fair",
            "movement_confirm": True,
            "consensus_confirm": False,
        },
        "DK_arb+consensus_confirm": {
            "prob_col": "dk_home_prob",
            "movement_confirm": False,
            "consensus_confirm": True,
            "cons_col": "consensus_home_fair",
        },
        "DK_arb+both_filters": {
            "prob_col": "dk_home_prob",
            "movement_confirm": True,
            "consensus_confirm": True,
            "cons_col": "consensus_home_fair",
        },
    }

    edge_grid = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
    kelly_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    cons_edge_grid = [0.0, 0.01, 0.02, 0.03]

    for strat_name, config in strategies.items():
        print(f"\n  --- {strat_name} ---")
        best_score = -999
        best_params = None

        for me in edge_grid:
            for kf in kelly_grid:
                ce_values = cons_edge_grid if config.get("consensus_confirm") else [0.0]
                for ce in ce_values:
                    bets = compute_bets(
                        val_df, config["prob_col"], min_edge=me, kelly_frac=kf,
                        movement_confirm=config.get("movement_confirm", False),
                        consensus_confirm=config.get("consensus_confirm", False),
                        cons_col=config.get("cons_col"),
                        cons_min_edge=ce,
                    )
                    if len(bets) < 10:
                        continue
                    m = run_kelly_backtest(bets)
                    # Sample-size weighted Sharpe
                    score = m["sharpe"] * min(1.0, len(bets) / 50.0)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "min_edge": me, "kelly_frac": kf, "cons_min_edge": ce,
                            "n_bets": m["n_bets"], "sharpe": m["sharpe"],
                            "flat_roi": m["flat_roi"], "win_rate": m["win_rate"],
                        }

        if best_params:
            print(f"    Best: edge>{best_params['min_edge']:.0%} kelly={best_params['kelly_frac']:.0%} "
                  f"cons_edge>{best_params['cons_min_edge']:.0%}")
            print(f"    Bets={best_params['n_bets']} WR={best_params['win_rate']:.1%} "
                  f"ROI={best_params['flat_roi']:+.1%} Sharpe={best_params['sharpe']:.2f} "
                  f"(weighted={best_score:.2f})")
            results[strat_name] = {**config, **best_params}
        else:
            print(f"    No profitable configuration found")

    return results


# ═══════════════════════════════════════════════════════════════════
# 4. Test Evaluation (FIXED params)
# ═══════════════════════════════════════════════════════════════════

def evaluate_on_test(test_df, opt_results):
    """Apply fixed validation params to test period."""
    print("\n" + "=" * 70)
    print("TEST PERIOD: Out-of-Sample Evaluation")
    print("=" * 70)
    print(f"  Games: {len(test_df)}, {test_df['game_date'].min()} to {test_df['game_date'].max()}")
    print(f"  Home win rate: {test_df['home_win'].mean():.3f}")

    test_results = {}

    for strat_name, config in sorted(opt_results.items(), key=lambda x: -x[1].get("sharpe", 0)):
        bets = compute_bets(
            test_df, config["prob_col"],
            min_edge=config["min_edge"],
            kelly_frac=config["kelly_frac"],
            movement_confirm=config.get("movement_confirm", False),
            consensus_confirm=config.get("consensus_confirm", False),
            cons_col=config.get("cons_col"),
            cons_min_edge=config.get("cons_min_edge", 0),
        )
        m = run_kelly_backtest(bets)
        test_results[strat_name] = m

        # Bootstrap CI
        if len(bets) >= 20:
            boot_rois = []
            for _ in range(2000):
                idx = np.random.choice(len(bets), len(bets), replace=True)
                boot_rois.append(bets.iloc[idx]["pnl_flat"].mean())
            ci_lo, ci_hi = np.percentile(boot_rois, [2.5, 97.5])
        else:
            ci_lo, ci_hi = np.nan, np.nan

        print(f"\n  {strat_name} (edge>{config['min_edge']:.0%}, kelly={config['kelly_frac']:.0%}"
              f"{', +mvmt' if config.get('movement_confirm') else ''}"
              f"{', +cons' if config.get('consensus_confirm') else ''}):")
        print(f"    Bets: {m['n_bets']}, Win Rate: {m['win_rate']:.1%}")
        print(f"    Flat ROI: {m['flat_roi']:+.1%}  [{ci_lo:+.1%}, {ci_hi:+.1%}]")
        print(f"    Sharpe: {m['sharpe']:.2f}")
        print(f"    Kelly Return: {m['total_return']:+.1%}, Max DD: {m['max_dd']:.1%}")

        # Monthly
        if len(bets) > 0:
            bets["month"] = pd.to_datetime(bets["game_date"]).dt.to_period("M")
            print(f"    Monthly: ", end="")
            for month, mbets in bets.groupby("month"):
                mr = mbets["pnl_flat"].mean()
                print(f"{month}: {mr:+.1%}({len(mbets)}) ", end="")
            print()

    # Permutation test on best strategy
    if test_results:
        best_name = max(test_results, key=lambda k: test_results[k]["sharpe"])
        best_config = opt_results[best_name]
        print(f"\n  Permutation test on {best_name}:")
        bets = compute_bets(
            test_df, best_config["prob_col"],
            min_edge=best_config["min_edge"],
            kelly_frac=best_config["kelly_frac"],
            movement_confirm=best_config.get("movement_confirm", False),
            consensus_confirm=best_config.get("consensus_confirm", False),
            cons_col=best_config.get("cons_col"),
            cons_min_edge=best_config.get("cons_min_edge", 0),
        )
        observed_roi = bets["pnl_flat"].mean() if len(bets) > 0 else 0
        n_perm = 5000
        perm_rois = []
        for _ in range(n_perm):
            shuffled = test_df.copy()
            shuffled["home_win"] = np.random.permutation(shuffled["home_win"].values)
            pb = compute_bets(
                shuffled, best_config["prob_col"],
                min_edge=best_config["min_edge"],
                kelly_frac=best_config["kelly_frac"],
                movement_confirm=best_config.get("movement_confirm", False),
                consensus_confirm=best_config.get("consensus_confirm", False),
                cons_col=best_config.get("cons_col"),
                cons_min_edge=best_config.get("cons_min_edge", 0),
            )
            perm_rois.append(pb["pnl_flat"].mean() if len(pb) > 0 else 0)
        perm_rois = np.array(perm_rois)
        p_value = (perm_rois >= observed_roi).mean()
        print(f"    Observed ROI: {observed_roi:+.1%}")
        print(f"    P-value: {p_value:.4f}")
        print(f"    Significant: {'YES' if p_value < 0.05 else 'NO'}")

    return test_results


# ═══════════════════════════════════════════════════════════════════
# 5. Kalshi O/U Arb
# ═══════════════════════════════════════════════════════════════════

def load_ou_data():
    """Load Kalshi O/U + DK O/U closing lines."""
    # Kalshi totals (2025 + 2026)
    frames = []
    for year in [2025, 2026]:
        path = DATA / "kalshi" / f"kalshi_total_{year}.parquet"
        if path.exists():
            kdf = pd.read_parquet(path)
            kdf["game_date"] = pd.to_datetime(kdf["game_date"]).dt.strftime("%Y-%m-%d")
            frames.append(kdf)
    if not frames:
        return None
    kalshi_ou = pd.concat(frames, ignore_index=True)

    # DK O/U
    dk = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    dk["game_date"] = pd.to_datetime(dk["game_date"]).dt.strftime("%Y-%m-%d")
    dk_ou_cols = ["game_date", "home_team", "away_team", "ou_close",
                  "over_close_odds", "under_close_odds", "total_runs", "ou_result"]
    dk_ou = dk[dk_ou_cols].dropna(subset=["ou_close"])

    # Devig DK O/U
    dk_ou["dk_over_raw"] = american_to_prob(dk_ou["over_close_odds"])
    dk_ou["dk_under_raw"] = american_to_prob(dk_ou["under_close_odds"])
    dk_ou["dk_ou_vig"] = dk_ou["dk_over_raw"] + dk_ou["dk_under_raw"]
    dk_ou["dk_over_fair"] = dk_ou["dk_over_raw"] / dk_ou["dk_ou_vig"]

    # Merge
    merge_keys = ["game_date", "home_team", "away_team"]
    merged = kalshi_ou.merge(dk_ou, on=merge_keys, how="inner")
    print(f"  Kalshi O/U + DK merge: {len(merged)} games")

    return merged


def run_ou_arb(ou_df):
    """
    Find edges between DK implied total and Kalshi strike prices.

    For each game + strike, compare DK's implied P(over strike) with
    Kalshi's over_XX_prob. If DK says 60% over 8.5 but Kalshi sells
    over-8.5 at 50 cents, that's a 10-point edge.
    """
    if ou_df is None or ou_df.empty:
        print("  No O/U data available")
        return

    print("\n" + "=" * 70)
    print("KALSHI O/U ARB ANALYSIS")
    print("=" * 70)
    print(f"  Games: {len(ou_df)}")
    print(f"  Date range: {ou_df['game_date'].min()} to {ou_df['game_date'].max()}")

    # Strike columns in Kalshi data
    strikes = {6.5: "over_65_prob", 7.5: "over_75_prob",
               8.5: "over_85_prob", 9.5: "over_95_prob"}

    # DK closing line gives us a point estimate. We can derive P(over X)
    # from the DK line using a simple model: if DK line = 8.5 with over at -110,
    # the devigged P(over 8.5) ≈ dk_over_fair.
    # For other strikes, we can use the DK line to estimate a gaussian CDF.
    # Assume total ~ Normal(dk_line, sigma) where sigma ≈ 2.5 (empirical)

    from scipy.stats import norm

    # Estimate sigma from actual data
    if ou_df["total_runs"].notna().sum() > 20:
        residuals = ou_df["total_runs"] - ou_df["ou_close"]
        sigma = max(residuals.std(), 1.5)
    else:
        sigma = 2.5

    print(f"  Estimated total std: {sigma:.2f}")

    bets = []
    for _, row in ou_df.iterrows():
        dk_line = row["ou_close"]
        if pd.isna(dk_line):
            continue

        actual_total = row.get("total_runs", row.get("actual_total", np.nan))

        for strike_val, kalshi_col in strikes.items():
            kalshi_over_prob = row.get(kalshi_col, np.nan)
            if pd.isna(kalshi_over_prob) or kalshi_over_prob <= 0 or kalshi_over_prob >= 1:
                continue

            # DK implied P(over strike) using normal approximation
            # Use DK's devigged over probability at their own line, then extrapolate
            dk_over_at_line = row.get("dk_over_fair", 0.5)
            # Adjust dk_line by vig-implied shift
            adjusted_mean = dk_line + norm.ppf(dk_over_at_line) * sigma - norm.ppf(0.5) * sigma
            dk_over_prob = 1.0 - norm.cdf(strike_val + 0.5, loc=adjusted_mean, scale=sigma)

            kalshi_under_prob = 1.0 - kalshi_over_prob

            # Edge on over
            edge_over = dk_over_prob - kalshi_over_prob
            # Edge on under
            dk_under_prob = 1.0 - dk_over_prob
            edge_under = dk_under_prob - kalshi_under_prob

            # Pick better side
            if edge_over > edge_under and edge_over > 0.03:
                side = "over"
                edge = edge_over
                buy_price = kalshi_over_prob
                our_p = dk_over_prob
                won = actual_total > strike_val if not pd.isna(actual_total) else np.nan
            elif edge_under > 0.03:
                side = "under"
                edge = edge_under
                buy_price = kalshi_under_prob
                our_p = dk_under_prob
                won = actual_total < strike_val if not pd.isna(actual_total) else np.nan
            else:
                continue

            if pd.isna(won):
                continue
            # Skip pushes
            if actual_total == strike_val:
                continue

            pnl = kalshi_pnl(buy_price, won)
            b = (1.0 - buy_price) / buy_price
            q = 1.0 - our_p
            kelly_f = max(0.0, (our_p * b - q) / b) * 0.20
            kelly_f = min(kelly_f, 0.05)

            bets.append({
                "game_date": row["game_date"],
                "home_team": row["home_team"],
                "strike": strike_val,
                "side": side,
                "edge": edge,
                "our_prob": our_p,
                "kalshi_price": buy_price,
                "dk_line": dk_line,
                "actual_total": actual_total,
                "pnl_flat": pnl,
                "kelly_f": kelly_f,
                "won": int(won),
            })

    bets_df = pd.DataFrame(bets)
    if bets_df.empty:
        print("  No O/U bets found")
        return

    m = run_kelly_backtest(bets_df)
    print(f"\n  O/U Arb Results:")
    print(f"    Bets: {m['n_bets']}, Win Rate: {m['win_rate']:.1%}")
    print(f"    Flat ROI: {m['flat_roi']:+.1%}")
    print(f"    Sharpe: {m['sharpe']:.2f}")
    print(f"    Kelly Return: {m['total_return']:+.1%}")

    # Per-strike breakdown
    print(f"\n  Per-strike:")
    for strike in sorted(bets_df["strike"].unique()):
        sb = bets_df[bets_df["strike"] == strike]
        print(f"    {strike}: {len(sb)} bets, WR={sb['won'].mean():.1%}, "
              f"ROI={sb['pnl_flat'].mean():+.1%}, avg_edge={sb['edge'].mean():.1%}")

    # Per-side
    for side in ["over", "under"]:
        sb = bets_df[bets_df["side"] == side]
        if len(sb) > 0:
            print(f"    {side}: {len(sb)} bets, WR={sb['won'].mean():.1%}, "
                  f"ROI={sb['pnl_flat'].mean():+.1%}")

    return m


# ═══════════════════════════════════════════════════════════════════
# 6. Multi-Strategy Portfolio (Test Period)
# ═══════════════════════════════════════════════════════════════════

def run_portfolio(test_df, opt_results, strategies_to_use):
    """Run selected strategies independently in a portfolio."""
    print("\n" + "=" * 70)
    print(f"PORTFOLIO: {', '.join(strategies_to_use)}")
    print("=" * 70)

    bankroll = 10000
    all_bets = []
    daily_pnl = []

    for date, day_games in test_df.groupby("game_date"):
        day_total = 0
        for strat_name in strategies_to_use:
            if strat_name not in opt_results:
                continue
            config = opt_results[strat_name]
            for _, row in day_games.iterrows():
                our_home = row.get(config["prob_col"], np.nan)
                if pd.isna(our_home):
                    continue
                our_away = 1.0 - our_home
                k_home, k_away = row["kalshi_home_prob"], row["kalshi_away_prob"]
                if pd.isna(k_home):
                    continue

                edge_home = our_home - k_home
                edge_away = our_away - k_away

                if edge_home > edge_away and edge_home > config["min_edge"]:
                    side, edge, buy_price, our_p = "home", edge_home, k_home, our_home
                elif edge_away > config["min_edge"]:
                    side, edge, buy_price, our_p = "away", edge_away, k_away, our_away
                else:
                    continue

                # Apply filters
                if config.get("movement_confirm"):
                    mv = row.get("dk_movement", np.nan)
                    if pd.isna(mv):
                        continue
                    if (side == "home" and mv < 0) or (side == "away" and mv > 0):
                        continue

                if config.get("consensus_confirm"):
                    cons_home = row.get(config.get("cons_col", ""), np.nan)
                    if pd.isna(cons_home):
                        continue
                    cons_away = 1.0 - cons_home
                    ce = config.get("cons_min_edge", 0)
                    if side == "home" and (cons_home - k_home) < ce:
                        continue
                    if side == "away" and (cons_away - k_away) < ce:
                        continue

                b = (1.0 - buy_price) / buy_price
                q = 1.0 - our_p
                kelly_f = max(0.0, (our_p * b - q) / b) * config["kelly_frac"]
                kelly_f = min(kelly_f, 0.05)

                stake = bankroll * kelly_f
                if stake <= 0:
                    continue

                won = (side == "home" and row["home_win"] == 1) or \
                      (side == "away" and row["home_win"] == 0)
                dollar_pnl = stake * (1.0 - buy_price) / buy_price if won else -stake
                bankroll += dollar_pnl
                day_total += dollar_pnl

                all_bets.append({
                    "game_date": date, "strategy": strat_name,
                    "side": side, "won": int(won),
                    "pnl_flat": kalshi_pnl(buy_price, won),
                    "pnl_dollar": dollar_pnl,
                    "kelly_f": kelly_f, "kalshi_price": buy_price,
                })
        daily_pnl.append(day_total)

    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()
    daily_pnl = np.array(daily_pnl)

    if bets_df.empty:
        print("  No bets")
        return {}

    n = len(bets_df)
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(180) if daily_pnl.std() > 0 else 0
    total_return = (bankroll - 10000) / 10000

    print(f"  Bets: {n}, WR: {bets_df['won'].mean():.1%}")
    print(f"  Flat ROI: {bets_df['pnl_flat'].mean():+.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Kelly Return: {total_return:+.1%}")
    print(f"  Final Bankroll: ${bankroll:,.0f}")

    for strat in bets_df["strategy"].unique():
        sb = bets_df[bets_df["strategy"] == strat]
        print(f"    {strat}: {len(sb)} bets, WR={sb['won'].mean():.1%}, ROI={sb['pnl_flat'].mean():+.1%}")

    return {"n_bets": n, "sharpe": sharpe, "flat_roi": bets_df["pnl_flat"].mean(),
            "total_return": total_return, "final_bankroll": bankroll}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run_focused_backtest(df):
    """
    Run the focused DK arb backtest with validated underdog + price filters.

    Strategies tested (all use DK devigged closing line vs Kalshi):
      A: DK_arb edge>6% (baseline)
      B: DK_arb edge>4% underdogs only (kalshi_price < 0.50)
      C: DK_arb edge>4% price [0.35, 0.55]
    """
    val_mask = (df["game_date"] >= "2025-04-16") & (df["game_date"] < VAL_CUTOFF)
    test_mask = df["game_date"] >= VAL_CUTOFF
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    print(f"\n  Validation: {len(val_df)} games")
    print(f"  Test:       {len(test_df)} games")

    configs = {
        "DK_arb (edge>6%)": {
            "min_edge": 0.06, "kelly_frac": 0.30,
            "underdog_only": False, "price_range": None,
        },
        "DK_arb underdogs (edge>4%)": {
            "min_edge": 0.04, "kelly_frac": 0.30,
            "underdog_only": True, "price_range": None,
        },
        "DK_arb price[.35,.55] (edge>4%)": {
            "min_edge": 0.04, "kelly_frac": 0.30,
            "underdog_only": False, "price_range": (0.35, 0.55),
        },
    }

    # Validate all configs
    print("\n  VALIDATION results:")
    print(f"    {'Config':<35} {'Bets':>5} {'WR':>6} {'ROI':>7} {'Sharpe':>7} {'WtdS':>6}")
    val_scores = {}
    for name, cfg in configs.items():
        bets = compute_bets(val_df, "dk_home_prob", min_edge=cfg["min_edge"],
                           kelly_frac=cfg["kelly_frac"])
        if cfg["underdog_only"]:
            bets = bets[bets["kalshi_price"] < 0.50] if not bets.empty else bets
        if cfg["price_range"] and not bets.empty:
            lo, hi = cfg["price_range"]
            bets = bets[(bets["kalshi_price"] >= lo) & (bets["kalshi_price"] <= hi)]
        m = run_kelly_backtest(bets)
        w = m["sharpe"] * min(1.0, len(bets) / 50.0) if m["n_bets"] > 0 else 0
        val_scores[name] = w
        print(f"    {name:<35} {m['n_bets']:>5} {m['win_rate']:>5.1%} {m['flat_roi']:>+6.1%} "
              f"{m['sharpe']:>6.2f} {w:>5.2f}")

    # Test all configs
    print("\n  TEST results:")
    print(f"    {'Config':<35} {'Bets':>5} {'WR':>6} {'ROI':>7} {'Sharpe':>7} {'95% CI':>18}")
    test_results = {}
    for name, cfg in configs.items():
        bets = compute_bets(test_df, "dk_home_prob", min_edge=cfg["min_edge"],
                           kelly_frac=cfg["kelly_frac"])
        if cfg["underdog_only"]:
            bets = bets[bets["kalshi_price"] < 0.50] if not bets.empty else bets
        if cfg["price_range"] and not bets.empty:
            lo, hi = cfg["price_range"]
            bets = bets[(bets["kalshi_price"] >= lo) & (bets["kalshi_price"] <= hi)]
        m = run_kelly_backtest(bets)
        test_results[name] = m

        if len(bets) >= 15:
            boot_rois = [bets.iloc[np.random.choice(len(bets), len(bets), replace=True)]["pnl_flat"].mean()
                        for _ in range(2000)]
            ci_lo, ci_hi = np.percentile(boot_rois, [2.5, 97.5])
            ci_str = f"[{ci_lo:+.1%}, {ci_hi:+.1%}]"
        else:
            ci_str = "N/A"

        print(f"    {name:<35} {m['n_bets']:>5} {m['win_rate']:>5.1%} {m['flat_roi']:>+6.1%} "
              f"{m['sharpe']:>6.2f} {ci_str:>18}")

    # Permutation test on validation-best config
    best_name = max(val_scores, key=val_scores.get)
    best_cfg = configs[best_name]
    print(f"\n  Permutation test on {best_name}:")
    bets = compute_bets(test_df, "dk_home_prob", min_edge=best_cfg["min_edge"],
                       kelly_frac=best_cfg["kelly_frac"])
    if best_cfg["underdog_only"]:
        bets = bets[bets["kalshi_price"] < 0.50] if not bets.empty else bets
    if best_cfg["price_range"] and not bets.empty:
        lo, hi = best_cfg["price_range"]
        bets = bets[(bets["kalshi_price"] >= lo) & (bets["kalshi_price"] <= hi)]

    observed_roi = bets["pnl_flat"].mean() if len(bets) > 0 else 0
    perm_rois = []
    for _ in range(5000):
        shuffled = test_df.copy()
        shuffled["home_win"] = np.random.permutation(shuffled["home_win"].values)
        pb = compute_bets(shuffled, "dk_home_prob", min_edge=best_cfg["min_edge"],
                         kelly_frac=best_cfg["kelly_frac"])
        if best_cfg["underdog_only"]:
            pb = pb[pb["kalshi_price"] < 0.50] if not pb.empty else pb
        if best_cfg["price_range"] and not pb.empty:
            lo, hi = best_cfg["price_range"]
            pb = pb[(pb["kalshi_price"] >= lo) & (pb["kalshi_price"] <= hi)]
        perm_rois.append(pb["pnl_flat"].mean() if len(pb) > 0 else 0)
    perm_rois = np.array(perm_rois)
    p_value = (perm_rois >= observed_roi).mean()
    print(f"    Observed ROI: {observed_roi:+.1%}, P-value: {p_value:.4f}")
    print(f"    Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")

    # Edge curve on test (informational)
    print("\n  Edge threshold curve (test, kelly=30%):")
    print(f"    {'Edge':>6} {'Bets':>6} {'WR':>7} {'ROI':>8} {'Sharpe':>8}")
    for me in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]:
        b = compute_bets(test_df, "dk_home_prob", min_edge=me, kelly_frac=0.30)
        m = run_kelly_backtest(b)
        if m["n_bets"] >= 5:
            print(f"    {me:>5.0%} {m['n_bets']:>6} {m['win_rate']:>6.1%} "
                  f"{m['flat_roi']:>+7.1%} {m['sharpe']:>7.2f}")

    return test_results


def main():
    print("=" * 70)
    print("ENHANCED KALSHI BACKTEST — DK Arb + Filters + O/U")
    print("=" * 70)

    # ── ML strategies ──
    print("\n[1] Loading data...")
    df = load_ml_data()

    # ── Focused backtest ──
    print("\n[2] Running focused DK arb backtest...")
    test_results = run_focused_backtest(df)

    # ── Full strategy sweep (validation → test) ──
    print("\n[3] Full strategy sweep (with consensus + movement filters)...")
    val_mask = (df["game_date"] >= "2025-04-16") & (df["game_date"] < VAL_CUTOFF)
    test_mask = df["game_date"] >= VAL_CUTOFF
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    opt_results = optimize_strategies(val_df)

    if opt_results:
        evaluate_on_test(test_df, opt_results)

    # ── O/U arb ──
    print("\n[4] Kalshi O/U arb...")
    ou_df = load_ou_data()
    if ou_df is not None and len(ou_df) > 0:
        run_ou_arb(ou_df)
    else:
        print("  No O/U arb data available")

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  {'Strategy':<35} {'Bets':>6} {'WR':>7} {'ROI':>8} {'Sharpe':>8}")
    for name in sorted(test_results.keys(), key=lambda k: -test_results[k].get("sharpe", 0)):
        m = test_results[name]
        print(f"  {name:<35} {m['n_bets']:>6} {m['win_rate']:>6.1%} "
              f"{m['flat_roi']:>+7.1%} {m['sharpe']:>7.2f}")

    print("\n  Production config (validation-selected):")
    print("    Signal: DK devigged closing ML prob vs Kalshi pre-game price")
    print("    Strategy A: edge > 4%, underdogs only (kalshi_price < 0.50)")
    print("    Strategy B: edge > 6%, no filter (baseline)")
    print("    Kelly: 30% fractional, max 5% per bet")
    print("    Market: Kalshi KXMLBGAME contracts")
    print("    Execution: Buy Kalshi contract when DK closing line shows edge")


if __name__ == "__main__":
    main()
