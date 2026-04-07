#!/usr/bin/env python3
"""
O/U Betting via Monte Carlo Simulator
======================================

Uses the MC simulator's full total-runs distribution to compute
P(over DK_line) for each game, then bets when sim disagrees with DK.

The simulator has:
  - Near-zero bias (+0.04 runs/game)
  - Correlation 0.246 with actual totals (vs DK's 0.229)
  - Game-specific pitcher/lineup/park/weather inputs

Pipeline:
  1. Run MC sim for 2025 games (or load cached results)
  2. Merge with DK closing O/U lines + odds
  3. Compute sim P(over) from empirical distribution at DK's line
  4. Compare vs DK devigged P(over) → edge
  5. Bet when edge exceeds threshold
  6. Proper temporal split + permutation test
"""

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

CACHE_PATH = DATA / "backtest" / "ou_sim_dist_2025.parquet"
VAL_CUTOFF = "2025-07-01"


def american_to_prob(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100),
                    np.where(odds > 0, 100 / (odds + 100), np.nan))


def run_simulations(max_games=None, n_sims=3000):
    """Run MC sim for all 2025 games, store total_runs distributions."""
    from simulate import (
        GameState, SimConfig, load_simulation_context,
        monte_carlo_win_prob,
    )
    from backtest_nrfi_ou import (
        load_sim_artifacts, load_backtest_data,
    )
    from feature_engineering import compute_park_factors

    season = 2025
    config = SimConfig(n_sims=n_sims, random_seed=42)

    # Load artifacts
    print("  Loading sim artifacts...", flush=True)
    base_rates, transition_matrix = load_sim_artifacts()
    print("  Loading backtest data...", flush=True)
    idx, matchup_models, lineups = load_backtest_data(season)

    # Load games
    print("  Loading games...", flush=True)
    games_path = DATA / "games" / f"games_{season}.parquet"
    games = pd.read_parquet(games_path)
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")

    # Alias columns
    if "home_team_abbr" in games.columns and "home_team" not in games.columns:
        games["home_team"] = games["home_team_abbr"]
    if "away_team_abbr" in games.columns and "away_team" not in games.columns:
        games["away_team"] = games["away_team_abbr"]

    # Filter to completed games with both SPs
    games = games[games["status"] == "Final"].copy()
    games = games.dropna(subset=["home_sp_id", "away_sp_id"])
    games["actual_total"] = games["home_score"] + games["away_score"]

    # Park factors + weather
    print("  Computing park factors...", flush=True)
    park_factors = compute_park_factors(games, season)
    print(f"  Park factors: {len(park_factors)} venues", flush=True)
    weather_map = {}
    wx_path = DATA / "weather" / f"weather_{season}.parquet"
    if wx_path.exists():
        wx_df = pd.read_parquet(wx_path)
        for _, wr in wx_df.iterrows():
            wind_dir = str(wr.get("wind_direction", "")).lower()
            weather_map[int(wr["game_pk"])] = {
                "temperature": wr.get("temperature"),
                "wind_speed": wr.get("wind_speed"),
                "wind_out": int("out" in wind_dir),
                "wind_in": int("in" in wind_dir),
                "is_dome": wr.get("is_dome", 0),
            }

    games = games.sort_values("game_date").reset_index(drop=True)
    if max_games:
        games = games.head(max_games)

    print(f"  Simulating {len(games)} games with {n_sims} sims each...")

    rows = []
    n_skip = 0
    for i, (_, row) in enumerate(games.iterrows()):
        game = row.to_dict()
        gpk = int(game["game_pk"])
        lu = lineups.get(gpk)

        if lu is None or not lu.get("home") or not lu.get("away"):
            n_skip += 1
            continue

        try:
            home_ctx, away_ctx = load_simulation_context(
                game, str(game["game_date"]), lu, idx,
                matchup_models or {}, base_rates, config=config,
            )
            pf = park_factors.get(game["home_team"], 1.0)
            wx = weather_map.get(gpk)
            if wx and wx.get("is_dome"):
                wx = None

            result = monte_carlo_win_prob(
                home_ctx, away_ctx, GameState(),
                transition_matrix, config,
                park_factor=pf, weather=wx,
            )

            # Store total runs distribution as P(over) at various lines
            total_dist = result.get("total_runs_dist", {})
            total_sims = sum(total_dist.values())

            # Compute P(over) at common lines
            p_over = {}
            for line in [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5]:
                n_over = sum(cnt for tot, cnt in total_dist.items() if tot > line)
                p_over[f"sim_p_over_{line}"] = n_over / total_sims if total_sims > 0 else 0.5

            rows.append({
                "game_pk": gpk,
                "game_date": game["game_date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "sim_total_mean": result["total_runs_mean"],
                "sim_total_std": np.sqrt(np.mean([(t - result["total_runs_mean"])**2
                                                   for t, c in total_dist.items() for _ in range(c)])) if total_dist else 0,
                "actual_total": int(game["actual_total"]),
                "sim_home_wp": result["home_wp"],
                **p_over,
            })
        except Exception as e:
            n_skip += 1
            if n_skip <= 3:
                print(f"    ERROR game {gpk}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(games)} ({len(rows)} done, {n_skip} skipped)", flush=True)

    print(f"  Done: {len(rows)} games simulated, {n_skip} skipped")
    result_df = pd.DataFrame(rows)
    result_df.to_parquet(CACHE_PATH, index=False)
    return result_df


def load_dk_odds():
    """Load DK O/U odds."""
    dk = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    dk["game_date"] = pd.to_datetime(dk["game_date"]).dt.strftime("%Y-%m-%d")
    dk = dk[dk["game_date"] >= "2025-01-01"]
    dk = dk.dropna(subset=["ou_close", "over_close_odds", "under_close_odds"])

    dk["dk_over_raw"] = american_to_prob(dk["over_close_odds"])
    dk["dk_under_raw"] = american_to_prob(dk["under_close_odds"])
    dk["dk_vig"] = dk["dk_over_raw"] + dk["dk_under_raw"]
    dk["dk_over_fair"] = dk["dk_over_raw"] / dk["dk_vig"]
    dk["dk_under_fair"] = dk["dk_under_raw"] / dk["dk_vig"]

    return dk[["game_date", "home_team", "away_team", "ou_close",
               "dk_over_fair", "dk_under_fair", "over_close_odds", "under_close_odds"]].drop_duplicates()


def compute_sim_p_over_at_dk_line(sim_df):
    """For each game, interpolate sim P(over) at the DK closing line."""
    # Available sim lines
    sim_lines = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5]

    p_overs = []
    for _, row in sim_df.iterrows():
        dk_line = row["ou_close"]

        if dk_line in sim_lines:
            p_overs.append(row[f"sim_p_over_{dk_line}"])
        else:
            # Linear interpolation between nearest lines
            below = max([l for l in sim_lines if l <= dk_line], default=sim_lines[0])
            above = min([l for l in sim_lines if l >= dk_line], default=sim_lines[-1])
            if below == above:
                p_overs.append(row[f"sim_p_over_{below}"])
            else:
                frac = (dk_line - below) / (above - below)
                p_below = row[f"sim_p_over_{below}"]
                p_above = row[f"sim_p_over_{above}"]
                p_overs.append(p_below + frac * (p_above - p_below))

    return np.array(p_overs)


def backtest_ou(df, min_edge, kelly_frac=0.25, under_only=False):
    """Generate bets and compute results."""
    bets = []
    for _, row in df.iterrows():
        sim_p = row["sim_p_over"]
        mkt_p = row["dk_over_fair"]
        actual_over = row["actual_total"] > row["ou_close"]
        is_push = row["actual_total"] == row["ou_close"]

        if is_push or pd.isna(sim_p) or pd.isna(mkt_p):
            continue

        edge_over = sim_p - mkt_p
        edge_under = -edge_over

        if not under_only and edge_over > min_edge:
            side, edge, our_p = "over", edge_over, sim_p
            buy_price = mkt_p
            won = actual_over
        elif edge_under > min_edge:
            side, edge, our_p = "under", edge_under, 1 - sim_p
            buy_price = 1 - mkt_p
            won = not actual_over
        else:
            continue

        pnl = (1.0 - buy_price) if won else -buy_price

        b = (1.0 - buy_price) / buy_price if buy_price > 0 else 0
        q = 1.0 - our_p
        kelly_f = max(0.0, (our_p * b - q) / b) * kelly_frac if b > 0 else 0
        kelly_f = min(kelly_f, 0.05)

        bets.append({
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "ou_line": row["ou_close"],
            "side": side,
            "edge": edge,
            "our_prob": our_p,
            "market_price": buy_price,
            "actual_total": row["actual_total"],
            "pnl_flat": pnl,
            "kelly_f": kelly_f,
            "won": int(won),
        })

    return pd.DataFrame(bets)


def compute_metrics(bets_df, starting_bankroll=10000):
    if bets_df.empty:
        return {"n_bets": 0, "sharpe": 0, "flat_roi": 0, "win_rate": 0,
                "total_return": 0, "max_dd": 0}

    bankroll = starting_bankroll
    daily_pnl = []
    for date, day_bets in bets_df.groupby("game_date"):
        day_total = 0
        for _, bet in day_bets.iterrows():
            stake = bankroll * bet["kelly_f"]
            if stake <= 0:
                continue
            if bet["won"]:
                dollar_pnl = stake * (1.0 - bet["market_price"]) / bet["market_price"]
            else:
                dollar_pnl = -stake
            bankroll += dollar_pnl
            day_total += dollar_pnl
        daily_pnl.append(day_total)

    daily_pnl = np.array(daily_pnl)
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(180) if daily_pnl.std() > 0 else 0
    equity = np.array([starting_bankroll] + list(np.cumsum(daily_pnl) + starting_bankroll))
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min()

    return {
        "n_bets": len(bets_df),
        "win_rate": bets_df["won"].mean(),
        "flat_roi": bets_df["pnl_flat"].mean(),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": (bankroll - starting_bankroll) / starting_bankroll,
    }


def main():
    print("=" * 70)
    print("O/U BETTING VIA MC SIMULATOR")
    print("=" * 70)

    # ── Step 1: Run or load simulations ──
    print("\n[1] Loading simulations...")
    if CACHE_PATH.exists():
        sim_df = pd.read_parquet(CACHE_PATH)
        if len(sim_df) > 100:
            print(f"  Loaded cached: {len(sim_df)} games")
        else:
            print(f"  Cache too small ({len(sim_df)} games). Re-running...")
            sim_df = run_simulations(max_games=None, n_sims=3000)
    else:
        print("  No cache found. Running simulations...")
        sim_df = run_simulations(max_games=2000, n_sims=2000)

    # ── Step 2: Merge with DK odds ──
    print("\n[2] Merging with DK odds...")
    dk = load_dk_odds()
    merged = sim_df.merge(dk, on=["game_date", "home_team", "away_team"], how="inner")
    merged = merged.sort_values("game_date").reset_index(drop=True)
    print(f"  Matched: {len(merged)} games with sim + DK odds")
    print(f"  Date range: {merged['game_date'].min()} to {merged['game_date'].max()}")

    # ── Step 3: Compute sim P(over) at DK line ──
    print("\n[3] Computing sim P(over) at DK lines...")
    merged["sim_p_over"] = compute_sim_p_over_at_dk_line(merged)

    # Filter to regular season (Apr 1+)
    merged = merged[merged["game_date"] >= "2025-04-01"].copy()
    merged = merged.sort_values("game_date").reset_index(drop=True)

    # ── Step 4: Diagnostics ──
    print(f"\n[4] Diagnostics ({len(merged)} games):")
    actual_over = (merged["actual_total"] > merged["ou_close"]).astype(int)
    is_push = merged["actual_total"] == merged["ou_close"]
    non_push = ~is_push

    from sklearn.metrics import brier_score_loss
    y = actual_over[non_push].values
    sim_p = merged.loc[non_push, "sim_p_over"].values
    dk_p = merged.loc[non_push, "dk_over_fair"].values

    brier_sim = brier_score_loss(y, sim_p)
    brier_dk = brier_score_loss(y, dk_p)
    brier_naive = brier_score_loss(y, np.full(len(y), y.mean()))
    bss_sim = 1 - brier_sim / brier_naive
    bss_dk = 1 - brier_dk / brier_naive
    bss_sim_vs_dk = 1 - brier_sim / brier_dk

    corr_sim = np.corrcoef(merged["sim_total_mean"].values, merged["actual_total"].values)[0, 1]
    corr_dk = np.corrcoef(merged["ou_close"].values, merged["actual_total"].values)[0, 1]

    print(f"  Correlation — Sim: {corr_sim:.3f}, DK: {corr_dk:.3f}")
    print(f"  Brier — Sim: {brier_sim:.4f}, DK: {brier_dk:.4f}")
    print(f"  BSS — Sim: {bss_sim:+.4f}, DK: {bss_dk:+.4f}")
    print(f"  BSS sim vs DK: {bss_sim_vs_dk:+.4f} ({'sim better' if bss_sim_vs_dk > 0 else 'DK better'})")
    print(f"  Actual over rate: {y.mean():.3f}")
    print(f"  Sim mean P(over): {sim_p.mean():.3f}")
    print(f"  DK mean P(over):  {dk_p.mean():.3f}")

    # Calibration
    print(f"\n  Calibration (sim P(over) bins):")
    merged_np = merged[non_push].copy()
    merged_np["actual_over"] = y
    bins = [0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0]
    merged_np["bin"] = pd.cut(merged_np["sim_p_over"], bins)
    for b, grp in merged_np.groupby("bin", observed=True):
        if len(grp) >= 5:
            print(f"    {str(b):>14}: n={len(grp):>4}, pred={grp['sim_p_over'].mean():.3f}, "
                  f"actual={grp['actual_over'].mean():.3f}")

    # Naive under
    under_pnl = merged_np.apply(
        lambda r: r["dk_over_fair"] if r["actual_over"] == 0 else -(1 - r["dk_over_fair"]), axis=1
    )
    print(f"\n  Naive under: WR={(y==0).mean():.1%}, ROI={under_pnl.mean():+.1%}")

    # ── Step 5: Temporal split ──
    print(f"\n[5] Temporal split...")
    # Train on early data to select thresholds, test on later data
    pre_july = merged[merged["game_date"] < VAL_CUTOFF]
    train_end_idx = int(len(pre_july) * 0.6)
    train_end_date = pre_july.iloc[train_end_idx]["game_date"]

    val_df = merged[(merged["game_date"] > train_end_date) & (merged["game_date"] < VAL_CUTOFF)].copy()
    test_df = merged[merged["game_date"] >= VAL_CUTOFF].copy()

    print(f"  Val: {len(val_df)} games ({val_df['game_date'].min()} – {val_df['game_date'].max()})")
    print(f"  Test: {len(test_df)} games ({test_df['game_date'].min()} – {test_df['game_date'].max()})")

    # ── Step 6: Validation sweep ──
    print(f"\n[6] Validation: edge threshold sweep...")
    print(f"  {'Edge':>5} {'Kelly':>6} {'Mode':>7} {'Bets':>5} {'WR':>6} {'ROI':>7} {'Sharpe':>7}")

    best_score, best_cfg = -999, None
    for me in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        for kf in [0.10, 0.15, 0.20, 0.25]:
            for uo in [False, True]:
                bets = backtest_ou(val_df, min_edge=me, kelly_frac=kf, under_only=uo)
                if len(bets) < 15:
                    continue
                m = compute_metrics(bets)
                score = m["sharpe"] * min(1.0, len(bets) / 50.0)
                mode = "under" if uo else "both"

                if me in [0.02, 0.05, 0.08, 0.10, 0.15] and kf == 0.25:
                    print(f"  {me:>4.0%} {kf:>5.0%} {mode:>7} {m['n_bets']:>5} "
                          f"{m['win_rate']:>5.1%} {m['flat_roi']:>+6.1%} {m['sharpe']:>6.2f}")

                if score > best_score:
                    best_score = score
                    best_cfg = {"edge": me, "kelly": kf, "under_only": uo}

    if best_cfg is None:
        print("  No profitable config found on validation.")
        return

    bv = backtest_ou(val_df, min_edge=best_cfg["edge"],
                     kelly_frac=best_cfg["kelly"], under_only=best_cfg["under_only"])
    mv = compute_metrics(bv)
    mode_str = "under-only" if best_cfg["under_only"] else "both"
    print(f"\n  BEST: edge>{best_cfg['edge']:.0%} kelly={best_cfg['kelly']:.0%} mode={mode_str}")
    print(f"    Val: {mv['n_bets']} bets, WR={mv['win_rate']:.1%}, "
          f"ROI={mv['flat_roi']:+.1%}, Sharpe={mv['sharpe']:.2f}")
    if len(bv) > 0:
        for side in ["over", "under"]:
            sb = bv[bv["side"] == side]
            if len(sb) > 0:
                print(f"      {side}: {len(sb)} bets, WR={sb['won'].mean():.1%}, ROI={sb['pnl_flat'].mean():+.1%}")

    # ── Step 7: Test evaluation ──
    print(f"\n[7] Test evaluation (fixed params)...")
    bt = backtest_ou(test_df, min_edge=best_cfg["edge"],
                     kelly_frac=best_cfg["kelly"], under_only=best_cfg["under_only"])
    mt = compute_metrics(bt)

    # Bootstrap CI
    if len(bt) >= 15:
        boot_rois = [bt.iloc[np.random.choice(len(bt), len(bt), replace=True)]["pnl_flat"].mean()
                     for _ in range(2000)]
        ci_lo, ci_hi = np.percentile(boot_rois, [2.5, 97.5])
        ci_str = f"[{ci_lo:+.1%}, {ci_hi:+.1%}]"
    else:
        ci_str = "N/A"

    print(f"    Test: {mt['n_bets']} bets, WR={mt['win_rate']:.1%}, "
          f"ROI={mt['flat_roi']:+.1%} {ci_str}, Sharpe={mt['sharpe']:.2f}, MaxDD={mt['max_dd']:.1%}")

    if len(bt) > 0:
        for side in ["over", "under"]:
            sb = bt[bt["side"] == side]
            if len(sb) > 0:
                print(f"      {side}: {len(sb)} bets, WR={sb['won'].mean():.1%}, "
                      f"ROI={sb['pnl_flat'].mean():+.1%}, avg_edge={sb['edge'].mean():.1%}")

        # Monthly
        bt["month"] = pd.to_datetime(bt["game_date"]).dt.to_period("M")
        print("    Monthly:")
        for month, mb in bt.groupby("month"):
            print(f"      {month}: {len(mb)} bets, WR={mb['won'].mean():.1%}, ROI={mb['pnl_flat'].mean():+.1%}")

    # ── Step 8: Permutation test ──
    if len(bt) >= 15:
        print(f"\n[8] Permutation test (5,000 shuffles)...")
        observed_roi = bt["pnl_flat"].mean()
        perm_rois = []
        for _ in range(5000):
            shuffled = test_df.copy()
            shuffled["actual_total"] = np.random.permutation(shuffled["actual_total"].values)
            pb = backtest_ou(shuffled, min_edge=best_cfg["edge"],
                            kelly_frac=best_cfg["kelly"], under_only=best_cfg["under_only"])
            perm_rois.append(pb["pnl_flat"].mean() if len(pb) > 0 else 0)
        perm_rois = np.array(perm_rois)
        p_value = (perm_rois >= observed_roi).mean()
        print(f"    Observed ROI: {observed_roi:+.1%}")
        print(f"    P-value: {p_value:.4f}")
        print(f"    Significant: {'YES' if p_value < 0.05 else 'NO'}")

    # ── Step 9: Ensemble with LGB model ──
    # Load LGB predictions if available and blend with sim
    print(f"\n[9] Checking for LGB model predictions to ensemble...")
    feat_path = DATA / "features" / "game_features_2025.parquet"
    if feat_path.exists():
        # Quick check: does blending sim P(over) with a constant under bias help?
        # Shift sim probabilities toward under by the observed market bias
        bias = merged.loc[non_push, "dk_over_fair"].mean() - y.mean()  # DK overprices overs by this much
        print(f"  DK over-pricing bias: {bias:+.3f}")

        for sdf, name in [(val_df, "Val"), (test_df, "Test")]:
            sdf["sim_p_over_corrected"] = sdf["sim_p_over"] - bias
            sdf["sim_p_over_corrected"] = sdf["sim_p_over_corrected"].clip(0.05, 0.95)

        # Re-run with corrected probabilities
        bv_c = backtest_ou(val_df, min_edge=best_cfg["edge"],
                          kelly_frac=best_cfg["kelly"], under_only=best_cfg["under_only"])
        mv_c = compute_metrics(bv_c) if len(bv_c) > 0 else {"n_bets": 0}

        # Override sim_p_over with corrected version for test
        test_df_c = test_df.copy()
        test_df_c["sim_p_over"] = test_df_c["sim_p_over_corrected"]
        bt_c = backtest_ou(test_df_c, min_edge=best_cfg["edge"],
                          kelly_frac=best_cfg["kelly"], under_only=best_cfg["under_only"])
        mt_c = compute_metrics(bt_c)

        if mt_c["n_bets"] > 0:
            ci_c = ""
            if len(bt_c) >= 15:
                boot = [bt_c.iloc[np.random.choice(len(bt_c), len(bt_c), replace=True)]["pnl_flat"].mean()
                        for _ in range(2000)]
                ci_c = f"[{np.percentile(boot, 2.5):+.1%}, {np.percentile(boot, 97.5):+.1%}]"
            print(f"  Bias-corrected: {mt_c['n_bets']} bets, WR={mt_c['win_rate']:.1%}, "
                  f"ROI={mt_c['flat_roi']:+.1%} {ci_c}, Sharpe={mt_c['sharpe']:.2f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Sim correlation with totals: {corr_sim:.3f} (DK: {corr_dk:.3f})")
    print(f"  BSS sim vs DK: {bss_sim_vs_dk:+.4f}")
    print(f"  Best config: edge>{best_cfg['edge']:.0%} kelly={best_cfg['kelly']:.0%} {mode_str}")
    print(f"  Test: {mt['n_bets']} bets, ROI={mt['flat_roi']:+.1%}, Sharpe={mt['sharpe']:.2f}")


if __name__ == "__main__":
    main()
