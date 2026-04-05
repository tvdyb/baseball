"""
ou_edge_analysis.py
-------------------
Explore O/U betting edges against DraftKings lines.

Approach 1: Blind under betting + model-enhanced under selection
Approach 2: Closing Line Value (CLV) exploitation via line-move prediction
Approach 3: Edge in specific game subsets (high/low totals, Coors/Oracle)

Usage:
    python src/ou_edge_analysis.py
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================================
# HELPERS
# ============================================================================

def american_to_raw_prob(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))


def american_to_decimal(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, a / 100 + 1, 100 / (-a) + 1)


def flat_bet_roi(wins, losses, pushes, odds=-110):
    """ROI for flat betting at given American odds. Push = stake returned."""
    dec = american_to_decimal(np.array([odds]))[0]
    profit = wins * (dec - 1) - losses * 1.0
    n_bets = wins + losses + pushes
    if n_bets == 0:
        return np.nan, 0, 0.0
    roi = profit / n_bets
    return roi, n_bets, profit


def bet_at_actual_odds(is_win, is_push, dec_payout):
    """Compute PnL using actual DK closing odds for each game."""
    profit = np.where(is_push, 0.0,
                      np.where(is_win, dec_payout - 1.0, -1.0))
    n = len(profit)
    return profit.sum() / n if n > 0 else np.nan, n, profit.sum()


# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 78)
print("O/U EDGE ANALYSIS vs DraftKings 2025 Lines")
print("=" * 78)

ODDS_PATH = "data/odds/sbr_mlb_2025.parquet"
LGB_PATH  = "data/backtest/total_runs_lgb_2025.parquet"
STK_PATH  = "data/backtest/total_runs_stacker_2025.parquet"

print("\nLoading data ...")
odds    = pd.read_parquet(ODDS_PATH)
lgb     = pd.read_parquet(LGB_PATH)
stacker = pd.read_parquet(STK_PATH)

for df_tmp in [odds, lgb, stacker]:
    df_tmp["game_date"] = pd.to_datetime(df_tmp["game_date"])

# Merge LGB + stacker
stk_cols = ["game_pk", "stk_pred_total", "sim_total_mean", "sim_line",
            "stk_p_over_7.5", "stk_p_over_8.0", "stk_p_over_8.5",
            "stk_p_over_9.0", "stk_p_over_9.5",
            "stk_qr_p_over_7.5", "stk_qr_p_over_8.0", "stk_qr_p_over_8.5",
            "stk_qr_p_over_9.0", "stk_qr_p_over_9.5"]
df = lgb.merge(stacker[stk_cols], on="game_pk", how="left")

# Merge with DK odds
dk_cols = ["game_date", "home_team", "away_team",
           "ou_open", "ou_close",
           "over_open_odds", "under_open_odds",
           "over_close_odds", "under_close_odds",
           "ou_result", "total_runs"]
dk = odds[dk_cols].copy()
dk = dk.rename(columns={"total_runs": "dk_total_runs"})

df = df.merge(dk, on=["game_date", "home_team", "away_team"], how="inner")
df = df.sort_values("game_date").reset_index(drop=True)

# Outcomes
df["actual_over"]  = (df["total_runs"] > df["ou_close"]).astype(int)
df["actual_under"] = (df["total_runs"] < df["ou_close"]).astype(int)
df["is_push"]      = (df["total_runs"] == df["ou_close"]).astype(int)

# Vig-free implied probs
over_imp  = american_to_raw_prob(df["over_close_odds"].values)
under_imp = american_to_raw_prob(df["under_close_odds"].values)
total_imp = over_imp + under_imp
df["vig_free_p_over"]  = np.where(total_imp > 0, over_imp / total_imp, np.nan)
df["vig_free_p_under"] = np.where(total_imp > 0, under_imp / total_imp, np.nan)

# Same for opening odds
over_imp_o  = american_to_raw_prob(df["over_open_odds"].values)
under_imp_o = american_to_raw_prob(df["under_open_odds"].values)
total_imp_o = over_imp_o + under_imp_o
df["vig_free_p_over_open"]  = np.where(total_imp_o > 0, over_imp_o / total_imp_o, np.nan)

# Decimal payouts
df["dec_over"]  = american_to_decimal(df["over_close_odds"].values)
df["dec_under"] = american_to_decimal(df["under_close_odds"].values)
df["dec_under_open"] = american_to_decimal(df["under_open_odds"].values)
df["dec_over_open"]  = american_to_decimal(df["over_open_odds"].values)

# Line movement
df["line_move"] = df["ou_close"] - df["ou_open"]

# Model predictions at DK line
LGB_LINES = [7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
STK_LINES = [7.5, 8.0, 8.5, 9.0, 9.5]


def assign_p_over(df_in, prefix, lines):
    result = pd.Series(np.nan, index=df_in.index)
    for line in lines:
        col = f"{prefix}_{line}"
        if col in df_in.columns:
            mask = (df_in["ou_close"] == line) & df_in[col].notna()
            result[mask] = df_in.loc[mask, col]
    # Fallback: normal CDF
    fb = result.isna() & df_in["lgb_pred_total"].notna()
    if fb.any() and "lgb_residual_std" in df_in.columns:
        result[fb] = stats.norm.sf(
            df_in.loc[fb, "ou_close"],
            loc=df_in.loc[fb, "lgb_pred_total"],
            scale=df_in.loc[fb, "lgb_residual_std"],
        )
    return result


df["lgb_p_over"] = assign_p_over(df, "lgb_p_over", LGB_LINES)
df["stk_p_over"] = assign_p_over(df, "stk_p_over", STK_LINES)
df["lgb_p_under"] = 1.0 - df["lgb_p_over"]
df["stk_p_under"] = 1.0 - df["stk_p_over"]

# Sim p_over (normal CDF)
sim_ok = df["sim_total_mean"].notna() & df["lgb_residual_std"].notna()
df["sim_p_over"] = np.nan
df.loc[sim_ok, "sim_p_over"] = stats.norm.sf(
    df.loc[sim_ok, "ou_close"],
    loc=df.loc[sim_ok, "sim_total_mean"],
    scale=df.loc[sim_ok, "lgb_residual_std"],
)
df["sim_p_under"] = 1.0 - df["sim_p_over"]

# Walk-forward 30/70 split
n_calib = int(len(df) * 0.30)
calib = df.iloc[:n_calib].copy()
test  = df.iloc[n_calib:].copy()

print(f"  Total merged games : {len(df):,}")
print(f"  Calibration set    : {len(calib):,} games  (through {calib['game_date'].max().date()})")
print(f"  Test set           : {len(test):,} games   ({test['game_date'].min().date()} onward)")

non_push_test = test[test["is_push"] == 0]
print(f"\n  Test set over rate  : {non_push_test['actual_over'].mean():.4f}")
print(f"  Test set under rate : {non_push_test['actual_under'].mean():.4f}")
print(f"  Test set push rate  : {test['is_push'].mean():.4f}")
print(f"  Test vig-free P(over) mean: {test['vig_free_p_over'].mean():.4f}")

# ============================================================================
# APPROACH 1: UNDER-SIDE BETTING
# ============================================================================
print("\n" + "=" * 78)
print("APPROACH 1: STRUCTURAL UNDER BIAS")
print("=" * 78)

# --- 1a. Blind flat-bet all unders ---
print("\n--- 1a. Blind flat-bet ALL unders at closing odds ---")
t = test.dropna(subset=["dec_under"]).copy()
roi, n, pnl = bet_at_actual_odds(
    t["actual_under"].values, t["is_push"].values, t["dec_under"].values
)
wins  = t["actual_under"].sum()
losses = ((t["is_push"] == 0) & (t["actual_under"] == 0)).sum()
pushes = t["is_push"].sum()
print(f"  Games: {n}  Wins: {wins}  Losses: {losses}  Pushes: {pushes}")
print(f"  Win rate (excl push): {wins / max(wins + losses, 1):.4f}")
print(f"  ROI at actual DK closing odds: {roi:+.4f}  ({roi*100:+.2f}%)")
print(f"  Total PnL (1u/game): {pnl:+.2f} units")

# Same but at standard -110
roi_110, _, pnl_110 = flat_bet_roi(wins, losses, pushes, odds=-110)
print(f"  ROI at standard -110:          {roi_110:+.4f}  ({roi_110*100:+.2f}%)")

# --- 1b. Blind flat-bet all unders at OPENING odds ---
print("\n--- 1b. Blind flat-bet ALL unders at OPENING odds ---")
t_open = test.dropna(subset=["dec_under_open"]).copy()
roi_o, n_o, pnl_o = bet_at_actual_odds(
    t_open["actual_under"].values, t_open["is_push"].values,
    t_open["dec_under_open"].values
)
print(f"  Games: {n_o}  ROI at actual DK opening odds: {roi_o:+.4f}  ({roi_o*100:+.2f}%)")

# --- 1c. Model-enhanced under selection ---
print("\n--- 1c. Model-enhanced under selection (test set only) ---")
print("  Bet under when model_p_under > vig_free_p_under + threshold\n")

THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
MODELS = {"lgb": "lgb_p_under", "stk": "stk_p_under", "sim": "sim_p_under"}

for model_name, p_col in MODELS.items():
    t = test.dropna(subset=[p_col, "vig_free_p_under", "dec_under"]).copy()
    if len(t) == 0:
        print(f"  [{model_name}] No data with non-null predictions.")
        continue

    print(f"  [{model_name}]  n_games_available = {len(t)}")
    print(f"  {'thr':>5}  {'n_bet':>6}  {'wins':>5}  {'losses':>6}  {'push':>5}  "
          f"{'WR':>6}  {'ROI_dk':>8}  {'ROI_110':>8}  {'PnL':>8}")
    print("  " + "-" * 72)

    for thr in THRESHOLDS:
        edge = t[p_col] - t["vig_free_p_under"]
        mask = edge > thr
        sub = t[mask]
        if len(sub) == 0:
            print(f"  {thr:>5.2f}  {0:>6d}  {'--':>5}  {'--':>6}  {'--':>5}  "
                  f"{'--':>6}  {'--':>8}  {'--':>8}  {'--':>8}")
            continue
        w = sub["actual_under"].sum()
        p = sub["is_push"].sum()
        l = len(sub) - w - p
        wr = w / max(w + l, 1)
        roi_dk, _, pnl_dk = bet_at_actual_odds(
            sub["actual_under"].values, sub["is_push"].values, sub["dec_under"].values
        )
        roi_s, _, pnl_s = flat_bet_roi(w, l, p, -110)
        print(f"  {thr:>5.2f}  {len(sub):>6d}  {w:>5d}  {l:>6d}  {p:>5d}  "
              f"{wr:>6.3f}  {roi_dk:>+8.4f}  {roi_s:>+8.4f}  {pnl_dk:>+8.1f}")
    print()

# --- 1d. Consensus: BOTH lgb AND stk say under ---
print("--- 1d. Consensus: LGB AND stk both agree on under edge ---")
t = test.dropna(subset=["lgb_p_under", "stk_p_under", "vig_free_p_under", "dec_under"]).copy()
print(f"  Games with both LGB + STK predictions: {len(t)}")

for thr in THRESHOLDS:
    lgb_edge = t["lgb_p_under"] - t["vig_free_p_under"]
    stk_edge = t["stk_p_under"] - t["vig_free_p_under"]
    mask = (lgb_edge > thr) & (stk_edge > thr)
    sub = t[mask]
    if len(sub) == 0:
        continue
    w = sub["actual_under"].sum()
    p = sub["is_push"].sum()
    l = len(sub) - w - p
    wr = w / max(w + l, 1)
    roi_dk, _, pnl_dk = bet_at_actual_odds(
        sub["actual_under"].values, sub["is_push"].values, sub["dec_under"].values
    )
    roi_s, _, pnl_s = flat_bet_roi(w, l, p, -110)
    print(f"  thr={thr:.2f}  n={len(sub):>5d}  W/L/P={w}/{l}/{p}  "
          f"WR={wr:.3f}  ROI_dk={roi_dk:+.4f}  ROI_110={roi_s:+.4f}  PnL={pnl_dk:+.1f}u")

# ============================================================================
# APPROACH 2: CLOSING LINE VALUE (CLV) EXPLOITATION
# ============================================================================
print("\n" + "=" * 78)
print("APPROACH 2: CLOSING LINE VALUE (CLV) EXPLOITATION")
print("=" * 78)

t = test.dropna(subset=["ou_open", "ou_close", "lgb_pred_total"]).copy()
t["line_move"] = t["ou_close"] - t["ou_open"]
t["model_vs_open"] = t["lgb_pred_total"] - t["ou_open"]  # positive = model says higher
t["model_vs_close"] = t["lgb_pred_total"] - t["ou_close"]

# --- 2a. Correlation: model vs line movement ---
print("\n--- 2a. Does model predict line movement direction? ---")
from scipy.stats import pearsonr, spearmanr

moved = t[t["line_move"] != 0].copy()
print(f"  Games with line movement: {len(moved)} / {len(t)}")

r_p, p_p = pearsonr(t["model_vs_open"], t["line_move"])
r_s, p_s = spearmanr(t["model_vs_open"], t["line_move"])
print(f"  Pearson  r(model_vs_open, line_move) = {r_p:.4f}  p = {p_p:.4f}")
print(f"  Spearman r(model_vs_open, line_move) = {r_s:.4f}  p = {p_s:.4f}")

# Also with stacker
t_stk = t.dropna(subset=["stk_pred_total"]).copy()
if len(t_stk) > 100:
    t_stk["stk_vs_open"] = t_stk["stk_pred_total"] - t_stk["ou_open"]
    r_p2, p_p2 = pearsonr(t_stk["stk_vs_open"], t_stk["line_move"])
    print(f"  Pearson  r(stk_vs_open, line_move)  = {r_p2:.4f}  p = {p_p2:.4f}  (n={len(t_stk)})")

# Sim
t_sim = t.dropna(subset=["sim_total_mean"]).copy()
if len(t_sim) > 100:
    t_sim["sim_vs_open"] = t_sim["sim_total_mean"] - t_sim["ou_open"]
    r_p3, p_p3 = pearsonr(t_sim["sim_vs_open"], t_sim["line_move"])
    print(f"  Pearson  r(sim_vs_open, line_move)  = {r_p3:.4f}  p = {p_p3:.4f}  (n={len(t_sim)})")

# --- 2b. Directional accuracy ---
print("\n--- 2b. Directional accuracy: model predicts move direction ---")

for model_lbl, col in [("lgb", "lgb_pred_total"), ("stk", "stk_pred_total"),
                        ("sim", "sim_total_mean")]:
    sub = t.dropna(subset=[col]).copy()
    sub["model_diff"] = sub[col] - sub["ou_open"]
    moved_sub = sub[sub["line_move"] != 0].copy()
    if len(moved_sub) < 50:
        continue
    correct = ((moved_sub["model_diff"] > 0) & (moved_sub["line_move"] > 0)) | \
              ((moved_sub["model_diff"] < 0) & (moved_sub["line_move"] < 0))
    acc = correct.mean()
    print(f"  [{model_lbl}] Directional accuracy on moved games: {acc:.3f}  "
          f"({correct.sum()}/{len(moved_sub)})  "
          f"{'*' if acc > 0.55 else ''}")

# --- 2c. CLV betting strategy ---
print("\n--- 2c. CLV strategy: bet opening line when model predicts favorable move ---")
print("  Bet UNDER at opening odds when model pred < ou_open (predicts line drop)")
print("  Bet OVER  at opening odds when model pred > ou_open (predicts line rise)\n")

for model_lbl, col in [("lgb", "lgb_pred_total"), ("stk", "stk_pred_total"),
                        ("sim", "sim_total_mean")]:
    sub = t.dropna(subset=[col, "dec_under_open", "dec_over_open"]).copy()
    if len(sub) < 100:
        continue

    for diff_thr in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Under bets: model < open - threshold
        u_mask = (sub["ou_open"] - sub[col]) > diff_thr
        u_sub = sub[u_mask]
        if len(u_sub) > 0:
            u_roi, u_n, u_pnl = bet_at_actual_odds(
                u_sub["actual_under"].values, u_sub["is_push"].values,
                u_sub["dec_under_open"].values
            )
        else:
            u_roi, u_n, u_pnl = np.nan, 0, 0.0

        # Over bets: model > open + threshold
        o_mask = (sub[col] - sub["ou_open"]) > diff_thr
        o_sub = sub[o_mask]
        if len(o_sub) > 0:
            o_roi, o_n, o_pnl = bet_at_actual_odds(
                o_sub["actual_over"].values, o_sub["is_push"].values,
                o_sub["dec_over_open"].values
            )
        else:
            o_roi, o_n, o_pnl = np.nan, 0, 0.0

        combined_pnl = u_pnl + o_pnl
        combined_n = u_n + o_n
        combined_roi = combined_pnl / combined_n if combined_n > 0 else np.nan

        print(f"  [{model_lbl}] diff>{diff_thr:.2f}  "
              f"UNDER: n={u_n:>4d} ROI={u_roi:>+.4f}  "
              f"OVER: n={o_n:>4d} ROI={o_roi:>+.4f}  "
              f"COMBINED: n={combined_n:>4d} ROI={combined_roi:>+.4f} PnL={combined_pnl:>+.1f}u"
              if combined_n > 0 else
              f"  [{model_lbl}] diff>{diff_thr:.2f}  No qualifying bets")
    print()

# --- 2d. CLV realized: did we actually get CLV on under bets? ---
print("--- 2d. Realized CLV check: closing line moved in our favor? ---")
for model_lbl, col in [("lgb", "lgb_pred_total"), ("stk", "stk_pred_total")]:
    sub = t.dropna(subset=[col]).copy()
    if len(sub) < 100:
        continue
    for diff_thr in [0.0, 0.5, 1.0]:
        u_mask = (sub["ou_open"] - sub[col]) > diff_thr
        u_sub = sub[u_mask]
        if len(u_sub) == 0:
            continue
        # CLV = line moved down (closing < opening) means under bettors got value
        got_clv = (u_sub["line_move"] < 0).mean()
        avg_move = u_sub["line_move"].mean()
        print(f"  [{model_lbl}] Under bets (model < open-{diff_thr}):  "
              f"n={len(u_sub):>4d}  "
              f"line_moved_down={got_clv:.3f}  "
              f"avg_line_move={avg_move:+.3f}")
    print()

# ============================================================================
# APPROACH 3: SPECIFIC GAME TYPE EDGES
# ============================================================================
print("=" * 78)
print("APPROACH 3: EDGES IN SPECIFIC GAME SUBSETS")
print("=" * 78)

# --- 3a. By DK line bucket ---
print("\n--- 3a. Under ROI by DK closing line bucket ---")
t_all = test.dropna(subset=["ou_close", "dec_under", "actual_under", "is_push"]).copy()

buckets = [
    ("Low (<=7.5)",  t_all["ou_close"] <= 7.5),
    ("Mid (8.0-8.5)", (t_all["ou_close"] >= 8.0) & (t_all["ou_close"] <= 8.5)),
    ("High (9.0-9.5)", (t_all["ou_close"] >= 9.0) & (t_all["ou_close"] <= 9.5)),
    ("Very High (>=10)", t_all["ou_close"] >= 10.0),
]

print(f"  {'Bucket':>20}  {'n':>5}  {'W':>5}  {'L':>5}  {'P':>4}  "
      f"{'WR':>6}  {'ROI_dk':>8}  {'ROI_110':>8}")
print("  " + "-" * 72)

for label, mask in buckets:
    sub = t_all[mask]
    w = sub["actual_under"].sum()
    p = sub["is_push"].sum()
    l = len(sub) - w - p
    wr = w / max(w + l, 1)
    roi_dk, _, pnl_dk = bet_at_actual_odds(
        sub["actual_under"].values, sub["is_push"].values, sub["dec_under"].values
    )
    roi_s, _, _ = flat_bet_roi(w, l, p, -110)
    print(f"  {label:>20}  {len(sub):>5d}  {w:>5d}  {l:>5d}  {p:>4d}  "
          f"{wr:>6.3f}  {roi_dk:>+8.4f}  {roi_s:>+8.4f}")

# --- 3b. Over ROI by DK line bucket ---
print("\n--- 3b. Over ROI by DK closing line bucket ---")
print(f"  {'Bucket':>20}  {'n':>5}  {'W':>5}  {'L':>5}  {'P':>4}  "
      f"{'WR':>6}  {'ROI_dk':>8}  {'ROI_110':>8}")
print("  " + "-" * 72)

for label, mask in buckets:
    sub = t_all[mask]
    w = sub["actual_over"].sum()
    p = sub["is_push"].sum()
    l = len(sub) - w - p
    wr = w / max(w + l, 1)
    roi_dk, _, _ = bet_at_actual_odds(
        sub["actual_over"].values, sub["is_push"].values, sub["dec_over"].values
    )
    roi_s, _, _ = flat_bet_roi(w, l, p, -110)
    print(f"  {label:>20}  {len(sub):>5d}  {w:>5d}  {l:>5d}  {p:>4d}  "
          f"{wr:>6.3f}  {roi_dk:>+8.4f}  {roi_s:>+8.4f}")

# --- 3c. Extreme parks ---
print("\n--- 3c. Extreme ballparks (by home team) ---")
COORS_TEAMS = ["COL"]
LOW_PARK_TEAMS = ["SF", "MIA", "CLE", "NYM"]  # Oracle, LoanDepot, Progressive, Citi

for park_label, teams in [("Coors (COL home)", COORS_TEAMS),
                          ("Low-scoring parks (SF/MIA/CLE/NYM)", LOW_PARK_TEAMS)]:
    sub = t_all[t_all["home_team"].isin(teams)]
    if len(sub) < 20:
        continue

    # Under
    w_u = sub["actual_under"].sum()
    p_u = sub["is_push"].sum()
    l_u = len(sub) - w_u - p_u
    wr_u = w_u / max(w_u + l_u, 1)
    roi_u, _, _ = bet_at_actual_odds(
        sub["actual_under"].values, sub["is_push"].values, sub["dec_under"].values
    )
    # Over
    w_o = sub["actual_over"].sum()
    l_o = len(sub) - w_o - p_u
    wr_o = w_o / max(w_o + l_o, 1)
    roi_o, _, _ = bet_at_actual_odds(
        sub["actual_over"].values, sub["is_push"].values, sub["dec_over"].values
    )

    print(f"\n  {park_label}  (n={len(sub)})")
    print(f"    Under: WR={wr_u:.3f}  ROI={roi_u:+.4f}")
    print(f"    Over:  WR={wr_o:.3f}  ROI={roi_o:+.4f}")

# --- 3d. Model-enhanced under by bucket ---
print("\n--- 3d. Model-enhanced under by DK line bucket ---")
print("  (LGB under edge > 0.03 within each bucket)\n")

t_model = test.dropna(subset=["lgb_p_under", "vig_free_p_under", "dec_under"]).copy()
t_model["lgb_under_edge"] = t_model["lgb_p_under"] - t_model["vig_free_p_under"]

buckets_model = [
    ("Low (<=7.5)",     t_model["ou_close"] <= 7.5),
    ("Mid (8.0-8.5)",   (t_model["ou_close"] >= 8.0) & (t_model["ou_close"] <= 8.5)),
    ("High (9.0-9.5)",  (t_model["ou_close"] >= 9.0) & (t_model["ou_close"] <= 9.5)),
    ("Very High (>=10)", t_model["ou_close"] >= 10.0),
    ("ALL",             pd.Series(True, index=t_model.index)),
]

THR = 0.03
print(f"  {'Bucket':>20}  {'n_all':>6}  {'n_bet':>6}  {'W':>5}  {'L':>5}  {'P':>4}  "
      f"{'WR':>6}  {'ROI_dk':>8}")
print("  " + "-" * 72)

for label, bucket_mask in buckets_model:
    sub = t_model[bucket_mask]
    bet_mask = sub["lgb_under_edge"] > THR
    bet_sub = sub[bet_mask]
    if len(bet_sub) == 0:
        print(f"  {label:>20}  {len(sub):>6d}  {0:>6d}  {'--':>5}  {'--':>5}  {'--':>4}  "
              f"{'--':>6}  {'--':>8}")
        continue
    w = bet_sub["actual_under"].sum()
    p = bet_sub["is_push"].sum()
    l = len(bet_sub) - w - p
    wr = w / max(w + l, 1)
    roi_dk, _, _ = bet_at_actual_odds(
        bet_sub["actual_under"].values, bet_sub["is_push"].values, bet_sub["dec_under"].values
    )
    print(f"  {label:>20}  {len(sub):>6d}  {len(bet_sub):>6d}  {w:>5d}  {l:>5d}  {p:>4d}  "
          f"{wr:>6.3f}  {roi_dk:>+8.4f}")

# --- 3e. Model disagreement with market as signal ---
print("\n--- 3e. Big model-vs-market disagreement (|pred - line| > threshold) ---")
print("  Bet the side our model agrees with\n")

t_dis = test.dropna(subset=["lgb_pred_total", "ou_close", "dec_over", "dec_under"]).copy()
t_dis["model_diff"] = t_dis["lgb_pred_total"] - t_dis["ou_close"]

for abs_thr in [0.5, 1.0, 1.5, 2.0]:
    # Over when model >> line
    o_mask = t_dis["model_diff"] > abs_thr
    o_sub = t_dis[o_mask]
    if len(o_sub) > 0:
        o_roi, o_n, _ = bet_at_actual_odds(
            o_sub["actual_over"].values, o_sub["is_push"].values, o_sub["dec_over"].values
        )
    else:
        o_roi, o_n = np.nan, 0

    # Under when model << line
    u_mask = t_dis["model_diff"] < -abs_thr
    u_sub = t_dis[u_mask]
    if len(u_sub) > 0:
        u_roi, u_n, _ = bet_at_actual_odds(
            u_sub["actual_under"].values, u_sub["is_push"].values, u_sub["dec_under"].values
        )
    else:
        u_roi, u_n = np.nan, 0

    print(f"  |diff|>{abs_thr:.1f}  "
          f"OVER(model>line): n={o_n:>4d} ROI={o_roi:>+.4f}  "
          f"UNDER(model<line): n={u_n:>4d} ROI={u_roi:>+.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 78)
print("EXECUTIVE SUMMARY")
print("=" * 78)
print("""
Key findings across all three approaches:

APPROACH 1 — Structural Under Bias:
  The market implies P(over) ~ 0.498 but actual over rate ~ 0.473.
  This 2.5pp gap suggests unders are systematically undervalued.
  Check the blind-under ROI above to see if this survives the vig.
  Model-enhanced selection (betting under only when model agrees)
  may improve selectivity at cost of fewer bets.

APPROACH 2 — Closing Line Value:
  Check the Pearson/Spearman correlations above.
  If r > 0.10 with p < 0.01, the model can partially predict line movement,
  which is the gold standard for +EV identification.
  The CLV strategy bets at opening odds when model predicts the close
  will move in the bettor's favor.

APPROACH 3 — Game Type Subsets:
  Certain line buckets or parks may have structural inefficiencies.
  Look for buckets where blind-under ROI is positive AND model-enhanced
  ROI is even higher — that's where the real edge lives.

  Reminder: small samples (n < 100) have wide confidence intervals.
  A 5% ROI on n=100 has a 95% CI of roughly +/-20%.
""")
print("Done.")
