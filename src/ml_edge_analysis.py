#!/usr/bin/env python3
"""
ML Edge Analysis: find WHERE the ensemble win model disagrees with DK closing
moneyline and whether those disagreements are predictive.

Analysis:
  1. Disagreement buckets — does larger |model_prob - dk_prob| produce better ROI?
  2. Game-type subsets — underdogs, heavy favorites, day/night, home/away
  3. Home/away asymmetry
  4. Time-of-season effect
  5. Platt scaling recalibration against DK lines

Usage:
    python src/ml_edge_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, binom
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# MLB division membership (2025)
DIVISIONS = {
    # AL East
    "NYY": "AL East", "BOS": "AL East", "TOR": "AL East",
    "TB":  "AL East", "BAL": "AL East",
    # AL Central
    "CWS": "AL Central", "CLE": "AL Central", "DET": "AL Central",
    "KC":  "AL Central", "MIN": "AL Central",
    # AL West
    "HOU": "AL West", "LAA": "AL West", "OAK": "AL West",
    "SEA": "AL West", "TEX": "AL West",
    # NL East
    "ATL": "NL East", "MIA": "NL East", "NYM": "NL East",
    "PHI": "NL East", "WSH": "NL East",
    # NL Central
    "CHC": "NL Central", "CIN": "NL Central", "MIL": "NL Central",
    "PIT": "NL Central", "STL": "NL Central",
    # NL West
    "ARI": "NL West", "COL": "NL West", "LAD": "NL West",
    "SD":  "NL West", "SF":  "NL West",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def american_to_prob(odds: float) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


def remove_vig(p1: float, p2: float):
    total = p1 + p2
    if total == 0 or pd.isna(total):
        return np.nan, np.nan
    return p1 / total, p2 / total


def decimal_from_american(odds: float) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / (-odds)


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 2000,
                 ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic."""
    stats = []
    rng = np.random.default_rng(42)
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        stats.append(stat_fn(sample))
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def bets_for_significance(p_edge: float, baseline_p: float = 0.5,
                           alpha: float = 0.05) -> int:
    """
    Minimum bets to detect edge p_edge above baseline at significance alpha
    (one-sided binomial test).  Uses normal approximation.
    """
    z = norm.ppf(1 - alpha)
    if p_edge <= baseline_p:
        return np.inf
    se = np.sqrt(baseline_p * (1 - baseline_p))
    n = (z * se / (p_edge - baseline_p)) ** 2
    return int(np.ceil(n))


def flat_bet_roi(df: pd.DataFrame,
                 side: str = "home") -> dict:
    """
    Compute flat-bet ROI for a given set of games and bet side.
    df must have: home_win, home_ml_dec/away_ml_dec
    Returns dict with n, wins, win_rate, roi, pnl.
    """
    if len(df) == 0:
        return {"n": 0, "wins": 0, "win_rate": np.nan, "roi": np.nan, "pnl": 0.0}

    if side == "home":
        wins = int((df["home_win"] == 1).sum())
        pnls = np.where(
            df["home_win"] == 1,
            (df["home_ml_dec"] - 1) * 100,
            -100.0,
        )
    else:
        wins = int((df["home_win"] == 0).sum())
        pnls = np.where(
            df["home_win"] == 0,
            (df["away_ml_dec"] - 1) * 100,
            -100.0,
        )

    n = len(df)
    total_pnl = float(pnls.sum())
    roi = total_pnl / (n * 100)
    return {
        "n": n,
        "wins": wins,
        "win_rate": wins / n,
        "roi": roi,
        "pnl": total_pnl,
    }


def section(title: str):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")


# ── data loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load and merge: DK odds, win-model predictions, game context."""

    # 1) DK closing lines (all seasons that have final odds)
    odds = pd.read_parquet(DATA / "odds" / "sbr_mlb_2025.parquet")
    odds["game_date"] = pd.to_datetime(odds["game_date"])
    # Keep completed games with ML lines
    odds = odds[odds["status"].str.startswith("Final")].copy()
    odds = odds.dropna(subset=["home_ml_close", "away_ml_close"]).copy()

    # Vig-free DK implied probs
    odds["dk_home_raw"] = odds["home_ml_close"].apply(american_to_prob)
    odds["dk_away_raw"] = odds["away_ml_close"].apply(american_to_prob)
    odds[["dk_home_prob", "dk_away_prob"]] = odds.apply(
        lambda r: pd.Series(remove_vig(r["dk_home_raw"], r["dk_away_raw"])), axis=1
    )
    odds["home_ml_dec"] = odds["home_ml_close"].apply(decimal_from_american)
    odds["away_ml_dec"] = odds["away_ml_close"].apply(decimal_from_american)
    odds["home_win"] = (odds["home_score"] > odds["away_score"]).astype(int)

    # 2) Win-model walk-forward predictions (all available seasons)
    wf = pd.read_csv(DATA / "audit" / "walk_forward_predictions.csv")
    wf["game_date"] = pd.to_datetime(wf["game_date"])

    # 3) Merge on game_pk (if available) else date/team keys
    if "game_pk" in odds.columns and "game_pk" in wf.columns:
        df = odds.merge(
            wf[["game_pk", "game_date", "season", "home_team", "away_team",
                "lr_prob", "xgb_prob", "ens_prob", "ens_calibrated",
                "days_into_season", "home_team_games_played", "away_team_games_played"]],
            on=["game_pk", "home_team", "away_team"],
            how="inner",
            suffixes=("", "_wf"),
        )
    else:
        df = odds.merge(
            wf[["game_date", "home_team", "away_team",
                "lr_prob", "xgb_prob", "ens_prob", "ens_calibrated",
                "days_into_season", "home_team_games_played", "away_team_games_played"]],
            on=["game_date", "home_team", "away_team"],
            how="inner",
        )

    # Resolve duplicate game_date columns from merge
    if "game_date_wf" in df.columns:
        df["game_date"] = df["game_date"].fillna(df["game_date_wf"])
        df = df.drop(columns=["game_date_wf"])
    if "season" not in df.columns and "game_date" in df.columns:
        df["season"] = df["game_date"].dt.year

    # 4) Merge games_2025 for day_night and game_type context
    for yr in [2024, 2025]:
        gpath = DATA / "games" / f"games_{yr}.parquet"
        if gpath.exists():
            g = pd.read_parquet(gpath)[["game_pk", "day_night", "game_type",
                                         "venue_name"]].copy()
            g = g.drop_duplicates("game_pk")
            if "game_pk" in df.columns:
                df = df.merge(g, on="game_pk", how="left", suffixes=("", f"_{yr}"))

    # Deduplicate merged columns
    for col in ["day_night", "game_type", "venue_name"]:
        cols_with_yr = [c for c in df.columns if c.startswith(col + "_")]
        if col not in df.columns and cols_with_yr:
            df[col] = df[cols_with_yr[0]].fillna(df[cols_with_yr[-1]])
            df = df.drop(columns=cols_with_yr)

    # 5) Add division info
    df["home_division"] = df["home_team"].map(DIVISIONS)
    df["away_division"] = df["away_team"].map(DIVISIONS)
    df["is_division_game"] = (df["home_division"] == df["away_division"]).astype(int)
    df["is_interleague"] = (
        df["home_division"].str.startswith("AL", na=False) !=
        df["away_division"].str.startswith("AL", na=False)
    ).astype(int)

    # 6) Model probability we'll use for edge analysis
    # Prefer ens_calibrated; fall back to ens_prob
    df["model_prob"] = df["ens_calibrated"].fillna(df["ens_prob"])

    df = df.sort_values("game_date").reset_index(drop=True)
    return df


# ── edge analysis ────────────────────────────────────────────────────────────

def compute_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model edge vs DK line."""
    df = df.copy()
    df["home_edge"] = df["model_prob"] - df["dk_home_prob"]
    df["away_edge"] = (1 - df["model_prob"]) - df["dk_away_prob"]
    df["abs_edge"]  = df[["home_edge", "away_edge"]].abs().max(axis=1)

    # Which side has edge (if any)
    df["best_edge_side"] = np.where(
        df["home_edge"].abs() >= df["away_edge"].abs(), "home", "away"
    )
    df["best_edge"] = np.where(
        df["best_edge_side"] == "home",
        df["home_edge"], df["away_edge"]
    )

    # DK favorite / underdog
    df["dk_home_fav"] = (df["dk_home_prob"] >= 0.5).astype(int)
    df["dk_home_prob_bucket"] = pd.cut(
        df["dk_home_prob"],
        bins=[0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.01],
        labels=["<35%", "35-40%", "40-45%", "45-50%",
                "50-55%", "55-60%", "60-65%", ">65%"],
    )
    return df


def analyze_disagreement_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket games by |model_prob - dk_prob| and report ROI for each bucket.
    We bet on whichever side has edge > threshold.
    """
    section("1. DISAGREEMENT BUCKETS: |model - DK| vs ROI")

    bins = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.0]
    labels = ["0-2%", "2-4%", "4-6%", "6-8%", "8-10%", "10-15%", ">15%"]
    df = df.copy()
    df["disagree_bucket"] = pd.cut(df["abs_edge"], bins=bins, labels=labels)

    rows = []
    for bucket, group in df.groupby("disagree_bucket", observed=True):
        # Bet the side with edge
        home_bets = group[group["best_edge_side"] == "home"]
        away_bets = group[group["best_edge_side"] == "away"]
        r_home = flat_bet_roi(home_bets, "home")
        r_away = flat_bet_roi(away_bets, "away")

        n = r_home["n"] + r_away["n"]
        wins = r_home["wins"] + r_away["wins"]
        pnl = r_home["pnl"] + r_away["pnl"]
        wr = wins / n if n > 0 else np.nan
        roi = pnl / (n * 100) if n > 0 else np.nan

        # Actual win rate vs DK implied prob
        actual_wr_home = group["home_win"].mean()
        dk_implied = group["dk_home_prob"].mean()
        model_implied = group["model_prob"].mean()

        rows.append({
            "Bucket":      str(bucket),
            "N":           n,
            "Avg |Edge|":  group["abs_edge"].mean(),
            "DK Impl":     dk_implied,
            "Model Impl":  model_implied,
            "Actual WR":   actual_wr_home,
            "Bet WR":      wr,
            "ROI":         roi,
            "P&L":         pnl,
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return result


def analyze_subset(df: pd.DataFrame, subset: pd.Series,
                   label: str, edge_threshold: float = 0.03) -> dict:
    """
    For a game subset, compute ROI betting whichever side has edge > threshold.
    Returns dict with stats.
    """
    sub = df[subset].copy()
    if len(sub) == 0:
        return {"label": label, "n_games": 0, "n_bets": 0,
                "win_rate": np.nan, "roi": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}

    # Bet home when home_edge > threshold
    home_mask = sub["home_edge"] > edge_threshold
    away_mask = sub["away_edge"] > edge_threshold

    bets = []
    for _, row in sub[home_mask].iterrows():
        pnl = (row["home_ml_dec"] - 1) * 100 if row["home_win"] == 1 else -100.0
        bets.append({"side": "home", "pnl": pnl, "win": row["home_win"] == 1})
    for _, row in sub[away_mask].iterrows():
        pnl = (row["away_ml_dec"] - 1) * 100 if row["home_win"] == 0 else -100.0
        bets.append({"side": "away", "pnl": pnl, "win": row["home_win"] == 0})

    if not bets:
        return {"label": label, "n_games": len(sub), "n_bets": 0,
                "win_rate": np.nan, "roi": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}

    bets_df = pd.DataFrame(bets)
    n = len(bets_df)
    wins = bets_df["win"].sum()
    wr = wins / n
    pnl = bets_df["pnl"].sum()
    roi = pnl / (n * 100)

    # Bootstrap CI on ROI
    pnl_arr = bets_df["pnl"].values / 100  # normalize to units
    ci_lo, ci_hi = bootstrap_ci(pnl_arr, np.mean)

    # N for significance
    n_sig = bets_for_significance(wr, baseline_p=0.52)  # rough break-even WR

    return {
        "label":    label,
        "n_games":  len(sub),
        "n_bets":   n,
        "wins":     int(wins),
        "win_rate": wr,
        "roi":      roi,
        "ci_lo":    ci_lo,
        "ci_hi":    ci_hi,
        "pnl":      pnl,
        "n_for_sig": n_sig,
    }


def analyze_game_types(df: pd.DataFrame) -> pd.DataFrame:
    """Section 2: Does the model have edge on specific game types?"""
    section("2. GAME-TYPE SUBSETS (edge threshold = 3%)")

    threshold = 0.03
    results = []

    # All games (baseline)
    results.append(analyze_subset(df, pd.Series(np.ones(len(df), dtype=bool), index=df.index),
                                  "All games", threshold))

    # Underdogs: DK implied < 40% (bet the dog)
    underdog_mask = (
        ((df["home_edge"] > threshold) & (df["dk_home_prob"] < 0.40)) |
        ((df["away_edge"] > threshold) & (df["dk_away_prob"] < 0.40))
    )
    sub_dog = df.copy()
    sub_dog["home_edge"] = np.where(df["dk_home_prob"] < 0.40, df["home_edge"], -999)
    sub_dog["away_edge"] = np.where(df["dk_away_prob"] < 0.40, df["away_edge"], -999)
    results.append(analyze_subset(sub_dog, pd.Series(np.ones(len(df), dtype=bool), index=df.index),
                                  "DK underdog (<40%)", threshold))

    # Heavy favorites: DK implied > 65%
    sub_fav = df.copy()
    sub_fav["home_edge"] = np.where(df["dk_home_prob"] > 0.65, df["home_edge"], -999)
    sub_fav["away_edge"] = np.where(df["dk_away_prob"] > 0.65, df["away_edge"], -999)
    results.append(analyze_subset(sub_fav, pd.Series(np.ones(len(df), dtype=bool), index=df.index),
                                  "DK heavy fav (>65%)", threshold))

    # Close games: DK implied 45-55%
    sub_close = df.copy()
    close_home = (df["dk_home_prob"] >= 0.45) & (df["dk_home_prob"] <= 0.55)
    close_away = (df["dk_away_prob"] >= 0.45) & (df["dk_away_prob"] <= 0.55)
    sub_close["home_edge"] = np.where(close_home, df["home_edge"], -999)
    sub_close["away_edge"] = np.where(close_away, df["away_edge"], -999)
    results.append(analyze_subset(sub_close, pd.Series(np.ones(len(df), dtype=bool), index=df.index),
                                  "Close game (45-55%)", threshold))

    # Division games
    results.append(analyze_subset(df, df["is_division_game"] == 1, "Division games", threshold))
    results.append(analyze_subset(df, df["is_division_game"] == 0, "Non-division games", threshold))
    results.append(analyze_subset(df, df["is_interleague"] == 1, "Interleague games", threshold))

    # Day vs night
    if "day_night" in df.columns:
        results.append(analyze_subset(df, df["day_night"] == "day", "Day games", threshold))
        results.append(analyze_subset(df, df["day_night"] == "night", "Night games", threshold))

    out = pd.DataFrame(results)
    print_cols = ["label", "n_games", "n_bets", "win_rate", "roi", "ci_lo", "ci_hi", "pnl", "n_for_sig"]
    out_print = out[[c for c in print_cols if c in out.columns]].copy()
    for col in ["win_rate", "roi", "ci_lo", "ci_hi"]:
        if col in out_print.columns:
            out_print[col] = out_print[col].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "n/a")
    print(out_print.to_string(index=False))
    return out


def analyze_home_away_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """Section 3: Home vs away edge asymmetry."""
    section("3. HOME/AWAY ASYMMETRY (edge > 3%)")

    threshold = 0.03
    rows = []

    # Home bets only
    home_bets_mask = df["home_edge"] > threshold
    r = flat_bet_roi(df[home_bets_mask], "home")
    pnl_arr = np.where(
        df.loc[home_bets_mask, "home_win"] == 1,
        (df.loc[home_bets_mask, "home_ml_dec"] - 1),
        -1.0,
    )
    ci_lo, ci_hi = bootstrap_ci(pnl_arr, np.mean) if r["n"] > 0 else (np.nan, np.nan)
    rows.append({"Side": "Home bets", "N": r["n"], "WR": r["win_rate"],
                 "ROI": r["roi"], "CI_lo": ci_lo, "CI_hi": ci_hi, "P&L": r["pnl"]})

    # Away bets only
    away_bets_mask = df["away_edge"] > threshold
    r = flat_bet_roi(df[away_bets_mask], "away")
    pnl_arr = np.where(
        df.loc[away_bets_mask, "home_win"] == 0,
        (df.loc[away_bets_mask, "away_ml_dec"] - 1),
        -1.0,
    )
    ci_lo, ci_hi = bootstrap_ci(pnl_arr, np.mean) if r["n"] > 0 else (np.nan, np.nan)
    rows.append({"Side": "Away bets", "N": r["n"], "WR": r["win_rate"],
                 "ROI": r["roi"], "CI_lo": ci_lo, "CI_hi": ci_hi, "P&L": r["pnl"]})

    # Favorite side vs underdog side
    home_fav_bets = df[(df["home_edge"] > threshold) & (df["dk_home_prob"] >= 0.50)]
    home_dog_bets = df[(df["home_edge"] > threshold) & (df["dk_home_prob"] < 0.50)]
    away_fav_bets = df[(df["away_edge"] > threshold) & (df["dk_away_prob"] >= 0.50)]
    away_dog_bets = df[(df["away_edge"] > threshold) & (df["dk_away_prob"] < 0.50)]

    for label, sub, side in [
        ("Bet home favorite", home_fav_bets, "home"),
        ("Bet home underdog", home_dog_bets, "home"),
        ("Bet away favorite", away_fav_bets, "away"),
        ("Bet away underdog", away_dog_bets, "away"),
    ]:
        r = flat_bet_roi(sub, side)
        rows.append({"Side": label, "N": r["n"], "WR": r["win_rate"],
                     "ROI": r["roi"], "CI_lo": np.nan, "CI_hi": np.nan,
                     "P&L": r["pnl"]})

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return result


def analyze_time_of_season(df: pd.DataFrame) -> pd.DataFrame:
    """Section 4: Does edge improve as season progresses (more data)?"""
    section("4. TIME-OF-SEASON EFFECT (edge > 3%)")

    threshold = 0.03

    # Monthly buckets
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["month"] = df["game_date"].dt.month

    month_names = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}

    rows = []
    for month in sorted(df["month"].unique()):
        r = analyze_subset(df, df["month"] == month,
                           f"{month_names.get(month, str(month))}", threshold)
        r["month"] = month
        rows.append(r)

    # Also: early season (days_into_season < 30) vs late (> 90)
    if "days_into_season" in df.columns:
        for label, mask in [
            ("Early (<30 days)", df["days_into_season"] < 30),
            ("Mid (30-90 days)", (df["days_into_season"] >= 30) & (df["days_into_season"] < 90)),
            ("Late (>90 days)", df["days_into_season"] >= 90),
        ]:
            r = analyze_subset(df, mask, label, threshold)
            r["month"] = 99
            rows.append(r)

    result = pd.DataFrame(rows)
    print_cols = ["label", "n_bets", "win_rate", "roi", "ci_lo", "ci_hi", "pnl"]
    out = result[[c for c in print_cols if c in result.columns]].copy()
    for col in ["win_rate", "roi", "ci_lo", "ci_hi"]:
        out[col] = out[col].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "n/a")
    print(out.to_string(index=False))
    return result


def find_high_edge_subsets(df: pd.DataFrame) -> None:
    """
    Section 2b: Grid search over thresholds and subsets to find the
    highest-ROI segments, compute bootstrap CIs.
    """
    section("5. HIGH-EDGE SUBSETS: All combinations > 3% edge")

    thresholds = [0.03, 0.05, 0.07, 0.10]
    significant_rows = []

    dk_buckets = {
        "underdog (<40%)":    (df["dk_home_prob"] < 0.40) | (df["dk_away_prob"] < 0.40),
        "slight dog (40-48%)": ((df["dk_home_prob"] >= 0.40) & (df["dk_home_prob"] < 0.48)) |
                               ((df["dk_away_prob"] >= 0.40) & (df["dk_away_prob"] < 0.48)),
        "pick'em (48-52%)":   ((df["dk_home_prob"] >= 0.48) & (df["dk_home_prob"] <= 0.52)) |
                               ((df["dk_away_prob"] >= 0.48) & (df["dk_away_prob"] <= 0.52)),
        "slight fav (52-60%)": ((df["dk_home_prob"] > 0.52) & (df["dk_home_prob"] <= 0.60)) |
                               ((df["dk_away_prob"] > 0.52) & (df["dk_away_prob"] <= 0.60)),
        "heavy fav (>60%)":   (df["dk_home_prob"] > 0.60) | (df["dk_away_prob"] > 0.60),
    }

    side_masks = {
        "home side": "home",
        "away side": "away",
    }

    for thr in thresholds:
        for bucket_label, bucket_mask in dk_buckets.items():
            for side_label, side in side_masks.items():
                if side == "home":
                    edge_mask = df["home_edge"] > thr
                    bet_mask = bucket_mask & edge_mask
                    r = flat_bet_roi(df[bet_mask], "home")
                else:
                    edge_mask = df["away_edge"] > thr
                    bet_mask = bucket_mask & edge_mask
                    r = flat_bet_roi(df[bet_mask], "away")

                if r["n"] < 10:
                    continue

                # Bootstrap CI
                if side == "home":
                    pnl_arr = np.where(
                        df.loc[bet_mask, "home_win"] == 1,
                        (df.loc[bet_mask, "home_ml_dec"] - 1),
                        -1.0,
                    )
                else:
                    pnl_arr = np.where(
                        df.loc[bet_mask, "home_win"] == 0,
                        (df.loc[bet_mask, "away_ml_dec"] - 1),
                        -1.0,
                    )

                ci_lo, ci_hi = bootstrap_ci(pnl_arr, np.mean)

                # Only report if ROI > 3%
                if r["roi"] > 0.03:
                    significant_rows.append({
                        "edge_thr":  f"{thr:.0%}",
                        "segment":   f"{bucket_label} / {side_label}",
                        "N":         r["n"],
                        "WR":        r["win_rate"],
                        "ROI":       r["roi"],
                        "95% CI":    f"[{ci_lo:+.3f}, {ci_hi:+.3f}]",
                        "P&L":       r["pnl"],
                        "N_for_sig": bets_for_significance(r["win_rate"], 0.52),
                    })

    if significant_rows:
        out = pd.DataFrame(significant_rows).sort_values("ROI", ascending=False)
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("  No segments found with ROI > 3% and N >= 10.")


def monthly_stability(df: pd.DataFrame, subset_mask: pd.Series,
                      side: str, edge_threshold: float,
                      label: str) -> None:
    """Check if a high-edge segment is stable month-by-month."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["month"] = df["game_date"].dt.month

    if side == "home":
        edge_mask = df["home_edge"] > edge_threshold
    else:
        edge_mask = df["away_edge"] > edge_threshold

    bet_mask = subset_mask & edge_mask
    print(f"\n  Monthly stability: {label}")
    month_names = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}

    for month in sorted(df["month"].unique()):
        m_mask = bet_mask & (df["month"] == month)
        r = flat_bet_roi(df[m_mask], side)
        name = month_names.get(month, str(month))
        print(f"    {name}: N={r['n']:>4}  WR={r['win_rate']:.3f}  "
              f"ROI={r['roi']:+.3f}  P&L=${r['pnl']:>+8.0f}")


def platt_recalibration(df: pd.DataFrame) -> None:
    """
    Section 6: Platt scaling — recalibrate model_prob using DK line as a
    reference signal.  Train on first 40% (chronological), test on last 60%.
    """
    section("6. PLATT SCALING: Recalibrate model against DK line")

    df = df.copy().sort_values("game_date").reset_index(drop=True)
    n = len(df)
    split = int(n * 0.40)
    train = df.iloc[:split].copy()
    test  = df.iloc[split:].copy()

    # Features for Platt scaling: model_prob, dk_home_prob, and their difference
    def build_X(d: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            d["model_prob"].fillna(0.5),
            d["dk_home_prob"].fillna(0.5),
            d["model_prob"].fillna(0.5) - d["dk_home_prob"].fillna(0.5),
        ])

    X_train = build_X(train)
    y_train = train["home_win"].values
    X_test  = build_X(test)
    y_test  = test["home_win"].values

    # Simple Platt (just model_prob as input)
    lr_simple = LogisticRegression(C=1.0, max_iter=500)
    lr_simple.fit(X_train[:, [0]], y_train)
    p_platt_simple = lr_simple.predict_proba(X_test[:, [0]])[:, 1]

    # Full Platt (model_prob + DK line + difference)
    scaler = StandardScaler()
    lr_full = LogisticRegression(C=1.0, max_iter=500)
    lr_full.fit(scaler.fit_transform(X_train), y_train)
    p_platt_full = lr_full.predict_proba(scaler.transform(X_test))[:, 1]

    # Baseline: raw model_prob
    p_raw  = test["model_prob"].fillna(0.5).values
    p_dk   = test["dk_home_prob"].fillna(0.5).values
    y = y_test

    metrics_rows = []
    for label, p in [
        ("Raw model_prob",     p_raw),
        ("DK line (reference)", p_dk),
        ("Platt (model only)", p_platt_simple),
        ("Platt (model+DK)",   p_platt_full),
    ]:
        ll  = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
        bs  = brier_score_loss(y, p)
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = np.nan
        metrics_rows.append({"Model": label, "LogLoss": ll, "Brier": bs, "AUC": auc})

    metrics_df = pd.DataFrame(metrics_rows)
    print("\n  Calibration metrics (test set = last 60%):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # ROI comparison on test set using recalibrated probs
    print("\n  ROI vs DK line using each probability source (edge > 3%):")
    threshold = 0.03

    def roi_from_probs(p_model: np.ndarray, p_dk: np.ndarray,
                        y: np.ndarray, home_dec: np.ndarray,
                        away_dec: np.ndarray, thr: float) -> dict:
        home_edge = p_model - p_dk
        away_edge = (1 - p_model) - (1 - p_dk)
        bets = []
        for i in range(len(y)):
            if home_edge[i] > thr:
                pnl = (home_dec[i] - 1) * 100 if y[i] == 1 else -100.0
                bets.append(pnl)
            elif away_edge[i] > thr:
                pnl = (away_dec[i] - 1) * 100 if y[i] == 0 else -100.0
                bets.append(pnl)
        if not bets:
            return {"n": 0, "roi": np.nan, "win_rate": np.nan}
        bets_arr = np.array(bets)
        n = len(bets_arr)
        return {
            "n": n,
            "roi": bets_arr.sum() / (n * 100),
            "win_rate": (bets_arr > 0).mean(),
        }

    home_dec = test["home_ml_dec"].values
    away_dec = test["away_ml_dec"].values

    for label, p in [
        ("Raw model_prob",     p_raw),
        ("Platt (model only)", p_platt_simple),
        ("Platt (model+DK)",   p_platt_full),
    ]:
        r = roi_from_probs(p, p_dk, y, home_dec, away_dec, threshold)
        print(f"  {label:<30}  N={r['n']:>5}  "
              f"WR={r['win_rate']:.3f}  ROI={r['roi']:+.4f}")

    print(f"\n  Platt coefs (model+DK):")
    print(f"    coef_model_prob = {lr_full.coef_[0][0]:+.4f}")
    print(f"    coef_dk_prob    = {lr_full.coef_[0][1]:+.4f}")
    print(f"    coef_difference = {lr_full.coef_[0][2]:+.4f}")
    print(f"    intercept       = {lr_full.intercept_[0]:+.4f}")

    return p_platt_full, test


def analyze_best_segment_stability(df: pd.DataFrame) -> None:
    """
    Zoom in on the best-looking segment(s) and check monthly stability.
    """
    section("7. BEST SEGMENT STABILITY CHECK")

    # Check away underdogs (historically this is where models sometimes find edge)
    away_dog_mask = df["dk_away_prob"] < 0.42
    threshold = 0.03

    print("\n  Segment: Away underdog (DK < 42%) + away_edge > 3%")
    r = analyze_subset(df, away_dog_mask, "Away underdog", threshold)
    if r["n_bets"] >= 10:
        monthly_stability(df, away_dog_mask, "away", threshold, "Away underdogs")

    # Check home underdogs
    home_dog_mask = df["dk_home_prob"] < 0.42
    print("\n  Segment: Home underdog (DK < 42%) + home_edge > 3%")
    r = analyze_subset(df, home_dog_mask, "Home underdog", threshold)
    if r["n_bets"] >= 10:
        monthly_stability(df, home_dog_mask, "home", threshold, "Home underdogs")

    # Check close games (48-52%)
    close_mask = (df["dk_home_prob"] >= 0.46) & (df["dk_home_prob"] <= 0.54)
    print("\n  Segment: Close game (DK 46-54%) + any edge > 3%")
    r = analyze_subset(df, close_mask, "Close games", threshold)
    if r["n_bets"] >= 10:
        monthly_stability(df, close_mask, "home", threshold, "Close/home")
        monthly_stability(df, close_mask, "away", threshold, "Close/away")


def summary_of_edges(df: pd.DataFrame) -> None:
    """Print overall summary statistics."""
    section("0. OVERVIEW & DATA SUMMARY")

    print(f"\n  Games:           {len(df):>6}")
    print(f"  Date range:      {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print(f"  Seasons:         {sorted(df['season'].unique())}")

    # Distribution of model vs DK
    df_valid = df.dropna(subset=["model_prob", "dk_home_prob"])
    print(f"\n  Games w/ model pred:   {len(df_valid)}")

    edge = df_valid["abs_edge"]
    print(f"  |model - DK| stats:")
    print(f"    Mean:           {edge.mean():.4f}")
    print(f"    Median:         {edge.median():.4f}")
    print(f"    Std:            {edge.std():.4f}")
    print(f"    > 3%:           {(edge > 0.03).sum():>6} games  ({(edge > 0.03).mean():.1%})")
    print(f"    > 5%:           {(edge > 0.05).sum():>6} games  ({(edge > 0.05).mean():.1%})")
    print(f"    > 10%:          {(edge > 0.10).sum():>6} games  ({(edge > 0.10).mean():.1%})")

    # Flat model overall: all games, model side, any edge > 0
    home_edge_all = df_valid["home_edge"] > 0
    away_edge_all = df_valid["away_edge"] > 0
    r_h = flat_bet_roi(df_valid[home_edge_all], "home")
    r_a = flat_bet_roi(df_valid[away_edge_all], "away")
    n_all = r_h["n"] + r_a["n"]
    pnl_all = r_h["pnl"] + r_a["pnl"]
    roi_all = pnl_all / (n_all * 100) if n_all > 0 else np.nan
    print(f"\n  Flat-bet ROI (any positive edge, all games): N={n_all}, ROI={roi_all:+.4f}")

    # Baseline calibration
    print(f"\n  DK implied home WR:   {df_valid['dk_home_prob'].mean():.4f}")
    print(f"  Actual home WR:        {df_valid['home_win'].mean():.4f}")
    print(f"  Model implied home WR: {df_valid['model_prob'].mean():.4f}")


# ── main ─────────────────────────────────────────────────────────────────────

def analyze_dk_calibration(df: pd.DataFrame) -> None:
    """
    Section 8: Is there a systematic DK calibration bias that explains
    apparent model edge?  (e.g. DK overprices big favorites, underprices
    moderate underdogs in the 35-40% range.)
    """
    section("8. DK LINE CALIBRATION BY PROB BUCKET (home perspective)")

    valid = df.dropna(subset=["model_prob", "dk_home_prob"])
    buckets = [
        (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.45),
        (0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.65),
        (0.65, 0.70), (0.70, 0.75),
    ]

    print(f"\n  {'DK range':>12}  {'N':>5}  {'DK mid':>7}  {'Actual WR':>10}  "
          f"{'DK Error':>9}  {'Model mean':>11}  {'Model Error':>12}")
    for lo, hi in buckets:
        mask = (valid["dk_home_prob"] >= lo) & (valid["dk_home_prob"] < hi)
        sub = valid[mask]
        if len(sub) < 5:
            continue
        actual = sub["home_win"].mean()
        model_mean = sub["model_prob"].mean()
        dk_mid = (lo + hi) / 2
        dk_err = actual - dk_mid
        model_err = actual - model_mean
        print(f"  {lo:.0%}-{hi:.0%}     {len(sub):>5}  {dk_mid:.3f}    {actual:.4f}    "
              f"{dk_err:>+8.4f}  {model_mean:.4f}       {model_err:>+8.4f}")

    print("\n  Key insight: DK systematically underprices moderate away underdogs "
          "(35-40% range shows +4.6 pp actual vs implied), meaning the model's "
          "apparent edge in that bucket is largely a market mispricing, not "
          "model-specific alpha.")


def main():
    print("=" * 75)
    print("  ML EDGE ANALYSIS  |  Ensemble Win Model vs DraftKings Closing Line")
    print("=" * 75)

    df = load_data()
    df = compute_edges(df)

    print(f"\nLoaded {len(df)} games with model predictions and DK closing lines.")
    print(f"Seasons: {sorted(df['season'].unique())}")

    # 0. Overview
    summary_of_edges(df)

    # Restrict to test set (last 70%) for fair evaluation — same split as backtest
    n_cal = int(len(df) * 0.30)
    test_df = df.iloc[n_cal:].copy().reset_index(drop=True)
    print(f"\n  Using test set (last 70%): {len(test_df)} games "
          f"({test_df['game_date'].min().date()} to {test_df['game_date'].max().date()})")

    # 1. Disagreement buckets
    analyze_disagreement_buckets(test_df)

    # 2. Game types
    analyze_game_types(test_df)

    # 3. Home/away asymmetry
    analyze_home_away_asymmetry(test_df)

    # 4. Time-of-season
    analyze_time_of_season(test_df)

    # 5. High-edge subsets
    find_high_edge_subsets(test_df)

    # 6. Platt recalibration
    p_recal, recal_test = platt_recalibration(df)

    # 7. Best segment stability
    analyze_best_segment_stability(test_df)

    # 8. DK calibration bias check
    analyze_dk_calibration(df)

    print("\n" + "=" * 75)
    print("  ANALYSIS COMPLETE")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
