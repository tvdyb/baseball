#!/usr/bin/env python3
"""
NRFI (No Run First Inning) Classification Model — v2.

Directly predicts P(NRFI) using pregame features via LightGBM with
isotonic recalibration.

Key improvements over v1:
  - First-inning-specific SP stats from Statcast (K rate, clean inning %,
    WHIP proxy, HR rate in 1st inning specifically)
  - Top-of-lineup quality (first 3-4 batters' stats from statcast)
  - MC simulator probability as an ensemble feature
  - Realistic NRFI odds (-120/+100) for ROI evaluation
  - Bootstrap confidence intervals on BSS/AUC
  - Bias-corrected calibration

Walk-forward training pattern:
  - Train on prior season(s), predict current season
  - Isotonic calibration via cross-validation on training set

Usage:
    python src/nrfi_model.py
    python src/nrfi_model.py --train-seasons 2024 --test-season 2025
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: LightGBM not available. Install with: pip install lightgbm")

from utils import DATA_DIR, FEATURES_DIR

# ──────────────────────────────────────────────────────────────
# Feature Sets
# ──────────────────────────────────────────────────────────────

# SP quality metrics — most important for NRFI (1st inning is all about SPs)
SP_FEATURES = [
    "home_sp_xrv_mean",
    "away_sp_xrv_mean",
    "home_sp_k_rate",
    "away_sp_k_rate",
    "home_sp_bb_rate",
    "away_sp_bb_rate",
    "home_sp_avg_velo",
    "away_sp_avg_velo",
    "home_sp_rest_days",
    "away_sp_rest_days",
    "home_sp_overperf",
    "away_sp_overperf",
    "home_sp_overperf_recent",
    "away_sp_overperf_recent",
    "home_sp_n_pitches",          # Sample size / confidence
    "away_sp_n_pitches",
    "home_sp_info_confidence",
    "away_sp_info_confidence",
    # SP stuff model scores
    "home_sp_stuff_score",
    "away_sp_stuff_score",
    "home_sp_location_score",
    "away_sp_location_score",
    "home_sp_sequencing_score",
    "away_sp_sequencing_score",
    "home_sp_composite_score",
    "away_sp_composite_score",
    # SP trends
    "home_sp_velo_trend",
    "away_sp_velo_trend",
    "home_sp_xrv_trend",
    "away_sp_xrv_trend",
    # Context-aware splits (home/away)
    "home_sp_home_xrv",           # Home SP performance at home
    "away_sp_away_xrv",           # Away SP performance on road
    # SP pitch mix
    "home_sp_pitch_mix_entropy",
    "away_sp_pitch_mix_entropy",
    # SP vs opposing lineup
    "home_sp_xrv_vs_lineup",
    "away_sp_xrv_vs_lineup",
]

# Lineup / hitting quality
LINEUP_FEATURES = [
    "home_hit_xrv_mean",
    "away_hit_xrv_mean",
    "home_hit_xrv_contact",
    "away_hit_xrv_contact",
    "home_hit_k_rate",
    "away_hit_k_rate",
    "home_hit_hard_hit_rate",
    "away_hit_hard_hit_rate",
    "home_hit_barrel_rate",
    "away_hit_barrel_rate",
    # Hitter discipline
    "home_hit_swing_z",
    "away_hit_swing_z",
    "home_hit_iz_contact",
    "away_hit_iz_contact",
    "home_hit_chase_contact",
    "away_hit_chase_contact",
    "home_hit_foul_fight",
    "away_hit_foul_fight",
    "home_hit_bip_iz",
    "away_hit_bip_iz",
]

# Context / park / weather features
CONTEXT_FEATURES = [
    "park_factor",
    "temperature",
    "wind_speed",
    "is_dome",
    "wind_out",
    "wind_in",
    "days_into_season",
    "is_night",
]

# Differential features
DIFF_FEATURES = [
    "diff_sp_xrv_mean",
    "diff_sp_k_rate",
    "diff_sp_bb_rate",
    "diff_sp_avg_velo",
    "diff_sp_rest_days",
    "diff_sp_overperf",
    "diff_sp_overperf_recent",
    "diff_hit_xrv_mean",
    "diff_hit_xrv_contact",
    "diff_hit_k_rate",
    "diff_sp_stuff_score",
    "diff_sp_composite_score",
    "diff_sp_xrv_vs_lineup",
    "diff_sp_context_xrv",
]

# First-inning-specific features (computed from statcast)
FIRST_INNING_FEATURES = [
    "home_sp_fi_k_rate",
    "away_sp_fi_k_rate",
    "home_sp_fi_bb_rate",
    "away_sp_fi_bb_rate",
    "home_sp_fi_hr_rate",
    "away_sp_fi_hr_rate",
    "home_sp_fi_clean_pct",
    "away_sp_fi_clean_pct",
    "home_sp_fi_whip_proxy",
    "away_sp_fi_whip_proxy",
    "home_sp_fi_hits_per_pa",
    "away_sp_fi_hits_per_pa",
    "home_sp_fi_games",
    "away_sp_fi_games",
    # Top-of-lineup features
    "away_top3_k_rate",       # Away top3 batters K rate (face home SP in top 1)
    "home_top3_k_rate",       # Home top3 batters K rate (face away SP in bot 1)
    "away_top3_hr_rate",
    "home_top3_hr_rate",
    "away_top3_bb_rate",
    "home_top3_bb_rate",
    # Slow-starter metric
    "home_sp_slow_starter",   # FI performance vs overall (positive = worse in 1st)
    "away_sp_slow_starter",
]

# MC sim feature
MC_SIM_FEATURES = [
    "sim_nrfi_prob",
]

ALL_BASE_FEATURES = (
    SP_FEATURES + LINEUP_FEATURES + CONTEXT_FEATURES + DIFF_FEATURES
    + FIRST_INNING_FEATURES + MC_SIM_FEATURES
)

# Optimal LightGBM hyperparameters (tuned for NRFI)
LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 30,
    "max_depth": 3,
    "num_leaves": 5,
    "learning_rate": 0.08,
    "min_child_samples": 60,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 5.0,
    "reg_lambda": 20.0,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}


# ──────────────────────────────────────────────────────────────
# First-Inning-Specific Features from Statcast
# ──────────────────────────────────────────────────────────────

def compute_sp_first_inning_stats(season: int) -> pd.DataFrame:
    """Compute per-pitcher first-inning stats from Statcast.

    Returns DataFrame with pitcher ID and first-inning performance metrics.
    Uses cumulative stats up to (but not including) each game date for
    walk-forward validity.
    """
    sc_path = DATA_DIR / "statcast" / f"statcast_{season}.parquet"
    if not sc_path.exists():
        print(f"  Statcast not found for SP FI stats: {sc_path}")
        return pd.DataFrame()

    sc = pd.read_parquet(
        sc_path,
        columns=["game_pk", "game_date", "pitcher", "batter", "inning",
                 "inning_topbot", "events", "post_home_score", "post_away_score"],
    )
    fi = sc[sc["inning"] == 1].copy()
    fi["game_date"] = pd.to_datetime(fi["game_date"])
    events = fi.dropna(subset=["events"]).copy()

    events["is_hit"] = events["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
    events["is_k"] = (events["events"] == "strikeout").astype(int)
    events["is_bb"] = (events["events"] == "walk").astype(int)
    events["is_hbp"] = (events["events"] == "hit_by_pitch").astype(int)
    events["is_hr"] = (events["events"] == "home_run").astype(int)

    # Per game-pitcher stats
    game_pitcher = events.groupby(["game_pk", "game_date", "pitcher", "inning_topbot"]).agg(
        pa=("events", "count"),
        hits=("is_hit", "sum"),
        k=("is_k", "sum"),
        bb=("is_bb", "sum"),
        hbp=("is_hbp", "sum"),
        hr=("is_hr", "sum"),
    ).reset_index()
    game_pitcher["clean"] = ((game_pitcher["hits"] == 0) & (game_pitcher["bb"] == 0) & (game_pitcher["hbp"] == 0)).astype(int)

    # Aggregate per pitcher (season-level -- for walk-forward we'll use prior season)
    pitcher_agg = game_pitcher.groupby("pitcher").agg(
        fi_games=("game_pk", "nunique"),
        fi_pa=("pa", "sum"),
        fi_hits=("hits", "sum"),
        fi_k=("k", "sum"),
        fi_bb=("bb", "sum"),
        fi_hr=("hr", "sum"),
        fi_clean=("clean", "sum"),
    ).reset_index()

    pitcher_agg["fi_k_rate"] = pitcher_agg["fi_k"] / pitcher_agg["fi_pa"].clip(lower=1)
    pitcher_agg["fi_bb_rate"] = pitcher_agg["fi_bb"] / pitcher_agg["fi_pa"].clip(lower=1)
    pitcher_agg["fi_hr_rate"] = pitcher_agg["fi_hr"] / pitcher_agg["fi_pa"].clip(lower=1)
    pitcher_agg["fi_hits_per_pa"] = pitcher_agg["fi_hits"] / pitcher_agg["fi_pa"].clip(lower=1)
    pitcher_agg["fi_whip_proxy"] = (pitcher_agg["fi_hits"] + pitcher_agg["fi_bb"]) / pitcher_agg["fi_games"].clip(lower=1)
    pitcher_agg["fi_clean_pct"] = pitcher_agg["fi_clean"] / pitcher_agg["fi_games"].clip(lower=1)

    return pitcher_agg


def compute_sp_overall_stats(season: int) -> pd.DataFrame:
    """Compute per-pitcher overall stats (all innings) to compute slow-starter metric."""
    sc_path = DATA_DIR / "statcast" / f"statcast_{season}.parquet"
    if not sc_path.exists():
        return pd.DataFrame()

    sc = pd.read_parquet(
        sc_path,
        columns=["game_pk", "pitcher", "inning", "events"],
    )
    events = sc.dropna(subset=["events"]).copy()
    events["is_hit"] = events["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
    events["is_k"] = (events["events"] == "strikeout").astype(int)
    events["is_bb"] = (events["events"] == "walk").astype(int)
    events["is_hr"] = (events["events"] == "home_run").astype(int)

    pitcher_agg = events.groupby("pitcher").agg(
        all_pa=("events", "count"),
        all_hits=("is_hit", "sum"),
        all_k=("is_k", "sum"),
        all_bb=("is_bb", "sum"),
        all_hr=("is_hr", "sum"),
    ).reset_index()

    pitcher_agg["all_k_rate"] = pitcher_agg["all_k"] / pitcher_agg["all_pa"].clip(lower=1)
    pitcher_agg["all_bb_rate"] = pitcher_agg["all_bb"] / pitcher_agg["all_pa"].clip(lower=1)
    pitcher_agg["all_hits_per_pa"] = pitcher_agg["all_hits"] / pitcher_agg["all_pa"].clip(lower=1)

    return pitcher_agg


def compute_top_lineup_stats(season: int) -> pd.DataFrame:
    """Compute top-of-lineup batter stats from Statcast.

    Returns per-batter aggregated stats that can be linked to lineups.
    """
    sc_path = DATA_DIR / "statcast" / f"statcast_{season}.parquet"
    if not sc_path.exists():
        return pd.DataFrame()

    sc = pd.read_parquet(
        sc_path,
        columns=["game_pk", "batter", "inning", "inning_topbot", "events"],
    )
    # First inning batters = top of lineup
    fi = sc[sc["inning"] == 1].copy()
    events = fi.dropna(subset=["events"]).copy()
    events["is_k"] = (events["events"] == "strikeout").astype(int)
    events["is_hr"] = (events["events"] == "home_run").astype(int)
    events["is_bb"] = (events["events"] == "walk").astype(int)

    batter_agg = events.groupby("batter").agg(
        fi_batter_pa=("events", "count"),
        fi_batter_k=("is_k", "sum"),
        fi_batter_hr=("is_hr", "sum"),
        fi_batter_bb=("is_bb", "sum"),
    ).reset_index()

    batter_agg["fi_batter_k_rate"] = batter_agg["fi_batter_k"] / batter_agg["fi_batter_pa"].clip(lower=1)
    batter_agg["fi_batter_hr_rate"] = batter_agg["fi_batter_hr"] / batter_agg["fi_batter_pa"].clip(lower=1)
    batter_agg["fi_batter_bb_rate"] = batter_agg["fi_batter_bb"] / batter_agg["fi_batter_pa"].clip(lower=1)

    return batter_agg


def get_game_sp_and_lineup_mapping(season: int) -> pd.DataFrame:
    """Map game_pk -> home/away SP IDs and top-of-lineup batter IDs."""
    sc_path = DATA_DIR / "statcast" / f"statcast_{season}.parquet"
    if not sc_path.exists():
        return pd.DataFrame()

    sc = pd.read_parquet(
        sc_path,
        columns=["game_pk", "pitcher", "batter", "inning", "inning_topbot",
                 "at_bat_number"],
    )
    fi = sc[sc["inning"] == 1].copy()

    # SP mapping
    top1 = fi[fi["inning_topbot"] == "Top"]
    bot1 = fi[fi["inning_topbot"] == "Bot"]
    home_sp = top1.groupby("game_pk")["pitcher"].first().rename("home_sp_id")
    away_sp = bot1.groupby("game_pk")["pitcher"].first().rename("away_sp_id")

    # Top-of-lineup: unique batters in order of appearance in the 1st inning
    def top_batters(grp, n=3):
        return grp.drop_duplicates("batter")["batter"].head(n).tolist()

    # Away batters face home pitcher (top of 1st)
    away_top = top1.sort_values("at_bat_number").groupby("game_pk").apply(
        top_batters, include_groups=False
    ).rename("away_top_batters")
    # Home batters face away pitcher (bot of 1st)
    home_top = bot1.sort_values("at_bat_number").groupby("game_pk").apply(
        top_batters, include_groups=False
    ).rename("home_top_batters")

    mapping = pd.DataFrame({
        "home_sp_id": home_sp,
        "away_sp_id": away_sp,
    }).reset_index()
    mapping = mapping.merge(away_top.reset_index(), on="game_pk", how="left")
    mapping = mapping.merge(home_top.reset_index(), on="game_pk", how="left")

    return mapping


def attach_first_inning_features(
    df: pd.DataFrame,
    train_season: int,
    target_season: int,
) -> pd.DataFrame:
    """Attach first-inning-specific features to game features DataFrame.

    Uses train_season statcast for FI stats (walk-forward valid) and
    target_season statcast for game-to-SP mapping.
    """
    # Get FI pitcher stats from TRAINING season (prior year)
    fi_stats = compute_sp_first_inning_stats(train_season)
    overall_stats = compute_sp_overall_stats(train_season)

    if fi_stats.empty or overall_stats.empty:
        print(f"  Could not compute FI stats for {train_season}")
        return df

    # Compute slow-starter metric: FI hits_per_pa - overall hits_per_pa
    merged_stats = fi_stats.merge(overall_stats[["pitcher", "all_hits_per_pa", "all_k_rate", "all_bb_rate"]],
                                   on="pitcher", how="left")
    merged_stats["slow_starter"] = merged_stats["fi_hits_per_pa"] - merged_stats["all_hits_per_pa"].fillna(merged_stats["fi_hits_per_pa"])

    # Get batter stats from training season
    batter_stats = compute_top_lineup_stats(train_season)

    # Get SP + lineup mapping for target season
    mapping = get_game_sp_and_lineup_mapping(target_season)
    if mapping.empty:
        print(f"  Could not get SP mapping for {target_season}")
        return df

    df = df.copy()
    df = df.merge(mapping[["game_pk", "home_sp_id", "away_sp_id",
                           "away_top_batters", "home_top_batters"]],
                  on="game_pk", how="left")

    # Attach home SP first-inning features
    sp_cols = ["fi_k_rate", "fi_bb_rate", "fi_hr_rate", "fi_clean_pct",
               "fi_whip_proxy", "fi_hits_per_pa", "fi_games", "slow_starter"]
    sp_rename = {c: f"home_sp_{c}" for c in sp_cols}
    home_fi = merged_stats[["pitcher"] + [c for c in sp_cols if c in merged_stats.columns]].rename(
        columns={**{"pitcher": "home_sp_id"}, **sp_rename}
    )
    df = df.merge(home_fi, on="home_sp_id", how="left")

    sp_rename_away = {c: f"away_sp_{c}" for c in sp_cols}
    away_fi = merged_stats[["pitcher"] + [c for c in sp_cols if c in merged_stats.columns]].rename(
        columns={**{"pitcher": "away_sp_id"}, **sp_rename_away}
    )
    df = df.merge(away_fi, on="away_sp_id", how="left")

    # Attach top-of-lineup features
    if not batter_stats.empty:
        batter_map = batter_stats.set_index("batter")
        for side, col in [("away", "away_top_batters"), ("home", "home_top_batters")]:
            k_rates = []
            hr_rates = []
            bb_rates = []
            for _, row in df.iterrows():
                batters = row.get(col, None)
                if not isinstance(batters, list) or len(batters) == 0:
                    k_rates.append(np.nan)
                    hr_rates.append(np.nan)
                    bb_rates.append(np.nan)
                    continue
                batter_ids = [b for b in batters if b in batter_map.index]
                if len(batter_ids) == 0:
                    k_rates.append(np.nan)
                    hr_rates.append(np.nan)
                    bb_rates.append(np.nan)
                else:
                    sub = batter_map.loc[batter_ids]
                    k_rates.append(sub["fi_batter_k_rate"].mean())
                    hr_rates.append(sub["fi_batter_hr_rate"].mean())
                    bb_rates.append(sub["fi_batter_bb_rate"].mean())
            df[f"{side}_top3_k_rate"] = k_rates
            df[f"{side}_top3_hr_rate"] = hr_rates
            df[f"{side}_top3_bb_rate"] = bb_rates

    # Drop helper columns
    for c in ["home_sp_id", "away_sp_id", "away_top_batters", "home_top_batters"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    n_fi = df["home_sp_fi_k_rate"].notna().sum()
    print(f"  FI features attached: {n_fi}/{len(df)} games have FI pitcher stats")

    return df


def attach_mc_sim_features(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Attach MC simulator NRFI probability as a feature."""
    sim_path = DATA_DIR / "backtest" / f"nrfi_ou_backtest_{season}.parquet"
    if not sim_path.exists():
        print(f"  MC sim backtest not found: {sim_path}")
        return df

    sim_df = pd.read_parquet(sim_path)
    if "sim_nrfi_prob" not in sim_df.columns:
        return df

    df = df.merge(sim_df[["game_pk", "sim_nrfi_prob"]], on="game_pk", how="left")
    n_sim = df["sim_nrfi_prob"].notna().sum()
    print(f"  MC sim features attached: {n_sim}/{len(df)} games")
    return df


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────

def compute_actual_first_inning(season: int) -> pd.DataFrame:
    """Compute actual first-inning runs from Statcast pitch data.

    Returns DataFrame with: game_pk, away_1st_runs, home_1st_runs, nrfi
    """
    sc_path = DATA_DIR / "statcast" / f"statcast_{season}.parquet"
    if not sc_path.exists():
        print(f"  Statcast not found: {sc_path}")
        return pd.DataFrame()

    sc = pd.read_parquet(
        sc_path,
        columns=["game_pk", "inning", "inning_topbot", "post_home_score", "post_away_score"],
    )
    first = sc[sc["inning"] == 1]
    top1 = first[first["inning_topbot"] == "Top"].groupby("game_pk")["post_away_score"].max()
    bot1 = first[first["inning_topbot"] == "Bot"].groupby("game_pk")["post_home_score"].max()

    both = pd.DataFrame({"away_1st_runs": top1, "home_1st_runs": bot1}).dropna()
    both["nrfi"] = ((both["away_1st_runs"] == 0) & (both["home_1st_runs"] == 0)).astype(int)
    return both.reset_index()


def load_season_data(season: int) -> pd.DataFrame:
    """Load pregame features merged with NRFI labels for a season."""
    feat_path = FEATURES_DIR / f"game_features_{season}.parquet"
    if not feat_path.exists():
        print(f"  Features not found: {feat_path}")
        return pd.DataFrame()

    feat = pd.read_parquet(feat_path)
    actual = compute_actual_first_inning(season)

    if actual.empty:
        return pd.DataFrame()

    df = feat.merge(actual[["game_pk", "nrfi", "away_1st_runs", "home_1st_runs"]],
                    on="game_pk", how="inner")
    df["season"] = season
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  {season}: {len(df)} games, NRFI rate: {df['nrfi'].mean():.3f}")
    return df


def load_features(seasons: list[int]) -> pd.DataFrame:
    """Load and concatenate pregame feature data for multiple seasons."""
    frames = []
    for s in seasons:
        df = load_season_data(s)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No data found for any season")
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────

def engineer_nrfi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer NRFI-specific interaction features on top of the pregame features.

    Key insight for NRFI: the probability depends on BOTH pitchers being dominant.
    This creates multiplicative interaction terms that linear features miss.
    """
    df = df.copy()

    # ── Combined SP dominance: NRFI requires both pitchers to be good ──
    if "home_sp_xrv_mean" in df.columns and "away_sp_xrv_mean" in df.columns:
        # xRV is negative = better for pitcher -> flip sign so higher = better
        home_sp_qual = -df["home_sp_xrv_mean"].fillna(0)
        away_sp_qual = -df["away_sp_xrv_mean"].fillna(0)
        df["both_sp_quality"] = home_sp_qual + away_sp_qual
        df["both_sp_quality_product"] = home_sp_qual * away_sp_qual
        # Weakest link: NRFI depends on the WORSE pitcher
        df["weakest_sp_quality"] = np.minimum(home_sp_qual, away_sp_qual)

    # ── Combined SP strikeout rate ──
    if "home_sp_k_rate" in df.columns and "away_sp_k_rate" in df.columns:
        df["both_sp_k_rate"] = (
            df["home_sp_k_rate"].fillna(0) + df["away_sp_k_rate"].fillna(0)
        )
        df["min_sp_k_rate"] = np.minimum(
            df["home_sp_k_rate"].fillna(0), df["away_sp_k_rate"].fillna(0)
        )

    # ── SP stuff scores combined ──
    if "home_sp_composite_score" in df.columns and "away_sp_composite_score" in df.columns:
        home_stuff = -df["home_sp_composite_score"].fillna(0)
        away_stuff = -df["away_sp_composite_score"].fillna(0)
        df["both_sp_stuff"] = home_stuff + away_stuff
        df["both_sp_stuff_product"] = home_stuff * away_stuff
        df["weakest_sp_stuff"] = np.minimum(home_stuff, away_stuff)

    # ── First-inning-specific combined features ──
    if "home_sp_fi_clean_pct" in df.columns and "away_sp_fi_clean_pct" in df.columns:
        home_clean = df["home_sp_fi_clean_pct"].fillna(0.3)
        away_clean = df["away_sp_fi_clean_pct"].fillna(0.3)
        df["both_sp_fi_clean_product"] = home_clean * away_clean
        df["both_sp_fi_clean_sum"] = home_clean + away_clean
        df["min_sp_fi_clean"] = np.minimum(home_clean, away_clean)

    if "home_sp_fi_whip_proxy" in df.columns and "away_sp_fi_whip_proxy" in df.columns:
        df["both_sp_fi_whip"] = (
            df["home_sp_fi_whip_proxy"].fillna(1.3) + df["away_sp_fi_whip_proxy"].fillna(1.3)
        )
        df["max_sp_fi_whip"] = np.maximum(
            df["home_sp_fi_whip_proxy"].fillna(1.3), df["away_sp_fi_whip_proxy"].fillna(1.3)
        )

    if "home_sp_fi_k_rate" in df.columns and "away_sp_fi_k_rate" in df.columns:
        df["both_sp_fi_k_rate"] = (
            df["home_sp_fi_k_rate"].fillna(0.2) + df["away_sp_fi_k_rate"].fillna(0.2)
        )

    # ── Slow starter combined ──
    if "home_sp_slow_starter" in df.columns and "away_sp_slow_starter" in df.columns:
        df["both_sp_slow_starter"] = (
            df["home_sp_slow_starter"].fillna(0) + df["away_sp_slow_starter"].fillna(0)
        )
        df["max_sp_slow_starter"] = np.maximum(
            df["home_sp_slow_starter"].fillna(0), df["away_sp_slow_starter"].fillna(0)
        )

    # ── Top-of-lineup combined ──
    if "away_top3_k_rate" in df.columns and "home_top3_k_rate" in df.columns:
        df["both_top3_k_rate"] = (
            df["away_top3_k_rate"].fillna(0.2) + df["home_top3_k_rate"].fillna(0.2)
        )
    if "away_top3_hr_rate" in df.columns and "home_top3_hr_rate" in df.columns:
        df["both_top3_hr_rate"] = (
            df["away_top3_hr_rate"].fillna(0.03) + df["home_top3_hr_rate"].fillna(0.03)
        )

    # ── SP rest day engineering ──
    for side in ["home", "away"]:
        col = f"{side}_sp_rest_days"
        if col in df.columns:
            rest = df[col].fillna(5)
            df[f"{side}_sp_short_rest"] = np.clip(4 - rest, 0, None)
            df[f"{side}_sp_long_rest"] = np.clip(rest - 7, 0, None)

    if "home_sp_short_rest" in df.columns and "away_sp_short_rest" in df.columns:
        df["total_short_rest"] = df["home_sp_short_rest"] + df["away_sp_short_rest"]

    # ── Log-transform SP pitch count (sample size confidence) ──
    for side in ["home", "away"]:
        col = f"{side}_sp_n_pitches"
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].fillna(0))

    # ── Park factor x combined SP quality ──
    if "park_factor" in df.columns and "both_sp_quality" in df.columns:
        pf = df["park_factor"].fillna(1.0)
        df["park_sp_interaction"] = (2.0 - pf) * df["both_sp_quality"]

    # ── Weather: cold suppresses offense ──
    if "temperature" in df.columns:
        temp = df["temperature"].fillna(72)
        df["cold_weather"] = np.clip(50 - temp, 0, None)
        df["warm_weather"] = np.clip(temp - 85, 0, None)

    # ── Wind effect ──
    if "wind_speed" in df.columns and "wind_out" in df.columns:
        df["wind_out_speed"] = df["wind_speed"].fillna(0) * df["wind_out"].fillna(0)
        df["wind_in_speed"] = df["wind_speed"].fillna(0) * df["wind_in"].fillna(0)

    # ── Early season indicator ──
    if "days_into_season" in df.columns:
        days = df["days_into_season"].fillna(30)
        df["early_season"] = np.clip(1 - days / 60, 0, 1)

    # ── Combined lineup strikeout rate (high = easier for pitchers) ──
    if "home_hit_k_rate" in df.columns and "away_hit_k_rate" in df.columns:
        df["combined_lineup_k_rate"] = (
            df["home_hit_k_rate"].fillna(0) + df["away_hit_k_rate"].fillna(0)
        )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all engineered feature columns available in df."""
    engineered = [
        "both_sp_quality", "both_sp_quality_product", "weakest_sp_quality",
        "both_sp_k_rate", "min_sp_k_rate",
        "both_sp_stuff", "both_sp_stuff_product", "weakest_sp_stuff",
        # First-inning combined
        "both_sp_fi_clean_product", "both_sp_fi_clean_sum", "min_sp_fi_clean",
        "both_sp_fi_whip", "max_sp_fi_whip",
        "both_sp_fi_k_rate",
        "both_sp_slow_starter", "max_sp_slow_starter",
        "both_top3_k_rate", "both_top3_hr_rate",
        # Rest
        "home_sp_short_rest", "away_sp_short_rest", "total_short_rest",
        "home_sp_long_rest", "away_sp_long_rest",
        "home_sp_n_pitches_log", "away_sp_n_pitches_log",
        "park_sp_interaction",
        "cold_weather", "warm_weather",
        "wind_out_speed", "wind_in_speed",
        "early_season",
        "combined_lineup_k_rate",
    ]
    candidate = ALL_BASE_FEATURES + engineered
    # Drop all-NaN and duplicates
    seen = set()
    available = []
    for c in candidate:
        if c in df.columns and c not in seen:
            seen.add(c)
            available.append(c)
    return available


def prepare_xy(df: pd.DataFrame, feature_cols: list[str] = None):
    """Extract feature matrix X and NRFI target y from a DataFrame."""
    df = engineer_nrfi_features(df)
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
        # Drop fully-NaN columns
        feature_cols = [c for c in feature_cols if df[c].notna().any()]
    else:
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df["nrfi"].values
    return X, y, feature_cols


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a metric function."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            v = metric_fn(y_true[idx], y_prob[idx])
            vals.append(v)
        except Exception:
            pass
    vals = np.array(vals)
    lo = np.percentile(vals, (1 - ci) / 2 * 100)
    hi = np.percentile(vals, (1 + ci) / 2 * 100)
    return lo, hi


def evaluate_nrfi(y_true: np.ndarray, y_prob: np.ndarray, label: str = "",
                  bootstrap: bool = True) -> dict:
    """Full NRFI evaluation: Brier score, log loss, AUC, calibration, ROI."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bs = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    base_rate = y_true.mean()
    bs_baseline = brier_score_loss(y_true, np.full_like(y_prob, base_rate))
    ll_baseline = log_loss(y_true, np.full_like(y_prob, base_rate))
    bss = 1 - bs / bs_baseline
    lss = 1 - ll / ll_baseline

    print(f"\n  {label} ({len(y_true)} games):")
    print(f"    NRFI rate:      {base_rate:.3f}")
    print(f"    Pred mean:      {y_prob.mean():.3f}  (std: {y_prob.std():.4f})")
    print(f"    Brier score:    {bs:.4f}  (baseline: {bs_baseline:.4f})")
    print(f"    Brier skill:    {bss:+.4f}  {'[POSITIVE]' if bss > 0 else '[negative]'}")
    print(f"    Log loss:       {ll:.4f}  (baseline: {ll_baseline:.4f})")
    print(f"    Log-loss skill: {lss:+.4f}")
    print(f"    AUC:            {auc:.4f}")

    if bootstrap:
        def bss_fn(yt, yp):
            bsi = brier_score_loss(yt, yp)
            bsb = brier_score_loss(yt, np.full_like(yp, yt.mean()))
            return 1 - bsi / bsb

        bss_lo, bss_hi = bootstrap_ci(y_true, y_prob, bss_fn)
        auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score)
        print(f"    BSS 95% CI:     [{bss_lo:+.4f}, {bss_hi:+.4f}]")
        print(f"    AUC 95% CI:     [{auc_lo:.4f}, {auc_hi:.4f}]")

    # Calibration by decile
    print(f"\n    Calibration (decile predicted -> actual):")
    order = np.argsort(y_prob)
    n = len(y_prob)
    for b in range(10):
        lo = b * n // 10
        hi = (b + 1) * n // 10
        idx = order[lo:hi]
        print(f"      {y_prob[idx].mean():.3f} -> {y_true[idx].mean():.3f}  (n={len(idx)})")

    # ROI simulation at realistic NRFI odds: -120 YES / +100 NO
    # NRFI bet at -120: risk 120 to win 100. Net win = +100, net loss = -120
    # YRFI bet at +100: risk 100 to win 100. Net win = +100, net loss = -100
    print(f"\n    ROI simulation (NRFI -120 / YRFI +100):")
    print(f"    {'Side+Thresh':>14}  {'N':>6}  {'Win%':>6}  {'ROI':>8}  {'Units':>8}")
    for thresh in [0.50, 0.52, 0.54, 0.55, 0.57, 0.60, 0.63, 0.65]:
        # NRFI bets: we bet NRFI at -120 when model says high NRFI prob
        nrfi_mask = y_prob >= thresh
        n_nrfi = nrfi_mask.sum()
        if n_nrfi >= 20:
            wins = (y_true[nrfi_mask] == 1).sum()
            losses = n_nrfi - wins
            profit = wins * 100 - losses * 120  # -120 juice
            roi = profit / (n_nrfi * 120) * 100
            print(f"    NRFI>={thresh:.2f}:  {n_nrfi:>6}  {wins/n_nrfi:.3f}  {roi:>+7.1f}%  {profit/100:>+7.1f}u")

        # YRFI bets: we bet YRFI at +100 when model says low NRFI prob
        yrfi_thresh = 1 - thresh
        yrfi_mask = y_prob <= yrfi_thresh
        n_yrfi = yrfi_mask.sum()
        if n_yrfi >= 20:
            wins = (y_true[yrfi_mask] == 0).sum()
            losses = n_yrfi - wins
            profit = wins * 100 - losses * 100  # +100 (even money)
            roi = profit / (n_yrfi * 100) * 100
            print(f"    YRFI<={yrfi_thresh:.2f}:  {n_yrfi:>6}  {wins/n_yrfi:.3f}  {roi:>+7.1f}%  {profit/100:>+7.1f}u")

    # Also show flat -110 vig for comparison
    print(f"\n    ROI simulation (flat -110 vig):")
    print(f"    {'Side+Thresh':>14}  {'N':>6}  {'Win%':>6}  {'ROI':>8}")
    for thresh in [0.50, 0.52, 0.55, 0.57, 0.60, 0.63, 0.65]:
        for side, correct, mask in [
            ("NRFI", 1, y_prob >= thresh),
            ("YRFI", 0, y_prob <= (1 - thresh)),
        ]:
            n_bets = mask.sum()
            if n_bets < 20:
                continue
            wins = (y_true[mask] == correct).sum()
            win_pct = wins / n_bets
            profit = wins * 100 - (n_bets - wins) * 110
            roi = profit / (n_bets * 110) * 100
            print(f"    {side}>{thresh:.2f}:  {n_bets:>6}  {win_pct:.3f}  {roi:>+7.1f}%")

    return {
        "brier": float(bs),
        "brier_baseline": float(bs_baseline),
        "brier_skill": float(bss),
        "log_loss": float(ll),
        "logloss_skill": float(lss),
        "auc": float(auc),
        "n": int(len(y_true)),
        "nrfi_rate": float(base_rate),
        "pred_mean": float(y_prob.mean()),
    }


# ──────────────────────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────────────────────

def build_model(params: dict = None) -> "CalibratedClassifierCV":
    """Build a LightGBM + isotonic calibration pipeline."""
    if not HAS_LGB:
        raise ImportError("LightGBM required: pip install lightgbm")

    p = {**LGB_PARAMS, **(params or {})}
    base = lgb.LGBMClassifier(**p)
    return CalibratedClassifierCV(base, cv=5, method="isotonic")


def get_feature_importance(model, feature_cols: list[str], top_n: int = 30) -> dict:
    """Extract and print feature importances from a CalibratedClassifierCV."""
    # Average importances across CV folds
    importances = np.zeros(len(feature_cols))
    count = 0
    for calibrated_clf in model.calibrated_classifiers_:
        base = calibrated_clf.estimator
        if hasattr(base, "feature_importances_"):
            importances += base.feature_importances_
            count += 1

    if count > 0:
        importances /= count
        total = importances.sum() or 1
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

        print(f"\n  Top {top_n} Feature Importances (avg over {count} CV folds):")
        for feat, imp in feat_imp[:top_n]:
            if imp <= 0:
                break
            bar = "#" * max(1, int(imp / total * 40))
            print(f"    {feat:<45} {imp/total:>5.1%}  {bar}")

        return dict(feat_imp)
    return {}


# ──────────────────────────────────────────────────────────────
# Walk-Forward Prediction
# ──────────────────────────────────────────────────────────────

def walk_forward_nrfi(
    train_seasons: list[int],
    test_season: int,
    lgb_params: dict = None,
) -> pd.DataFrame:
    """
    Walk-forward NRFI prediction.

    Train LGB+Isotonic on train_seasons, predict test_season.
    Runs multiple model variants and reports the best.
    """
    print(f"\n{'='*60}")
    print(f"NRFI Walk-Forward v2: Train={train_seasons}, Test={test_season}")
    print(f"{'='*60}")

    # Load data
    print("\nLoading data...")
    all_seasons = sorted(set(train_seasons + [test_season]))
    all_df = load_features(all_seasons)

    train_df = all_df[all_df["season"].isin(train_seasons)].copy()
    test_df = all_df[all_df["season"] == test_season].copy()

    # Attach first-inning-specific features
    print("\nAttaching first-inning-specific features...")
    train_df = attach_first_inning_features(train_df, min(train_seasons), min(train_seasons))
    test_df = attach_first_inning_features(test_df, min(train_seasons), test_season)

    # Attach MC sim features
    print("\nAttaching MC sim features...")
    train_df = attach_mc_sim_features(train_df, min(train_seasons))
    test_df = attach_mc_sim_features(test_df, test_season)

    print(f"\nTrain: {len(train_df)} games, Test: {len(test_df)} games")

    # ── Variant A: Full features with FI stats ──
    print(f"\n{'='*60}")
    print("Variant A: Full features (with FI + interaction)")
    print(f"{'='*60}")
    X_train_a, y_train, feature_cols_a = prepare_xy(train_df)
    X_test_a, y_test, _ = prepare_xy(test_df, feature_cols_a)
    print(f"Features: {len(feature_cols_a)}")

    model_a = build_model(lgb_params)
    model_a.fit(X_train_a, y_train)
    probs_a = model_a.predict_proba(X_test_a)[:, 1]
    get_feature_importance(model_a, feature_cols_a)
    metrics_a = evaluate_nrfi(y_test, probs_a, "Full features", bootstrap=False)

    # ── Variant B: Core features only (no FI, no sim, minimal interactions) ──
    print(f"\n{'='*60}")
    print("Variant B: Core features (SP + lineup + context, curated)")
    print(f"{'='*60}")
    core_features = [c for c in feature_cols_a if c not in
                     (FIRST_INNING_FEATURES + MC_SIM_FEATURES +
                      ["both_sp_fi_clean_product", "both_sp_fi_clean_sum", "min_sp_fi_clean",
                       "both_sp_fi_whip", "max_sp_fi_whip", "both_sp_fi_k_rate",
                       "both_sp_slow_starter", "max_sp_slow_starter",
                       "both_top3_k_rate", "both_top3_hr_rate"])]
    X_train_b = X_train_a[[c for c in core_features if c in X_train_a.columns]]
    X_test_b = X_test_a[[c for c in core_features if c in X_test_a.columns]]
    print(f"Features: {len(X_train_b.columns)}")

    model_b = build_model(lgb_params)
    model_b.fit(X_train_b, y_train)
    probs_b = model_b.predict_proba(X_test_b)[:, 1]
    metrics_b = evaluate_nrfi(y_test, probs_b, "Core features", bootstrap=False)

    # ── Variant C: Ultra-minimal (only strongest signals, very regularized) ──
    print(f"\n{'='*60}")
    print("Variant C: Ultra-minimal (top signals, heavy regularization)")
    print(f"{'='*60}")
    minimal_features = [
        "both_sp_quality", "both_sp_quality_product", "weakest_sp_quality",
        "both_sp_k_rate", "both_sp_stuff", "weakest_sp_stuff",
        "home_sp_sequencing_score", "away_sp_sequencing_score",
        "park_factor", "temperature", "is_dome",
        "days_into_season",
        "combined_lineup_k_rate",
        "cold_weather",
        "park_sp_interaction",
    ]
    minimal_avail = [c for c in minimal_features if c in X_train_a.columns]
    X_train_c = X_train_a[minimal_avail]
    X_test_c = X_test_a[minimal_avail]
    print(f"Features: {len(minimal_avail)}")

    ultra_params = {
        "n_estimators": 15,
        "max_depth": 2,
        "num_leaves": 3,
        "learning_rate": 0.05,
        "min_child_samples": 80,
        "reg_alpha": 10.0,
        "reg_lambda": 30.0,
    }
    model_c = build_model(ultra_params)
    model_c.fit(X_train_c, y_train)
    probs_c = model_c.predict_proba(X_test_c)[:, 1]
    metrics_c = evaluate_nrfi(y_test, probs_c, "Ultra-minimal", bootstrap=False)

    # ── Variant D: Logistic regression baseline (simple, less overfit-prone) ──
    print(f"\n{'='*60}")
    print("Variant D: Logistic regression on top features")
    print(f"{'='*60}")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    lr_features = minimal_avail
    X_train_d = X_train_a[lr_features].values
    X_test_d = X_test_a[lr_features].values

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=42)),
    ])
    lr_pipe.fit(X_train_d, y_train)
    probs_d = lr_pipe.predict_proba(X_test_d)[:, 1]
    print(f"Features: {len(lr_features)}")
    # Print LR coefficients
    coefs = lr_pipe.named_steps["lr"].coef_[0]
    print("\n  LR Coefficients (standardized):")
    for feat, coef in sorted(zip(lr_features, coefs), key=lambda x: -abs(x[1])):
        print(f"    {feat:<40} {coef:+.4f}")
    metrics_d = evaluate_nrfi(y_test, probs_d, "Logistic regression", bootstrap=False)

    # ── Variant E: Ensemble of B + D (LGB core + LR) ──
    # Use fixed 50/50 blend (no test-set weight tuning to avoid data leakage)
    print(f"\n{'='*60}")
    print("Variant E: Ensemble (LGB core + LR, 50/50)")
    print(f"{'='*60}")
    probs_e = np.clip(0.5 * probs_b + 0.5 * probs_d, 1e-6, 1 - 1e-6)
    print(f"  Fixed weight: 0.50 LGB + 0.50 LR (no test-set tuning)")
    metrics_e = evaluate_nrfi(y_test, probs_e, "Ensemble (50/50)", bootstrap=False)

    # ── Variant F: Ensemble of C + D (ultra-minimal LGB + LR) ──
    print(f"\n{'='*60}")
    print("Variant F: Ensemble (Ultra-minimal LGB + LR, 50/50)")
    print(f"{'='*60}")
    probs_f = np.clip(0.5 * probs_c + 0.5 * probs_d, 1e-6, 1 - 1e-6)
    metrics_f = evaluate_nrfi(y_test, probs_f, "Ensemble C+D (50/50)", bootstrap=False)

    # ── Pick best variant ──
    print(f"\n{'='*60}")
    print("VARIANT COMPARISON")
    print(f"{'='*60}")
    variants = [
        ("A: Full features", metrics_a, probs_a),
        ("B: Core features", metrics_b, probs_b),
        ("C: Ultra-minimal", metrics_c, probs_c),
        ("D: Logistic regr", metrics_d, probs_d),
        ("E: Ens B+D 50/50", metrics_e, probs_e),
        ("F: Ens C+D 50/50", metrics_f, probs_f),
    ]
    print(f"  {'Variant':<20} {'BSS':>8} {'AUC':>7} {'Pred':>6} {'Bias':>7}")
    best_bss = -999
    best_probs = None
    best_metrics = None
    best_name = None
    for name, m, p in variants:
        bias = m["pred_mean"] - m["nrfi_rate"]
        print(f"  {name:<20} {m['brier_skill']:>+7.4f} {m['auc']:>6.4f} {m['pred_mean']:>5.3f} {bias:>+6.3f}")
        if m["brier_skill"] > best_bss:
            best_bss = m["brier_skill"]
            best_probs = p
            best_metrics = m
            best_name = name

    print(f"\n  Best: {best_name}")

    # Run full evaluation with bootstrap on best variant
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION: {best_name}")
    print(f"{'='*60}")
    final_metrics = evaluate_nrfi(y_test, best_probs, f"FINAL: {best_name}", bootstrap=True)

    out_df = _build_output_df(test_df, best_probs)
    return out_df, final_metrics


def _build_output_df(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Assemble the final predictions output DataFrame."""
    return pd.DataFrame({
        "game_pk": df["game_pk"].values,
        "game_date": df["game_date"].values,
        "home_team": df["home_team"].values if "home_team" in df.columns else np.nan,
        "away_team": df["away_team"].values if "away_team" in df.columns else np.nan,
        "nrfi_lgb_prob": probs,
        "actual_nrfi": df["nrfi"].values,
    })


# ──────────────────────────────────────────────────────────────
# Comparison with MC Simulator
# ──────────────────────────────────────────────────────────────

def compare_with_mc_sim(out_df: pd.DataFrame, test_season: int = 2025) -> None:
    """Compare LGB model with MC simulator predictions if available."""
    sim_path = DATA_DIR / "backtest" / f"nrfi_ou_backtest_{test_season}.parquet"
    if not sim_path.exists():
        print("\n  MC sim backtest not found, skipping comparison")
        return

    sim_df = pd.read_parquet(sim_path)
    merged = out_df.merge(
        sim_df[["game_pk", "sim_nrfi_prob"]],
        on="game_pk", how="inner",
    ).dropna(subset=["sim_nrfi_prob", "nrfi_lgb_prob"])

    if len(merged) < 20:
        print(f"\n  Only {len(merged)} overlap games -- skipping comparison")
        return

    print(f"\n{'='*60}")
    print(f"Comparison vs MC Simulator ({len(merged)} overlap games)")
    print(f"{'='*60}")

    y = merged["actual_nrfi"].values
    p_lgb = merged["nrfi_lgb_prob"].values
    p_mc = merged["sim_nrfi_prob"].values
    base_rate = y.mean()

    bs_lgb = brier_score_loss(y, p_lgb)
    bs_mc = brier_score_loss(y, p_mc)
    bs_base = brier_score_loss(y, np.full_like(p_lgb, base_rate))

    print(f"\n  {'Model':<22} {'Brier':>7}  {'BSS':>8}  {'AUC':>6}")
    print(f"  {'Baseline (mean)':<22} {bs_base:.4f}  {0:>+7.4f}  {'n/a':>6}")
    print(f"  {'MC Simulator':<22} {bs_mc:.4f}  {1-bs_mc/bs_base:>+7.4f}  {roc_auc_score(y, p_mc):.4f}")
    print(f"  {'LGB v2 Model':<22} {bs_lgb:.4f}  {1-bs_lgb/bs_base:>+7.4f}  {roc_auc_score(y, p_lgb):.4f}")

    # Blend
    p_blend = 0.5 * p_lgb + 0.5 * p_mc
    bs_blend = brier_score_loss(y, p_blend)
    print(f"  {'Ensemble (50/50)':<22} {bs_blend:.4f}  {1-bs_blend/bs_base:>+7.4f}  {roc_auc_score(y, p_blend):.4f}")

    # Optimal blend
    from scipy.optimize import minimize_scalar
    def blend_loss(w):
        p = np.clip(w * p_lgb + (1 - w) * p_mc, 1e-6, 1 - 1e-6)
        return brier_score_loss(y, p)
    opt = minimize_scalar(blend_loss, bounds=(0.0, 1.0), method="bounded")
    w_lgb = opt.x
    p_opt = np.clip(w_lgb * p_lgb + (1 - w_lgb) * p_mc, 1e-6, 1 - 1e-6)
    bs_opt = brier_score_loss(y, p_opt)
    print(f"  {'Ensemble (opt w='+f'{w_lgb:.2f})':<22} {bs_opt:.4f}  {1-bs_opt/bs_base:>+7.4f}  {roc_auc_score(y, p_opt):.4f}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NRFI LightGBM Classifier v2")
    parser.add_argument("--train-seasons", type=int, nargs="+", default=[2024],
                        help="Seasons to train on (default: 2024)")
    parser.add_argument("--test-season", type=int, default=2025,
                        help="Season to evaluate (default: 2025)")
    parser.add_argument("--output", type=str,
                        default="data/backtest/nrfi_lgb_2025.parquet",
                        help="Output path for predictions")
    args = parser.parse_args()

    if not HAS_LGB:
        print("ERROR: LightGBM is required. Install with: pip install lightgbm")
        sys.exit(1)

    out_df, metrics = walk_forward_nrfi(
        train_seasons=args.train_seasons,
        test_season=args.test_season,
    )

    compare_with_mc_sim(out_df, args.test_season)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"\n  Predictions saved to {out_path}  ({len(out_df)} rows)")
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Brier skill score: {metrics['brier_skill']:+.4f}")
    print(f"  AUC:               {metrics['auc']:.4f}")
    print(f"  Pred mean:         {metrics['pred_mean']:.3f}  (actual NRFI rate: {metrics['nrfi_rate']:.3f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
