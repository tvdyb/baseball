#!/usr/bin/env python3
"""
Ablation study: old matchup model vs arsenal matchup model vs simple stuff grade.

Compares walk-forward AUC/log-loss for:
  1. Base model (no pitcher quality features)
  2. Stuff-only (sp_xrv_mean, sp_xrv_vs_lineup — no hitter-specific info)
  3. Old matchup (sparse pitcher-hitter effects)
  4. Arsenal matchup (hitter response to pitcher arsenal type)
  5. Both matchup models combined
  6. Full model (all features)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_DIR = DATA_DIR / "features"

# ── Feature groups ──────────────────────────────────────

BASE_FEATURES = [
    "diff_hit_xrv_mean",
    "diff_hit_xrv_contact",
    "diff_hit_k_rate",
    "diff_def_xrv_delta",
    "diff_recent_form",
    "park_factor",
]

STUFF_FEATURES = [
    "diff_sp_xrv_mean",
    "diff_sp_k_rate",
    "diff_sp_bb_rate",
    "diff_sp_avg_velo",
    "diff_sp_rest_days",
    "diff_sp_overperf",
    "diff_sp_overperf_recent",
    "diff_bp_xrv_mean",
    "diff_bp_fatigue_score",
    "home_sp_pitch_mix_entropy",
    "away_sp_pitch_mix_entropy",
    "home_sp_n_pitches",
    "away_sp_n_pitches",
]

PLATOON_FEATURES = [
    "diff_sp_xrv_vs_lineup",
    "diff_platoon_pct",
]

# Old matchup model (sparse pitcher-hitter effects)
OLD_MATCHUP_FEATURES = [
    "diff_matchup_xrv_mean",
    "diff_matchup_xrv_sum",
    "home_matchup_n_known",
    "away_matchup_n_known",
    "diff_bp_matchup_xrv_mean",
]

# Arsenal matchup model (hitter response to pitcher arsenal type)
ARSENAL_MATCHUP_FEATURES = [
    "diff_arsenal_matchup_xrv_mean",
    "diff_arsenal_matchup_xrv_sum",
    "home_arsenal_matchup_n_known",
    "away_arsenal_matchup_n_known",
    "diff_bp_arsenal_matchup_xrv_mean",
]

WEATHER_FEATURES = [
    "temperature",
    "wind_speed",
    "is_dome",
    "wind_out",
    "wind_in",
]


def load_features(years):
    frames = []
    for year in years:
        path = FEATURES_DIR / f"game_features_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["season"] = year
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def eval_feature_set(df, feature_list, all_years, label, min_train=2):
    """Walk-forward LR evaluation for a given feature set."""
    results = []
    available = [f for f in feature_list if f in df.columns]

    for test_year in all_years[min_train:]:
        train = df[df["season"] < test_year]
        test = df[df["season"] == test_year]
        if len(train) < 100 or len(test) < 50:
            continue

        X_train = train[available].fillna(0)
        X_test = test[available].fillna(0)
        y_train = train["home_win"].values
        y_test = test["home_win"].values

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_tr, y_train)
        probs = lr.predict_proba(X_te)[:, 1]

        results.append({
            "year": test_year,
            "auc": roc_auc_score(y_test, probs),
            "log_loss": log_loss(y_test, probs),
            "n": len(y_test),
        })

    rdf = pd.DataFrame(results)
    avg_auc = rdf["auc"].mean()
    avg_ll = rdf["log_loss"].mean()
    return rdf, avg_auc, avg_ll


def main():
    all_years = list(range(2018, 2026))
    print("Loading features...")
    df = load_features(all_years)
    print(f"  {len(df)} total games across {df['season'].nunique()} seasons")

    # Show which arsenal features are available
    arsenal_avail = [f for f in ARSENAL_MATCHUP_FEATURES if f in df.columns]
    print(f"  Arsenal features available: {arsenal_avail}")
    if arsenal_avail:
        for f in arsenal_avail:
            non_null = df[f].notna().sum()
            print(f"    {f}: {non_null}/{len(df)} non-null ({non_null/len(df):.1%})")

    STUFF_PLATOON = BASE_FEATURES + STUFF_FEATURES + PLATOON_FEATURES

    configs = [
        ("1. Base (hitting/defense/form)",      BASE_FEATURES),
        ("2. + Stuff grade",                    BASE_FEATURES + STUFF_FEATURES),
        ("3. + Platoon",                        STUFF_PLATOON),
        ("4. + Old matchup",                    STUFF_PLATOON + OLD_MATCHUP_FEATURES),
        ("5. + Arsenal matchup (new)",          STUFF_PLATOON + ARSENAL_MATCHUP_FEATURES),
        ("6. + Old + Arsenal (both)",           STUFF_PLATOON + OLD_MATCHUP_FEATURES + ARSENAL_MATCHUP_FEATURES),
        ("7. Full (+ weather)",                 STUFF_PLATOON + OLD_MATCHUP_FEATURES + ARSENAL_MATCHUP_FEATURES + WEATHER_FEATURES),
    ]

    print(f"\n{'Model':<40} {'Feats':>5} {'Avg AUC':>8} {'Avg LL':>8} {'vs Base':>10} {'vs Stuff':>10}")
    print("-" * 85)

    base_auc = None
    stuff_auc = None
    yearly_results = {}

    for label, features in configs:
        rdf, avg_auc, avg_ll = eval_feature_set(df, features, all_years, label)
        n_feats = len([f for f in features if f in df.columns])

        if base_auc is None:
            base_auc = avg_auc
        if "Stuff grade" in label:
            stuff_auc = avg_auc

        vs_base = f"{avg_auc - base_auc:+.4f}" if base_auc else ""
        vs_stuff = f"{avg_auc - stuff_auc:+.4f}" if stuff_auc else ""

        print(f"{label:<40} {n_feats:>5} {avg_auc:>8.4f} {avg_ll:>8.4f} {vs_base:>10} {vs_stuff:>10}")
        yearly_results[label] = rdf

    # Per-year breakdown: stuff vs old matchup vs arsenal matchup
    print(f"\n{'='*85}")
    print("Per-year: Stuff grade vs Old matchup vs Arsenal matchup")
    print(f"{'='*85}")

    stuff_rdf = yearly_results.get("2. + Stuff grade", pd.DataFrame())
    old_rdf = yearly_results.get("4. + Old matchup", pd.DataFrame())
    arsenal_rdf = yearly_results.get("5. + Arsenal matchup (new)", pd.DataFrame())

    if not stuff_rdf.empty and not arsenal_rdf.empty:
        merged = stuff_rdf.rename(columns={"auc": "auc_stuff", "log_loss": "ll_stuff"})
        if not old_rdf.empty:
            merged = merged.merge(
                old_rdf[["year", "auc", "log_loss"]].rename(columns={"auc": "auc_old", "log_loss": "ll_old"}),
                on="year", how="left"
            )
        merged = merged.merge(
            arsenal_rdf[["year", "auc", "log_loss"]].rename(columns={"auc": "auc_arsenal", "log_loss": "ll_arsenal"}),
            on="year", how="left"
        )

        header = f"{'Year':>6} {'Stuff':>8} {'Old MU':>8} {'Arsenal':>8} {'Old-Stuff':>10} {'Ars-Stuff':>10} {'Ars-Old':>10}"
        print(f"\n{header}")
        print("-" * 70)
        for _, row in merged.iterrows():
            old_d = row.get("auc_old", np.nan) - row["auc_stuff"] if pd.notna(row.get("auc_old")) else np.nan
            ars_d = row["auc_arsenal"] - row["auc_stuff"]
            ars_old = row["auc_arsenal"] - row.get("auc_old", np.nan) if pd.notna(row.get("auc_old")) else np.nan

            old_str = f"{row.get('auc_old', np.nan):>8.4f}" if pd.notna(row.get("auc_old")) else f"{'N/A':>8}"
            old_d_str = f"{old_d:>+10.4f}" if pd.notna(old_d) else f"{'N/A':>10}"
            ars_old_str = f"{ars_old:>+10.4f}" if pd.notna(ars_old) else f"{'N/A':>10}"

            print(f"{int(row['year']):>6} {row['auc_stuff']:>8.4f} {old_str} {row['auc_arsenal']:>8.4f} {old_d_str} {ars_d:>+10.4f} {ars_old_str}")

        avg_stuff = merged["auc_stuff"].mean()
        avg_old = merged["auc_old"].mean() if "auc_old" in merged else np.nan
        avg_arsenal = merged["auc_arsenal"].mean()
        print(f"\n  Avg AUC - Stuff:   {avg_stuff:.4f}")
        if pd.notna(avg_old):
            print(f"  Avg AUC - Old MU:  {avg_old:.4f} ({avg_old - avg_stuff:+.4f} vs stuff)")
        print(f"  Avg AUC - Arsenal: {avg_arsenal:.4f} ({avg_arsenal - avg_stuff:+.4f} vs stuff)")
        if pd.notna(avg_old):
            print(f"  Arsenal vs Old:    {avg_arsenal - avg_old:+.4f}")


if __name__ == "__main__":
    main()
