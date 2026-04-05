#!/usr/bin/env python3
"""
Hitter evaluation models.

H1: Swing Decision — scores each batter's pitch selection quality.
H2: Contact Rates — P(contact|swing) split by in-zone vs chase.
H3: Foul Fighting — rewards batters who fight off quality 2-strike pitches.
H4: BIP Quality — expected run value on balls in play.

All models use the statcast xRV parquet data. Per-batter scores are extracted
as residuals from population-level models.

Usage:
    python src/hitter_eval.py --season 2024
    python src/hitter_eval.py --season 2025
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import MODEL_DIR, XRV_DIR

# Swing descriptions
SWING_DESCRIPTIONS = {
    "foul", "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
    "hit_into_play_no_out", "hit_into_play_score",
}

# Take descriptions
TAKE_DESCRIPTIONS = {
    "called_strike", "ball", "blocked_ball", "automatic_ball", "automatic_strike",
    "hit_by_pitch", "pitchout",
}

# Contact on swing
CONTACT_DESCRIPTIONS = {
    "foul", "foul_tip", "foul_bunt", "bunt_foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}

# Whiff on swing
WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
}


def _load_xrv(season: int) -> pd.DataFrame:
    """Load xRV parquet for a season."""
    path = XRV_DIR / f"statcast_xrv_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No xRV data for {season}: {path}")
    df = pd.read_parquet(path)
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    return df


def _classify_swing(description: str) -> str | None:
    """Classify a pitch as swing, take, or None (unknown)."""
    if description in SWING_DESCRIPTIONS:
        return "swing"
    elif description in TAKE_DESCRIPTIONS:
        return "take"
    return None


# ─── H1: Swing Decision ─────────────────────────────────────────────

def compute_swing_decision_scores(xrv: pd.DataFrame) -> dict:
    """Score each batter's pitch selection quality.

    For each (pitch_type, zone, count) bucket, compute xRV|swing vs xRV|take.
    A batter should swing when xRV|swing > xRV|take (positive xRV = good for batter).
    Penalty for wrong decisions scales with |delta|.

    Returns {batter_id: {"score": float, "n_pitches": int, "pct_correct": float}}
    """
    df = xrv.dropna(subset=["xrv", "zone", "description"]).copy()
    df["action"] = df["description"].map(_classify_swing)
    df = df.dropna(subset=["action"])
    df["cnt"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)

    # Build reference table: mean xRV for swing vs take per bucket
    group_cols = ["pitch_type", "zone", "cnt"]
    ref = df.groupby(group_cols + ["action"])["xrv"].agg(["mean", "count"]).reset_index()
    ref.columns = group_cols + ["action", "xrv_mean", "n"]

    # Pivot to get xRV_swing and xRV_take side by side
    swing_ref = ref[ref["action"] == "swing"].set_index(group_cols)["xrv_mean"].rename("xrv_swing")
    take_ref = ref[ref["action"] == "take"].set_index(group_cols)["xrv_mean"].rename("xrv_take")
    lookup = pd.concat([swing_ref, take_ref], axis=1).dropna()

    # Require minimum observations in both swing and take
    swing_n = ref[ref["action"] == "swing"].set_index(group_cols)["n"]
    take_n = ref[ref["action"] == "take"].set_index(group_cols)["n"]
    min_n = pd.concat([swing_n.rename("sn"), take_n.rename("tn")], axis=1).dropna()
    valid_buckets = min_n[(min_n["sn"] >= 20) & (min_n["tn"] >= 20)].index
    lookup = lookup.loc[lookup.index.isin(valid_buckets)]

    lookup["delta"] = lookup["xrv_swing"] - lookup["xrv_take"]

    # Score each batter — vectorized
    df = df.join(lookup["delta"], on=["pitch_type", "zone", "cnt"])
    df = df.dropna(subset=["delta"])

    # Penalty: swung when delta < 0 OR took when delta > 0
    wrong_swing = (df["action"] == "swing") & (df["delta"] < 0)
    wrong_take = (df["action"] == "take") & (df["delta"] > 0)
    df["penalty"] = 0.0
    df.loc[wrong_swing | wrong_take, "penalty"] = df.loc[wrong_swing | wrong_take, "delta"].abs()
    df["correct"] = (~wrong_swing & ~wrong_take).astype(int)

    # Aggregate per batter
    agg = df.groupby("batter").agg(
        total_penalty=("penalty", "sum"),
        n_correct=("correct", "sum"),
        n_pitches=("penalty", "count"),
    )
    agg = agg[agg["n_pitches"] >= 200]
    agg["mean_penalty"] = agg["total_penalty"] / agg["n_pitches"]
    agg["pct_correct"] = agg["n_correct"] / agg["n_pitches"]

    scores = {}
    for batter, row in agg.iterrows():
        scores[int(batter)] = {
            "score": -float(row["mean_penalty"]),
            "n_pitches": int(row["n_pitches"]),
            "pct_correct": float(row["pct_correct"]),
        }

    # Z-score across batters
    if scores:
        raw_scores = np.array([v["score"] for v in scores.values()])
        mean_s = raw_scores.mean()
        std_s = raw_scores.std()
        if std_s > 0:
            for batter in scores:
                scores[batter]["z_score"] = float((scores[batter]["score"] - mean_s) / std_s)
        else:
            for batter in scores:
                scores[batter]["z_score"] = 0.0

    return scores


# ─── H2: Contact Rates ──────────────────────────────────────────────

def compute_contact_rate_scores(xrv: pd.DataFrame) -> dict:
    """Compute per-batter contact skills split by in-zone vs chase.

    Returns {batter_id: {"iz_contact_skill": float, "chase_contact_skill": float,
                          "iz_n": int, "chase_n": int}}
    """
    # Filter to swing pitches only
    df = xrv.dropna(subset=["description", "zone"]).copy()
    df["action"] = df["description"].map(_classify_swing)
    df = df[df["action"] == "swing"]

    # Contact = 1 if foul/BIP, 0 if whiff
    df["contact"] = df["description"].isin(CONTACT_DESCRIPTIONS).astype(int)

    # Zone split
    df["is_iz"] = df["zone"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])
    df["is_chase"] = df["zone"].isin([11, 12, 13, 14])

    # Population contact rates by (pitch_type, zone, count) — vectorized
    df["cnt"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    pop_rates_df = df.groupby(["pitch_type", "zone", "cnt"])["contact"].mean().rename("expected_contact")
    df = df.join(pop_rates_df, on=["pitch_type", "zone", "cnt"])
    df["expected_contact"] = df["expected_contact"].fillna(0.65)  # fallback
    df["residual"] = df["contact"] - df["expected_contact"]

    scores = {}

    # In-zone
    iz = df[df["is_iz"]]
    iz_agg = iz.groupby("batter").agg(
        iz_skill=("residual", "mean"),
        iz_n=("residual", "count"),
    )
    iz_agg = iz_agg[iz_agg["iz_n"] >= 100]

    # Chase
    chase = df[df["is_chase"]]
    chase_agg = chase.groupby("batter").agg(
        chase_skill=("residual", "mean"),
        chase_n=("residual", "count"),
    )
    chase_agg = chase_agg[chase_agg["chase_n"] >= 50]

    # Combine
    all_batters = set(iz_agg.index) | set(chase_agg.index)
    for batter in all_batters:
        batter = int(batter)
        iz_row = iz_agg.loc[batter] if batter in iz_agg.index else None
        chase_row = chase_agg.loc[batter] if batter in chase_agg.index else None
        scores[batter] = {
            "iz_contact_skill": float(iz_row["iz_skill"]) if iz_row is not None else None,
            "chase_contact_skill": float(chase_row["chase_skill"]) if chase_row is not None else None,
            "iz_n": int(iz_row["iz_n"]) if iz_row is not None else 0,
            "chase_n": int(chase_row["chase_n"]) if chase_row is not None else 0,
        }

    return scores


# ─── H3: Foul Fighting ──────────────────────────────────────────────

def compute_foul_fighting_scores(xrv: pd.DataFrame) -> dict:
    """Score batters on ability to fight off quality 2-strike pitches.

    Returns {batter_id: {"foul_fight_rate": float, "foul_fight_score": float, "n": int}}
    """
    # 2-strike, in-zone pitches
    df = xrv[
        (xrv["strikes"] == 2)
        & (xrv["zone"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        & (xrv["description"].notna())
        & (xrv["xrv"].notna())
    ].copy()

    df["is_foul"] = df["description"].isin({"foul", "foul_tip"}).astype(int)

    scores = {}
    for batter, bdf in df.groupby("batter"):
        batter = int(batter)
        if len(bdf) < 80:
            continue

        foul_fight_rate = float(bdf["is_foul"].mean())
        foul_pitches = bdf[bdf["is_foul"] == 1]

        if len(foul_pitches) >= 10:
            # Lower xRV on fouled pitches = fighting off pitches that were
            # favorable for the pitcher (bad for batter)
            mean_xrv_fouls = float(foul_pitches["xrv"].mean())
        else:
            mean_xrv_fouls = 0.0

        scores[batter] = {
            "foul_fight_rate": foul_fight_rate,
            "mean_xrv_fouls": mean_xrv_fouls,
            "n": len(bdf),
        }

    # Composite: rate * (-mean_xrv_fouls_z)
    # Low xRV fouls = fighting off good pitcher pitches = good
    if scores:
        rates = np.array([v["foul_fight_rate"] for v in scores.values()])
        xrvs = np.array([v["mean_xrv_fouls"] for v in scores.values()])
        xrv_z = (xrvs - xrvs.mean()) / (xrvs.std() + 1e-9)
        for i, batter in enumerate(scores):
            # Negative xrv_z = fighting off pitcher-favorable pitches = good
            scores[batter]["foul_fight_score"] = float(rates[i] * (-xrv_z[i]))

    return scores


# ─── H4: Ball-in-Play Quality ───────────────────────────────────────

def compute_bip_quality_scores(xrv: pd.DataFrame) -> dict:
    """Score batters on quality of their batted ball outcomes.

    Returns {batter_id: {"bip_quality_iz": float, "bip_quality_chase": float,
                          "bip_n": int}}
    """
    df = xrv[
        (xrv["type"] == "X")
        & (xrv["xrv"].notna())
        & (xrv["zone"].notna())
    ].copy()

    scores = {}
    for batter, bdf in df.groupby("batter"):
        batter = int(batter)

        iz = bdf[bdf["zone"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])]
        chase = bdf[bdf["zone"].isin([11, 12, 13, 14])]

        if len(iz) < 50:
            continue

        bip_iz = float(iz["xrv"].mean())
        bip_chase = float(chase["xrv"].mean()) if len(chase) >= 20 else None

        scores[batter] = {
            "bip_quality_iz": bip_iz,
            "bip_quality_chase": bip_chase,
            "bip_n": len(bdf),
        }

    return scores


# ─── Composite ───────────────────────────────────────────────────────

def build_hitter_profiles(season: int) -> dict:
    """Build complete hitter profiles for a season.

    Returns dict with keys:
        swing_decision: {batter_id: {...}}
        contact_rates: {batter_id: {...}}
        foul_fighting: {batter_id: {...}}
        bip_quality: {batter_id: {...}}
        composite: {batter_id: {"embedding": [5 dims], ...}}
    """
    # Load current + prior season
    frames = []
    for yr in [season - 1, season]:
        try:
            frames.append(_load_xrv(yr))
        except FileNotFoundError:
            pass
    if not frames:
        raise FileNotFoundError(f"No xRV data for {season}")
    xrv = pd.concat(frames, ignore_index=True)

    # Only use current season for scoring (prior season for population baselines)
    current = xrv[xrv["season"] == season] if "season" in xrv.columns else xrv

    print(f"\n{'='*60}")
    print(f"  Hitter Evaluation — {season}")
    print(f"{'='*60}")
    print(f"  Total pitches: {len(xrv):,} (current season: {len(current):,})")

    print("\n  H1: Swing Decision...")
    swing_scores = compute_swing_decision_scores(current)
    print(f"    Scored {len(swing_scores)} batters")
    if swing_scores:
        zs = [v["z_score"] for v in swing_scores.values() if "z_score" in v]
        pcts = [v["pct_correct"] for v in swing_scores.values()]
        print(f"    Correct decision rate: {np.mean(pcts):.1%} avg")
        print(f"    Z-score range: [{min(zs):.2f}, {max(zs):.2f}]")

    print("\n  H2: Contact Rates...")
    contact_scores = compute_contact_rate_scores(current)
    print(f"    Scored {len(contact_scores)} batters")
    iz_skills = [v["iz_contact_skill"] for v in contact_scores.values() if v["iz_contact_skill"] is not None]
    if iz_skills:
        print(f"    IZ contact skill range: [{min(iz_skills):.3f}, {max(iz_skills):.3f}]")

    print("\n  H3: Foul Fighting...")
    foul_scores = compute_foul_fighting_scores(current)
    print(f"    Scored {len(foul_scores)} batters")
    if foul_scores:
        rates = [v["foul_fight_rate"] for v in foul_scores.values()]
        print(f"    Foul fight rate: {np.mean(rates):.1%} avg")

    print("\n  H4: BIP Quality...")
    bip_scores = compute_bip_quality_scores(current)
    print(f"    Scored {len(bip_scores)} batters")
    iz_quals = [v["bip_quality_iz"] for v in bip_scores.values()]
    if iz_quals:
        print(f"    BIP quality (IZ): [{min(iz_quals):.4f}, {max(iz_quals):.4f}]")

    # Build 5D embeddings for batters with all scores
    print("\n  Building composite embeddings...")
    composite = {}
    all_batters = set(swing_scores) & set(contact_scores) & set(bip_scores)
    for batter in all_batters:
        sw = swing_scores[batter]
        ct = contact_scores[batter]
        ff = foul_scores.get(batter, {})
        bp = bip_scores[batter]

        embedding = [
            sw.get("z_score", 0.0),
            ct.get("iz_contact_skill", 0.0) or 0.0,
            ct.get("chase_contact_skill", 0.0) or 0.0,
            ff.get("foul_fight_score", 0.0),
            bp.get("bip_quality_iz", 0.0),
        ]
        composite[batter] = {"embedding": embedding}

    print(f"    Complete embeddings: {len(composite)} batters")

    return {
        "swing_decision": swing_scores,
        "contact_rates": contact_scores,
        "foul_fighting": foul_scores,
        "bip_quality": bip_scores,
        "composite": composite,
        "season": season,
    }


def save_hitter_profiles(profiles: dict):
    """Save hitter profiles."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    season = profiles["season"]
    path = MODEL_DIR / f"hitter_eval_{season}.pkl"
    with open(path, "wb") as f:
        pickle.dump(profiles, f)
    print(f"\n  Saved to {path}")


def load_hitter_profiles(season: int) -> dict:
    """Load hitter profiles."""
    path = MODEL_DIR / f"hitter_eval_{season}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Hitter Evaluation Models")
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    profiles = build_hitter_profiles(args.season)
    save_hitter_profiles(profiles)


if __name__ == "__main__":
    main()
