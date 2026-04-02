#!/usr/bin/env python3
"""
Build simulation lookup tables from historical Statcast data.

Produces three artifacts used by the Monte Carlo game simulator:
  1. League-average PA outcome base rates
  2. xRV-to-outcome calibration (log-linear sensitivity coefficients)
  3. Base-running transition matrix

Usage:
    python src/build_transition_matrix.py --seasons 2021 2022 2023 2024
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import DATA_DIR, XRV_DIR

SIM_DIR = DATA_DIR / "sim"

# ── Outcome mapping ────────────────────────────────────────
# Map Statcast `events` to simulation outcome categories.
EVENT_MAP = {
    "strikeout": "K",
    "strikeout_double_play": "K",
    "walk": "BB",
    "hit_by_pitch": "HBP",
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "grounded_into_double_play": "dp",
    "double_play": "dp",
    # field_out handled separately using bb_type
    "sac_fly": "out_fly",
    "sac_fly_double_play": "out_fly",
    "sac_bunt": "out_ground",
    "sac_bunt_double_play": "out_ground",
    "force_out": "out_ground",
    "fielders_choice": "out_ground",
    "fielders_choice_out": "out_ground",
    "field_error": "1B",  # runner reaches; treat like single for base running
    "catcher_interf": "1B",
}

BBTYPE_MAP = {
    "ground_ball": "out_ground",
    "fly_ball": "out_fly",
    "popup": "out_fly",
    "line_drive": "out_line",
}

OUTCOME_ORDER = ["K", "BB", "HBP", "1B", "2B", "3B", "HR",
                 "dp", "out_ground", "out_fly", "out_line"]

STATCAST_DIR = DATA_DIR / "statcast"


def _classify_events_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized classification of Statcast events to sim outcome categories."""
    outcome = df["events"].map(EVENT_MAP)

    field_out_mask = df["events"] == "field_out"
    if field_out_mask.any():
        bb_mapped = df.loc[field_out_mask, "bb_type"].map(BBTYPE_MAP).fillna("out_ground")
        outcome.loc[field_out_mask] = bb_mapped

    still_null = outcome.isna() & df["events"].notna()
    if still_null.any():
        has_dp = df.loc[still_null, "events"].str.contains("double_play", na=False)
        outcome.loc[still_null & has_dp] = "dp"
        has_out = df.loc[still_null & ~has_dp, "events"].str.contains("out", na=False)
        outcome.loc[still_null & ~has_dp & has_out] = "out_ground"

    return outcome


def _classify_event(row) -> str | None:
    """Map a Statcast row to a simulation outcome category."""
    event = row.get("events")
    if pd.isna(event) or event == "":
        return None

    if event in EVENT_MAP:
        return EVENT_MAP[event]

    if event == "field_out":
        bb = row.get("bb_type", "")
        return BBTYPE_MAP.get(bb, "out_ground")

    # Catch-all for rare events (e.g. "triple_play", "runner_double_play")
    if "double_play" in str(event):
        return "dp"
    if "out" in str(event):
        return "out_ground"

    return None


def _base_state(row) -> tuple[bool, bool, bool]:
    """Extract (1B, 2B, 3B) occupancy from a Statcast row.

    A base is occupied if the column exists and has a non-null player ID.
    """
    def _occupied(col: str) -> bool:
        val = row.get(col) if hasattr(row, "get") else row[col] if col in row.index else None
        if val is None:
            return False
        return not pd.isna(val)

    return (_occupied("on_1b"), _occupied("on_2b"), _occupied("on_3b"))


# ── Step 1: League-average base rates ─────────────────────

def compute_base_rates(df: pd.DataFrame) -> dict[str, float]:
    """Compute league-average PA outcome frequencies."""
    pa = df[df["events"].notna()].copy()
    pa["outcome"] = _classify_events_vectorized(pa)
    pa = pa[pa["outcome"].notna()]

    counts = pa["outcome"].value_counts()
    total = counts.sum()
    rates = {o: counts.get(o, 0) / total for o in OUTCOME_ORDER}

    print(f"  Total PAs: {total:,}")
    for o in OUTCOME_ORDER:
        print(f"    {o:>10}: {rates[o]:.4f}")
    return rates


# ── Step 2: xRV-to-outcome calibration ────────────────────

def compute_xrv_calibration(df: pd.DataFrame, base_rates: dict) -> dict:
    """
    Fit log-linear sensitivity coefficients mapping matchup xRV
    to PA outcome probability shifts.

    P(outcome | xrv) = base_rate * exp(alpha * xrv * scale) / Z
    """
    # Get PA-level mean xRV (average across all pitches in the at-bat)
    pa = df[df["events"].notna()].copy()
    pa["outcome"] = _classify_events_vectorized(pa)
    pa = pa[pa["outcome"].notna()]

    pa_xrv = pa.groupby(["game_pk", "at_bat_number"])["xrv"].mean().reset_index()
    pa_xrv.columns = ["game_pk", "at_bat_number", "pa_xrv"]
    pa = pa.merge(pa_xrv, on=["game_pk", "at_bat_number"], how="left")
    pa = pa.dropna(subset=["pa_xrv"])

    # Bin into deciles
    pa["xrv_bin"] = pd.qcut(pa["pa_xrv"], q=10, labels=False, duplicates="drop")

    # Compute outcome rates per bin
    bin_rates = {}
    bin_xrv_means = {}
    for b, grp in pa.groupby("xrv_bin"):
        counts = grp["outcome"].value_counts()
        total = counts.sum()
        bin_rates[b] = {o: counts.get(o, 0) / total for o in OUTCOME_ORDER}
        bin_xrv_means[b] = grp["pa_xrv"].mean()

    # Fit alphas via least squares across bins
    # For each outcome: minimize sum over bins of
    #   (observed_rate - base_rate * exp(alpha * xrv_mean) / Z)^2
    bins = sorted(bin_rates.keys())
    xrv_vals = np.array([bin_xrv_means[b] for b in bins])

    # Scale factor: xRV is per-pitch (~±0.005 range) but outcomes happen per-PA.
    # With ~3.9 pitches/PA and wanting ~±20% shifts at ±1σ, scale=8.0 gives
    # exp(8 * 0.005 * 3.9) ≈ 1.17, a reasonable ~17% multiplicative shift.
    scale = 8.0

    alphas = {}
    for outcome in OUTCOME_ORDER:
        observed = np.array([bin_rates[b][outcome] for b in bins])
        br = base_rates[outcome]
        if br < 1e-6:
            alphas[outcome] = 0.0
            continue

        def loss(a):
            raw = br * np.exp(a * xrv_vals * scale)
            # Normalize across outcomes isn't trivial per-outcome,
            # so fit each independently and renormalize at prediction time
            return np.sum((raw - observed) ** 2)

        result = minimize(loss, x0=0.0, method="Nelder-Mead")
        alphas[outcome] = float(result.x[0])

    print("\n  Fitted calibration alphas:")
    for o in OUTCOME_ORDER:
        print(f"    {o:>10}: {alphas[o]:+.4f}")

    return {"alphas": alphas, "scale": scale}


# ── Step 3: Base-running transition matrix ─────────────────

# Deterministic fallback rules for base running
def _deterministic_transition(
    outcome: str, bases: tuple[bool, bool, bool], outs: int
) -> list[tuple[tuple[bool, bool, bool], int, float]]:
    """
    Return list of (new_bases, runs_scored, probability) for deterministic rules.
    """
    b1, b2, b3 = bases
    runners = int(b1) + int(b2) + int(b3)

    if outcome == "HR":
        runs = 1 + runners
        return [((False, False, False), runs, 1.0)]

    if outcome == "3B":
        runs = runners
        return [((False, False, True), runs, 1.0)]

    if outcome == "2B":
        runs = int(b3) + int(b2)
        # Runner from 1B: 60% scores, 40% to 3B
        if b1:
            return [
                ((False, True, True), runs, 0.4),
                ((False, True, False), runs + 1, 0.6),
            ]
        return [((False, True, False), runs, 1.0)]

    if outcome == "1B":
        runs = int(b3)
        new_b3 = False
        # Runner from 2B: 60% scores, 40% to 3B
        if b2:
            r2_scores_prob = 0.6
            branch_a = (True, False, True)   # R2 holds at 3B
            branch_b_runs = runs + 1         # R2 scores
            # Runner from 1B: 70% to 2B, 30% to 3B
            if b1:
                return [
                    ((True, True, True), runs, 0.4 * 0.7),
                    ((True, False, True), runs, 0.4 * 0.3),   # R1 to 3B, R2 holds 3B — can't both be there
                    ((True, True, False), runs + 1, 0.6 * 0.7),
                    ((True, False, True), runs + 1, 0.6 * 0.3),
                ]
            return [
                ((True, False, True), runs, 0.4),
                ((True, False, False), runs + 1, 0.6),
            ]
        if b1:
            return [((True, True, False), runs, 0.7),
                    ((True, False, True), runs, 0.3)]
        return [((True, False, False), runs, 1.0)]

    if outcome in ("BB", "HBP"):
        runs = 0
        if b1 and b2 and b3:
            runs = 1  # bases loaded walk
        new_b1 = True
        new_b2 = b2 or b1
        new_b3 = b3 or (b1 and b2)
        return [((new_b1, new_b2, new_b3), runs, 1.0)]

    if outcome == "K":
        return [(bases, 0, 1.0)]  # outs handled by caller

    if outcome == "out_fly":
        runs = 0
        new_bases = bases
        if b3 and outs < 2:
            runs = 1
            new_bases = (b1, b2, False)
        return [(new_bases, runs, 1.0)]

    if outcome == "out_ground":
        runs = 0
        if outs < 2:
            # Runners advance one base
            new_b3 = b2
            new_b2 = b1
            new_b1 = False
            if b3:
                runs = 1
            return [((new_b1, new_b2, new_b3), runs, 1.0)]
        return [(bases, 0, 1.0)]

    if outcome == "out_line":
        return [(bases, 0, 1.0)]

    if outcome == "dp":
        # Lead runner out, batter out
        if b1:
            new_b1 = False
            new_b2 = b2  # other runners advance one
            new_b3 = b2 if not b3 else b3
            runs = 1 if b3 else 0
            return [((new_b1, new_b2, new_b3), runs, 1.0)]
        # If no runner on 1B somehow, just a ground out
        return [(bases, 0, 1.0)]

    return [(bases, 0, 1.0)]


def compute_transition_matrix(df: pd.DataFrame, min_count: int = 20) -> dict:
    """
    Build empirical base-running transition matrix from Statcast data.

    For each (outcome, base_state, outs), compute probability distribution
    over (new_base_state, runs_scored).

    Falls back to deterministic rules for sparse combinations.

    Requires `post_home_score` and `post_away_score` columns (or `home_score`
    and `away_score`) to compute runs scored per PA.  If neither pair is
    present the function uses deterministic rules for all transitions.
    """
    # Get PA-level rows
    pa = df[df["events"].notna()].copy()
    pa["outcome"] = _classify_events_vectorized(pa)
    pa = pa[pa["outcome"].notna()]
    pa["bases"] = pa.apply(_base_state, axis=1)

    # For post-PA state, we need to look at the next PA in the same half-inning.
    # Group by game + half-inning, sort by at_bat_number.
    pa = pa.sort_values(["game_pk", "inning", "inning_topbot", "at_bat_number"])

    # Determine best way to compute runs scored per PA.
    # Preferred: post_home_score / post_away_score vs home_score / away_score
    # (post_* gives us runs scored in THIS PA directly)
    has_post_scores = (
        "post_home_score" in pa.columns and "post_away_score" in pa.columns
        and "home_score" in pa.columns and "away_score" in pa.columns
    )
    has_scores = "home_score" in pa.columns and "away_score" in pa.columns
    if has_post_scores:
        print("  Using post_home_score/post_away_score for run inference")
    elif has_scores:
        print("  Using home_score/away_score delta for run inference")
    else:
        print("  WARNING: No score columns found — using deterministic transitions only")

    transitions = {}  # (outcome, bases, outs) -> list of (new_bases, runs, count)

    for hi_key, grp in pa.groupby(["game_pk", "inning", "inning_topbot"]):
        grp = grp.sort_values("at_bat_number").reset_index(drop=True)

        for i in range(len(grp)):
            row = grp.iloc[i]
            outcome = row["outcome"]
            bases = row["bases"]
            outs = int(row["outs_when_up"]) if pd.notna(row.get("outs_when_up")) else 0

            key = (outcome, bases, outs)

            if i + 1 < len(grp):
                next_row = grp.iloc[i + 1]
                new_bases = _base_state(next_row)

                # Infer runs scored from score columns
                if has_post_scores:
                    # post_*_score is the score AFTER this PA resolves
                    is_top = row["inning_topbot"] == "Top"
                    if is_top:
                        pre = row.get("away_score")
                        post = row.get("post_away_score")
                    else:
                        pre = row.get("home_score")
                        post = row.get("post_home_score")
                    if pd.notna(pre) and pd.notna(post):
                        runs = int(post) - int(pre)
                    else:
                        runs = _deterministic_transition(outcome, bases, outs)[0][1]
                elif has_scores:
                    is_top = row["inning_topbot"] == "Top"
                    if is_top:
                        runs = int(next_row["away_score"]) - int(row["away_score"]) if (
                            pd.notna(next_row.get("away_score")) and pd.notna(row.get("away_score"))
                        ) else 0
                    else:
                        runs = int(next_row["home_score"]) - int(row["home_score"]) if (
                            pd.notna(next_row.get("home_score")) and pd.notna(row.get("home_score"))
                        ) else 0
                else:
                    runs = _deterministic_transition(outcome, bases, outs)[0][1]
                runs = max(0, min(runs, 4))  # clamp
            else:
                # Last PA of half-inning — outcome led to 3rd out or inning end
                if outcome == "HR":
                    runners = int(bases[0]) + int(bases[1]) + int(bases[2])
                    new_bases = (False, False, False)
                    runs = 1 + runners
                else:
                    fallback = _deterministic_transition(outcome, bases, outs)
                    new_bases = fallback[0][0]
                    runs = fallback[0][1]

            entry = (new_bases, runs)
            transitions.setdefault(key, {})
            transitions[key][entry] = transitions[key].get(entry, 0) + 1

    # Convert counts to probability distributions
    matrix = {}
    n_empirical = 0
    n_fallback = 0

    for key, entries in transitions.items():
        total = sum(entries.values())
        if total >= min_count:
            dist = [(nb, r, c / total) for (nb, r), c in entries.items()]
            matrix[key] = dist
            n_empirical += 1
        else:
            outcome, bases, outs = key
            matrix[key] = _deterministic_transition(outcome, bases, outs)
            n_fallback += 1

    # Fill in missing combinations with deterministic rules
    all_bases = [(b1, b2, b3) for b1 in [False, True] for b2 in [False, True] for b3 in [False, True]]
    for outcome in OUTCOME_ORDER:
        for bases in all_bases:
            for outs in range(3):
                key = (outcome, bases, outs)
                if key not in matrix:
                    matrix[key] = _deterministic_transition(outcome, bases, outs)
                    n_fallback += 1

    print(f"\n  Transition matrix: {n_empirical} empirical, {n_fallback} fallback")
    return matrix


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build simulation lookup tables")
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2021, 2026)))
    args = parser.parse_args()

    # Load xRV data and merge score columns from raw Statcast
    frames = []
    for year in sorted(args.seasons):
        path = XRV_DIR / f"statcast_xrv_{year}.parquet"
        if not path.exists():
            print(f"  {path} not found, skipping")
            continue
        xrv = pd.read_parquet(path)
        if "game_type" in xrv.columns:
            xrv = xrv[xrv["game_type"] == "R"]

        # Merge score columns from raw Statcast (needed for transition matrix)
        score_cols = ["home_score", "away_score", "post_home_score", "post_away_score"]
        if not all(c in xrv.columns for c in score_cols):
            raw_path = STATCAST_DIR / f"statcast_{year}.parquet"
            if raw_path.exists():
                raw = pd.read_parquet(raw_path, columns=["game_pk", "at_bat_number", "pitch_number"] + score_cols)
                # Drop any score cols already in xrv to avoid _x/_y suffixes
                xrv = xrv.drop(columns=[c for c in score_cols if c in xrv.columns], errors="ignore")
                xrv = xrv.merge(raw[["game_pk", "at_bat_number", "pitch_number"] + score_cols],
                                on=["game_pk", "at_bat_number", "pitch_number"], how="left")
                print(f"  Loaded {year}: {len(xrv):,} pitches (merged score cols from raw)")
            else:
                print(f"  Loaded {year}: {len(xrv):,} pitches (no raw Statcast for scores)")
        else:
            print(f"  Loaded {year}: {len(xrv):,} pitches")

        frames.append(xrv)

    if not frames:
        print("No data found. Run build_xrv.py first.")
        return

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Total: {len(df):,} pitches across {len(frames)} seasons")

    # Step 1: Base rates
    print("\n" + "=" * 60)
    print("Step 1: League-average PA outcome base rates")
    print("=" * 60)
    base_rates = compute_base_rates(df)

    # Step 2: Transition matrix
    print("\n" + "=" * 60)
    print("Step 2: Base-running transition matrix")
    print("=" * 60)
    transition_matrix = compute_transition_matrix(df)

    # Save artifacts
    SIM_DIR.mkdir(parents=True, exist_ok=True)

    with open(SIM_DIR / "league_base_rates.pkl", "wb") as f:
        pickle.dump(base_rates, f)
    print(f"\n  Saved league_base_rates.pkl")

    with open(SIM_DIR / "transition_matrix.pkl", "wb") as f:
        pickle.dump(transition_matrix, f)
    print(f"  Saved transition_matrix.pkl")

    print("\nDone.")


if __name__ == "__main__":
    main()
