#!/usr/bin/env python3
"""
Multi-Output Matchup Model

Predicts 11-category PA outcome distributions for each hitter-pitcher matchup
using a hierarchical model in arsenal feature space. Replaces the scalar xRV
matchup model + log-linear calibration with direct outcome prediction.

Inference: PyTorch with L2 regularization (equivalent to hierarchical Bayesian priors).
Models split by batter handedness (vs LHH, vs RHH).

Usage:
    python src/multi_output_matchup_model.py --season 2024
    python src/multi_output_matchup_model.py --season 2024 --epochs 200 --lr 0.005
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arsenal_matchup_model import ARSENAL_FEATURES
from build_transition_matrix import OUTCOME_ORDER, EVENT_MAP, BBTYPE_MAP
from feature_engineering import (
    _compute_pitcher_arsenal_live,
    _standardize_arsenal,
    _get_before,
    filter_competitive,
)
from utils import DATA_DIR, XRV_DIR, MODEL_DIR, HARD_TYPES, BREAK_TYPES, OFFSPEED_TYPES

N_OUTCOMES = len(OUTCOME_ORDER)  # 11
N_ARSENAL = len(ARSENAL_FEATURES)  # 13
N_LOGITS = N_OUTCOMES - 1  # 10 (last outcome is reference)

# Common pitch types (must be consistent between training and prediction)
PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH", "ST", "SV", "SW", "KC", "FS"]
N_PTYPES = len(PITCH_TYPES)
PTYPE_MAP = {pt: i for i, pt in enumerate(PITCH_TYPES)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MultiOutputMatchupModel(nn.Module):
    def __init__(self, n_hitters, n_arsenal=N_ARSENAL, n_ptypes=N_PTYPES, n_logits=N_LOGITS):
        super().__init__()
        self.n_hitters = n_hitters
        self.n_logits = n_logits

        # Population-level parameters
        self.intercept = nn.Parameter(torch.zeros(n_logits))
        self.register_buffer("intercept_prior", torch.zeros(n_logits))
        self.beta_arsenal = nn.Parameter(torch.zeros(n_arsenal, n_logits))
        self.beta_ptmix = nn.Parameter(torch.zeros(n_ptypes, n_logits))

        # Per-hitter parameters (hierarchical -- regularized toward zero)
        self.hitter_base = nn.Parameter(torch.zeros(n_hitters, n_logits))
        self.hitter_arsenal = nn.Parameter(torch.zeros(n_hitters, n_arsenal, n_logits))
        self.hitter_ptype = nn.Parameter(torch.zeros(n_hitters, n_ptypes, n_logits))

    def forward(self, arsenal_z, pitch_mix, hitter_idx):
        """
        Args:
            arsenal_z: (batch, n_arsenal) standardized arsenal features
            pitch_mix: (batch, n_ptypes) pitch-type mix fractions
            hitter_idx: (batch,) integer hitter indices

        Returns:
            probs: (batch, N_OUTCOMES) outcome probabilities
        """
        batch_size = arsenal_z.shape[0]

        # Population-level logits
        logits = self.intercept.unsqueeze(0).expand(batch_size, -1)  # (B, 10)
        logits = logits + arsenal_z @ self.beta_arsenal  # (B, 13) @ (13, 10) -> (B, 10)
        logits = logits + pitch_mix @ self.beta_ptmix  # (B, P) @ (P, 10) -> (B, 10)

        # Per-hitter effects
        h_base = self.hitter_base[hitter_idx]  # (B, 10)
        logits = logits + h_base

        # Per-hitter arsenal sensitivity: sum over arsenal features
        h_arsenal = self.hitter_arsenal[hitter_idx]  # (B, 13, 10)
        arsenal_contrib = (arsenal_z.unsqueeze(2) * h_arsenal).sum(dim=1)  # (B, 10)
        logits = logits + arsenal_contrib

        # Per-hitter pitch-type response: sum over pitch types
        h_ptype = self.hitter_ptype[hitter_idx]  # (B, P, 10)
        ptype_contrib = (pitch_mix.unsqueeze(2) * h_ptype).sum(dim=1)  # (B, 10)
        logits = logits + ptype_contrib

        # Append zero logit for reference category, then softmax
        zero_col = torch.zeros(batch_size, 1, device=logits.device)
        full_logits = torch.cat([logits, zero_col], dim=1)  # (B, 11)

        probs = torch.softmax(full_logits, dim=1)
        return probs

    def l2_penalty(self, lambda_base=0.1, lambda_arsenal=0.5, lambda_ptype=0.2,
                    lambda_pop=0.01):
        """L2 regularization (hierarchical shrinkage)."""
        # Per-hitter parameters: shrink toward zero (population average)
        penalty = lambda_base * (self.hitter_base ** 2).sum()
        penalty += lambda_arsenal * (self.hitter_arsenal ** 2).sum()
        penalty += lambda_ptype * (self.hitter_ptype ** 2).sum()
        # Population intercepts: shrink toward empirical base rates (not zero)
        penalty += lambda_pop * ((self.intercept - self.intercept_prior) ** 2).sum()
        # Population slopes: shrink toward zero
        penalty += lambda_pop * (self.beta_arsenal ** 2).sum()
        penalty += lambda_pop * (self.beta_ptmix ** 2).sum()
        return penalty

    def predict_single(self, arsenal_z, pitch_mix, hitter_idx=None):
        """Predict for a single matchup. Returns numpy array of 11 probabilities."""
        self.eval()
        with torch.no_grad():
            a = torch.tensor(np.nan_to_num(np.asarray(arsenal_z, dtype=np.float32), 0.0),
                             dtype=torch.float32).unsqueeze(0)
            p = torch.tensor(np.nan_to_num(np.asarray(pitch_mix, dtype=np.float32), 0.0),
                             dtype=torch.float32).unsqueeze(0)

            if hitter_idx is not None:
                h = torch.tensor([hitter_idx], dtype=torch.long)
                return self.forward(a, p, h).numpy()[0]
            else:
                # Unknown hitter: use population average (hitter effects = 0)
                logits = self.intercept.unsqueeze(0)
                logits = logits + a @ self.beta_arsenal
                logits = logits + p @ self.beta_ptmix
                zero_col = torch.zeros(1, 1)
                full_logits = torch.cat([logits, zero_col], dim=1)
                return torch.softmax(full_logits, dim=1).numpy()[0]


# ---------------------------------------------------------------------------
# PA outcome classification
# ---------------------------------------------------------------------------


def _classify_pa_outcome(events_str, bb_type_str=None):
    """Classify a single PA outcome to one of 11 categories."""
    if events_str is None:
        return None

    # field_out needs special handling via bb_type
    if events_str == "field_out":
        if bb_type_str:
            return BBTYPE_MAP.get(bb_type_str, "out_ground")
        return "out_ground"

    mapped = EVENT_MAP.get(events_str)
    if mapped is not None:
        return mapped

    # Fallback heuristics
    if "double_play" in events_str:
        return "dp"
    if "out" in events_str:
        return "out_ground"
    return None


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_training_data(season: int, stand: str, min_pitcher_pitches=300, min_hitter_pa=30):
    """
    Prepare PA-level training data for the multi-output model.

    Returns:
        pa_df: DataFrame with columns [hitter_idx, outcome_idx, game_date, game_pk]
        arsenal_z: (N, 13) float32 array of standardized arsenal features per PA
        ptmix: (N, n_ptypes) float32 array of pitch-type mix per PA
        hitter_map: {mlb_id: idx}
        arsenal_stats: feature standardization stats {feat: {mean, std}}
    """
    # Load xRV data
    path = XRV_DIR / f"statcast_xrv_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"xRV data not found for {season}: {path}")
    xrv = pd.read_parquet(path)
    if "game_type" in xrv.columns:
        xrv = xrv[xrv["game_type"] == "R"]

    # Filter by batter handedness
    xrv = xrv[xrv["stand"] == stand].copy()

    # Get PA rows (events not null)
    pa = xrv[xrv["events"].notna()].copy()

    # Classify outcomes
    pa["outcome"] = pa.apply(
        lambda r: _classify_pa_outcome(r["events"], r.get("bb_type")), axis=1
    )
    pa = pa[pa["outcome"].notna()].copy()
    outcome_to_idx = {o: i for i, o in enumerate(OUTCOME_ORDER)}
    pa["outcome_idx"] = pa["outcome"].map(outcome_to_idx)
    pa = pa[pa["outcome_idx"].notna()].copy()
    pa["outcome_idx"] = pa["outcome_idx"].astype(int)

    # Filter pitchers with enough data
    pitcher_counts = xrv.groupby("pitcher").size()
    valid_pitchers = pitcher_counts[pitcher_counts >= min_pitcher_pitches].index
    pa = pa[pa["pitcher"].isin(valid_pitchers)]

    # Filter hitters with enough PAs
    hitter_pa_counts = pa.groupby("batter").size()
    valid_hitters = hitter_pa_counts[hitter_pa_counts >= min_hitter_pa].index
    pa = pa[pa["batter"].isin(valid_hitters)]

    # Build hitter map
    hitter_ids = sorted(pa["batter"].unique())
    hitter_map = {int(hid): i for i, hid in enumerate(hitter_ids)}
    pa["hitter_idx"] = pa["batter"].map(hitter_map)

    # ------------------------------------------------------------------
    # Compute per-pitcher arsenal profiles and pitch-type mixes
    # ------------------------------------------------------------------
    print("  Computing pitcher arsenal profiles...")
    pitcher_ids = pa["pitcher"].unique()
    arsenal_profiles = {}
    pitch_mixes = {}

    for pid in pitcher_ids:
        pitcher_pitches = xrv[xrv["pitcher"] == pid]
        n = len(pitcher_pitches)
        if n < 100:
            continue

        pt_counts = pitcher_pitches["pitch_type"].value_counts()
        pt_pcts = pt_counts / n

        hard_pitches = pitcher_pitches[pitcher_pitches["pitch_type"].isin(HARD_TYPES)]
        break_pitches = pitcher_pitches[pitcher_pitches["pitch_type"].isin(BREAK_TYPES)]

        hard_pct = sum(pt_pcts.get(t, 0) for t in HARD_TYPES)
        break_pct = sum(pt_pcts.get(t, 0) for t in BREAK_TYPES)
        offspeed_pct = sum(pt_pcts.get(t, 0) for t in OFFSPEED_TYPES)

        pcts = pt_pcts.values
        pcts = pcts[pcts > 0]
        entropy = -float(np.sum(pcts * np.log2(pcts))) if len(pcts) > 0 else 0.0

        hard_velo = float(hard_pitches["release_speed"].mean()) if len(hard_pitches) > 0 else 0.0

        pt_velos = pitcher_pitches.groupby("pitch_type")["release_speed"].mean()
        velo_spread = float(pt_velos.max() - pt_velos.min()) if len(pt_velos) > 1 else 0.0

        pt_hmov = pitcher_pitches.groupby("pitch_type")["pfx_x"].mean()
        hmov_range = float(pt_hmov.max() - pt_hmov.min()) if len(pt_hmov) > 1 else 0.0

        hard_spin = (
            float(hard_pitches["release_spin_rate"].mean())
            if len(hard_pitches) > 0 and "release_spin_rate" in pitcher_pitches.columns
            else 0.0
        )
        break_spin = (
            float(break_pitches["release_spin_rate"].mean())
            if len(break_pitches) > 0 and "release_spin_rate" in pitcher_pitches.columns
            else 0.0
        )
        hard_ivb = float(hard_pitches["pfx_z"].mean()) if len(hard_pitches) > 0 else 0.0
        hard_ext = (
            float(hard_pitches["release_extension"].mean())
            if len(hard_pitches) > 0 and "release_extension" in pitcher_pitches.columns
            else 0.0
        )

        if "release_pos_x" in pitcher_pitches.columns:
            pt_rel_x = pitcher_pitches.groupby("pitch_type")["release_pos_x"].mean().dropna()
            rel_x_spread = (
                float(pt_rel_x.std())
                if len(pt_rel_x) > 1 and pd.notna(pt_rel_x.std())
                else 0.0
            )
        else:
            rel_x_spread = 0.0

        pt_vmov = pitcher_pitches.groupby("pitch_type")["pfx_z"].mean()
        vmov_range = float(pt_vmov.max() - pt_vmov.min()) if len(pt_vmov) > 1 else 0.0

        arsenal_profiles[pid] = {
            "hard_velo": hard_velo,
            "hard_pct": hard_pct,
            "break_pct": break_pct,
            "offspeed_pct": offspeed_pct,
            "velo_spread": velo_spread,
            "hmov_range": hmov_range,
            "entropy": entropy,
            "hard_spin": hard_spin,
            "break_spin": break_spin,
            "hard_ivb": hard_ivb,
            "hard_ext": hard_ext,
            "rel_x_spread": rel_x_spread,
            "vmov_range": vmov_range,
        }

        # Pitch-type mix
        mix = np.zeros(N_PTYPES, dtype=np.float32)
        for pt, count in pt_counts.items():
            if pt in PTYPE_MAP:
                mix[PTYPE_MAP[pt]] = count / n
        pitch_mixes[pid] = mix

    # Filter PAs to pitchers with computed profiles
    pa = pa[pa["pitcher"].isin(arsenal_profiles)]

    # ------------------------------------------------------------------
    # Standardize arsenal features
    # ------------------------------------------------------------------
    arsenal_df = pd.DataFrame.from_dict(arsenal_profiles, orient="index")
    arsenal_stats = {}
    for feat in ARSENAL_FEATURES:
        if feat in arsenal_df.columns:
            vals = arsenal_df[feat].replace([np.inf, -np.inf], np.nan).dropna()
            mean = float(vals.mean()) if len(vals) > 0 else 0.0
            std = float(vals.std()) if len(vals) > 0 else 1.0
            arsenal_stats[feat] = {"mean": mean, "std": std if std > 0 else 1.0}
        else:
            arsenal_stats[feat] = {"mean": 0.0, "std": 1.0}

    # ------------------------------------------------------------------
    # Build feature arrays aligned to PA rows
    # ------------------------------------------------------------------
    pitcher_vals = pa["pitcher"].values
    arsenal_z_rows = np.zeros((len(pa), N_ARSENAL), dtype=np.float32)
    ptmix_rows = np.zeros((len(pa), N_PTYPES), dtype=np.float32)

    for row_i, pid in enumerate(pitcher_vals):
        raw = arsenal_profiles.get(pid)
        if raw is None:
            continue
        for feat_j, feat in enumerate(ARSENAL_FEATURES):
            stats = arsenal_stats[feat]
            val = raw.get(feat, 0.0)
            arsenal_z_rows[row_i, feat_j] = (val - stats["mean"]) / stats["std"]
        ptmix_rows[row_i] = pitch_mixes.get(pid, np.zeros(N_PTYPES, dtype=np.float32))

    # Clean up NaN/Inf
    arsenal_z_rows = np.nan_to_num(arsenal_z_rows, nan=0.0, posinf=0.0, neginf=0.0)
    ptmix_rows = np.nan_to_num(ptmix_rows, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  {len(pa):,} PAs, {len(hitter_ids)} hitters, {len(arsenal_profiles)} pitchers")

    pa_out = pa[["hitter_idx", "outcome_idx", "game_date", "game_pk"]].reset_index(drop=True)
    return pa_out, arsenal_z_rows, ptmix_rows, hitter_map, arsenal_stats


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    pa_df,
    arsenal_z,
    ptmix,
    hitter_map,
    n_epochs=150,
    lr=0.01,
    batch_size=4096,
    lambda_base=0.1,
    lambda_arsenal=0.5,
    lambda_ptype=0.2,
    lambda_pop=0.01,
    val_frac=0.2,
    patience=15,
):
    """Train the multi-output matchup model."""
    n_hitters = len(hitter_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiOutputMatchupModel(n_hitters).to(device)

    # Initialize intercepts from empirical base rates (log-odds vs reference category)
    outcome_counts = pa_df["outcome_idx"].value_counts().sort_index()
    n_outcomes = len(OUTCOME_ORDER)
    empirical_probs = np.zeros(n_outcomes)
    for idx, count in outcome_counts.items():
        empirical_probs[idx] = count
    empirical_probs = empirical_probs / empirical_probs.sum()
    empirical_probs = np.clip(empirical_probs, 1e-6, None)  # avoid log(0)
    # Reference category is last (out_line); logits are log(p_i / p_ref)
    ref_prob = empirical_probs[-1]
    init_logits = np.log(empirical_probs[:-1] / ref_prob).astype(np.float32)
    with torch.no_grad():
        init_tensor = torch.tensor(init_logits, device=device)
        model.intercept.copy_(init_tensor)
        model.intercept_prior.copy_(init_tensor)
    print(f"  Initialized intercepts from empirical rates")
    for i, o in enumerate(OUTCOME_ORDER[:-1]):
        print(f"    {o:>12}: {init_logits[i]:+.4f} (rate={empirical_probs[i]:.4f})")
    print(f"    {'out_line':>12}: ref     (rate={empirical_probs[-1]:.4f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Chronological train/val split
    game_dates = pa_df["game_date"].values
    sorted_dates = np.sort(np.unique(game_dates))
    if len(sorted_dates) == 0:
        raise ValueError("No game dates found in training data")
    split_idx = max(1, int(len(sorted_dates) * (1 - val_frac)))
    split_date = sorted_dates[split_idx]
    train_mask = game_dates <= split_date
    val_mask = ~train_mask

    # Convert to tensors
    hitter_idx = torch.tensor(pa_df["hitter_idx"].values, dtype=torch.long, device=device)
    outcome_idx = torch.tensor(pa_df["outcome_idx"].values, dtype=torch.long, device=device)
    arsenal_t = torch.tensor(arsenal_z, dtype=torch.float32, device=device)
    ptmix_t = torch.tensor(ptmix, dtype=torch.float32, device=device)

    train_idx = torch.where(torch.tensor(train_mask, device=device))[0]
    val_idx = torch.where(torch.tensor(val_mask, device=device))[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"Train/val split produced empty set: train={len(train_idx)}, val={len(val_idx)}"
        )

    print(f"  Train: {len(train_idx):,} PAs, Val: {len(val_idx):,} PAs")
    print(f"  Hitters: {n_hitters}, Device: {device}")

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()

        # Shuffle training indices
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        epoch_loss = 0.0
        n_samples = 0

        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            if len(batch_idx) == 0:
                continue

            probs = model(
                arsenal_t[batch_idx],
                ptmix_t[batch_idx],
                hitter_idx[batch_idx],
            )

            # Cross-entropy loss
            targets = outcome_idx[batch_idx]
            log_probs = torch.log(probs + 1e-10)
            ce_loss = nn.functional.nll_loss(log_probs, targets)

            # L2 regularization scaled by dataset size
            reg_loss = model.l2_penalty(lambda_base, lambda_arsenal, lambda_ptype, lambda_pop)

            loss = ce_loss + reg_loss / len(train_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += ce_loss.item() * len(batch_idx)
            n_samples += len(batch_idx)

        train_loss = epoch_loss / max(n_samples, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            # Process validation in chunks to avoid OOM on large datasets
            val_loss_sum = 0.0
            val_n = 0
            for v_start in range(0, len(val_idx), batch_size):
                v_batch = val_idx[v_start : v_start + batch_size]
                v_probs = model(arsenal_t[v_batch], ptmix_t[v_batch], hitter_idx[v_batch])
                v_log_probs = torch.log(v_probs + 1e-10)
                v_loss = nn.functional.nll_loss(v_log_probs, outcome_idx[v_batch], reduction="sum")
                val_loss_sum += v_loss.item()
                val_n += len(v_batch)
            val_loss = val_loss_sum / max(val_n, 1)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch + 1:3d}: train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(
                    f"    Early stopping at epoch {epoch + 1} "
                    f"(best val_loss={best_val_loss:.4f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.cpu()
    return model


# ---------------------------------------------------------------------------
# Save / Load / Predict
# ---------------------------------------------------------------------------


def save_model_artifacts(model, hitter_map, arsenal_stats, season, hand):
    """Save trained model artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "year": season,
        "hand": hand,
        "state_dict": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        "hitter_map": hitter_map,
        "arsenal_stats": arsenal_stats,
        "n_hitters": model.n_hitters,
        "n_arsenal": N_ARSENAL,
        "n_ptypes": N_PTYPES,
        "n_outcomes": N_OUTCOMES,
        "outcome_order": list(OUTCOME_ORDER),
        "pitch_types": list(PITCH_TYPES),
        "arsenal_features": list(ARSENAL_FEATURES),
    }

    out_path = MODEL_DIR / f"multi_output_matchup_{season}_vs{hand}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"  Saved to {out_path}")


def load_multi_output_models(season):
    """Load trained multi-output models for both hands."""
    models = {}
    for hand in ["L", "R"]:
        path = MODEL_DIR / f"multi_output_matchup_{season}_vs{hand}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                artifacts = pickle.load(f)

            n_hitters = artifacts["n_hitters"]
            model = MultiOutputMatchupModel(n_hitters)

            # Load state dict (convert numpy back to tensors)
            state_dict = {k: torch.tensor(v) for k, v in artifacts["state_dict"].items()}
            model.load_state_dict(state_dict)
            model.eval()

            artifacts["model"] = model
            models[hand] = artifacts
            print(f"  Loaded multi_output_matchup_{season}_vs{hand}")
    return models


def predict_matchup_distribution(
    model_artifacts: dict,
    arsenal_z: np.ndarray,
    pitch_mix: np.ndarray,
    hitter_id: int,
) -> np.ndarray:
    """
    Predict outcome distribution for a single hitter-pitcher matchup.

    Args:
        model_artifacts: loaded model dict (with "model" key)
        arsenal_z: (13,) standardized arsenal features of the pitcher
        pitch_mix: (n_ptypes,) pitch-type mix fractions
        hitter_id: MLB player ID

    Returns:
        (11,) numpy array of outcome probabilities in OUTCOME_ORDER
    """
    model = model_artifacts["model"]
    hitter_map = model_artifacts["hitter_map"]

    hitter_idx = hitter_map.get(hitter_id)
    return model.predict_single(arsenal_z, pitch_mix, hitter_idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Multi-Output Matchup Model")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lambda-base", type=float, default=0.1)
    parser.add_argument("--lambda-arsenal", type=float, default=0.5)
    parser.add_argument("--lambda-ptype", type=float, default=0.2)
    parser.add_argument("--lambda-pop", type=float, default=0.01)
    parser.add_argument(
        "--pool-seasons",
        type=int,
        nargs="+",
        default=None,
        help="Pool multiple seasons (e.g. --pool-seasons 2022 2023 2024)",
    )
    args = parser.parse_args()

    for hand in ["L", "R"]:
        print(f"\n{'=' * 60}")
        print(f"Training multi-output model vs {hand}HH")
        print(f"{'=' * 60}")

        seasons = args.pool_seasons or [args.season]
        all_pa = []
        all_arsenal_z = []
        all_ptmix = []
        combined_hitter_map: dict[int, int] = {}
        combined_arsenal_stats = None

        for yr in seasons:
            print(f"\n  Loading season {yr}...")
            pa_df, arsenal_z, ptmix, hitter_map, arsenal_stats = prepare_training_data(
                yr,
                hand,
            )

            # Build inverse map for this season: idx -> mlb_id
            idx_to_hid = {idx: hid for hid, idx in hitter_map.items()}

            # Register any new hitters into the combined map
            for hid in hitter_map:
                if hid not in combined_hitter_map:
                    combined_hitter_map[hid] = len(combined_hitter_map)

            # Remap hitter indices for combined dataset
            pa_df = pa_df.copy()
            pa_df["hitter_idx"] = pa_df["hitter_idx"].map(
                lambda old_idx: combined_hitter_map[idx_to_hid[old_idx]]
            )

            all_pa.append(pa_df)
            all_arsenal_z.append(arsenal_z)
            all_ptmix.append(ptmix)
            combined_arsenal_stats = arsenal_stats  # Use latest season's stats

        pa_df = pd.concat(all_pa, ignore_index=True)
        arsenal_z = np.concatenate(all_arsenal_z)
        ptmix = np.concatenate(all_ptmix)

        print(f"\n  Combined: {len(pa_df):,} PAs, {len(combined_hitter_map)} hitters")

        model = train_model(
            pa_df,
            arsenal_z,
            ptmix,
            combined_hitter_map,
            n_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            lambda_base=args.lambda_base,
            lambda_arsenal=args.lambda_arsenal,
            lambda_ptype=args.lambda_ptype,
            lambda_pop=args.lambda_pop,
        )

        save_model_artifacts(
            model, combined_hitter_map, combined_arsenal_stats, args.season, hand
        )

        # Print diagnostics
        print(f"\n  Population intercepts (logits):")
        intercept = model.intercept.detach().numpy()
        for i, o in enumerate(OUTCOME_ORDER[:-1]):
            print(f"    {o:>12}: {intercept[i]:+.4f}")

        # Show population probabilities
        pop_probs = torch.softmax(
            torch.cat([model.intercept, torch.zeros(1)]), dim=0
        ).detach().numpy()
        print(f"\n  Population-average outcome probabilities:")
        for i, o in enumerate(OUTCOME_ORDER):
            print(f"    {o:>12}: {pop_probs[i]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
