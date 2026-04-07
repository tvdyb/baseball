#!/usr/bin/env python3
"""
Neural Net Win Probability Model

Predicts P(home_win) from per-hitter PA outcome distributions and game context.
Uses a MatchupAggregationNetwork with self-attention over lineup embeddings.

Architecture (~5K parameters):
  - Lineup Encoder (shared): per-hitter 11+4 -> 16 -> 8, self-attention, pool -> 16
  - Pitcher Encoder (shared): SP(3) + rest(1) + n_pitches(1) -> 8, BP(2) -> 4, concat -> 12
  - Game Head: 16+16+12+12+5 = 61 -> 32 -> 16 -> 1 (sigmoid)

Usage:
    python src/nn_win_model.py --train
    python src/nn_win_model.py --evaluate
    python src/nn_win_model.py --predict --season 2025
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import DATA_DIR, MODEL_DIR, FEATURES_DIR
from build_transition_matrix import OUTCOME_ORDER

N_OUTCOMES = len(OUTCOME_ORDER)  # 11
N_HITTERS = 9
POS_EMBED_DIM = 4

# Context features: park_factor, temperature, wind_speed, is_dome, is_night
N_CONTEXT = 5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HitterEncoder(nn.Module):
    """Encode a single hitter's PA distribution + batting order position."""

    def __init__(self, n_outcomes=N_OUTCOMES, pos_dim=POS_EMBED_DIM, hidden=16, out_dim=8):
        super().__init__()
        self.pos_embed = nn.Embedding(N_HITTERS, pos_dim)
        self.fc1 = nn.Linear(n_outcomes + pos_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, pa_dist, position_idx):
        """
        Args:
            pa_dist: (batch, 9, 11) PA outcome distributions
            position_idx: (batch, 9) lineup positions [0..8]

        Returns:
            (batch, 9, out_dim) hitter embeddings
        """
        pos_emb = self.pos_embed(position_idx)  # (B, 9, pos_dim)
        x = torch.cat([pa_dist, pos_emb], dim=-1)  # (B, 9, 15)
        x = F.relu(self.fc1(x))  # (B, 9, 16)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, 9, 8)
        return x


class LineupAttention(nn.Module):
    """Single-head self-attention over 9 hitter embeddings."""

    def __init__(self, dim=8):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, 9, dim)
            mask: (batch, 9) bool, True = valid hitter, False = missing

        Returns:
            (batch, 9, dim)
        """
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, 9, 9)

        if mask is not None:
            # Mask out attention to missing hitters
            mask_2d = mask.unsqueeze(1).expand_as(attn)  # (B, 9, 9)
            attn = attn.masked_fill(~mask_2d, -1e9)

        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, V)  # (B, 9, dim)
        return out


class LineupEncoder(nn.Module):
    """Full lineup encoder: per-hitter MLP + self-attention + pooling."""

    def __init__(self, hitter_dim=8, pool_dim=16):
        super().__init__()
        self.hitter_enc = HitterEncoder(out_dim=hitter_dim)
        self.attention = LineupAttention(dim=hitter_dim)
        self.pool_dim = pool_dim
        # pool_dim = 2 * hitter_dim (mean + max)
        assert pool_dim == 2 * hitter_dim

    def forward(self, pa_dist, position_idx, mask):
        """
        Args:
            pa_dist: (batch, 9, 11)
            position_idx: (batch, 9) [0..8]
            mask: (batch, 9) bool

        Returns:
            (batch, pool_dim) lineup representation
        """
        h = self.hitter_enc(pa_dist, position_idx)  # (B, 9, 8)
        h = self.attention(h, mask)  # (B, 9, 8)

        # Masked pooling
        mask_f = mask.unsqueeze(-1).float()  # (B, 9, 1)
        h_masked = h * mask_f

        # Mean pool (only over valid hitters)
        n_valid = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_pool = h_masked.sum(dim=1) / n_valid  # (B, 8)

        # Max pool (set masked to -inf)
        h_for_max = h_masked + (1 - mask_f) * (-1e9)
        max_pool = h_for_max.max(dim=1).values  # (B, 8)

        return torch.cat([mean_pool, max_pool], dim=-1)  # (B, 16)


class PitcherEncoder(nn.Module):
    """Encode SP + bullpen features."""

    def __init__(self, sp_in=5, bp_in=2, sp_out=8, bp_out=4):
        super().__init__()
        self.sp_fc = nn.Linear(sp_in, sp_out)
        self.bp_fc = nn.Linear(bp_in, bp_out)
        self.dropout = nn.Dropout(0.3)
        self.out_dim = sp_out + bp_out

    def forward(self, sp_features, bp_features):
        """
        Args:
            sp_features: (batch, 5) [stuff, location, sequencing, rest_days, n_pitches]
            bp_features: (batch, 2) [bp_xrv, bp_fatigue]

        Returns:
            (batch, 12)
        """
        sp = F.relu(self.sp_fc(sp_features))
        sp = self.dropout(sp)
        bp = F.relu(self.bp_fc(bp_features))
        return torch.cat([sp, bp], dim=-1)


class MatchupAggregationNetwork(nn.Module):
    """Full model: lineup encoders + pitcher encoders + game head -> P(home_win)."""

    def __init__(self):
        super().__init__()
        self.lineup_enc = LineupEncoder(hitter_dim=8, pool_dim=16)
        self.pitcher_enc = PitcherEncoder(sp_in=5, bp_in=2, sp_out=8, bp_out=4)

        head_in = 16 + 16 + 12 + 12 + N_CONTEXT  # = 61
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, home_pa, home_pos, home_mask,
                away_pa, away_pos, away_mask,
                home_sp, home_bp, away_sp, away_bp,
                context):
        """
        Returns:
            (batch, 1) logits for P(home_win)
        """
        home_lineup = self.lineup_enc(home_pa, home_pos, home_mask)  # (B, 16)
        away_lineup = self.lineup_enc(away_pa, away_pos, away_mask)  # (B, 16)
        home_pitch = self.pitcher_enc(home_sp, home_bp)  # (B, 12)
        away_pitch = self.pitcher_enc(away_sp, away_bp)  # (B, 12)

        x = torch.cat([home_lineup, away_lineup, home_pitch, away_pitch, context], dim=-1)
        return self.head(x)  # (B, 1)

    def predict_proba(self, *args):
        """Returns P(home_win) as numpy array."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(*args)
            return torch.sigmoid(logits).squeeze(-1).numpy()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalize_feature(values: np.ndarray, mean: float = None, std: float = None):
    """Z-score normalize, returning (normalized, mean, std)."""
    if mean is None:
        mean = float(np.nanmean(values))
    if std is None:
        std = float(np.nanstd(values))
    if std < 1e-8:
        std = 1.0
    normed = (values - mean) / std
    return np.nan_to_num(normed, nan=0.0).astype(np.float32), mean, std


def load_nn_features(years: list[int]) -> pd.DataFrame:
    """Load and concatenate NN features for multiple years."""
    frames = []
    for year in years:
        path = FEATURES_DIR / f"nn_features_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["season"] = year
            frames.append(df)
            print(f"  Loaded {len(df)} games from {year}")
        else:
            print(f"  WARNING: {path} not found, skipping")
    if not frames:
        raise FileNotFoundError("No NN feature files found")
    return pd.concat(frames, ignore_index=True)


def prepare_tensors(df: pd.DataFrame, norm_stats: dict = None, fit_norm: bool = True):
    """Convert DataFrame to model input tensors.

    Args:
        df: NN features DataFrame
        norm_stats: dict of {feature: (mean, std)} for normalization
        fit_norm: if True, compute normalization stats from this data

    Returns:
        tensor_dict, norm_stats, valid_mask (games with enough data)
    """
    n = len(df)

    if norm_stats is None:
        norm_stats = {}

    # --- PA distributions: (n, 9, 11) per side ---
    home_pa = np.full((n, N_HITTERS, N_OUTCOMES), np.nan, dtype=np.float32)
    away_pa = np.full((n, N_HITTERS, N_OUTCOMES), np.nan, dtype=np.float32)

    for slot in range(N_HITTERS):
        for oi, outcome in enumerate(OUTCOME_ORDER):
            col_h = f"home_h{slot}_{outcome}"
            col_a = f"away_h{slot}_{outcome}"
            if col_h in df.columns:
                home_pa[:, slot, oi] = df[col_h].values.astype(np.float32)
            if col_a in df.columns:
                away_pa[:, slot, oi] = df[col_a].values.astype(np.float32)

    # Create masks: a hitter slot is valid if any outcome is not NaN
    home_mask = ~np.isnan(home_pa[:, :, 0])  # (n, 9)
    away_mask = ~np.isnan(away_pa[:, :, 0])  # (n, 9)

    # Fill NaN distributions with uniform (1/11 each) — these are masked out anyway
    uniform = np.full(N_OUTCOMES, 1.0 / N_OUTCOMES, dtype=np.float32)
    for i in range(n):
        for slot in range(N_HITTERS):
            if np.isnan(home_pa[i, slot, 0]):
                home_pa[i, slot] = uniform
            if np.isnan(away_pa[i, slot, 0]):
                away_pa[i, slot] = uniform

    # Position indices (static)
    positions = np.tile(np.arange(N_HITTERS, dtype=np.int64), (n, 1))  # (n, 9)

    # --- Pitcher features ---
    sp_feature_names = ["sp_stuff", "sp_location", "sp_sequencing", "sp_rest_days", "sp_n_pitches"]

    home_sp = np.zeros((n, 5), dtype=np.float32)
    away_sp = np.zeros((n, 5), dtype=np.float32)

    for fi, feat in enumerate(sp_feature_names):
        for side, arr in [("home", home_sp), ("away", away_sp)]:
            col = f"{side}_{feat}"
            if col in df.columns:
                raw = df[col].values.astype(np.float64)
            else:
                raw = np.full(n, np.nan, dtype=np.float64)

            key = f"sp_{feat}"
            if fit_norm and key not in norm_stats:
                _, mean, std = _normalize_feature(raw)
                norm_stats[key] = (mean, std)
            mean, std = norm_stats.get(key, (0.0, 1.0))
            arr[:, fi] = np.nan_to_num((raw - mean) / std, nan=0.0).astype(np.float32)

    # Bullpen features
    home_bp = np.zeros((n, 2), dtype=np.float32)
    away_bp = np.zeros((n, 2), dtype=np.float32)
    bp_feats = ["bp_xrv", "bp_fatigue"]

    for fi, feat in enumerate(bp_feats):
        for side, arr in [("home", home_bp), ("away", away_bp)]:
            col = f"{side}_{feat}"
            if col in df.columns:
                raw = df[col].values.astype(np.float64)
            else:
                raw = np.full(n, np.nan, dtype=np.float64)

            key = f"bp_{feat}"
            if fit_norm and key not in norm_stats:
                _, mean, std = _normalize_feature(raw)
                norm_stats[key] = (mean, std)
            mean, std = norm_stats.get(key, (0.0, 1.0))
            arr[:, fi] = np.nan_to_num((raw - mean) / std, nan=0.0).astype(np.float32)

    # --- Context features ---
    context = np.zeros((n, N_CONTEXT), dtype=np.float32)
    ctx_feats = ["park_factor", "temperature", "wind_speed", "is_dome", "is_night"]

    for fi, feat in enumerate(ctx_feats):
        if feat in df.columns:
            raw = df[feat].values.astype(np.float64)
        else:
            raw = np.full(n, np.nan, dtype=np.float64)

        key = f"ctx_{feat}"
        if fit_norm and key not in norm_stats:
            _, mean, std = _normalize_feature(raw)
            norm_stats[key] = (mean, std)
        mean, std = norm_stats.get(key, (0.0, 1.0))
        context[:, fi] = np.nan_to_num((raw - mean) / std, nan=0.0).astype(np.float32)

    # --- Labels ---
    labels = df["home_win"].values.astype(np.float32) if "home_win" in df.columns else np.full(n, np.nan)

    # Valid mask: at least 3 hitters per side with distributions
    valid = (home_mask.sum(axis=1) >= 3) & (away_mask.sum(axis=1) >= 3) & ~np.isnan(labels)

    tensors = {
        "home_pa": torch.tensor(home_pa),
        "home_pos": torch.tensor(positions),
        "home_mask": torch.tensor(home_mask),
        "away_pa": torch.tensor(away_pa),
        "away_pos": torch.tensor(positions),
        "away_mask": torch.tensor(away_mask),
        "home_sp": torch.tensor(home_sp),
        "home_bp": torch.tensor(home_bp),
        "away_sp": torch.tensor(away_sp),
        "away_bp": torch.tensor(away_bp),
        "context": torch.tensor(context),
        "labels": torch.tensor(labels),
        "valid": torch.tensor(valid),
    }

    return tensors, norm_stats


def _augment_swap(tensors: dict, idx: torch.Tensor) -> dict:
    """Data augmentation: swap home/away with flipped label. Returns new tensors for the batch."""
    return {
        "home_pa": tensors["away_pa"][idx],
        "home_pos": tensors["away_pos"][idx],
        "home_mask": tensors["away_mask"][idx],
        "away_pa": tensors["home_pa"][idx],
        "away_pos": tensors["home_pos"][idx],
        "away_mask": tensors["home_mask"][idx],
        "home_sp": tensors["away_sp"][idx],
        "home_bp": tensors["away_bp"][idx],
        "away_sp": tensors["home_sp"][idx],
        "away_bp": tensors["home_bp"][idx],
        "context": tensors["context"][idx],  # context stays the same (park is home-centric but OK for augment)
        "labels": 1.0 - tensors["labels"][idx],
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    train_tensors: dict,
    val_tensors: dict,
    n_epochs: int = 200,
    lr: float = 3e-3,
    batch_size: int = 128,
    weight_decay: float = 1e-3,
    patience: int = 20,
    augment: bool = True,
) -> tuple[MatchupAggregationNetwork, dict]:
    """Train the MatchupAggregationNetwork.

    Returns:
        (model, training_info)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatchupAggregationNetwork().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5, min_lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()

    # Get valid indices
    train_valid = torch.where(train_tensors["valid"])[0]
    val_valid = torch.where(val_tensors["valid"])[0]

    if len(train_valid) == 0:
        raise ValueError("No valid training samples")
    if len(val_valid) == 0:
        raise ValueError("No valid validation samples")

    print(f"  Train: {len(train_valid)} games, Val: {len(val_valid)} games")

    # Move tensors to device
    def to_device(t_dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t_dict.items()}

    train_t = to_device(train_tensors)
    val_t = to_device(val_tensors)

    best_val_brier = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()

        # Shuffle train indices
        perm = train_valid[torch.randperm(len(train_valid))]
        epoch_loss = 0.0
        n_samples = 0

        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start:start + batch_size]
            if len(batch_idx) == 0:
                continue

            # Original batch
            logits = model(
                train_t["home_pa"][batch_idx], train_t["home_pos"][batch_idx],
                train_t["home_mask"][batch_idx],
                train_t["away_pa"][batch_idx], train_t["away_pos"][batch_idx],
                train_t["away_mask"][batch_idx],
                train_t["home_sp"][batch_idx], train_t["home_bp"][batch_idx],
                train_t["away_sp"][batch_idx], train_t["away_bp"][batch_idx],
                train_t["context"][batch_idx],
            ).squeeze(-1)
            labels = train_t["labels"][batch_idx]
            loss = criterion(logits, labels)

            # Augmented batch (swap home/away)
            if augment:
                aug = _augment_swap(train_t, batch_idx)
                aug_logits = model(
                    aug["home_pa"], aug["home_pos"], aug["home_mask"],
                    aug["away_pa"], aug["away_pos"], aug["away_mask"],
                    aug["home_sp"], aug["home_bp"],
                    aug["away_sp"], aug["away_bp"],
                    aug["context"],
                ).squeeze(-1)
                aug_loss = criterion(aug_logits, aug["labels"])
                loss = (loss + aug_loss) / 2.0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)
            n_samples += len(batch_idx)

        train_loss = epoch_loss / max(n_samples, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(
                val_t["home_pa"][val_valid], val_t["home_pos"][val_valid],
                val_t["home_mask"][val_valid],
                val_t["away_pa"][val_valid], val_t["away_pos"][val_valid],
                val_t["away_mask"][val_valid],
                val_t["home_sp"][val_valid], val_t["home_bp"][val_valid],
                val_t["away_sp"][val_valid], val_t["away_bp"][val_valid],
                val_t["context"][val_valid],
            ).squeeze(-1)
            val_labels = val_t["labels"][val_valid]
            val_loss = criterion(val_logits, val_labels).item()

            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_labels_np = val_labels.cpu().numpy()
            val_brier = float(np.mean((val_probs - val_labels_np) ** 2))

        scheduler.step(val_brier)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch + 1:3d}: train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_brier={val_brier:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        if val_brier < best_val_brier:
            best_val_brier = val_brier
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {epoch + 1} "
                      f"(best val_brier={best_val_brier:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()

    info = {
        "best_val_brier": best_val_brier,
        "n_train": len(train_valid),
        "n_val": len(val_valid),
        "n_params": n_params,
    }
    return model, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(model: MatchupAggregationNetwork, tensors: dict, label: str = "Test"):
    """Evaluate model on a dataset. Returns dict of metrics."""
    valid_idx = torch.where(tensors["valid"])[0]
    if len(valid_idx) == 0:
        print(f"  {label}: no valid samples")
        return {}

    model.eval()
    with torch.no_grad():
        logits = model(
            tensors["home_pa"][valid_idx], tensors["home_pos"][valid_idx],
            tensors["home_mask"][valid_idx],
            tensors["away_pa"][valid_idx], tensors["away_pos"][valid_idx],
            tensors["away_mask"][valid_idx],
            tensors["home_sp"][valid_idx], tensors["home_bp"][valid_idx],
            tensors["away_sp"][valid_idx], tensors["away_bp"][valid_idx],
            tensors["context"][valid_idx],
        ).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()
        labels = tensors["labels"][valid_idx].numpy()

    brier = float(np.mean((probs - labels) ** 2))
    log_loss = float(-np.mean(labels * np.log(probs + 1e-10) +
                               (1 - labels) * np.log(1 - probs + 1e-10)))

    # Calibration: bin probabilities
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    cal_error = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            bin_mean_pred = probs[mask].mean()
            bin_mean_actual = labels[mask].mean()
            cal_error += abs(bin_mean_pred - bin_mean_actual) * mask.sum()
    cal_error /= len(probs)  # Expected Calibration Error

    # Accuracy
    pred_labels = (probs >= 0.5).astype(float)
    accuracy = float((pred_labels == labels).mean())

    # Home win rate sanity check
    home_win_rate = labels.mean()
    mean_pred = probs.mean()

    print(f"\n  {label} Results ({len(valid_idx)} games):")
    print(f"    Brier score:    {brier:.4f}")
    print(f"    Log loss:       {log_loss:.4f}")
    print(f"    ECE:            {cal_error:.4f}")
    print(f"    Accuracy:       {accuracy:.4f}")
    print(f"    Home win rate:  {home_win_rate:.4f}")
    print(f"    Mean prediction:{mean_pred:.4f}")

    return {
        "brier": brier, "log_loss": log_loss, "ece": cal_error,
        "accuracy": accuracy, "n_games": len(valid_idx),
        "home_win_rate": home_win_rate, "mean_pred": mean_pred,
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_model(model: MatchupAggregationNetwork, norm_stats: dict, info: dict):
    """Save model and normalization stats."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "nn_win_model.pkl"

    artifacts = {
        "state_dict": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        "norm_stats": norm_stats,
        "info": info,
        "outcome_order": list(OUTCOME_ORDER),
        "n_hitters": N_HITTERS,
        "n_outcomes": N_OUTCOMES,
        "n_context": N_CONTEXT,
    }
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n  Model saved to {path}")


def load_model() -> tuple[MatchupAggregationNetwork, dict]:
    """Load trained model and normalization stats."""
    path = MODEL_DIR / "nn_win_model.pkl"
    with open(path, "rb") as f:
        artifacts = pickle.load(f)

    model = MatchupAggregationNetwork()
    state_dict = {k: torch.tensor(v) for k, v in artifacts["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    return model, artifacts["norm_stats"]


def predict_game(model: MatchupAggregationNetwork, nn_features_row: pd.Series,
                 norm_stats: dict) -> float | None:
    """Predict P(home_win) for a single game from its nn_features row.

    Returns None if insufficient data.
    """
    df = pd.DataFrame([nn_features_row.to_dict()])
    tensors, _ = prepare_tensors(df, norm_stats=norm_stats, fit_norm=False)

    if not tensors["valid"][0]:
        return None

    model.eval()
    with torch.no_grad():
        logit = model(
            tensors["home_pa"][:1], tensors["home_pos"][:1], tensors["home_mask"][:1],
            tensors["away_pa"][:1], tensors["away_pos"][:1], tensors["away_mask"][:1],
            tensors["home_sp"][:1], tensors["home_bp"][:1],
            tensors["away_sp"][:1], tensors["away_bp"][:1],
            tensors["context"][:1],
        ).squeeze()
        return float(torch.sigmoid(logit).item())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="NN Win Probability Model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    parser.add_argument("--predict", action="store_true", help="Generate predictions")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    if args.train:
        print("\n" + "=" * 60)
        print("  Training NN Win Model")
        print("=" * 60)

        # Load features
        print("\nLoading features...")
        df = load_nn_features([2024, 2025])

        # Temporal split: train on 2024, validate on Apr-Jun 2025, test on Jul+ 2025
        train_mask = df["season"] == 2024
        val_mask = (df["season"] == 2025) & (df["game_date"] < "2025-07-01")
        test_mask = (df["season"] == 2025) & (df["game_date"] >= "2025-07-01")

        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[val_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        print(f"\n  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Prepare tensors (fit normalization on train only)
        print("\nPreparing tensors...")
        train_tensors, norm_stats = prepare_tensors(train_df, fit_norm=True)
        val_tensors, _ = prepare_tensors(val_df, norm_stats=norm_stats, fit_norm=False)

        # Train
        print("\nTraining...")
        model, info = train_model(
            train_tensors, val_tensors,
            n_epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, patience=args.patience,
        )

        # Evaluate on train and val
        evaluate_model(model, train_tensors, "Train")
        evaluate_model(model, val_tensors, "Validation")

        # Test set if available
        if len(test_df) > 0:
            test_tensors, _ = prepare_tensors(test_df, norm_stats=norm_stats, fit_norm=False)
            test_metrics = evaluate_model(model, test_tensors, "Test")
            info["test_metrics"] = test_metrics

        # Save
        save_model(model, norm_stats, info)

    elif args.evaluate:
        print("\n" + "=" * 60)
        print("  Evaluating NN Win Model")
        print("=" * 60)

        model, norm_stats = load_model()
        df = load_nn_features([args.season])
        tensors, _ = prepare_tensors(df, norm_stats=norm_stats, fit_norm=False)
        evaluate_model(model, tensors, f"Season {args.season}")

    elif args.predict:
        print("\n" + "=" * 60)
        print(f"  Generating predictions for {args.season}")
        print("=" * 60)

        model, norm_stats = load_model()
        df = load_nn_features([args.season])
        tensors, _ = prepare_tensors(df, norm_stats=norm_stats, fit_norm=False)

        valid_idx = torch.where(tensors["valid"])[0]
        model.eval()

        with torch.no_grad():
            logits = model(
                tensors["home_pa"][valid_idx], tensors["home_pos"][valid_idx],
                tensors["home_mask"][valid_idx],
                tensors["away_pa"][valid_idx], tensors["away_pos"][valid_idx],
                tensors["away_mask"][valid_idx],
                tensors["home_sp"][valid_idx], tensors["home_bp"][valid_idx],
                tensors["away_sp"][valid_idx], tensors["away_bp"][valid_idx],
                tensors["context"][valid_idx],
            ).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()

        # Build output dataframe
        valid_rows = df.iloc[valid_idx.numpy()]
        out = valid_rows[["game_pk", "game_date", "home_team", "away_team"]].copy()
        out["nn_home_prob"] = probs
        if "home_win" in valid_rows.columns:
            out["home_win"] = valid_rows["home_win"].values

        out_path = FEATURES_DIR / f"nn_predictions_{args.season}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"\n  Predictions saved to {out_path} ({len(out)} games)")

        # Summary
        print(f"  Mean P(home_win): {probs.mean():.4f}")
        print(f"  Std P(home_win):  {probs.std():.4f}")
        if "home_win" in valid_rows.columns:
            labels = valid_rows["home_win"].values
            brier = float(np.mean((probs - labels) ** 2))
            print(f"  Brier score:      {brier:.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
