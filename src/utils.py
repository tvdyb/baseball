#!/usr/bin/env python3
"""
Shared utilities and constants used across the MLB model pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ── Directory constants ──────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
XRV_DIR = DATA_DIR / "xrv"
GAMES_DIR = DATA_DIR / "games"
WEATHER_DIR = DATA_DIR / "weather"
OAA_DIR = DATA_DIR / "oaa"
MODEL_DIR = PROJECT_DIR / "models"
FEATURES_DIR = DATA_DIR / "features"
STATCAST_DIR = DATA_DIR / "statcast"
SPRINT_DIR = DATA_DIR / "sprint_speed"
ROSTER_DIR = DATA_DIR / "rosters"

# ── Pitch type groupings ────────────────────────────────────
HARD_TYPES = {"FF", "SI", "FC"}
BREAK_TYPES = {"SL", "CU", "KC", "SV", "ST", "SW"}
OFFSPEED_TYPES = {"CH", "FS"}


def filter_competitive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to pitches in competitive counts.
    Excludes 0-2 (waste pitches), 3-0 (hitter takes), and intentional balls.
    """
    if "balls" not in df.columns or "strikes" not in df.columns:
        return df
    exclude_mask = (
        ((df["balls"] == 0) & (df["strikes"] == 2)) |
        ((df["balls"] == 3) & (df["strikes"] == 0))
    )
    if "description" in df.columns:
        exclude_mask = exclude_mask | df["description"].str.contains(
            "intentional", case=False, na=False
        )
    return df[~exclude_mask]
