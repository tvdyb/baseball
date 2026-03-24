#!/usr/bin/env python3
"""
Build the xRV (expected run value) model.

Every pitch gets an xRV:
  - Non-contact (B/S): delta_run_exp is deterministic from count transitions.
    Use as-is — no noise to strip.
  - Contact (X): actual delta_run_exp has BABIP noise. Replace with expected
    value from a model of: exit_velo + launch_angle + spray_angle + sprint_speed + park.

The contact model is a gradient-boosted tree (LightGBM) trained on
batted ball outcomes. We train one model per season (or pooled) to
predict delta_run_exp from batted ball characteristics.

Output: augmented Statcast parquet files with an `xrv` column for every pitch.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATCAST_DIR = DATA_DIR / "statcast"
SPRINT_DIR = DATA_DIR / "sprint_speed"
OUTPUT_DIR = DATA_DIR / "xrv"


def compute_spray_angle(hc_x: pd.Series, hc_y: pd.Series) -> pd.Series:
    """
    Compute spray angle from hit coordinates.
    hc_x, hc_y are in the Statcast coordinate system where home plate is
    approximately at (125.42, 198.27). Spray angle in degrees:
      0 = straight up the middle
      negative = pull side (left for RHH, right for LHH)
      positive = opposite field
    """
    # Statcast coordinates: x increases left-to-right from catcher's view
    # y increases from home to outfield (y decreases as ball goes further out)
    dx = hc_x - 125.42
    dy = 198.27 - hc_y
    angle_rad = np.arctan2(dx, dy)
    return np.degrees(angle_rad)


def load_and_prepare(year: int) -> pd.DataFrame:
    """Load Statcast data for a year and add derived columns."""
    path = STATCAST_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        print(f"  {path} not found, skipping")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["season"] = year

    # Spray angle from hit coordinates
    df["spray_angle"] = compute_spray_angle(df["hc_x"], df["hc_y"])

    # Join sprint speed
    sprint_path = SPRINT_DIR / "sprint_speed_all.parquet"
    if sprint_path.exists():
        sprint = pd.read_parquet(sprint_path)
        sprint = sprint[sprint["season"] == year][["batter", "sprint_speed"]]
        df = df.merge(sprint, on="batter", how="left")
    else:
        df["sprint_speed"] = np.nan

    # Park: use home_team as proxy (each team has a home park)
    # We'll encode this as a categorical feature
    df["park"] = df["home_team"]

    return df


def prepare_contact_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare feature matrix for contact events.
    Returns (X, y, feature_names).
    """
    contact = df[df["type"] == "X"].copy()
    contact = contact.dropna(subset=["delta_run_exp", "launch_speed", "launch_angle"])

    features = ["launch_speed", "launch_angle", "spray_angle", "sprint_speed"]
    cat_features = ["park"]

    # Fill missing sprint speed with league average
    league_avg_sprint = contact["sprint_speed"].median()
    if pd.isna(league_avg_sprint):
        league_avg_sprint = 27.0  # reasonable default
    contact["sprint_speed"] = contact["sprint_speed"].fillna(league_avg_sprint)

    # Encode park as category codes
    contact["park"] = contact["park"].astype("category")

    X = contact[features + cat_features].copy()
    y = contact["delta_run_exp"]

    return X, y, features, cat_features


def train_xrv_model(X, y, features, cat_features):
    """Train LightGBM model to predict run value from batted ball characteristics."""

    if not HAS_LGB:
        # Fallback: sklearn gradient boosting
        from sklearn.ensemble import GradientBoostingRegressor
        print("  LightGBM not available, using sklearn GBR (slower)")

        # Need to encode categoricals manually
        X_encoded = X.copy()
        for c in cat_features:
            X_encoded[c] = X_encoded[c].cat.codes

        model = GradientBoostingRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=50,
        )
        model.fit(X_encoded, y)
        return model, "sklearn"

    # LightGBM with categorical support
    cat_indices = [list(X.columns).index(c) for c in cat_features]

    # Convert categoricals to int codes for lgb
    X_lgb = X.copy()
    for c in cat_features:
        X_lgb[c] = X_lgb[c].cat.codes

    train_data = lgb.Dataset(X_lgb, label=y, categorical_feature=cat_indices)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 7,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=800,
    )

    return model, "lightgbm"


def predict_xrv(model, model_type, X, features, cat_features):
    """Generate predictions from the trained model."""
    X_pred = X.copy()
    for c in cat_features:
        if hasattr(X_pred[c], "cat"):
            X_pred[c] = X_pred[c].cat.codes
        else:
            X_pred[c] = X_pred[c].astype(int)

    if model_type == "lightgbm":
        return model.predict(X_pred)
    else:
        return model.predict(X_pred)


def cross_validate(X, y, features, cat_features, n_folds=5):
    """Quick CV to assess model quality."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    X_cv = X.copy()
    for c in cat_features:
        X_cv[c] = X_cv[c].cat.codes

    rmses = []
    r2s = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv)):
        X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if HAS_LGB:
            cat_indices = [list(X_cv.columns).index(c) for c in cat_features]
            train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_indices)

            params = {
                "objective": "regression", "metric": "rmse",
                "learning_rate": 0.05, "num_leaves": 64, "max_depth": 7,
                "min_child_samples": 100, "subsample": 0.8,
                "colsample_bytree": 0.8, "verbose": -1,
            }
            model = lgb.train(params, train_data, num_boost_round=800)
            preds = model.predict(X_val)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=50,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

        rmse = mean_squared_error(y_val, preds) ** 0.5
        r2 = r2_score(y_val, preds)
        rmses.append(rmse)
        r2s.append(r2)

    return np.mean(rmses), np.mean(r2s)


def build_xrv_for_year(year: int, cv: bool = False):
    """Build xRV for a single season."""
    print(f"\n{'='*60}")
    print(f"Building xRV for {year}")
    print(f"{'='*60}")

    df = load_and_prepare(year)
    if df.empty:
        return

    print(f"  Loaded {len(df):,} pitches")
    print(f"  Contact events: {(df['type']=='X').sum():,}")

    # Prepare contact model data
    X, y, features, cat_features = prepare_contact_features(df)
    print(f"  Contact with features: {len(X):,}")
    print(f"  Sprint speed coverage: {X['sprint_speed'].notna().mean():.1%}")

    # Cross-validate
    if cv:
        print("  Running 5-fold CV...")
        rmse, r2 = cross_validate(X, y, features, cat_features)
        print(f"  CV RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Train on full data
    print("  Training contact xRV model...")
    model, model_type = train_xrv_model(X, y, features, cat_features)
    print(f"  Model type: {model_type}")

    # Feature importance
    if model_type == "lightgbm":
        importance = dict(zip(
            features + cat_features,
            model.feature_importance(importance_type="gain")
        ))
        total = sum(importance.values())
        print("  Feature importance (gain):")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"    {feat}: {imp/total:.1%}")

    # Generate xRV for all pitches
    print("  Generating xRV for all pitches...")

    # Start with delta_run_exp for non-contact (it's already "expected" for B/S)
    df["xrv"] = df["delta_run_exp"].copy()

    # For contact events with sufficient features, replace with model prediction
    contact_mask = (df["type"] == "X") & df["launch_speed"].notna() & df["launch_angle"].notna()
    contact_rows = df.loc[contact_mask].copy()

    if len(contact_rows) > 0:
        # Fill missing sprint speed
        league_avg_sprint = df.loc[contact_mask, "sprint_speed"].median()
        if pd.isna(league_avg_sprint):
            league_avg_sprint = 27.0
        contact_rows["sprint_speed"] = contact_rows["sprint_speed"].fillna(league_avg_sprint)
        contact_rows["park"] = contact_rows["park"].astype("category")

        X_contact = contact_rows[features + cat_features]
        preds = predict_xrv(model, model_type, X_contact, features, cat_features)
        df.loc[contact_mask, "xrv"] = preds

    # Stats
    non_null = df["xrv"].notna().sum()
    print(f"  xRV assigned: {non_null:,}/{len(df):,} ({non_null/len(df):.1%})")
    print(f"  xRV mean: {df['xrv'].mean():.6f}")
    print(f"  xRV std: {df['xrv'].std():.4f}")

    # Compare contact: actual vs expected
    contact_df = df[contact_mask]
    if len(contact_df) > 0:
        actual_mean = contact_df["delta_run_exp"].mean()
        expected_mean = contact_df["xrv"].mean()
        print(f"  Contact actual RV mean: {actual_mean:.6f}")
        print(f"  Contact xRV mean: {expected_mean:.6f}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"statcast_xrv_{year}.parquet"

    # Save key columns + xrv (not entire 118 col dataset)
    keep_cols = [
        "game_pk", "game_date", "season", "at_bat_number", "pitch_number",
        "pitcher", "batter", "home_team", "away_team",
        "inning", "inning_topbot",
        "pitch_type", "release_speed", "release_spin_rate",
        "pfx_x", "pfx_z", "plate_x", "plate_z",
        "effective_speed", "release_extension",
        "release_pos_x", "release_pos_z", "spin_axis",
        "balls", "strikes", "outs_when_up",
        "on_1b", "on_2b", "on_3b",
        "stand", "p_throws",
        "type", "description", "events",
        "launch_speed", "launch_angle", "spray_angle",
        "sprint_speed", "park",
        "bb_type", "zone",
        "delta_run_exp", "xrv",
        "game_type",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_parquet(out_path, index=False)
    print(f"  Saved to {out_path}")

    return model, model_type


def main():
    parser = argparse.ArgumentParser(description="Build xRV model")
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2017, 2026)))
    parser.add_argument("--cv", action="store_true", help="Run cross-validation")
    args = parser.parse_args()

    for year in sorted(args.seasons):
        build_xrv_for_year(year, cv=args.cv)

    print(f"\n{'='*60}")
    print("Done building xRV for all seasons.")


if __name__ == "__main__":
    main()
