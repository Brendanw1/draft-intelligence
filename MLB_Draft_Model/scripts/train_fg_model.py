#!/usr/bin/env python3
"""
train_fg_model.py — Train Tier 1 draft pick model (XGBoost).

Predicts MLB draft pick number from FanGraphs college stats plus
new features (height, BMI, conference_tier, position).

Trains separate models for hitters and pitchers.
Saves model artifacts and feature importance.

Usage:
  python3 scripts/train_fg_model.py [--output-dir models/artifacts]
"""

import json, sys, os, argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "training" / "fg_training_set.json"
DEFAULT_OUTPUT = BASE / "models" / "artifacts"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def safe_float(v):
    """Convert value to float, return None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def parse_height(height_str):
    """Convert height like \"6' 4\\\"\" or \"6-4\" to inches."""
    if not height_str or height_str == "":
        return None
    try:
        s = str(height_str).strip().replace("'", "-").replace('"', "").replace(" ", "")
        parts = s.split("-")
        if len(parts) == 2:
            feet, inches = int(parts[0]), int(parts[1])
            return feet * 12 + inches
    except (ValueError, IndexError):
        pass
    return None


# ── Hitter feature columns (same as manifest + new features) ──
HITTER_FEATURES = [
    # FG college stats
    "fg_AVG", "fg_OBP", "fg_SLG", "fg_OPS", "fg_ISO",
    "fg_wOBA", "fg_wRC_plus", "fg_wRC", "fg_wRAA", "fg_wBsR",
    "fg_BB_pct", "fg_K_pct", "fg_BB/K",
    "fg_BABIP", "fg_Spd",
    "fg_G", "fg_PA", "fg_AB",
    "fg_H", "fg_1B", "fg_2B", "fg_3B", "fg_HR",
    "fg_R", "fg_RBI", "fg_BB", "fg_SO", "fg_SB", "fg_CS",
    "fg_HBP", "fg_SF", "fg_SH", "fg_GDP",
    "fg_Age",
    # New features
    "height_inches", "bmi", "conference_tier",
]

# ── Pitcher feature columns ──
PITCHER_FEATURES = [
    "fg_ERA", "fg_FIP", "fg_WHIP",
    "fg_K_pct", "fg_BB_pct", "fg_KBB",
    "fg_K_per_nine", "fg_BB_per_nine",
    "fg_AVG", "fg_BABIP", "fg_HR_per_nine",
    "fg_LOB_pct", "fg_ERA_minus_FIP", "fg_K_minus_BB_pct",
    "fg_G", "fg_GS", "fg_IP", "fg_TBF",
    "fg_H", "fg_R", "fg_ER", "fg_HR", "fg_BB", "fg_SO",
    "fg_HBP", "fg_WP", "fg_BK",
    "fg_W", "fg_L", "fg_SHO", "fg_SV", "fg_CG",
    "fg_Age",
    # New features
    "height_inches", "bmi", "conference_tier",
]


def prepare_data(records, player_type, feature_cols):
    """Convert records to feature matrix X and target vector y."""
    rows = []
    targets = []
    player_names = []
    player_ids = []

    for r in records:
        if r.get("player_type") != player_type:
            continue

        # Target: draft_pick (lower = drafted earlier = better)
        pick = safe_float(r.get("draft_pick"))
        if pick is None or pick <= 0:
            continue

        # Build feature vector
        row = []
        valid = True

        # Parse height
        height_raw = r.get("height", "")
        height_inches = parse_height(height_raw) if isinstance(height_raw, str) else height_raw

        for col in feature_cols:
            if col == "height_inches":
                val = height_inches
            elif col == "bmi":
                val = safe_float(r.get("bmi"))
                # Compute BMI from weight if not present
                if val is None:
                    weight = safe_float(r.get("weight"))
                    if weight and height_inches:
                        val = round(weight * 703 / (height_inches ** 2), 1)
            elif col == "conference_tier":
                val = safe_float(r.get("conference_tier"))
                if val is None:
                    val = 4.0  # default for unknown
            else:
                val = safe_float(r.get(col))

            if val is None or np.isnan(val):
                val = 0.0  # fill missing with 0
            row.append(val)

        rows.append(row)
        targets.append(pick)
        player_names.append(r.get("player_name", ""))
        player_ids.append(r.get("person_id", ""))

    X = np.array(rows)
    y = np.array(targets)

    return X, y, player_names, player_ids


def compute_bmi_from_weight_height(weight_lbs, height_inches):
    if not weight_lbs or not height_inches or weight_lbs <= 0 or height_inches <= 0:
        return None
    return round(weight_lbs * 703 / (height_inches ** 2), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TIER 1: DRAFT PICK PREDICTION (XGBoost)")
    print("=" * 60)

    # Load data
    print("\nLoading training data...")
    data = load_json(DATA_PATH)
    hitters = [r for r in data if r.get("player_type") == "hitter"]
    pitchers = [r for r in data if r.get("player_type") == "pitcher"]
    print(f"  Hitters: {len(hitters)}")
    print(f"  Pitchers: {len(pitchers)}")

    results = {}

    for pt, features, label in [
        ("hitter", HITTER_FEATURES, "HITTERS"),
        ("pitcher", PITCHER_FEATURES, "PITCHERS"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {label}")
        print(f"{'=' * 60}")

        records = hitters if pt == "hitter" else pitchers
        X, y, names, ids = prepare_data(data, pt, features)

        print(f"  Training samples: {len(y)}")
        print(f"  Features: {len(features)}")

        if len(y) < 50:
            print(f"  SKIP: Too few samples ({len(y)})")
            continue

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train XGBoost
        print("\n  Training XGBoost regressor...")
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print(f"\n  Performance:")
        print(f"    Test R²:  {r2:.4f}")
        print(f"    Test RMSE: {rmse:.1f} picks")
        print(f"    Test MAE:  {mae:.1f} picks")

        # Feature importance
        importance = model.feature_importances_
        feat_imp = sorted(
            zip(features, importance),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"\n  Top 10 Feature Importances:")
        for feat, imp in feat_imp[:10]:
            print(f"    {feat:<20s} {imp:.4f}")

        # Cross-validation
        cv_scores = cross_val_score(
            model, X, y, cv=5,
            scoring="r2",
        )
        print(f"\n  5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Full retrain
        print("\n  Retraining on full dataset...")
        model_full = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model_full.fit(X, y)

        # Save model
        model_path = output_dir / f"fg_draft_{pt}.json"
        model_full.save_model(str(model_path))
        print(f"  Model saved: {model_path}")

        # Feature metadata
        features_path = output_dir / f"fg_features_{pt}.json"
        with open(features_path, "w") as f:
            json.dump({
                "model_type": f"fg_draft_{pt}",
                "features": features,
                "n_train": int(len(y)),
                "n_features": int(len(features)),
                "r2_test": float(r2),
                "rmse_test": float(rmse),
                "mae_test": float(mae),
                "cv_r2_mean": float(cv_scores.mean()),
                "cv_r2_std": float(cv_scores.std()),
                "feature_importance": [
                    {"feature": feat, "importance": float(imp)}
                    for feat, imp in feat_imp
                ],
            }, f, indent=2)
        print(f"  Features saved: {features_path}")

        results[pt] = {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "cv_r2_mean": float(cv_scores.mean()),
            "n_train": int(len(y)),
        }

    # Summary
    print(f"\n{'=' * 60}")
    print("TIER 1 TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for pt, res in results.items():
        print(f"  {pt.upper()}: R²={res['r2']:.4f}  RMSE={res['rmse']:.1f}  MAE={res['mae']:.1f}  n={res['n_train']}")


if __name__ == "__main__":
    main()
