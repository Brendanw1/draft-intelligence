#!/usr/bin/env python3
"""
train_tier2_model.py — Train Tier 2 MLB probability model (XGBoost classifier).

Predicts the probability that a drafted college player reaches MLB, using
FG college stats plus new features (height, BMI, conference_tier).

Target: reached_mlb (binary) from tier2_training_set.json
Uses Platt scaling and isotonic calibration for well-calibrated probabilities.

Usage:
  python3 scripts/train_tier2_model.py [--output-dir models/artifacts]
"""

import json, sys, os, argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "training" / "tier2_training_set.json"
DEFAULT_OUTPUT = BASE / "models" / "artifacts"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def safe_float(v):
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


# ── Hitter features (FG stats + height + weight) ──
HITTER_FEATURES = [
    "fg_Age", "fg_G", "fg_AB", "fg_PA",
    "fg_H", "fg_1B", "fg_2B", "fg_3B", "fg_HR",
    "fg_R", "fg_RBI", "fg_BB", "fg_SO",
    "fg_HBP", "fg_SF", "fg_SH", "fg_SB", "fg_CS", "fg_GDP",
    "fg_AVG", "fg_BB_pct", "fg_K_pct",
    "fg_OBP", "fg_SLG", "fg_OPS", "fg_ISO", "fg_Spd",
    "fg_BABIP", "fg_wOBA", "fg_wRC_plus",
    "fg_wRC", "fg_wRAA", "fg_wBsR", "fg_BB/K",
    # New features
    "height_inches", "bmi", "conference_tier",
]

# ── Pitcher features ──
PITCHER_FEATURES = [
    "fg_Age", "fg_G", "fg_GS", "fg_CG", "fg_SHO", "fg_SV",
    "fg_IP", "fg_TBF",
    "fg_H", "fg_R", "fg_ER", "fg_HR", "fg_BB", "fg_SO",
    "fg_HBP", "fg_WP", "fg_BK",
    "fg_W", "fg_L",
    "fg_ERA", "fg_WHIP", "fg_FIP", "fg_ERA_minus_FIP",
    "fg_K_pct", "fg_BB_pct", "fg_KBB",
    "fg_K_per_nine", "fg_BB_per_nine", "fg_HR_per_nine",
    "fg_AVG", "fg_BABIP", "fg_LOB_pct", "fg_K_minus_BB_pct",
    # New features
    "height_inches", "bmi", "conference_tier",
]


def prepare_data(records, player_type, feature_cols):
    """Convert records to feature matrix X and binary target y."""
    rows = []
    targets = []
    player_names = []
    player_ids = []

    for r in records:
        if r.get("player_type") != player_type:
            continue

        # Target: reached_mlb
        reached = r.get("reached_mlb")
        if reached is None:
            continue
        y_val = 1 if str(reached).lower() in ("true", "1", "yes") else 0

        # Build feature vector
        row = []

        # Parse height
        height_raw = r.get("height", "")
        height_inches = parse_height(height_raw) if isinstance(height_raw, str) else height_raw

        for col in feature_cols:
            if col == "height_inches":
                val = height_inches
            elif col == "bmi":
                val = safe_float(r.get("bmi"))
                if val is None:
                    weight = safe_float(r.get("weight"))
                    if weight and height_inches:
                        val = round(weight * 703 / (height_inches ** 2), 1)
            elif col == "conference_tier":
                val = safe_float(r.get("conference_tier"))
                if val is None:
                    val = 4.0
            else:
                val = safe_float(r.get(col))

            if val is None or np.isnan(val):
                val = 0.0
            row.append(val)

        rows.append(row)
        targets.append(y_val)
        player_names.append(r.get("player_name", ""))
        player_ids.append(r.get("person_id", ""))

    X = np.array(rows)
    y = np.array(targets)

    return X, y, player_names, player_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TIER 2: MLB PROBABILITY PREDICTION (XGBoost Classifier)")
    print("=" * 60)

    # Load data
    print("\nLoading training data...")
    data = load_json(DATA_PATH)
    hitters = [r for r in data if r.get("player_type") == "hitter"]
    pitchers = [r for r in data if r.get("player_type") == "pitcher"]

    # Count target prevalence
    for label, records in [("Hitters", hitters), ("Pitchers", pitchers)]:
        n_reached = sum(1 for r in records if str(r.get("reached_mlb", "")).lower() in ("true", "1"))
        n_total = len(records)
        print(f"  {label}: {n_total} ({n_reached} reached MLB = {100 * n_reached / n_total:.1f}%)")

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
        print(f"  Class balance: {y.sum()}/{len(y)} ({(y.sum() / len(y)) * 100:.1f}%)")

        if len(y) < 50:
            print(f"  SKIP: Too few samples ({len(y)})")
            continue

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train base XGBoost classifier
        print("\n  Training XGBoost classifier...")
        base_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=(len(y) - y.sum()) / y.sum(),  # handle imbalance
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
        )

        base_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate raw model
        y_prob = base_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ll = log_loss(y_test, y_prob)

        print(f"\n  Base XGBoost Performance:")
        print(f"    AUC-ROC:     {auc:.4f}")
        print(f"    Brier Score: {brier:.4f}")

        # Platt calibration (sigmoid)
        print("\n  Applying Platt (sigmoid) calibration...")
        platt = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
        platt.fit(X, y)
        y_prob_platt = platt.predict_proba(X_test)[:, 1]
        auc_platt = roc_auc_score(y_test, y_prob_platt)
        brier_platt = brier_score_loss(y_test, y_prob_platt)

        print(f"    Platt AUC:   {auc_platt:.4f}")
        print(f"    Platt Brier: {brier_platt:.4f}")

        # Isotonic calibration
        print("  Applying isotonic calibration...")
        iso = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
        iso.fit(X, y)
        y_prob_iso = iso.predict_proba(X_test)[:, 1]
        auc_iso = roc_auc_score(y_test, y_prob_iso)
        brier_iso = brier_score_loss(y_test, y_prob_iso)

        print(f"    Isotonic AUC:   {auc_iso:.4f}")
        print(f"    Isotonic Brier: {brier_iso:.4f}")

        # Feature importance (from base model)
        importance = base_model.feature_importances_
        feat_imp = sorted(
            zip(features, importance),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"\n  Top 10 Feature Importances:")
        for feat, imp in feat_imp[:10]:
            print(f"    {feat:<20s} {imp:.4f}")

        # Cross-validation
        try:
            cv_scores = cross_val_score(
                base_model, X, y, cv=5,
                scoring="roc_auc",
            )
            print(f"\n  5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception:
            cv_scores = None
            print("\n  CV failed (small dataset or class imbalance)")

        # Retrain on full dataset
        print("\n  Retraining on full dataset...")
        final_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=(len(y) - y.sum()) / y.sum(),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
        )
        final_model.fit(X, y)

        # Save models
        model_path = output_dir / f"tier2_{pt}.json"
        final_model.save_model(str(model_path))
        print(f"  Model saved: {model_path}")

        # Calibrated versions (retrain on full data)
        for method, suffix in [("sigmoid", "platt"), ("isotonic", "isotonic")]:
            cal = CalibratedClassifierCV(
                xgb.XGBClassifier(
                    n_estimators=500, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                    reg_alpha=0.1, reg_lambda=1.0,
                    scale_pos_weight=(len(y) - y.sum()) / y.sum(),
                    random_state=42, n_jobs=-1, verbosity=0,
                ),
                method=method, cv=5,
            )
            cal.fit(X, y)

            # Save the calibrated model
            # Note: sklearn's CalibratedClassifierCV doesn't export like XGBoost,
            # so we save it via pickle via the feature metadata
            cal_path = output_dir / f"calibrator_{suffix}_{pt}.pkl"
            import pickle
            with open(cal_path, "wb") as f:
                pickle.dump(cal, f)
            print(f"  Calibrator ({method}) saved: {cal_path}")

        # Feature metadata
        features_path = output_dir / f"tier2_features_{pt}.json"
        with open(features_path, "w") as f:
            meta = {
                "model_type": f"tier2_{pt}",
                "features": features,
                "n_train": int(len(y)),
                "n_features": int(len(features)),
                "n_reached_mlb": int(y.sum()),
                "auc_base": float(auc),
                "brier_base": float(brier),
                "auc_platt": float(auc_platt),
                "brier_platt": float(brier_platt),
                "auc_isotonic": float(auc_iso),
                "brier_isotonic": float(brier_iso),
                "feature_importance": [
                    {"feature": feat, "importance": float(imp)}
                    for feat, imp in feat_imp
                ],
            }
            if cv_scores is not None:
                meta["cv_auc_mean"] = float(cv_scores.mean())
                meta["cv_auc_std"] = float(cv_scores.std())
            json.dump(meta, f, indent=2)
        print(f"  Features saved: {features_path}")

        results[pt] = {
            "auc": float(auc),
            "auc_platt": float(auc_platt),
            "auc_isotonic": float(auc_iso),
            "n_train": int(len(y)),
        }

    # Summary
    print(f"\n{'=' * 60}")
    print("TIER 2 TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for pt, res in results.items():
        print(f"  {pt.upper()}: AUC={res['auc']:.4f}  Platt={res['auc_platt']:.4f}  Iso={res['auc_isotonic']:.4f}  n={res['n_train']}")


if __name__ == "__main__":
    main()
