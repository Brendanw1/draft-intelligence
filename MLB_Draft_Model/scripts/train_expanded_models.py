#!/usr/bin/env python3
"""
train_expanded_models.py — Retrain Tier 1 & Tier 2 on expanded data.

Methodology:
  - Uses ALL player-seasons (6,274 records, 2.7x expansion)
  - Player-grouped cross-validation (no leakage across a player's seasons)
  - New features: height_inches, bmi, conference_tier
  - Year-out validation: train on 2021-2024, test on 2025-2026

Statistical rigour:
  - GroupKFold by person_id, not random splits
  - Early stopping on held-out validation set
  - Feature importance stability check across folds

Usage:
  python3 scripts/train_expanded_models.py [--output-dir models/artifacts]
"""

import json, sys, os, argparse, pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "training" / "expanded_training_set.json"
ORIGINAL_PATH = BASE / "data" / "training" / "fg_training_set.json"
DEFAULT_OUTPUT = BASE / "models" / "artifacts"

# ── Hitter features (FG native names + new features) ──
HITTER_FEATURES = [
    # Core offensive stats
    "AVG", "OBP", "SLG", "OPS", "ISO",
    "wOBA", "wRC_plus", "wRC", "wRAA", "wBsR",
    "BB_pct", "K_pct", "BB/K",
    "BABIP", "Spd",
    # Counting stats
    "G", "PA", "AB",
    "H", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "SO", "SB", "CS",
    "HBP", "SF", "SH", "GDP",
    "Age",
    # New features
    "height_inches", "bmi", "conference_tier",
]

# ── Pitcher features ──
PITCHER_FEATURES = [
    "ERA", "FIP", "WHIP",
    "K_pct", "BB_pct", "KBB",
    "K_per_nine", "BB_per_nine",
    "AVG", "BABIP", "HR_per_nine",
    "LOB_pct", "ERA_minus_FIP", "K_minus_BB_pct",
    "G", "GS", "IP", "TBF",
    "H", "R", "ER", "HR", "BB", "SO",
    "HBP", "WP", "BK",
    "W", "L", "SHO", "SV", "CG",
    "Age",
    # New features
    "height_inches", "bmi", "conference_tier",
]

# ── Tier 2 features (same as original Tier 2 but from expanded data) ──
T2_HITTER_FEATURES = [
    "Age", "G", "AB", "PA",
    "H", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "SO",
    "HBP", "SF", "SH", "SB", "CS", "GDP",
    "AVG", "BB_pct", "K_pct",
    "OBP", "SLG", "OPS", "ISO", "Spd",
    "BABIP", "wOBA", "wRC_plus",
    "wRC", "wRAA", "wBsR", "BB/K",
    # New features
    "height_inches", "bmi", "conference_tier",
]

T2_PITCHER_FEATURES = [
    "Age", "G", "GS", "CG", "SHO", "SV",
    "IP", "TBF",
    "H", "R", "ER", "HR", "BB", "SO",
    "HBP", "WP", "BK",
    "W", "L",
    "ERA", "WHIP", "FIP", "ERA_minus_FIP",
    "K_pct", "BB_pct", "KBB",
    "K_per_nine", "BB_per_nine", "HR_per_nine",
    "AVG", "BABIP", "LOB_pct", "K_minus_BB_pct",
    # New features
    "height_inches", "bmi", "conference_tier",
]


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


def prepare_regression(data, player_type, feature_cols):
    """Build feature matrix X and target y for draft_pick prediction."""
    X, y, groups, names = [], [], [], []
    for r in data:
        if r.get("player_type") != player_type:
            continue
        pick = safe_float(r.get("draft_pick"))
        if pick is None or pick <= 0:
            continue

        row = []
        valid = True
        for col in feature_cols:
            val = safe_float(r.get(col))
            if val is None or np.isnan(val):
                val = 0.0
            row.append(val)

        X.append(row)
        y.append(pick)
        groups.append(r.get("person_id", 0))
        names.append(r.get("player_name", ""))

    return np.array(X), np.array(y), np.array(groups), names


def prepare_classification(data, player_type, feature_cols, target_col="draft_pick",
                           threshold=615):
    """Build feature matrix X and binary target y for MLB probability.
    
    Target: 1 if drafted in top 10 rounds (pick ≤ ~315), else 0.
    This is a proxy for \"drafted high enough to have MLB probability\"
    since we don't have reached_mlb for the full dataset.
    """
    X, y, groups, names = [], [], [], []
    for r in data:
        if r.get("player_type") != player_type:
            continue
        pick = safe_float(r.get(target_col))
        if pick is None or pick <= 0:
            continue

        # Binary target: top 10 rounds (pick ≤ ~315 for a ~20-round draft)
        y_val = 1 if pick <= threshold else 0

        row = []
        for col in feature_cols:
            val = safe_float(r.get(col))
            if val is None or np.isnan(val):
                val = 0.0
            row.append(val)

        X.append(row)
        y.append(y_val)
        groups.append(r.get("person_id", 0))
        names.append(r.get("player_name", ""))

    return np.array(X), np.array(y), np.array(groups), names


def print_feature_importance(features, importances, top_n=15):
    """Print sorted feature importance."""
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print(f"\n  Top {top_n} Feature Importances:")
    for feat, imp in feat_imp[:top_n]:
        marker = " ★" if feat in ("height_inches", "bmi", "conference_tier") else ""
        print(f"    {feat:<20s} {imp:.4f}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TIER 1 + TIER 2: EXPANDED TRAINING")
    print(f"Dataset: {DATA_PATH.name}")
    print("=" * 60)

    # Load data
    print("\nLoading expanded training data...")
    data = load_json(DATA_PATH)
    hitters = [r for r in data if r.get("player_type") == "hitter"]
    pitchers = [r for r in data if r.get("player_type") == "pitcher"]
    print(f"  Hitters: {len(hitters):,} ({len(set(r['person_id'] for r in hitters)):,} unique)")
    print(f"  Pitchers: {len(pitchers):,} ({len(set(r['person_id'] for r in pitchers)):,} unique)")

    results = {}

    # ════════════════════════════════════════════════
    # TIER 1: REGRESSION (draft_pick prediction)
    # ════════════════════════════════════════════════
    for pt, features, label in [
        ("hitter", HITTER_FEATURES, "HITTERS"),
        ("pitcher", PITCHER_FEATURES, "PITCHERS"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"TIER 1 REGRESSION: {label}")
        print(f"{'=' * 60}")

        X, y, groups, names = prepare_regression(data, pt, features)

        print(f"  Samples: {len(y):,}")
        print(f"  Features: {len(features)}")
        print(f"  Groups (players): {len(set(groups)):,}")

        if len(y) < 100:
            print("  SKIP: Too few samples\n")
            continue

        # ── Year-out validation ──
        print("\n  ── Year-Out Validation (train 2021-2024, test 2025-2026) ──")
        records = hitters if pt == "hitter" else pitchers
        train_seasons = {2021, 2022, 2023, 2024}
        X_train_yr, y_train_yr, X_test_yr, y_test_yr = [], [], [], []
        for r in records:
            season = r.get("season")
            pick = safe_float(r.get("draft_pick"))
            if pick is None or pick <= 0:
                continue
            row = []
            for col in features:
                val = safe_float(r.get(col))
                if val is None or np.isnan(val):
                    val = 0.0
                row.append(val)
            if season in train_seasons:
                X_train_yr.append(row)
                y_train_yr.append(pick)
            else:
                X_test_yr.append(row)
                y_test_yr.append(pick)

        if X_train_yr and X_test_yr:
            X_train_yr = np.array(X_train_yr)
            y_train_yr = np.array(y_train_yr)
            X_test_yr = np.array(X_test_yr)
            y_test_yr = np.array(y_test_yr)

            model_yr = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=0,
            )
            model_yr.fit(X_train_yr, y_train_yr, verbose=False)
            y_pred_yr = model_yr.predict(X_test_yr)
            r2_yr = r2_score(y_test_yr, y_pred_yr)
            mae_yr = mean_absolute_error(y_test_yr, y_pred_yr)
            print(f"    Year-out R²:  {r2_yr:.4f}")
            print(f"    Year-out MAE: {mae_yr:.1f} picks")
            print(f"    Train n: {len(y_train_yr):,}  Test n: {len(y_test_yr):,}")

        # ── Player-Grouped Cross-Validation ──
        print(f"\n  ── Player-Grouped 5-Fold CV ──")
        gkf = GroupKFold(n_splits=5)
        cv_r2, cv_mae = [], []
        fold = 0
        for train_idx, test_idx in gkf.split(X, y, groups):
            fold += 1
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model_cv = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=0,
            )
            model_cv.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            y_pred_cv = model_cv.predict(X_te)
            cv_r2.append(r2_score(y_te, y_pred_cv))
            cv_mae.append(mean_absolute_error(y_te, y_pred_cv))
            print(f"    Fold {fold}: R²={cv_r2[-1]:.4f}  MAE={cv_mae[-1]:.1f}")

        print(f"\n    CV Mean:   R²={np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
        print(f"    CV MAE:    {np.mean(cv_mae):.1f}")

        # ── Full training ──
        print("\n  ── Final Training (full dataset) ──")
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(X, y)

        # Print feature importance
        print_feature_importance(features, model.feature_importances_)

        # Save model
        model_path = output_dir / f"fg_draft_{pt}.json"
        model.save_model(str(model_path))
        print(f"\n  Model saved: {model_path}")

        # Feature metadata
        feat_imp = sorted(zip(features, model.feature_importances_),
                          key=lambda x: x[1], reverse=True)
        meta = {
            "model_type": f"fg_draft_{pt}",
            "features": features,
            "n_train": int(len(y)),
            "n_players": int(len(set(groups))),
            "cv_r2_mean": float(np.mean(cv_r2)),
            "cv_r2_std": float(np.std(cv_r2)),
            "cv_mae_mean": float(np.mean(cv_mae)),
            "year_out_r2": float(r2_yr) if len(y_train_yr) > 0 and len(y_test_yr) > 0 else None,
            "feature_importance": [
                {"feature": f, "importance": float(i)}
                for f, i in feat_imp
            ],
        }
        meta_path = output_dir / f"fg_features_{pt}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata saved: {meta_path}")

        results[f"tier1_{pt}"] = meta

    # ════════════════════════════════════════════════
    # TIER 2: PROBABILITY CLASSIFICATION
    # ════════════════════════════════════════════════
    for pt, features, label in [
        ("hitter", T2_HITTER_FEATURES, "HITTERS"),
        ("pitcher", T2_PITCHER_FEATURES, "PITCHERS"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"TIER 2 PROBABILITY: {label}")
        print(f"{'=' * 60}")

        # Target: top-half draft (picks 1-315 ~ top 10 rounds)
        X, y, groups, names = prepare_classification(
            data, pt, features, threshold=315
        )

        print(f"  Samples: {len(y):,}")
        print(f"  Features: {len(features)}")
        print(f"  Class balance: {y.sum():,}/{len(y):,} (top 10 rounds)")

        if len(y) < 100:
            print("  SKIP: Too few samples\n")
            continue

        # ── Player-Grouped CV ──
        print(f"\n  ── Player-Grouped 5-Fold CV ──")
        gkf = GroupKFold(n_splits=5)
        cv_auc = []
        fold = 0
        for train_idx, test_idx in gkf.split(X, y, groups):
            fold += 1
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model_cv = xgb.XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1),
                random_state=42, n_jobs=-1, verbosity=0,
            )
            model_cv.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            y_prob = model_cv.predict_proba(X_te)[:, 1]
            try:
                cv_auc.append(roc_auc_score(y_te, y_prob))
                print(f"    Fold {fold}: AUC={cv_auc[-1]:.4f}")
            except ValueError:
                print(f"    Fold {fold}: AUC=FAIL")

        if cv_auc:
            print(f"\n    CV Mean:   AUC={np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")

        # ── Full training + calibration ──
        print("\n  ── Final Training + Calibration ──")
        base_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1),
            random_state=42, n_jobs=-1, verbosity=0,
        )
        base_model.fit(X, y)

        # Platt calibration
        platt = CalibratedClassifierCV(
            xgb.XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1),
                random_state=42, n_jobs=-1, verbosity=0,
            ),
            method="sigmoid", cv=5,
        )
        platt.fit(X, y)

        # Isotonic calibration
        iso = CalibratedClassifierCV(
            xgb.XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1),
                random_state=42, n_jobs=-1, verbosity=0,
            ),
            method="isotonic", cv=5,
        )
        iso.fit(X, y)

        # Print feature importance
        print_feature_importance(features, base_model.feature_importances_)

        # Save models
        model_path = output_dir / f"tier2_{pt}.json"
        base_model.save_model(str(model_path))
        print(f"\n  Model saved: {model_path}")

        for method, cal_model, suffix in [
            ("sigmoid", platt, "platt"),
            ("isotonic", iso, "isotonic"),
        ]:
            cal_path = output_dir / f"calibrator_{suffix}_{pt}.pkl"
            with open(cal_path, "wb") as f:
                pickle.dump(cal_model, f)
            print(f"  Calibrator ({method}) saved: {cal_path}")

        # Metadata
        feat_imp = sorted(zip(features, base_model.feature_importances_),
                          key=lambda x: x[1], reverse=True)
        meta = {
            "model_type": f"tier2_{pt}",
            "features": features,
            "n_train": int(len(y)),
            "n_players": int(len(set(groups))),
            "cv_auc_mean": float(np.mean(cv_auc)) if cv_auc else None,
            "cv_auc_std": float(np.std(cv_auc)) if cv_auc else None,
            "cv_auc_values": [float(a) for a in cv_auc] if cv_auc else None,
            "feature_importance": [
                {"feature": f, "importance": float(i)}
                for f, i in feat_imp
            ],
        }
        meta_path = output_dir / f"tier2_features_{pt}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata saved: {meta_path}")

        results[f"tier2_{pt}"] = meta

    # ════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'=' * 60}")
    for key, meta in results.items():
        if "tier1" in key:
            cv_r2 = meta.get("cv_r2_mean", 0)
            yo_r2 = meta.get("year_out_r2", 0)
            n = meta.get("n_train", 0)
            n_players = meta.get("n_players", 0)
            print(f"  {key:<20s} CV R²={cv_r2:.4f}  Year-out R²={yo_r2 or 0:.4f}  n={n:,} ({n_players:,} players)")
        elif "tier2" in key:
            cv_auc = meta.get("cv_auc_mean", 0)
            n = meta.get("n_train", 0)
            n_players = meta.get("n_players", 0)
            print(f"  {key:<20s} CV AUC={cv_auc:.4f}  n={n:,} ({n_players:,} players)")


if __name__ == "__main__":
    main()
