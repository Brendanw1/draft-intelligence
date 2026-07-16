#!/usr/bin/env python3
"""
train_tier2_full.py — Tier 2 classifier trained on complete D1 population.

Uses ALL 63,524 FG player-seasons with verified drafted/undrafted labels.
27x more training data than previous Tier 2 model (was 2,366).

Target: drafted in top 10 rounds (picks 1-315)
Features: FG college stats + height_inches + bmi + conference_tier

Usage:
  python3 scripts/train_tier2_full.py
"""

import json, pickle, warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
POSITIVES_PATH = BASE / "data" / "training" / "expanded_training_set.json"
NEGATIVES_PATH = BASE / "data" / "training" / "tier2_negatives.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"

# Features (same as expanded model Tier 2 features)
T2_HITTER_FEATURES = [
    "Age", "G", "AB", "PA", "H", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "SO", "HBP", "SF", "SH", "SB", "CS", "GDP",
    "AVG", "BB_pct", "K_pct", "OBP", "SLG", "OPS", "ISO", "Spd",
    "BABIP", "wOBA", "wRC_plus", "wRC", "wRAA", "wBsR", "BB/K",
    "height_inches", "bmi", "conference_tier",
]

T2_PITCHER_FEATURES = [
    "Age", "G", "GS", "CG", "SHO", "SV", "IP", "TBF",
    "H", "R", "ER", "HR", "BB", "SO", "HBP", "WP", "BK",
    "W", "L", "ERA", "WHIP", "FIP", "ERA_minus_FIP",
    "K_pct", "BB_pct", "KBB", "K_per_nine", "BB_per_nine", "HR_per_nine",
    "AVG", "BABIP", "LOB_pct", "K_minus_BB_pct",
    "height_inches", "bmi", "conference_tier",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def safe_float(v):
    if v is None: return None
    try: return float(v)
    except: return None


def prepare_classification(records, feature_cols, threshold=315):
    """Build X, y, groups from mixed positive/negative records."""
    X, y, groups = [], [], []
    for r in records:
        # Determine target
        draft_pick = safe_float(r.get("draft_pick"))
        is_undrafted = r.get("is_undrafted_verified", False)

        if is_undrafted:
            y_val = 0  # Confirmed negative
        elif draft_pick and draft_pick > 0:
            y_val = 1 if draft_pick <= threshold else 0  # Top 10 rounds or not
        else:
            continue  # Skip records we can't classify

        row = []
        for col in feature_cols:
            val = safe_float(r.get(col))
            if val is None or np.isnan(val):
                val = 0.0
            row.append(val)

        X.append(row)
        y.append(y_val)
        # Use person_id if available, else hash of player_name for grouping
        pid = r.get("person_id")
        if pid is None:
            pid = hash(r.get("player_name", "")) % (10**10)
        groups.append(pid)

    return np.array(X), np.array(y), np.array(groups)


def print_feature_importance(features, importances, top_n=15):
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print(f"\n  Top {top_n} Features:")
    for feat, imp in feat_imp[:top_n]:
        marker = " ★" if feat in ("height_inches", "bmi", "conference_tier") else ""
        print(f"    {feat:<22s} {imp:.4f}{marker}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TIER 2: FULL POPULATION CLASSIFICATION")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    positives = load_json(POSITIVES_PATH)
    negatives = load_json(NEGATIVES_PATH)
    print(f"  Positives (drafted): {len(positives):,}")
    print(f"  Negatives (undrafted): {len(negatives):,}")

    results = {}

    for pt, features, label, pos_key in [
        ("hitter", T2_HITTER_FEATURES, "HITTERS", "hitter"),
        ("pitcher", T2_PITCHER_FEATURES, "PITCHERS", "pitcher"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {label}")
        print(f"{'=' * 60}")

        # Filter to position
        pos_records = [r for r in positives if r.get("player_type") == pt]
        neg_records = [r for r in negatives if r.get("player_type") == pt]
        all_records = pos_records + neg_records

        print(f"  Positives: {len(pos_records):,}")
        print(f"  Negatives: {len(neg_records):,}")
        print(f"  Total: {len(all_records):,}")
        print(f"  Class balance: {100*len(pos_records)/len(all_records):.1f}% drafted")

        # Build feature matrix
        X, y, groups = prepare_classification(all_records, features, threshold=315)
        print(f"  Training samples: {len(y):,}")
        print(f"  Features: {len(features)}")
        print(f"  Positive rate: {100*y.sum()/len(y):.1f}%")

        # ── Player-grouped CV ──
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
                auc = roc_auc_score(y_te, y_prob)
                brier = brier_score_loss(y_te, y_prob)
                cv_auc.append(auc)
                print(f"    Fold {fold}: AUC={auc:.4f}  Brier={brier:.4f}")
            except ValueError:
                print(f"    Fold {fold}: AUC=FAIL")

        cv_auc_mean = np.mean(cv_auc) if cv_auc else 0
        cv_auc_std = np.std(cv_auc) if cv_auc else 0
        print(f"\n    CV Mean: AUC={cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

        # ── Year-out validation ──
        print(f"\n  ── Year-Out Validation ──")
        train_mask = np.array([all_records[i].get("season", 0) in (2021, 2022, 2023, 2024) for i in range(len(all_records)) if i < len(y)])
        # Need to rebuild X,y for year-out
        X_yr, y_yr, _ = prepare_classification(
            [r for r in all_records if r.get("season", 0) in (2021, 2022, 2023, 2024)],
            features, threshold=315
        )
        X_yr_test, y_yr_test, _ = prepare_classification(
            [r for r in all_records if r.get("season", 0) in (2025, 2026)],
            features, threshold=315
        )

        if len(X_yr) > 100 and len(X_yr_test) > 100:
            model_yr = xgb.XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=(len(X_yr) - y_yr.sum()) / max(y_yr.sum(), 1),
                random_state=42, n_jobs=-1, verbosity=0,
            )
            model_yr.fit(X_yr, y_yr, verbose=False)
            y_prob_yr = model_yr.predict_proba(X_yr_test)[:, 1]
            auc_yr = roc_auc_score(y_yr_test, y_prob_yr)
            print(f"    Year-out AUC: {auc_yr:.4f} (train={len(X_yr):,}, test={len(X_yr_test):,})")
        else:
            auc_yr = None
            print(f"    Year-out: insufficient data (train={len(X_yr):,}, test={len(X_yr_test):,})")

        # ── Full training ──
        print(f"\n  ── Final Training ──")
        base = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1),
            random_state=42, n_jobs=-1, verbosity=0,
        )
        base.fit(X, y)

        print_feature_importance(features, base.feature_importances_)

        # Calibrate
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

        # Save
        model_path = OUTPUT_DIR / f"tier2_full_{pt}.json"
        base.save_model(str(model_path))

        cal_path = OUTPUT_DIR / f"calibrator_platt_{pt}.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump(platt, f)

        # Feature metadata
        feat_imp = sorted(zip(features, base.feature_importances_),
                          key=lambda x: x[1], reverse=True)
        meta = {
            "model_type": f"tier2_full_{pt}",
            "features": features,
            "n_train": int(len(y)),
            "n_positives": int(y.sum()),
            "n_negatives": int(len(y) - y.sum()),
            "cv_auc_mean": float(cv_auc_mean),
            "cv_auc_std": float(cv_auc_std),
            "year_out_auc": float(auc_yr) if auc_yr else None,
            "feature_importance": [
                {"feature": f, "importance": float(i)} for f, i in feat_imp
            ],
        }
        meta_path = OUTPUT_DIR / f"tier2_full_features_{pt}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n  Model: {model_path}")
        print(f"  Calibrator: {cal_path}")
        print(f"  Metadata: {meta_path}")

        results[f"tier2_full_{pt}"] = meta

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Model':<25s} {'n':>7s} {'Pos':>7s} {'Neg':>7s} {'CV AUC':>7s}")
    print(f"  {'-'*53}")
    for key, m in results.items():
        print(f"  {key:<25s} {m['n_train']:>7,} {m['n_positives']:>7,} {m['n_negatives']:>7,} {m['cv_auc_mean']:.4f}")


if __name__ == "__main__":
    main()
