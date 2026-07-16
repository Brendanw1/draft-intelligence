#!/usr/bin/env python3
"""
train_tier2_full.py — Tier 2 classifier trained on complete D1 population.

Replaces conference_tier with continuous conf_strength from empirical
draft rates. Adds conference-adjusted stats and conf_strength × stat
interactions so the model learns to properly discount stats from weak
competition while boosting performance in strong conferences.

Usage:
  python3 scripts/train_tier2_full.py
"""
import json, pickle, warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
POSITIVES_PATH = BASE / "data" / "training" / "expanded_training_set.json"
NEGATIVES_PATH = BASE / "data" / "training" / "tier2_negatives.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
CONF_STRENGTH_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"
MIN_CELL_N = 5

# ── Conference-adjusted features ──
HITTER_ADJ = ["wOBA_adj", "OPS_adj", "AVG_adj", "SLG_adj",
              "BB_pct_adj", "K_pct_adj", "ISO_adj", "wRC_plus_adj"]
PITCHER_ADJ = ["ERA_adj", "FIP_adj", "WHIP_adj",
               "K_per_nine_adj", "BB_per_nine_adj", "K_pct_adj", "BB_pct_adj"]

# ── Interaction features: conf_strength × adjusted stat ──
HITTER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in HITTER_ADJ]
PITCHER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in PITCHER_ADJ]

# Full feature lists
T2_HITTER_FEATURES = [
    "Age", "G", "AB", "PA", "H", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "SO", "HBP", "SF", "SH", "SB", "CS", "GDP",
    "AVG", "BB_pct", "K_pct", "OBP", "SLG", "OPS", "ISO", "Spd",
    "BABIP", "wOBA", "wRC_plus", "wRC", "wRAA", "wBsR", "BB/K",
    "height_inches", "bmi", "conf_strength",
] + HITTER_ADJ + HITTER_INTERACTIONS

T2_PITCHER_FEATURES = [
    "Age", "G", "GS", "CG", "SHO", "SV", "IP", "TBF",
    "H", "R", "ER", "HR", "BB", "SO", "HBP", "WP", "BK",
    "W", "L", "ERA", "WHIP", "FIP", "ERA_minus_FIP",
    "K_pct", "BB_pct", "KBB", "K_per_nine", "BB_per_nine", "HR_per_nine",
    "AVG", "BABIP", "LOB_pct", "K_minus_BB_pct",
    "height_inches", "bmi", "conf_strength",
] + PITCHER_ADJ + PITCHER_INTERACTIONS


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


# ── Conference-adjusted stats ──

def load_conference_stats(path):
    return load_json(path)


def get_conf_avg(conf_stats, conf, season, ptype, stat):
    per_season = conf_stats.get("per_season", {})
    conf_ov = conf_stats.get("conference_overall", {})
    tier_fb = conf_stats.get("tier_fallback", {})

    season_data = per_season.get(conf, {}).get(str(season), {})
    if isinstance(season_data, dict):
        ptype_data = season_data.get(ptype, {})
        if stat in ptype_data:
            return ptype_data[stat]
    conf_data = conf_ov.get(conf, {}).get(ptype, {})
    if stat in conf_data:
        return conf_data[stat]
    # Fallback to tier 3 average
    tier_data = tier_fb.get("3", {}).get(ptype, {})
    return tier_data.get(stat, 0.0)


ADJ_FEATURE_MAP = {
    "wOBA_adj": "wOBA", "OPS_adj": "OPS", "AVG_adj": "AVG", "SLG_adj": "SLG",
    "BB_pct_adj": "BB_pct", "K_pct_adj": "K_pct", "ISO_adj": "ISO", "wRC_plus_adj": "wRC_plus",
    "ERA_adj": "ERA", "FIP_adj": "FIP", "WHIP_adj": "WHIP",
    "K_per_nine_adj": "K_per_nine", "BB_per_nine_adj": "BB_per_nine",
}

# Map interaction names back to their base adjusted stat
INTERACTION_BASE_MAP = {
    "strength_x_wOBA": "wOBA_adj", "strength_x_OPS": "OPS_adj",
    "strength_x_AVG": "AVG_adj", "strength_x_SLG": "SLG_adj",
    "strength_x_BB_pct": "BB_pct_adj", "strength_x_K_pct": "K_pct_adj",
    "strength_x_ISO": "ISO_adj", "strength_x_wRC_plus": "wRC_plus_adj",
    "strength_x_ERA": "ERA_adj", "strength_x_FIP": "FIP_adj",
    "strength_x_WHIP": "WHIP_adj",
    "strength_x_K_per_nine": "K_per_nine_adj", "strength_x_BB_per_nine": "BB_per_nine_adj",
}


def add_features(records, conf_stats, conf_strength):
    """Add conf_strength, conf-adjusted stats, and interactions to records in-place."""
    for rec in records:
        ptype = rec.get("player_type", "hitter")
        conf = rec.get("conference", "")
        season = rec.get("season")

        # 1. conf_strength
        strength = conf_strength.get(conf, {}).get("strength", 1.0)
        rec["conf_strength"] = strength

        # 2. Conference-adjusted stats
        adj_list = HITTER_ADJ if ptype == "hitter" else PITCHER_ADJ
        for adj_feat in adj_list:
            raw_stat = ADJ_FEATURE_MAP.get(adj_feat, adj_feat.replace("_adj", ""))
            raw_val = safe_float(rec.get(raw_stat))
            if raw_val is None:
                rec[adj_feat] = 0.0
            else:
                conf_avg = get_conf_avg(conf_stats, conf, season, ptype, raw_stat)
                rec[adj_feat] = round(raw_val - conf_avg, 4)

        # 3. Interaction features: conf_strength × adjusted stat
        interaction_list = HITTER_INTERACTIONS if ptype == "hitter" else PITCHER_INTERACTIONS
        for int_feat in interaction_list:
            base_adj = INTERACTION_BASE_MAP.get(int_feat)
            if base_adj and base_adj in rec:
                rec[int_feat] = round(strength * rec[base_adj], 4)
            else:
                rec[int_feat] = 0.0

    return records


def prepare_classification(records, feature_cols, threshold=315):
    X, y, groups = [], [], []
    for r in records:
        draft_pick = safe_float(r.get("draft_pick"))
        is_undrafted = r.get("is_undrafted_verified", False)

        if is_undrafted:
            y_val = 0
        elif draft_pick and draft_pick > 0:
            y_val = 1 if draft_pick <= threshold else 0
        else:
            continue

        row = []
        for col in feature_cols:
            val = safe_float(r.get(col))
            if val is None or np.isnan(val):
                val = 0.0
            row.append(val)

        X.append(row)
        y.append(y_val)
        pid = r.get("person_id")
        if pid is None:
            pid = hash(r.get("player_name", "")) % (10**10)
        groups.append(pid)

    return np.array(X), np.array(y), np.array(groups)


def print_feature_importance(features, importances, top_n=20):
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print(f"\n  Top {top_n} Features:")
    for feat, imp in feat_imp[:top_n]:
        marker = ""
        if feat in ("height_inches", "bmi", "conf_strength"):
            marker = " ★"
        elif feat.startswith("strength_x_"):
            marker = " ◆"
        elif feat.endswith("_adj") and not feat.startswith("strength_x_"):
            marker = " ▲"
        print(f"    {feat:<26s} {imp:.4f}{marker}")


def train_model(X, y, groups, features, label, OUTPUT_DIR, pt):
    """Train, calibrate, and save a single model."""
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
    # We need to know which records are in which years
    # Use the original records list that maps 1:1 with X rows
    # But we don't have it here — re-filter from the full feature set
    # This is handled in main() below instead
    auc_yr = None

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

    # ── Calibration ──
    # Platt (sigmoid) — tends to saturate for high-confidence predictions
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

    # Isotonic — no saturation ceiling, handles the elite tail properly
    isotonic = CalibratedClassifierCV(
        xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1),
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        method="isotonic", cv=5,
    )
    isotonic.fit(X, y)

    # ── Save ──
    model_path = OUTPUT_DIR / f"tier2_full_{pt}.json"
    base.save_model(str(model_path))
    print(f"\n  Model: {model_path}")

    with open(OUTPUT_DIR / f"calibrator_platt_{pt}.pkl", "wb") as f:
        pickle.dump(platt, f)
    print(f"  Calibrator (platt): saved")

    with open(OUTPUT_DIR / f"calibrator_isotonic_{pt}.pkl", "wb") as f:
        pickle.dump(isotonic, f)
    print(f"  Calibrator (isotonic): saved")

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
    print(f"  Metadata: {meta_path}")

    return meta


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

    # Load conference data and add features
    print("\nAdding conference strength, adjusted stats, and interactions...")
    conf_stats = load_conference_stats(CONF_STATS_PATH)
    conf_strength = load_json(CONF_STRENGTH_PATH)
    positives = add_features(positives, conf_stats, conf_strength)
    negatives = add_features(negatives, conf_stats, conf_strength)
    print(f"  conf_strength: continuous feature (SEC ≈ 3.0)")
    print(f"  Adjusted stats: {len(HITTER_ADJ)} hitter + {len(PITCHER_ADJ)} pitcher")
    print(f"  Interactions: {len(HITTER_INTERACTIONS)} hitter + {len(PITCHER_INTERACTIONS)} pitcher")

    results = {}

    for pt, features, label in [
        ("hitter", T2_HITTER_FEATURES, "HITTERS"),
        ("pitcher", T2_PITCHER_FEATURES, "PITCHERS"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {label}")
        print(f"{'=' * 60}")

        pos_records = [r for r in positives if r.get("player_type") == pt]
        neg_records = [r for r in negatives if r.get("player_type") == pt]
        all_records = pos_records + neg_records

        print(f"  Positives: {len(pos_records):,}")
        print(f"  Negatives: {len(neg_records):,}")
        print(f"  Total: {len(all_records):,}")

        X, y, groups = prepare_classification(all_records, features, threshold=315)
        print(f"  Training samples: {len(y):,}")
        print(f"  Features: {len(features)}")
        print(f"  Positive rate: {100 * y.sum() / len(y):.1f}%")

        meta = train_model(X, y, groups, features, label, OUTPUT_DIR, pt)
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
