#!/usr/bin/env python3
"""
train_tier1_model.py — Retrain the Tier 1 draft-pick regressor with
conf_strength, conference-adjusted stats, and interaction features.

Predicts: draft pick number (1-620) from college stats + conference strength.

Replaces the old Tier 1 model that used the broken 4-tier conference_tier.

Usage:
    python3 scripts/train_tier1_model.py
"""
import json, pickle, warnings
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "data" / "training" / "expanded_training_set.json"
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
CONF_STRENGTH_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"

# Feature definitions — same as Tier 2
HITTER_ADJ = ["wOBA_adj", "OPS_adj", "AVG_adj", "SLG_adj",
              "BB_pct_adj", "K_pct_adj", "ISO_adj", "wRC_plus_adj"]
PITCHER_ADJ = ["ERA_adj", "FIP_adj", "WHIP_adj",
               "K_per_nine_adj", "BB_per_nine_adj", "K_pct_adj", "BB_pct_adj"]
HITTER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in HITTER_ADJ]
PITCHER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in PITCHER_ADJ]

T1_HITTER_FEATURES = [
    "Age", "G", "AB", "PA", "H", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "SO", "HBP", "SF", "SH", "SB", "CS", "GDP",
    "AVG", "BB_pct", "K_pct", "OBP", "SLG", "OPS", "ISO", "Spd",
    "BABIP", "wOBA", "wRC_plus", "wRC", "wRAA", "wBsR", "BB/K",
    "height_inches", "bmi", "conf_strength",
] + HITTER_ADJ + HITTER_INTERACTIONS

T1_PITCHER_FEATURES = [
    "Age", "G", "GS", "CG", "SHO", "SV", "IP", "TBF",
    "H", "R", "ER", "HR", "BB", "SO", "HBP", "WP", "BK",
    "W", "L", "ERA", "WHIP", "FIP", "ERA_minus_FIP",
    "K_pct", "BB_pct", "KBB", "K_per_nine", "BB_per_nine", "HR_per_nine",
    "AVG", "BABIP", "LOB_pct", "K_minus_BB_pct",
    "height_inches", "bmi", "conf_strength",
] + PITCHER_ADJ + PITCHER_INTERACTIONS

ADJ_FEATURE_MAP = {
    "wOBA_adj": "wOBA", "OPS_adj": "OPS", "AVG_adj": "AVG", "SLG_adj": "SLG",
    "BB_pct_adj": "BB_pct", "K_pct_adj": "K_pct", "ISO_adj": "ISO", "wRC_plus_adj": "wRC_plus",
    "ERA_adj": "ERA", "FIP_adj": "FIP", "WHIP_adj": "WHIP",
    "K_per_nine_adj": "K_per_nine", "BB_per_nine_adj": "BB_per_nine",
}
INTERACTION_BASE_MAP = {k: v for k, v in [
    ("strength_x_wOBA", "wOBA_adj"), ("strength_x_OPS", "OPS_adj"),
    ("strength_x_AVG", "AVG_adj"), ("strength_x_SLG", "SLG_adj"),
    ("strength_x_BB_pct", "BB_pct_adj"), ("strength_x_K_pct", "K_pct_adj"),
    ("strength_x_ISO", "ISO_adj"), ("strength_x_wRC_plus", "wRC_plus_adj"),
    ("strength_x_ERA", "ERA_adj"), ("strength_x_FIP", "FIP_adj"),
    ("strength_x_WHIP", "WHIP_adj"),
    ("strength_x_K_per_nine", "K_per_nine_adj"), ("strength_x_BB_per_nine", "BB_per_nine_adj"),
]}


def load_json(path):
    return json.load(open(path))


def safe_float(v):
    if v is None: return None
    try: return float(v)
    except: return None


def get_conf_avg(conf_stats, conf, season, ptype, stat):
    ps = conf_stats.get("per_season", {})
    co = conf_stats.get("conference_overall", {})
    tf = conf_stats.get("tier_fallback", {})
    sd = ps.get(conf, {}).get(str(season), {})
    if isinstance(sd, dict):
        pd = sd.get(ptype, {})
        if stat in pd: return pd[stat]
    cd = co.get(conf, {}).get(ptype, {})
    if stat in cd: return cd[stat]
    return tf.get("3", {}).get(ptype, {}).get(stat, 0.0)


def add_features(records, conf_stats, conf_strength):
    """Add conf_strength, adj stats, interactions to each record in-place."""
    for rec in records:
        ptype = rec.get("player_type", "hitter")
        conf = rec.get("conference", "")
        season = rec.get("season")
        strength = conf_strength.get(conf, {}).get("strength", 1.0)
        rec["conf_strength"] = strength

        adj_list = HITTER_ADJ if ptype == "hitter" else PITCHER_ADJ
        for af in adj_list:
            rs = ADJ_FEATURE_MAP.get(af, af.replace("_adj", ""))
            rv = safe_float(rec.get(rs))
            rec[af] = round(rv - get_conf_avg(conf_stats, conf, season, ptype, rs), 4) if rv is not None else 0.0

        il = HITTER_INTERACTIONS if ptype == "hitter" else PITCHER_INTERACTIONS
        for inf in il:
            ba = INTERACTION_BASE_MAP.get(inf)
            rec[inf] = round(strength * rec.get(ba, 0), 4) if ba else 0.0
    return records


def build_dataset(records, features, target_col="draft_pick"):
    """Build X, y, groups from records, grouping by player (use latest season)."""
    # Group by person_id -> take latest season record for each player
    player_groups = defaultdict(list)
    for r in records:
        pid = r.get("person_id")
        if pid is None:
            pid = hash(r.get("player_name", "")) % (10**10)
        player_groups[pid].append(r)

    X, y, groups, names = [], [], [], []
    for pid, recs in player_groups.items():
        # Take the most recent season
        latest = max(recs, key=lambda x: x.get("season", 0) or 0)
        dp = safe_float(latest.get(target_col))
        if dp is None or dp <= 0:
            continue  # Skip undrafted

        row = [safe_float(latest.get(f)) or 0.0 for f in features]
        X.append(row)
        y.append(dp)
        groups.append(pid)
        names.append(latest.get("player_name", "?"))

    return np.array(X), np.array(y), np.array(groups), names


def main():
    print("=" * 60)
    print("TIER 1: DRAFT PICK REGRESSOR (RETRAIN)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    train = load_json(TRAIN_PATH)
    conf_stats = load_json(CONF_STATS_PATH)
    conf_strength = load_json(CONF_STRENGTH_PATH)

    # Add features
    print("Adding features...")
    train = add_features(train, conf_stats, conf_strength)

    results = {}
    for pt, features, label in [
        ("hitter", T1_HITTER_FEATURES, "HITTERS"),
        ("pitcher", T1_PITCHER_FEATURES, "PITCHERS"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {label}")
        print(f"{'=' * 60}")

        pos = [r for r in train if r.get("player_type") == pt]
        print(f"  Player-seasons: {len(pos)}")

        X, y, groups, names = build_dataset(pos, features)
        print(f"  Unique players: {len(y)}")
        print(f"  Features: {len(features)}")
        print(f"  Pick range: {int(min(y))}-{int(max(y))}, mean={np.mean(y):.0f}")

        # ── Player-grouped CV ──
        print(f"\n  ── Player-Grouped 5-Fold CV ──")
        gkf = GroupKFold(n_splits=5)
        cv_mae = []
        fold = 0
        for train_idx, test_idx in gkf.split(X, y, groups):
            fold += 1
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model = xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=0,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            y_pred = model.predict(X_te)
            mae = mean_absolute_error(y_te, y_pred)
            r2 = r2_score(y_te, y_pred)
            cv_mae.append(mae)
            print(f"    Fold {fold}: MAE={mae:.1f}  R²={r2:.3f}")

        cv_mae_mean = np.mean(cv_mae)
        cv_mae_std = np.std(cv_mae)
        print(f"\n    CV Mean MAE: {cv_mae_mean:.1f} ± {cv_mae_std:.1f}")

        # ── Full training ──
        print(f"\n  ── Final Training ──")
        final = xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        final.fit(X, y)

        # Feature importance
        feat_imp = sorted(zip(features, final.feature_importances_),
                          key=lambda x: x[1], reverse=True)
        print(f"\n  Top 15 Features:")
        for feat, imp in feat_imp[:15]:
            marker = ""
            if feat in ("height_inches", "bmi", "conf_strength"): marker = " ★"
            elif feat.startswith("strength_x_"): marker = " ◆"
            elif feat.endswith("_adj") and not feat.startswith("strength_x_"): marker = " ▲"
            print(f"    {feat:<26s} {imp:.4f}{marker}")

        # Save
        model_path = OUTPUT_DIR / f"fg_draft_{pt}.json"
        final.save_model(str(model_path))
        print(f"\n  Saved: {model_path}")

        # Metadata
        meta = {
            "model_type": f"tier1_{pt}",
            "features": features,
            "n_train": int(len(y)),
            "pick_range": [int(min(y)), int(max(y))],
            "cv_mae_mean": float(cv_mae_mean),
            "cv_mae_std": float(cv_mae_std),
            "feature_importance": [
                {"feature": f, "importance": float(i)} for f, i in feat_imp
            ],
        }
        meta_path = OUTPUT_DIR / f"tier1_features_{pt}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata: {meta_path}")

        results[pt] = meta

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Model':<15s} {'n':>6s} {'CV MAE':>8s}")
    for pt, m in results.items():
        print(f"  {pt:<15s} {m['n_train']:>6d} {m['cv_mae_mean']:.1f} ± {m['cv_mae_std']:.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
