#!/usr/bin/env python3
"""
train_fg_model.py — Train Tier 1 draft pick model (XGBoost).
COMPATIBLE with infer_2026.py feature pipeline.

Predicts MLB draft pick number from FanGraphs college stats plus
new features (height, BMI, conference_strength, conference-adjusted stats).

Trains separate models for hitters and pitchers.
Saves model artifacts and feature importance.

Evaluation (audit 2026-10-29): forward by draft year. For each test year Y the model is fit on draft years < Y,
and conf_strength is refit on Tier 2 rows whose label was complete before draft Y (season <= Y - 3), so no
draft-Y outcome enters the features. The old random 80/20 split is gone. The model is trained on drafted players
only, so `proj_pick` reads "projected pick if drafted".

Usage:
  python3 scripts/train_fg_model.py [--output-dir models/artifacts]
  python3 scripts/train_fg_model.py --output-dir models/artifacts_full
"""

import argparse, json, sys, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_tier2_full as t2  # noqa: E402  (labelled rows for a point-in-time conf_strength)
from mlb_draft_dashboard.leakfree import conf_strength_table  # noqa: E402

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "training" / "fg_training_set.json"
DEFAULT_OUTPUT = BASE / "models" / "artifacts"
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
CONF_STRENGTH_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"

# ── Conference-adjusted feature helpers (same as infer_2026.py) ──
HITTER_ADJ = ["wOBA_adj", "OPS_adj", "AVG_adj", "SLG_adj",
              "BB_pct_adj", "K_pct_adj", "ISO_adj", "wRC_plus_adj"]
PITCHER_ADJ = ["ERA_adj", "FIP_adj", "WHIP_adj",
               "K_per_nine_adj", "BB_per_nine_adj", "K_pct_adj", "BB_pct_adj"]
HITTER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in HITTER_ADJ]
PITCHER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in PITCHER_ADJ]

ADJ_FEATURE_MAP = {
    "wOBA_adj": "wOBA", "OPS_adj": "OPS", "AVG_adj": "AVG", "SLG_adj": "SLG",
    "BB_pct_adj": "BB_pct", "K_pct_adj": "K_pct", "ISO_adj": "ISO", "wRC_plus_adj": "wRC_plus",
    "ERA_adj": "ERA", "FIP_adj": "FIP", "WHIP_adj": "WHIP",
    "K_per_nine_adj": "K_per_nine", "BB_per_nine_adj": "BB_per_nine",
}

# Map non-prefixed feature names to fg_ prefixed training data columns
FG_COLUMN_MAP = {
    "Age": "fg_Age", "G": "fg_G", "AB": "fg_AB", "PA": "fg_PA",
    "H": "fg_H", "1B": "fg_1B", "2B": "fg_2B", "3B": "fg_3B", "HR": "fg_HR",
    "R": "fg_R", "RBI": "fg_RBI", "BB": "fg_BB", "SO": "fg_SO",
    "HBP": "fg_HBP", "SF": "fg_SF", "SH": "fg_SH", "SB": "fg_SB", "CS": "fg_CS",
    "GDP": "fg_GDP",
    "AVG": "fg_AVG", "BB_pct": "fg_BB_pct", "K_pct": "fg_K_pct",
    "OBP": "fg_OBP", "SLG": "fg_SLG", "OPS": "fg_OPS", "ISO": "fg_ISO",
    "Spd": "fg_Spd", "BABIP": "fg_BABIP", "wOBA": "fg_wOBA",
    "wRC_plus": "fg_wRC_plus", "wRC": "fg_wRC", "wRAA": "fg_wRAA", "wBsR": "fg_wBsR",
    "BB/K": "fg_BB/K",
    # Pitcher-specific
    "ERA": "fg_ERA", "FIP": "fg_FIP", "WHIP": "fg_WHIP",
    "K_per_nine": "fg_K_per_nine", "BB_per_nine": "fg_BB_per_nine",
    "KBB": "fg_KBB", "HR_per_nine": "fg_HR_per_nine",
    "LOB_pct": "fg_LOB_pct", "ERA_minus_FIP": "fg_ERA_minus_FIP",
    "K_minus_BB_pct": "fg_K_minus_BB_pct",
    "TBF": "fg_TBF", "ER": "fg_ER", "WP": "fg_WP", "BK": "fg_BK",
    "W": "fg_W", "L": "fg_L", "SHO": "fg_SHO", "SV": "fg_SV", "CG": "fg_CG",
    "GS": "fg_GS", "IP": "fg_IP",
}

# ── Feature lists matching infer_2026.py ──
HITTER_FEATURES = [
    "Age", "G", "AB", "PA", "H", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "SO", "HBP", "SF", "SH", "SB", "CS", "GDP",
    "AVG", "BB_pct", "K_pct", "OBP", "SLG", "OPS", "ISO", "Spd",
    "BABIP", "wOBA", "wRC_plus", "wRC", "wRAA", "wBsR", "BB/K",
    "height_inches", "bmi", "conf_strength",
] + HITTER_ADJ + HITTER_INTERACTIONS

PITCHER_FEATURES = [
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


def parse_height(height_str):
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
    tier_data = tier_fb.get("3", {}).get(ptype, {})
    return tier_data.get(stat, 0.0)


def prepare_data(records, player_type, feature_cols, conf_stats=None, conf_strength_data=None):
    """Convert records to feature matrix X and target vector y.
    Compatible with infer_2026.py feature pipeline."""
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

        # Player info
        conf = r.get("conference") or ""
        season = r.get("draft_year") or 2021
        strength = conf_strength_data.get(conf, {}).get("strength", 1.0) if conf_strength_data else 1.0

        # Parse height
        height_raw = r.get("height", "")
        height_inches = parse_height(height_raw) if isinstance(height_raw, str) else height_raw

        # Build feature vector
        row = []
        for feat in feature_cols:
            if feat == "height_inches":
                val = height_inches
            elif feat == "bmi":
                val = safe_float(r.get("bmi"))
                if val is None:
                    weight = safe_float(r.get("weight"))
                    if weight and height_inches:
                        val = round(weight * 703 / (height_inches ** 2), 1)
                    else:
                        val = 0.0
            elif feat == "conf_strength":
                val = strength
            elif feat == "conference_tier":
                val = safe_float(r.get("conference_tier", 4))
                if val is None:
                    val = 4.0
            elif feat.endswith("_adj") and not feat.startswith("strength_x_"):
                # Adjusted stat: player stat - conference average
                raw_stat = ADJ_FEATURE_MAP.get(feat, feat.replace("_adj", ""))
                fg_col = FG_COLUMN_MAP.get(raw_stat)
                if fg_col:
                    raw_val = safe_float(r.get(fg_col))
                else:
                    raw_val = safe_float(r.get(raw_stat))
                if raw_val is not None and conf_stats is not None:
                    conf_avg = get_conf_avg(conf_stats, conf, season, player_type, raw_stat)
                    val = round(raw_val - conf_avg, 4)
                else:
                    val = 0.0
            elif feat.startswith("strength_x_"):
                # Interaction: conf_strength × adjusted stat
                base_adj = None
                for k, v in ADJ_FEATURE_MAP.items():
                    if f"strength_x_{v}" == feat:
                        base_adj = k
                        break
                if base_adj is None:
                    base_adj = feat.replace("strength_x_", "") + "_adj"

                raw_stat = ADJ_FEATURE_MAP.get(base_adj, base_adj.replace("_adj", ""))
                fg_col = FG_COLUMN_MAP.get(raw_stat)
                if fg_col:
                    raw_val = safe_float(r.get(fg_col))
                else:
                    raw_val = safe_float(r.get(raw_stat))
                if raw_val is not None and conf_stats is not None:
                    conf_avg = get_conf_avg(conf_stats, conf, season, player_type, raw_stat)
                    adj_val = round(raw_val - conf_avg, 4)
                    val = round(strength * adj_val, 4)
                else:
                    val = 0.0
            else:
                # Try non-prefixed first, then fg_ prefixed
                fg_col = FG_COLUMN_MAP.get(feat)
                if fg_col:
                    val = safe_float(r.get(fg_col))
                else:
                    val = safe_float(r.get(feat))
                if val is None:
                    val = 0.0

            if val is None or np.isnan(val):
                val = 0.0
            row.append(val)

        rows.append(row)
        targets.append(pick)
        player_names.append(r.get("player_name", ""))
        player_ids.append(r.get("person_id", ""))

    X = np.array(rows)
    y = np.array(targets)

    return X, y, player_names, player_ids


FORWARD_YEARS = (2024, 2025)


def make_reg():
    return xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42, n_jobs=4, verbosity=0)


def forward_backtest(data, pt, features, conf_stats, t2_rows, years=FORWARD_YEARS):
    """Train on draft years < Y, score draft year Y; conf_strength from labels complete before draft Y."""
    y2, _, s2, c2 = t2.arrays(t2_rows)
    out = []
    for Y in years:
        sel = s2 <= Y - t2.HORIZON - 1
        strength = conf_strength_table(c2[sel], y2[sel])
        tr = [r for r in data if r.get("draft_year") < Y]
        te = [r for r in data if r.get("draft_year") == Y]
        Xtr, ytr, _, _ = prepare_data(tr, pt, features, conf_stats, strength)
        Xte, yte, _, _ = prepare_data(te, pt, features, conf_stats, strength)
        pred = make_reg().fit(Xtr, ytr).predict(Xte)
        out.append({"test_year": Y, "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "mae": float(mean_absolute_error(yte, pred)),
                    "baseline_mae": float(mean_absolute_error(yte, np.full(len(yte), ytr.mean()))),
                    "r2": float(r2_score(yte, pred)),
                    "spearman_rho": float(pd.Series(pred).corr(pd.Series(yte), method="spearman"))})
    return out


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

    # Load conference data (optional — for adj stats and conf_strength)
    conf_stats = None
    conf_strength_data = None
    if CONF_STATS_PATH.exists():
        conf_stats = load_json(CONF_STATS_PATH)
        print(f"  Loaded conference stats: {len(conf_stats.get('per_season', {}))} conferences")
    if CONF_STRENGTH_PATH.exists():
        conf_strength_data = load_json(CONF_STRENGTH_PATH)
        print(f"  Loaded conf_strength: {len(conf_strength_data)} conferences")

    # Load data
    print("\nLoading training data...")
    data = load_json(DATA_PATH)
    # Also filter out 2026 (already done if we ran filter_training_no2026.py)
    data = [r for r in data if r.get("draft_year") != 2026]
    t2_rows, _ = t2.load_all()
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
        X, y, names, ids = prepare_data(data, pt, features,
                                         conf_stats=conf_stats,
                                         conf_strength_data=conf_strength_data)

        print(f"  Training samples: {len(y)}")
        print(f"  Features: {len(features)}")

        if len(y) < 50:
            print(f"  SKIP: Too few samples ({len(y)})")
            continue

        # Forward-by-draft-year backtest (replaces the random 80/20 split)
        bt = forward_backtest(data, pt, features, conf_stats, t2_rows)
        for row in bt:
            print(f"    {row['test_year']}: MAE {row['mae']:.1f} (train-mean baseline {row['baseline_mae']:.1f}) "
                  f"R2 {row['r2']:.3f} n_test {row['n_test']}")
        mae = float(np.mean([r["mae"] for r in bt]))
        r2 = float(np.mean([r["r2"] for r in bt]))

        model_full = make_reg().fit(X, y)
        model_path = output_dir / f"fg_draft_{pt}.json"
        model_full.save_model(str(model_path))
        feat_imp = sorted(zip(features, model_full.feature_importances_), key=lambda x: x[1], reverse=True)

        # Feature metadata
        features_path = output_dir / f"fg_features_{pt}.json"
        with open(features_path, "w") as f:
            json.dump({
                "model_type": f"fg_draft_{pt}",
                "features": features,
                "n_train": int(len(y)),
                "n_features": int(len(features)),
                "target": "draft pick number, drafted players only: proj_pick reads 'if drafted'",
                "mae_forward": mae,
                "r2_forward": r2,
                "forward_backtest": bt,
                "eval_note": "forward by draft year: fit on years < Y, conf_strength from labels complete before Y",
                "feature_importance": [
                    {"feature": feat, "importance": float(imp)}
                    for feat, imp in feat_imp
                ],
            }, f, indent=2)
        print(f"  Features saved: {features_path}")

        results[pt] = {"r2": r2, "mae": mae, "n_train": int(len(y))}

    # Summary
    print(f"\n{'=' * 60}")
    print("TIER 1 TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for pt, res in results.items():
        print(f"  {pt.upper()}: forward MAE={res['mae']:.1f}  R²={res['r2']:.4f}  n={res['n_train']}")


if __name__ == "__main__":
    main()
