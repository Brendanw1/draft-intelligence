#!/usr/bin/env python3
"""
infer_2026.py — Run Tier 1 and Tier 2 model inference on 2026 prospects.

Loads trained models from models/artifacts_full/ and applies them to the
2026 FanGraphs D1 data joined with roster enrichment data. Writes updated
projections including mlb_prob_platt, mlb_probability, mlb_prob_isotonic,
projected_pick, projected_round, composite_score, and value_grade.

Usage:
  python3 scripts/infer_2026.py
  python3 scripts/infer_2026.py --model-dir models/artifacts_full
"""

import json, sys, os, pickle, warnings
from pathlib import Path

import numpy as np
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = BASE / "models" / "artifacts_full"
ENRICHED_PATH = BASE / "data" / "training" / "projections_2026_enriched.json"
RAW_BATTERS_PATH = BASE / "data" / "fangraphs" / "raw" / "batters_2026.json"
RAW_PITCHERS_PATH = BASE / "data" / "fangraphs" / "raw" / "pitchers_2026.json"
OUTPUT_PATH = BASE / "data" / "training" / "projections_2026_enriched.json"


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


# ── Hitter features (full-population model) ──
HITTER_FEATURES = [
    "Age", "G", "AB", "PA", "H", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "SO", "HBP", "SF", "SH", "SB", "CS", "GDP",
    "AVG", "BB_pct", "K_pct", "OBP", "SLG", "OPS", "ISO",
    "Spd", "BABIP", "wOBA", "wRC_plus", "wRC", "wRAA", "wBsR", "BB/K",
    "height_inches", "bmi", "conference_tier",
]

PITCHER_FEATURES = [
    "Age", "G", "GS", "CG", "SHO", "SV",
    "IP", "TBF", "H", "R", "ER", "HR", "BB", "SO",
    "HBP", "WP", "BK", "W", "L",
    "ERA", "WHIP", "FIP", "ERA_minus_FIP",
    "K_pct", "BB_pct", "KBB",
    "K_per_nine", "BB_per_nine", "HR_per_nine",
    "AVG", "BABIP", "LOB_pct", "K_minus_BB_pct",
    "height_inches", "bmi", "conference_tier",
]

# Lower-is-better stats (for percentile direction)
LOWER_BETTER = {"K_pct", "BB_pct", "ERA", "FIP", "WHIP", "BB_per_nine",
                "HR_per_nine", "LOB_pct"}


def build_feature_row(raw_fg: dict, roster: dict, features: list) -> list | None:
    """Build a feature vector from raw FG data + roster enrichment."""
    row = []
    for feat in features:
        if feat == "height_inches":
            v = safe_float(roster.get("height_inches"))
            if v is None:
                h = roster.get("height", "")
                v = parse_height(h) if isinstance(h, str) else None
        elif feat == "bmi":
            v = safe_float(roster.get("bmi"))
            if v is None:
                height = safe_float(roster.get("height_inches"))
                weight = safe_float(roster.get("weight_lbs"))
                if weight and height:
                    v = round(weight * 703 / (height ** 2), 1)
                else:
                    v = 0.0  # fallback
        elif feat == "conference_tier":
            v = safe_float(roster.get("conference_tier", 4))
        elif feat == "Age":
            v = safe_float(raw_fg.get("Age"))
        else:
            v = safe_float(raw_fg.get(feat))
        if v is None or np.isnan(v):
            v = 0.0
        row.append(v)
    return row


def main():
    model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--model-dir" else DEFAULT_MODEL_DIR

    print("=" * 60)
    print("2026 INFERENCE — Full-Population Models")
    print("=" * 60)

    # ── 1. Load data ──
    print("\nLoading data...")
    enriched = load_json(ENRICHED_PATH)
    raw_batters = load_json(RAW_BATTERS_PATH)["data"]
    raw_pitchers = load_json(RAW_PITCHERS_PATH)["data"]

    # Index raw FG by {player_name|team_abb}
    raw_fg_idx = {}
    for rec in raw_batters + raw_pitchers:
        key = f"{rec.get('Player', '').strip().lower()}|{rec.get('team_name_abb', '').strip().lower()}"
        raw_fg_idx[key] = rec

    print(f"  Enriched records: {len(enriched)}")
    print(f"  Raw FG batters:  {len(raw_batters)}")
    print(f"  Raw FG pitchers: {len(raw_pitchers)}")

    # ── 2. Load models ──
    print("\nLoading models...")

    models = {}
    for pt in ["hitter", "pitcher"]:
        # Tier 2 — MLB probability (full-population)
        tier2_path = model_dir / f"tier2_full_{pt}.json"
        if not tier2_path.exists():
            tier2_path = model_dir / f"tier2_{pt}.json"
        if tier2_path.exists():
            model = xgb.XGBClassifier()
            model.load_model(str(tier2_path))
            models[f"tier2_{pt}"] = model
            print(f"  Tier 2 {pt}: loaded from {tier2_path.name}")

        # Calibrators
        for cal_type in ["platt", "isotonic"]:
            cal_path = model_dir / f"calibrator_{cal_type}_{pt}.pkl"
            if cal_path.exists():
                with open(cal_path, "rb") as f:
                    models[f"cal_{cal_type}_{pt}"] = pickle.load(f)
                print(f"  Calibrator ({cal_type}) {pt}: loaded")

        # Tier 1 — draft position
        t1_path = model_dir / f"fg_draft_{pt}.json"
        # Also check the base artifacts directory
        if not t1_path.exists():
            t1_path = BASE / "models" / "artifacts" / f"fg_draft_{pt}.json"
        if t1_path.exists():
            t1_model = xgb.XGBRegressor()
            t1_model.load_model(str(t1_path))
            models[f"tier1_{pt}"] = t1_model
            print(f"  Tier 1 {pt}: loaded from {t1_path.name}")

    # Determine inference year and season from data
    season = 2026

    # ── 3. Run inference ──
    print(f"\nRunning inference on {len(enriched)} players...")

    hitter_raw_probs = []
    hitter_cal_probs = []
    pitcher_raw_probs = []
    pitcher_cal_probs = []

    for i, rec in enumerate(enriched):
        ptype = rec.get("player_type", "hitter")
        name = rec.get("player_name", "")
        team_abb = rec.get("team_abb", "").strip()
        lookup_key = f"{name.strip().lower()}|{team_abb.strip().lower()}"
        raw_fg = raw_fg_idx.get(lookup_key, {})

        features = HITTER_FEATURES if ptype == "hitter" else PITCHER_FEATURES
        feat_row = build_feature_row(raw_fg, rec, features)
        X = np.array([feat_row])

        # Tier 2: MLB probability (full-population)
        t2_key = f"tier2_{ptype}"
        if t2_key in models:
            raw_prob = models[t2_key].predict_proba(X)[0, 1]
            rec["mlb_probability"] = round(float(raw_prob), 4)

            # Platt calibration
            cal_platt_key = f"cal_platt_{ptype}"
            if cal_platt_key in models:
                platt_prob = models[cal_platt_key].predict_proba(X)[0, 1]
                rec["mlb_prob_platt"] = round(float(platt_prob), 4)
            else:
                rec["mlb_prob_platt"] = round(float(raw_prob), 4)

            # Isotonic calibration
            cal_iso_key = f"cal_isotonic_{ptype}"
            if cal_iso_key in models:
                iso_prob = models[cal_iso_key].predict_proba(X)[0, 1]
                rec["mlb_prob_isotonic"] = round(float(iso_prob), 4)
            else:
                rec["mlb_prob_isotonic"] = round(float(raw_prob), 4)

            if ptype == "hitter":
                hitter_raw_probs.append(raw_prob)
                hitter_cal_probs.append(rec.get("mlb_prob_platt", raw_prob))
            else:
                pitcher_raw_probs.append(raw_prob)
                pitcher_cal_probs.append(rec.get("mlb_prob_platt", raw_prob))

        # Tier 1: Draft position
        t1_key = f"tier1_{ptype}"
        if t1_key in models:
            proj_pick = float(models[t1_key].predict(X)[0])
            rec["projected_pick"] = round(proj_pick, 1)
            rec["projected_round"] = max(1, min(20, int(np.ceil(proj_pick / 30.75))))
        else:
            rec["projected_pick"] = rec.get("projected_pick")
            rec["projected_round"] = rec.get("projected_round")

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(enriched)}...")

    # ── 4. Compute composite scores and grades ──
    print("\nComputing composites and grades...")

    # Composite: 40% slot value of projected round + 60% calibrated MLB%
    for rec in enriched:
        ptype = rec.get("player_type", "hitter")
        proj_pick = safe_float(rec.get("projected_pick"))
        mlb_p = safe_float(rec.get("mlb_prob_platt"))

        # Slot value: pick 1 = 100, pick 600 = 0, scaled inverse-linearly
        slot_score = 0
        if proj_pick is not None and proj_pick > 0:
            slot_score = max(0, 100 - (proj_pick / 620) * 100)

        # MLB% score: scale to 0-100 linearly
        mlb_score = (mlb_p or 0) * 100

        # Composite: 40% slot + 60% MLB%
        composite = slot_score * 0.4 + mlb_score * 0.6
        rec["composite_score"] = round(composite, 1)

        # Confidence tier for Tier 1 (based on projected pick depth)
        if proj_pick is not None:
            if proj_pick <= 150:
                rec["tier1_confidence"] = "high"
            elif proj_pick <= 300:
                rec["tier1_confidence"] = "medium"
            else:
                rec["tier1_confidence"] = "low"
        else:
            rec["tier1_confidence"] = "low"

    # Grades: threshold-based within qualified players per type
    qualified = [
        r for r in enriched
        if not (
            r.get("player_type") == "hitter"
            and (r.get("college_BB_pct") is not None and safe_float(r.get("college_PA", 0)) or 0 < 50)
        )
    ]
    # Actually use mlb probability as a rough filter for qualified
    for pt in ["hitter", "pitcher"]:
        pool = [r for r in enriched if r.get("player_type") == pt and r.get("composite_score") is not None]
        composites = sorted([r["composite_score"] for r in pool], reverse=True)
        if len(composites) < 100:
            continue
        elite_n = max(1, len(composites) // 100)
        high_n = max(1, len(composites) // 20)
        medium_n = max(1, len(composites) // 5)
        elite_th = composites[elite_n - 1]
        high_th = composites[high_n - 1]
        medium_th = composites[medium_n - 1]

        for r in pool:
            if r["composite_score"] >= elite_th:
                r["value_grade"] = "elite"
            elif r["composite_score"] >= high_th:
                r["value_grade"] = "high"
            elif r["composite_score"] >= medium_th:
                r["value_grade"] = "medium"
            else:
                r["value_grade"] = "low"

    # ── 5. Summary statistics ──
    print(f"\n  Hitter: raw mean={np.mean(hitter_raw_probs):.4f}, cal mean={np.mean(hitter_cal_probs):.4f}")
    print(f"  Pitcher: raw mean={np.mean(pitcher_raw_probs):.4f}, cal mean={np.mean(pitcher_cal_probs):.4f}")

    grade_counts = {"elite": 0, "high": 0, "medium": 0, "low": 0}
    for r in enriched:
        g = r.get("value_grade", "low")
        grade_counts[g] = grade_counts.get(g, 0) + 1
    print(f"  Grades: {json.dumps(grade_counts)}")
    print(f"  Mean projected pick: {np.mean([safe_float(r.get('projected_pick', 0)) or 0 for r in enriched]):.1f}")

    # ── 6. Save ──
    print(f"\nSaving {len(enriched)} enriched records to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(enriched, f, separators=(",", ":"))

    print("\nDone.")


if __name__ == "__main__":
    main()
