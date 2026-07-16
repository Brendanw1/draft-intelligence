#!/usr/bin/env python3
"""
infer_2026.py — Run Tier 1 and Tier 2 model inference on 2026 prospects.

Loads trained models from models/artifacts_full/ and applies them to the
2026 FanGraphs D1 data joined with roster enrichment data.

Features: continuous conf_strength + conference-adjusted stats + interactions.

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
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
CONF_STRENGTH_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"


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


# ── Conference-adjusted features ──
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

INTERACTION_BASE_MAP = {
    "strength_x_wOBA": "wOBA_adj", "strength_x_OPS": "OPS_adj",
    "strength_x_AVG": "AVG_adj", "strength_x_SLG": "SLG_adj",
    "strength_x_BB_pct": "BB_pct_adj", "strength_x_K_pct": "K_pct_adj",
    "strength_x_ISO": "ISO_adj", "strength_x_wRC_plus": "wRC_plus_adj",
    "strength_x_ERA": "ERA_adj", "strength_x_FIP": "FIP_adj",
    "strength_x_WHIP": "WHIP_adj",
    "strength_x_K_per_nine": "K_per_nine_adj", "strength_x_BB_per_nine": "BB_per_nine_adj",
}


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


def get_conf_strength(conf_strength, conf):
    return conf_strength.get(conf, {}).get("strength", 1.0)


# ── Feature lists ──
# Tier 1 (position regressor) — now retrained with conf_strength + adj + interactions
HITTER_FEATURES_T1 = [
    "Age", "G", "AB", "PA", "H", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "SO", "HBP", "SF", "SH", "SB", "CS", "GDP",
    "AVG", "BB_pct", "K_pct", "OBP", "SLG", "OPS", "ISO", "Spd",
    "BABIP", "wOBA", "wRC_plus", "wRC", "wRAA", "wBsR", "BB/K",
    "height_inches", "bmi", "conf_strength",
] + HITTER_ADJ + HITTER_INTERACTIONS
PITCHER_FEATURES_T1 = [
    "Age", "G", "GS", "CG", "SHO", "SV",
    "IP", "TBF", "H", "R", "ER", "HR", "BB", "SO",
    "HBP", "WP", "BK", "W", "L",
    "ERA", "WHIP", "FIP", "ERA_minus_FIP",
    "K_pct", "BB_pct", "KBB",
    "K_per_nine", "BB_per_nine", "HR_per_nine",
    "AVG", "BABIP", "LOB_pct", "K_minus_BB_pct",
    "height_inches", "bmi", "conf_strength",
] + PITCHER_ADJ + PITCHER_INTERACTIONS

# Tier 2 uses the new conf_strength + adj stats + interactions
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

# Lower-is-better stats (for percentile direction)
LOWER_BETTER = {"K_pct", "BB_pct", "ERA", "FIP", "WHIP", "BB_per_nine",
                "HR_per_nine", "LOB_pct"}


def build_feature_row(raw_fg: dict, roster: dict, features: list,
                       conf_stats: dict = None, conf_strength_data: dict = None,
                       ptype: str = "hitter") -> list | None:
    """Build a feature vector from raw FG data + roster enrichment.

    Handles all feature types: raw stats, conf_strength, _adj stats, strength_x_ interactions.
    """
    conf = roster.get("conference", "")
    season = roster.get("season", 2026)

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
                    v = 0.0
        elif feat == "conference_tier":
            v = safe_float(roster.get("conference_tier", 4))
        elif feat == "conf_strength":
            v = get_conf_strength(conf_strength_data, conf) if conf_strength_data else 1.0
        elif feat == "Age":
            v = safe_float(raw_fg.get("Age"))
        elif feat.endswith("_adj") and not feat.startswith("strength_x_"):
            raw_stat = ADJ_FEATURE_MAP.get(feat, feat.replace("_adj", ""))
            raw_val = safe_float(raw_fg.get(raw_stat))
            if raw_val is None or conf_stats is None:
                v = 0.0
            else:
                conf_avg = get_conf_avg(conf_stats, conf, season, ptype, raw_stat)
                v = round(raw_val - conf_avg, 4)
        elif feat.startswith("strength_x_"):
            # Interaction: conf_strength × adjusted stat
            base_adj = INTERACTION_BASE_MAP.get(feat)
            if base_adj and conf_strength_data:
                strength = get_conf_strength(conf_strength_data, conf)
                # Get the adj stat value — compute it now since we may not have precomputed it
                if base_adj in raw_fg:
                    adj_val = safe_float(raw_fg.get(base_adj, 0))
                else:
                    raw_stat = ADJ_FEATURE_MAP.get(base_adj, base_adj.replace("_adj", ""))
                    raw_val = safe_float(raw_fg.get(raw_stat))
                    if raw_val is not None and conf_stats is not None:
                        conf_avg = get_conf_avg(conf_stats, conf, season, ptype, raw_stat)
                        adj_val = round(raw_val - conf_avg, 4)
                    else:
                        adj_val = 0.0
                v = round(strength * adj_val, 4)
            else:
                v = 0.0
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
        tier2_path = model_dir / f"tier2_full_{pt}.json"
        if not tier2_path.exists():
            tier2_path = model_dir / f"tier2_{pt}.json"
        if tier2_path.exists():
            model = xgb.XGBClassifier()
            model.load_model(str(tier2_path))
            models[f"tier2_{pt}"] = model
            print(f"  Tier 2 {pt}: loaded from {tier2_path.name}")

        for cal_type in ["platt", "isotonic"]:
            cal_path = model_dir / f"calibrator_{cal_type}_{pt}.pkl"
            if cal_path.exists():
                with open(cal_path, "rb") as f:
                    models[f"cal_{cal_type}_{pt}"] = pickle.load(f)
                print(f"  Calibrator ({cal_type}) {pt}: loaded")

        t1_path = model_dir / f"fg_draft_{pt}.json"
        if not t1_path.exists():
            t1_path = BASE / "models" / "artifacts" / f"fg_draft_{pt}.json"
        if t1_path.exists():
            t1_model = xgb.XGBRegressor()
            t1_model.load_model(str(t1_path))
            models[f"tier1_{pt}"] = t1_model
            print(f"  Tier 1 {pt}: loaded from {t1_path.name}")

    season = 2026

    # Load Tier 3 models (MLB arrival prediction)
    print("\nLoading Tier 3 models...")
    tier3_models = {}
    tier3_nns = {}
    for pt in ["hitter", "pitcher"]:
        t3_path = model_dir / f"tier3_mlb_{pt}.pkl"
        if t3_path.exists():
            with open(t3_path, "rb") as f:
                tier3_data = pickle.load(f)
            tier3_models[pt] = tier3_data
            print(f"  Tier 3 {pt}: loaded ({len(tier3_data['features'])} features)")

        nn_path = model_dir / f"tier3_nn_{pt}.pkl"
        if nn_path.exists():
            with open(nn_path, "rb") as f:
                tier3_nns[pt] = pickle.load(f)
            print(f"  Tier 3 NN index {pt}: loaded")

    # Pre-compute feature vectors for NN reference pool
    print("\nPreparing NN reference pool for Tier 3...")
    nn_ref_pool = {}
    for pt in ["hitter", "pitcher"]:
        if pt not in tier3_nns:
            continue
        nn_info = tier3_nns[pt]
        nn_ref_pool[pt] = {
            "nn": nn_info["nn"],
            "scaler": nn_info["scaler"],
            "sim_stats": nn_info["sim_stats"],
        }

    # Also load the historical drafted players for NN lookup — we need their
    # MLB debut status stored with the NN index. The NN index was built on
    # all_pt players, but we need to know which ones debuted.
    # The index aligns with the player order from training.
    # For inference-time NN, we only need the index — the 2026 players
    # are compared to the same reference pool.
    # Note: nn_mlb_rate is computed at training time and stored as a model coefficient.
    # During inference, we compute it fresh for each 2026 player against the reference pool.
    print("  Reference pools ready.")

    # Load conference data
    print("\nLoading conference data...")
    conf_stats = load_json(CONF_STATS_PATH) if CONF_STATS_PATH.exists() else None
    conf_strength_data = load_json(CONF_STRENGTH_PATH) if CONF_STRENGTH_PATH.exists() else None
    if conf_strength_data:
        n_conf = len(conf_strength_data)
        print(f"  conf_strength.json: {n_conf} conferences loaded")
    else:
        print("  WARNING: conference_strength.json not found — strength = 1.0 default")

    # ── 3. Run inference ──
    print(f"\nRunning inference on {len(enriched)} players...")

    # Pre-compute player types for efficient batch processing
    player_ptypes = [rec.get("player_type", "hitter") for rec in enriched]

    # Batch NN computation: build similarity vectors for all players at once
    print("  Computing NN mlb rates (batched)...")
    nn_rates_by_idx = {}
    for pt in ["hitter", "pitcher"]:
        if pt not in tier3_models or pt not in nn_ref_pool:
            continue
        nn_info = nn_ref_pool[pt]
        nn_pickled = tier3_nns[pt]
        sim_stats = nn_info["sim_stats"]
        scaler = nn_info["scaler"]
        nn_model = nn_info["nn"]
        ref_labels = nn_pickled.get("mlb_debut_labels", [])

        # Get indices of this player type
        type_indices = [i for i, t in enumerate(player_ptypes) if t == pt]
        if not type_indices:
            continue

        # Build similarity matrix for all players of this type
        sim_matrix = []
        for idx in type_indices:
            rec = enriched[idx]
            row = [safe_float(rec.get(s, 0)) or 0 for s in sim_stats]
            sim_matrix.append(row)
        sim_matrix = np.array(sim_matrix)

        # Batch normalize and predict
        sim_norm = scaler.transform(sim_matrix)
        n_neighbors = min(21, len(ref_labels))
        distances, indices = nn_model.kneighbors(sim_norm, n_neighbors=n_neighbors)

        # Compute rates and store by original index
        for j, idx in enumerate(type_indices):
            neighbor_labels = [ref_labels[k] for k in indices[j] if k < len(ref_labels)]
            nn_rate = np.mean(neighbor_labels) if neighbor_labels else 0.0
            nn_rates_by_idx[idx] = round(float(nn_rate), 4)

    print(f"  Computed NN rates for {len(nn_rates_by_idx)} players")
    for i, rec in enumerate(enriched):
        ptype = rec.get("player_type", "hitter")
        name = rec.get("player_name", "")
        team_abb = rec.get("team_abb", "").strip()
        lookup_key = f"{name.strip().lower()}|{team_abb.strip().lower()}"
        raw_fg = raw_fg_idx.get(lookup_key, {})

        features_t2 = HITTER_FEATURES if ptype == "hitter" else PITCHER_FEATURES
        features_t1 = HITTER_FEATURES_T1 if ptype == "hitter" else PITCHER_FEATURES_T1

        # Tier 2 feature vector
        feat_row = build_feature_row(raw_fg, rec, features_t2,
                                      conf_stats=conf_stats,
                                      conf_strength_data=conf_strength_data,
                                      ptype=ptype)
        X_t2 = np.array([feat_row])

        # Tier 1 feature vector (now uses conf_strength)
        feat_row_t1 = build_feature_row(raw_fg, rec, features_t1,
                                         conf_stats=conf_stats,
                                         conf_strength_data=conf_strength_data,
                                         ptype=ptype)
        X_t1 = np.array([feat_row_t1])

        # Tier 2: MLB probability
        t2_key = f"tier2_{ptype}"
        if t2_key in models:
            raw_prob = models[t2_key].predict_proba(X_t2)[0, 1]
            rec["mlb_probability"] = round(float(raw_prob), 4)

            # Platt calibration
            cal_platt_key = f"cal_platt_{ptype}"
            if cal_platt_key in models:
                platt_prob = models[cal_platt_key].predict_proba(X_t2)[0, 1]
                rec["mlb_prob_platt"] = round(float(platt_prob), 4)
            else:
                rec["mlb_prob_platt"] = round(float(raw_prob), 4)

            # Isotonic calibration (preferred — no ceiling)
            cal_iso_key = f"cal_isotonic_{ptype}"
            if cal_iso_key in models:
                iso_prob = models[cal_iso_key].predict_proba(X_t2)[0, 1]
                rec["mlb_prob_isotonic"] = round(float(iso_prob), 4)
            else:
                rec["mlb_prob_isotonic"] = round(float(raw_prob), 4)

        # Tier 1: Draft position
        t1_key = f"tier1_{ptype}"
        if t1_key in models:
            proj_pick = float(models[t1_key].predict(X_t1)[0])
            rec["projected_pick"] = round(proj_pick, 1)
            rec["projected_round"] = max(1, min(20, int(np.ceil(proj_pick / 30.75))))
        else:
            rec["projected_pick"] = rec.get("projected_pick")
            rec["projected_round"] = rec.get("projected_round")

        # ── Tier 3: MLB arrival probability ──
        t3_mlb_arrival = None
        nn_rate = nn_rates_by_idx.get(i)
        rec["nn_mlb_rate"] = nn_rate

        if ptype in tier3_models and nn_rate is not None:
            try:
                t3_info = tier3_models[ptype]
                t3_features = t3_info["features"]
                t3_model = t3_info["model"]

                # Build Tier 3 feature vector using pre-computed nn_rate
                t3_row = []
                for feat in t3_features:
                    if feat == "nn_mlb_rate":
                        t3_row.append(nn_rate)
                    elif feat == "round_logit_prior":
                        # Map projected_round to its empirical MLB debut logit
                        rnd = rec.get("projected_round", 10)
                        # Load round rates from model artifact if available (saved at training time)
                        # For now, use a default mapping based on training data
                        round_logit_map = {
                            1: -0.259, 2: -0.818, 3: -1.337, 4: -2.046, 5: -1.963,
                            6: -1.737, 7: -2.653, 8: -2.752, 9: -4.317, 10: -2.786,
                            11: -2.159, 12: -2.364, 13: -3.227, 14: -3.714, 15: -4.159,
                            16: -4.103, 17: -4.754, 18: -2.986, 19: -4.533, 20: -2.273,
                        }
                        t3_row.append(round_logit_map.get(rnd, -2.0))
                    elif feat == "conf_strength":
                        conf = rec.get("conference", "")
                        t3_row.append(get_conf_strength(conf_strength_data, conf) if conf_strength_data else 1.0)
                    elif feat == "Age":
                        t3_row.append(safe_float(rec.get("age", 21)) or 21)
                    elif feat == "height_inches":
                        v = safe_float(rec.get("height_inches"))
                        if v is None:
                            h = rec.get("height", "")
                            v = parse_height(h) if isinstance(h, str) else 0.0
                        t3_row.append(v or 0.0)
                    elif feat == "bmi":
                        t3_row.append(safe_float(rec.get("bmi", 0)) or 0.0)
                    elif feat.endswith("_adj"):
                        raw_stat = ADJ_FEATURE_MAP.get(feat, feat.replace("_adj", ""))
                        # Try raw FG first, then fall back to enriched (college_ prefixed) data
                        raw_val = safe_float(raw_fg.get(raw_stat))
                        if raw_val is None:
                            raw_val = safe_float(rec.get(f"college_{raw_stat}"))
                        if raw_val is not None and conf_stats is not None:
                            conf = rec.get("conference", "")
                            conf_avg = get_conf_avg(conf_stats, conf, 2026, ptype, raw_stat)
                            t3_row.append(round(raw_val - conf_avg, 4))
                        else:
                            t3_row.append(0.0)
                    elif feat.startswith("strength_x_"):
                        # Interaction: conf_strength × adjusted stat
                        base_adj = INTERACTION_BASE_MAP.get(feat)
                        if base_adj and conf_stats is not None and conf_strength_data is not None:
                            raw_stat = ADJ_FEATURE_MAP.get(base_adj, base_adj.replace("_adj", ""))
                            raw_val = safe_float(raw_fg.get(raw_stat))
                            if raw_val is None:
                                raw_val = safe_float(rec.get(f"college_{raw_stat}"))
                            if raw_val is not None:
                                conf = rec.get("conference", "")
                                strength = get_conf_strength(conf_strength_data, conf)
                                conf_avg = get_conf_avg(conf_stats, conf, 2026, ptype, raw_stat)
                                adj_val = round(raw_val - conf_avg, 4)
                                t3_row.append(round(strength * adj_val, 4))
                            else:
                                t3_row.append(0.0)
                        else:
                            t3_row.append(0.0)
                    else:
                        t3_row.append(0.0)

                t3_X = np.array([t3_row])
                t3_prob = t3_model.predict_proba(t3_X)[0, 1]
                t3_mlb_arrival = round(float(t3_prob), 4)
            except Exception:
                pass

        rec["mlb_arrival_prob"] = t3_mlb_arrival

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(enriched)}...")

    # ── 4. Compute composite scores and grades ──
    print("\nComputing composites and grades...")
    for rec in enriched:
        proj_pick = safe_float(rec.get("projected_pick"))
        # Use raw probability as primary (conference-adjusted, no calibration ceiling)
        mlb_p = safe_float(rec.get("mlb_probability")) or 0

        slot_score = 0
        if proj_pick is not None and proj_pick > 0:
            slot_score = max(0, 100 - (proj_pick / 620) * 100)

        mlb_score = (mlb_p or 0) * 100
        composite = slot_score * 0.4 + mlb_score * 0.6
        rec["composite_score"] = round(composite, 1)

        if proj_pick is not None:
            if proj_pick <= 150:
                rec["tier1_confidence"] = "high"
            elif proj_pick <= 300:
                rec["tier1_confidence"] = "medium"
            else:
                rec["tier1_confidence"] = "low"
        else:
            rec["tier1_confidence"] = "low"

    # Grades
    for pt in ["hitter", "pitcher"]:
        pool = [r for r in enriched if r.get("player_type") == pt and r.get("composite_score") is not None]
        composites = sorted([r["composite_score"] for r in pool], reverse=True)
        if len(composites) < 100:
            continue
        elite_th = composites[max(1, len(composites) // 100) - 1]
        high_th = composites[max(1, len(composites) // 20) - 1]
        medium_th = composites[max(1, len(composites) // 5) - 1]
        for r in pool:
            if r["composite_score"] >= elite_th:
                r["value_grade"] = "elite"
            elif r["composite_score"] >= high_th:
                r["value_grade"] = "high"
            elif r["composite_score"] >= medium_th:
                r["value_grade"] = "medium"
            else:
                r["value_grade"] = "low"

    # ── 5. Summary ──
    print(f"\n  Hitter: raw mean={np.mean([p.get('mlb_probability',0) or 0 for p in enriched if p.get('player_type')=='hitter']):.4f}")
    print(f"  Pitcher: raw mean={np.mean([p.get('mlb_probability',0) or 0 for p in enriched if p.get('player_type')=='pitcher']):.4f}")

    grade_counts = {"elite": 0, "high": 0, "medium": 0, "low": 0}
    for r in enriched:
        g = r.get("value_grade", "low")
        grade_counts[g] = grade_counts.get(g, 0) + 1
    print(f"  Grades: {json.dumps(grade_counts)}")

    # ── 6. Save ──
    print(f"\nSaving {len(enriched)} enriched records to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(enriched, f, separators=(",", ":"))

    print("\nDone.")


if __name__ == "__main__":
    main()
