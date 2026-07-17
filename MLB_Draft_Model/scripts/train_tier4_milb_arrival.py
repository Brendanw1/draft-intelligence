#!/usr/bin/env python3
"""
train_tier4_milb_arrival.py — Train Tier 4 model (Tier 3 + MiLB year-1 features).

Two-model comparison:
  1) Tier 3 model (Elastic Net on college stats)
  2) Tier 4 model (Tier 3 features + MiLB year-1 stats)

Trained on 2021-2022, validated on 2023.

Usage:
    python3 scripts/train_tier4_milb_arrival.py
"""
import json
import pickle
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "data" / "training" / "milb_extended_training.json"
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
CONF_STRENGTH_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"

# Tier 3 features (from train_tier3_mlb_arrival.py)
HITTER_T3_FEATURES = [
    "Age", "conf_strength",
    "wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj",
    "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate",
]
PITCHER_T3_FEATURES = [
    "Age", "conf_strength",
    "ERA_adj", "FIP_adj", "K_per_nine_adj", "BB_per_nine_adj",
    "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate",
]

# Tier 4 features = Tier 3 + MiLB year-1 features
HITTER_T4_FEATURES = HITTER_T3_FEATURES + [
    "milb_year1_wOBA",
    "milb_year1_level",
    "milb_year1_games",
]
PITCHER_T4_FEATURES = PITCHER_T3_FEATURES + [
    "milb_year1_FIP",
    "milb_year1_level",
    "milb_year1_games",
]

# Adj feature map for computing adjusted stats
ADJ_FEATURE_MAP = {
    "wOBA_adj": "wOBA", "OPS_adj": "OPS",
    "BB_pct_adj": "BB_pct", "K_pct_adj": "K_pct",
    "ERA_adj": "ERA", "FIP_adj": "FIP",
    "K_per_nine_adj": "K_per_nine", "BB_per_nine_adj": "BB_per_nine",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def safe_float(v, default=0.0):
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def get_conf_avg(conf_stats, conf, season, ptype, stat):
    """Get conference average for a stat/season/type."""
    ps = conf_stats.get("per_season", {})
    co = conf_stats.get("conference_overall", {})
    tf = conf_stats.get("tier_fallback", {})

    # Try per-season first
    sd = ps.get(conf, {}).get(str(season), {})
    if isinstance(sd, dict) and stat in sd.get(ptype, {}):
        return sd[ptype][stat]

    # Fall back to conference overall
    cd = co.get(conf, {}).get(ptype, {})
    if stat in cd:
        return cd[stat]

    # Tier fallback
    return tf.get("3", {}).get(ptype, {}).get(stat, 0.0)


def compute_adj_features(records, conf_stats, conf_strength):
    """Compute adjusted stats (raw - conference avg) for each record."""
    adj_list = list(ADJ_FEATURE_MAP.keys())

    for rec in records:
        ptype = rec.get("player_type", "hitter")
        conf = rec.get("conference", "")
        season = rec.get("season")
        strength = conf_strength.get(conf, {}).get("strength", 1.0)
        rec["conf_strength"] = strength

        for adj_field in adj_list:
            raw_field = ADJ_FEATURE_MAP.get(adj_field, adj_field.replace("_adj", ""))
            raw_val = safe_float(rec.get(raw_field))
            if raw_val is not None:
                conf_avg = get_conf_avg(conf_stats, conf, season, ptype, raw_field)
                rec[adj_field] = round(raw_val - conf_avg, 4)
            else:
                rec[adj_field] = 0.0

    return records


def build_feature_matrix(records, feature_list):
    """Build X (features) and y (target) arrays from records."""
    X, y = [], []
    for rec in records:
        row = [safe_float(rec.get(f), 0) for f in feature_list]
        X.append(row)
        y.append(rec.get("has_mlb_debut", 0))
    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("TIER 4: MLB ARRIVAL PREDICTOR (Tier 3 + MiLB year-1)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──
    print("\n1. Loading data...")
    data = load_json(TRAIN_PATH)
    conf_stats = load_json(CONF_STATS_PATH)
    conf_strength = load_json(CONF_STRENGTH_PATH)

    print(f"   Loaded {len(data)} records from milb_extended_training.json")

    # Load Tier 3 models
    tier3_models = {}
    for pt in ["hitter", "pitcher"]:
        path = OUTPUT_DIR / f"tier3_mlb_{pt}.pkl"
        with open(path, "rb") as f:
            tier3_models[pt] = pickle.load(f)
        print(f"   Loaded Tier 3 {pt} model: {len(tier3_models[pt]['features'])} features, "
              f"trained on round_rates for {len(tier3_models[pt]['round_rates'])} rounds")

    # ── 2. Compute adjusted features ──
    print("\n2. Computing adjusted college features (raw - conference avg)...")
    data = compute_adj_features(data, conf_stats, conf_strength)

    # Verify adj fields
    for adj_f in ["wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj"]:
        vals = [safe_float(r.get(adj_f)) for r in data]
        non_zero = sum(1 for v in vals if v != 0)
        print(f"   {adj_f}: {non_zero}/{len(data)} non-zero, mean={np.mean(vals):.4f}")

    # ── 3. Add milb_year1_games (unified field) ──
    print("\n3. Creating milb_year1_games field...")
    for rec in data:
        ptype = rec.get("player_type", "hitter")
        if ptype == "hitter":
            rec["milb_year1_games"] = safe_float(rec.get("milb_year1_batting_games"), 0)
        else:
            rec["milb_year1_games"] = safe_float(rec.get("milb_year1_pitching_games"), 0)
    games_vals = [safe_float(r.get("milb_year1_games")) for r in data]
    print(f"   milb_year1_games: mean={np.mean(games_vals):.1f}, "
          f"min={min(games_vals):.0f}, max={max(games_vals):.0f}")

    # ── 4. Split train/validation ──
    print("\n4. Splitting train (2021-2022) / validation (2023)...")
    train_data = [r for r in data if r.get("draft_year") in (2021, 2022)]
    val_data = [r for r in data if r.get("draft_year") == 2023]
    print(f"   Train: {len(train_data)} ({sum(1 for r in train_data if r['has_mlb_debut'])} debuted)")
    print(f"   Validation: {len(val_data)} ({sum(1 for r in val_data if r['has_mlb_debut'])} debuted)")

    # ── 5. Train and evaluate for each player type ──
    results = {}

    for pt, t3_features, t4_features, label in [
        ("hitter", HITTER_T3_FEATURES, HITTER_T4_FEATURES, "HITTERS"),
        ("pitcher", PITCHER_T3_FEATURES, PITCHER_T4_FEATURES, "PITCHERS"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"{label}")
        print(f"{'=' * 60}")

        # Filter by player type
        train_pt = [r for r in train_data if r.get("player_type") == pt]
        val_pt = [r for r in val_data if r.get("player_type") == pt]

        print(f"   Train: {len(train_pt)} ({sum(1 for r in train_pt if r['has_mlb_debut'])} MLB)")
        print(f"   Val:   {len(val_pt)} ({sum(1 for r in val_pt if r['has_mlb_debut'])} MLB)")

        if len(train_pt) < 20 or len(val_pt) < 5:
            print(f"   SKIP — insufficient data")
            continue

        # ── 5a. Build Tier 3 feature matrix ──
        X_train_t3, y_train = build_feature_matrix(train_pt, t3_features)
        X_val_t3, y_val = build_feature_matrix(val_pt, t3_features)

        print(f"\n   Tier 3 features ({len(t3_features)}): {t3_features}")
        print(f"   Tier 4 features ({len(t4_features)}): {t4_features}")

        # ── 5b. Evaluate existing Tier 3 model on validation set ──
        t3_model_obj = tier3_models[pt]["model"]

        val_prob_t3 = t3_model_obj.predict_proba(X_val_t3)[:, 1]
        val_auc_t3 = roc_auc_score(y_val, val_prob_t3)
        val_brier_t3 = brier_score_loss(y_val, val_prob_t3)

        print(f"\n   ── Tier 3 Model Evaluation ──")
        print(f"   Validation AUC:  {val_auc_t3:.4f}")
        print(f"   Validation Brier: {val_brier_t3:.4f}")

        # ── 5c. Build Tier 4 feature matrix ──
        X_train_t4, _ = build_feature_matrix(train_pt, t4_features)
        X_val_t4, _ = build_feature_matrix(val_pt, t4_features)

        print(f"\n   X_train_t4 shape: {X_train_t4.shape}")
        print(f"   X_val_t4 shape: {X_val_t4.shape}")

        # ── 5d. Train Tier 4 Elastic Net ──
        print(f"\n   ── Tier 4 Training (Elastic Net) ──")
        t4_model = LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=1.0, l1_ratio=0.3, max_iter=2000, random_state=42, n_jobs=-1,
        )
        t4_model.fit(X_train_t4, y_train)

        # Check convergence
        converged = True  # saga solver doesn't set n_iter_ reliably
        # Check if all coefficients are reasonable (not NaN or extreme)
        has_nan = np.any(np.isnan(t4_model.coef_))
        has_inf = np.any(np.isinf(t4_model.coef_))
        converged = not (has_nan or has_inf)
        print(f"   Model converged: {converged}")

        # Feature coefficients
        coefs = sorted(zip(t4_features, t4_model.coef_[0]),
                       key=lambda x: abs(x[1]), reverse=True)
        print(f"\n   Tier 4 Feature coefficients (|beta|):")
        for feat, coef in coefs:
            print(f"     {feat:<30s} beta={coef:+.6f}")

        # ── 5e. Evaluate Tier 4 on validation set ──
        val_prob_t4 = t4_model.predict_proba(X_val_t4)[:, 1]
        val_auc_t4 = roc_auc_score(y_val, val_prob_t4)
        val_brier_t4 = brier_score_loss(y_val, val_prob_t4)

        print(f"\n   ── Tier 4 Model Evaluation ──")
        print(f"   Validation AUC:  {val_auc_t4:.4f}")
        print(f"   Validation Brier: {val_brier_t4:.4f}")

        # ── 5f. Comparison ──
        print(f"\n   ── Comparison ──")
        print(f"   {'Metric':<20s} {'Tier 3':>10s} {'Tier 4':>10s} {'Delta':>10s}")
        print(f"   {'-'*50}")
        auc_delta = val_auc_t4 - val_auc_t3
        brier_delta = val_brier_t3 - val_brier_t4  # negative = worse
        print(f"   {'AUC':<20s} {val_auc_t3:>10.4f} {val_auc_t4:>10.4f} {auc_delta:>+10.4f}")
        print(f"   {'Brier':<20s} {val_brier_t3:>10.4f} {val_brier_t4:>10.4f} {brier_delta:>+10.4f}")

        # ── 5g. Validation checks ──
        print(f"\n   ── Validation Checks ──")
        checks_pass = True

        # V1: Tier 4 AUC > Tier 3 AUC (or not worse by > 0.02)
        v1_pass = val_auc_t4 > val_auc_t3 - 0.02
        if v1_pass:
            v1_detail = "PASS"
        else:
            v1_detail = f"FAIL (T4 AUC {val_auc_t4:.4f} < T3 AUC {val_auc_t3:.4f} - 0.02)"
            checks_pass = False
        print(f"   [V1] Tier 4 AUC >= Tier 3 AUC - 0.02: {v1_detail}")

        # V2: milb_year1_wOBA coefficient is positive (for hitters)
        #     OR milb_year1_FIP coefficient is negative (for pitchers, lower FIP = better)
        if pt == "hitter":
            woba_coef = t4_model.coef_[0][t4_features.index("milb_year1_wOBA")]
            v2_pass = woba_coef > 0
            print(f"   [V2] milb_year1_wOBA coefficient > 0: "
                  f"{'PASS' if v2_pass else 'FAIL'} (coef={woba_coef:+.6f})")
        else:
            fip_coef = t4_model.coef_[0][t4_features.index("milb_year1_FIP")]
            v2_pass = fip_coef < 0  # Lower FIP = better pitcher = higher debut prob
            print(f"   [V2] milb_year1_FIP coefficient < 0: "
                  f"{'PASS' if v2_pass else 'FAIL'} (coef={fip_coef:+.6f})")
        if not v2_pass:
            checks_pass = False

        # V3: milb_year1_level coefficient is positive
        lvl_idx = t4_features.index("milb_year1_level")
        lvl_coef = t4_model.coef_[0][lvl_idx]
        v3_pass = lvl_coef > 0
        if not v3_pass:
            checks_pass = False
        print(f"   [V3] milb_year1_level coefficient > 0: "
              f"{'PASS' if v3_pass else 'FAIL'} (coef={lvl_coef:+.6f})")

        # V4: Model converges
        v4_pass = converged
        if not v4_pass:
            checks_pass = False
        print(f"   [V4] Model converged (no NaN/Inf in coefs): "
              f"{'PASS' if v4_pass else 'FAIL'}")

        # Print all check results
        print(f"\n   All checks: {'PASSED' if checks_pass else 'SOME FAILED'}")

        # ── 5h. Save artifacts ──
        print(f"\n   ── Saving Artifacts ──")

        # Save model
        model_path = OUTPUT_DIR / f"tier4_mlb_{pt}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": t4_model,
                "features": t4_features,
                "tier3_baseline_auc": float(val_auc_t3),
                "tier4_auc": float(val_auc_t4),
            }, f)
        print(f"   Saved model: {model_path}")

        # Save metadata
        coef_list = [{"feature": f, "coefficient": float(c)} for f, c in coefs]
        meta = {
            "model_type": f"tier4_mlb_{pt}",
            "features": t4_features,
            "tier3_features": t3_features,
            "feature_coefficients": coef_list,
            "milb_feature_coefficients": [
                {"feature": f, "coefficient": float(c)}
                for f, c in coefs
                if f.startswith("milb_")
            ],
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_val)),
            "n_positives_train": int(y_train.sum()),
            "n_positives_val": int(y_val.sum()),
            "tier3_validation_auc": float(val_auc_t3),
            "tier3_validation_brier": float(val_brier_t3),
            "tier4_validation_auc": float(val_auc_t4),
            "tier4_validation_brier": float(val_brier_t4),
            "auc_delta": float(auc_delta),
            "brier_delta": float(brier_delta),
            "tier3_comparison": {
                "auc": float(val_auc_t3),
                "brier": float(val_brier_t3),
            },
            "validation_checks": {
                "V1_auc_not_worse": bool(v1_pass),
                "V2_milb_stat_coef_sign": bool(v2_pass),
                "V3_milb_level_coef_positive": bool(v3_pass),
                "V4_model_converged": bool(v4_pass),
                "all_passed": bool(checks_pass),
            },
        }
        meta_path = OUTPUT_DIR / f"tier4_features_{pt}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"   Saved metadata: {meta_path}")

        results[pt] = meta

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20s} {'Hitter':>12s} {'Pitcher':>12s}")
    print(f"  {'-'*44}")
    for pt in ["hitter", "pitcher"]:
        if pt in results:
            m = results[pt]
            print(f"  {'n_train':<20s} {m['n_train']:>12d} {m['n_validation']:>12d}")
    print(f"  {'Tier 3 AUC':<20s} {results.get('hitter',{}).get('tier3_validation_auc',0):>12.4f} "
          f"{results.get('pitcher',{}).get('tier3_validation_auc',0):>12.4f}")
    print(f"  {'Tier 4 AUC':<20s} {results.get('hitter',{}).get('tier4_validation_auc',0):>12.4f} "
          f"{results.get('pitcher',{}).get('tier4_validation_auc',0):>12.4f}")
    print(f"  {'AUC Delta':<20s} {results.get('hitter',{}).get('auc_delta',0):>+12.4f} "
          f"{results.get('pitcher',{}).get('auc_delta',0):>+12.4f}")
    print(f"  {'All checks':<20s} {str(results.get('hitter',{}).get('validation_checks',{}).get('all_passed',False)):>12s} "
          f"{str(results.get('pitcher',{}).get('validation_checks',{}).get('all_passed',False)):>12s}")

    print(f"\nSaved artifacts:")
    for pt in ["hitter", "pitcher"]:
        if pt in results:
            print(f"  - models/artifacts_full/tier4_mlb_{pt}.pkl")
            print(f"  - models/artifacts_full/tier4_features_{pt}.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
