#!/usr/bin/env python3
"""
build_calibration_curve.py — Build a quantile calibration curve from
out-of-fold model predictions.

The curve maps raw model scores to empirical draft rates:
    calibrated_p = empirical_rate_of_players_scored_like_this

This avoids Platt's ceiling problem (caps all elite players at ~0.85)
and isotonic's compression problem (all top players mapped to ~0.81).

Output: models/artifacts_full/calibration_curve_{pt}.json per position.

Usage:
    python3 scripts/build_calibration_curve.py
"""
import json, pickle, warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import cross_val_predict, GroupKFold
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
POSITIVES_PATH = BASE / "data" / "training" / "expanded_training_set.json"
NEGATIVES_PATH = BASE / "data" / "training" / "tier2_negatives.json"
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
CONF_STRENGTH_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"

# ── Feature definitions (must match train_tier2_full.py) ──
HITTER_ADJ = ["wOBA_adj", "OPS_adj", "AVG_adj", "SLG_adj",
              "BB_pct_adj", "K_pct_adj", "ISO_adj", "wRC_plus_adj"]
PITCHER_ADJ = ["ERA_adj", "FIP_adj", "WHIP_adj",
               "K_per_nine_adj", "BB_per_nine_adj", "K_pct_adj", "BB_pct_adj"]
HITTER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in HITTER_ADJ]
PITCHER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in PITCHER_ADJ]

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

ADJ_FEATURE_MAP = {
    "wOBA_adj": "wOBA", "OPS_adj": "OPS", "AVG_adj": "AVG", "SLG_adj": "SLG",
    "BB_pct_adj": "BB_pct", "K_pct_adj": "K_pct", "ISO_adj": "ISO", "wRC_plus_adj": "wRC_plus",
    "ERA_adj": "ERA", "FIP_adj": "FIP", "WHIP_adj": "WHIP",
    "K_per_nine_adj": "K_per_nine", "BB_per_nine_adj": "BB_per_nine",
}
INTERACTION_BASE_MAP = {
    k: v for k, v in [
        ("strength_x_wOBA", "wOBA_adj"), ("strength_x_OPS", "OPS_adj"),
        ("strength_x_AVG", "AVG_adj"), ("strength_x_SLG", "SLG_adj"),
        ("strength_x_BB_pct", "BB_pct_adj"), ("strength_x_K_pct", "K_pct_adj"),
        ("strength_x_ISO", "ISO_adj"), ("strength_x_wRC_plus", "wRC_plus_adj"),
        ("strength_x_ERA", "ERA_adj"), ("strength_x_FIP", "FIP_adj"),
        ("strength_x_WHIP", "WHIP_adj"),
        ("strength_x_K_per_nine", "K_per_nine_adj"),
        ("strength_x_BB_per_nine", "BB_per_nine_adj"),
    ]
}

HITTER_FEATURE_NAMES = T2_HITTER_FEATURES
PITCHER_FEATURE_NAMES = T2_PITCHER_FEATURES


def load_json(path):
    return json.load(open(path))


def safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def get_conf_avg(conf_stats, conf, season, ptype, stat):
    per_season = conf_stats.get("per_season", {})
    conf_ov = conf_stats.get("conference_overall", {})
    tier_fb = conf_stats.get("tier_fallback", {})
    sd = per_season.get(conf, {}).get(str(season), {})
    if isinstance(sd, dict):
        pd = sd.get(ptype, {})
        if stat in pd:
            return pd[stat]
    cd = conf_ov.get(conf, {}).get(ptype, {})
    if stat in cd:
        return cd[stat]
    return tier_fb.get("3", {}).get(ptype, {}).get(stat, 0.0)


def add_features(records, conf_stats, conf_strength):
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


def prepare(records, features, threshold=315):
    X, y, groups = [], [], []
    for r in records:
        dp = safe_float(r.get("draft_pick"))
        und = r.get("is_undrafted_verified", False)
        if und:
            yv = 0
        elif dp and dp > 0:
            yv = 1 if dp <= threshold else 0
        else:
            continue
        row = [safe_float(r.get(f)) or 0.0 for f in features]
        X.append(row)
        y.append(yv)
        pid = r.get("person_id") or hash(r.get("player_name", "")) % (10**10)
        groups.append(pid)
    return np.array(X), np.array(y), np.array(groups)


def build_curve(oof_preds, oof_true, n_bins=100):
    """Build quantile calibration curve from OOF predictions and true labels."""
    order = np.argsort(oof_preds)
    sorted_preds = oof_preds[order]
    sorted_true = oof_true[order]
    n = len(sorted_preds)
    bin_size = n // n_bins

    bins = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        bp = sorted_preds[start:end]
        bt = sorted_true[start:end]
        rate = float(np.mean(bt))
        n_in_bin = int(end - start)
        # Binomial confidence interval (Wilson score)
        z = 1.96  # 95% CI
        p_hat = rate
        denom = 1 + z**2 / n_in_bin
        center = (p_hat + z**2 / (2 * n_in_bin)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_in_bin)) / n_in_bin) / denom
        bins.append({
            "bin": i,
            "raw_min": float(np.min(bp)),
            "raw_max": float(np.max(bp)),
            "raw_mean": float(np.mean(bp)),
            "empirical_rate": rate,
            "ci_lower": max(0, center - margin),
            "ci_upper": min(1, center + margin),
            "n": n_in_bin,
            "n_pos": int(np.sum(bt)),
        })

    # Diagnostic: calibration slope, intercept
    # Regress logit(empirical) ~ logit(raw) for well-calibrated: slope=1, intercept=0
    eps = 1e-6
    logit_emp = np.array([np.log(max(b["empirical_rate"], eps) / max(1 - b["empirical_rate"], eps)) for b in bins])
    logit_raw = np.array([np.log(max(b["raw_mean"], eps) / max(1 - b["raw_mean"], eps)) for b in bins])
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(logit_raw.reshape(-1, 1), logit_emp)
    slope = float(lr.coef_[0])
    intercept = float(lr.intercept_)
    r2 = float(lr.score(logit_raw.reshape(-1, 1), logit_emp))

    return {"bins": bins, "n_bins": n_bins, "n_total": n,
            "calibration_slope": round(slope, 4),
            "calibration_intercept": round(intercept, 4),
            "calibration_r2": round(r2, 4)}


def validate_curve(curve, label):
    """Print validation diagnostics for a calibration curve."""
    bins = curve["bins"]
    print(f"\n  ── Calibration Diagnostics: {label} ──")
    print(f"    Samples: {curve['n_total']:,}  Bins: {curve['n_bins']}")
    print(f"    Slope: {curve['calibration_slope']:.4f} (1.0 = perfect)  "
          f"Intercept: {curve['calibration_intercept']:.4f} (0.0 = perfect)")
    print(f"    R²: {curve['calibration_r2']:.4f}")

    # Check monotonicity
    rates = [b["empirical_rate"] for b in bins]
    is_monotonic = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    print(f"    Monotonic: {'✅' if is_monotonic else '❌'}")

    # Check the extremes
    top5 = bins[-5:]
    bottom5 = bins[:5]
    print(f"    Bottom bins (lowest raw scores):")
    for b in bottom5:
        print(f"      raw={b['raw_mean']:.4f} → empirical={b['empirical_rate']:.4f} [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}] (n={b['n']}, pos={b['n_pos']})")
    print(f"    Top bins (highest raw scores):")
    for b in top5:
        print(f"      raw={b['raw_mean']:.4f} → empirical={b['empirical_rate']:.4f} [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}] (n={b['n']}, pos={b['n_pos']})")

    # Check: what's the max empirical rate?
    max_rate = max(b["empirical_rate"] for b in bins)
    max_raw = max(b["raw_mean"] for b in bins)
    print(f"    Max calibrated probability: {max_rate:.4f} (at raw≈{max_raw:.4f})")
    print(f"    Platt ceiling was: 0.8484 — improvement: {max_rate - 0.8484:+.4f}")


def main():
    print("=" * 60)
    print("QUANTILE CALIBRATION CURVE")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    positives = load_json(POSITIVES_PATH)
    negatives = load_json(NEGATIVES_PATH)
    conf_stats = load_json(CONF_STATS_PATH)
    conf_strength = load_json(CONF_STRENGTH_PATH)

    # Add features
    print("Adding features...")
    positives = add_features(positives, conf_stats, conf_strength)
    negatives = add_features(negatives, conf_stats, conf_strength)

    results = {}

    for pt, features, fname in [
        ("hitter", T2_HITTER_FEATURES, "calibration_curve_hitter"),
        ("pitcher", T2_PITCHER_FEATURES, "calibration_curve_pitcher"),
    ]:
        print(f"\n{'─' * 60}")
        print(f"Processing: {pt.upper()}S")
        print(f"{'─' * 60}")

        pos = [r for r in positives if r.get("player_type") == pt]
        neg = [r for r in negatives if r.get("player_type") == pt]
        all_recs = pos + neg

        X, y, groups = prepare(all_recs, features, threshold=315)
        print(f"  Samples: {len(y):,}  Features: {len(features)}  Positive rate: {100*y.sum()/len(y):.1f}%")

        # Get OOF predictions via cross_val_predict
        print("  Computing out-of-fold predictions (5-fold)...")
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1),
            random_state=42, n_jobs=-1, verbosity=0,
        )

        gkf = GroupKFold(n_splits=5)
        oof_preds = cross_val_predict(model, X, y, cv=gkf.split(X, y, groups),
                                      method="predict_proba", n_jobs=-1, verbose=1)[:, 1]
        oof_true = y

        # Build calibration curve
        curve = build_curve(oof_preds, oof_true, n_bins=100)

        # Validate
        validate_curve(curve, pt.upper())

        # Save
        path = OUTPUT_DIR / f"{fname}.json"
        with open(path, "w") as f:
            json.dump(curve, f, indent=2)
        print(f"\n  Saved: {path}")

        # Also save a lightweight version for fast lookup during inference
        lookup = {
            "raw_centers": [b["raw_mean"] for b in curve["bins"]],
            "calibrated": [b["empirical_rate"] for b in curve["bins"]],
            "ci_lower": [b["ci_lower"] for b in curve["bins"]],
            "ci_upper": [b["ci_upper"] for b in curve["bins"]],
            "n_bins": curve["n_bins"],
            "n_total": curve["n_total"],
            "calibration_slope": curve["calibration_slope"],
        }
        lookup_path = OUTPUT_DIR / f"calibration_lookup_{pt}.json"
        with open(lookup_path, "w") as f:
            json.dump(lookup, f, separators=(",", ":"))
        print(f"  Saved (lookup): {lookup_path}")

        results[pt] = curve

    # ── Summary comparison ──
    print(f"\n{'=' * 60}")
    print("CALIBRATION COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'':>15s} {'Platt Ceil':>12s} {'Quantile Top':>13s} {'Slope':>8s} {'R²':>6s}")
    print(f"{'─'*54}")
    for pt in ["hitter", "pitcher"]:
        c = results[pt]
        max_cal = max(b["empirical_rate"] for b in c["bins"])
        print(f"  {pt:<12s} {'0.8484':>12s} {max_cal:>12.4f}  "
              f"{c['calibration_slope']:>6.3f} {c['calibration_r2']:>5.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
