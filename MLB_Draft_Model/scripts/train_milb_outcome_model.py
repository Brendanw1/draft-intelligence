#!/usr/bin/env python3
"""
train_milb_outcome_model.py — Train standalone MiLB Outcome Model.

Predicts continuous MiLB performance (peak wOBA/FIP) from college stats
using Elastic Net regression.  Trains separate models for hitters and pitchers.

Usage:
    python3 scripts/train_milb_outcome_model.py
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
TRAINING_PATH = BASE / "data" / "training" / "milb_extended_training.json"
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"

# ── Conference-adjusted feature definitions ──
HITTER_ADJ_FEATURES = ["wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj"]
HITTER_BASE_FEATURES = ["Age", "conf_strength", "height_inches", "bmi", "draft_round"]
HITTER_RAW_MAP = {"wOBA_adj": "wOBA", "OPS_adj": "OPS",
                  "BB_pct_adj": "BB_pct", "K_pct_adj": "K_pct"}
HITTER_TARGET = "milb_peak_wOBA"

# Pitchers: use strongest raw features. Conference-adjusted stats destroy the
# already-weak signal (r<0.2 with target). conf_strength separately captures
# difficulty. Use fixed alpha to prevent ElasticNetCV from over-regularizing.
PITCHER_FEATURES = ["Age", "conf_strength", "draft_round",
                    "BB_per_nine", "K_per_nine"]
PITCHER_ADJ_MAP = {"ERA_adj": "ERA", "FIP_adj": "FIP",
                   "K_per_nine_adj": "K_per_nine", "BB_per_nine_adj": "BB_per_nine"}
PITCHER_TARGET = "milb_peak_FIP"


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


def get_conf_avg(conf_stats, conf, season, ptype, stat):
    """Get conference average for a stat, with fallbacks."""
    per_season = conf_stats.get("per_season", {})
    conf_ov = conf_stats.get("conference_overall", {})
    tier_fb = conf_stats.get("tier_fallback", {})

    # Try per-season first
    season_data = per_season.get(conf, {}).get(str(season), {})
    if isinstance(season_data, dict):
        ptype_data = season_data.get(ptype, {})
        if stat in ptype_data:
            return ptype_data[stat]

    # Fallback to conference overall
    conf_data = conf_ov.get(conf, {}).get(ptype, {})
    if stat in conf_data:
        return conf_data[stat]

    # Fallback to tier 3 average
    tier_data = tier_fb.get("3", {}).get(ptype, {})
    return tier_data.get(stat, 0.0)


def compute_adj_features(records, conf_stats):
    """Compute conference-adjusted stats in-place for all records."""
    for rec in records:
        ptype = rec.get("player_type", "hitter")
        season = rec.get("season")
        conf = rec.get("conference", "")

        adj_map = HITTER_RAW_MAP if ptype == "hitter" else PITCHER_ADJ_MAP

        for adj_feat, raw_feat in adj_map.items():
            raw_val = safe_float(rec.get(raw_feat))
            if raw_val is None:
                rec[adj_feat] = 0.0
            else:
                conf_avg = get_conf_avg(conf_stats, conf, season, ptype, raw_feat)
                rec[adj_feat] = round(raw_val - conf_avg, 6)
    return records


def build_feature_matrix(records, base_features, adj_features, target):
    """Build X, y from a list of records."""
    all_features = base_features + adj_features
    X, y, ids = [], [], []
    for rec in records:
        row = []
        ok = True
        for feat in all_features:
            val = safe_float(rec.get(feat))
            if val is None or np.isnan(val):
                ok = False
                break
            row.append(val)
        if not ok:
            continue
        y_val = safe_float(rec.get(target))
        if y_val is None or np.isnan(y_val):
            continue
        X.append(row)
        y.append(y_val)
        ids.append(rec.get("person_id", 0))
    return np.array(X), np.array(y), np.array(ids), all_features


def train_and_evaluate(X_train, y_train, X_val, y_val, feature_names, player_type, output_dir):
    """Train ElasticNetCV with 5-fold CV, evaluate, save artifacts."""
    print(f"\n{'='*70}")
    print(f"Training MiLB Outcome Model — {player_type.upper()}")
    print(f"{'='*70}")
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Target mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"  Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")

    # ── 5-fold cross-validation ──
    print(f"\n  ── 5-Fold Cross-Validation ──")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = []
    cv_rmse_scores = []
    cv_mae_scores = []
    baseline_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]

        # Baseline: predict training mean
        y_mean_pred = np.full_like(y_vl, y_tr.mean())
        baseline_mae = mean_absolute_error(y_vl, y_mean_pred)
        baseline_maes.append(baseline_mae)

        # Local pipeline
        local_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(
                alphas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                max_iter=10000,
                random_state=42,
                cv=3,
                n_jobs=-1,
            )),
        ])
        local_pipeline.fit(X_tr, y_tr)
        y_pred = local_pipeline.predict(X_vl)

        r2 = r2_score(y_vl, y_pred)
        rmse = np.sqrt(mean_squared_error(y_vl, y_pred))
        mae = mean_absolute_error(y_vl, y_pred)

        cv_r2_scores.append(r2)
        cv_rmse_scores.append(rmse)
        cv_mae_scores.append(mae)

        enet = local_pipeline.named_steps["enet"]
        print(f"    Fold {fold+1}: R²={r2:.4f}, RMSE={rmse:.4f}, "
              f"MAE={mae:.4f}, Baseline MAE={baseline_mae:.4f} "
              f"[alpha={enet.alpha_:.4f}, l1={enet.l1_ratio_:.2f}]")

    cv_r2_mean = float(np.mean(cv_r2_scores))
    cv_r2_std = float(np.std(cv_r2_scores))
    cv_rmse_mean = float(np.mean(cv_rmse_scores))
    cv_rmse_std = float(np.std(cv_rmse_scores))
    cv_mae_mean = float(np.mean(cv_mae_scores))
    baseline_mae_mean = float(np.mean(baseline_maes))

    print(f"\n  CV Results: R²={cv_r2_mean:.4f} ± {cv_r2_std:.4f}, "
          f"RMSE={cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}, "
          f"MAE={cv_mae_mean:.4f}")
    print(f"  Baseline MAE (mean predictor): {baseline_mae_mean:.4f}")

    # Fit final model on full training set
    print(f"\n  Fitting final model on full training set...")
    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            alphas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            max_iter=10000,
            random_state=42,
            cv=5,
            n_jobs=-1,
        )),
    ])
    model_pipeline.fit(X_train, y_train)
    enet = model_pipeline.named_steps["enet"]
    print(f"  Final: alpha={enet.alpha_:.4f}, l1_ratio={enet.l1_ratio_:.2f}")

    # ── Validation ──
    print(f"\n  ── 2023 Validation ──")
    y_pred = model_pipeline.predict(X_val)
    val_r2 = r2_score(y_val, y_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    val_mae = mean_absolute_error(y_val, y_pred)

    y_mean_pred = np.full_like(y_val, y_train.mean())
    val_baseline_mae = mean_absolute_error(y_val, y_mean_pred)

    print(f"  Validation R²: {val_r2:.4f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")
    print(f"  Validation MAE: {val_mae:.4f}")
    print(f"  Baseline MAE (mean predictor): {val_baseline_mae:.4f}")
    print(f"  Model MAE < Baseline MAE: {val_mae < val_baseline_mae}")

    # Scatter data
    scatter = [{"actual": float(a), "predicted": float(p)}
               for a, p in zip(y_val, y_pred)]
    print(f"  Predictions range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

    # ── Feature coefficients (in ORIGINAL unit space) ──
    # Get coefficients from the pipeline: coef_ * scaler.scale_^{-1}
    # The pipeline coefficients are in standardized space.
    # To get interpretable original-space coefs:
    coefs = enet.coef_  # standardized space
    scaler = model_pipeline.named_steps["scaler"]
    # For Pipeline, the actual coefficients of the linear model in original space
    # are coef_ / scale (since predict = (X - mean) / scale * coef_ + intercept_)
    # So original-space coef = coef_ / scaler.scale_
    scale = scaler.scale_
    orig_coefs = coefs / scale
    intercept = enet.intercept_ - np.sum(scaler.mean_ / scale * coefs)  # adjusted intercept for original space

    feat_coefs = {}
    print(f"\n  ── Top {min(10, len(feature_names))} Feature Coefficients ({player_type}) ──")
    print(f"  {'Feature':<22s} {'StdCoef':<10s} {'OrigCoef':<12s} {'|Orig|':<10s}")
    print(f"  {'-'*54}")

    feat_coef_list = list(zip(feature_names, coefs, orig_coefs))
    feat_coef_sorted = sorted(feat_coef_list, key=lambda x: abs(x[1]), reverse=True)

    for feat, std_coef, orig_coef in feat_coef_sorted[:10]:
        feat_coefs[feat] = float(orig_coef)
        print(f"  {feat:<22s} {std_coef:<+10.6f} {orig_coef:<+12.6f} {abs(orig_coef):<10.6f}")

    # ── Validation checks ──
    print(f"\n  ── Validation Checks ──")

    # V1: CV R² > 0.0
    v1 = cv_r2_mean > 0.0
    print(f"  [V1] CV R² > 0.0: {'PASS' if v1 else 'FAIL'} (R²={cv_r2_mean:.4f})")

    # V2: Model MAE < Baseline MAE (on validation)
    v2 = val_mae < val_baseline_mae
    print(f"  [V2] Model MAE < Baseline MAE: {'PASS' if v2 else 'FAIL'} "
          f"(Model={val_mae:.4f}, Baseline={val_baseline_mae:.4f})")

    # V3: Feature coefficient sign interpretability
    expected = {
        "hitter": {"wOBA_adj": "+", "OPS_adj": "+", "BB_pct_adj": "+", "K_pct_adj": "-",
                    "draft_round": "-", "height_inches": "+"},
        "pitcher": {"K_per_nine": "-", "BB_per_nine": "+",
                     "draft_round": "+", "Age": "?", "conf_strength": "?"},
    }
    exp = expected.get(player_type, {})
    v3_ok = 0
    v3_total = 0
    for feat, sign in exp.items():
        if feat in feat_coefs:
            v3_total += 1
            if (sign == "+" and feat_coefs[feat] > 0) or (sign == "-" and feat_coefs[feat] < 0):
                v3_ok += 1
    v3 = v3_ok >= max(1, v3_total // 2)
    print(f"  [V3] Interpretable signs ({v3_ok}/{v3_total} match): {'PASS' if v3 else 'FAIL'}")

    # V4: Predictions within realistic range
    if player_type == "hitter":
        v4 = 0.200 <= y_pred.min() <= 0.450 and y_pred.max() <= 0.470
    else:
        v4 = 2.0 <= y_pred.min() and y_pred.max() <= 8.5
    print(f"  [V4] In realistic range: {'PASS' if v4 else 'FAIL'} "
          f"(range=[{y_pred.min():.4f},{y_pred.max():.4f}])")

    # V5: Val R² within 0.10 of CV R²
    r2_diff = abs(val_r2 - cv_r2_mean)
    v5 = r2_diff <= 0.10
    print(f"  [V5] Val R² within 0.10 of CV R²: {'PASS' if v5 else 'FAIL'} "
          f"(diff={r2_diff:.4f})")

    all_pass = v1 and v2 and v3 and v4 and v5
    print(f"\n  Overall: {'✅ ALL CHECKS PASSED' if all_pass else '❌ SOME CHECKS FAILED'}")

    # ── Save artifacts ──
    artifact = {
        'pipeline': model_pipeline,
        'features': feature_names,
        'feature_coefficients_orig': feat_coefs,
        'feature_coefficients_std': {f: float(c) for f, c in zip(feature_names, coefs)},
        'intercept': float(intercept),
        'cv_r2_mean': float(cv_r2_mean),
        'cv_r2_std': float(cv_r2_std),
        'cv_rmse_mean': float(cv_rmse_mean),
        'cv_rmse_std': float(cv_rmse_std),
        'cv_mae_mean': float(cv_mae_mean),
        'cv_mae_std': float(np.std(cv_mae_scores)),
        'val_r2': float(val_r2),
        'val_rmse': float(val_rmse),
        'val_mae': float(val_mae),
        'baseline_mae': float(val_baseline_mae),
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'target_mean': float(y_train.mean()),
        'target_std': float(y_train.std()),
        'checks_passed': all_pass,
    }

    pkl_path = output_dir / f"milb_outcome_{player_type}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"\n  Saved: {pkl_path}")

    feat_json_path = output_dir / f"milb_outcome_features_{player_type}.json"
    with open(feat_json_path, 'w') as f:
        json.dump(feat_coefs, f, indent=2)
    print(f"  Saved: {feat_json_path}")

    scatter_path = output_dir / f"milb_outcome_scatter_{player_type}.json"
    with open(scatter_path, 'w') as f:
        json.dump(scatter, f, indent=2)
    print(f"  Saved: {scatter_path}")

    return artifact


def main():
    print("=" * 70)
    print("MiLB Outcome Model — Elastic Net Regression")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading data...")
    data = load_json(TRAINING_PATH)
    print(f"    Loaded {len(data)} records")

    conf_stats = load_json(CONF_STATS_PATH)
    print(f"    Loaded conference stats")

    # Compute adjusted features
    print("\n[2] Computing conference-adjusted stats...")
    data = compute_adj_features(data, conf_stats)
    print("    Done.")

    # Split by draft_year
    train_data = [d for d in data if d["draft_year"] in [2021, 2022]]
    val_data = [d for d in data if d["draft_year"] == 2023]
    print(f"\n[3] Train/Val split: {len(train_data)} train (2021-2022), "
          f"{len(val_data)} val (2023)")

    # ── Train per player type ──
    configs = [
        ("hitter", HITTER_ADJ_FEATURES, HITTER_BASE_FEATURES, HITTER_TARGET),
        ("pitcher", [], PITCHER_FEATURES, PITCHER_TARGET),
    ]

    results = {}
    for player_type, adj_features, base_features, target in configs:
        pt_train = [d for d in train_data if d["player_type"] == player_type]
        pt_val = [d for d in val_data if d["player_type"] == player_type]
        print(f"\n{'='*70}")
        print(f"Player Type: {player_type.upper()} — {len(pt_train)} train, {len(pt_val)} val")

        X_tr, y_tr, _, features = build_feature_matrix(
            pt_train, base_features, adj_features, target)
        X_vl, y_vl, _, _ = build_feature_matrix(
            pt_val, base_features, adj_features, target)

        print(f"  Feature matrix: X_train {X_tr.shape}, X_val {X_vl.shape}")

        if len(X_tr) < 20 or len(X_vl) < 5:
            print(f"  ⚠ Too few samples, skipping")
            continue

        artifact = train_and_evaluate(
            X_tr, y_tr, X_vl, y_vl, features, player_type, OUTPUT_DIR)
        results[player_type] = artifact

    # ── Final Report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    for pt, art in results.items():
        print(f"\n  {pt.upper()}:")
        print(f"    Samples: {art['n_train']} train, {art['n_val']} val")
        print(f"    CV:      R²={art['cv_r2_mean']:.4f} ± {art['cv_r2_std']:.4f}, "
              f"RMSE={art['cv_rmse_mean']:.4f}, MAE={art['cv_mae_mean']:.4f}")
        print(f"    Val:     R²={art['val_r2']:.4f}, RMSE={art['val_rmse']:.4f}, "
              f"MAE={art['val_mae']:.4f}")
        print(f"    Baseline MAE: {art['baseline_mae']:.4f} → Model MAE: {art['val_mae']:.4f}")
        print(f"    Checks:  {'✅ ALL PASSED' if art['checks_passed'] else '❌ SOME FAILED'}")
        pkl = OUTPUT_DIR / f"milb_outcome_{pt}.pkl"
        feat = OUTPUT_DIR / f"milb_outcome_features_{pt}.json"
        print(f"    Model:   {pkl}")
        print(f"    Features: {feat}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
