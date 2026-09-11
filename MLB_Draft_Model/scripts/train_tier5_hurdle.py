#!/usr/bin/env python3
"""Train the Tier 5 Stage A hurdle classifier (Elastic Net logistic).

Predicts P(war_years_1_through_5 > 2.0) = "meaningfully productive" for
drafted players, split by role (hitter/pitcher), using the exact Tier 3
feature sets. Honest cohort reality: train = draft_year 2021, heldout =
2022-2023 (right-censored), 2.94% positive base rate (33/1124).

Artifacts (models/artifacts_full/):
    tier5_hurdle_{role}.pkl              -> model + features + metadata
    tier5_hurdle_{role}_features.json    -> coefficients + metrics
    tier5_hurdle_calibration.json        -> decile calibration (both roles)

Usage:
    python scripts/train_tier5_hurdle.py [--verify] [--training PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = BASE / "data" / "training" / "tier5_training_set.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"

HURDLE_THRESHOLD = 2.0
HURDLE_AUC_GATE = 0.65
N_BOOTSTRAP = 1000
L1_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
N_SPLITS = 5

# Feature sets copied verbatim from train_tier3_mlb_arrival.py (T3 features).
# Hitters deliberately exclude milb_year1_wOBA (AUC -0.025 in T3 exploration).
HITTER_T5_FEATURES = [
    "Age", "conf_strength",
    "wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj",
    "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate",
]
PITCHER_T5_FEATURES = [
    "Age", "conf_strength",
    "ERA_adj", "FIP_adj", "K_per_nine_adj", "BB_per_nine_adj",
    "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate",
]

ROLE_FEATURES = {"hitter": HITTER_T5_FEATURES, "pitcher": PITCHER_T5_FEATURES}


def safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_records(path):
    return json.load(open(path))


def split_by_role(records):
    return {
        "hitter": [r for r in records if r.get("player_type") == "hitter"],
        "pitcher": [r for r in records if r.get("player_type") == "pitcher"],
    }


def temporal_split(records):
    train = [r for r in records if r.get("split") == "train"]
    heldout = [r for r in records if r.get("split") == "heldout"]
    return train, heldout


def build_matrix(records, features):
    X = np.array(
        [[safe_float(r.get(f)) or 0.0 for f in features] for r in records],
        dtype=float,
    )
    y = np.array([1 if r.get("meaningfully_productive") else 0 for r in records])
    groups = [r.get("person_id") for r in records]
    return X, y, groups


def fit_hurdle_model(X, y, l1_ratio=0.3, class_weight="balanced"):
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=1.0,
        l1_ratio=l1_ratio,
        max_iter=2000,
        class_weight=class_weight,
        random_state=42,
    )
    model.fit(X, y)
    return model


def _fold_score(X_train, y_train, X_test, y_test, l1_ratio, class_weight):
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auc": float("nan"), "brier": float("nan")}
    model = fit_hurdle_model(X_train, y_train, l1_ratio, class_weight)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
    }


def cv_auc_brier(X, y, groups, l1_ratio=0.3, class_weight="balanced", n_splits=N_SPLITS):
    aucs, briers = [], []
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        score = _fold_score(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], l1_ratio, class_weight)
        aucs.append(score["auc"])
        briers.append(score["brier"])
    return {
        "auc_mean": float(np.nanmean(aucs)) if not all(math.isnan(a) for a in aucs) else float("nan"),
        "auc_std": float(np.nanstd(aucs)) if not all(math.isnan(a) for a in aucs) else float("nan"),
        "brier_mean": float(np.nanmean(briers)) if not all(math.isnan(b) for b in briers) else float("nan"),
        "auc_folds": [a for a in aucs],
        "brier_folds": [b for b in briers],
        "n_splits": n_splits,
    }


def tune_l1_ratio(X, y, groups, grid=L1_GRID, class_weight="balanced", n_splits=N_SPLITS):
    best, best_score = 0.3, float("-inf")
    for l1 in grid:
        cv = cv_auc_brier(X, y, groups, l1_ratio=l1, class_weight=class_weight, n_splits=n_splits)
        score = cv["auc_mean"]
        if math.isnan(score):
            score = 0.0
        if score > best_score:
            best, best_score = l1, score
    return best


def evaluate(model, X, y):
    if model is None or len(y) == 0 or len(np.unique(y)) < 2:
        return {"auc": float("nan"), "brier": float("nan")}
    proba = model.predict_proba(X)[:, 1]
    return {
        "auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }


def bootstrap_auc_ci(model, X, y, n_iter=N_BOOTSTRAP, seed=42):
    point = evaluate(model, X, y)
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y)
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        y_boot = y[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        proba = model.predict_proba(X[idx])[:, 1]
        aucs.append(roc_auc_score(y_boot, proba))
    low = high = point["auc"]
    if aucs:
        low = float(np.percentile(aucs, 2.5))
        high = float(np.percentile(aucs, 97.5))
    return {
        "n_iter": n_iter,
        "n_valid": len(aucs),
        "auc": point["auc"],
        "low": low,
        "high": high,
    }


def calibration_deciles(model, X, y, n_bins=10):
    if model is None or len(y) == 0:
        return []
    proba = model.predict_proba(X)[:, 1]
    order = np.argsort(proba)
    bins = np.array_split(order, n_bins)
    deciles = []
    for b in bins:
        if len(b) == 0:
            continue
        deciles.append({
            "n": int(len(b)),
            "pred_mean": float(np.mean(proba[b])),
            "obs_mean": float(np.mean(y[b])),
        })
    return deciles


def check_auc_gate(auc, gate=HURDLE_AUC_GATE):
    passed = (not math.isnan(auc)) and auc >= gate
    return {"status": "PASS" if passed else "WARN", "auc": float(auc), "gate": float(gate)}


def train_hurdle(records, role, features, l1_grid=L1_GRID, n_splits=N_SPLITS,
                 n_bootstrap=N_BOOTSTRAP, class_weight="balanced",
                 gate=HURDLE_AUC_GATE):
    pool = split_by_role(records)[role]
    train, heldout = temporal_split(pool)
    X_train, y_train, groups_train = build_matrix(train, features)
    X_held, y_held, _ = build_matrix(heldout, features)

    n_train = len(train)
    n_heldout = len(heldout)
    n_train_pos = int(y_train.sum()) if len(y_train) else 0
    n_heldout_pos = int(y_held.sum()) if len(y_held) else 0

    if n_train == 0 or len(np.unique(y_train)) < 2:
        return {
            "role": role,
            "features": list(features),
            "n_train": n_train,
            "n_train_pos": n_train_pos,
            "n_heldout": n_heldout,
            "n_heldout_pos": n_heldout_pos,
            "l1_ratio": None,
            "class_weight": class_weight,
            "cv": {"auc_mean": float("nan"), "auc_std": float("nan"),
                   "brier_mean": float("nan"), "n_splits": n_splits},
            "heldout": {"auc": float("nan"), "brier": float("nan")},
            "bootstrap": {"n_iter": n_bootstrap, "n_valid": 0,
                          "auc": float("nan"), "low": float("nan"), "high": float("nan")},
            "calibration": [],
            "gate": check_auc_gate(float("nan"), gate),
            "note": "insufficient positives to train",
            "model": None,
        }

    best_l1 = tune_l1_ratio(X_train, y_train, groups_train, grid=l1_grid,
                            class_weight=class_weight, n_splits=n_splits)
    cv = cv_auc_brier(X_train, y_train, groups_train, l1_ratio=best_l1,
                      class_weight=class_weight, n_splits=n_splits)
    model = fit_hurdle_model(X_train, y_train, l1_ratio=best_l1, class_weight=class_weight)
    heldout_metrics = evaluate(model, X_held, y_held)
    ci = bootstrap_auc_ci(model, X_held, y_held, n_iter=n_bootstrap)
    deciles = calibration_deciles(model, X_held, y_held)
    gate_result = check_auc_gate(heldout_metrics["auc"], gate)

    coefs = []
    if model is not None:
        coefs = sorted(
            zip(features, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True
        )

    return {
        "role": role,
        "features": list(features),
        "n_train": n_train,
        "n_train_pos": n_train_pos,
        "n_heldout": n_heldout,
        "n_heldout_pos": n_heldout_pos,
        "l1_ratio": best_l1,
        "class_weight": class_weight,
        "cv": cv,
        "heldout": heldout_metrics,
        "bootstrap": ci,
        "calibration": deciles,
        "gate": gate_result,
        "coefficients": [{"feature": f, "coefficient": float(c)} for f, c in coefs],
        "model": model,
    }


def save_artifacts(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration = {}
    for role, result in results.items():
        with open(OUTPUT_DIR / f"tier5_hurdle_{role}.pkl", "wb") as f:
            pickle.dump({
                "model": result["model"],
                "features": result["features"],
                "role": role,
                "l1_ratio": result["l1_ratio"],
                "class_weight": result["class_weight"],
                "hurdle_threshold": HURDLE_THRESHOLD,
                "gate": result["gate"],
                "n_train": result["n_train"],
                "n_heldout": result["n_heldout"],
            }, f)
        meta = {
            "model_type": f"tier5_hurdle_{role}",
            "role": role,
            "features": result["features"],
            "feature_coefficients": result["coefficients"],
            "n_train": result["n_train"],
            "n_train_pos": result["n_train_pos"],
            "n_heldout": result["n_heldout"],
            "n_heldout_pos": result["n_heldout_pos"],
            "l1_ratio": result["l1_ratio"],
            "class_weight": result["class_weight"],
            "cv": result["cv"],
            "heldout": result["heldout"],
            "bootstrap": result["bootstrap"],
            "gate": result["gate"],
            "hurdle_threshold": HURDLE_THRESHOLD,
            "hurdle_auc_gate": HURDLE_AUC_GATE,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(OUTPUT_DIR / f"tier5_hurdle_{role}_features.json", "w") as f:
            json.dump(meta, f, indent=2)
        calibration[role] = result["calibration"]

    with open(OUTPUT_DIR / "tier5_hurdle_calibration.json", "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "hurdle_threshold": HURDLE_THRESHOLD,
            "calibration": calibration,
        }, f, indent=2)


def print_report(results):
    print("=" * 64)
    print("TIER 5 STAGE A — HURDLE CLASSIFIER (Elastic Net logistic)")
    print("=" * 64)
    print(f"  hurdle: war_years_1_through_5 > {HURDLE_THRESHOLD}")
    print(f"  AUC gate: {HURDLE_AUC_GATE} (WARN, never crash)")
    print(f"  class_weight: balanced (base rate ~2.94%)")
    print("  " + "-" * 56)
    for role, r in results.items():
        held = r["heldout"]
        print(f"  {role:<8s} n_train={r['n_train']:>3d} (pos {r['n_train_pos']})  "
              f"n_heldout={r['n_heldout']:>3d} (pos {r['n_heldout_pos']})")
        print(f"           l1_ratio={r['l1_ratio']}  "
              f"CV AUC={r['cv']['auc_mean']:.4f}±{r['cv']['auc_std']:.4f}  "
              f"CV Brier={r['cv']['brier_mean']:.4f}")
        print(f"           heldout AUC={held['auc']:.4f}  Brier={held['brier']:.4f}  "
              f"bootstrap 95% CI=[{r['bootstrap']['low']:.3f}, {r['bootstrap']['high']:.3f}]")
        print(f"           GATE: {r['gate']['status']}")
    print("  " + "-" * 56)
    for role, r in results.items():
        print(f"  {role} calibration (decile: pred_mean vs obs_mean):")
        for i, d in enumerate(r["calibration"], 1):
            print(f"    bin {i:>2d}: n={d['n']:>4d}  pred={d['pred_mean']:.4f}  obs={d['obs_mean']:.4f}")
    print("  " + "-" * 56)
    print("  NOTE: 2.94% positive base rate (33/1124). Right-censored heldout")
    print("        (2022-2023). AUC < 0.65 is the expected honest outcome for")
    print("        a data-starved hurdle; the gate WARNS rather than crashing.")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train Tier 5 Stage A hurdle classifier.")
    p.add_argument("--training", default=str(DEFAULT_TRAINING), help="tier5_training_set.json path")
    p.add_argument("--verify", action="store_true", help="print AUC/Brier report")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    records = load_records(args.training)
    results = {}
    for role, features in ROLE_FEATURES.items():
        results[role] = train_hurdle(records, role, features)
    save_artifacts(results)
    print_report(results)
    if not args.verify:
        print("\nRun with --verify to print the AUC/Brier verification report.")


if __name__ == "__main__":
    main()
