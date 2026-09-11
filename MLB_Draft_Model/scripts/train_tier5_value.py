#!/usr/bin/env python3
"""Train the Tier 5 Stage B value regressor (LightGBM).

Predicts ``signed_log_war`` for drafted players, split by role
(hitter/pitcher), using the exact Tier 3 feature sets.  Honest cohort
reality: train = draft_year 2021, heldout = 2022-2023 (right-censored).

The target is the signed-log transform of 5-year cumulative WAR
(``sign(war) * log1p(|war|)``), already computed by
``build_tier5_training.py``.  This handles the zero-inflated WAR
distribution while preserving directionality.

Artifacts (models/artifacts_full/):
    tier5_value_{role}.pkl              -> model + features + metadata
    tier5_value_{role}_features.json    -> importances + metrics
    tier5_value_calibration.json        -> decile calibration (both roles)

Usage:
    python scripts/train_tier5_value.py [--verify] [--training PATH]
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "lightgbm is required for Tier 5 value regression.  "
        "Install with: pip install lightgbm>=4.0.0"
    ) from exc

from sklearn.ensemble import RandomForestRegressor

BASE = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = BASE / "data" / "training" / "tier5_training_set_v2.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"

# LightGBM defaults — conservative to prevent overfitting on ~1,200 train records.
# Increased n_estimators and reduced regularization for larger dataset.
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_samples": 10,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "random_state": 42,
    "verbose": -1,
    "objective": "regression",
    "metric": "rmse",
}

N_SPLITS = 5
N_BOOTSTRAP = 1000
VALUE_R2_GATE = 0.0  # must beat predicting the mean

# Feature sets copied verbatim from train_tier5_hurdle.py (Tier 3 features).
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


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_records(path, real_war_only=False):
    data = json.load(open(path))
    if isinstance(data, dict) and "records" in data:
        records = data["records"]
    else:
        records = data
    
    if real_war_only:
        # Filter to records with real WAR data (not imputed zeros)
        records = [r for r in records if not r.get("war_imputed", False) and r.get("signed_log_war") is not None]
    
    return records


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
    y = np.array([safe_float(r.get("signed_log_war")) or 0.0 for r in records])
    groups = [r.get("person_id") for r in records]
    # Sample weights: real WAR records get 20x weight vs imputed zeros
    # This prevents the model from just learning "predict 0 for everyone"
    weights = np.array([
        20.0 if not r.get("war_imputed", False) else 1.0
        for r in records
    ])
    return X, y, groups, weights


def signed_log_to_war(signed_log):
    """Inverse of signed_log_war: ``sign(x) * (exp(|x|) - 1)``."""
    if signed_log == 0.0:
        return 0.0
    sign = -1.0 if signed_log < 0 else 1.0
    return sign * (math.expm1(abs(signed_log)))


def fit_value_model(X, y, params=None, sample_weight=None, model_type="lightgbm"):
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        model.fit(X, y)
        return model
    else:
        params = params or dict(LGBM_PARAMS)
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y, sample_weight=sample_weight)
        return model


def evaluate(model, X, y):
    if model is None or len(y) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                "corr": float("nan")}
    pred = model.predict(X)
    rmse = float(math.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))
    if len(y) >= 2 and np.std(y) > 0 and np.std(pred) > 0:
        corr = float(np.corrcoef(y, pred)[0, 1])
        if math.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "corr": corr}


def _fold_score(X_train, y_train, X_test, y_test, params, w_train=None):
    if len(y_train) == 0 or len(y_test) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    model = fit_value_model(X_train, y_train, params, sample_weight=w_train)
    return evaluate(model, X_test, y_test)


def cv_rmse_r2(X, y, groups, params=None, n_splits=N_SPLITS, weights=None):
    params = params or dict(LGBM_PARAMS)
    rmses, maes, r2s = [], [], []
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        w_tr = weights[tr_idx] if weights is not None else None
        score = _fold_score(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], params, w_tr)
        rmses.append(score["rmse"])
        maes.append(score["mae"])
        r2s.append(score["r2"])
    return {
        "rmse_mean": float(np.nanmean(rmses)) if rmses else float("nan"),
        "rmse_std": float(np.nanstd(rmses)) if rmses else float("nan"),
        "mae_mean": float(np.nanmean(maes)) if maes else float("nan"),
        "r2_mean": float(np.nanmean(r2s)) if r2s else float("nan"),
        "r2_folds": [r for r in r2s],
        "n_splits": n_splits,
    }


def bootstrap_rmse_ci(model, X, y, n_iter=N_BOOTSTRAP, seed=42):
    point = evaluate(model, X, y)
    rng = np.random.RandomState(seed)
    rmses = []
    n = len(y)
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        y_boot = y[idx]
        pred_boot = model.predict(X[idx])
        rmses.append(math.sqrt(mean_squared_error(y_boot, pred_boot)))
    low = high = point["rmse"]
    if rmses:
        low = float(np.percentile(rmses, 2.5))
        high = float(np.percentile(rmses, 97.5))
    return {
        "n_iter": n_iter,
        "n_valid": len(rmses),
        "rmse": point["rmse"],
        "low": low,
        "high": high,
    }


def calibration_deciles(model, X, y, n_bins=10):
    if model is None or len(y) == 0:
        return []
    pred = model.predict(X)
    order = np.argsort(pred)
    bins = np.array_split(order, n_bins)
    deciles = []
    for b in bins:
        if len(b) == 0:
            continue
        deciles.append({
            "n": int(len(b)),
            "pred_mean": float(np.mean(pred[b])),
            "pred_war_mean": float(np.mean([signed_log_to_war(p) for p in pred[b]])),
            "obs_mean": float(np.mean(y[b])),
            "obs_war_mean": float(np.mean([signed_log_to_war(v) for v in y[b]])),
        })
    return deciles


def check_r2_gate(r2, gate=VALUE_R2_GATE):
    passed = (not math.isnan(r2)) and r2 >= gate
    return {"status": "PASS" if passed else "WARN", "r2": float(r2), "gate": float(gate)}


def feature_importances(model, features):
    if model is None:
        return []
    imp = model.feature_importances_
    return sorted(
        [{"feature": f, "importance": int(v)} for f, v in zip(features, imp)],
        key=lambda x: x["importance"],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Baseline comparators
# ---------------------------------------------------------------------------

def baseline_predict_median(y_train, y_test):
    """Predict the training median for every heldout record."""
    if len(y_train) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    median = float(np.median(y_train))
    pred = np.full_like(y_test, median, dtype=float)
    rmse = float(math.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def baseline_hurdle_only(hurdle_probs, y_test, hurdle_threshold=0.5):
    """Predict 0 for non-productive, training-median-of-positives for productive.

    This is the 'hurdle classifier only' baseline — no value regression,
    just the two-stage approach with a constant magnitude.
    """
    if len(y_test) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    positive_mask = np.array(hurdle_probs) >= hurdle_threshold
    pred = np.zeros(len(y_test))
    if positive_mask.any():
        pred[positive_mask] = 0.0  # placeholder; magnitude comes from value model
    # For a fair comparison, use the mean of positive training labels
    # (this is computed externally and passed in)
    return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
            "note": "computed externally with training positive mean"}


def baseline_direct_regression(X_train, y_train, X_test, y_test, params=None, w_train=None):
    """Single LightGBM on all records (no two-stage split).

    This is the 'direct regression' baseline — predict signed_log_war
    directly without the hurdle gate.
    """
    if len(y_train) == 0 or len(y_test) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    model = fit_value_model(X_train, y_train, params, sample_weight=w_train)
    return evaluate(model, X_test, y_test)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_value(records, role, features, use_imputed=False, model_type="random_forest"):
    """Train value regressor for one role.
    
    Args:
        records: list of player records
        role: 'hitter' or 'pitcher'
        features: feature list
        use_imputed: if False, filter to real WAR only
        model_type: 'lightgbm' or 'random_forest'
    """
    role_records = [r for r in records if r.get("player_type") == role]
    
    if not use_imputed:
        role_records = [r for r in role_records if not r.get("war_imputed", False)]
    
    train, heldout = temporal_split(role_records)
    X_train, y_train, groups_train, weights_train = build_matrix(train, features)
    X_test, y_test, _, _ = build_matrix(heldout, features)
    
    print(f"\n  {role:<8} n_train={len(train):>3}  n_heldout={len(heldout):>3}")
    
    # Cross-validation
    cv = cv_rmse_r2(X_train, y_train, groups_train, weights=weights_train)
    
    # Train final model
    model = fit_value_model(X_train, y_train, sample_weight=weights_train, model_type=model_type)
    
    # Evaluate
    heldout_metrics = evaluate(model, X_test, y_test)
    bootstrap_ci = bootstrap_rmse_ci(model, X_test, y_test)
    
    # Baseline
    baseline_pred = [np.median(y_train)] * len(y_test)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_r2 = float(r2_score(y_test, baseline_pred))
    
    gate_status = "PASS" if heldout_metrics["r2"] >= VALUE_R2_GATE else "WARN"
    
    return {
        "role": role,
        "n_train": len(train),
        "n_heldout": len(heldout),
        "cv": cv,
        "heldout": heldout_metrics,
        "bootstrap_ci": bootstrap_ci,
        "baseline": {"rmse": baseline_rmse, "mae": baseline_mae, "r2": baseline_r2},
        "gate": {"status": gate_status, "r2": heldout_metrics["r2"], "threshold": VALUE_R2_GATE},
        "model": model,
        "features": features,
        "model_type": model_type,
    }


def save_artifacts(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration = {}
    for role, result in results.items():
        with open(OUTPUT_DIR / f"tier5_value_{role}.pkl", "wb") as f:
            pickle.dump({
                "model": result["model"],
                "features": result["features"],
                "role": role,
                "gate": result["gate"],
                "n_train": result["n_train"],
                "n_heldout": result["n_heldout"],
                "model_type": result.get("model_type", "random_forest"),
            }, f)
        meta = {
            "model_type": f"tier5_value_{role}",
            "role": role,
            "features": result["features"],
            "n_train": result["n_train"],
            "n_heldout": result["n_heldout"],
            "cv": result["cv"],
            "heldout": result["heldout"],
            "bootstrap": result["bootstrap_ci"],
            "gate": result["gate"],
            "baseline": result.get("baseline", {}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(OUTPUT_DIR / f"tier5_value_{role}_features.json", "w") as f:
            json.dump(meta, f, indent=2)
        calibration[role] = result.get("calibration", [])

    with open(OUTPUT_DIR / "tier5_value_calibration.json", "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "calibration": calibration,
        }, f, indent=2)


def print_report(results):
    print("=" * 64)
    print("TIER 5 STAGE B — VALUE REGRESSOR")
    print("=" * 64)
    print(f"  target: signed_log_war  (sign(war) * log1p(|war|))")
    print(f"  R² gate: {VALUE_R2_GATE} (must beat predicting the mean)")
    print("  " + "-" * 56)
    for role, r in results.items():
        held = r["heldout"]
        bl = r.get("baseline", {})
        print(f"  {role:<8s} n_train={r['n_train']:>3d}  "
              f"n_heldout={r['n_heldout']:>3d}  "
              f"model={r.get('model_type', 'random_forest')}")
        print(f"           CV  RMSE={r['cv']['rmse_mean']:.4f}±{r['cv']['rmse_std']:.4f}  "
              f"R²={r['cv']['r2_mean']:.4f}")
        print(f"           heldout RMSE={held['rmse']:.4f}  MAE={held['mae']:.4f}  "
              f"R²={held['r2']:.4f}  corr={held['corr']:+.4f}")
        print(f"           bootstrap 95% CI=[{r['bootstrap_ci']['low']:.3f}, "
              f"{r['bootstrap_ci']['high']:.3f}]")
        if bl and not math.isnan(bl.get("rmse", float("nan"))):
            print(f"           baseline(median) RMSE={bl['rmse']:.4f}  "
                  f"MAE={bl['mae']:.4f}  R²={bl['r2']:.4f}")
        print(f"           GATE: {r['gate']['status']}")
    print("  " + "-" * 56)
    print("  NOTE: Right-censored heldout. R² < 0.0 means the")
    print("        model performs worse than predicting the training median.")
    print("        The gate WARNS rather than crashing.")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train Tier 5 Stage B value regressor.")
    p.add_argument("--training", default=str(DEFAULT_TRAINING),
                   help="tier5_training_set.json path")
    p.add_argument("--verify", action="store_true",
                   help="print RMSE/MAE/R² verification report")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    records = load_records(args.training, real_war_only=True)
    print(f"Training on {len(records)} records with real WAR data only")
    results = {}
    for role, features in ROLE_FEATURES.items():
        results[role] = train_value(records, role, features, use_imputed=False)
    save_artifacts(results)
    print_report(results)
    if not args.verify:
        print("\nRun with --verify to print the RMSE/MAE/R² verification report.")


if __name__ == "__main__":
    main()
