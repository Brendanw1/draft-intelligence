#!/usr/bin/env python3
"""
Tier 5 Value Regression — Phase B (Optuna Tuning + Feature Selection + Ensemble)

This script runs ONLY if Phase A passes kill criteria (R² ≥ 0.0 for both roles).
If Phase A failed, this script exits gracefully with a warning.

Phase B enhancements:
1. Optuna hyperparameter tuning (100 trials)
2. Recursive feature elimination (RFE)
3. Ensemble of top-3 models
4. Comparison to Phase A baseline
5. Save artifacts with _phaseB suffix

Usage:
    python scripts/train_tier5_value_phaseB.py [--trials 100] [--verify]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# LightGBM + Optuna
try:
    import lightgbm as lgb
    import optuna
except ImportError as e:
    print(f"ERROR: Missing dependencies: {e}")
    print("Install with: pip install lightgbm optuna")
    sys.exit(1)

BASE = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE / "models/artifacts_full"
TRAINING_SET_PATH = BASE / "data/training/tier5_training_set.json"


def load_training_set():
    """Load Tier 5 training set."""
    with open(TRAINING_SET_PATH) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_phase_a_artifacts(role):
    """Load Phase A model and metrics for comparison."""
    pkl_path = ARTIFACTS_DIR / f"tier5_value_{role}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def check_phase_a_gate(phase_a_hitter, phase_a_pitcher):
    """Check if Phase A passed kill criteria."""
    if phase_a_hitter is None or phase_a_pitcher is None:
        print("WARNING: Phase A artifacts not found. Cannot proceed with Phase B.")
        return False

    hitter_r2 = phase_a_hitter.get("gate", {}).get("r2", -999)
    pitcher_r2 = phase_a_pitcher.get("gate", {}).get("r2", -999)

    print(f"Phase A Hitter R²: {hitter_r2:.4f}")
    print(f"Phase A Pitcher R²: {pitcher_r2:.4f}")

    if hitter_r2 < 0.0 or pitcher_r2 < 0.0:
        print("WARNING: Phase A failed kill criteria (R² < 0.0 for one or both roles).")
        print("Phase B is designed to refine, not rescue. Skipping.")
        return False

    print("Phase A passed kill criteria. Proceeding with Phase B.")
    return True


def objective(trial, X_train, y_train, groups, role):
    """Optuna objective for LightGBM hyperparameter tuning."""
    # Hyperparameter search space
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    # GroupKFold cross-validation
    gkf = GroupKFold(n_splits=3)
    r2_scores = []

    for train_idx, val_idx in gkf.split(X_train, y_train, groups):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMRegressor(**param)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )

        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        r2_scores.append(r2)

    return np.mean(r2_scores)


def recursive_feature_elimination(X, y, groups, role, n_features_to_select=8):
    """Simple RFE: iteratively remove least important feature."""
    # Train initial model
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, verbosity=-1)
    model.fit(X, y)

    feature_names = [f"f{i}" for i in range(X.shape[1])]
    remaining_features = list(range(X.shape[1]))

    while len(remaining_features) > n_features_to_select:
        # Get feature importances
        importances = model.feature_importances_
        remaining_importances = importances[remaining_features]

        # Remove least important
        min_idx = np.argmin(remaining_importances)
        removed = remaining_features.pop(min_idx)
        print(f"  RFE: Removed feature {feature_names[removed]} (importance={remaining_importances[min_idx]:.2f})")

        # Retrain
        X_subset = X[:, remaining_features]
        model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, verbosity=-1)
        model.fit(X_subset, y)

    return remaining_features


def train_ensemble(X_train, y_train, groups, best_params_list, role):
    """Train ensemble of top-3 models."""
    models = []
    for i, params in enumerate(best_params_list[:3]):
        print(f"  Training ensemble model {i+1}/3...")
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        models.append(model)

    return models


def compute_metrics(y_true, y_pred, y_baseline):
    """Compute regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)

    # Baseline metrics
    rmse_baseline = np.sqrt(mean_squared_error(y_true, y_baseline))
    r2_baseline = r2_score(y_true, y_baseline)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": corr,
        "rmse_baseline": rmse_baseline,
        "r2_baseline": r2_baseline,
    }


def train_phase_b_role(df_role, role, n_trials=100):
    """Train Phase B model for one role (hitter or pitcher)."""
    print(f"\n{'='*60}")
    print(f"Phase B: {role.upper()}")
    print(f"{'='*60}")

    # Prepare data
    feature_cols = [c for c in df_role.columns if c.startswith("feature_")]
    X = df_role[feature_cols].values
    y = df_role["signed_log_war"].values
    groups = df_role["draft_year"].values

    # Split: train=2021, heldout=2022-2023
    train_mask = df_role["draft_year"] == 2021
    heldout_mask = df_role["draft_year"].isin([2022, 2023])

    X_train, X_heldout = X[train_mask], X[heldout_mask]
    y_train, y_heldout = y[train_mask], y[heldout_mask]
    groups_train = groups[train_mask]

    print(f"Train: {X_train.shape[0]}, Heldout: {X_heldout.shape[0]}")

    # 1. Optuna tuning
    print(f"\n[1/4] Optuna hyperparameter tuning ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize", study_name=f"tier5_value_{role}")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, groups_train, role),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best_params = study.best_params
    print(f"  Best CV R²: {study.best_value:.4f}")
    print(f"  Best params: {best_params}")

    # 2. Feature selection
    print(f"\n[2/4] Recursive feature elimination...")
    selected_features = recursive_feature_elimination(X_train, y_train, groups_train, role, n_features_to_select=8)
    X_train_sel = X_train[:, selected_features]
    X_heldout_sel = X_heldout[:, selected_features]
    print(f"  Selected {len(selected_features)} features")

    # 3. Train top-3 models for ensemble
    print(f"\n[3/4] Training ensemble (top-3 models)...")
    best_params_list = [study.trials[i].params for i in range(min(3, len(study.trials)))]
    ensemble_models = train_ensemble(X_train_sel, y_train, groups_train, best_params_list, role)

    # 4. Evaluate on heldout
    print(f"\n[4/4] Evaluating on heldout (2022-2023)...")
    # Ensemble prediction (average)
    y_pred_ensemble = np.mean([m.predict(X_heldout_sel) for m in ensemble_models], axis=0)

    # Baseline: predict training median
    y_baseline = np.full_like(y_heldout, np.median(y_train))

    metrics = compute_metrics(y_heldout, y_pred_ensemble, y_baseline)
    print(f"  RMSE: {metrics['rmse']:.4f} (baseline: {metrics['rmse_baseline']:.4f})")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f} (baseline: {metrics['r2_baseline']:.4f})")
    print(f"  Correlation: {metrics['correlation']:.4f}")

    # Gate check
    if metrics["r2"] < 0.0:
        print(f"  WARNING: R² < 0.0 — Phase B also failed kill criteria")
    else:
        print(f"  PASS: R² ≥ 0.0")

    # Save artifacts
    artifact = {
        "ensemble_models": ensemble_models,
        "selected_features": selected_features,
        "feature_names": [feature_cols[i] for i in selected_features],
        "best_params": best_params,
        "validation_metrics": metrics,
        "n_train": len(y_train),
        "n_heldout": len(y_heldout),
    }

    pkl_path = ARTIFACTS_DIR / f"tier5_value_{role}_phaseB.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  Saved: {pkl_path}")

    # Save feature list
    feat_path = ARTIFACTS_DIR / f"tier5_value_{role}_phaseB_features.json"
    with open(feat_path, "w") as f:
        json.dump({
            "features": artifact["feature_names"],
            "n_train": artifact["n_train"],
            "n_heldout": artifact["n_heldout"],
            "best_params": best_params,
        }, f, indent=2)
    print(f"  Saved: {feat_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Tier 5 Value Regression Phase B")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--verify", action="store_true", help="Run verification checks")
    args = parser.parse_args()

    print("="*60)
    print("Tier 5 Value Regression — Phase B")
    print("="*60)

    # Load training set
    df = load_training_set()
    print(f"Training set: {len(df)} records")

    # Check Phase A gate
    phase_a_hitter = load_phase_a_artifacts("hitter")
    phase_a_pitcher = load_phase_a_artifacts("pitcher")

    if not check_phase_a_gate(phase_a_hitter, phase_a_pitcher):
        print("\nPhase B SKIPPED — Phase A did not pass kill criteria.")
        sys.exit(0)

    # Split by role
    df_hitter = df[df["player_type"] == "hitter"].copy()
    df_pitcher = df[df["player_type"] == "pitcher"].copy()

    print(f"\nHitters: {len(df_hitter)}, Pitchers: {len(df_pitcher)}")

    # Train Phase B for each role
    metrics_hitter = train_phase_b_role(df_hitter, "hitter", n_trials=args.trials)
    metrics_pitcher = train_phase_b_role(df_pitcher, "pitcher", n_trials=args.trials)

    # Summary
    print(f"\n{'='*60}")
    print("Phase B Summary")
    print(f"{'='*60}")
    print(f"Hitter R²: {metrics_hitter['r2']:.4f}")
    print(f"Pitcher R²: {metrics_pitcher['r2']:.4f}")

    if args.verify:
        print(f"\n[Verification] Comparing to Phase A...")
        if phase_a_hitter and phase_a_pitcher:
            hitter_improvement = metrics_hitter["r2"] - phase_a_hitter["validation_metrics"]["r2"]
            pitcher_improvement = metrics_pitcher["r2"] - phase_a_pitcher["validation_metrics"]["r2"]
            print(f"  Hitter improvement: {hitter_improvement:+.4f}")
            print(f"  Pitcher improvement: {pitcher_improvement:+.4f}")

    print("\nPhase B complete.")


if __name__ == "__main__":
    main()
