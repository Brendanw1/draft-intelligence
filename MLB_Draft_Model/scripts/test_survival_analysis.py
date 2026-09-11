#!/usr/bin/env python3
"""Test survival analysis (time-to-debut) as alternative target.

Instead of predicting WAR magnitude, predict:
1. Will the player debut in MLB? (binary classification)
2. How many years after draft until debut? (regression for debuted players)

This separates the "if" from the "how much" question.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
TRAINING_PATH = BASE / "data" / "training" / "tier5_training_set_v2.json"

HITTER_FEATURES = [
    "Age", "conf_strength",
    "wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj",
    "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate",
]
PITCHER_FEATURES = [
    "Age", "conf_strength",
    "ERA_adj", "FIP_adj", "K_per_nine_adj", "BB_per_nine_adj",
    "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate",
]


def load_data():
    with open(TRAINING_PATH) as f:
        data = json.load(f)
    records = data["records"]
    # Filter to real WAR only
    records = [r for r in records if not r.get("war_imputed", False) and r.get("signed_log_war") is not None]
    return records


def build_debut_matrix(records, features):
    """Binary classification: will player debut?"""
    X = np.array([[r.get(f, 0.0) or 0.0 for f in features] for r in records])
    # Player debuted if signed_log_war is not None and war_years_1_through_5 is not None
    y = np.array([1 if r.get("war_years_1_through_5") is not None else 0 for r in records])
    return X, y


def build_time_to_debut_matrix(records, features):
    """Regression: years from draft to debut (for debuted players only)."""
    debuted = [r for r in records if r.get("war_years_1_through_5") is not None and r.get("mlb_debut_season") is not None]
    X = np.array([[r.get(f, 0.0) or 0.0 for f in features] for r in debuted])
    # Years to debut = debut season - draft year
    y = np.array([r["mlb_debut_season"] - r["draft_year"] for r in debuted])
    return X, y, debuted


def main():
    records = load_data()
    print(f"Total records (real WAR): {len(records)}")
    
    for role, features in [("hitter", HITTER_FEATURES), ("pitcher", PITCHER_FEATURES)]:
        role_records = [r for r in records if r.get("player_type") == role]
        train = [r for r in role_records if r.get("split") == "train"]
        heldout = [r for r in role_records if r.get("split") == "heldout"]
        
        print(f"\n{'='*60}")
        print(f"{role.upper()} ({len(train)} train, {len(heldout)} heldout)")
        print(f"{'='*60}")
        
        # Task 1: Will player debut? (binary classification)
        print(f"\n1. DEBUT PREDICTION (Binary Classification)")
        X_train, y_train = build_debut_matrix(train, features)
        X_test, y_test = build_debut_matrix(heldout, features)
        
        print(f"  Train debut rate: {y_train.mean():.1%}")
        print(f"  Heldout debut rate: {y_test.mean():.1%}")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, pred)
        baseline_accuracy = y_train.mean()  # Predict majority class
        
        print(f"  Accuracy: {accuracy:.3f} (baseline: {baseline_accuracy:.3f})")
    # Check if there's variation in debut
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        print(f"  SKIP: No variation in debut status (all debuted)")
        print(f"  Note: Survival analysis requires mix of debuted/non-debuted players")
        print()
    else:
        print(f"  Classification Report:")
        print(classification_report(y_test, pred, target_names=["No Debut", "Debut"], zero_division=0))
        print()
    # Task 2: Time to debut (regression, for debuted players only)
    print(f"\n2. TIME TO DEBUT (Regression, debuted players only)")
    X_train_ttd, y_train_ttd, _ = build_time_to_debut_matrix(train, features)
    X_test_ttd, y_test_ttd, _ = build_time_to_debut_matrix(heldout, features)
    
    if len(y_train_ttd) > 10 and len(y_test_ttd) > 10:
        print(f"  Train: {len(y_train_ttd)} debuted players")
        print(f"  Heldout: {len(y_test_ttd)} debuted players")
        print(f"  Train mean years to debut: {y_train_ttd.mean():.2f}")
        print(f"  Heldout mean years to debut: {y_test_ttd.mean():.2f}")
        
        scaler_ttd = StandardScaler()
        X_train_ttd_scaled = scaler_ttd.fit_transform(X_train_ttd)
        X_test_ttd_scaled = scaler_ttd.transform(X_test_ttd)
        
        model_ttd = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model_ttd.fit(X_train_ttd_scaled, y_train_ttd)
        pred_ttd = model_ttd.predict(X_test_ttd_scaled)
        r2 = r2_score(y_test_ttd, pred_ttd)
        baseline_r2 = r2_score(y_test_ttd, [y_train_ttd.mean()] * len(y_test_ttd))
        
        print(f"  R²: {r2:.4f} (baseline: {baseline_r2:.4f})")
        print(f"  RMSE: {np.sqrt(np.mean((y_test_ttd - pred_ttd)**2)):.2f} years")
    else:
        print(f"  Insufficient data (train: {len(y_train_ttd)}, heldout: {len(y_test_ttd)})")


if __name__ == "__main__":
    main()
