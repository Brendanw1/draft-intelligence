#!/usr/bin/env python3
"""Test WAR bands (ordinal classification) as alternative target.

Instead of predicting continuous WAR, predict WAR bands:
  0: WAR <= 0 (bust/no contribution)
  1: 0 < WAR <= 1 (replacement level)
  2: 1 < WAR <= 2 (solid contributor)
  3: 2 < WAR <= 5 (good player)
  4: WAR > 5 (star player)

This reduces variance and may work better with small samples.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
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


def assign_war_band(signed_log_war):
    """Convert signed_log_war to ordinal band (0-4)."""
    if signed_log_war <= 0:
        return 0  # WAR <= 0
    elif signed_log_war <= 0.693:  # log1p(1) = 0.693
        return 1  # 0 < WAR <= 1
    elif signed_log_war <= 1.099:  # log1p(2) = 1.099
        return 2  # 1 < WAR <= 2
    elif signed_log_war <= 1.792:  # log1p(5) = 1.792
        return 3  # 2 < WAR <= 5
    else:
        return 4  # WAR > 5


def load_data():
    with open(TRAINING_PATH) as f:
        data = json.load(f)
    records = data["records"]
    records = [r for r in records if not r.get("war_imputed", False) and r.get("signed_log_war") is not None]
    return records


def build_matrix(records, features):
    X = np.array([[r.get(f, 0.0) or 0.0 for f in features] for r in records])
    y = np.array([assign_war_band(r["signed_log_war"]) for r in records])
    return X, y


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
        
        X_train, y_train = build_matrix(train, features)
        X_test, y_test = build_matrix(heldout, features)
        
        # Check class distribution
        print(f"Train class distribution: {np.bincount(y_train, minlength=5)}")
        print(f"Heldout class distribution: {np.bincount(y_test, minlength=5)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = [
            ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ]
        
        for name, model in models:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, pred)
            baseline_accuracy = 1.0 / 5  # Random guess
            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.3f} (baseline: {baseline_accuracy:.3f})")
            print(f"  Classification Report:")
            print(classification_report(y_test, pred, labels=[0,1,2,3,4], 
                                       target_names=["WAR<=0", "0-1", "1-2", "2-5", "5+"],
                                       zero_division=0))


if __name__ == "__main__":
    main()
