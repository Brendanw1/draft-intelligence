#!/usr/bin/env python3
"""Test multiple model architectures on Tier 5 value regression.

Compares LightGBM, XGBoost, Random Forest, Ridge, and MLP on the balanced
2021→2022 temporal split with real WAR only.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
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


def build_matrix(records, features):
    X = np.array([[r.get(f, 0.0) or 0.0 for f in features] for r in records])
    y = np.array([r["signed_log_war"] for r in records])
    return X, y


def test_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    baseline_rmse = np.sqrt(mean_squared_error(y_test, [np.mean(y_train)] * len(y_test)))
    baseline_r2 = r2_score(y_test, [np.mean(y_train)] * len(y_test))
    return {
        "name": name,
        "r2": r2,
        "rmse": rmse,
        "baseline_r2": baseline_r2,
        "baseline_rmse": baseline_rmse,
        "beats_baseline": r2 > baseline_r2,
    }


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
        
        # Scale features for Ridge and MLP
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = [
            ("LightGBM", None),  # Will skip, already tested
            ("XGBoost", None),   # Will skip if not installed
            ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
            ("Ridge", Ridge(alpha=1.0)),
            ("MLP", MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)),
        ]
        
        results = []
        for name, model in models:
            if model is None:
                continue
            try:
                if name in ["Ridge", "MLP"]:
                    result = test_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test)
                else:
                    result = test_model(name, model, X_train, y_train, X_test, y_test)
                results.append(result)
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
        
        # Sort by R²
        results.sort(key=lambda x: x["r2"], reverse=True)
        
        print(f"\n{'Model':<20} {'R²':>8} {'RMSE':>8} {'Baseline R²':>12} {'Beats Baseline':>15}")
        print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*12} {'-'*15}")
        for r in results:
            beats = "✅" if r["beats_baseline"] else "❌"
            print(f"{r['name']:<20} {r['r2']:>8.4f} {r['rmse']:>8.4f} {r['baseline_r2']:>12.4f} {beats:>15}")


if __name__ == "__main__":
    main()
