#!/usr/bin/env python3
"""validate_models.py — Compare old vs expanded models on same test sets."""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

BASE = Path(__file__).resolve().parents[1]

def safe_float(v):
    if v is None: return None
    try: return float(v)
    except: return None

def parse_height(s):
    if not s: return None
    s = str(s).strip().replace("'", "-").replace('"', "").replace(" ", "")
    try:
        parts = s.split("-")
        if len(parts) == 2: return int(parts[0])*12 + int(parts[1])
    except: pass
    return None

print("=" * 60)
print("MODEL VALIDATION — Old vs Expanded")
print("=" * 60)

# ── Expanded Data Test Set (2025-2026 year-out) ──
with open(BASE / "data/training/expanded_training_set.json") as f:
    exp = json.load(f)

# Load expanded model
model_exp_h = xgb.XGBRegressor()
model_exp_h.load_model(str(BASE / "models/artifacts_expanded/fg_draft_hitter.json"))

with open(BASE / "models/artifacts_expanded/fg_features_hitter.json") as f:
    feat_meta = json.load(f)
hitter_feats = feat_meta["features"]

# Build year-out test set (2025-2026)
X_test_yr, y_test_yr = [], []
for r in exp:
    if r.get("player_type") != "hitter": continue
    season = r.get("season")
    if season not in (2025, 2026): continue
    pick = safe_float(r.get("draft_pick"))
    if pick is None: continue

    row = []
    for col in hitter_feats:
        val = safe_float(r.get(col))
        if col == "height_inches" and val is None:
            val = parse_height(r.get("height_raw", ""))
        if col == "conference_tier" and val is None:
            val = 4.0
        row.append(float(val) if val is not None else 0.0)
    X_test_yr.append(row)
    y_test_yr.append(pick)

X_test_yr = np.array(X_test_yr)
y_test_yr = np.array(y_test_yr)
y_pred_exp = model_exp_h.predict(X_test_yr)

r2_exp = r2_score(y_test_yr, y_pred_exp)
mae_exp = mean_absolute_error(y_test_yr, y_pred_exp)

print(f"\nExpanded Model on 2025-2026 test set:")
print(f"  R²:  {r2_exp:.4f}")
print(f"  MAE: {mae_exp:.1f} picks")
print(f"  n:   {len(y_test_yr)}")

# ── Old Model on Same Test Set ──
# Need to build features in OLD format (fg_XXX)
print(f"\nOld model comparison: needs feature format conversion —")
print(f"  The old model was trained on fg_XXX-prefixed columns")
print(f"  from the original training set format.")

# For a direct comparison, let's test on the original fg_training_set
# using the OLD model (already trained)
model_old_h = xgb.XGBRegressor()
model_old_h.load_model(str(BASE / "models/artifacts/fg_draft_hitter.json"))

with open(BASE / "models/artifacts/fg_features_hitter.json") as f:
    old_meta = json.load(f)
old_feats = old_meta["features"]

# Test old model on original data
with open(BASE / "data/training/fg_training_set.json") as f:
    orig = json.load(f)

X_old, y_old = [], []
for r in orig:
    if r.get("player_type") != "hitter": continue
    pick = safe_float(r.get("draft_pick"))
    if pick is None: continue
    row = []
    for col in old_feats:
        val = safe_float(r.get(col))
        row.append(float(val) if val is not None else 0.0)
    X_old.append(row)
    y_old.append(pick)

X_old = np.array(X_old)
y_old = np.array(y_old)
y_pred_old = model_old_h.predict(X_old)

r2_old = r2_score(y_old, y_pred_old)
mae_old = mean_absolute_error(y_old, y_pred_old)

print(f"\nOld Model on original training set (n={len(y_old)}):")
print(f"  R²:  {r2_old:.4f}")
print(f"  MAE: {mae_old:.1f}")

# ── Expanded model on same data ──
print(f"\nExpanded Model on original training set:")
# Need to transform features — map old columns to new model's expectations
X_transformed = []
for r in orig:
    if r.get("player_type") != "hitter": continue
    pick = safe_float(r.get("draft_pick"))
    if pick is None: continue
    row = []
    for col in hitter_feats:
        if col in ("height_inches", "bmi", "conference_tier"):
            val = safe_float(r.get(col))
            if col == "height_inches" and val is None:
                val = parse_height(r.get("height", ""))
            if col == "conference_tier" and val is None:
                val = 4.0
        elif col in ("AVG", "OBP", "SLG", "OPS", "ISO", "wOBA", "wRC_plus",
                     "wRC", "wRAA", "wBsR", "BB_pct", "K_pct", "BB/K",
                     "BABIP", "Spd", "G", "PA", "AB", "H", "1B", "2B", "3B",
                     "HR", "R", "RBI", "BB", "SO", "SB", "CS", "HBP", "SF",
                     "SH", "GDP", "Age"):
            # Old data uses fg_ prefix
            val = safe_float(r.get(f"fg_{col}"))
        else:
            val = 0.0
        row.append(float(val) if val is not None else 0.0)
    X_transformed.append(row)
    y_old.append(pick)  # already collected

X_transformed = np.array(X_transformed)
y_pred_exp_old = model_exp_h.predict(X_transformed)

# Trim to same length
min_len = min(len(y_old), len(y_pred_exp_old))
r2_exp_old = r2_score(y_old[:min_len], y_pred_exp_old[:min_len])
mae_exp_old = mean_absolute_error(y_old[:min_len], y_pred_exp_old[:min_len])

print(f"  R²:  {r2_exp_old:.4f}")
print(f"  MAE: {mae_exp_old:.1f}")

print(f"\n{'=' * 60}")
print(f"COMPARISON SUMMARY")
print(f"{'=' * 60}")
print(f"")
print(f"Old model on original data  (n={len(y_old)}): R²={r2_old:.4f}  MAE={mae_old:.1f}")
print(f"Expanded model on original  (n={len(y_old)}): R²={r2_exp_old:.4f}  MAE={mae_exp_old:.1f}")
print(f"Expanded model on year-out  (n={len(y_test_yr)}): R²={r2_exp:.4f}  MAE={mae_exp:.1f}")
print(f"")
r2_improvement = r2_exp_old - r2_old
print(f"R² improvement on original: {r2_improvement:+.4f}")
