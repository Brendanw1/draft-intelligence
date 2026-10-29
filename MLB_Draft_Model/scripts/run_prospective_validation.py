#!/usr/bin/env python3
"""
run_prospective_validation.py — End-to-end prospective validation of the MLB draft model.

Steps:
  1. Filter training data to exclude 2026 outcomes (backup originals)
  2. Retrain Tier 1 (draft position regressor)
  3. Retrain Tier 2 (full population classifier)
  4. Retrain Tier 3 (MLB arrival predictor)
  5. Run inference on 2026 players
  6. Compute accuracy against actual 2026 draft results
  7. Generate accuracy report at analysis/2026_draft_accuracy_prospective.md
  8. Restore original training data

Usage:
  cd /Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model
  python3 scripts/run_prospective_validation.py
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "training"
MODEL_DIR = BASE / "models" / "artifacts_full"

def log(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")
    sys.stdout.flush()

def run_cmd(cmd, cwd=None, label=None):
    """Run a command and print output."""
    if label:
        log(label)
    print(f"  Running: {' '.join(cmd)}")
    sys.stdout.flush()
    result = subprocess.run(
        cmd,
        cwd=cwd or BASE,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.stdout:
        # Print last 30 lines
        lines = result.stdout.strip().split("\n")
        for line in lines[-40:]:
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  ERROR (exit code {result.returncode}):")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-20:]:
                print(f"  ERR: {line}")
        return False
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-10:]:
            print(f"  (stderr) {line}")
    sys.stdout.flush()
    return True

def step_filter_training():
    """Step 1: Filter training data to remove 2026 outcomes."""
    log("STEP 1: FILTERING TRAINING DATA — Removing 2026 outcomes")
    
    files_to_filter = [
        ("expanded_training_set.json", "draft_year", 2026, "Positives (drafted players)"),
        ("tier2_negatives.json", "season", 2026, "Negatives (undrafted players)"),
        ("fg_training_set.json", "draft_year", 2026, "FG training set (Tier 1 features)"),
    ]
    
    results = {}
    for fname, field, value, label in files_to_filter:
        fpath = DATA_DIR / fname
        bakpath = DATA_DIR / (fname + ".bak")
        
        print(f"\n  {label}: {fname}")
        
        # Load
        with open(fpath) as f:
            data = json.load(f)
        
        total = len(data)
        removed = [r for r in data if r.get(field) == value]
        filtered = [r for r in data if r.get(field) != value]
        
        print(f"    Total: {total:,}")
        print(f"    {field}={value}: {len(removed):,}")
        print(f"    After filter: {len(filtered):,}")
        
        # Backup (preserve existing backup)
        if not bakpath.exists():
            shutil.copy2(fpath, bakpath)
            print(f"    Backup: {fname}.bak created")
        else:
            print(f"    Backup: {fname}.bak exists (not overwritten)")
        
        # Write filtered
        with open(fpath, "w") as f:
            json.dump(filtered, f)
        print(f"    Saved filtered {fname}")
        
        results[fname] = {"total": total, "removed": len(removed), "remaining": len(filtered)}
    
    print(f"\n  Filter summary:")
    for fname, r in results.items():
        print(f"    {fname}: {r['total']:,} → {r['remaining']:,} ({r['removed']} removed)")
    
    return True

def step_retrain_tier1():
    """Step 2: Retrain Tier 1 (draft position regressor)."""
    log("STEP 2: RETRAINING TIER 1 — Draft Pick Regressor")
    return run_cmd(
        ["python3", "scripts/train_fg_model.py", "--output-dir", "models/artifacts_full"],
        label="Tier 1 Training (XGBoost regressor)"
    )

def step_retrain_tier2():
    """Step 3: Retrain Tier 2 (full population classifier)."""
    log("STEP 3: RETRAINING TIER 2 — Full Population Classifier")
    return run_cmd(
        ["python3", "scripts/train_tier2_full.py"],
        label="Tier 2 Training (XGBoost + Platt/Isotonic)"
    )

def step_retrain_tier3():
    """Step 4: Retrain Tier 3 (MLB arrival predictor)."""
    log("STEP 4: RETRAINING TIER 3 — MLB Arrival Predictor")
    return run_cmd(
        ["python3", "scripts/train_tier3_mlb_arrival.py"],
        label="Tier 3 Training (Elastic Net + NN)"
    )

def step_run_inference():
    """Step 5: Run inference on 2026 players."""
    log("STEP 5: RUNNING INFERENCE ON 2026 PROSPECTS")
    return run_cmd(
        ["python3", "scripts/infer_2026.py", "--model-dir", "models/artifacts_full"],
        label="2026 Inference"
    )

def step_compute_accuracy():
    """Step 6: Compute accuracy and generate report."""
    log("STEP 6: COMPUTING ACCURACY AND GENERATING REPORT")
    return run_cmd(
        ["python3", "scripts/compute_2026_accuracy_prospective.py"],
        label="Accuracy Computation"
    )

def step_restore_training():
    """Step 7: Restore original training data."""
    log("STEP 7: RESTORING ORIGINAL TRAINING DATA")
    
    files_to_restore = [
        "expanded_training_set.json",
        "tier2_negatives.json",
        "fg_training_set.json",
    ]
    
    for fname in files_to_restore:
        orig = DATA_DIR / fname
        bak = DATA_DIR / (fname + ".bak")
        if bak.exists():
            shutil.copy2(bak, orig)
            bak.unlink()
            print(f"  Restored: {fname}")
        else:
            print(f"  No backup found for {fname}")
    
    return True

def main():
    print("=" * 70)
    print("  MLB DRAFT MODEL — PROSPECTIVE VALIDATION")
    print("  Training data excludes 2026 outcomes")
    print("  Models retrained from scratch")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    sys.stdout.flush()
    
    steps = [
        ("Filter training data", step_filter_training),
        ("Retrain Tier 1 (draft position)", step_retrain_tier1),
        ("Retrain Tier 2 (MLB probability)", step_retrain_tier2),
        ("Retrain Tier 3 (MLB arrival)", step_retrain_tier3),
        ("Run inference on 2026", step_run_inference),
        ("Compute accuracy + generate report", step_compute_accuracy),
        ("Restore original training data", step_restore_training),
    ]
    
    results = []
    for name, func in steps:
        print(f"\n{'#'*70}")
        print(f"# {name}")
        print(f"{'#'*70}")
        sys.stdout.flush()
        try:
            ok = func()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            ok = False
        results.append((name, ok))
        
        if not ok:
            print(f"\n  ⚠ STEP FAILED: {name}")
            print(f"  Attempting to continue with remaining steps...")
    
    # Summary
    log("VALIDATION SUMMARY")
    print(f"\n  {'Step':<45s} {'Result':<10s}")
    print(f"  {'-'*55}")
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:<45s} {status:<10s}")
    
    report_path = BASE / "analysis" / "2026_draft_accuracy_prospective.md"
    if report_path.exists():
        print(f"\n  Report: {report_path}")
    
    print(f"\n  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  {'='*70}")
    print(f"  NOTE: If training data was modified, it was restored in step 7.")
    print(f"  If a step failed, check error output above. You may need to restore")
    print(f"  manually with: python3 scripts/restore_training_data.py")
    print(f"  {'='*70}")

if __name__ == "__main__":
    main()
