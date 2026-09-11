#!/usr/bin/env python3
"""
restore_training_data.py — Restore original training data from .bak files.
"""
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "training"

files_to_restore = [
    "expanded_training_set.json",
    "tier2_negatives.json",
    "fg_training_set.json",
]

print("=" * 60)
print("RESTORING ORIGINAL TRAINING DATA")
print("=" * 60)

for fname in files_to_restore:
    orig = DATA_DIR / fname
    bak = DATA_DIR / (fname + ".bak")
    if bak.exists():
        shutil.copy2(bak, orig)
        bak.unlink()
        print(f"  Restored: {fname} (from {bak.name})")
    else:
        print(f"  No backup found for {fname} — cannot restore")

print("\nRestore complete.")
