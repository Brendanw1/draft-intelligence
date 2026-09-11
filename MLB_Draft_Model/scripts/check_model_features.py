#!/usr/bin/env python3
"""Check saved model features."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

for pt in ["hitter", "pitcher"]:
    fpath = BASE / "models" / "artifacts_full" / f"fg_features_{pt}.json"
    if fpath.exists():
        f = json.load(open(fpath))
        print(f"{pt}: {len(f['features'])} features")
        print(f"  Features: {f['features']}")
        print(f"  n_train: {f['n_train']}")
        print(f"  r2_test: {f['r2_test']}")
    else:
        print(f"No features file for {pt}")
