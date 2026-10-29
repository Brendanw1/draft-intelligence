#!/usr/bin/env python3
"""Check for 2026 data leakage in training sets."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

# Check expanded training set
with open(BASE / "data" / "training" / "expanded_training_set.json") as f:
    data = json.load(f)

draft_years = {}
for r in data:
    dy = r.get("draft_year")
    draft_years[dy] = draft_years.get(dy, 0) + 1

print("Expanded training set draft_year distribution:")
for k in sorted(draft_years.keys()):
    print(f"  {k}: {draft_years[k]}")
print(f"\n  2026 records: {draft_years.get(2026, 0)}")
print(f"  Total: {len(data)}")

# Check tier2 negatives
with open(BASE / "data" / "training" / "tier2_negatives.json") as f:
    negs = json.load(f)

seasons = {}
for r in negs:
    s = r.get("season")
    seasons[s] = seasons.get(s, 0) + 1

print("\nTier 2 negatives season distribution:")
for k in sorted(seasons.keys()):
    print(f"  {k}: {seasons[k]}")
print(f"  2026 records: {seasons.get(2026, 0)}")
print(f"  Total: {len(negs)}")
