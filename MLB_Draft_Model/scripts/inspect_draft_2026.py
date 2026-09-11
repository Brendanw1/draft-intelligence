#!/usr/bin/env python3
"""Inspect 2026 draft data."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

with open(BASE / "data" / "draft" / "draft_2026.json") as f:
    d2026 = json.load(f)

print(f"2026 draft picks: {len(d2026)}")
print(f"Keys: {list(d2026[0].keys()) if d2026 else 'empty'}")

# Count college picks
college = [p for p in d2026 if p.get("school_class") in ("JR","SR","SO","FR","GR")]
print(f"College picks: {len(college)}")
non_college = [p for p in d2026 if p.get("school_class") not in ("JR","SR","SO","FR","GR")]
print(f"HS/JUCO/other picks: {len(non_college)}")

# Show school_class distribution
sclass = {}
for p in d2026:
    sc = p.get("school_class","?")
    sclass[sc] = sclass.get(sc, 0) + 1
print(f"School class distribution: {sclass}")

print(f"Sample: {d2026[0].get('full_name')} pick={d2026[0].get('pick_number')} school={d2026[0].get('school')} class={d2026[0].get('school_class')}")
