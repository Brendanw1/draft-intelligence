#!/usr/bin/env python3
"""Quick exploration of milb_extended_training.json"""
import json, sys
sys.path.insert(0, '.')
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
path = BASE / "data" / "training" / "milb_extended_training.json"

d = json.load(open(path))
print(f"Total records: {len(d)}")

# Get first record's keys
print(f"\nAll keys in first record:")
for k in sorted(d[0].keys()):
    print(f"  {k}: {type(d[0][k]).__name__} = {d[0][k]}")

# Count by year
from collections import Counter
dy = Counter(r.get("draft_year") for r in d)
print(f"\nBy draft year: {dict(sorted(dy.items()))}")

# By player type
pt = Counter(r.get("player_type") for r in d)
print(f"By player type: {dict(pt)}")

# Level distribution
ld = Counter(r.get("milb_year1_level") for r in d)
print(f"By milb_year1_level: {dict(sorted(ld.items()))}")

# MLB debut rate
debuted = sum(1 for r in d if r.get("has_mlb_debut"))
print(f"MLB debut rate: {debuted}/{len(d)} ({100*debuted/len(d):.1f}%)")

# Check for milb_year1_wOBA availability
has_woba = sum(1 for r in d if r.get("milb_year1_wOBA") is not None and r.get("milb_year1_wOBA", 0) > 0)
has_fip = sum(1 for r in d if r.get("milb_year1_FIP") is not None and r.get("milb_year1_FIP", 0) > 0)
print(f"\nHitters with milb_year1_wOBA>0: {has_woba}")
print(f"Pitchers with milb_year1_FIP>0: {has_fip}")

# Check key fields
for field in ["milb_year1_level", "milb_year1_games", "milb_year1_pa", "milb_year1_ip"]:
    nulls = sum(1 for r in d if r.get(field) is None)
    print(f"  {field}: {nulls} nulls")

# Check adjusted fields exist
for field in ["wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj", "ERA_adj", "FIP_adj", "K_per_nine_adj", "BB_per_nine_adj"]:
    present = sum(1 for r in d if field in r)
    print(f"  {field}: present in {present}/{len(d)} records")

# Check milb_year1_wOBA distribution
hitters = [r for r in d if r.get("player_type") == "hitter"]
wobas = [r.get("milb_year1_wOBA", 0) for r in hitters if r.get("milb_year1_wOBA") is not None]
if wobas:
    import numpy as np
    print(f"\nHitter milb_year1_wOBA: mean={np.mean(wobas):.4f}, min={min(wobas):.4f}, max={max(wobas):.4f}")

pitchers = [r for r in d if r.get("player_type") == "pitcher"]
fips = [r.get("milb_year1_FIP", 0) for r in pitchers if r.get("milb_year1_FIP") is not None]
if fips:
    import numpy as np
    print(f"Pitcher milb_year1_FIP: mean={np.mean(fips):.4f}, min={min(fips):.4f}, max={max(fips):.4f}")

# Check round_logit_prior and nn_mlb_rate
print(f"\nround_logit_prior: mean={np.mean([r.get('round_logit_prior',0) for r in d]):.4f}")
print(f"nn_mlb_rate: mean={np.mean([r.get('nn_mlb_rate',0) for r in d]):.4f}")
