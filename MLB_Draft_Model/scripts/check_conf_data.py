#!/usr/bin/env python3
"""Check if conference stats/strength include 2026 data."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

cs = json.load(open(BASE / "models" / "artifacts_full" / "conference_stats.json"))
print(f"conf_stats keys: {list(cs.keys())}")

if "per_season" in cs:
    seasons = set()
    for conf, yr_data in cs["per_season"].items():
        if isinstance(yr_data, dict):
            for yr in yr_data.keys():
                seasons.add(int(yr))
    print(f"Seasons in per_season: {sorted(seasons)}")

strength = json.load(open(BASE / "models" / "artifacts_full" / "conference_strength.json"))
print(f"\nconf_strength keys: {list(strength.keys())[:5]}...")
print(f"conf_strength count: {len(strength)}")
if strength:
    sample_conf = list(strength.keys())[0]
    print(f"Sample: {sample_conf} -> {strength[sample_conf]}")
