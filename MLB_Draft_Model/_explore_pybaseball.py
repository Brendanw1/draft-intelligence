#!/usr/bin/env python3
"""Explore pybaseball MiLB data integration feasibility."""

import sys, json, os
from collections import Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Check pybaseball
print("=" * 60)
print("STEP 1: Check pybaseball availability")
print("=" * 60)
try:
    import pybaseball
    print(f"pybaseball version: {pybaseball.__version__}")
    print(f"pybaseball location: {pybaseball.__file__}")
except ImportError:
    print("pybaseball NOT installed. Installing...")
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "install", "pybaseball"],
                          capture_output=True, text=True)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    try:
        import pybaseball
        print(f"pybaseball version after install: {pybaseball.__version__}")
    except ImportError:
        print("FAILED to install pybaseball")
        sys.exit(1)

# 2. Load draft data
print("\n" + "=" * 60)
print("STEP 2: Load draft data structure")
print("=" * 60)
with open('data/draft/draft_all_picks.json') as f:
    draft_data = json.load(f)

print(f"Total picks: {len(draft_data)}")
print(f"Keys in first item: {list(draft_data[0].keys())}")
pids = set(d.get('person_id') for d in draft_data)
print(f"Unique person_ids: {len(pids)}")

years = Counter(d.get('year') for d in draft_data)
for y in sorted(years):
    print(f"  {y}: {years[y]} picks")

# Sample high-profile players
samples = [d for d in draft_data if d.get('person_id') in [621020, 669242, 686616]]
for s in samples[:3]:
    print(f"  person_id={s['person_id']}, name={s['full_name']}, year={s['year']}")

# Count college subset 2021-2025
college = [d for d in draft_data if d.get('year',0) >= 2021 and d.get('school_class') in ['JR','SR','SO','GR']]
print(f"College draftees 2021-2025 with school_class: {len(college)}")

# 3. Test MiLB endpoints
print("\n" + "=" * 60)
print("STEP 3: Test pybaseball MiLB endpoints")
print("=" * 60)

from pybaseball import milb_batting_stats, milb_pitching_stats, milb_standings
from pybaseball import playerid_reverse_lookup

# 3a. Test milb_batting_stats for a few years
for year in [2021, 2022, 2023, 2024]:
    try:
        df = milb_batting_stats(year)
        print(f"\n{year} milb_batting_stats: {len(df)} rows, columns ({len(df.columns)}): {list(df.columns[:15])}...")
        if len(df) > 0:
            print(f"  Sample rows:")
            print(f"  {df.head(2).to_string()}")
    except Exception as e:
        print(f"\n{year} milb_batting_stats: ERROR - {e}")

# 3b. Test milb_pitching_stats
for year in [2021, 2022, 2023, 2024]:
    try:
        df = milb_pitching_stats(year)
        print(f"\n{year} milb_pitching_stats: {len(df)} rows, columns ({len(df.columns)}): {list(df.columns[:15])}...")
        if len(df) > 0:
            print(f"  Sample rows:")
            print(f"  {df.head(2).to_string()}")
    except Exception as e:
        print(f"\n{year} milb_pitching_stats: ERROR - {e}")

# 3c. Check what parameters milb_batting_stats accepts
import inspect
print("\n\nmilb_batting_stats signature:")
print(inspect.signature(milb_batting_stats))
print(inspect.getsource(milb_batting_stats)[:1000])

print("\nmilb_pitching_stats signature:")
print(inspect.signature(milb_pitching_stats))
print(inspect.getsource(milb_pitching_stats)[:1000])

# 4. Check if we can query by player ID
print("\n" + "=" * 60)
print("STEP 4: Try player-specific queries")
print("=" * 60)

# Some known MLB players in the draft data
test_ids = [669242, 686616, 669242]  # These are known players
result = playerid_reverse_lookup(test_ids)
print(f"playerid_reverse_lookup: {result}")

# 5. Check the existing milb JSON data
print("\n" + "=" * 60)
print("STEP 5: Examine existing MiLB JSON data")
print("=" * 60)
for year in [2021, 2022, 2023, 2024, 2025]:
    path = f'data/milb/milb_{year}.json'
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        players = data.get('players', [])
        print(f"{year}: {len(players)} players, keys={list(data.keys())}")
        if players:
            print(f"  Sample player keys: {list(players[0].keys())}")
            # Check batting/pitching structure
            p0 = players[0]
            if 'batting' in p0:
                print(f"  Batting keys: {list(p0['batting'].keys())}")
            if 'pitching' in p0:
                print(f"  Pitching keys: {list(p0['pitching'].keys())[:15]}")
    else:
        print(f"{year}: file not found")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
