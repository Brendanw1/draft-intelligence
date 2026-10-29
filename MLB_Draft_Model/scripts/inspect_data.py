"""Inspect data structures for milb and expanded_training_set."""
import json
import sys
import os

# Load milb data
print("=== MiLB 2021 ===")
with open('data/milb/milb_2021.json') as f:
    d = json.load(f)
players = d['players']
print(f"Total players: {len(players)}")

# Find a player with level
for p in players:
    if p.get('level'):
        print(f"\nSample player keys: {list(p.keys())}")
        print(f"person_id: {p['person_id']}")
        print(f"full_name: {p['full_name']}")
        print(f"level: {p['level']}")
        print(f"team_name: {p.get('team_name')}")
        print(f"season: {p.get('season')}")
        print(f"batting keys: {list(p.get('batting', {}).keys()) if p.get('batting') else 'None'}")
        print(f"pitching keys: {list(p.get('pitching', {}).keys()) if p.get('pitching') else 'None'}")
        break

# Count unique person_ids across all years
print("\n=== All MiLB years ===")
all_ids = set()
year_counts = {}
for year in [2021, 2022, 2023, 2024, 2025]:
    with open(f'data/milb/milb_{year}.json') as f:
        dd = json.load(f)
    pp = dd['players']
    year_counts[year] = len(pp)
    ids = set(p['person_id'] for p in pp)
    all_ids.update(ids)
    print(f"{year}: {len(pp)} players, {len(ids)} unique IDs")
print(f"Total unique person_ids across all years: {len(all_ids)}")

# Count levels
levels = {}
for year in [2021, 2022, 2023, 2024, 2025]:
    with open(f'data/milb/milb_{year}.json') as f:
        dd = json.load(f)
    for p in dd['players']:
        lvl = p.get('level')
        levels[lvl] = levels.get(lvl, 0) + 1
print(f"\nLevel distribution: {levels}")

print("\n=== Expanded Training Set ===")
with open('data/training/expanded_training_set.json') as f:
    ets = json.load(f)
print(f"Total records: {len(ets)}")
print(f"Keys in first record: {list(ets[0].keys())}")

# Count signed draftees
signed = [r for r in ets if r.get('draft_pick') and r['draft_pick'] > 0]
print(f"Signed draftees (draft_pick > 0): {len(signed)}")

# Count by player_type
from collections import Counter
pt = Counter(r.get('player_type') for r in signed)
print(f"Player type distribution: {dict(pt)}")

# Count by draft_year
dy = Counter(r.get('draft_year') for r in signed)
print(f"Draft year distribution: {dict(sorted(dy.items()))}")

# Sample hitter and pitcher
for r in signed:
    if r.get('player_type') == 'hitter':
        print(f"\nSample hitter: {json.dumps(r, indent=2)[:500]}")
        break
for r in signed:
    if r.get('player_type') == 'pitcher':
        print(f"\nSample pitcher: {json.dumps(r, indent=2)[:500]}")
        break

print("\nDone!")
