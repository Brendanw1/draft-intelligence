"""Check specific players in projections."""
import json

BASE = '/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model'

proj = json.load(open(f'{BASE}/data/training/projections_2026_enriched.json'))
draft = json.load(open(f'{BASE}/data/draft/draft_2026.json'))

# Check Chris Hacopian
print("=== Checking Chris Hacopian ===")
for p in proj:
    if 'hacopian' in p.get('player_name', '').lower():
        print(f"  Found: {p['player_name']} | {p['team_abb']} | Proj={p['projected_pick']} | {p.get('value_grade','')}")

# Check Cam Kozeal
print("\n=== Checking Cam Kozeal ===")
for p in proj:
    if 'kozeal' in p.get('player_name', '').lower():
        print(f"  Found: {p['player_name']} | {p['team_abb']} | Proj={p['projected_pick']} | {p.get('value_grade','')}")

# Check Cade Townsend matches
print("\n=== Checking Cade Townsend ===")
for p in proj:
    if 'tounsend' in p.get('player_name', '').lower() or 'tounsend' in p.get('player_name', '').lower():
        print(f"  Found: {p['player_name']} | {p['team_abb']} | Proj={p['projected_pick']} | {p.get('value_grade','')}")

# Quick check: how many name-only matches are there with 2+ projections same name?
from collections import defaultdict
name_only_matches = json.load(open(f'{BASE}/analysis/eval_results.json'))['matched_details']
name_only = [m for m in name_only_matches if m['match_type'] == 'name_only']

name_counts = defaultdict(list)
for m in name_only:
    name_counts[m['player']].append((m['pick'], m['school']))

multi = {k: v for k, v in name_counts.items() if len(v) > 1}
print(f"\n=== Name-only matches with multiple candidates ({len(multi)}) ===")
for name, details in sorted(multi.items())[:10]:
    print(f"  {name}: {details}")

# Check if the unmatched Chris Hacopian is really in projections
print("\n=== Searching for Texas A&M players ===")
for p in proj:
    if p.get('team_abb', '') in ['TAMU', 'ATM', 'TEXAM'] and 'Chris' in p.get('player_name', ''):
        print(f"  {p['player_name']} | {p['team_abb']} | Proj={p['projected_pick']}")
