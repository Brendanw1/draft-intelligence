"""Inspect pitcher records in expanded_training_set."""
import json

with open('data/training/expanded_training_set.json') as f:
    ets = json.load(f)

# Find pitchers with more data
pitchers = [r for r in ets if r.get('player_type') == 'pitcher']
print(f"Total pitchers: {len(pitchers)}")
print(f"Sample pitcher keys: {list(pitchers[0].keys())}")

# Check what pitching-specific fields exist
pitching_fields = set()
for p in pitchers:
    pitching_fields.update(k for k in p.keys() if k not in ('person_id', 'player_name', 'player_type', 'season', 'draft_year', 'draft_pick', 'draft_round', 'draft_bonus', 'draft_team', 'draft_school', 'draft_position', 'height_raw', 'height_inches', 'weight', 'bmi', 'bats', 'throws', 'conference', 'conference_tier'))

# Actually let me check for FIP, ERA, K/9 etc.
interesting = ['FIP', 'fip', 'ERA', 'K_9', 'BB_9', 'HR_9', 'K_pct', 'BB_pct', 'WHIP', 'whip']
for field in interesting:
    count = sum(1 for p in pitchers if p.get(field) is not None)
    if count > 0:
        print(f"  {field}: present in {count}/{len(pitchers)} pitchers")

# Show full first pitcher record to understand pitcher stats
print("\n=== Full first pitcher ===")
print(json.dumps(pitchers[0], indent=2)[:800])

# Check if there are pitcher-specific fields like W, L, ERA
has_pitching_stats = [k for k in pitchers[0].keys() if k not in ('person_id', 'player_name', 'player_type', 'season', 'draft_year', 'draft_pick', 'draft_round', 'draft_bonus', 'draft_team', 'draft_school', 'draft_position', 'height_raw', 'height_inches', 'weight', 'bmi', 'bats', 'throws', 'conference', 'conference_tier')]
print(f"\nFirst pitcher non-base fields: {has_pitching_stats}")
print(f"\nFirst pitcher record:")
for k in ['W', 'L', 'ERA', 'G', 'GS', 'IP', 'SO', 'BB', 'H', 'HR']:
    print(f"  {k}: {pitchers[0].get(k)}")
