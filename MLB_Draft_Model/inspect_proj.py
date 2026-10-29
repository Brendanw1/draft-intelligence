import json

# Check enriched projections structure
proj = json.load(open('data/training/projections_2026_enriched.json'))
print('=== ENRICHED PROJECTIONS ===')
print(f'Total records: {len(proj)}')

# Check a few records
for r in proj[:3]:
    keys = list(r.keys())
    print(f'Keys: {keys}')
    print(f'  name: {r.get("name", r.get("Name", "N/A"))}')
    print(f'  team_abb: {r.get("team_abb", r.get("Team", r.get("School", "N/A")))}')
    print(f'  pick_pred: {r.get("pick_pred", r.get("pick_number_pred", "N/A"))}')
    print()

# Check what we have for matching
sample = proj[0]
for k, v in sample.items():
    if k in ['name', 'Name', 'team_abb', 'Team', 'School', 'school_name', 'person_id', 'xMLBAMID', 'pitcher', 'hitter', 'position', 'pos', 'pick_pred', 'round_pred', 'pick_number_pred', 'predicted_pick', 'draft_year', 'source']:
        print(f'{k}: {v}')
    elif k in ['full_name', 'player_name', 'first_name', 'last_name']:
        print(f'{k}: {v}')

# Check all keys
print('\nAll keys (first 50):')
all_keys = set()
for r in proj:
    all_keys.update(r.keys())
print(sorted(list(all_keys))[:50])
print(f'Total unique keys: {len(all_keys)}')
