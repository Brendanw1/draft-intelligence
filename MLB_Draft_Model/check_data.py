import json

# Check expanded training
data = json.load(open('data/training/expanded_training_set.json'))
years = set(r.get('draft_year') for r in data)
count_2026 = sum(1 for r in data if r.get('draft_year') == 2026)
print(f'expanded_training_set: {len(data)} records, years={sorted(years)}, 2026_count={count_2026}')

# Check negatives
data = json.load(open('data/training/tier2_negatives.json'))
years = set(r.get('season') for r in data)
count_2026 = sum(1 for r in data if r.get('season') == 2026)
print(f'tier2_negatives: {len(data)} records, seasons={sorted(years)}, 2026_count={count_2026}')

# Check fg_training_set
data = json.load(open('data/training/fg_training_set.json'))
years = set(r.get('draft_year') for r in data)
count_2026 = sum(1 for r in data if r.get('draft_year') == 2026)
print(f'fg_training_set: {len(data)} records, years={sorted(years)}, 2026_count={count_2026}')

# Check draft 2026 file
data = json.load(open('data/draft/draft_2026.json'))
print(f'draft_2026.json: {len(data)} records')
for r in data[:3]:
    print(f'  Sample: {r.get("full_name")} - pick {r.get("pick_number")} - school {r.get("school_name")}')
