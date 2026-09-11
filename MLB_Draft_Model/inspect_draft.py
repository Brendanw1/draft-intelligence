import json

# Check draft 2026 structure
draft = json.load(open('data/draft/draft_2026.json'))
print('=== DRAFT 2026 ===')
print(f'Total records: {len(draft)}')

# Check a few records
for r in draft[:3]:
    keys = list(r.keys())
    print(f'Keys: {keys}')
    for k, v in r.items():
        print(f'  {k}: {v}')
    print()

# Check all keys
print('All keys:')
all_keys = set()
for r in draft:
    all_keys.update(r.keys())
print(sorted(all_keys))

# Check for person_id availability
has_person_id = sum(1 for r in draft if r.get('person_id'))
print(f'\nHas person_id: {has_person_id}/{len(draft)}')

# Check school_name
for r in draft[:5]:
    print(f'  {r.get("full_name")}: school="{r.get("school_name")}", person_id={r.get("person_id")}')

# Check draft_all_picks for 2026
all_picks = json.load(open('data/draft/draft_all_picks.json'))
draft_2026_from_all = [r for r in all_picks if r.get('draft_year') == 2026]
print(f'\n=== DRAFT ALL PICKS 2026 ===')
print(f'Total 2026 records: {len(draft_2026_from_all)}')
if draft_2026_from_all:
    r = draft_2026_from_all[0]
    print(f'Keys: {list(r.keys())}')
    for k, v in list(r.items())[:15]:
        print(f'  {k}: {v}')
