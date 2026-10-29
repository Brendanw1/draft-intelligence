import json
with open('data/training/milb_extended_training.json') as f:
    data = json.load(f)

# Check for adjusted features
adj_keys = set()
for d in data:
    for k in d:
        if 'adj' in k.lower():
            adj_keys.add(k)
print("Fields with 'adj':", sorted(adj_keys))

# Check missing values
for d in data:
    if d['player_type'] == 'hitter':
        missing = [k for k in ['wOBA_adj','OPS_adj','BB_pct_adj','K_pct_adj'] if k not in d]
        if missing:
            print(f"Hitter missing fields: {missing}")
        break
    if d['player_type'] == 'pitcher':
        missing = [k for k in ['ERA_adj','FIP_adj','K_per_nine_adj','BB_per_nine_adj'] if k not in d]
        if missing:
            print(f"Pitcher missing fields: {missing}")
        break

# List ALL keys
print("\nAll keys in first record:")
for d in data:
    print(sorted(d.keys()))
    break

print("\nCount missing milb_peak_wOBA for hitters:", sum(1 for d in data if d['player_type']=='hitter' and d.get('milb_peak_wOBA') is None))
print("Count missing milb_peak_FIP for pitchers:", sum(1 for d in data if d['player_type']=='pitcher' and d.get('milb_peak_FIP') is None))
