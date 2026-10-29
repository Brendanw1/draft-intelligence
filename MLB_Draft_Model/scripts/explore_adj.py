import json
with open('data/training/milb_extended_training.json') as f:
    data = json.load(f)

# Show pitcher-specific keys (non-null values)
for d in data:
    if d['player_type'] == 'pitcher':
        print("PITCHER keys with non-null/non-zero values:")
        pitcher_keys = {}
        for k, v in d.items():
            if v is not None and v != 0 and v != 0.0 and v != '':
                if isinstance(v, (int, float)) and k not in ['person_id','draft_year','draft_pick','draft_round','season','milb_person_id']:
                    pitcher_keys[k] = v
        for k, v in sorted(pitcher_keys.items()):
            print(f"  {k}: {v}")
        break

print("\n--- Checking what stats exist for pitchers ---")
for d in data:
    if d['player_type'] == 'pitcher':
        for k in ['ERA','FIP','K_per_nine','BB_per_nine','wOBA','OPS','BB_pct','K_pct']:
            if k in d:
                print(f"  {k}: {d[k]}")
        break

print("\n--- Hitter stats ---")
for d in data:
    if d['player_type'] == 'hitter':
        for k in ['wOBA','OPS','BB_pct','K_pct','ERA','FIP','K_per_nine','BB_per_nine']:
            if k in d:
                print(f"  {k}: {d[k]}")
        break

# Check target ranges
hitter_woba = [d['milb_peak_wOBA'] for d in data if d['player_type']=='hitter' and d['milb_peak_wOBA']]
pitcher_fip = [d['milb_peak_FIP'] for d in data if d['player_type']=='pitcher' and d['milb_peak_FIP']]
print(f"\nHitter milb_peak_wOBA: min={min(hitter_woba):.4f}, max={max(hitter_woba):.4f}, mean={sum(hitter_woba)/len(hitter_woba):.4f}")
print(f"Pitcher milb_peak_FIP: min={min(pitcher_fip):.4f}, max={max(pitcher_fip):.4f}, mean={sum(pitcher_fip)/len(pitcher_fip):.4f}")

# Check nulls in features
print("\n--- Null counts in key features ---")
features = ['Age','conf_strength','wOBA','OPS','BB_pct','K_pct','height_inches','bmi','draft_round','milb_peak_wOBA','milb_peak_FIP']
for feat in features:
    nulls = sum(1 for d in data if d.get(feat) is None)
    if nulls > 0:
        print(f"  {feat}: {nulls} nulls")

# Check pitcher pitching stats
print("\n--- Pitcher stat availability ---")
for d in data:
    if d['player_type'] == 'pitcher':
        for k in ['ERA','FIP','K_per_nine','BB_per_nine','K_pct','BB_pct']:
            has_not_null = sum(1 for d2 in data if d2['player_type']=='pitcher' and d2.get(k) is not None)
            print(f"  {k}: {has_not_null}/{len([x for x in data if x['player_type']=='pitcher'])} have non-null")
        break

# Check FIP_adj etc existence
print("\n--- Looking for adjusted fields by name ---")
all_keys = set()
for d in data:
    all_keys.update(d.keys())
adjusted = [k for k in all_keys if 'adj' in k.lower()]
print(f"Any adj keys: {adjusted}")
