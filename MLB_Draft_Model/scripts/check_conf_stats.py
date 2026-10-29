import json

cs = json.load(open('models/artifacts_full/conference_stats.json'))
print('Keys:', list(cs.keys()))
print('per_season keys sample:', list(cs.get('per_season', {}).keys())[:5])
print('conference_overall keys sample:', list(cs.get('conference_overall', {}).keys())[:5])
# Check one conference's data
for conf, seasons in list(cs.get('per_season', {}).items())[:3]:
    for yr, data in list(seasons.items())[:1]:
        if isinstance(data, dict) and 'hitter' in data:
            print(f'\n{conf}, year {yr}:')
            print(f'  hitter keys: {list(data["hitter"].keys())[:10]}')
            print(f'  pitcher keys: {list(data["pitcher"].keys())[:10] if "pitcher" in data else "N/A"}')
            print(f'  hitter wOBA: {data["hitter"].get("wOBA")}')
            print(f'  hitter OPS: {data["hitter"].get("OPS")}')
            break
