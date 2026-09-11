#!/usr/bin/env python3
"""Deep analysis of existing MiLB JSON stats."""
import json
from collections import Counter

# Load the most complete data
print("=" * 60)
print("DEEP MiLB JSON ANALYSIS")
print("=" * 60)

# Get all batting and pitching columns across all years
batting_cols = set()
pitching_cols = set()

all_batting_stats = {}  # column -> [sample values]
all_pitching_stats = {}

for year in [2021, 2022, 2023, 2024, 2025]:
    path = f'/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/milb/milb_{year}.json'
    with open(path) as f:
        data = json.load(f)
    
    for p in data.get('players', []):
        b = p.get('batting', {})
        if b:
            batting_cols.update(b.keys())
            for k, v in b.items():
                if k not in all_batting_stats:
                    all_batting_stats[k] = set()
                all_batting_stats[k].add(str(v)[:30])
        
        pch = p.get('pitching', {})
        if pch:
            pitching_cols.update(pch.keys())
            for k, v in pch.items():
                if k not in all_pitching_stats:
                    all_pitching_stats[k] = set()
                all_pitching_stats[k].add(str(v)[:30])

print(f"\nAll Batting Columns ({len(batting_cols)}):")
for col in sorted(batting_cols):
    sample_vals = list(all_batting_stats.get(col, set()))[:3]
    print(f"  {col}: {sample_vals}")

print(f"\nAll Pitching Columns ({len(pitching_cols)}):")
for col in sorted(pitching_cols):
    sample_vals = list(all_pitching_stats.get(col, set()))[:3]
    print(f"  {col}: {sample_vals}")

# Check player fields (non-batting/pitching)
print(f"\n=== Player-level fields ===")
with open('/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/milb/milb_2023.json') as f:
    data = json.load(f)
p = data['players'][0]
player_fields = [k for k in p.keys() if k not in ('batting', 'pitching')]
print(f"Player fields: {player_fields}")

# Check multiple seasons for same player
print(f"\n=== Multi-season stats (person_id stability check) ===")
# Find a player present in multiple years
from collections import defaultdict
player_years = defaultdict(list)
for year in [2021, 2022, 2023, 2024]:
    path = f'/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/milb/milb_{year}.json'
    with open(path) as f:
        data = json.load(f)
    for p in data.get('players', []):
        pid = p.get('person_id')
        if pid:
            player_years[pid].append(year)

multi_year = [(pid, years) for pid, years in player_years.items() if len(years) >= 3]
print(f"Players with 3+ years of data: {len(multi_year)}")
if multi_year:
    pid = multi_year[0][0]
    print(f"\nExample: person_id={pid}")
    for year in [2021, 2022, 2023, 2024]:
        path = f'/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/milb/milb_{year}.json'
        with open(path) as f:
            data = json.load(f)
        for p in data.get('players', []):
            if p.get('person_id') == pid:
                b = p.get('batting', {})
                pch = p.get('pitching', {})
                level = p.get('level', '?')
                if b and int(b.get('gamesPlayed', 0)) > 0:
                    print(f"  {year} ({level}): Batting G={b['gamesPlayed']}, PA={b.get('plateAppearances','?')}, AVG={b.get('avg','?')}")
                if pch and int(pch.get('gamesPlayed', 0)) > 0:
                    print(f"  {year} ({level}): Pitching G={pch['gamesPlayed']}, ERA={pch.get('era','?')}, IP={pch.get('inningsPitched','?')}")
                break

# Check how levels change over time for same players
print(f"\n=== Level progression (first 5 multi-year players) ===")
for pid, years in multi_year[:5]:
    levels = {}
    for year in years:
        path = f'/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/milb/milb_{year}.json'
        with open(path) as f:
            data = json.load(f)
        for p in data.get('players', []):
            if p.get('person_id') == pid:
                levels[year] = p.get('level', '?')
                break
    print(f"  {pid}: {levels}")

# Count level distribution across all years
print(f"\n=== Level Distribution (2023) ===")
level_counts = Counter()
for p in data.get('players', []):
    level_counts[p.get('level', 'unknown')] += 1
for level, count in level_counts.most_common():
    print(f"  {level}: {count}")

# Key: how many draftees 2021-2025 have stats by level?
print(f"\n=== Draftee MiLB stats by level (2023 sample) ===")
with open('/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/draft/draft_all_picks.json') as f:
    draft = json.load(f)
draft_pids = set(d['person_id'] for d in draft if d.get('year', 0) >= 2021)

# For 2023 milb data, count by level for draft-relevant players
level_draftee_counts = Counter()
for p in data.get('players', []):
    if p.get('person_id') in draft_pids:
        level_draftee_counts[p.get('level', 'unknown')] += 1
print("  Drafted players in 2023 milb data by level:")
for level, count in level_draftee_counts.most_common():
    print(f"    {level}: {count}")

# Estimate how many batters and pitchers we have
signed_draftees = [d for d in draft if d.get('year', 0) >= 2021 and d.get('signing_bonus') is not None]
signed_pids = set(d['person_id'] for d in signed_draftees)

# Load all milb data and check position
all_pitcher_pids = set()
all_batter_pids = set()
for year in [2021, 2022, 2023, 2024]:
    path = f'/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/milb/milb_{year}.json'
    with open(path) as f:
        data = json.load(f)
    for p in data.get('players', []):
        pid = p.get('person_id')
        if pid in signed_pids:
            b = p.get('batting', {})
            pch = p.get('pitching', {})
            if b and int(b.get('gamesPlayed', 0)) > 5:
                all_batter_pids.add(pid)
            if pch and int(pch.get('gamesPlayed', 0)) > 5:
                all_pitcher_pids.add(pid)

print(f"\n=== Signed draftees 2021-2025 with meaningful stats (>5 games) ===")
print(f"  Signed draftees: {len(signed_pids)}")
print(f"  With batting stats: {len(all_batter_pids)}")
print(f"  With pitching stats: {len(all_pitcher_pids)}")
print(f"  With either: {len(all_batter_pids | all_pitcher_pids)}")

# What features would be useful?
print("\n=== USEFUL FEATURES FOR MODELING ===")
# For hitters: wOBA, wRC+, AVG, OBP, SLG, OPS, BB%, K%, ISO, BABIP
useful_batting = ['avg', 'obp', 'slg', 'ops', 'babip', 'homeRuns', 'strikeOuts', 'baseOnBalls', 'stolenBases', 'rbi', 'runs', 'doubles', 'triples', 'plateAppearances']
print(f"Useful batting features available:")
for col in useful_batting:
    available = col in batting_cols
    print(f"  {col}: {'✅' if available else '❌'}")

# For pitchers: ERA, FIP, WHIP, K/9, BB/9, HR/9, BABIP
useful_pitching = ['era', 'whip', 'wins', 'losses', 'saves', 'inningsPitched', 'strikeOuts', 'baseOnBalls', 'homeRuns', 'holds', 'blownSaves', 'era', 'strikeOutsPer9Inn', 'walksPer9Inn', 'homeRunsPer9', 'babip']
print(f"\nUseful pitching features available:")
for col in useful_pitching:
    available = col in pitching_cols
    print(f"  {col}: {'✅' if available else '❌'}")

print("\nDone!")
