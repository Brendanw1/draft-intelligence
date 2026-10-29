#!/usr/bin/env python3
"""Analyze existing MiLB JSON data and draft data coverage."""
import json
from collections import Counter, defaultdict

# Load draft data
with open('/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/draft/draft_all_picks.json') as f:
    draft = json.load(f)

print("=" * 60)
print("DRAFT DATA OVERVIEW")
print("=" * 60)
print(f"Total picks: {len(draft)}")
print(f"Year range: {min(d.get('year',0) for d in draft)} - {max(d.get('year',0) for d in draft)}")

# Unique person_ids
all_pids = set(d['person_id'] for d in draft)
print(f"Unique person_ids: {len(all_pids)}")

# Count by year
year_counts = Counter(d.get('year') for d in draft)
for y in sorted(year_counts):
    print(f"  {y}: {year_counts[y]}")

# Count draftees 2021-2025 (our training pool)
recent = [d for d in draft if d.get('year', 0) >= 2021]
recent_pids = set(d['person_id'] for d in recent)
print(f"\nTraining pool (2021-2025): {len(recent)} picks, {len(recent_pids)} unique person_ids")

# Position breakdown
pos_counts = Counter(d.get('position_type') for d in recent)
print(f"Position breakdown (2021-2025):")
for p, c in pos_counts.most_common():
    print(f"  {p}: {c}")

print("\n" + "=" * 60)
print("EXISTING MiLB JSON DATA COVERAGE")
print("=" * 60)

# Load all milb data
milb_players_by_year = {}
all_milb_players = set()
milb_player_details = {}  # person_id -> {years, has_batting, has_pitching, levels}

for year in [2021, 2022, 2023, 2024, 2025]:
    path = f'/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/milb/milb_{year}.json'
    try:
        with open(path) as f:
            data = json.load(f)
        players = data.get('players', [])
        year_pids = set()
        for p in players:
            pid = p.get('person_id')
            if pid:
                year_pids.add(pid)
                if pid not in milb_player_details:
                    milb_player_details[pid] = {'years': set(), 'levels': set(), 'has_batting': False, 'has_pitching': False}
                milb_player_details[pid]['years'].add(year)
                
                batting = p.get('batting', {})
                if batting and int(batting.get('gamesPlayed', 0)) > 0:
                    milb_player_details[pid]['has_batting'] = True
                    if 'batting_g' not in milb_player_details[pid]:
                        milb_player_details[pid]['batting_g'] = 0
                    milb_player_details[pid]['batting_g'] += int(batting.get('gamesPlayed', 0))
                
                pitching = p.get('pitching', {})
                if pitching and int(pitching.get('gamesPlayed', 0)) > 0:
                    milb_player_details[pid]['has_pitching'] = True
                    if 'pitching_g' not in milb_player_details[pid]:
                        milb_player_details[pid]['pitching_g'] = 0
                    milb_player_details[pid]['pitching_g'] += int(pitching.get('gamesPlayed', 0))
        
        milb_players_by_year[year] = year_pids
        all_milb_players.update(year_pids)
        print(f"  {year}: {len(players)} total, {len(year_pids)} unique person_ids")
    except Exception as e:
        print(f"  {year}: error - {e}")

print(f"\nTotal unique milb players: {len(all_milb_players)}")
overlap = recent_pids & all_milb_players
print(f"Draftees 2021-2025 with MiLB data: {len(overlap)} / {len(recent_pids)} ({100*len(overlap)/len(recent_pids):.1f}%)")

# Detailed coverage by draft year
print(f"\n=== Coverage by Draft Year (2021-2025) ===")
for year in [2021, 2022, 2023, 2024, 2025]:
    year_draftees = [d for d in recent if d.get('year') == year]
    year_pids = set(d['person_id'] for d in year_draftees)
    covered = year_pids & all_milb_players
    pct = 100 * len(covered) / len(year_pids) if year_pids else 0
    print(f"  {year}: {len(covered)}/{len(year_pids)} covered ({pct:.1f}%)")

# Stats coverage analysis
print(f"\n=== Stats Quality (for covered players) ===")
with_stats_batting = sum(1 for pid, info in milb_player_details.items() if info['has_batting'])
with_stats_pitching = sum(1 for pid, info in milb_player_details.items() if info['has_pitching'])

covered_details = [(pid, milb_player_details[pid]) for pid in overlap if pid in milb_player_details]
with_batting = sum(1 for _, info in covered_details if info['has_batting'])
with_pitching = sum(1 for _, info in covered_details if info['has_pitching'])
with_both = sum(1 for _, info in covered_details if info['has_batting'] and info['has_pitching'])

print(f"Covered draftees with batting stats: {with_batting}/{len(overlap)}")
print(f"Covered draftees with pitching stats: {with_pitching}/{len(overlap)}")

# Check the milb JSON structure more carefully
print(f"\n=== MiLB JSON Structure ===")
with open('/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/milb/milb_2023.json') as f:
    m23 = json.load(f)
print(f"Top-level keys: {list(m23.keys())}")
print(f"Players count: {len(m23.get('players', []))}")

# Sample first player with real stats
for p in m23.get('players', []):
    b = p.get('batting', {})
    if b and int(b.get('gamesPlayed', 0)) > 0:
        print(f"\nSample player with batting stats:")
        print(f"  person_id: {p['person_id']}")
        print(f"  Batting fields ({len(b)}): {list(b.keys())[:15]}")
        print(f"  Sample values: gamesPlayed={b.get('gamesPlayed')}, hits={b.get('hits')}, atBats={b.get('atBats')}, avg={b.get('avg')}")
        break

for p in m23.get('players', []):
    pch = p.get('pitching', {})
    if pch and int(pch.get('gamesPlayed', 0)) > 0:
        print(f"\nSample player with pitching stats:")
        print(f"  person_id: {p['person_id']}")
        print(f"  Pitching fields ({len(pch)}): {list(pch.keys())[:15]}")
        print(f"  Sample values: gamesPlayed={pch.get('gamesPlayed')}, wins={pch.get('wins')}, losses={pch.get('losses')}, era={pch.get('era')}")
        break

# Check if there's a "level" field in the milb data
has_level = sum(1 for p in m23.get('players', []) if 'level' in p)
print(f"\nPlayers with 'level' field: {has_level}/{len(m23.get('players', []))}")
if has_level > 0:
    level_sample = [p.get('level') for p in m23.get('players', []) if 'level' in p][:10]
    print(f"Sample levels: {level_sample}")

# Count how many have season field
has_season = sum(1 for p in m23.get('players', []) if 'season' in p)
print(f"Players with 'season' field: {has_season}")

# Check what data sources the milb JSON came from
if 'source' in m23:
    print(f"Data source: {m23['source']}")
if 'season' in m23:
    print(f"Season: {m23['season']}")

print("\n" + "=" * 60)
print("COVERAGE ESTIMATE FOR TARGET POPULATION (~1,500 draftees)")
print("=" * 60)

# Count target population
# Recent draftees 2021-2025 who signed (not necessarily MLB debut)
signed_recent = [d for d in recent if d.get('signing_bonus') is not None]
print(f"Signed draftees 2021-2025: {len(signed_recent)}")
signed_pids = set(d['person_id'] for d in signed_recent)
signed_covered = signed_pids & all_milb_players
print(f"Signed with MiLB data: {len(signed_covered)}/{len(signed_pids)} ({100*len(signed_covered)/len(signed_pids):.1f}%)")

# Players who actually made MiLB (i.e., not just drafted but played)
# vs players who never played (busts) 
college_draftees = [d for d in recent if d.get('school_class')]
print(f"\nCollege draftees 2021-2025: {len(college_draftees)}")

# Check MLB debut status
has_debut = [d for d in recent if d.get('mlb_debut_date')]
no_debut = [d for d in recent if not d.get('mlb_debut_date')]
print(f"Already debuted in MLB: {len(has_debut)}")
print(f"Not yet debuted: {len(no_debut)}")
debuted_covered = set(d['person_id'] for d in has_debut) & all_milb_players
not_debuted_covered = set(d['person_id'] for d in no_debut) & all_milb_players
print(f"Debuted with MiLB data: {len(debuted_covered)}/{len(has_debut)}")
print(f"Not-debuted with MiLB data: {len(not_debuted_covered)}/{len(no_debut)}")

print("\nDone!")
