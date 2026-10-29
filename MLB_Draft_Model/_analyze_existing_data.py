#!/usr/bin/env python3
"""Check existing milb JSON data and test one BRef query."""
import os, json, sys
sys.path.insert(0, '/opt/anaconda3/lib/python3.12/site-packages')

# First check existing data
print("=" * 60)
print("EXISTING MiLB JSON DATA ANALYSIS")
print("=" * 60)

import json
with open('data/draft/draft_all_picks.json') as f:
    draft = json.load(f)

# Get person_ids of drafted players 2021-2025
recent_draftees = [d for d in draft if d.get('year', 0) >= 2021]
recent_pids = set(d['person_id'] for d in recent_draftees)
print(f"Draftees 2021-2025: {len(recent_draftees)} picks, {len(recent_pids)} unique person_ids")

# Check the existing milb json for coverage
milb_players_by_year = {}
all_milb_players = set()
for year in [2021, 2022, 2023, 2024, 2025]:
    path = f'data/milb/milb_{year}.json'
    try:
        with open(path) as f:
            data = json.load(f)
        players = data.get('players', [])
        milb_players_by_year[year] = set(p.get('person_id') for p in players)
        all_milb_players.update(p.get('person_id') for p in players)
        print(f"  {year}: {len(players)} players in milb JSON")
    except Exception as e:
        print(f"  {year}: error - {e}")

overlap = recent_pids & all_milb_players
print(f"\nTotal unique milb JSON players: {len(all_milb_players)}")
print(f"Overlap with recent draftees: {len(overlap)} / {len(recent_pids)} ({100*len(overlap)/len(recent_pids):.1f}%)")

# Sample milb JSON structure
with open('data/milb/milb_2023.json') as f:
    milb_data = json.load(f)
print(f"\nMilb JSON top-level keys: {list(milb_data.keys())}")

players = milb_data.get('players', [])
print(f"Sample player (first with non-zero batting):")
for p in players[:20]:
    batting = p.get('batting', {})
    if batting and int(batting.get('gamesPlayed', 0)) > 0:
        print(f"  person_id={p['person_id']}")
        print(f"  Batting gamesPlayed: {batting['gamesPlayed']}")
        print(f"  Batting keys: {list(batting.keys())}")
        break
for p in players[:20]:
    pitching = p.get('pitching', {})
    if pitching and int(pitching.get('gamesPlayed', 0)) > 0:
        print(f"\n  Pitching player person_id={p['person_id']}")
        print(f"  Pitching gamesPlayed: {pitching['gamesPlayed']}")
        print(f"  Pitching keys: {list(pitching.keys())[:20]}")
        break

# Check what levels are in the data
levels = set()
for p in players[:500]:
    if 'level' in p:
        levels.add(p.get('level'))
print(f"\nLevel values found: {sorted(levels)[:20] if levels else 'No level field'}")

# Check how many have season-level data vs peak-only
has_stats = 0
for p in players[:1000]:
    if p.get('batting', {}).get('gamesPlayed', 0) > 0 or p.get('pitching', {}).get('gamesPlayed', 0) > 0:
        has_stats += 1
print(f"Players with stats (first 1000 checked): {has_stats}")

# Quick check for 2025 data
print("\n--- 2025 milb JSON check ---")
with open('data/milb/milb_2025.json') as f:
    m25 = json.load(f)
p25 = m25.get('players', [])
print(f"2025 players: {len(p25)}")
# Count with stats
with_stats = sum(1 for p in p25 if p.get('batting', {}).get('gamesPlayed', 0) > 0 or p.get('pitching', {}).get('gamesPlayed', 0) > 0)
print(f"With non-zero games: {with_stats}")

# Now try ONE BRef query to see if MiLB data is accessible
print("\n" + "=" * 60)
print("ONE BRef MiLB test")
print("=" * 60)
from pybaseball.datasources.bref import BRefSession
from bs4 import BeautifulSoup

session = BRefSession()

# Try the BRef MiLB register/affiliates page
url = "https://www.baseball-reference.com/register/affiliate.cgi"
try:
    resp = session.get(url)
    print(f"Affiliate page: {resp.status_code}, {len(resp.content)} bytes")
    # Check for minor league content
    text = resp.text.lower()
    has_milb = 'minor' in text or 'aaa' in text or 'aa' in text
    print(f"Has MiLB content: {has_milb}")
except Exception as e:
    print(f"ERROR: {e}")

# Try the BRef minor league batting leaders
url = "https://www.baseball-reference.com/register/leader.cgi?type=b&group=Minor&class=AAA"
try:
    resp = session.get(url)
    print(f"\nMiLB AAA batting leaders: {resp.status_code}, {len(resp.content)} bytes")
    soup = BeautifulSoup(resp.content, 'lxml')
    tables = soup.find_all('table')
    print(f"Tables: {len(tables)}")
    if tables:
        rows = tables[0].find_all('tr')[:5]
        for r in rows:
            print(f"  {[c.get_text().strip() for c in r.find_all(['th','td'])][:8]}")
except Exception as e:
    print(f"ERROR: {e}")

print("\nDone!")
