#!/usr/bin/env python3
"""Quick test of BRef MiLB endpoints - smaller scope."""
from pybaseball.datasources.bref import BRefSession
from bs4 import BeautifulSoup
import pandas as pd

session = BRefSession()

# Test just a few MiLB levels with smaller date range
levels = ['AAA', 'AA', 'A+', 'A']

print("=" * 60)
print("TEST: MiLB levels on BRef daily.cgi (small date range)")
print("=" * 60)

for level in levels:
    # Use just one month to keep response manageable
    url = f"http://www.baseball-reference.com/leagues/daily.cgi?type=b&level={level}&lastndays=30&dates=fromandto&fromandto=2024-06-01.2024-06-30"
    resp = session.get(url)
    soup = BeautifulSoup(resp.content, 'lxml')
    tables = soup.find_all('table')
    if tables:
        rows = tables[0].find_all('tr')
        data_rows = [r for r in rows if r.find_all('td')]
        print(f"  {level} batting: {len(data_rows)} players, {len(resp.content)} bytes")
    else:
        print(f"  {level}: No table found")
    
    # Pitching too
    url = f"http://www.baseball-reference.com/leagues/daily.cgi?type=p&level={level}&lastndays=30&dates=fromandto&fromandto=2024-06-01.2024-06-30"
    resp = session.get(url)
    soup = BeautifulSoup(resp.content, 'lxml')
    tables = soup.find_all('table')
    if tables:
        rows = tables[0].find_all('tr')
        data_rows = [r for r in rows if r.find_all('td')]
        print(f"  {level} pitching: {len(data_rows)} players, {len(resp.content)} bytes")
    else:
        print(f"  {level}: No table found")

print("\n" + "=" * 60)
print("TEST: Full season AAA 2024 with smaller scope (first 2 cols only)")
print("=" * 60)

# Try with stat parameter to limit columns
url = "http://www.baseball-reference.com/leagues/daily.cgi?type=b&level=AAA&lastndays=365&dates=fromandto&fromandto=2024-04-01.2024-10-01&stat=G"
resp = session.get(url)
print(f"AAA full season batting: {len(resp.content)} bytes")

soup = BeautifulSoup(resp.content, 'lxml')
table = soup.find_all('table')[0]

# Just check headers and count rows
headers = [th.get_text().strip() for th in table.find("tr").find_all("th")]
print(f"Headers: {headers}")

rows = table.find('tbody').find_all('tr')
data_rows = [r for r in rows if r.find_all('td')]
print(f"Total data rows: {len(data_rows)}")

# Count mlbID presence
mlbid_count = 0
for row in data_rows[:100]:  # Check first 100
    row_anchor = row.find("a")
    if row_anchor and "mlb_ID=" in row_anchor.get("href", ""):
        mlbid_count += 1
print(f"First 100 rows with mlbID: {mlbid_count}")

print("\n" + "=" * 60)
print("TEST: Quick check of existing milb data for player coverage")
print("=" * 60)

import json
with open('data/draft/draft_all_picks.json') as f:
    draft = json.load(f)

# Get person_ids of drafted players 2021-2025
recent_draftees = [d for d in draft if d.get('year', 0) >= 2021]
recent_pids = set(d['person_id'] for d in recent_draftees)
print(f"Draftees 2021-2025: {len(recent_draftees)} picks, {len(recent_pids)} unique person_ids")

# Check the existing milb json for coverage
milb_players = set()
for year in [2021, 2022, 2023, 2024, 2025]:
    path = f'data/milb/milb_{year}.json'
    try:
        with open(path) as f:
            data = json.load(f)
        for p in data.get('players', []):
            milb_players.add(p.get('person_id'))
    except:
        pass

overlap = recent_pids & milb_players
print(f"Milb JSON has {len(milb_players)} unique person_ids")
print(f"Overlap with recent draftees: {len(overlap)} / {len(recent_pids)}")

# Sample the milb json structure more carefully
with open('data/milb/milb_2023.json') as f:
    milb_data = json.load(f)
print(f"\nMilb 2023 structure: {list(milb_data.keys())}")
if milb_data.get('players'):
    p = milb_data['players'][0]
    print(f"Sample player keys: {list(p.keys())}")
    if 'batting' in p:
        print(f"Batting keys: {list(p['batting'].keys())[:10]}")
    if 'pitching' in p:
        print(f"Pitching keys: {list(p['pitching'].keys())[:10]}")
    if 'level' in p:
        print(f"Level: {p['level']}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
