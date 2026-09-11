#!/usr/bin/env python3
"""Deep test of BRef stats and crosswalk."""
import sys

from pybaseball import batting_stats_bref, pitching_stats_bref
from pybaseball import playerid_reverse_lookup, playerid_lookup
import pandas as pd

print("=" * 60)
print("TEST: batting_stats_bref - check ALL columns")
print("=" * 60)
try:
    df = batting_stats_bref(2024)
    print(f"Rows: {len(df)}")
    print(f"ALL columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"  {i}: '{col}'")
    print(f"\nLev column unique values: {sorted(df['Lev'].unique())}")
    print(f"\nSample data:")
    print(df.head(3).to_string())
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("TEST: pitching_stats_bref - check ALL columns")
print("=" * 60)
try:
    df = pitching_stats_bref(2024)
    print(f"Rows: {len(df)}")
    print(f"ALL columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"  {i}: '{col}'")
    print(f"\nLev column unique values: {sorted(df['Lev'].unique())}")
    print(df.head(3).to_string())
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("TEST: Check if we can modify BRef URL for MiLB")
print("=" * 60)
# Look at the source of batting_stats_range
from pybaseball import league_batting_stats
import inspect
print("batting_stats_range source:")
src = inspect.getsource(league_batting_stats)
print(src[:2000])

print("\n" + "=" * 60)
print("TEST: playerid_reverse_lookup - all columns")
print("=" * 60)
test_ids = [621020, 669242, 592450, 660271, 686616, 545361]
try:
    result = playerid_reverse_lookup(test_ids, key_type='mlbam')
    print(f"ALL columns ({len(result.columns)}):")
    for i, col in enumerate(result.columns):
        print(f"  {i}: '{col}'")
    print(f"\nData:")
    print(result.to_string())
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("TEST: Check the BRef session for custom URL support")
print("=" * 60)
from pybaseball.datasources.bref import BRefSession
session = BRefSession()
try:
    # Try the daily.cgi URL with different level
    url = "http://www.baseball-reference.com/leagues/daily.cgi?type=b&level=AAA&lastndays=7"
    resp = session.get(url)
    print(f"AAA batting status: {resp.status_code}, len={len(resp.content)}")
    if 'minor' in resp.text.lower() or 'aaa' in resp.text.lower():
        print("  Contains MiLB reference!")
    else:
        print(f"  Content preview: {resp.text[:500]}")
except Exception as e:
    print(f"ERROR: {e}")

try:
    # Try the full daily.cgi URL with a date range for MiLB
    url = "http://www.baseball-reference.com/leagues/daily.cgi?user_team=&bust_cache=&type=b&lastndays=7&dates=fromandto&fromandto=2024-04-01.2024-10-01&level=AAA&franch=&stat=&stat_value=0"
    resp = session.get(url)
    print(f"\nAAA specific batting status: {resp.status_code}, len={len(resp.content)}")
    if resp.status_code == 200:
        # Try to parse table
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.content, 'lxml')
        tables = soup.find_all('table')
        print(f"Tables found: {len(tables)}")
        if tables:
            rows = tables[0].find_all('tr')
            print(f"Rows in first table: {len(rows)}")
            if len(rows) > 1:
                headers = [th.get_text() for th in rows[0].find_all('th')]
                print(f"Headers: {headers}")
    else:
        print(f"  Content: {resp.text[:500]}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
