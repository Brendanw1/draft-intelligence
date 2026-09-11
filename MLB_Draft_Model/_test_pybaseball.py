#!/usr/bin/env python3
"""Test pybaseball endpoints and explore alternatives."""
import sys, json, time

from pybaseball import batting_stats, pitching_stats
from pybaseball import batting_stats_bref, pitching_stats_bref
from pybaseball import playerid_reverse_lookup, playerid_lookup
from pybaseball import amateur_draft

print("=" * 60)
print("TEST 1: batting_stats (Fangraphs) - MLB 2024")
print("=" * 60)
try:
    df = batting_stats(2024, qual=1)
    print(f"Rows: {len(df)}, Columns: {list(df.columns[:15])}")
    print(f"First 2 rows:")
    print(df.head(2).to_string())
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("TEST 2: Check if 'league' param changes output")
print("=" * 60)
# Check if different league codes work
from pybaseball.datasources.fangraphs import fg_batting_data
import inspect
print(f"fg_batting_data signature: {inspect.signature(fg_batting_data)}")

# Try with a known player's Fangraphs ID
# Judge has FG id 15650, Trout has 545361
# These are NOT the same as MLBAM person_ids
print("\nTesting playerid_reverse_lookup...")
test_ids = [545361, 592450, 660271, 621020]  # Some MLBAM IDs
try:
    result = playerid_reverse_lookup(test_ids, key_type='mlbam')
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("TEST 3: Try playerid_lookup by name")
print("=" * 60)
try:
    result = playerid_lookup('Swanson', 'Dansby')
    print(f"Swanson lookup: {result}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("TEST 4: Try amateur_draft endpoint")
print("=" * 60)
try:
    df = amateur_draft(2021)
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    print(df.head(3).to_string())
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("TEST 5: Try batting_stats_bref (Baseball Reference)")
print("=" * 60)
try:
    df = batting_stats_bref(2024)
    print(f"Rows: {len(df)}, Columns: {list(df.columns[:15])}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
