#!/usr/bin/env python3
"""Explore pybaseball batting_stats and pitching_stats for MiLB capabilities."""
import inspect, sys

# First, figure out the source module
import pybaseball
from pybaseball import batting_stats, pitching_stats
from pybaseball import batting_stats_bref, pitching_stats_bref
from pybaseball import playerid_reverse_lookup, playerid_lookup

print("=" * 60)
print("batting_stats signature:")
print("=" * 60)
try:
    sig = inspect.signature(batting_stats)
    print(f"  {sig}")
except:
    pass
src = inspect.getsource(batting_stats)
print(f"\nSource (first 50 lines):")
for line in src.split('\n')[:50]:
    print(f"  {line}")

print("\n" + "=" * 60)
print("pitching_stats signature:")
print("=" * 60)
try:
    sig = inspect.signature(pitching_stats)
    print(f"  {sig}")
except:
    pass
src = inspect.getsource(pitching_stats)
print(f"\nSource (first 50 lines):")
for line in src.split('\n')[:50]:
    print(f"  {line}")

print("\n" + "=" * 60)
print("batting_stats_bref signature:")
print("=" * 60)
try:
    sig = inspect.signature(batting_stats_bref)
    print(f"  {sig}")
except:
    pass
src = inspect.getsource(batting_stats_bref)
print(f"\nSource (first 50 lines):")
for line in src.split('\n')[:50]:
    print(f"  {line}")

print("\n" + "=" * 60)
print("playerid_reverse_lookup signature:")
print("=" * 60)
try:
    sig = inspect.signature(playerid_reverse_lookup)
    print(f"  {sig}")
except:
    pass
