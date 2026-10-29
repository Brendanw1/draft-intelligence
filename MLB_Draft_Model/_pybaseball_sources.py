#!/usr/bin/env python3
"""Inspect batting_leaders and pitching_leaders source code."""
import inspect, os

pkg_path = '/opt/anaconda3/lib/python3.12/site-packages/pybaseball'

print("=" * 60)
print("batting_leaders.py FULL SOURCE")
print("=" * 60)
with open(os.path.join(pkg_path, 'batting_leaders.py')) as f:
    print(f.read())

print("\n" + "=" * 60)
print("pitching_leaders.py FULL SOURCE")
print("=" * 60)
with open(os.path.join(pkg_path, 'pitching_leaders.py')) as f:
    print(f.read())

print("\n" + "=" * 60)
print("league_batting_stats.py FULL SOURCE (for reference)")
print("=" * 60)
with open(os.path.join(pkg_path, 'league_batting_stats.py')) as f:
    print(f.read())

print("\n" + "=" * 60)
print("league_pitching_stats.py FULL SOURCE (for reference)")
print("=" * 60)
with open(os.path.join(pkg_path, 'league_pitching_stats.py')) as f:
    print(f.read())
