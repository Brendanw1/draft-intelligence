#!/usr/bin/env python3
"""Deep dive into fangraphs datasources."""
import inspect, os

pkg_path = '/opt/anaconda3/lib/python3.12/site-packages/pybaseball'

# Read the fangraphs datasource
print("=" * 60)
print("fangraphs.py FULL SOURCE")
print("=" * 60)
with open(os.path.join(pkg_path, 'datasources', 'fangraphs.py')) as f:
    content = f.read()
    print(content)

print("\n" + "=" * 60)
print("fangraphs/league.py FULL SOURCE")
print("=" * 60)
with open(os.path.join(pkg_path, 'enums', 'fangraphs', 'league.py')) as f:
    print(f.read())

print("\n" + "=" * 60)
print("fangraphs/batting_data_enum.py FULL SOURCE")
print("=" * 60)
with open(os.path.join(pkg_path, 'enums', 'fangraphs', 'batting_data_enum.py')) as f:
    print(f.read())

print("\n" + "=" * 60)
print("fangraphs/fangraphs_stats_base.py FULL SOURCE")
print("=" * 60)
with open(os.path.join(pkg_path, 'enums', 'fangraphs', 'fangraphs_stats_base.py')) as f:
    print(f.read())
