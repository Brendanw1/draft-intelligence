#!/usr/bin/env python3
"""Explore pybaseball datasources for MiLB support."""
import inspect, os

pkg_path = '/opt/anaconda3/lib/python3.12/site-packages/pybaseball'

print("=== datasources directory ===")
ds_path = os.path.join(pkg_path, 'datasources')
for root, dirs, files in os.walk(ds_path):
    for f in files:
        if f.endswith('.py'):
            rel = os.path.relpath(os.path.join(root, f), ds_path)
            print(f"  {rel}")

print("\n=== enums directory ===")
enum_path = os.path.join(pkg_path, 'enums')
for root, dirs, files in os.walk(enum_path):
    for f in files:
        if f.endswith('.py'):
            rel = os.path.relpath(os.path.join(root, f), enum_path)
            print(f"  {rel}")

# Check fg_batting_data
print("\n=== fg_batting_data signature ===")
from pybaseball.datasources.fangraphs import fg_batting_data
print(f"type: {type(fg_batting_data)}")
try:
    sig = inspect.signature(fg_batting_data.fetch)
    print(f"fetch signature: {sig}")
except Exception as e:
    print(f"Error on fetch sig: {e}")

# Check enums for league options
print("\n=== Checking enum files ===")
enum_path = os.path.join(pkg_path, 'enums')
import importlib.util
for root, dirs, files in os.walk(enum_path):
    for f in files:
        if f.endswith('.py') and not f.startswith('__'):
            rel = os.path.relpath(os.path.join(root, f), enum_path)
            # Try to load it
            module_path = os.path.join(root, f)
            with open(module_path) as fh:
                content = fh.read()
            if 'milb' in content.lower() or 'minor' in content.lower() or 'league' in content.lower():
                print(f"\n  {rel} contains league-related content:")
                for line in content.split('\n')[:30]:
                    if 'milb' in line.lower() or 'minor' in line.lower() or 'league' in line.lower() or 'qual' in line.lower():
                        print(f"    {line}")
