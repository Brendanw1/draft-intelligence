#!/usr/bin/env python3
"""Deep dive into pybaseball's data layer."""
import pybaseball
import inspect, os

# Find the module
pkg_path = os.path.dirname(pybaseball.__file__)
print("=== Package files ===")
for f in sorted(os.listdir(pkg_path)):
    if f.endswith('.py') and not f.startswith('_'):
        print(f"  {f}")

# Check the datalayer module
from pybaseball import datalayer
print("\n=== datalayer contents ===")
print([x for x in dir(datalayer) if not x.startswith('_')])

# Look at batting_stats more carefully
print("\n=== batting_stats type and attributes ===")
bs = pybaseball.batting_stats
print(f"type: {type(bs)}")
print(f"dir: {[x for x in dir(bs) if not x.startswith('_')]}")

# Try to get the actual fetch signature
try:
    src = inspect.getsource(bs.__class__)
    print(f"\nClass source (first 80 lines):")
    for line in src.split('\n')[:80]:
        print(f"  {line}")
except Exception as e:
    print(f"Error getting source: {e}")

# Check what datasources are registered
print("\n=== datasources ===")
from pybaseball import datasources
print(f"type: {type(datasources)}")
print([x for x in dir(datasources) if not x.startswith('_')])

# Check enums for possible parameters
from pybaseball import enums
print("\n=== enums ===")
print([x for x in dir(enums) if not x.startswith('_')])
