#!/usr/bin/env python3
"""Check pybaseball source code for MiLB-related functions."""
import inspect
import pybaseball
import pybaseball.batting_stats as bs_module
import pybaseball.pitching_stats as ps_module

# Check what functions exist in the batting_stats module
print("=== batting_stats module contents ===")
print([x for x in dir(bs_module) if not x.startswith('_')])

# Check signature of the main function
print("\n=== batting_stats signature ===")
print(inspect.signature(bs_module.batting_stats))
src = inspect.getsource(bs_module.batting_stats)
# Print first 30 lines
for line in src.split('\n')[:30]:
    print(line)

# Check if there are MiLB-specific functions anywhere
print("\n=== Searching for 'milb' in pybaseball package ===")
import os
pkg_path = os.path.dirname(pybaseball.__file__)
for root, dirs, files in os.walk(pkg_path):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
                if 'milb' in content.lower() or 'minor' in content.lower():
                    if 'milb' in content.lower():
                        print(f"  Contains 'milb': {path}")
                    break

print("\n=== pybaseball directory listing ===")
for item in sorted(os.listdir(pkg_path)):
    print(f"  {item}")
