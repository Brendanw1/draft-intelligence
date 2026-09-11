#!/usr/bin/env python3
"""Explore pybaseball package structure."""
import pybaseball, os, inspect

pkg_path = os.path.dirname(pybaseball.__file__)

# List ALL files including __init__
print("=== All package files ===")
for f in sorted(os.listdir(pkg_path)):
    print(f"  {f}")

# Read __init__.py
init_path = os.path.join(pkg_path, '__init__.py')
with open(init_path) as f:
    init_src = f.read()
print("\n=== __init__.py (first 2000 chars) ===")
print(init_src[:2000])

# Check if there's a _batting.py or similar
print("\n=== Looking for batting/pitching source files ===")
for root, dirs, files in os.walk(pkg_path):
    for f in files:
        if f.endswith('.py') and ('bat' in f or 'pitch' in f):
            print(f"  {os.path.join(root, f)}")
