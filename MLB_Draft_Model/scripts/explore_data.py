#!/usr/bin/env python3
"""Explore data coverage — field completion, year coverage, source gaps."""
import json, os, sys
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[1] / "data"

def load(path):
    try:
        with open(BASE / path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  FAILED: {path} — {e}")
        return None

def analyze_field_coverage(records, label, max_fields=50):
    if not records:
        return
    n = len(records)
    print(f"\n{'='*60}")
    print(f"FIELD COVERAGE: {label} ({n:,} records)")
    print(f"{'='*60}")
    keys = list(records[0].keys())
    if len(keys) > max_fields:
        print(f"  ({len(keys)} total fields — showing first {max_fields})")
        keys = keys[:max_fields]
    for key in keys:
        filled = sum(1 for r in records if r.get(key) is not None
                     and str(r.get(key, "")).strip() not in ("", "None", "0", "0.0"))
        pct = 100 * filled / n
        bar = "#" * int(pct / 5) + " " * (20 - int(pct / 5))
        print(f"  {key:<30s} {bar} {pct:5.1f}% ({filled:>7,}/{n:,})")

def analyze_year_coverage(records, label):
    if not records:
        return
    n = len(records)
    print(f"\nYEAR COVERAGE: {label} ({n:,} records)")
    years = Counter()
    for r in records:
        for y_key in ["draft_year", "fg_season", "season", "year"]:
            if y_key in r:
                years[r[y_key]] += 1
    if years:
        for yr in sorted(years.keys()):
            print(f"  {yr}: {years[yr]:>6,} records")
    else:
        print("  No year field found")

# ── PROJECTIONS ──
p = load("training/projections_2026_enriched.json")
analyze_field_coverage(p, "2026 ENRICHED PROJECTIONS", max_fields=50)

# ── FG TRAINING SET ──
fg = load("training/fg_training_set.json")
if fg:
    analyze_year_coverage(fg, "FG TRAINING SET")
    hitters = [r for r in fg if r.get("player_type") == "hitter"]
    pitchers = [r for r in fg if r.get("player_type") == "pitcher"]
    print(f"\n  Hitters: {len(hitters):,}  Pitchers: {len(pitchers):,}")

# ── TIER 2 TRAINING SET ──
t2 = load("training/tier2_training_set.json")
if t2:
    analyze_field_coverage(t2, "TIER 2 TRAINING SET", max_fields=30)
    analyze_year_coverage(t2, "TIER 2 TRAINING SET")
    reached = sum(1 for r in t2 if str(r.get("reached_mlb", "")).lower() in ("true", "1"))
    print(f"\n  Reached MLB: {reached:,}/{len(t2):,} ({100*reached/len(t2):.1f}%)")
    # Check peak_level distribution
    peak_dist = Counter(r.get("peak_level", "unknown") for r in t2)
    print("\n  Peak level reached:")
    for level, count in peak_dist.most_common():
        print(f"    {level:<15s} {count:>5,}")

# ── FANGRAPHS RAW DATA ──
import os
fg_dir = BASE / "fangraphs"
if fg_dir.exists():
    print(f"\n{'='*60}")
    print(f"FANGRAPHS RAW DATA")
    print(f"{'='*60}")
    for f in sorted(fg_dir.glob("*.json")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:<40s} {size_mb:>6.1f} MB")

# ── ROSTERS ──
rosters = load("rosters/d1_rosters_2026.json")
if rosters:
    analyze_year_coverage(rosters, "D1 ROSTERS")
    # Conference distribution
    conf_dist = Counter(r.get("conference", "unknown") for r in rosters)
    print("\n  Top conferences by player count:")
    for conf, count in conf_dist.most_common(15):
        print(f"    {conf:<25s} {count:>5,}")

# ── DRAFT DATA ──
dd = load("draft/draft_all_picks.json")
if dd:
    analyze_year_coverage(dd, "DRAFT DATA (ALL PICKS)")
    # Signed rate
    signed = sum(1 for r in dd if r.get("signing_bonus") is not None)
    print(f"\n  Signed: {signed:,}/{len(dd):,}")

# ── MLB DATA ──
mlb_dir = BASE / "milb"
if mlb_dir.exists():
    print(f"\n{'='*60}")
    print(f"MiLB DATA")
    print(f"{'='*60}")
    for f in sorted(mlb_dir.glob("*.json")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:<40s} {size_mb:>6.1f} MB")

print("\n\nDone.")
