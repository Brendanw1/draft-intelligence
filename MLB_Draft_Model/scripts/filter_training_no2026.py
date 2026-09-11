#!/usr/bin/env python3
"""
filter_training_no2026.py — Create filtered training data excluding 2026 outcomes.

Backs up originals, then writes filtered versions of:
- expanded_training_set.json (filtered by draft_year != 2026)
- tier2_negatives.json (filtered by season != 2026)
- fg_training_set.json (filtered by draft_year != 2026)

Prints counts before/after.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "training"

def backup_file(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        import shutil
        shutil.copy2(path, bak)
        print(f"  Backed up: {path.name} → {bak.name}")
    else:
        print(f"  Backup exists: {bak.name}")
    return bak

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved: {path.name} ({len(data):,} records)")

def filter_and_save(json_path, filter_field, filter_value, label):
    """Filter records where filter_field == filter_value, backup + save filtered."""
    print(f"\n{'='*60}")
    print(f"{label}: {json_path.name}")
    print(f"{'='*60}")

    data = load_json(json_path)
    removed = [r for r in data if r.get(filter_field) == filter_value]
    filtered = [r for r in data if r.get(filter_field) != filter_value]

    print(f"  Total records: {len(data):,}")
    print(f"  Records with {filter_field}={filter_value}: {len(removed):,}")
    print(f"  Remaining after filter: {len(filtered):,}")

    # Backup
    backup_file(json_path)

    # Save filtered
    save_json(filtered, json_path)

    return len(data), len(removed), len(filtered)

def main():
    print("=" * 60)
    print("FILTERING TRAINING DATA — REMOVING 2026 OUTCOMES")
    print("=" * 60)

    # 1. expanded_training_set.json — filter by draft_year != 2026
    filter_and_save(
        DATA_DIR / "expanded_training_set.json",
        "draft_year", 2026,
        "TRAINING SET (positives, drafted players)"
    )

    # 2. tier2_negatives.json — filter by season != 2026
    filter_and_save(
        DATA_DIR / "tier2_negatives.json",
        "season", 2026,
        "TIER 2 NEGATIVES (undrafted players)"
    )

    # 3. fg_training_set.json — filter by draft_year != 2026
    filter_and_save(
        DATA_DIR / "fg_training_set.json",
        "draft_year", 2026,
        "FG TRAINING SET (Tier 1 features)"
    )

    print(f"\n{'='*60}")
    print("FILTERING COMPLETE — Training data now excludes 2026 outcomes")
    print("Backup files (*.bak) created for all modified files.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
