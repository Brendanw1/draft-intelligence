#!/usr/bin/env python3
"""
compute_conference_strength.py — conference strength = conference rate of the Tier 2 label / pooled rate.

Target-encoded from the Tier 2 fixed-horizon label (drafted at pick <= 315 within three drafts), on the labelled
seasons only (2021 through LAST_LABELLED), via the same function the trainers use on their own training rows
(`leakfree.conf_strength_table`). The 2026 board and any season after LAST_LABELLED never feed it. Run after
build_tier2_negatives.py and before train_fg_model.py / train_tier3_mlb_arrival.py, which read the file.

Usage:
    PYTHONPATH=src python3 scripts/compute_conference_strength.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_tier2_full as t2  # noqa: E402
from mlb_draft_dashboard.leakfree import conf_strength_table  # noqa: E402


def main():
    rows, _ = t2.load_all()
    y, _, seasons, confs = t2.arrays(rows)
    table = conf_strength_table(confs, y)
    t2.CONF_STRENGTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(t2.CONF_STRENGTH_PATH, "w") as f:
        json.dump(table, f, indent=2)
    top = sorted(table.items(), key=lambda kv: -kv[1]["strength"])
    print(f"seasons {seasons.min()}-{seasons.max()}, {len(y):,} rows, pooled rate {y.mean():.4f}, "
          f"{len(table)} conferences -> {t2.CONF_STRENGTH_PATH}")
    for c, d in top[:5] + top[-3:]:
        print(f"  {c:<20s} {d['strength']:.3f} ({d['drafted']}/{d['total']})")


if __name__ == "__main__":
    main()
