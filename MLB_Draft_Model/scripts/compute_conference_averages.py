#!/usr/bin/env python3
"""
compute_conference_averages.py — Pre-compute per-conference, per-year stat averages
from the training set and save as reference data for the adjusted-feature pipeline.

Usage:
    python3 scripts/compute_conference_averages.py
"""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"

# Rate stats most influenced by competition quality
HITTER_ADJ_STATS = ["wOBA", "OPS", "AVG", "SLG", "BB_pct", "K_pct", "ISO", "wRC_plus"]
PITCHER_ADJ_STATS = ["ERA", "FIP", "WHIP", "K_per_nine", "BB_per_nine", "K_pct", "BB_pct"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def compute_mean(vals):
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def main():
    print("=" * 60)
    print("CONFERENCE STAT AVERAGES")
    print("=" * 60)

    train = load_json(BASE / "data" / "training" / "expanded_training_set.json")
    print(f"\nTraining records: {len(train)}")

    # ── 1. Collect values per (conf, season, ptype, stat) ──
    cell_values = defaultdict(list)  # key=(conf, season, ptype, stat) → [val1, val2, ...]
    conf_all_values = defaultdict(list)  # key=(conf, ptype, stat) → [val1, ...]
    tier_all_values = defaultdict(list)  # key=(tier, ptype, stat) → [val1, ...]

    for rec in train:
        conf = rec.get("conference", "")
        season = rec.get("season")
        ptype = rec.get("player_type", "hitter")
        tier = rec.get("conference_tier")

        if not conf or season is None:
            continue

        stats = HITTER_ADJ_STATS if ptype == "hitter" else PITCHER_ADJ_STATS
        for stat in stats:
            val = safe_float(rec.get(stat))
            if val is not None:
                cell_values[(conf, season, ptype, stat)].append(val)
                conf_all_values[(conf, ptype, stat)].append(val)
                if tier is not None:
                    tier_all_values[(tier, ptype, stat)].append(val)

    # ── 2. Compute means ──
    # Per (conf, season, ptype)
    conf_season_ptype = defaultdict(dict)
    for (conf, season, ptype, stat), vals in cell_values.items():
        key = (conf, season, ptype)
        conf_season_ptype[key][stat] = round(compute_mean(vals), 4)

    # Per (conf, ptype) — fallback when season cell is small
    conf_overall = defaultdict(dict)
    for (conf, ptype, stat), vals in conf_all_values.items():
        conf_overall[(conf, ptype)][stat] = round(compute_mean(vals), 4)

    # Per (tier, ptype) — ultimate fallback
    tier_fallback = defaultdict(dict)
    for (tier, ptype, stat), vals in tier_all_values.items():
        tier_fallback[(tier, ptype)][stat] = round(compute_mean(vals), 4)

    # ── 3. Restructure into nested dict for easy lookup ──
    # per_season[conf][season][ptype][stat] = mean
    per_season = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for (conf, season, ptype), stats in conf_season_ptype.items():
        per_season[conf][season][ptype] = stats

    # conference_overall[conf][ptype][stat] = mean
    conf_ov = defaultdict(lambda: defaultdict(dict))
    for (conf, ptype), stats in conf_overall.items():
        conf_ov[conf][ptype] = stats

    # tier_fallback[tier][ptype][stat] = mean
    tier_fb = defaultdict(lambda: defaultdict(dict))
    for (tier, ptype), stats in tier_fallback.items():
        tier_fb[str(tier)][ptype] = stats

    # ── 4. Report ──
    print(f"\nUnique conferences: {len(per_season)}")
    seasons = sorted(set(s for conf in per_season.values() for s in conf.keys()))
    print(f"Seasons in data: {seasons}")
    print(f"Hitter stats: {HITTER_ADJ_STATS}")
    print(f"Pitcher stats: {PITCHER_ADJ_STATS}")

    # Count small cells
    small = 0
    total = 0
    for conf, seasons_dict in per_season.items():
        for season, ptypes in seasons_dict.items():
            for ptype in ptypes:
                total += 1
                n = len(cell_values.get((conf, season, ptype, ptype), []))
                if n < 5:
                    small += 1
    # Better: count unique player records per (conf, season, ptype)
    cell_counts = defaultdict(set)
    for rec in train:
        conf = rec.get("conference", "")
        season = rec.get("season")
        ptype = rec.get("player_type", "hitter")
        if conf and season is not None:
            cell_counts[(conf, season, ptype)].add(rec.get("player_name", ""))
    small = sum(1 for key, names in cell_counts.items() if len(names) < 5)
    total = len(cell_counts)
    print(f"\nConference-season-type cells: {total}")
    print(f"  Small (< 5 players): {small} ({100*small/max(total,1):.0f}%)")

    # ── 5. Save ──
    output = {
        "per_season": per_season,
        "conference_overall": conf_ov,
        "tier_fallback": tier_fb,
        "stats": {"hitter": HITTER_ADJ_STATS, "pitcher": PITCHER_ADJ_STATS},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print(f"\nSaved: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")

    # Sample
    print("\nSample — SEC hitters 2024:")
    sec = per_season.get("SEC", {}).get(2024, {}).get("hitter", {})
    for stat, val in sorted(sec.items())[:6]:
        print(f"  {stat}: {val}")

    print("\nDone.")


if __name__ == "__main__":
    main()
