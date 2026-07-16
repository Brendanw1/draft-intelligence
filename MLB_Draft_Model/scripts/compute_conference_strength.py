#!/usr/bin/env python3
"""
compute_conference_strength.py — Compute empirical conference strength index
from historical draft rates using logit transformation.

Strength = logit(draft_rate) = log(rate / (1 - rate))
Normalized so SEC = 1.0 for interpretability.

This replaces the flawed 4-tier categorical system with a continuous,
data-driven measure of competition quality.

Usage:
    python3 scripts/compute_conference_strength.py
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("CONFERENCE STRENGTH INDEX")
    print("=" * 60)

    # Load training data (positives + negatives)
    positives = load_json(BASE / "data" / "training" / "expanded_training_set.json")
    negatives = load_json(BASE / "data" / "training" / "tier2_negatives.json")
    all_records = positives + negatives

    print(f"Positives (drafted): {len(positives):,}")
    print(f"Negatives (undrafted): {len(negatives):,}")
    print(f"Total: {len(all_records):,}")

    # Count drafted and total per conference
    conf_data = defaultdict(lambda: {"drafted": 0, "total": 0, "picks": [], "bonuses": []})

    for rec in positives:
        conf = rec.get("conference", "")
        if conf:
            conf_data[conf]["drafted"] += 1
            conf_data[conf]["total"] += 1
            pick = rec.get("draft_pick")
            if pick and pick > 0:
                conf_data[conf]["picks"].append(float(pick))
            bonus = rec.get("draft_bonus", 0)
            if bonus:
                conf_data[conf]["bonuses"].append(float(bonus))

    for rec in negatives:
        conf = rec.get("conference", "")
        if conf:
            conf_data[conf]["total"] += 1

    # Compute strength metrics
    global_rate = sum(d["drafted"] for d in conf_data.values()) / max(
        sum(d["total"] for d in conf_data.values()), 1
    )
    print(f"\nGlobal draft rate: {global_rate:.4f}")

    # ── Compute conference strength as draft rate relative to global average ──
    # strength = draft_rate / global_draft_rate
    # > 1.0 = stronger than average, < 1.0 = weaker than average
    # SEC ≈ 3.0 (players 3× more likely to be drafted than D1 average)
    # SWAC ≈ 0.09 (players 11× less likely)

    strengths = {}
    for conf, d in conf_data.items():
        if d["total"] < 10:
            continue
        rate = d["drafted"] / max(d["total"], 1)
        strength = rate / global_rate if global_rate > 0 else 1.0
        avg_pick = np.mean(d["picks"]) if d["picks"] else 500
        med_pick = np.median(d["picks"]) if d["picks"] else 500
        avg_bonus = np.mean(d["bonuses"]) if d["bonuses"] else 0

        strengths[conf] = {
            "strength": round(float(strength), 4),
            "draft_rate": round(float(rate), 4),
            "global_rate": round(float(global_rate), 4),
            "total": d["total"],
            "drafted": d["drafted"],
            "avg_pick": round(float(avg_pick), 1),
            "med_pick": round(float(med_pick), 1),
            "avg_bonus": round(float(avg_bonus), 0),
        }

    # Print sorted by strength
    print(f"\n{'Conference':<20s} {'Drafted':>8s} {'Total':>7s} {'Rate':>7s} {'Strength':>9s} {'AvgPick':>7s}")
    print("-" * 60)
    for conf in sorted(strengths, key=lambda c: strengths[c]["strength"], reverse=True):
        d = strengths[conf]
        print(f"{conf:<20s} {d['drafted']:>8d} {d['total']:>7d} {d['draft_rate']:>6.1%} "
              f"{d['strength']:>+8.3f} {d['avg_pick']:>6.0f}")

    # Show Tier comparison
    print(f"\n{'Conference':<20s} {'Strength':>9s} {'Old Tier':>9s} {'Interpretation':>25s}")
    print("-" * 63)
    reference_confs = ["DI Independent", "SEC", "ACC", "Big 12", "Big Ten",
                       "American", "Big West", "CUSA", "Sun Belt", "WCC",
                       "ASUN", "Southland", "SWAC"]
    for conf in reference_confs:
        if conf in strengths:
            s = strengths[conf]
            old_tier = "?"
            for r in all_records:
                if r.get("conference") == conf:
                    old_tier = str(r.get("conference_tier", "?"))
                    break
            if s["strength"] > 2.5:
                interp = "Elite (3×+ avg)"
            elif s["strength"] > 1.5:
                interp = "Power 5 (1.5-2.5× avg)"
            elif s["strength"] > 0.8:
                interp = "Solid mid-major"
            elif s["strength"] > 0.5:
                interp = "Avg mid-major"
            else:
                interp = "Low major (<0.5× avg)"
            print(f"{conf:<20s} {s['strength']:>+8.3f} {old_tier:>8s} {interp:>25s}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(strengths, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  {len(strengths)} conferences")
    print(f"  Range: {min(s['strength'] for s in strengths.values()):.3f} to {max(s['strength'] for s in strengths.values()):.3f}")
    print(f"  Global draft rate: {global_rate:.4f}")


if __name__ == "__main__":
    main()
