#!/usr/bin/env python3
"""
calibrate_probs.py — Calibrate raw model probabilities using the quantile
calibration curve. Loads the curve, applies it to enriched predictions,
and writes calibrated probabilities.

Can run standalone (post-inference) or be called from infer_2026.py.

Usage:
    python3 scripts/calibrate_probs.py
"""
import json, sys
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
ENRICHED_PATH = BASE / "data" / "training" / "projections_2026_enriched.json"
CAL_DIR = BASE / "models" / "artifacts_full"


def load_json(path):
    return json.load(open(path))


def calibrate(raw_score, raw_centers, calibrated):
    """Map a raw score to calibrated probability via linear interpolation.

    raw_centers and calibrated are PAVA-smoothed bin centers/rates.
    Handles out-of-range scores by using the nearest endpoint.
    """
    raw_centers = np.array(raw_centers)
    calibrated = np.array(calibrated)

    # Clip to range
    if raw_score <= raw_centers[0]:
        return float(calibrated[0])
    if raw_score >= raw_centers[-1]:
        return float(calibrated[-1])

    # Binary search for bracketing bins
    idx = np.searchsorted(raw_centers, raw_score, side="right") - 1
    idx = max(0, min(idx, len(raw_centers) - 2))

    # Linear interpolation
    x0, x1 = raw_centers[idx], raw_centers[idx + 1]
    y0, y1 = calibrated[idx], calibrated[idx + 1]

    if x1 == x0:
        return float(y0)
    return float(y0 + (raw_score - x0) * (y1 - y0) / (x1 - x0))


def main():
    print("=" * 60)
    print("CALIBRATE PROBABILITIES")
    print("=" * 60)

    # Load enriched predictions
    enriched = load_json(ENRICHED_PATH)
    print(f"\nEnriched records: {len(enriched)}")

    # Load calibration curves
    h_curve = load_json(CAL_DIR / "calibration_lookup_hitter.json")
    p_curve = load_json(CAL_DIR / "calibration_lookup_pitcher.json")

    h_raw = h_curve["raw_centers"]
    h_cal = h_curve["calibrated"]
    p_raw = p_curve["raw_centers"]
    p_cal = p_curve["calibrated"]

    print(f"  Hitter curve: {len(h_raw)} bins, range [{h_raw[0]:.4f}, {h_raw[-1]:.4f}] → [{h_cal[0]:.4f}, {h_cal[-1]:.4f}]")
    print(f"  Pitcher curve: {len(p_raw)} bins, range [{p_raw[0]:.4f}, {p_raw[-1]:.4f}] → [{p_cal[0]:.4f}, {p_cal[-1]:.4f}]")

    # Apply calibration
    n_calibrated = 0
    for rec in enriched:
        ptype = rec.get("player_type", "hitter")
        raw = rec.get("mlb_probability")
        if raw is None:
            continue

        raw_centers = h_raw if ptype == "hitter" else p_raw
        cal_values = h_cal if ptype == "hitter" else p_cal
        cal_prob = calibrate(raw, raw_centers, cal_values)
        rec["mlb_prob_calibrated"] = round(cal_prob, 4)
        n_calibrated += 1

    print(f"\nCalibrated {n_calibrated} records")

    # Summary stats
    all_cal = [rec["mlb_prob_calibrated"] for rec in enriched if "mlb_prob_calibrated" in rec]
    all_raw = [rec["mlb_probability"] for rec in enriched if "mlb_probability" in rec]

    print(f"\n  Raw vs Calibrated comparison:")
    print(f"  {'Percentile':>12s} {'Raw':>8s} {'Calibrated':>12s}")
    print(f"  {'─'*32}")
    for pctl in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        rv = np.percentile(all_raw, pctl)
        cv = np.percentile(all_cal, pctl)
        print(f"  p{pctl:>3d}:     {rv:.4f}      {cv:.4f}")

    # Check specific players
    print(f"\n  Specific players:")
    targets = {}
    for rec in enriched:
        name = rec.get("player_name", "")
        for target in ["Drew Burress", "Vahn Lackey", "Roch Cholowsky",
                        "Evan Dempsey", "Jackson Flora", "Ben Blair"]:
            if target.lower() in name.lower():
                targets[target] = rec

    for name, rec in sorted(targets.items()):
        raw = rec.get("mlb_probability", 0)
        cal = rec.get("mlb_prob_calibrated", 0)
        platt = rec.get("mlb_prob_platt", 0)
        pick = rec.get("projected_pick", "?")
        if not isinstance(pick, str): pick = f"{pick:.0f}"
        print(f"    {name:<20s} raw={raw:.4f} cal={cal:.4f} platt={platt:.4f} pick={pick}")

    # Save
    OUTPUT_PATH = ENRICHED_PATH
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(enriched, f, separators=(",", ":"))
    print("Done.")


if __name__ == "__main__":
    main()
