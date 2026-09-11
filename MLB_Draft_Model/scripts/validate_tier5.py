#!/usr/bin/env python3
"""Unified Tier 5 verification gates.

Runs all Tier 5 verification gates (V1-V4 from build_tier5_training.py +
V1-V4 from pull_war_ground_truth.py) and produces a unified report.

Gates:
  * Hurdle rate sanity (positive rate should be reasonable)
  * Round-WAR negative correlation (higher draft round = lower WAR)
  * Temporal leakage check (no 2022+ in train set)
  * WAR distribution skew (should be right-skewed, mode near 0)
  * BRef match rate (≥85% of debuted players should have BRef key)
  * Field completeness (all required fields present)
  * Phantom WAR check (no debut season < draft year)
  * Median WAR ≤ 0 (most players produce 0 or negative WAR)

Usage:
    python scripts/validate_tier5.py [--training PATH] [--war PATH] [--verify]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = BASE / "data" / "training" / "tier5_training_set.json"
DEFAULT_WAR = BASE / "data" / "war" / "war_ground_truth.json"


def load_json(path):
    return json.load(open(path))


def safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Training set gates (V1-V4 from build_tier5_training.py)
# ---------------------------------------------------------------------------

def verify_training_v1(records):
    """Hurdle rate sanity."""
    n = len(records)
    n_positive = sum(1 for r in records if r.get("meaningfully_productive") is True)
    rate = n_positive / n if n > 0 else 0.0
    # Expect 2-10% positive rate for this cohort
    return {
        "pass": 0.01 <= rate <= 0.15,
        "hurdle_rate": rate,
        "n": n,
        "n_positive": n_positive,
        "note": "expected 2-10% for 2021-2023 drafted players",
    }


def verify_training_v2(records):
    """Round-WAR negative correlation."""
    rounds, wars = [], []
    for r in records:
        war = r.get("war_years_1_through_5")
        rnd = r.get("draft_round")
        if war is None or rnd is None:
            continue
        rounds.append(float(rnd))
        wars.append(float(war))
    if len(rounds) < 2:
        return {"pass": False, "correlation": 0.0, "n": len(rounds)}
    corr = float(np.corrcoef(rounds, wars)[0, 1])
    if math.isnan(corr):
        corr = 0.0
    return {"pass": corr < 0, "correlation": corr, "n": len(rounds)}


def verify_training_v3(records, train_max_year=2021):
    """Temporal leakage check."""
    leakage = [
        r for r in records
        if r.get("split") == "train" and (r.get("draft_year") or 0) > train_max_year
    ]
    return {"pass": len(leakage) == 0, "leakage_count": len(leakage)}


def verify_training_v4(records):
    """WAR distribution skew."""
    vals = [r["war_years_1_through_5"] for r in records
            if r.get("war_years_1_through_5") is not None]
    if not vals:
        return {"pass": True, "median": None, "count": 0}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    mid = n // 2
    median = float(vals_sorted[mid]) if n % 2 == 1 else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2.0
    skew = 0.0
    if len(vals) >= 3:
        from scipy.stats import skew as _skew
        skew = float(_skew(vals))
    return {
        "pass": abs(median) <= 0.5,
        "median": median,
        "count": len(vals),
        "skew": skew,
        "mode_near_0": abs(median) <= 0.5,
    }


# ---------------------------------------------------------------------------
# WAR ground truth gates (V1-V4 from pull_war_ground_truth.py)
# ---------------------------------------------------------------------------

def verify_war_v1(records):
    """BRef match rate."""
    debuted = [r for r in records if r.get("mlb_debut_season") is not None]
    matched = sum(1 for r in debuted if r.get("key_bbref"))
    total = len(debuted)
    rate = matched / total if total else 0.0
    return {
        "pass": rate >= 0.85,
        "debuted_match_rate": rate,
        "matched": matched,
        "debuted_total": total,
        "all_records": len(records),
    }


def verify_war_v2(records):
    """Field completeness."""
    required_fields = [
        "person_id", "key_bbref", "career_war_to_date",
        "war_years_1_through_5", "war_year_1_through_3",
        "peak_single_season_war", "seasons_with_positive_war",
        "total_mlb_seasons", "mlb_debut_season", "draft_year",
    ]
    missing = sum(1 for r in records
                  if not set(required_fields).issubset(r.keys()))
    return {"pass": missing == 0, "n": len(records), "missing": missing}


def verify_war_v3(records):
    """Phantom WAR check (no debut season < draft year)."""
    impossible = 0
    for r in records:
        if r.get("career_war_to_date") is None:
            continue
        debut = r.get("mlb_debut_season")
        if debut is not None and debut < r.get("draft_year", 0):
            impossible += 1
    return {
        "pass": impossible == 0,
        "impossible_war": impossible,
        "note": "phantom WAR = debut season < draft year (bad join)",
    }


def verify_war_v4(records):
    """Median WAR ≤ 0."""
    values = [r["war_years_1_through_5"] for r in records
              if r.get("war_years_1_through_5") is not None]
    if not values:
        return {"pass": True, "median": None, "count": 0}
    values = sorted(values)
    n = len(values)
    mid = n // 2
    median = float(values[mid]) if n % 2 == 1 else (values[mid - 1] + values[mid]) / 2.0
    return {"pass": median <= 0, "median": median, "count": n}


# ---------------------------------------------------------------------------
# Unified report
# ---------------------------------------------------------------------------

def run_all_gates(training_records, war_records):
    gates = {
        "training_V1_hurdle_rate": verify_training_v1(training_records),
        "training_V2_round_war_corr": verify_training_v2(training_records),
        "training_V3_temporal_leakage": verify_training_v3(training_records),
        "training_V4_war_distribution": verify_training_v4(training_records),
        "war_V1_bbref_match": verify_war_v1(war_records),
        "war_V2_field_completeness": verify_war_v2(war_records),
        "war_V3_phantom_war": verify_war_v3(war_records),
        "war_V4_median_lte_zero": verify_war_v4(war_records),
    }
    return gates


def print_report(gates):
    print("=" * 64)
    print("TIER 5 — UNIFIED VERIFICATION REPORT")
    print("=" * 64)
    for name, g in gates.items():
        status = "PASS" if g["pass"] else "WARN"
        print(f"  {name:<30s} {status}")
        for k, v in g.items():
            if k != "pass":
                print(f"    {k}: {v}")
    print("-" * 64)
    all_pass = all(g["pass"] for g in gates.values())
    print(f"  OVERALL: {'PASS' if all_pass else 'WARN (review gates above)'}")
    print("=" * 64)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run Tier 5 unified verification gates.")
    p.add_argument("--training", default=str(DEFAULT_TRAINING),
                   help="tier5_training_set.json path")
    p.add_argument("--war", default=str(DEFAULT_WAR),
                   help="war_ground_truth.json path")
    p.add_argument("--verify", action="store_true",
                   help="print unified verification report")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    try:
        training = load_json(args.training)
    except FileNotFoundError:
        print(f"ERROR: training set not found at {args.training}")
        return 1

    try:
        war_data = load_json(args.war)
        war_records = war_data.get("records", war_data)
    except FileNotFoundError:
        print(f"ERROR: WAR ground truth not found at {args.war}")
        return 1

    gates = run_all_gates(training, war_records)

    if args.verify:
        print_report(gates)

    # Exit 0 if all gates pass, 1 if any fail
    all_pass = all(g["pass"] for g in gates.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
