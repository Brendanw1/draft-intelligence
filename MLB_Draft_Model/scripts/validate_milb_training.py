#!/usr/bin/env python3
"""
validate_milb_training.py — Validate MiLB extended training set.

Validates milb_extended_training.json by checking:
- Total records, distributions, correlations
- Print 5 random players with all computed fields
- Print pass/fail for existing validation checks

Usage:
    python3 scripts/validate_milb_training.py
"""
import json
import random
import sys
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "data" / "training" / "milb_extended_training.json"


def safe_float(v, default=0.0):
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def main():
    print("=" * 60)
    print("VALIDATION: MiLB Extended Training Set")
    print("=" * 60)

    # ── 1. Load ──
    print("\n1. Loading milb_extended_training.json...")
    data = json.load(open(TRAIN_PATH))
    print(f"   Total records: {len(data)}")

    # ── 2. Distributions ──
    print("\n2. Distributions:")
    print(f"\n   [Records by draft year]:")
    from collections import Counter
    dy_counts = Counter(r.get("draft_year") for r in data)
    for year in sorted(dy_counts.keys()):
        cnt = dy_counts[year]
        print(f"     {year}: {cnt} ({100 * cnt / len(data):.1f}%)")

    print(f"\n   [Records by player_type]:")
    pt_counts = Counter(r.get("player_type") for r in data)
    for pt, cnt in sorted(pt_counts.items()):
        print(f"     {pt}: {cnt} ({100 * cnt / len(data):.1f}%)")

    print(f"\n   [milb_year1_level distribution]:")
    level_labels = {1: "A", 2: "A+", 3: "AA", 4: "AAA"}
    lv_counts = Counter(r.get("milb_year1_level") for r in data)
    for lvl in sorted(lv_counts.keys()):
        cnt = lv_counts[lvl]
        label = level_labels.get(lvl, f"Level {lvl}")
        print(f"     {label} ({lvl}): {cnt} ({100 * cnt / len(data):.1f}%)")

    print(f"\n   [milb_highest_level distribution]:")
    hl_counts = Counter(r.get("milb_highest_level") for r in data)
    for lvl in sorted(hl_counts.keys()):
        cnt = hl_counts[lvl]
        label = level_labels.get(lvl, f"Level {lvl}")
        print(f"     {label} ({lvl}): {cnt} ({100 * cnt / len(data):.1f}%)")

    # Peak wOBA/FIP
    hitters = [r for r in data if r.get("player_type") == "hitter"]
    pitchers = [r for r in data if r.get("player_type") == "pitcher"]

    print(f"\n   [milb_peak_wOBA distribution — hitters]:")
    peak_wobas = [safe_float(r.get("milb_peak_wOBA"), 0) for r in hitters]
    if peak_wobas:
        print(f"     Mean={np.mean(peak_wobas):.4f} Median={np.median(peak_wobas):.4f}")
        print(f"     Min={min(peak_wobas):.4f} Max={max(peak_wobas):.4f}")
        print(f"     P10={np.percentile(peak_wobas, 10):.4f} P90={np.percentile(peak_wobas, 90):.4f}")

    print(f"\n   [milb_peak_FIP distribution — pitchers]:")
    peak_fips = [safe_float(r.get("milb_peak_FIP"), 0) for r in pitchers]
    if peak_fips:
        print(f"     Mean={np.mean(peak_fips):.4f} Median={np.median(peak_fips):.4f}")
        print(f"     Min={min(peak_fips):.4f} Max={max(peak_fips):.4f}")
        print(f"     P10={np.percentile(peak_fips, 10):.4f} P90={np.percentile(peak_fips, 90):.4f}")

    # mlb_debut rate
    debuted = sum(1 for r in data if r.get("has_mlb_debut"))
    print(f"\n   [MLB debut rate]: {debuted}/{len(data)} ({100 * debuted / len(data):.1f}%)")
    for pt in ["hitter", "pitcher"]:
        sub = [r for r in data if r.get("player_type") == pt]
        d_sub = sum(1 for r in sub if r.get("has_mlb_debut"))
        print(f"     {pt}: {d_sub}/{len(sub)} ({100 * d_sub / max(len(sub), 1):.1f}%)")

    # ── 3. Correlations ──
    print("\n3. Correlations:")

    # 3a. college_wOBA vs milb_peak_wOBA (expect > 0.10)
    if len(hitters) >= 10:
        college_woba = [safe_float(r.get("wOBA"), 0) for r in hitters]
        peak_woba = [safe_float(r.get("milb_peak_wOBA"), 0) for r in hitters]
        corr_1 = np.corrcoef(college_woba, peak_woba)[0, 1]
        v5_pass = corr_1 > 0.10
        print(f"   [V5] Correlation(college wOBA, milb_peak_wOBA): {corr_1:.4f} "
              f"({'PASS' if v5_pass else 'FAIL'}, threshold: > 0.10)")

    # 3b. draft_round vs milb_peak_wOBA (expect < -0.05)
    if len(hitters) >= 10:
        draft_rounds = [safe_float(r.get("draft_round"), 0) for r in hitters]
        peak_woba2 = [safe_float(r.get("milb_peak_wOBA"), 0) for r in hitters]
        corr_2 = np.corrcoef(draft_rounds, peak_woba2)[0, 1]
        v6_pass = corr_2 < -0.05
        print(f"   [V6] Correlation(draft_round, milb_peak_wOBA): {corr_2:.4f} "
              f"({'PASS' if v6_pass else 'FAIL'}, threshold: < -0.05)")

    # 3c. conf_strength vs milb_peak_wOBA (exploratory)
    if len(hitters) >= 10:
        conf_str = [safe_float(r.get("conf_strength"), 1.0) for r in hitters]
        peak_woba3 = [safe_float(r.get("milb_peak_wOBA"), 0) for r in hitters]
        corr_3 = np.corrcoef(conf_str, peak_woba3)[0, 1]
        print(f"   [Exploratory] Correlation(conf_strength, milb_peak_wOBA): {corr_3:.4f} (no threshold)")

    # Also check college wOBA vs milb_year1_wOBA
    if len(hitters) >= 10:
        yr1_woba = [safe_float(r.get("milb_year1_wOBA"), 0) for r in hitters]
        corr_y1 = np.corrcoef(college_woba, yr1_woba)[0, 1]
        print(f"   [Exploratory] Correlation(college wOBA, milb_year1_wOBA): {corr_y1:.4f}")

    # For pitchers: college FIP vs milb_peak_FIP
    if len(pitchers) >= 10:
        college_fip = [safe_float(r.get("FIP"), 0) for r in pitchers]
        peak_fip = [safe_float(r.get("milb_peak_FIP"), 0) for r in pitchers]
        corr_fip = np.corrcoef(college_fip, peak_fip)[0, 1]
        print(f"   [Exploratory] Correlation(college FIP, milb_peak_FIP): {corr_fip:.4f}")

        # draft_round vs milb_peak_FIP (lower FIP = better, so should be positive or no clear signal)
        draft_rounds_p = [safe_float(r.get("draft_round"), 0) for r in pitchers]
        corr_fip_dr = np.corrcoef(draft_rounds_p, peak_fip)[0, 1]
        print(f"   [Exploratory] Correlation(draft_round, milb_peak_FIP): {corr_fip_dr:.4f}")

    # ── 4. Random sample ──
    print("\n4. Five random players with all computed fields:")
    COMPUTED_FIELDS = [
        "player_name", "draft_year", "player_type", "draft_round",
        "milb_year1_wOBA", "milb_year1_FIP", "milb_year1_level",
        "milb_year1_batting_games", "milb_year1_pitching_games",
        "milb_peak_wOBA", "milb_peak_FIP", "milb_highest_level",
        "milb_years_count", "has_mlb_debut",
        "conf_strength", "round_logit_prior", "nn_mlb_rate",
        "wOBA", "OPS", "BB_pct", "K_pct",
        "ERA", "FIP", "K_per_nine", "BB_per_nine",
        "height_inches", "bmi", "Age",
    ]
    random.seed(42)
    sample = random.sample(data, min(5, len(data)))
    for i, rec in enumerate(sample, 1):
        print(f"\n   Player {i}:")
        for field in COMPUTED_FIELDS:
            val = rec.get(field, "MISSING")
            if val is None:
                val = "None"
            print(f"     {field:<30s} {val}")

    # ── 5. Re-run validation checks ──
    print("\n\n5. Re-running build_milb_training.py validation checks:")
    print("   (Checks V1-V6 as defined in build_milb_training.py)")

    # V1: Year distribution ranges (check if years are reasonable)
    total = len(data)
    y2021_pct = 100 * dy_counts.get(2021, 0) / max(total, 1)
    y2022_pct = 100 * dy_counts.get(2022, 0) / max(total, 1)
    y2023_pct = 100 * dy_counts.get(2023, 0) / max(total, 1)
    v1_pass = (10 <= y2021_pct <= 40) and (20 <= y2022_pct <= 50) and (10 <= y2023_pct <= 55)
    print(f"   [V1] Year dist: 2021={y2021_pct:.1f}% 2022={y2022_pct:.1f}% 2023={y2023_pct:.1f}% "
          f"— {'PASS' if v1_pass else 'FAIL'}")

    # V2: Hitter/pitcher split
    h_pct = 100 * pt_counts.get("hitter", 0) / max(total, 1)
    p_pct = 100 * pt_counts.get("pitcher", 0) / max(total, 1)
    v2_pass = 35 <= h_pct <= 65 and 35 <= p_pct <= 65
    print(f"   [V2] Hitter/pitcher split: {pt_counts.get('hitter',0)} ({h_pct:.1f}%) / "
          f"{pt_counts.get('pitcher',0)} ({p_pct:.1f}%) — {'PASS' if v2_pass else 'FAIL'}")

    # V3: Non-null peak stats
    hit_null_peak = sum(1 for r in hitters if r.get("milb_peak_wOBA") is None)
    pit_null_peak = sum(1 for r in pitchers if r.get("milb_peak_FIP") is None)
    v3_pass = hit_null_peak == 0 and pit_null_peak == 0
    print(f"   [V3] Non-null peak stats: hitter null_peak={hit_null_peak}, pitcher null_peak={pit_null_peak} "
          f"— {'PASS' if v3_pass else 'FAIL'}")

    # V4: Correlation check (same as V5 in build script — college wOBA vs milb_peak_wOBA > 0.10)
    if len(hitters) >= 10:
        v4_pass = corr_1 > 0.10
    else:
        v4_pass = True
    print(f"   [V4] Correlation(college wOBA, milb_peak_wOBA) > 0.10: "
          f"{'PASS' if v4_pass else 'FAIL'}")

    # V5: Correlation(draft_round, milb_peak_wOBA) < -0.05
    if len(hitters) >= 10:
        v5_pass = corr_2 < -0.05
    else:
        v5_pass = True
    print(f"   [V5] Correlation(draft_round, milb_peak_wOBA) < -0.05: "
          f"{'PASS' if v5_pass else 'FAIL'}")

    # V6: Total records check
    v6_pass = len(data) >= 800
    print(f"   [V6] Total records >= 800: {len(data)} — {'PASS' if v6_pass else 'FAIL'}")

    all_pass = v1_pass and v2_pass and v3_pass and v4_pass and v5_pass and v6_pass
    print(f"\n   Overall: {'ALL 6 CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total records: {len(data)}")
    print(f"  Years: {dict(sorted(dy_counts.items()))}")
    print(f"  Hitter/pitcher: {dict(pt_counts)}")
    print(f"  MLB debut rate: {100 * debuted / max(len(data), 1):.1f}%")
    print(f"  Validation: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"  File: {TRAIN_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
