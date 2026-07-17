#!/usr/bin/env python3
"""
build_milb_training.py — Build MiLB extended training set.

Joins milb_{year}.json stats with expanded_training_set.json,
computes peak wOBA/FIP/level features, applies filters for 2021-2023
draftees, and writes to data/training/milb_extended_training.json.

Usage:
    python3 scripts/build_milb_training.py
"""

import json
import math
import sys
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]

MILB_DIR = BASE / "data" / "milb"
TRAINING_DIR = BASE / "data" / "training"
ARTIFACTS_DIR = BASE / "models" / "artifacts_full"

EXPANDED_TRAINING_PATH = TRAINING_DIR / "expanded_training_set.json"
CONF_STRENGTH_PATH = ARTIFACTS_DIR / "conference_strength.json"
OUTPUT_PATH = TRAINING_DIR / "milb_extended_training.json"

YEARS = [2021, 2022, 2023, 2024, 2025]

LEVEL_MAP = {"A": 1, "A+": 2, "AA": 3, "AAA": 4}

WOBACONST = {"bb": 0.69, "hbp": 0.72, "single": 0.88,
             "double": 1.24, "triple": 1.56, "hr": 1.95}
FIP_CONSTANT = 3.10


def load_json(path):
    with open(path) as f:
        return json.load(f)


def safe_float(v, default=None):
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def parse_ip(ip_str):
    """Parse innings pitched string like '65.1' to decimal innings."""
    if ip_str is None:
        return 0.0
    try:
        ip_str = str(ip_str)
        if "." in ip_str:
            parts = ip_str.split(".")
            if len(parts) == 2 and parts[1] in ("1", "2"):
                whole = int(parts[0])
                frac = int(parts[1])
                return whole + (1.0 if frac == 1 else 2.0) / 3.0
        return float(ip_str)
    except (ValueError, TypeError):
        return 0.0


def compute_woba(batting):
    """Compute wOBA from batting stats dict."""
    bb = safe_float(batting.get("baseOnBalls"), 0)
    hbp = safe_float(batting.get("hitByPitch"), 0)
    hits = safe_float(batting.get("hits"), 0)
    doubles = safe_float(batting.get("doubles"), 0)
    triples = safe_float(batting.get("triples"), 0)
    hr = safe_float(batting.get("homeRuns"), 0)
    pa = safe_float(batting.get("plateAppearances"), 0)

    if pa == 0:
        ab = safe_float(batting.get("atBats"), 0)
        sf = safe_float(batting.get("sacFlies"), 0)
        pa = ab + bb + hbp + sf

    if pa == 0:
        return 0.0, 0

    singles = hits - doubles - triples - hr
    numerator = (WOBACONST["bb"] * bb + WOBACONST["hbp"] * hbp +
                 WOBACONST["single"] * singles + WOBACONST["double"] * doubles +
                 WOBACONST["triple"] * triples + WOBACONST["hr"] * hr)
    return numerator / pa, pa


def compute_fip(pitching):
    """Compute FIP from pitching stats dict."""
    hr = safe_float(pitching.get("homeRuns"), 0)
    bb = safe_float(pitching.get("baseOnBalls"), 0)
    hbp = safe_float(pitching.get("hitBatsmen"), 0)
    so = safe_float(pitching.get("strikeOuts"), 0)
    ip_str = pitching.get("inningsPitched", "0")
    ip = parse_ip(ip_str)

    if ip == 0:
        return 0.0, 0.0

    fip = (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT
    return max(fip, 0.0), ip


def compute_milb_features(players_by_pid):
    """
    For each player, compute MiLB hitting AND pitching features.
    Returns dict with all computed stats for every player.
    """
    results = {}

    for pid, seasons in players_by_pid.items():
        seasons = sorted(seasons, key=lambda s: s.get("season", 0))

        # === HITTING features ===
        # Track wOBA per season with enough PA
        season_wobas = []  # (woba, pa, season_idx)
        season_pa_list = []
        for s in seasons:
            b = s.get("batting")
            if b:
                woba, pa = compute_woba(b)
                if pa > 0:
                    season_wobas.append((woba, pa))
                season_pa_list.append(pa)
            else:
                season_pa_list.append(0)

        # Year 1 hitting
        if season_wobas:
            s1_woba, s1_pa = season_wobas[0]
        else:
            s1_woba, s1_pa = 0.0, 0

        # For year1_games, use first season's batting gamesPlayed
        b1 = seasons[0].get("batting", {})
        s1_batting_games = safe_float(b1.get("gamesPlayed"), 0) if b1 else 0

        # Peak wOBA from seasons 2-3 (index 1, 2)
        later_wobas = [w for i, (w, pa) in enumerate(season_wobas) if i >= 1 and i < 4 and pa >= 30]
        peak_woba = max(later_wobas) if later_wobas else s1_woba

        # === PITCHING features ===
        season_fips = []  # (fip, ip, season_idx)
        season_ip_list = []
        for s in seasons:
            p = s.get("pitching")
            if p:
                fip_val, ip = compute_fip(p)
                if ip > 0:
                    season_fips.append((fip_val, ip))
                season_ip_list.append(ip)
            else:
                season_ip_list.append(0.0)

        # Year 1 pitching
        if season_fips:
            s1_fip, s1_ip = season_fips[0]
        else:
            s1_fip, s1_ip = 0.0, 0.0

        # For year1_games (pitching), use first season's pitching gamesPlayed
        p1 = seasons[0].get("pitching", {})
        s1_pitching_games = safe_float(p1.get("gamesPlayed"), 0) if p1 else 0

        # Peak FIP from seasons 2-3 (lowest = best)
        later_fips = [f for i, (f, ip) in enumerate(season_fips) if i >= 1 and i < 4 and ip >= 10]
        peak_fip = min(later_fips) if later_fips else s1_fip

        # Highest level reached
        highest_level = max(LEVEL_MAP.get(s.get("level", "A"), 1) for s in seasons)

        # Year 1 level
        s1_level = LEVEL_MAP.get(seasons[0].get("level"), 0)

        results[pid] = {
            # Hitting stats
            "milb_year1_wOBA": round(s1_woba, 4),
            "milb_year1_pa": s1_pa,
            "milb_year1_batting_games": s1_batting_games,
            "milb_peak_wOBA": round(peak_woba, 4),
            # Pitching stats
            "milb_year1_FIP": round(s1_fip, 4),
            "milb_year1_ip": s1_ip,
            "milb_year1_pitching_games": s1_pitching_games,
            "milb_peak_FIP": round(peak_fip, 4),
            # Level
            "milb_year1_level": s1_level,
            "milb_highest_level": highest_level,
            "milb_years_count": len(seasons),
            "milb_person_id": pid,
        }

    return results


def compute_round_logit_prior(records, training_years=None):
    """Compute round_logit_prior from empirical MLB debut rates per round."""
    round_stats = defaultdict(lambda: {"total": 0, "debut": 0})

    for r in records:
        dy = r.get("draft_year")
        if training_years and dy not in training_years:
            continue
        rnd = r.get("draft_round")
        if rnd is None:
            continue
        try:
            rnd = int(float(rnd))
        except (ValueError, TypeError):
            rnd = 20
        has_debut = 1 if r.get("mlb_debut_date") and r["mlb_debut_date"] != "None" else 0
        round_stats[rnd]["total"] += 1
        if has_debut:
            round_stats[rnd]["debut"] += 1

    round_rates = {}
    for rnd, stats in round_stats.items():
        rate = stats["debut"] / max(stats["total"], 1)
        logit = math.log(max(rate, 0.001) / max(1 - rate, 0.001))
        round_rates[rnd] = logit

    return round_rates


def compute_nn_mlb_rate(records, sim_stats, n_neighbors=20):
    """For each player, find n nearest neighbors and compute MLB debut rate."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    n = len(records)
    if n < 2:
        return [0.0] * n

    X_sim = []
    for p in records:
        row = [safe_float(p.get(s), 0) or 0 for s in sim_stats]
        X_sim.append(row)
    X_sim = np.array(X_sim)

    scaler = StandardScaler()
    X_sim_norm = scaler.fit_transform(X_sim)

    k = min(n_neighbors + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(X_sim_norm)

    distances, indices = nn.kneighbors(X_sim_norm, n_neighbors=k)

    nn_rates = []
    for i in range(n):
        neighbor_indices = indices[i][1:] if len(indices[i]) > 1 else indices[i]
        neighbor_debuts = [
            1 if records[j].get("mlb_debut_date") and records[j]["mlb_debut_date"] != "None"
            else 0
            for j in neighbor_indices
        ]
        nn_rates.append(np.mean(neighbor_debuts) if neighbor_debuts else 0.0)

    return nn_rates


def main():
    print("=" * 60)
    print("BUILD MILB EXTENDED TRAINING SET")
    print("=" * 60)

    # ── 1. Load all MiLB data ──
    print("\n1. Loading MiLB data (2021-2025)...")
    all_milb_players = []
    for year in YEARS:
        path = MILB_DIR / f"milb_{year}.json"
        print(f"   Loading {path.name}...")
        data = load_json(path)
        players = data.get("players", [])
        for p in players:
            p["year"] = year
        all_milb_players.extend(players)
        print(f"     {len(players)} players")

    total_milb = len(all_milb_players)
    unique_pids = len(set(p["person_id"] for p in all_milb_players))
    print(f"   Total MiLB records: {total_milb}")
    print(f"   Unique person_ids: {unique_pids}")

    # ── 2. Index by person_id ──
    print("\n2. Indexing by person_id...")
    milb_by_pid = defaultdict(list)
    for p in all_milb_players:
        milb_by_pid[p["person_id"]].append(p)
    print(f"   {len(milb_by_pid)} unique players")

    # ── 3. Compute MiLB features ──
    print("\n3. Computing MiLB features (wOBA/FIP/level)...")
    milb_features = compute_milb_features(milb_by_pid)
    print(f"   Features computed for {len(milb_features)} players")

    # ── 4. Load expanded training set ──
    print("\n4. Loading expanded training set...")
    ets = load_json(EXPANDED_TRAINING_PATH)
    print(f"   {len(ets)} records (player-seasons)")

    # Get signed draftees only
    signed = [r for r in ets if r.get("draft_pick") and r["draft_pick"] > 0]

    # Compute unique-player stats for reference
    signed_by_pid = {}
    for r in signed:
        pid = r.get("person_id")
        if pid is None:
            continue
        if pid not in signed_by_pid or (r.get("season") or 0) > (signed_by_pid[pid].get("season") or 0):
            signed_by_pid[pid] = r
    signed_unique = list(signed_by_pid.values())
    unique_count = len(signed_unique)

    print(f"   Signed draftees (player-seasons): {len(signed)}")
    print(f"   Signed draftees (unique players): {unique_count}")

    # Count by draft year and player_type for unique players (for reference only)
    dy_unique = defaultdict(int)
    pt_unique = defaultdict(int)
    for r in signed_unique:
        dy_unique[r.get("draft_year")] += 1
        pt_unique[r.get("player_type", "unknown")] += 1
    print(f"   Year distribution (unique): {dict(sorted(dy_unique.items()))}")
    print(f"   Player type distribution (unique): {dict(pt_unique)}")

    # Load draft data for MLB debut dates
    draft_path = BASE / "data" / "draft" / "draft_all_picks.json"
    draft_data = load_json(draft_path)
    debut_idx = {}
    for p in draft_data:
        pid = p.get("person_id")
        if pid:
            debut = p.get("mlb_debut_date")
            debut_idx[pid] = debut if debut and debut != "None" else None
    print(f"   Draft records with debut dates: {sum(1 for v in debut_idx.values() if v)}")

    # ── 5. Join MiLB features with expanded training set ──
    print("\n5. Joining MiLB features with expanded training set (all player-seasons)...")
    joined = []
    matched_pids = set()
    total_pids = set()

    for rec in signed:
        pid = rec.get("person_id")
        total_pids.add(pid)
        mf = milb_features.get(pid)

        if mf is None:
            continue

        matched_pids.add(pid)

        combined = dict(rec)
        combined.update(mf)

        # Add MLB debut label
        debut = debut_idx.get(pid)
        combined["mlb_debut_date"] = debut if debut else None
        combined["has_mlb_debut"] = 1 if debut else 0

        # Add a field indicating which MiLB stat is primary based on college player_type
        college_type = rec.get("player_type", "hitter")
        combined["milb_college_type"] = college_type

        joined.append(combined)

    # Compute per-year match rates (unique players)
    matched_years = defaultdict(int)
    total_by_year = defaultdict(int)
    for rec in signed_unique:
        pid = rec.get("person_id")
        dy = rec.get("draft_year")
        total_by_year[dy] += 1
        if pid in matched_pids:
            matched_years[dy] += 1

    total_signed = len(total_pids)
    matched_count = len(matched_pids)
    join_rate = matched_count / max(total_signed, 1) * 100

    target_total = sum(total_by_year.get(y, 0) for y in (2021, 2022, 2023))
    target_matched = sum(matched_years.get(y, 0) for y in (2021, 2022, 2023))
    target_join_rate = target_matched / max(target_total, 1) * 100 if target_total > 0 else 0

    print(f"   Matched {matched_count}/{total_signed} unique players overall ({join_rate:.1f}%)")
    print(f"   Matched (2021-2023 only): {target_matched}/{target_total} ({target_join_rate:.1f}%)")
    for y in sorted(total_by_year.keys()):
        print(f"     {y}: {matched_years.get(y, 0)}/{total_by_year[y]} "
              f"({100*matched_years.get(y,0)/max(total_by_year[y],1):.1f}%)")
    print(f"   Joined records (player-seasons): {len(joined)}")

    # ── 6. Apply filters ──
    print("\n6. Applying filters...")

    # Filter 1: Minimum playing time using COLLEGE player_type
    filtered = []
    for r in joined:
        college_type = r.get("player_type", "hitter")
        if college_type == "hitter":
            pa = r.get("milb_year1_pa", 0) or 0
            if pa >= 50:
                filtered.append(r)
        else:  # pitcher
            ip = r.get("milb_year1_ip", 0) or 0
            if ip >= 20:
                filtered.append(r)

    before_pt = len(filtered)
    print(f"   After minimum PA/IP filter (by college type): {len(filtered)} (from {len(joined)})")

    # Check split by college type
    h_before = sum(1 for r in filtered if r.get("player_type") == "hitter")
    p_before = sum(1 for r in filtered if r.get("player_type") == "pitcher")
    print(f"     Hitters (college): {h_before}, Pitchers (college): {p_before}")

    # Filter 2: Draft year 2021-2023
    filtered = [r for r in filtered if r.get("draft_year") in (2021, 2022, 2023)]
    print(f"   After draft year filter (2021-2023): {len(filtered)} (from {before_pt})")

    # Separate training and validation
    training = [r for r in filtered if r.get("draft_year") in (2021, 2022)]
    validation = [r for r in filtered if r.get("draft_year") == 2023]
    print(f"   Training (2021-2022): {len(training)}")
    print(f"   Validation (2023): {len(validation)}")

    # ── 7. Add derived college features ──
    print("\n7. Adding derived college features...")

    # 7a. conf_strength
    conf_strength_data = load_json(CONF_STRENGTH_PATH)
    for r in filtered:
        conf = r.get("conference", "")
        strength = conf_strength_data.get(conf, {}).get("strength", 1.0)
        r["conf_strength"] = strength
    print(f"   Added conf_strength for {len(filtered)} records")

    # 7b. round_logit_prior
    round_rates = compute_round_logit_prior(filtered, training_years={2021, 2022})
    print(f"   Round logit priors computed for {len(round_rates)} rounds:")
    for rnd in sorted(round_rates.keys()):
        print(f"     Round {rnd}: {round_rates[rnd]:+.3f}")

    for r in filtered:
        rnd = r.get("draft_round")
        try:
            rnd = int(float(rnd))
        except (ValueError, TypeError):
            rnd = 20
        r["round_logit_prior"] = round_rates.get(rnd, -1.0)
    print(f"   Added round_logit_prior")

    # 7c. nn_mlb_rate
    hitter_sim_stats = ["wOBA", "OBP", "SLG", "BB_pct", "K_pct", "ISO"]
    pitcher_sim_stats = ["FIP", "ERA", "WHIP", "K_per_nine", "BB_per_nine", "K_pct"]

    hitters_nn = [r for r in filtered if r.get("player_type") == "hitter"]
    pitchers_nn = [r for r in filtered if r.get("player_type") == "pitcher"]

    if hitters_nn:
        print(f"\n   Computing NN MLB rates for {len(hitters_nn)} hitters...")
        h_nn_rates = compute_nn_mlb_rate(hitters_nn, hitter_sim_stats)
        for i, r in enumerate(hitters_nn):
            r["nn_mlb_rate"] = h_nn_rates[i]

    if pitchers_nn:
        print(f"   Computing NN MLB rates for {len(pitchers_nn)} pitchers...")
        p_nn_rates = compute_nn_mlb_rate(pitchers_nn, pitcher_sim_stats)
        for i, r in enumerate(pitchers_nn):
            r["nn_mlb_rate"] = p_nn_rates[i]

    print(f"   Added nn_mlb_rate")

    # ── 8. Validation checks ──
    print("\n8. Running validation checks...")
    checks_passed = True
    fail_reasons = []

    # V1: Join rate >= 90% for 2021-2023 target draftees
    v1_pass = target_join_rate >= 90.0
    if not v1_pass:
        fail_reasons.append(
            f"V1 FAIL: Join rate {target_join_rate:.1f}% < 90% (2021-2023 draftees)"
        )
        checks_passed = False
    print(f"   [V1] Join rate (2021-2023): {target_join_rate:.1f}% "
          f"({target_matched}/{target_total}) — {'PASS' if v1_pass else 'FAIL'}")

    # V2: Year distribution (based on draft_year of filtered records)
    year_counts = defaultdict(int)
    for r in filtered:
        year_counts[r.get("draft_year")] += 1
    total_filtered = len(filtered)
    y2021_pct = 100 * year_counts.get(2021, 0) / max(total_filtered, 1)
    y2022_pct = 100 * year_counts.get(2022, 0) / max(total_filtered, 1)
    y2023_pct = 100 * year_counts.get(2023, 0) / max(total_filtered, 1)

    in_range_2021 = 10 <= y2021_pct <= 40
    in_range_2022 = 20 <= y2022_pct <= 50
    in_range_2023 = 10 <= y2023_pct <= 55

    v2_pass = in_range_2021 and in_range_2022 and in_range_2023
    if not v2_pass:
        fail_reasons.append(
            f"V2 FAIL: Year distribution 2021={y2021_pct:.1f}% 2022={y2022_pct:.1f}% "
            f"2023={y2023_pct:.1f}% — expected ~10-40%, ~20-50%, ~10-55%"
        )
        checks_passed = False
    print(f"   [V2] Year distribution: 2021={y2021_pct:.1f}% 2022={y2022_pct:.1f}% "
          f"2023={y2023_pct:.1f}% (target: 10-40/20-50/10-55) — {'PASS' if v2_pass else 'FAIL'}")

    # V3: Hitter/pitcher split (using college player_type)
    h_count = sum(1 for r in filtered if r.get("player_type") == "hitter")
    p_count = sum(1 for r in filtered if r.get("player_type") == "pitcher")
    h_pct = 100 * h_count / max(total_filtered, 1)
    p_pct = 100 * p_count / max(total_filtered, 1)
    v3_pass = 35 <= h_pct <= 65 and 35 <= p_pct <= 65
    if not v3_pass:
        fail_reasons.append(
            f"V3 FAIL: Hitter/pitcher split {h_pct:.1f}%/{p_pct:.1f}% — "
            f"not balanced (~35-65% each)"
        )
        checks_passed = False
    print(f"   [V3] Hitter/pitcher split (college type): {h_count} ({h_pct:.1f}%) / "
          f"{p_count} ({p_pct:.1f}%) — {'PASS' if v3_pass else 'FAIL'}")

    # V4: Non-null peak wOBA/FIP
    hitters_ck = [r for r in filtered if r.get("player_type") == "hitter"]
    pitchers_ck = [r for r in filtered if r.get("player_type") == "pitcher"]
    hit_peak_null = sum(1 for r in hitters_ck if r.get("milb_peak_wOBA") is None)
    pit_peak_null = sum(1 for r in pitchers_ck if r.get("milb_peak_FIP") is None)
    v4_pass = hit_peak_null == 0 and pit_peak_null == 0
    if not v4_pass:
        fail_reasons.append(
            f"V4 FAIL: {hit_peak_null} hitters with null milb_peak_wOBA, "
            f"{pit_peak_null} pitchers with null milb_peak_FIP"
        )
        checks_passed = False
    print(f"   [V4] Non-null peak stats: hitters {hit_peak_null} null, "
          f"pitchers {pit_peak_null} null — {'PASS' if v4_pass else 'FAIL'}")

    # V5: Correlation(college wOBA, milb_peak_wOBA) > 0.10
    if len(hitters_ck) >= 10:
        college_wobas = [safe_float(r.get("wOBA"), 0) for r in hitters_ck]
        peak_wobas = [safe_float(r.get("milb_peak_wOBA"), 0) for r in hitters_ck]
        corr_matrix = np.corrcoef(college_wobas, peak_wobas)
        v5_corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0
        v5_pass = v5_corr > 0.10
        if not v5_pass:
            fail_reasons.append(
                f"V5 FAIL: Correlation(college wOBA, milb_peak_wOBA) = {v5_corr:.4f} <= 0.10"
            )
            checks_passed = False
        print(f"   [V5] Correlation(college wOBA, milb_peak_wOBA) = {v5_corr:.4f} — "
              f"{'PASS' if v5_pass else 'FAIL'}")
    else:
        v5_pass = True
        v5_corr = 0.0
        print(f"   [V5] Not enough hitters ({len(hitters_ck)}) to compute — SKIP")

    # V6: Correlation(draft_round, milb_peak_wOBA) < -0.05
    if len(hitters_ck) >= 10:
        draft_rounds = [safe_float(r.get("draft_round"), 0) for r in hitters_ck]
        peak_wobas = [safe_float(r.get("milb_peak_wOBA"), 0) for r in hitters_ck]
        corr_matrix = np.corrcoef(draft_rounds, peak_wobas)
        v6_corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0
        v6_pass = v6_corr < -0.05
        if not v6_pass:
            fail_reasons.append(
                f"V6 FAIL: Correlation(draft_round, milb_peak_wOBA) = {v6_corr:.4f} >= -0.05"
            )
            checks_passed = False
        print(f"   [V6] Correlation(draft_round, milb_peak_wOBA) = {v6_corr:.4f} — "
              f"{'PASS' if v6_pass else 'FAIL'}")
    else:
        v6_pass = True
        v6_corr = 0.0
        print(f"   [V6] Not enough hitters ({len(hitters_ck)}) to compute — SKIP")

    # ── 9. Write output ──
    print(f"\n9. Writing output to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"   Written {len(filtered)} records")

    # ── 10. Print summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal records: {len(filtered)}")

    print(f"\nRecords by draft year:")
    for year in sorted(year_counts.keys()):
        count = year_counts[year]
        pct = 100 * count / max(total_filtered, 1)
        print(f"  {year}: {count} ({pct:.1f}%)")

    print(f"\nRecords by player type (college classification):")
    print(f"  Hitters: {h_count} ({h_pct:.1f}%)")
    print(f"  Pitchers: {p_count} ({p_pct:.1f}%)")

    if h_count > 0:
        mean_peak_woba = np.mean([
            safe_float(r.get("milb_peak_wOBA"), 0)
            for r in filtered if r.get("player_type") == "hitter"
        ])
        print(f"\nMean milb_peak_wOBA (hitters): {mean_peak_woba:.4f}")

    if p_count > 0:
        mean_peak_fip = np.mean([
            safe_float(r.get("milb_peak_FIP"), 0)
            for r in filtered if r.get("player_type") == "pitcher"
        ])
        print(f"Mean milb_peak_FIP (pitchers): {mean_peak_fip:.4f}")

    level_dist = defaultdict(int)
    for r in filtered:
        lvl = r.get("milb_year1_level")
        level_dist[lvl] += 1
    level_labels = {1: "A", 2: "A+", 3: "AA", 4: "AAA"}
    print(f"\nmilb_year1_level distribution:")
    for lvl in sorted(level_dist.keys()):
        label = level_labels.get(lvl, f"Level {lvl}")
        count = level_dist[lvl]
        pct = 100 * count / max(total_filtered, 1)
        print(f"  {label}: {count} ({pct:.1f}%)")

    print(f"\nJoin rate (2021-2023): {target_join_rate:.1f}% "
          f"({target_matched}/{target_total})")

    if len(hitters_ck) >= 10:
        print(f"\nCorrelation(college wOBA, milb_peak_wOBA): {v5_corr:.4f}")
        print(f"Correlation(draft_round, milb_peak_wOBA): {v6_corr:.4f}")

    print(f"\nValidation checks: {'ALL PASSED' if checks_passed else 'SOME FAILED'}")
    if not checks_passed:
        for reason in fail_reasons:
            print(f"  - {reason}")

    print(f"\nOutput file: {OUTPUT_PATH}")
    print(f"Output records: {len(filtered)}")

    return checks_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
