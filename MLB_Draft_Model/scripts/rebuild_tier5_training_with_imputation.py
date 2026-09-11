#!/usr/bin/env python3
"""Rebuild Tier 5 training set with WAR=0 imputation for non-debuted players.

For players drafted 2015-2019 who haven't debuted in MLB by 2025, we can safely
impute WAR=0. They've had 5+ years to reach the majors; if they haven't debuted
by now, they're extremely unlikely to ever accumulate positive WAR.

This increases the training set from ~216 records with WAR to ~1,500+ records,
providing much better signal for the value regression model.

Pipeline:
  1. Load WAR ground truth (2015-2023)
  2. For draft years 2015-2019: if no MLB debut, set WAR=0
  3. For draft years 2020-2023: keep as-is (still pending, <5 years)
  4. Join onto expanded training set (college stats)
  5. Compute signed_log_war and hurdle labels
  6. Apply temporal split: train=2015-2020, heldout=2021-2022, test=2023

Usage:
    python scripts/rebuild_tier5_training_with_imputation.py [--verify]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
WAR_PATH = BASE / "data" / "war" / "war_ground_truth.json"
EXPANDED_PATH = BASE / "data" / "training" / "expanded_training_set.json"
OUTPUT_PATH = BASE / "data" / "training" / "tier5_training_set_v2.json"
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
CONF_STRENGTH_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"
TRACKMAN_PATH = BASE / "data" / "trackman" / "trackman_player_features.json"

# Draft years where we can safely impute WAR=0 for non-debuted players
# (5+ years have passed since draft, as of 2026)
# 2021: 5 years passed (2021-2026)
# 2022: 4 years passed (borderline, but reasonable)
# 2023: 3 years passed (too early, keep as null)
IMPUTE_ZERO_YEARS = range(2015, 2023)  # 2015-2022

# Tier 3 feature engineering constants (copied from build_tier5_training.py)
HITTER_SIM_STATS = ["wOBA_adj", "OPS_adj", "AVG_adj", "SLG_adj", "BB_pct_adj", "K_pct_adj", "ISO_adj"]
PITCHER_SIM_STATS = ["ERA_adj", "FIP_adj", "WHIP_adj", "K_per_nine_adj", "BB_per_nine_adj", "K_pct_adj"]

HITTER_ADJ = HITTER_SIM_STATS + ["wRC_plus_adj"]
PITCHER_ADJ = PITCHER_SIM_STATS + ["BB_pct_adj"]

HITTER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in HITTER_ADJ]
PITCHER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in PITCHER_ADJ]

ADJ_FEATURE_MAP = {
    "wOBA_adj": "wOBA", "OPS_adj": "OPS", "AVG_adj": "AVG", "SLG_adj": "SLG",
    "BB_pct_adj": "BB_pct", "K_pct_adj": "K_pct", "ISO_adj": "ISO", "wRC_plus_adj": "wRC_plus",
    "ERA_adj": "ERA", "FIP_adj": "FIP", "WHIP_adj": "WHIP",
    "K_per_nine_adj": "K_per_nine", "BB_per_nine_adj": "BB_per_nine",
}
INTERACTION_BASE_MAP = {k: v for k, v in [
    ("strength_x_wOBA", "wOBA_adj"), ("strength_x_OPS", "OPS_adj"),
    ("strength_x_AVG", "AVG_adj"), ("strength_x_SLG", "SLG_adj"),
    ("strength_x_BB_pct", "BB_pct_adj"), ("strength_x_K_pct", "K_pct_adj"),
    ("strength_x_ISO", "ISO_adj"), ("strength_x_wRC_plus", "wRC_plus_adj"),
    ("strength_x_ERA", "ERA_adj"), ("strength_x_FIP", "FIP_adj"),
    ("strength_x_WHIP", "WHIP_adj"),
    ("strength_x_K_per_nine", "K_per_nine_adj"), ("strength_x_BB_per_nine", "BB_per_nine_adj"),
]}
# College stats only exist for 2021-2025, so we use:
# - 2021-2022: train (4-5 years of WAR data through 2026)
# - 2023: heldout (3 years of WAR data, some right-censoring)
# 2024-2025: inference only (too recent for WAR)
# Temporal split: 2021 train, 2022 heldout (balanced WAR distribution)
# 2023 dropped due to right-censoring bias (2x higher WAR than 2021-2022)
TRAIN_YEARS = (2021,)
HELDOUT_YEARS = (2022,)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def signed_log_war(war: float) -> float:
    """sign(war) * log1p(|war|)"""
    if war is None:
        return None
    sign = 1 if war >= 0 else -1
    return sign * math.log1p(abs(war))


def safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_conf_avg(conf_stats, conf, season, ptype, stat):
    ps = conf_stats.get("per_season", {})
    co = conf_stats.get("conference_overall", {})
    tf = conf_stats.get("tier_fallback", {})
    sd = ps.get(conf, {}).get(str(season), {})
    if isinstance(sd, dict) and stat in sd.get(ptype, {}):
        return sd[ptype][stat]
    cd = co.get(conf, {}).get(ptype, {})
    if stat in cd:
        return cd[stat]
    return tf.get("3", {}).get(ptype, {}).get(stat, 0.0)


def add_features(records, conf_stats, conf_strength):
    """Add conference-adjusted features (Tier 3 feature engineering)."""
    for rec in records:
        ptype = rec.get("player_type", "hitter")
        conf = rec.get("conference", "")
        season = rec.get("season")
        strength = conf_strength.get(conf, {}).get("strength", 1.0)
        rec["conf_strength"] = strength
        adj_list = HITTER_ADJ if ptype == "hitter" else PITCHER_ADJ
        for af in adj_list:
            rs = ADJ_FEATURE_MAP.get(af, af.replace("_adj", ""))
            rv = safe_float(rec.get(rs))
            rec[af] = round(rv - get_conf_avg(conf_stats, conf, season, ptype, rs), 4) if rv is not None else 0.0
        il = HITTER_INTERACTIONS if ptype == "hitter" else PITCHER_INTERACTIONS
        for inf in il:
            ba = INTERACTION_BASE_MAP.get(inf)
            rec[inf] = round(strength * rec.get(ba, 0), 4) if ba else 0.0
    return records


def compute_nn_mlb_rates(players, sim_stats):
    """Compute nearest-neighbor MLB debut rate."""
    n = len(players)
    if n == 0:
        return []
    X_sim = []
    for p in players:
        row = [safe_float(p.get(s, 0)) or 0 for s in sim_stats]
        X_sim.append(row)
    X_sim = np.array(X_sim)

    scaler = StandardScaler()
    X_sim_norm = scaler.fit_transform(X_sim)

    nn = NearestNeighbors(n_neighbors=min(21, n), metric="euclidean", n_jobs=-1)
    nn.fit(X_sim_norm)

    distances, indices = nn.kneighbors(X_sim_norm, n_neighbors=min(21, n))

    nn_mlb_rates = []
    for i in range(n):
        neighbor_indices = indices[i][1:] if len(indices[i]) > 1 else indices[i]
        neighbor_rates = [players[j].get("has_mlb_debut", 0) for j in neighbor_indices]
        nn_mlb_rates.append(np.mean(neighbor_rates) if neighbor_rates else 0.0)

    return nn_mlb_rates


def build_round_rates(players):
    """Build draft round -> MLB debut rate mapping."""
    round_rates = {}
    for p in players:
        rnd = p.get("draft_round")
        try:
            rnd = int(float(rnd)) if rnd is not None else 20
        except (TypeError, ValueError):
            rnd = 20
        if rnd not in round_rates:
            round_rates[rnd] = {"total": 0, "debut": 0}
        round_rates[rnd]["total"] += 1
        if p.get("has_mlb_debut"):
            round_rates[rnd]["debut"] += 1
    for rnd in round_rates:
        rr = round_rates[rnd]
        rate = rr["debut"] / max(rr["total"], 1)
        logit = math.log(max(rate, 0.001) / max(1 - rate, 0.001))
        rr["rate"] = rate
        rr["logit"] = float(logit)
    return round_rates


def assign_round_logit_prior(players, round_rates):
    """Assign round logit prior to each player."""
    for p in players:
        rnd = p.get("draft_round", 10)
        rr = round_rates.get(rnd, {"logit": -1.0})
        p["round_logit_prior"] = rr["logit"]
    return players


def merge_trackman_features(players, trackman_data):
    """Merge TrackMan features if available.
    
    Note: TrackMan uses internal 5-11 digit IDs, not MLBAM person_ids.
    This function attempts to join on player_id/person_id but will gracefully
    handle the case where no matches are found.
    """
    if not trackman_data:
        for p in players:
            p["avg_ev_wood_adj"] = None
            p["avg_velo"] = None
        return players, 0

    pitchers = trackman_data.get("pitchers", {})
    hitters = trackman_data.get("hitters", {})
    joined = 0
    
    for p in players:
        key = str(p.get("person_id"))
        if p.get("player_type") == "hitter":
            matched = hitters.get(key)
            if matched:
                joined += 1
                latest = max(matched, key=lambda r: r.get("season", 0) or 0)
                p["avg_ev_wood_adj"] = latest.get("avg_ev_wood_adj")
            else:
                p["avg_ev_wood_adj"] = None
            p["avg_velo"] = None
        else:
            matched = pitchers.get(key)
            if matched:
                joined += 1
                latest = max(matched, key=lambda r: r.get("season", 0) or 0)
                p["avg_velo"] = latest.get("avg_velo")
            else:
                p["avg_velo"] = None
            p["avg_ev_wood_adj"] = None
    
    return players, joined


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild Tier 5 training set with WAR=0 imputation")
    parser.add_argument("--war-path", type=Path, default=WAR_PATH)
    parser.add_argument("--expanded-path", type=Path, default=EXPANDED_PATH)
    parser.add_argument("--conf-stats-path", type=Path, default=CONF_STATS_PATH)
    parser.add_argument("--conf-strength-path", type=Path, default=CONF_STRENGTH_PATH)
    parser.add_argument("--trackman-path", type=Path, default=TRACKMAN_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--verify", action="store_true", help="Print summary stats")
    args = parser.parse_args(argv)

    # Load WAR ground truth
    war_payload = load_json(args.war_path)
    war_records = war_payload.get("records", [])
    print(f"Loaded WAR ground truth: {len(war_records)} records")

    # Build person_id -> WAR lookup with imputation
    war_lookup: Dict[int, dict] = {}
    for r in war_records:
        pid = r.get("person_id")
        if pid is None:
            continue
        draft_year = r.get("draft_year")
        war_5yr = r.get("war_years_1_through_5")
        debuted = r.get("mlb_debut_season") is not None

        # Impute WAR=0 for non-debuted players drafted 2015-2022
        if war_5yr is None and not debuted and draft_year in IMPUTE_ZERO_YEARS:
            war_5yr = 0.0

        war_lookup[pid] = {
            "war_years_1_through_5": war_5yr,
            "signed_log_war": signed_log_war(war_5yr) if war_5yr is not None else None,
            "mlb_debut_season": r.get("mlb_debut_season"),
            "career_war_to_date": r.get("career_war_to_date"),
            "imputed": (war_5yr == 0.0 and not debuted and draft_year in IMPUTE_ZERO_YEARS),
            "debuted": debuted,
        }

    imputed_count = sum(1 for v in war_lookup.values() if v.get("imputed"))
    print(f"WAR lookup: {len(war_lookup)} players, {imputed_count} imputed WAR=0")

    # Load expanded training set (college stats)
    expanded = load_json(args.expanded_path)
    print(f"Loaded expanded training set: {len(expanded)} records")

    # Load conference stats and strength for feature engineering
    conf_stats = load_json(args.conf_stats_path)
    conf_strength = load_json(args.conf_strength_path)
    print(f"Loaded conference stats and strength")

    # Load TrackMan features if available
    trackman_data = None
    if args.trackman_path.exists():
        trackman_data = load_json(args.trackman_path)
        print(f"Loaded TrackMan features: {len(trackman_data.get('pitchers', {}))} pitchers, {len(trackman_data.get('hitters', {}))} hitters")
    else:
        print(f"TrackMan features not found at {args.trackman_path}, skipping")

    # Feature engineering: add conference-adjusted features
    print("Computing conference-adjusted features...")
    expanded = add_features(expanded, conf_stats, conf_strength)

    # Merge TrackMan features if available
    if trackman_data:
        print("Merging TrackMan features...")
        expanded, tm_joined = merge_trackman_features(expanded, trackman_data)
        print(f"  Joined {tm_joined}/{len(expanded)} records with TrackMan data")
    else:
        # Initialize TrackMan fields as None
        for rec in expanded:
            rec["avg_ev_wood_adj"] = None
            rec["avg_velo"] = None

    # Set has_mlb_debut flag from WAR data
    for rec in expanded:
        pid = rec.get("person_id")
        war_info = war_lookup.get(pid)
        rec["has_mlb_debut"] = war_info["debuted"] if war_info else 0

    # Get latest record per player (most recent season)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in expanded:
        pid = r.get("person_id") or hash(r.get("player_name", "")) % (10 ** 10)
        groups[pid].append(r)
    all_players = []
    for pid, recs in groups.items():
        all_players.append(max(recs, key=lambda x: x.get("season", 0) or 0))
    print(f"Unique players (latest season): {len(all_players)}")

    # Compute NN MLB rates by role
    print("Computing nearest-neighbor MLB debut rates...")
    hitter_sim_stats = ["wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj", "conf_strength", "Age", "height_inches", "bmi"]
    pitcher_sim_stats = ["ERA_adj", "FIP_adj", "K_per_nine_adj", "BB_per_nine_adj", "conf_strength", "Age", "height_inches", "bmi"]

    hitters = [p for p in all_players if p.get("player_type") == "hitter"]
    pitchers = [p for p in all_players if p.get("player_type") == "pitcher"]

    hitter_nn_rates = compute_nn_mlb_rates(hitters, hitter_sim_stats)
    pitcher_nn_rates = compute_nn_mlb_rates(pitchers, pitcher_sim_stats)

    # Map NN rates back to all_players by index
    hitter_idx = 0
    pitcher_idx = 0
    for p in all_players:
        if p.get("player_type") == "hitter":
            p["nn_mlb_rate"] = hitter_nn_rates[hitter_idx] if hitter_idx < len(hitter_nn_rates) else 0.0
            hitter_idx += 1
        else:
            p["nn_mlb_rate"] = pitcher_nn_rates[pitcher_idx] if pitcher_idx < len(pitcher_nn_rates) else 0.0
            pitcher_idx += 1

    # Build round rates and assign logit priors
    print("Computing round logit priors...")
    round_rates = build_round_rates(all_players)
    assign_round_logit_prior(all_players, round_rates)

    # Rebuild expanded dict from all_players (now with features)
    expanded_with_features = {p.get("person_id"): p for p in all_players}

    # Join WAR onto expanded training set
    training_records = []
    inference_records = []

    for rec in expanded:
        pid = rec.get("person_id")
        if pid is None:
            continue
        draft_year = rec.get("draft_year")
        if draft_year is None:
            continue

        # Only include 2021-2025 draft classes
        if draft_year < 2021 or draft_year > 2025:
            continue

        war_info = war_lookup.get(pid)

        # For 2021-2023: must have WAR data (actual or imputed)
        if draft_year <= 2023:
            if war_info is None:
                continue  # No WAR data at all
            if war_info["war_years_1_through_5"] is None:
                continue  # WAR still null (shouldn't happen with imputation)

            # Assign split based on draft year
            if draft_year in TRAIN_YEARS:
                split = "train"
            elif draft_year in HELDOUT_YEARS:
                split = "heldout"
            else:
                continue  # Skip years not in split

            # Merge WAR fields into record (use feature-enriched version)
            merged = dict(expanded_with_features.get(pid, rec))
            merged["war_years_1_through_5"] = war_info["war_years_1_through_5"]
            merged["signed_log_war"] = war_info["signed_log_war"]
            merged["mlb_debut_season"] = war_info["mlb_debut_season"]
            merged["war_imputed"] = war_info["imputed"]
            merged["split"] = split
            training_records.append(merged)

        # For 2024-2025: inference-only (no WAR data yet)
        else:
            merged = dict(expanded_with_features.get(pid, rec))
            merged["war_years_1_through_5"] = None
            merged["signed_log_war"] = None
            merged["mlb_debut_season"] = None
            merged["war_imputed"] = False
            merged["split"] = "inference"
            inference_records.append(merged)

    print(f"Training set after join: {len(training_records)} records")
    print(f"Inference set (2024-2025): {len(inference_records)} records")

    # Summary stats
    from collections import Counter
    year_counts = Counter(r["draft_year"] for r in training_records)
    split_counts = Counter(r["split"] for r in training_records)
    type_counts = Counter(r.get("player_type", "unknown") for r in training_records)
    imputed_by_split = Counter(
        r["split"] for r in training_records if r.get("war_imputed")
    )

    print("\n=== Training Set Summary ===")
    print(f"Total records: {len(training_records)}")
    print(f"By draft year: {dict(sorted(year_counts.items()))}")
    print(f"By split: {dict(split_counts)}")
    print(f"By type: {dict(type_counts)}")
    print(f"Imputed WAR=0 by split: {dict(imputed_by_split)}")

    # Save output
    write_json(args.out, {
        "generated_at": "2026-08-21",
        "imputation_years": list(IMPUTE_ZERO_YEARS),
        "temporal_split": {
            "train": list(TRAIN_YEARS),
            "heldout": list(HELDOUT_YEARS),
        },
        "n_records": len(training_records),
        "n_inference": len(inference_records),
        "n_imputed": imputed_count,
        "records": training_records,
        "inference_records": inference_records,
    })
    print(f"\nWrote {args.out}")

    if args.verify:
        # Print WAR distribution
        wars = [r["signed_log_war"] for r in training_records if r["signed_log_war"] is not None]
        if wars:
            wars_sorted = sorted(wars)
            n = len(wars_sorted)
            print(f"\n=== WAR Distribution ===")
            print(f"N: {n}")
            print(f"Min: {min(wars):.3f}")
            print(f"25th: {wars_sorted[n//4]:.3f}")
            print(f"Median: {wars_sorted[n//2]:.3f}")
            print(f"75th: {wars_sorted[3*n//4]:.3f}")
            print(f"Max: {max(wars):.3f}")
            print(f"Mean: {sum(wars)/n:.3f}")
            positive = sum(1 for w in wars if w > 0)
            print(f"Positive WAR: {positive}/{n} ({100*positive/n:.1f}%)")

        # Print feature coverage
        print(f"\n=== Feature Coverage ===")
        adj_features = ["wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj", "conf_strength", "round_logit_prior", "nn_mlb_rate"]
        for feat in adj_features:
            non_null = sum(1 for r in training_records if r.get(feat) is not None)
            print(f"  {feat}: {non_null}/{len(training_records)} ({100*non_null/len(training_records):.0f}%)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
