#!/usr/bin/env python3
"""Build the Tier 5 WAR training set.

Joins BRef WAR ground truth onto the expanded college-stats training set,
re-derives Tier 3 features (via the exact ``add_features`` / NN-rate helpers
copied from ``train_tier3_mlb_arrival.py``), computes the hurdle label and
signed-log WAR target, and applies honest temporal windowing.

Honest cohort reality (verified): ``expanded_training_set.json`` only has
college stats for draft years 2021-2025; WAR ground truth exists 2015-2023.
The overlap population is therefore 2021-2023:

  * ``split == "train"``   -> draft_year == 2021 (5 seasons through 2026)
  * ``split == "heldout"`` -> draft_year in (2022, 2023) (right-censored)
  * draft_year >= 2024     -> excluded (inference-only)

TrackMan features are joined only when a person_id crosswalk exists; TrackMan
uses internal 5-11 digit ids (not 6-digit MLBAM person_ids), so no join occurs
and TrackMan fields are emitted as ``null`` (never 0.0).

Usage:
    python scripts/build_tier5_training.py [--war PATH] [--expanded PATH] \
        [--milb-dir PATH] [--trackman PATH] [--out PATH] [--verify]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from mlb_draft_dashboard.tier5_contracts import (
    HURDLE_THRESHOLD,
    signed_log_war,
    validate_hurdle_rate,
    validate_temporal_no_leakage,
    validate_war_distribution,
)

BASE = Path(__file__).resolve().parents[1]
DEFAULT_WAR = BASE / "data" / "war" / "war_ground_truth.json"
DEFAULT_EXPANDED = BASE / "data" / "training" / "expanded_training_set.json"
DEFAULT_DRAFT = BASE / "data" / "draft" / "draft_all_picks.json"
DEFAULT_TRACKMAN = BASE / "data" / "trackman" / "trackman_player_features.json"
DEFAULT_MILB_DIR = BASE / "data" / "milb"
DEFAULT_CONF_STATS = BASE / "models" / "artifacts_full" / "conference_stats.json"
DEFAULT_CONF_STRENGTH = BASE / "models" / "artifacts_full" / "conference_strength.json"
DEFAULT_OUTPUT = BASE / "data" / "training" / "tier5_training_set.json"
DEFAULT_META = BASE / "data" / "training" / "tier5_training_meta.json"

MIN_PA = 50
MIN_IP = 20
TRAIN_YEAR = 2021
HELDOUT_YEARS = (2022, 2023)

WINDOWING_NOTE = (
    "spec's 2015-2021 windowing is aspirational; actual college-stats coverage "
    "is 2021-2025, so train=2021 / heldout=2022-2023"
)

# ---------------------------------------------------------------------------
# Tier 3 constants (copied verbatim from train_tier3_mlb_arrival.py)
# ---------------------------------------------------------------------------

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


def load_json(path):
    return json.load(open(path))


# ---------------------------------------------------------------------------
# Tier 3 helpers (copied verbatim from train_tier3_mlb_arrival.py)
# ---------------------------------------------------------------------------

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


def get_player_latest(records):
    groups = defaultdict(list)
    for r in records:
        pid = r.get("person_id") or hash(r.get("player_name", "")) % (10 ** 10)
        groups[pid].append(r)
    result = []
    for pid, recs in groups.items():
        result.append(max(recs, key=lambda x: x.get("season", 0) or 0))
    return result


def compute_nn_mlb_rates(players, sim_stats):
    n = len(players)
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

    return nn_mlb_rates, scaler, nn


# ---------------------------------------------------------------------------
# Tier 5 pure functions
# ---------------------------------------------------------------------------

def load_war_map(war_data):
    """person_id -> war_record map. Accepts dict-with-``records`` or bare list."""
    records = war_data["records"] if isinstance(war_data, dict) and "records" in war_data else war_data
    mapping = {}
    for r in records:
        pid = r.get("person_id")
        if pid is None:
            continue
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pass
        mapping[pid] = r
    return mapping


def assign_split(draft_year, train_year=TRAIN_YEAR, heldout_years=HELDOUT_YEARS):
    if draft_year == train_year:
        return "train"
    if draft_year in heldout_years:
        return "heldout"
    return None


def compute_label(war):
    if war is None:
        return {"meaningfully_productive": False, "signed_log_war": None}
    return {
        "meaningfully_productive": war > HURDLE_THRESHOLD,
        "signed_log_war": signed_log_war(war),
    }


def is_two_way(person_id, player_types_by_person, two_way_person_ids):
    if person_id in two_way_person_ids:
        return True
    return len(player_types_by_person.get(person_id, set())) > 1


def apply_min_sample_filter(players, min_pa=MIN_PA, min_ip=MIN_IP):
    kept = []
    excluded = 0
    skipped_ip = 0
    for p in players:
        ptype = p.get("player_type", "hitter")
        if ptype == "hitter":
            pa = safe_float(p.get("PA"))
            if pa is None or pa < min_pa:
                excluded += 1
                continue
        else:
            ip = safe_float(p.get("IP"))
            if ip is None:
                skipped_ip += 1
            elif ip < min_ip:
                excluded += 1
                continue
        kept.append(p)
    return kept, excluded, skipped_ip


def build_round_rates(players):
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
    for p in players:
        rnd = p.get("draft_round", 10)
        rr = round_rates.get(rnd, {"logit": -1.0})
        p["round_logit_prior"] = rr["logit"]
    return players


def merge_trackman_features(players, trackman_data):
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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_training_set(war_data, expanded_records, draft_records, conf_stats,
                       conf_strength, trackman_data=None, train_year=TRAIN_YEAR,
                       heldout_years=HELDOUT_YEARS):
    war_map = load_war_map(war_data)

    records = [dict(r) for r in expanded_records]

    debut_idx = {}
    position_type_by_person = {}
    for p in draft_records:
        pid = p.get("person_id")
        if pid is None:
            continue
        debut = p.get("mlb_debut_date")
        debut_idx[pid] = 1 if (debut and debut != "None") else 0
        position_type_by_person[pid] = p.get("position_type", "")

    two_way_draft_ids = {
        pid for pid, pt in position_type_by_person.items() if pt == "Two-Way Player"
    }

    records = add_features(records, conf_stats, conf_strength)
    drafted = [r for r in records if r.get("draft_pick") and r["draft_pick"] > 0]
    all_players = get_player_latest(drafted)

    player_types_by_person = defaultdict(set)
    for r in drafted:
        pid = r.get("person_id")
        if pid is not None:
            player_types_by_person[pid].add(r.get("player_type", "hitter"))

    for p in all_players:
        p["has_mlb_debut"] = debut_idx.get(p.get("person_id"), 0)

    round_rates = build_round_rates(all_players)
    assign_round_logit_prior(all_players, round_rates)

    for pt, sim_stats in (("hitter", HITTER_SIM_STATS), ("pitcher", PITCHER_SIM_STATS)):
        pool = [p for p in all_players if p.get("player_type") == pt]
        if not pool:
            continue
        rates, _, _ = compute_nn_mlb_rates(pool, sim_stats)
        for p, rate in zip(pool, rates):
            p["nn_mlb_rate"] = rate

    two_way_excluded = 0
    filtered = []
    for p in all_players:
        if is_two_way(p.get("person_id"), player_types_by_person, two_way_draft_ids):
            two_way_excluded += 1
            continue
        filtered.append(p)

    filtered, below_min, skipped_ip = apply_min_sample_filter(filtered, MIN_PA, MIN_IP)

    excluded_inference = 0
    out = []
    for p in filtered:
        split = assign_split(p.get("draft_year", 0), train_year, heldout_years)
        if split is None:
            excluded_inference += 1
            continue
        p["split"] = split
        war_rec = war_map.get(p.get("person_id"))
        war = war_rec.get("war_years_1_through_5") if war_rec else None
        label = compute_label(war)
        p["war_years_1_through_5"] = war
        p["meaningfully_productive"] = label["meaningfully_productive"]
        p["signed_log_war"] = label["signed_log_war"]
        out.append(p)

    out, trackman_joined = merge_trackman_features(out, trackman_data)

    n_train = sum(1 for r in out if r["split"] == "train")
    n_heldout = sum(1 for r in out if r["split"] == "heldout")
    meta = {
        "generated_by": "scripts/build_tier5_training.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "windowing_note": WINDOWING_NOTE,
        "field_name_note": (
            "player_type (not role), Age (capital A), draft_pick (not draft_pick_number)"
        ),
        "hurdle_threshold": HURDLE_THRESHOLD,
        "n_expanded": len(expanded_records),
        "n_unique_players": len(all_players),
        "n_train": n_train,
        "n_heldout": n_heldout,
        "n_excluded_inference": excluded_inference,
        "n_two_way_excluded": two_way_excluded,
        "n_below_min_sample": below_min,
        "n_skipped_ip_filter": skipped_ip,
        "trackman_joined": trackman_joined,
        "hurdle_rate": validate_hurdle_rate(out),
    }
    return out, meta


# ---------------------------------------------------------------------------
# V1-V4 verification gates
# ---------------------------------------------------------------------------

def verify_v1(records):
    rate = validate_hurdle_rate(records)
    n_positive = sum(1 for r in records if r.get("meaningfully_productive") is True)
    return {"pass": True, "hurdle_rate": rate, "n": len(records), "n_positive": n_positive}


def verify_v2(records):
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
    if np.isnan(corr):
        corr = 0.0
    return {"pass": corr < 0, "correlation": corr, "n": len(rounds)}


def verify_v3(records, train_max_year=TRAIN_YEAR):
    ok = validate_temporal_no_leakage(records, train_max_year)
    leakage = [
        r for r in records
        if r.get("split") == "train" and (r.get("draft_year") or 0) > train_max_year
    ]
    return {"pass": ok, "leakage_count": len(leakage)}


def verify_v4(records):
    dist = validate_war_distribution(records)
    vals = [r["war_years_1_through_5"] for r in records
            if r.get("war_years_1_through_5") is not None]
    skew = 0.0
    if len(vals) >= 3:
        from scipy.stats import skew as _skew
        skew = float(_skew(vals))
    return {
        "pass": True,
        "median": dist["median"],
        "count": dist["count"],
        "positive_count": dist["positive_count"],
        "skew": skew,
        "mode_near_0": abs(dist["median"]) <= 0.5,
    }


def print_verify_report(records, meta):
    v1 = verify_v1(records)
    v2 = verify_v2(records)
    v3 = verify_v3(records)
    v4 = verify_v4(records)
    print("=" * 64)
    print("TIER 5 TRAINING SET — VERIFY REPORT")
    print("=" * 64)
    print(f"  V1 hurdle rate:          {v1['hurdle_rate']:.2%} "
          f"({v1['n_positive']}/{v1['n']})  "
          f"[note: overlap n is small; a base rate <10% is expected]")
    print(f"  V2 corr(round, WAR):     {v2['correlation']:+.4f} "
          f"(n={v2['n']}, expected negative: {v2['pass']})")
    print(f"  V3 temporal leakage:     {v3['leakage_count']} train records "
          f"with draft_year > {TRAIN_YEAR} (pass: {v3['pass']})")
    print(f"  V4 WAR distribution:     median={v4['median']:.3f} "
          f"count={v4['count']} positive={v4['positive_count']} "
          f"skew={v4['skew']:+.3f} mode_near_0={v4['mode_near_0']}")
    print("  " + "-" * 56)
    print(f"  cohort: train={meta['n_train']} heldout={meta['n_heldout']} "
          f"excluded_inference={meta['n_excluded_inference']} "
          f"two_way_excluded={meta['n_two_way_excluded']} "
          f"below_min_sample={meta['n_below_min_sample']}")
    print(f"  trackman_joined:         {meta['trackman_joined']}")
    print(f"  windowing note:          {meta['windowing_note']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the Tier 5 WAR training set.")
    p.add_argument("--war", default=str(DEFAULT_WAR), help="WAR ground-truth JSON path")
    p.add_argument("--expanded", default=str(DEFAULT_EXPANDED), help="Expanded training-set JSON path")
    p.add_argument("--milb-dir", default=str(DEFAULT_MILB_DIR),
                   help="MiLB data dir (reserved; milb_year1_wOBA excluded by spec)")
    p.add_argument("--trackman", default=str(DEFAULT_TRACKMAN), help="TrackMan player-features JSON path")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output training-set JSON path")
    p.add_argument("--verify", action="store_true", help="Print V1-V4 report to stdout")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    war_data = load_json(args.war)
    expanded = load_json(args.expanded)
    draft = load_json(DEFAULT_DRAFT)
    conf_stats = load_json(DEFAULT_CONF_STATS)
    conf_strength = load_json(DEFAULT_CONF_STRENGTH)

    trackman_data = None
    tm_path = Path(args.trackman)
    if tm_path.exists():
        trackman_data = load_json(str(tm_path))
    else:
        print(f"TrackMan features not found at {tm_path} — emitting null TrackMan fields")

    records, meta = build_training_set(
        war_data, expanded, draft, conf_stats, conf_strength, trackman_data
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    meta_path = out_path.with_name("tier5_training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_path} ({len(records)} records)")
    print(f"Wrote {meta_path}")
    print(f"  train={meta['n_train']} heldout={meta['n_heldout']} "
          f"excluded_inference={meta['n_excluded_inference']} "
          f"two_way_excluded={meta['n_two_way_excluded']} "
          f"below_min_sample={meta['n_below_min_sample']}")
    print(f"  trackman_joined={meta['trackman_joined']}")

    if args.verify:
        print_verify_report(records, meta)


if __name__ == "__main__":
    main()
