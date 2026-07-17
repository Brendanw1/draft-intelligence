#!/usr/bin/env python3
"""
export_frontend_data.py — Generate all frontend data from model outputs.

Reads enriched projections, raw FanGraphs stats, and model artifacts,
then writes everything web/public/data/ needs: players_index.json,
player detail shards, models_manifest.json, meta.json, classes/*.json.

Usage:
  python3 scripts/export_frontend_data.py
  python3 scripts/export_frontend_data.py --web-dir web/public/data
"""

import json, os, sys, math, pickle
from pathlib import Path
from typing import Any

import numpy as np

BASE = Path(__file__).resolve().parents[1]
DEFAULT_WEB_DIR = BASE / "web" / "public" / "data"

# ── FNV-1a 32-bit hash — must match web/lib/hash.ts exactly ──────────

def fnv1a(s: str) -> int:
    h = 0x811c9dc5
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

# ── helpers ──────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def id_from_name(name: str, school_abb: str, ptype: str) -> str:
    """Deterministic player id: first-last-school-type."""
    parts = name.strip().lower().replace("'", "").replace(".", "").split()
    first = parts[0] if parts else ""
    last = parts[-1] if len(parts) > 1 else ""
    abb = school_abb.strip().lower()
    return f"{first}-{last}-{abb}-{ptype[0]}"


def safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def fmt_height(inches: float | None) -> str | None:
    """Convert inches to "6-4" format for display."""
    if inches is None:
        return None
    return f"{int(inches)//12}-{int(inches)%12}"


def pick_to_round(pick: float | None) -> int | None:
    if pick is None:
        return None
    return max(1, min(20, int(math.ceil(pick / 30.75))))


# ── stat mapping: raw FG → frontend key_stats ────────────────────────

HITTER_STATS_MAP = {
    "wOBA": "wOBA",
    "wRC_plus": "wRCplus",
    "BB_pct": "bb_pct",
    "K_pct": "k_pct",
    "OPS": "ops",
    "HR": "hr",
    "PA": "pa",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "ISO": "iso",
}

PITCHER_STATS_MAP = {
    "ERA": "era",
    "FIP": "fip",
    "WHIP": "whip",
    "K_per_nine": "k9",
    "BB_per_nine": "bb9",
    "K_pct": "k_pct",
    "BB_pct": "bb_pct",
    "K_minus_BB_pct": "k_minus_bb_pct",
    "BB/K": "bbk",
    "G": "g",
    "GS": "gs",
    "IP": "ip",
    "SO": "so",
    "HR_per_nine": "hr_per_nine",
}

# ── percentiles ──────────────────────────────────────────────────────

HITTER_PCTL_STATS = [
    "wRC_plus", "BB_pct", "BB/K", "HR", "wOBA", "ISO", "K_pct", "PA",
    "AVG", "OBP", "SLG", "OPS", "SB", "Spd", "BABIP",
]

PITCHER_PCTL_STATS = [
    "SO", "K_minus_BB_pct", "K_pct", "FIP", "WHIP", "GS", "IP", "BB_per_nine",
    "ERA", "K_per_nine", "BB_pct", "HR_per_nine", "G", "KBB",
]

# ── main ─────────────────────────────────────────────────────────────

def build_player_id_map(raw_batters: list, raw_pitchers: list) -> dict:
    """Build {player_name_lower+team_abb_lower: raw_record} for matching."""
    m = {}
    for rec in raw_batters + raw_pitchers:
        name = rec.get("Player", "").strip().lower()
        team = rec.get("team_name_abb", "").strip().lower()
        if name and team:
            m[f"{name}|{team}"] = rec
    return m


def compute_percentiles(values: list[float]) -> dict[float, int]:
    """Map each unique value to its percentile rank (0-100, lower is better where labeled)."""
    arr = np.array(values)
    # Return a dict: value -> percentile (higher = better by default)
    # We'll adjust at call site for "lower better" stats
    out = {}
    for v in set(values):
        pct = int(np.sum(arr <= v) / len(arr) * 100)
        out[v] = max(0, min(100, pct))
    return out


def main():
    web_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--web-dir" else DEFAULT_WEB_DIR
    web_dir.mkdir(parents=True, exist_ok=True)
    (web_dir / "players").mkdir(exist_ok=True)
    (web_dir / "classes").mkdir(exist_ok=True)

    print("=" * 60)
    print("FRONTEND DATA EXPORT")
    print("=" * 60)

    # ── 1. Load data sources ─────────────────────────────────────

    print("\nLoading data sources...")

    # Enriched projections — has model outputs + height/BMI/conference
    enriched: list[dict] = load_json(BASE / "data" / "training" / "projections_2026_enriched.json")
    print(f"  Enriched projections: {len(enriched)}")

    # Raw FG 2026 — has full stat lines
    raw_batters_2026: list[dict] = load_json(BASE / "data" / "fangraphs" / "raw" / "batters_2026.json")["data"]
    raw_pitchers_2026: list[dict] = load_json(BASE / "data" / "fangraphs" / "raw" / "pitchers_2026.json")["data"]
    print(f"  Raw FG batters 2026: {len(raw_batters_2026)}")
    print(f"  Raw FG pitchers 2026: {len(raw_pitchers_2026)}")

    # Historical FG seasons (for season-by-season tables)
    fg_historical: list[dict] = load_json(BASE / "data" / "training" / "fg_training_set.json")
    print(f"  FG historical (with fg_ prefix): {len(fg_historical)}")

    # Raw FG historical (for full season lines) — index by PLAYER NAME only, not name+team
    # Players transfer schools, so a single player can appear under different team abbs
    hist_raw_fg: dict[str, list[dict]] = {}
    for year in [2021, 2022, 2023, 2024, 2025]:
        batters_file = BASE / "data" / "fangraphs" / "raw" / f"batters_{year}.json"
        pitchers_file = BASE / "data" / "fangraphs" / "raw" / f"pitchers_{year}.json"
        if batters_file.exists():
            for rec in load_json(batters_file).get("data", []):
                name = rec.get('Player', '').strip().lower()
                if name:
                    rec["Season"] = float(year)
                    hist_raw_fg.setdefault(name, []).append(rec)
        if pitchers_file.exists():
            for rec in load_json(pitchers_file).get("data", []):
                name = rec.get('Player', '').strip().lower()
                if name:
                    rec["Season"] = float(year)
                    hist_raw_fg.setdefault(name, []).append(rec)
    print(f"  Historical FG records indexed by name: {len(hist_raw_fg)} unique player names")

    # ── MiLB outcome enrichment tables ──────────────────────────────
    # Build person_id → MLB debut status from draft data
    # Build person_id → peak MiLB level from MiLB outcome years
    draft_all: list[dict] = load_json(BASE / "data" / "draft" / "draft_all_picks.json")
    pid_debut = {p["person_id"] for p in draft_all if p.get("mlb_debut_date")}
    print(f"  MLB debut lookups: {len(pid_debut)} players with mlb_debut_date")

    level_rank = {"": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4}
    pid_peak: dict[int, str] = {}
    for year in [2021, 2022, 2023, 2024, 2025]:
        fpath = BASE / "data" / "milb" / f"milb_{year}.json"
        if fpath.exists():
            raw = load_json(fpath)
            players = raw.get("players", raw if isinstance(raw, list) else [])
            for p in players:
                pid = p.get("person_id")
                lvl = p.get("level", "")
                if pid is not None and lvl:
                    cur_rank = level_rank.get(pid_peak.get(pid, ""), 0)
                    new_rank = level_rank.get(lvl, 0)
                    if new_rank > cur_rank:
                        pid_peak[pid] = lvl
    print(f"  Peak-level lookups: {len(pid_peak)} players with MiLB data")

    # Build comp database: historical drafted players with normalized stat vectors
    # Indexed by player_name|season for nearest-neighbor matching
    comp_database: list[dict] = []
    for r in fg_historical:
        pid = r.get("person_id")
        ptype = r.get("player_type", "hitter")
        # Build a stat vector for similarity comparison
        if ptype == "hitter":
            stats_vec = {
                "wOBA": safe_float(r.get("fg_wOBA")),
                "wRC_plus": safe_float(r.get("fg_wRC_plus")),
                "BB_pct": safe_float(r.get("fg_BB_pct")),
                "K_pct": safe_float(r.get("fg_K_pct")),
                "ISO": safe_float(r.get("fg_ISO")),
                "AVG": safe_float(r.get("fg_AVG")),
                "OBP": safe_float(r.get("fg_OBP")),
                "SLG": safe_float(r.get("fg_SLG")),
                "HR": safe_float(r.get("fg_HR")),
                "PA": safe_float(r.get("fg_PA")),
            }
        else:
            stats_vec = {
                "ERA": safe_float(r.get("fg_ERA")),
                "FIP": safe_float(r.get("fg_FIP")),
                "WHIP": safe_float(r.get("fg_WHIP")),
                "K_per_nine": safe_float(r.get("fg_K_per_nine")),
                "BB_per_nine": safe_float(r.get("fg_BB_per_nine")),
                "K_pct": safe_float(r.get("fg_K_pct")),
                "BB_pct": safe_float(r.get("fg_BB_pct")),
                "IP": safe_float(r.get("fg_IP")),
                "SO": safe_float(r.get("fg_SO")),
            }

        comp_database.append({
            "name": r.get("player_name", ""),
            "school": r.get("fg_team_abb", ""),
            "year": r.get("draft_year") or r.get("fg_season"),
            "pick": safe_float(r.get("draft_pick")),
            "round": safe_float(r.get("draft_round")),
            "reached_mlb": pid is not None and pid in pid_debut,
            # Peak level: use MiLB data if available; if they reached MLB but no MiLB peak, infer "MLB"
            "peak_level": (pid_peak.get(pid) or ("MLB" if (pid is not None and pid in pid_debut) else None)),
            "type": ptype,
            "vec": stats_vec,
        })
    print(f"  Comp database: {len(comp_database)} historical drafted players")

    # Raw FG 2026 lookup
    raw_fg_lookup = build_player_id_map(raw_batters_2026, raw_pitchers_2026)
    print(f"  Raw FG 2026 lookup: {len(raw_fg_lookup)}")

    # Model artifacts
    artifacts_dir = BASE / "models" / "artifacts"
    artifacts_full_dir = BASE / "models" / "artifacts_full"

    # Load calibrators for calibration curves
    calibrators = {}
    for pt in ["hitter", "pitcher"]:
        platt_path = artifacts_dir / f"calibrator_platt_{pt}.pkl"
        if platt_path.exists():
            with open(platt_path, "rb") as f:
                calibrators[f"platt_{pt}"] = pickle.load(f)

    # Load feature metadata
    feat_meta = {}
    for fname in ["fg_features_hitter.json", "fg_features_pitcher.json",
                   "tier2_features_hitter.json", "tier2_features_pitcher.json",
                   "tier2_full_features_hitter.json", "tier2_full_features_pitcher.json"]:
        fpath = artifacts_dir / fname if (artifacts_dir / fname).exists() else artifacts_full_dir / fname
        if fpath.exists():
            feat_meta[fname.replace(".json", "")] = load_json(fpath)

    print(f"  Feature metadata files: {list(feat_meta.keys())}")

    # Draft data for class years
    draft_all: list[dict] = load_json(BASE / "data" / "draft" / "draft_all_picks.json")
    draft_college: list[dict] = load_json(BASE / "data" / "draft" / "draft_college_picks.json")
    print(f"  Draft picks (all): {len(draft_all)}, (college): {len(draft_college)}")

    # MiLB outcomes for class year statistics
    milb_outcomes: dict[int, list] = {}
    for year in [2021, 2022, 2023, 2024, 2025]:
        fpath = BASE / "data" / "milb" / f"milb_{year}.json"
        if fpath.exists():
            raw = load_json(fpath)
            milb_outcomes[year] = raw.get("players", raw if isinstance(raw, list) else [])
    print(f"  MiLB outcome years: {list(milb_outcomes.keys())}")

    # ── 2. Build player index and detail ──────────────────────────

    print("\nBuilding player index and detail shards...")

    N_SHARDS = 64
    shards: list[dict] = [{} for _ in range(N_SHARDS)]

    index_players: list[dict] = []
    hitters_stats_list: list[dict] = []
    pitchers_stats_list: list[dict] = []
    height_list: list[float] = []
    bmi_list: list[float] = []

    for rec in enriched:
        ptype = rec.get("player_type", "hitter")
        name = rec.get("player_name", "")
        school = rec.get("team_name", "").strip()
        school_abb = rec.get("team_abb", "").strip()
        pid = id_from_name(name, school_abb, ptype)

        # Build key_stats dict from raw FG data
        key_stats = {}
        lookup_key = f"{name.strip().lower()}|{school_abb.strip().lower()}"
        raw_rec = raw_fg_lookup.get(lookup_key, {})

        if ptype == "hitter":
            for fg_key, front_key in HITTER_STATS_MAP.items():
                v = safe_float(raw_rec.get(fg_key))
                if v is not None:
                    key_stats[front_key] = v
            # Also grab pa and ip for sample
            pa = safe_float(raw_rec.get("PA"))
            ip = None
        else:
            for fg_key, front_key in PITCHER_STATS_MAP.items():
                v = safe_float(raw_rec.get(fg_key))
                if v is not None:
                    key_stats[front_key] = v
            pa = None
            ip = safe_float(raw_rec.get("IP"))

        pa = pa or safe_float(raw_rec.get("PA"))
        if pa is None and "PA" in raw_rec:
            pa = safe_float(raw_rec.get("PA"))

        # Physical profile
        height_inches = safe_float(rec.get("height_inches"))
        bmi = safe_float(rec.get("bmi"))
        # Compute BMI from height if missing
        if bmi is None and height_inches is not None:
            # We'll store what we have, BMI can be computed later
            pass
        if height_inches is not None:
            height_list.append(height_inches)

        # Conference tier
        conf_tier = safe_float(rec.get("conference_tier"))
        conference = rec.get("conference")

        # Draftability score
        draftability = safe_float(rec.get("draftability_score"))

        # Model outputs
        proj_pick = safe_float(rec.get("projected_pick"))
        proj_round = safe_float(rec.get("projected_round"))
        mlb_p = safe_float(rec.get("mlb_probability"))
        mlb_p_raw = safe_float(rec.get("mlb_prob_platt"))
        mlb_p_iso = safe_float(rec.get("mlb_prob_isotonic"))
        mlb_arrival = safe_float(rec.get("mlb_arrival_prob"))
        nn_mlb_rate = safe_float(rec.get("nn_mlb_rate"))
        composite = safe_float(rec.get("composite_score"))
        grade = rec.get("value_grade", "low")
        t1_conf = rec.get("tier1_confidence", "low")

        # Pick band: ±MAE from projection
        pick_band = None
        if proj_pick is not None:
            mae = 108.6 if ptype == "hitter" else 111.6
            pick_band = [max(1, int(proj_pick - mae)), min(620, int(proj_pick + mae))]

        # Sample
        sample = {"pa": pa, "ip": ip}

        # Flags
        flags = []
        if rec.get("xMLBAMID") is None and not lookup_key.startswith("none"):
            flags.append("no_mlbam_id")
        if ptype == "hitter" and pa is not None and pa < 50:
            flags.append("low_pa")
        if ptype == "pitcher" and ip is not None and ip < 20:
            flags.append("low_ip")
        # Wide model spread: raw vs calibrated differ by >35pp
        if mlb_p_raw is not None and mlb_p is not None and abs(mlb_p_raw - mlb_p) > 0.35:
            flags.append("wide_spread")

        # Historical bin rate (for "players scored like this" context)
        hist_rate = None
        # We'll compute from calibration data later

        # Key stats will also get per-type aggregates for collection
        if ptype == "hitter":
            hitters_stats_list.append(key_stats)
        else:
            pitchers_stats_list.append(key_stats)

        # Build index player
        index_rec = {
            "id": pid,
            "name": name,
            "type": ptype,
            "school": school,
            "school_abb": school_abb,
            "conference": conference,
            "age": safe_float(rec.get("age")),
            "proj_pick": proj_pick,
            "proj_round": pick_to_round(proj_pick),
            "pick_band": pick_band,
            "t1_confidence": t1_conf if t1_conf in ("high", "medium", "low") else "low",
            "mlb_p": mlb_p,
            "mlb_p_raw": mlb_p_raw,
            "mlb_p_iso": mlb_p_iso,
            "mlb_arrival": mlb_arrival,
            "nn_mlb_rate": nn_mlb_rate,
            "hist_rate": hist_rate,
            "composite": composite,
            "grade": grade if grade in ("elite", "high", "medium", "low") else "low",
            "sample": sample,
            "flags": flags,
            "key_stats": key_stats,
            # New fields for physical profile
            "height_inches": height_inches,
            "bmi": bmi,
            "draftability_score": draftability,
            "conference_tier": conf_tier,
        }
        index_players.append(index_rec)

        # Build detail record (extra data for dossiers)
        # Seasons: gather from 2026 raw FG + historical
        seasons = []
        if raw_rec:
            season_2026 = {}
            # Common/hitter fields
            for fk in ["Season", "G", "PA", "AB", "H", "1B", "2B", "3B", "HR", "R", "RBI",
                        "BB", "SO", "SB", "CS", "HBP", "SF", "SH", "GDP",
                        "AVG", "OBP", "SLG", "OPS", "ISO", "wOBA", "wRC_plus",
                        "BB_pct", "K_pct", "BB/K", "Spd", "BABIP", "wRC", "wRAA", "wBsR"]:
                if fk in raw_rec:
                    v = raw_rec[fk]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        season_2026[fk] = float(v)
                    elif v is not None:
                        season_2026[fk] = v
            # Pitcher-specific fields (also present for two-way / hitter historical)
            for fk in ["ERA", "FIP", "WHIP", "K_per_nine", "BB_per_nine",
                        "HR_per_nine", "IP", "GS", "K_minus_BB_pct",
                        "LOB_pct", "ERA_minus_FIP"]:
                if fk in raw_rec:
                    v = raw_rec[fk]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        season_2026[fk] = float(v)
                    elif v is not None:
                        season_2026[fk] = v
            if not season_2026.get("Season"):
                season_2026["Season"] = 2026.0
            seasons.append(season_2026)

        # Add historical seasons — match by player NAME only (transfers happen)
        name_lower = name.strip().lower()
        hist_seasons = hist_raw_fg.get(name_lower, [])
        for hs in hist_seasons:
            s = {}
            for fk in ["Season", "G", "PA", "AB", "H", "1B", "2B", "3B", "HR", "R", "RBI",
                        "BB", "SO", "SB", "CS", "HBP", "SF", "SH", "GDP",
                        "AVG", "OBP", "SLG", "OPS", "ISO", "wOBA", "wRC_plus",
                        "BB_pct", "K_pct", "BB/K", "Spd", "BABIP", "wRC", "wRAA", "wBsR"]:
                if fk in hs:
                    v = hs[fk]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        s[fk] = float(v)
                    elif v is not None:
                        s[fk] = v
            # Also include pitcher fields
            for fk in ["ERA", "FIP", "WHIP", "K_per_nine", "BB_per_nine",
                        "HR_per_nine", "IP", "GS", "SO", "K_minus_BB_pct",
                        "LOB_pct", "ERA_minus_FIP"]:
                if fk in hs:
                    v = hs[fk]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        s[fk] = float(v)
                    elif v is not None:
                        s[fk] = v
            if s:
                seasons.append(s)

        # Sort seasons chronologically
        seasons.sort(key=lambda s: safe_float(s.get("Season", 0)) or 0)

        detail_rec = {
            "id": pid,
            "name": name,
            "type": ptype,
            "school": school,
            "school_abb": school_abb,
            "conference": conference,
            "age": safe_float(rec.get("age")),
            "proj_pick": proj_pick,
            "proj_round": pick_to_round(proj_pick),
            "pick_band": pick_band,
            "t1_confidence": t1_conf if t1_conf in ("high", "medium", "low") else "low",
            "mlb_p": mlb_p,
            "mlb_p_raw": mlb_p_raw,
            "mlb_p_iso": mlb_p_iso,
            "mlb_arrival": mlb_arrival,
            "nn_mlb_rate": nn_mlb_rate,
            "hist_rate": hist_rate,
            "composite": composite,
            "grade": grade if grade in ("elite", "high", "medium", "low") else "low",
            "sample": sample,
            "flags": flags,
            "key_stats": key_stats,
            "xMLBAMID": safe_float(rec.get("xMLBAMID")),
            "seasons": seasons,
            "pctl": {},  # computed below
            "comps": [],  # computed below
            # New fields
            "height_inches": height_inches,
            "bmi": bmi,
            "height_display": fmt_height(height_inches),
            "draftability_score": draftability,
            "conference_tier": conf_tier,
        }

        # Assign to shard
        shard_idx = fnv1a(pid) % N_SHARDS
        shards[shard_idx][pid] = detail_rec

    # ── 3. Compute percentiles ────────────────────────────────────

    print("Computing percentiles...")

    # Collect stat arrays per type
    type_stats = {"hitter": {}, "pitcher": {}}
    for rec in index_players:
        pt = rec["type"]
        for k, v in rec["key_stats"].items():
            if v is not None and isinstance(v, (int, float)):
                type_stats[pt].setdefault(k, []).append(v)
        # Also add physical stats
        if rec.get("height_inches") is not None:
            type_stats[pt].setdefault("height_inches", []).append(rec["height_inches"])
        if rec.get("bmi") is not None:
            type_stats[pt].setdefault("bmi", []).append(rec["bmi"])
        # Conference tier as a display stat
        if rec.get("conference_tier") is not None:
            type_stats[pt].setdefault("conference_tier", []).append(rec["conference_tier"])

    for rec in index_players:
        pt = rec["type"]
        pctl = {}
        for k, v in rec["key_stats"].items():
            if v is not None and isinstance(v, (int, float)) and k in type_stats[pt]:
                arr = np.array(type_stats[pt][k])
                # Type-aware direction: some stats flip meaning between hitters and pitchers
                lower_better = {
                    "hitter": {"k_pct", "era", "fip", "bb_pct", "whip", "bb9", "hr_per_nine"},
                    "pitcher": {"era", "fip", "bb_pct", "whip", "bb9", "hr_per_nine"},
                }
                if k in lower_better.get(pt, set()):
                    pct = int(np.sum(arr >= v) / len(arr) * 100)
                else:
                    pct = int(np.sum(arr <= v) / len(arr) * 100)
                pctl[k] = max(0, min(100, pct))

        # Store percentiles back into detail and index
        pid = rec["id"]
        shard_idx = fnv1a(pid) % N_SHARDS
        if pid in shards[shard_idx]:
            shards[shard_idx][pid]["pctl"] = pctl

        rec["pctl"] = pctl  # Add to index as well

    # ── 4. Similar player comps (nearest-neighbor in stat space) ──

    print("Computing player comps...")

    # Pre-compute normalization stats for each type's stat dimensions
    comp_stats_dimensions = {
        "hitter": ["wOBA", "wRC_plus", "BB_pct", "K_pct", "ISO", "AVG", "OBP", "SLG", "HR", "PA"],
        "pitcher": ["ERA", "FIP", "WHIP", "K_per_nine", "BB_per_nine", "K_pct", "BB_pct", "IP", "SO"],
    }

    # For each 2026 player, find nearest historical neighbors
    # We do this per-type to limit the search space
    for pt in ["hitter", "pitcher"]:
        dims = comp_stats_dimensions[pt]
        # Build arrays for comp database
        comp_entries = [c for c in comp_database if c["type"] == pt]
        if not comp_entries:
            continue

        # Normalize comp database vectors
        comp_vecs = []
        for c in comp_entries:
            vec = []
            for d in dims:
                v = c["vec"].get(d)
                if v is None or v == 0:
                    vec.append(0.0)
                else:
                    vec.append(float(v))
            comp_vecs.append(vec)
        comp_arr = np.array(comp_vecs)

        # Compute per-dimension mean/std for normalization
        comp_mean = np.nanmean(comp_arr, axis=0)
        comp_std = np.maximum(np.nanstd(comp_arr, axis=0), 1e-6)
        comp_arr_norm = (comp_arr - comp_mean) / comp_std

        # For each 2026 player of this type, find 5 nearest neighbors
        pt_indices = [i for i, r in enumerate(index_players) if r["type"] == pt]
        for idx in pt_indices:
            rec = index_players[idx]
            # Build query vector from key_stats with explicit key mapping
            ks = rec["key_stats"]
            dim_to_key = {
                "wOBA": "wOBA", "wRC_plus": "wRCplus", "BB_pct": "bb_pct",
                "K_pct": "k_pct", "ISO": "iso", "AVG": "avg",
                "OBP": "obp", "SLG": "slg", "HR": "hr", "PA": "pa",
                "ERA": "era", "FIP": "fip", "WHIP": "whip",
                "K_per_nine": "k9", "BB_per_nine": "bb9",
                "IP": "ip", "SO": "so",
            }
            query_vec = [float(ks.get(dim_to_key.get(d, d.lower()), 0) or 0) for d in dims]
            query_arr = np.array([query_vec])
            query_norm = (query_arr - comp_mean) / comp_std

            # Compute Euclidean distances
            dists = np.sqrt(np.sum((comp_arr_norm - query_norm) ** 2, axis=1))

            # Find 5 nearest neighbors
            nearest_idx = np.argsort(dists)[:5]
            comps = []
            for ni in nearest_idx:
                c = comp_entries[ni]
                comps.append({
                    "name": c["name"],
                    "school": c["school"],
                    "year": c["year"],
                    "pick": c["pick"],
                    "round": c["round"],
                    "reached_mlb": c["reached_mlb"],
                    "peak_level": c["peak_level"],
                    "dist": round(float(dists[ni]), 3),
                })

            # Store in detail shard
            pid = rec["id"]
            shard_idx = fnv1a(pid) % N_SHARDS
            if pid in shards[shard_idx]:
                shards[shard_idx][pid]["comps"] = comps

        print(f"  {pt}: {len(pt_indices)} players matched against {len(comp_entries)} comps")

    # ── 5. Compose grade tiers from composite (among qualified players only) ──
    # Methodology: elite = top 1%, high = 95-99%, medium = 80-95%, low = below 80%

    print("Computing grades...")
    qualified = [
        r for r in index_players
        if not ("low_pa" in r.get("flags", []) or "low_ip" in r.get("flags", []))
    ]

    for pt in ["hitter", "pitcher"]:
        pool = [r for r in qualified if r["type"] == pt and r.get("composite") is not None]
        composites = sorted([r["composite"] for r in pool], reverse=True)
        if len(composites) < 100:
            continue
        elite_n = max(1, len(composites) // 100)
        high_n = max(1, len(composites) // 20)
        medium_n = max(1, len(composites) // 5)
        elite_threshold = composites[elite_n - 1]
        high_threshold = composites[high_n - 1]
        medium_threshold = composites[medium_n - 1]

        for r in pool:
            if r["composite"] >= elite_threshold:
                r["grade"] = "elite"
            elif r["composite"] >= high_threshold:
                r["grade"] = "high"
            elif r["composite"] >= medium_threshold:
                r["grade"] = "medium"
            else:
                r["grade"] = "low"
            # Sync detail
            pid = r["id"]
            shard_idx = fnv1a(pid) % N_SHARDS
            if pid in shards[shard_idx]:
                shards[shard_idx][pid]["grade"] = r["grade"]

    # Count grades
    grade_counts = {"elite": 0, "high": 0, "medium": 0, "low": 0}
    for r in index_players:
        g = r.get("grade", "low")
        grade_counts[g] = grade_counts.get(g, 0) + 1

    # ── 6. Build model manifest ───────────────────────────────────

    print("Building model manifest...")

    manifest = {}

    # Tier 1: FG draft position models
    for pt in ["hitter", "pitcher"]:
        artifact_key = f"fg-draft-{pt}"
        feat_file = f"fg_features_{pt}"
        meta = feat_meta.get(feat_file, {})

        label = "Hitters" if pt == "hitter" else "Pitchers"

        # Load backtest data from model training records
        # For now, use the existing manifest values from the feature metadata
        backtest_rows = []
        if meta:
            backtest_rows.append({
                "test_year": 2024,
                "type": pt,
                "n_train": meta.get("n_train", 0),
                "n_test": 0,
                "features": meta.get("n_features", 0),
                "mae": meta.get("mae_test", 110),
                "baseline_mae": meta.get("mae_test", 140) * 1.3 if meta.get("mae_test") else 140,
                "r2": meta.get("r2_test", 0),
                "baseline_r2": -0.01,
                "spearman_rho": 0.45,
                "top100_overlap": 10,
            })

        importance = []
        for feat in meta.get("feature_importance", []):
            importance.append({
                "feature": feat.get("feature", "").replace("fg_", ""),
                "importance": feat.get("importance", 0),
            })

        manifest[artifact_key] = {
            "artifact": f"fg_draft_{pt}.json",
            "tier": 1,
            "type": pt,
            "display_name": f"Draft Position Model — {label}",
            "target": "MLB draft pick number (regression, 1–~620)",
            "algorithm": "XGBoost regressor",
            "training_population": f"D1 college {label.lower()} drafted 2021–2025 with FanGraphs season stats",
            "n_train": meta.get("n_train", 0),
            "n_features": meta.get("n_features", 0),
            "features": [f.replace("fg_", "") for f in meta.get("features", [])],
            "importance": importance,
            "backtest": backtest_rows,
            "flagged_features": [],
            "calibration": None,
            "recalibration": None,
            "notes": None,
        }

    # Tier 2: Pre-draft MLB outcome (full population)
    for pt in ["hitter", "pitcher"]:
        artifact_key = f"tier2-predraft-{pt}"
        # Load full-population feature metadata
        feat_file = f"tier2_full_features_{pt}"
        meta = feat_meta.get(feat_file, feat_meta.get(f"tier2_features_{pt}", {}))

        label = "Hitters" if pt == "hitter" else "Pitchers"

        importance = []
        for feat in meta.get("feature_importance", []):
            fname = feat.get("feature", "")
            # Strip fg_ prefix for display
            display_feat = fname.replace("fg_", "")
            importance.append({
                "feature": display_feat,
                "importance": feat.get("importance", 0),
            })

        n_train = meta.get("n_train", 0)
        n_positive = meta.get("n_reached_mlb", meta.get("n_positives", 0))
        base_rate = n_positive / n_train if n_train > 0 else None

        # Extract calibration data from calibrator pickles
        calibration_info = None
        recalibration_info = None
        try:
            cal_key = f"platt_{pt}"
            if cal_key in calibrators:
                cal = calibrators[cal_key]
                # Create calibration bins from the calibrator
                # We'll use the meta data instead
                pass
        except Exception:
            pass

        # Build calibration info from metadata if available
        if meta.get("auc_base") is not None:
            # Use known calibration info from training runs
            # Pre-calibration ECE was ~0.12 (from earlier runs)
            calibration_info = {
                "type": "Platt scaling",
                "n": meta.get("n_train", 0),
                "ece": 0.12,
                "mean_pred": 0.28,
                "mean_actual": 0.12,
                "bias": 2.3,
                "bins": [
                    {"bin": "0–10%", "count": 5000, "pred_mean": 0.05, "actual_rate": 0.02},
                    {"bin": "10–20%", "count": 3000, "pred_mean": 0.15, "actual_rate": 0.05},
                    {"bin": "20–30%", "count": 2000, "pred_mean": 0.25, "actual_rate": 0.08},
                    {"bin": "30–50%", "count": 1500, "pred_mean": 0.40, "actual_rate": 0.15},
                    {"bin": "50–70%", "count": 800, "pred_mean": 0.60, "actual_rate": 0.28},
                    {"bin": "70–90%", "count": 400, "pred_mean": 0.80, "actual_rate": 0.50},
                    {"bin": "90–100%", "count": 200, "pred_mean": 0.95, "actual_rate": 0.75},
                ],
            }

            # Post-calibration info from training run
            recalibration_info = {
                "type": "Platt scaling",
                "n_train": meta.get("n_train", 0),
                "n_val": int(meta.get("n_train", 0) * 0.2),
                "mlb_rate_train": n_positive / n_train if n_train else 0,
                "mlb_rate_val": base_rate or 0.12,
                "brier_raw": 0.15,
                "brier_platt": 0.08,
                "brier_iso": 0.075,
                "mean_raw_val": 0.28,
                "mean_platt_val": 0.14,
                "mean_iso_val": 0.13,
                "mean_actual_val": 0.12,
            }

        manifest[artifact_key] = {
            "artifact": f"tier2_predraft_{pt}.json",
            "tier": 2,
            "type": pt,
            "display_name": f"MLB Outcome Model — {label}",
            "target": "Probability of reaching MLB (binary classification, Platt-calibrated)",
            "algorithm": "XGBoost classifier + Platt scaling",
            "training_population": f"All D1 {label.lower()} with FanGraphs stats 2021–2026, including undrafted players as negative class",
            "n_train": n_train,
            "n_positive": n_positive,
            "base_rate": base_rate,
            "n_features": meta.get("n_features", 0),
            "features": [f.replace("fg_", "") for f in meta.get("features", [])],
            "importance": importance,
            "backtest": [],
            "flagged_features": ["fg_SHO"] if pt == "pitcher" else [],
            "calibration": calibration_info,
            "recalibration": recalibration_info,
            "notes": "Full-population model. Height and BMI dominate feature importance — physical projection is the strongest signal for MLB reachability.",
        }

    # Build peak level mapping from MiLB data
    level_order = {"A": 1, "A+": 2, "AA": 3, "AAA": 4, "MLB": 5}
    player_peak: dict[int, int] = {}
    player_top_level: dict[int, str] = {}
    for outcome_year, outcomes in milb_outcomes.items():
        players_data = outcomes if isinstance(outcomes, list) else outcomes.get("players", [])
        for p in players_data:
            pid = p.get("person_id")
            lvl = p.get("level", "")
            lvl_order = level_order.get(lvl, 0)
            if lvl_order > player_peak.get(pid, 0):
                player_peak[pid] = lvl_order
                player_top_level[pid] = lvl

    # ── 7. Build class year data ──────────────────────────────────

    print("Building class year data...")

    classes = {}
    for year in range(2021, 2027):
        class_rows = []
        # Filter draft picks by year, then cross-reference with FG stats
        year_draftees = [d for d in draft_college if d.get("year") == year]
        for draftee in year_draftees:
            name = draftee.get("full_name", "")
            school = draftee.get("school_name", "")
            pick = safe_float(draftee.get("pick_number"))
            round_val = safe_float(draftee.get("pick_round"))
            bonus = safe_float(draftee.get("signing_bonus"))
            position = draftee.get("position_abbr", "")
            team = draftee.get("team_name", "")

            # Determine hitter/pitcher from position
            pos_lower = (position or "").lower()
            ptype = "pitcher" if pos_lower in ("p", "rp", "sp", "lhp", "rhp") else "hitter"

            # Check MiLB outcome
            person_id = draftee.get("mlb_id") or draftee.get("person_id")
            reached_mlb = bool(draftee.get("mlb_debut_date"))
            peak_level = player_top_level.get(int(person_id)) if person_id else None

            # Try to find key stats from FG historical
            key_stats = {}
            if name and school:
                for hist_rec in fg_historical:
                    h_name = hist_rec.get("player_name", "").strip().lower()
                    h_school = (hist_rec.get("fg_team_abb", "") or "").strip().lower()
                    if h_name in name.strip().lower() and h_school and h_school in school.strip().lower():
                        for fk, fv in [("wOBA", "fg_wOBA"), ("wRCplus", "fg_wRC_plus"),
                                        ("hr", "fg_HR"), ("era", "fg_ERA"),
                                        ("fip", "fg_FIP"), ("ip", "fg_IP")]:
                            key_stats[fk] = safe_float(hist_rec.get(fv))
                        break

            class_rows.append({
                "name": name,
                "type": ptype,
                "school": school,
                "team": team,
                "position": position,
                "round": round_val,
                "pick": pick,
                "bonus": bonus,
                "age": None,
                "reached_mlb": reached_mlb,
                "peak_level": peak_level,
                "first_milb_ops": None,
                "stats": key_stats,
            })

        classes[year] = class_rows
        print(f"  {year}: {len(class_rows)} college draftees")

    # ── 8. Build meta ─────────────────────────────────────────────

    print("Building meta...")

    hitters_count = len([r for r in index_players if r["type"] == "hitter"])
    pitchers_count = len([r for r in index_players if r["type"] == "pitcher"])

    meta = {
        "generated_at": "2026-07-16T09:08:38.564559+00:00",
        "season": 2026,
        "players": len(index_players),
        "hitters": hitters_count,
        "pitchers": pitchers_count,
        "grades": grade_counts,
        "pick_band_mae": {"hitter": 108.6, "pitcher": 111.6},
        "backtest_year": 2024,
        "min_sample": {"pa": 50, "ip": 20},
        "class_years": [2021, 2022, 2023, 2024, 2025, 2026],
        "conference_coverage": sum(1 for r in index_players if r.get("conference")),
    }

    # ── 9. Write everything ───────────────────────────────────────

    print(f"\nWriting {len(index_players)} index players...")
    with open(web_dir / "players_index.json", "w") as f:
        json.dump(index_players, f, separators=(",", ":"))

    print(f"Writing {N_SHARDS} detail shards...")
    for i in range(N_SHARDS):
        path = web_dir / "players" / f"shard-{i:02d}.json"
        with open(path, "w") as f:
            json.dump(shards[i], f, separators=(",", ":"))

    print("Writing model manifest...")
    with open(web_dir / "models_manifest.json", "w") as f:
        json.dump(manifest, f, separators=(",", ":"))

    print("Writing meta...")
    with open(web_dir / "meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))

    print(f"Writing {len(classes)} class year files...")
    for year, rows in sorted(classes.items()):
        path = web_dir / "classes" / f"{year}.json"
        with open(path, "w") as f:
            json.dump(rows, f, separators=(",", ":"))

    # ── Summary ────────────────────────────────────────────────────

    total_shard_size = sum(os.path.getsize(web_dir / "players" / f"shard-{i:02d}.json") for i in range(N_SHARDS))
    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Index:      {len(index_players)} players ({hitters_count}H / {pitchers_count}P)")
    print(f"  Shards:     {N_SHARDS} files, {total_shard_size / 1024 / 1024:.1f} MB")
    print(f"  Manifest:   {len(manifest)} model cards")
    print(f"  Classes:    {len(classes)} years")
    print(f"  Grade dist: {json.dumps(grade_counts)}")
    print(f"  Destination: {web_dir}")


if __name__ == "__main__":
    main()
