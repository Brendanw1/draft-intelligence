#!/usr/bin/env python3
"""
enrich_projections.py — Enrich FG projections with NCAA roster data.

Joins roster bio data (height, position, class, conference) into the
FG projections by matching on team name (via crosswalk) and player
name (fuzzy normalized). Also computes BMI where weight is available.

Input:
  - data/training/projections_2026.json        (raw FG projections)
  - data/rosters/d1_rosters_2026.json          (NCAA roster data)
  - data/rosters/fg_to_roster_crosswalk.json   (team name crosswalk)
  - data/training/fg_training_set.json         (optional, for BMI from weight)

Output:
  - data/training/projections_2026_enriched.json

Usage:
  python3 scripts/enrich_projections.py
"""

import json, re
from pathlib import Path
from difflib import SequenceMatcher

BASE = Path(__file__).resolve().parents[1]
PROJECTIONS_PATH = BASE / "data" / "training" / "projections_2026.json"
ROSTER_PATH = BASE / "data" / "rosters" / "d1_rosters_2026.json"
CROSSWALK_PATH = BASE / "data" / "rosters" / "fg_to_roster_crosswalk.json"
DRAFT_PATH = BASE / "data" / "draft" / "draft_all_picks.json"
OUTPUT_PATH = BASE / "data" / "training" / "projections_2026_enriched.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def normalize_name(name):
    """Normalize player name for matching: lowercase, strip Jr./III/etc."""
    if not name:
        return ""
    name = name.lower().strip()
    # Remove suffixes
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    # Remove punctuation except hyphens
    name = re.sub(r"[^a-z\-\s]", "", name)
    return name


def parse_height(height_str):
    """Convert height string to total inches.
    Handles '6-7' (roster), '6' 7"' (draft), and freeform formats."""
    if not height_str or height_str == "":
        return None
    try:
        s = str(height_str).strip()
        # Try "6-7" format (roster)
        if "-" in s:
            parts = s.split("-")
            if len(parts) == 2:
                feet, inches = int(parts[0]), int(parts[1])
                return feet * 12 + inches
        # Try "6' 7"" or "6' 7'" format (draft records)
        import re
        m = re.match(r"(\d+)\s*['\u2019\u2018]\s*(\d*)\s*[\"\u201d\u201c]?", s)
        if m:
            feet = int(m.group(1))
            inches = int(m.group(2)) if m.group(2) else 0
            return feet * 12 + inches
        # Try bare number (already inches)
        return int(float(s))
    except (ValueError, IndexError):
        pass
    return None


def height_display(inches):
    """Convert inches to '6-4' format."""
    if inches is None:
        return None
    feet = inches // 12
    remaining = inches % 12
    return f"{feet}-{remaining}"


def build_roster_index(rosters):
    """
    Build index of roster players by (normalized_team_name, normalized_player_name).
    Returns dict: team_key → { player_key → [player_records] }
    """
    index = {}
    for player in rosters:
        team_name = player.get("team_name", "")
        team_key = normalize_name(team_name)
        player_name = player.get("player_name", "") or player.get("full_name", "")
        player_key = normalize_name(player_name)

        if not team_key or not player_key:
            continue

        if team_key not in index:
            index[team_key] = {}
        if player_key not in index[team_key]:
            index[team_key][player_key] = []
        index[team_key][player_key].append(player)
    return index


def build_fg_team_index(crosswalk):
    """Build mapping: fg_abb → roster team name."""
    cw = crosswalk.get("crosswalk", {})
    mapping = {}
    for fg_abb, info in cw.items():
        mapping[fg_abb.upper()] = info["roster_name"]
    return mapping


def find_best_match(projection_player_name, roster_players, threshold=0.75):
    """
    Find best fuzzy match for a player name within a team's roster.
    Returns the matched roster record or None.
    """
    if not roster_players:
        return None, None, 0.0

    norm_proj = normalize_name(projection_player_name)

    # Exact match first
    if norm_proj in roster_players:
        return roster_players[norm_proj][0], norm_proj, 1.0

    # Fuzzy match
    best_score = 0
    best_key = None
    for player_key in roster_players:
        score = SequenceMatcher(None, norm_proj, player_key).ratio()
        if score > best_score:
            best_score = score
            best_key = player_key

    if best_score >= threshold and best_key:
        return roster_players[best_key][0], best_key, best_score

    return None, None, 0.0


def compute_bmi(weight_lbs, height_inches):
    """Compute BMI from weight (lbs) and height (inches)."""
    if not weight_lbs or not height_inches or weight_lbs <= 0 or height_inches <= 0:
        return None
    return round(weight_lbs * 703 / (height_inches ** 2), 1)


def main():
    print("Loading data...")
    projections = load_json(PROJECTIONS_PATH)
    rosters = load_json(ROSTER_PATH)
    crosswalk = load_json(CROSSWALK_PATH)

    print(f"  Projections: {len(projections)}")
    print(f"  Rosters: {len(rosters)}")
    print(f"  Crosswalk: {crosswalk['summary']['matched']} teams matched")

    # Load draft records for height/weight fallback
    draft_records = load_json(DRAFT_PATH)
    draft_by_pid = {}
    for p in draft_records:
        pid = p.get("person_id")
        if pid and pid not in draft_by_pid:
            draft_by_pid[pid] = p
    print(f"  Draft records indexed: {len(draft_by_pid)}")

    # Build indexes
    roster_idx = build_roster_index(rosters)
    fg_to_roster_team = build_fg_team_index(crosswalk)

    # Conference tier mapping (approximate tiers by conference strength)
    CONFERENCE_TIERS = {
        "SEC": 1,
        "ACC": 1,
        "Big 12": 1,
        "Big Ten": 2,
        "Pac-12": 2,
        "Pac 12": 2,
        "Pac12": 2,
        "DI Independent": 3,
        "Big East": 2,
        "West Coast": 2,
        "Sun Belt": 2,
        "American": 2,
        "Mountain West": 2,
        "CUSA": 3,
        "MAC": 3,
        "Mid-American": 3,
        "SoCon": 3,
        "Southern": 3,
        "MVC": 3,
        "Missouri Valley": 3,
        "Atlantic 10": 3,
        "Ivy League": 3,
        "Ivy": 3,
        "Colonial": 3,
        "CAA": 3,
        "ASUN": 3,
        "ASUN W-B": 3,
        "Southland": 3,
        "Big South": 3,
        "Big West": 3,
        "Horizon": 3,
        "WAC": 3,
        "Patriot": 4,
        "Patriot League": 4,
        "America East": 4,
        "NEC": 4,
        "Northeast": 4,
        "MAAC": 4,
        "Metro Atlantic": 4,
        "Summit League": 4,
        "SWAC": 4,
        "OVC": 4,
        "Ohio Valley": 4,
        "MEAC": 4,
        "Big Sky": 4,
    }

    def get_conference_tier(conference):
        if not conference:
            return 4
        conf = conference.strip()
        # Try direct lookup
        if conf in CONFERENCE_TIERS:
            return CONFERENCE_TIERS[conf]
        # Try case-insensitive
        for key, tier in CONFERENCE_TIERS.items():
            if key.lower() == conf.lower():
                return tier
        # Default to 4 for unknown conferences
        return 4

    # Enrich each projection
    enriched = []
    matched_count = 0
    unmatched_name = 0
    unmatched_team = 0

    for proj in projections:
        record = dict(proj)

        player_name = record.get("player_name", "")
        team_abb = (record.get("team_abb") or record.get("fg_team_abb", "")).upper()
        player_type = record.get("player_type", "")

        # Preserve existing fields, only set defaults for missing ones
        for k in ["height", "height_inches", "position", "class_year",
                   "bats", "throws", "conference", "conference_tier", "bmi"]:
            if k not in record or record[k] is None:
                record[k] = None

        # Determine roster team name via crosswalk
        roster_team_name = fg_to_roster_team.get(team_abb)
        if not roster_team_name:
            unmatched_team += 1
            enriched.append(record)
            continue

        team_key = normalize_name(roster_team_name)
        team_roster = roster_idx.get(team_key, {})

        if not team_roster:
            unmatched_team += 1
            enriched.append(record)
            continue

        # Match player by name
        match, matched_key, score = find_best_match(player_name, team_roster)

        if match:
            matched_count += 1
            height_str = match.get("height", "")
            height_inches = parse_height(height_str)

            record["height"] = height_display(height_inches)
            record["height_inches"] = height_inches
            record["position"] = match.get("position", None)
            record["class_year"] = match.get("class", None)
            record["bats"] = match.get("bats", None)
            record["throws"] = match.get("throws", None)
            record["conference"] = match.get("conference", None)

            # Determine conference tier
            conf = record.get("conference", "")
            record["conference_tier"] = get_conference_tier(conf)

            # Compute BMI if weight is available in projection
            weight = record.get("weight") or record.get("fg_weight")
            if weight and height_inches:
                record["bmi"] = compute_bmi(float(weight), height_inches)

        else:
            unmatched_name += 1

        # Fallback: if height still missing from roster, try draft records by xMLBAMID
        if record.get("height_inches") is None:
            xid = record.get("xMLBAMID") or record.get("mlb_id")
            if xid is not None:
                try:
                    draft_rec = draft_by_pid.get(int(xid))
                except (ValueError, TypeError):
                    draft_rec = None
                if draft_rec:
                    h = draft_rec.get("height", "")
                    if h:
                        hi = parse_height(h)
                        if hi:
                            record["height_inches"] = hi
                            record["height"] = height_display(hi)
                            # Also get weight and compute BMI
                            w = draft_rec.get("weight")
                            if w and hi:
                                record["bmi"] = compute_bmi(float(w), hi)
                # Try person_id as alternate key if xMLBAMID didn't match
                if record.get("height_inches") is None:
                    for pid_key in ["person_id", "mlb_id"]:
                        alt_id = record.get(pid_key)
                        if alt_id is not None and str(alt_id) != str(xid):
                            try:
                                draft_rec = draft_by_pid.get(int(alt_id))
                            except (ValueError, TypeError):
                                draft_rec = None
                            if draft_rec:
                                h = draft_rec.get("height", "")
                                if h:
                                    hi = parse_height(h)
                                    if hi:
                                        record["height_inches"] = hi
                                        record["height"] = height_display(hi)
                                        break

        enriched.append(record)

    # Summary
    print(f"\n=== Enrichment Results ===")
    print(f"  Total projections: {len(enriched)}")
    print(f"  Matched (with roster data): {matched_count}")
    print(f"  Unmatched (no roster team found): {unmatched_team}")
    print(f"  Unmatched (team found, no player match): {unmatched_name}")
    print(f"  Coverage: {matched_count / len(enriched) * 100:.1f}%")

    # Check feature coverage
    with_height = sum(1 for r in enriched if r.get("height_inches"))
    with_position = sum(1 for r in enriched if r.get("position"))
    with_conference = sum(1 for r in enriched if r.get("conference"))
    with_conf_tier = sum(1 for r in enriched if r.get("conference_tier"))
    with_bmi = sum(1 for r in enriched if r.get("bmi"))

    print(f"  With height: {with_height}")
    print(f"  With position: {with_position}")
    print(f"  With conference: {with_conference}")
    print(f"  With conference_tier: {with_conf_tier}")
    print(f"  With BMI: {with_bmi}")

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"\nWrote {len(enriched)} enriched projections to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
