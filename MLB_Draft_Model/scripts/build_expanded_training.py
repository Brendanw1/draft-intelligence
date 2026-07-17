#!/usr/bin/env python3
"""
build_expanded_training.py — Build expanded training set using ALL
player-seasons from FG data (not just the draft year).

Methodology:
  - FG drafted files contain all college seasons for drafted players
  - Join with draft data via xMLBAMID → person_id
  - Each player-season becomes a training example
  - Height/weight from draft data (MLB combine measurements)
  - Conference via team crosswalk
  - Multiple seasons per player → grouped cross-validation

Statistical rationale:
  - Using all seasons (freshman, sophomore, junior) captures development
  - Multiple seasons per player provides 2.7x more training data
  - Year-over-year changes help the model learn growth vs. stagnation
  - Grouped CV prevents player-level leakage across folds

Industry context:
  - Standard in MLB front office models (per Fangraphs, The Athletic reports)
  - Multi-year college performance is more predictive than single-season
  - Age relative to class year is a known signal (young-for-grade = projection)
  - Conference adjustment is standard (SEC/ACC/Big 12 premium)

Usage:
  python3 scripts/build_expanded_training.py [--train-until YEAR]

By default trains on all available data (2015-2026). Pass --train-until 2025
to exclude the most recent draft class for retrospective validation.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

FG_BATTERS = DATA / "fangraphs" / "fg_batters_drafted.json"
FG_PITCHERS = DATA / "fangraphs" / "fg_pitchers_drafted.json"
DRAFT_DATA = DATA / "draft" / "draft_all_picks.json"
ROSTER_PATH = DATA / "rosters" / "d1_rosters_2026.json"
CROSSWALK_PATH = DATA / "rosters" / "fg_to_roster_crosswalk.json"
OUTPUT_PATH = DATA / "training" / "expanded_training_set.json"
TIER2_OUTPUT = DATA / "training" / "expanded_tier2_set.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_height(height_str):
    """Convert height like \"6' 4\\\"\" or \"6-4\" to inches."""
    if not height_str:
        return None
    s = str(height_str).strip().replace("'", "-").replace('"', "").replace(" ", "")
    try:
        parts = s.split("-")
        if len(parts) == 2:
            return int(parts[0]) * 12 + int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def compute_bmi(weight_lbs, height_inches):
    if not weight_lbs or not height_inches or weight_lbs <= 0 or height_inches <= 0:
        return None
    return round(weight_lbs * 703 / (height_inches ** 2), 1)


# ── Conference tier mapping ─────────────────────────────
CONFERENCE_TIERS = {
    "SEC": 1, "ACC": 1, "Big 12": 1,
    "Big Ten": 2, "Pac-12": 2, "Pac 12": 2, "Big East": 2,
    "West Coast": 2, "Sun Belt": 2, "American": 2, "Mountain West": 2,
    "CUSA": 3, "MAC": 3, "Mid-American": 3, "SoCon": 3, "Southern": 3,
    "MVC": 3, "Missouri Valley": 3, "Atlantic 10": 3, "Ivy League": 3,
    "Ivy": 3, "CAA": 3, "Colonial": 3, "ASUN": 3, "Southland": 3,
    "Big South": 3, "Big West": 3, "Horizon": 3, "WAC": 3,
    "Patriot": 4, "Patriot League": 4, "America East": 4, "NEC": 4,
    "Northeast": 4, "MAAC": 4, "Metro Atlantic": 4, "Summit League": 4,
    "SWAC": 4, "OVC": 4, "Ohio Valley": 4, "MEAC": 4, "Big Sky": 4,
    "DI Independent": 3,
}


def get_conference_tier(conf):
    if not conf:
        return 4
    conf = conf.strip()
    if conf in CONFERENCE_TIERS:
        return CONFERENCE_TIERS[conf]
    for key, tier in CONFERENCE_TIERS.items():
        if key.lower() == conf.lower():
            return tier
    return 4


def build_fg_team_conference_index(crosswalk):
    """Extract fg_abb → conference from our existing crosswalk."""
    mapping = {}
    cw = crosswalk.get("crosswalk", {})
    for fg_abb, info in cw.items():
        conf = info.get("conference", "")
        if conf:
            mapping[fg_abb.upper()] = conf
    return mapping


def normalize_name(name):
    """Normalize team/school name for fuzzy matching."""
    if not name:
        return ""
    name = name.lower().strip()
    name = name.replace("st.", "state").replace("st ", "state ")
    name = name.replace("univ.", "university").replace("univ ", "university ")
    import re
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def main():
    parser = argparse.ArgumentParser(description="Build expanded training set")
    parser.add_argument("--train-until", type=int, default=2025,
                        help="Max draft year to include (e.g. 2025 to exclude 2026 outcomes). Default 2025.")
    args = parser.parse_args()

    print("=" * 60)
    print("BUILDING EXPANDED TRAINING SET")
    if args.train_until:
        print(f"  Training data limited to draft year ≤ {args.train_until}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    fg_batters = load_json(FG_BATTERS)
    fg_pitchers = load_json(FG_PITCHERS)
    draft = load_json(DRAFT_DATA)

    print(f"  FG batters (drafted seasons): {len(fg_batters):,}")
    print(f"  FG pitchers (drafted seasons): {len(fg_pitchers):,}")
    print(f"  Draft picks (2015-2026): {len(draft):,}")

    # Index draft data by person_id
    draft_index = {}
    for pick in draft:
        pid = pick.get("person_id")
        if pid:
            draft_index[pid] = pick

    print(f"  Draft index: {len(draft_index):,} unique person_ids")

    # Load crosswalk for conference mapping
    crosswalk = load_json(CROSSWALK_PATH)

    # Build conference index from crosswalk
    conf_index = build_fg_team_conference_index(crosswalk)
    print(f"  Teams in conference index: {len(conf_index):,}")

    # Also build name-based lookup: full FG team name → fg_abb
    # This lets us map team_name (like "Virginia Tech") to an fg_abb
    cw = crosswalk.get("crosswalk", {})
    fg_name_to_abb = {}
    roster_name_to_abb = {}
    for fg_abb, info in cw.items():
        fg_full = normalize_name(info.get("fg_full_name", ""))
        roster = normalize_name(info.get("roster_name", ""))
        if fg_full:
            fg_name_to_abb[fg_full] = fg_abb.upper()
        if roster:
            roster_name_to_abb[roster] = fg_abb.upper()

    # Build player-seasons
    hitter_seasons = []
    pitcher_seasons = []
    matched_h = 0
    matched_p = 0
    unmatched = 0

    def process_fg_record(r, player_type):
        nonlocal matched_h, matched_p, unmatched
        xid = r.get("xMLBAMID")
        if not xid:
            unmatched += 1
            return None

        draft_record = draft_index.get(int(xid)) if xid else None
        if not draft_record:
            unmatched += 1
            return None

        if player_type == "hitter":
            matched_h += 1
        else:
            matched_p += 1

        # Height from draft data
        height_raw = draft_record.get("height", "")
        height_inches = None
        if isinstance(height_raw, str):
            height_inches = parse_height(height_raw)
        elif isinstance(height_raw, (int, float)):
            height_inches = float(height_raw)

        weight = draft_record.get("weight")
        bmi = compute_bmi(weight, height_inches) if weight and height_inches else None

        # Conference: try team_name_abb first, then team_name
        team_abb = (r.get("team_name_abb") or "").strip().upper()
        conference = conf_index.get(team_abb, "")
        if not conference:
            # Try matching by normalized team name
            team_name_norm = normalize_name(r.get("team_name", ""))
            # Try roster name first (more likely to match)
            for rn, abb in roster_name_to_abb.items():
                if team_name_norm == rn or (len(team_name_norm) > 3 and rn in team_name_norm):
                    conference = conf_index.get(abb, "")
                    break
            if not conference:
                for fn, abb in fg_name_to_abb.items():
                    if team_name_norm == fn or (len(team_name_norm) > 3 and fn in team_name_norm):
                        conference = conf_index.get(abb, "")
                        break
        conf_tier = get_conference_tier(conference)

        return {
            "person_id": int(xid),
            "player_name": r.get("Player", ""),
            "player_type": player_type,
            "season": r.get("Season"),
            "draft_year": draft_record.get("year"),
            "draft_pick": draft_record.get("pick_number"),
            "draft_round": draft_record.get("pick_round"),
            "draft_bonus": draft_record.get("signing_bonus"),
            "draft_team": draft_record.get("team_name"),
            "draft_school": draft_record.get("school_name"),
            "draft_position": draft_record.get("position_name"),
            "height_raw": draft_record.get("height"),
            "height_inches": height_inches,
            "weight": weight,
            "bmi": bmi,
            "bats": draft_record.get("bats"),
            "throws": draft_record.get("throws"),
            "conference": conference,
            "conference_tier": conf_tier,
            # FG stats — copy all stat fields
            **{k: r.get(k) for k in r.keys()
               if k not in ("Player", "Season", "team_name", "team_name_abb",
                            "UPID", "UPURL", "xMLBAMID", "teamid",
                            "conference", "Conf")}
        }

    print("\nProcessing hitter seasons...")
    for r in fg_batters:
        result = process_fg_record(r, "hitter")
        if result and (not args.train_until or result.get("draft_year", 9999) <= args.train_until):
            hitter_seasons.append(result)

    print("Processing pitcher seasons...")
    for r in fg_pitchers:
        result = process_fg_record(r, "pitcher")
        if result and (not args.train_until or result.get("draft_year", 9999) <= args.train_until):
            pitcher_seasons.append(result)

    print(f"\n=== Expansion Results ===")
    print(f"  Hitter seasons: {len(hitter_seasons):,} (matched: {matched_h:,})")
    print(f"  Pitcher seasons: {len(pitcher_seasons):,} (matched: {matched_p:,})")
    print(f"  Unmatched: {unmatched:,}")
    print(f"  Total: {len(hitter_seasons) + len(pitcher_seasons):,}")
    print(f"  vs. original training set: 2,366")
    print(f"  Expansion factor: {(len(hitter_seasons) + len(pitcher_seasons)) / 2366:.1f}x")

    # Unique players
    unique_players = set()
    for s in hitter_seasons + pitcher_seasons:
        unique_players.add(s["person_id"])
    print(f"  Unique players: {len(unique_players):,}")

    # Season distribution
    from collections import Counter
    player_season_counts = Counter()
    for s in hitter_seasons + pitcher_seasons:
        player_season_counts[s["person_id"]] += 1

    season_dist = Counter(player_season_counts.values())
    print(f"\n  Seasons per player:")
    for n in sorted(season_dist.keys()):
        print(f"    {n} season(s): {season_dist[n]:,} players")

    # Year distribution
    year_dist = Counter()
    for s in hitter_seasons + pitcher_seasons:
        year_dist[s["season"]] += 1
    print(f"\n  Records by season:")
    for yr in sorted(year_dist.keys()):
        print(f"    {yr}: {year_dist[yr]:,}")

    # Write output
    all_seasons = hitter_seasons + pitcher_seasons
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_seasons, f, indent=2)
    print(f"\nWrote {len(all_seasons):,} records to {OUTPUT_PATH}")

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPANSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Original:  2,366 records (single-season per player)")
    print(f"  Expanded: {len(all_seasons):,} records (multi-season per player)")
    print(f"  Coverage: {len(fg_batters) + len(fg_pitchers):,} FG seasons → {len(all_seasons):,} usable")


if __name__ == "__main__":
    main()
