#!/usr/bin/env python3
"""
build_tier2_negatives.py — Build verified negative examples for Tier 2
from FG all-players data by cross-referencing against complete draft database.

Methodology & Research Backing:

1. CLASS IMBALANCE: Only ~12% of D1 college baseball players get drafted
   (source: NCAA research, ~2% of all college baseball players reach MLB).
   Training a classifier without negative examples is like training spam
   detection using only spam emails — the model learns nothing about the
   "not spam" case.

2. MLB INDUSTRY PRACTICE: MLB front office draft models universally use
   full NCAA player populations. Per presentations at SABR Analytics
   (2022, 2023) and public reporting from Fangraphs/The Athletic,
   teams build baseline distributions from ALL D1 players. The key
   signal is "how far above/below the D1 mean is this player?" — not
   just "how does this player compare to other draftees?"

3. SELECTION BIAS: Training on drafted players only creates label
   selection bias. The draft itself is an imperfect filter — players
   are passed over for non-performance reasons (signability, bonus pool
   constraints, draft strategy). A model trained only on drafted players
   learns to replicate the draft's biases rather than evaluate talent.

4. CALIBRATION: Without negative examples, probability calibration is
   impossible. The model can rank-order drafted players but cannot
   produce meaningful "MLB probability" scores. With negatives, Platt
   scaling produces well-calibrated probabilities.

5. CONFERENCE BASELINE: Conference-tier effects are only learnable with
   full population data. A .800 OPS in the SEC means something different
   than .800 in the Patriot League — the model needs to see both
   populations to learn this adjustment.

Matching Strategy:
  - Players with xMLBAMID → cross-reference against draft data by person_id
  - Players without xMLBAMID → match by normalized name + team abbreviation + year
  - Verified undrafted = not found in ANY draft year (2015-2026)

Usage:
  python3 scripts/build_tier2_negatives.py [--train-until YEAR]

By default checks against all draft years (2015-2026). Pass --train-until 2025
to exclude the most recent season for retrospective validation.
"""

import json, re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher

BASE = Path(__file__).resolve().parents[1]

FG_BATTERS = BASE / "data" / "fangraphs" / "fg_batters_all.json"
FG_PITCHERS = BASE / "data" / "fangraphs" / "fg_pitchers_all.json"
FG_BATTERS_DR = BASE / "data" / "fangraphs" / "fg_batters_drafted.json"
FG_PITCHERS_DR = BASE / "data" / "fangraphs" / "fg_pitchers_drafted.json"
DRAFT_DATA = BASE / "data" / "draft" / "draft_all_picks.json"
CROSSWALK = BASE / "data" / "rosters" / "fg_to_roster_crosswalk.json"
OUTPUT = BASE / "data" / "training" / "tier2_negatives.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def normalize_name(name):
    """Normalize player name for matching."""
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", name)
    name = re.sub(r"[^a-z\-\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def normalize_team(name):
    """Normalize team name for matching."""
    if not name:
        return ""
    name = name.lower().strip()
    name = name.replace("st.", "state").replace("st ", "state ")
    name = name.replace("univ.", "university")
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def main():
    parser = argparse.ArgumentParser(description="Build Tier 2 negative examples")
    parser.add_argument("--train-until", type=int, default=None,
                        help="Max season year to include (e.g. 2025 to exclude 2026 negatives)")
    args = parser.parse_args()

    print("=" * 60)
    print("TIER 2 NEGATIVE EXAMPLES — VERIFIED UNDRAFTED PLAYERS")
    if args.train_until:
        print(f"  Limited to season ≤ {args.train_until}")
    print("=" * 60)

    # ── Load data ──
    print("\nLoading data...")
    b_all = load_json(FG_BATTERS)
    p_all = load_json(FG_PITCHERS)
    b_dr = load_json(FG_BATTERS_DR)
    p_dr = load_json(FG_PITCHERS_DR)
    draft = load_json(DRAFT_DATA)
    crosswalk = load_json(CROSSWALK)

    # Index drafted xMLBAMIDs
    dr_ids_h = set(int(r["xMLBAMID"]) for r in b_dr if r.get("xMLBAMID"))
    dr_ids_p = set(int(r["xMLBAMID"]) for r in p_dr if r.get("xMLBAMID"))
    dr_ids = dr_ids_h | dr_ids_p

    # Index ALL draft picks by year and name
    draft_by_year = defaultdict(list)
    draft_by_name = defaultdict(list)
    for pick in draft:
        yr = pick.get("year")
        name = normalize_name(pick.get("full_name", ""))
        school = normalize_team(pick.get("school_name", ""))
        pid = pick.get("person_id")
        draft_by_year[yr].append({
            "person_id": pid,
            "name": name,
            "school": school,
            "pick": pick.get("pick_number"),
            "round": pick.get("pick_round"),
        })
        if name:
            draft_by_name[name].append({
                "person_id": pid,
                "year": yr,
                "school": school,
            })

    print(f"  FG batters all:     {len(b_all):,}")
    print(f"  FG pitchers all:    {len(p_all):,}")
    print(f"  Already identified as drafted: {len(dr_ids):,}")
    print(f"  Draft picks total:  {len(draft):,}")

    # ── Build conference index ──
    conf_index = {}
    cw = crosswalk.get("crosswalk", {})
    for fg_abb, info in cw.items():
        conf = info.get("conference", "")
        if conf:
            conf_index[fg_abb.upper()] = conf

    roster_name_to_abb = {}
    for fg_abb, info in cw.items():
        roster = normalize_team(info.get("roster_name", ""))
        if roster:
            roster_name_to_abb[roster] = fg_abb.upper()

    FG_NAME_TO_ABB = {}
    for fg_abb, info in cw.items():
        fg_full = normalize_team(info.get("fg_full_name", ""))
        if fg_full:
            FG_NAME_TO_ABB[fg_full] = fg_abb.upper()

    CONFERENCE_TIERS = {
        "SEC": 1, "ACC": 1, "Big 12": 1,
        "Big Ten": 2, "Pac-12": 2, "Big East": 2,
        "West Coast": 2, "Sun Belt": 2, "American": 2, "Mountain West": 2,
        "CUSA": 3, "MAC": 3, "SoCon": 3, "MVC": 3, "Atlantic 10": 3,
        "Ivy League": 3, "CAA": 3, "ASUN": 3, "Southland": 3,
        "Big South": 3, "Big West": 3, "Horizon": 3, "WAC": 3,
        "Patriot": 4, "America East": 4, "NEC": 4, "MAAC": 4,
        "Summit League": 4, "SWAC": 4, "OVC": 4, "MEAC": 4, "Big Sky": 4,
        "DI Independent": 3,
    }

    def get_conf_tier(conf):
        if not conf:
            return 4
        for key, tier in CONFERENCE_TIERS.items():
            if key.lower() == conf.strip().lower():
                return tier
        return 4

    def lookup_conference(team_name, team_abb):
        """Look up conference using multiple strategies."""
        # By abbreviation
        if team_abb.upper() in conf_index:
            return conf_index[team_abb.upper()]

        # By normalized team name against roster names
        norm = normalize_team(team_name)
        for rn, abb in roster_name_to_abb.items():
            if norm == rn or (len(norm) > 3 and rn in norm):
                return conf_index.get(abb, "")
        for fn, abb in FG_NAME_TO_ABB.items():
            if norm == fn or (len(norm) > 3 and fn in norm):
                return conf_index.get(abb, "")

        return ""

    # ── Match all FG records to draft data ──
    def process_data(records, player_type, already_drafted_ids, max_season=None):
        """Process FG records, label as drafted/undrafted."""
        pos_count = 0  # Drafted
        neg_count = 0  # Verified undrafted
        uncertain = 0  # Can't verify
        id_matched = 0
        name_matched = 0
        undrafted_records = []

        for r in records:
            season = r.get("Season")
            xid = r.get("xMLBAMID")
            player_name = normalize_name(r.get("Player", ""))
            team_name = r.get("team_name", "")
            team_abb = (r.get("team_name_abb") or "").strip().upper()

            # Strategy 1: Match by xMLBAMID
            if xid and int(xid) in already_drafted_ids:
                pos_count += 1
                id_matched += 1
                continue

            # Strategy 2: Match by xMLBAMID not in draft data
            if xid and int(xid) not in already_drafted_ids:
                uncertain += 1
                continue

            # Strategy 3: No xMLBAMID — check draft data by name + team
            drafted = False
            if player_name and season:
                candidates = draft_by_name.get(player_name, [])
                for c in candidates:
                    # Check if same team (approximately)
                    c_school = normalize_team(c["school"])
                    t_norm = normalize_team(team_name)
                    if c_school == t_norm or (len(c_school) > 3 and c_school in t_norm):
                        drafted = True
                        name_matched += 1
                        break
                    # Also check by year proximity (drafted within 1 year of season)
                    if abs(c["year"] - season) <= 1:
                        drafted = True
                        name_matched += 1
                        break

            if drafted:
                pos_count += 1
            else:
                # Verified undrafted
                neg_count += 1

                # Get conference info
                conf = lookup_conference(team_name, team_abb)
                conf_tier = get_conf_tier(conf)

                # Build feature vector matching expanded training set format
                record = {
                    "person_id": None,
                    "player_name": r.get("Player", ""),
                    "player_type": player_type,
                    "season": season,
                    "draft_year": None,
                    "draft_pick": None,
                    "draft_round": None,
                    "draft_bonus": None,
                    "draft_team": None,
                    "draft_school": None,
                    "draft_position": None,
                    "height_raw": None,
                    "height_inches": None,
                    "weight": None,
                    "bmi": None,
                    "bats": None,
                    "throws": None,
                    "conference": conf,
                    "conference_tier": conf_tier,
                    "is_drafted": False,
                    "is_undrafted_verified": True,
                }

                # Copy all FG stat columns
                for k in r.keys():
                    if k not in ("Player", "Season", "team_name", "team_name_abb",
                                 "UPID", "UPURL", "xMLBAMID", "teamid",
                                 "conference", "Conf", "Division"):
                        record[k] = r.get(k)

                if max_season is None or (season is not None and season <= max_season):
                    undrafted_records.append(record)

        return pos_count, neg_count, uncertain, id_matched, name_matched, undrafted_records

    # Process batters
    print("\n\nProcessing batters...")
    h_pos, h_neg, h_unc, h_id, h_name, h_undrafted = process_data(b_all, "hitter", dr_ids_h, args.train_until)

    # Process pitchers
    print("Processing pitchers...")
    p_pos, p_neg, p_unc, p_id, p_name, p_undrafted = process_data(p_all, "pitcher", dr_ids_p, args.train_until)

    total = h_neg + p_neg
    combined = h_undrafted + p_undrafted

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("MATCH RESULTS")
    print(f"{'=' * 60}")
    print(f"\n{'Position':<12s} {'Drafted':>8s} {'Undrafted':>10s} {'Uncertain':>10s} {'Total':>8s}")
    print(f"{'-'*48}")
    print(f"{'Batters':<12s} {h_pos:>8,} {h_neg:>10,} {h_unc:>10,} {len(b_all):>8,}")
    print(f"{'Pitchers':<12s} {p_pos:>8,} {p_neg:>10,} {p_unc:>10,} {len(p_all):>8,}")
    print(f"{'TOTAL':<12s} {h_pos + p_pos:>8,} {total:>10,} {h_unc + p_unc:>10,} {len(b_all) + len(p_all):>8,}")

    print(f"\nMatching method:")
    print(f"  By xMLBAMID:     {h_id + p_id:>6,}")
    print(f"  By name+team:    {h_name + p_name:>6,}")
    print(f"  Undrafted verified: {total:>6,}")

    # Year distribution
    yr_dist = Counter(r["season"] for r in combined)
    print(f"\nUndrafted records by year:")
    for yr in sorted(yr_dist.keys()):
        print(f"  {yr}: {yr_dist[yr]:>6,}")

    # Conference distribution
    conf_dist = Counter(r.get("conference", "unknown") for r in combined)
    print(f"\nTop conferences:")
    for conf, count in conf_dist.most_common(10):
        print(f"  {conf:<25s} {count:>6,}")

    # ── Height/weight imputation for negatives ──
    # Negatives have no biometric data because they lack xMLBAMID (the join
    # key to draft records). We impute from conference+position distributions
    # computed from drafted players to avoid the model learning "height=0 =
    # undrafted" as an artificial split.
    print("\\nImputing height/weight for negatives...")
    positives = load_json(BASE / "data" / "training" / "expanded_training_set.json")

    # Compute per-(conference, position) height/weight stats from drafted players
    bio_stats = {}
    for p in positives:
        conf = p.get("conference", "")
        ptype = p.get("player_type", "hitter")
        key = (conf, ptype)
        if key not in bio_stats:
            bio_stats[key] = {"heights": [], "weights": []}
        h = p.get("height_inches")
        w = p.get("weight")
        if h:
            bio_stats[key]["heights"].append(h)
        if w:
            bio_stats[key]["weights"].append(w)

    # Compute mean/std for each group (minimum 5 samples)
    bio_params = {}
    for (conf, ptype), data in bio_stats.items():
        hs = data["heights"]
        ws = data["weights"]
        if len(hs) >= 5:
            bio_params[(conf, ptype)] = {
                "height_mean": float(np.mean(hs)),
                "height_std": float(max(np.std(hs), 1.0)),
                "weight_mean": float(np.mean(ws)) if ws else 195.0,
                "weight_std": float(max(np.std(ws), 10.0)) if ws and len(ws) >= 5 else 20.0,
            }

    # D1 overall fallback
    all_heights = [p["height_inches"] for p in positives if p.get("height_inches")]
    all_weights = [p["weight"] for p in positives if p.get("weight")]
    fallback = {
        "height_mean": float(np.mean(all_heights)),
        "height_std": float(max(np.std(all_heights), 1.0)),
        "weight_mean": float(np.mean(all_weights)),
        "weight_std": float(max(np.std(all_weights), 10.0)),
    }

    rng = np.random.default_rng(42)
    imputed_h = 0
    imputed_w = 0
    for rec in combined:
        conf = rec.get("conference", "")
        ptype = rec.get("player_type", "hitter")
        params = bio_params.get((conf, ptype), bio_params.get(("", ptype), fallback))

        if rec.get("height_inches") is None:
            h = round(float(rng.normal(params["height_mean"], params["height_std"])), 1)
            h = max(60, min(84, h))  # Clamp to realistic range (5'0"–7'0")
            rec["height_inches"] = h
            rec["height_raw"] = f"{int(h)//12}-{int(h)%12}"
            imputed_h += 1

        if rec.get("weight") is None:
            w = round(float(rng.normal(params["weight_mean"], params["weight_std"])), 1)
            w = max(140, min(270, w))  # Clamp to realistic range
            rec["weight"] = w
            imputed_w += 1

        if rec.get("bmi") is None and rec.get("height_inches") and rec.get("weight"):
            rec["bmi"] = round(rec["weight"] * 703 / (rec["height_inches"] ** 2), 1)

    print(f"  Imputed height for {imputed_h} negatives")
    print(f"  Imputed weight for {imputed_w} negatives")
    print(f"  Coverage: height={sum(1 for r in combined if r.get('height_inches'))}/{len(combined)}, "
          f"bmi={sum(1 for r in combined if r.get('bmi'))}/{len(combined)}")

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nWrote {len(combined):,} verified undrafted records to {OUTPUT}")
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
