#!/usr/bin/env python3
"""
build_team_crosswalk.py — Build crosswalk between FanGraphs team abbreviations
and NCAA roster team names.

Strategy:
  1. Manual overrides for known ambiguous/incorrect matches
  2. abb_in_name: FG abbreviation appears in roster team name
  3. name_overlap: shared words between FG full name and roster name

Output: data/rosters/fg_to_roster_crosswalk.json

Usage:
  python3 scripts/build_team_crosswalk.py
"""

import json, re, sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROSTER_PATH = BASE / "data" / "rosters" / "d1_rosters_2026.json"
FG_HITTERS_PATH = BASE / "data" / "fangraphs" / "fg_batters_all.json"
FG_PITCHERS_PATH = BASE / "data" / "fangraphs" / "fg_pitchers_all.json"
OUTPUT_PATH = BASE / "data" / "rosters" / "fg_to_roster_crosswalk.json"

# ── Manual overrides ──────────────────────────────────────────
# Some FG abbreviations map unambiguously to a roster name but the
# heuristic gets it wrong. These override any heuristic match.
# Format: fg_abb → roster_name (exact as in roster data)
MANUAL_OVERRIDES = {
    "LAF": "Lafayette",
    "MICH": "Michigan",
    "IU": "Indiana",
    "IONA": "Iona",
    "KENN": "Kennesaw St.",
    "MCN": "McNeese",
    "LMU": "LMU (CA)",
    "TAR": "Tarleton St.",
    "BUCK": "Bucknell",
    "CONN": "Central Conn. St.",
    "FAIR": "Fairfield",
    "NEB": "Nebraska",
    "MEM": "Memphis",
    "VMI": "VMI",
    "VCU": "VCU",
    "LSU": "LSU",
    "UNLV": "UNLV",
    "GT": "Georgia Tech",
    "UTAH": "Utah Tech",
    "TEX": "Texas",
    "RAD": "Radford",
    "HAW": "Hawaii",
    "VAN": "Vanderbilt",
    "RICH": "Richmond",
    "SIE": "Siena",
    "CAL": "California",
    "BRY": "Bryant",
    "ELON": "Elon",
    "NEV": "Nevada",
    "WOF": "Wofford",
    "HALL": "Seton Hall",
    "NJIT": "NJIT",
    "ARIZ": "Arizona",
    "TOL": "Toledo",
    "XAV": "Xavier",
    "CSUN": "CSUN",
    "NAVY": "Navy",
    "BC": "Boston College",
    "TCU": "TCU",
    "CAM": "Campbell",
    "CREI": "Creighton",
    "LIU": "LIU",
    "COPP": "Coppin St.",
    "ARK": "Arkansas",
    "CAN": "Canisius",
    "LIP": "Lipscomb",
    "UMBC": "UMBC",
    "PAC": "Pacific",
    "NE": "Northeastern",
    "PEPP": "Pepperdine",
    "HOF": "Hofstra",
    "ORE": "Oregon",
    "GRAM": "Grambling",
    "IOWA": "Iowa",
    "AKR": "Akron",
    "GONZ": "Gonzaga",
    "MORE": "Morehead St.",
    "NIA": "Niagara",
    "UCLA": "UCLA",
    "MAN": "Manhattan",
    "DBU": "DBU",
    "ULM": "ULM",
    "LIB": "Liberty",
    "TROY": "Troy",
    "DAY": "Dayton",
    "BUT": "Butler",
    "AFA": "Air Force",
    "LEH": "Lehigh",
    "FGCU": "FGCU",
    "MONM": "Monmouth",
    "USC": "Southern California",
    "OU": "Oklahoma",
    "SAC": "Sacramento St.",
    "STET": "Stetson",
    "PENN": "Penn",
    "DUKE": "Duke",
    "KENT": "Kent St.",
    "DAV": "Davidson",
    "BYU": "BYU",
    "SBU": "St. Bonaventure",
    "CIN": "Cincinnati",
    "STAN": "Stanford",
    "OMA": "Omaha",
    "UPST": "USC Upstate",
    "RGV": "UTRGV",
    "GW": "George Washington",
    "HOU": "Houston",
    "UTA": "UT Arlington",
    "PRES": "Presbyterian",
    "TOW": "Towson",
    "VAL": "Valparaiso",
    "BING": "Binghamton",
    "WASH": "Washington",
    "APP": "App State",
    "MILW": "Milwaukee",
    "UAB": "UAB",
    "YALE": "Yale",
    "COLU": "Columbia",
    "UMES": "UMES",
    "HARV": "Harvard",
    "PRIN": "Princeton",
    "DART": "Dartmouth",
    "COR": "Cornell",
    "STO": "Stonehill",
    "LIN": "Lindenwood",
    "LEM": "Le Moyne",
    "WES": "West Ga.",
    "MERC": "Mercyhurst",
    "ECU": "East Carolina",
    "JOES": "Saint Joseph's",
    "SHU": "Sacred Heart",
    "ORU": "Oral Roberts",
    "NCSU": "NC State",
    "TA&M": "Texas A&M",
    "LR": "Little Rock",
    "LAS": "La Salle",
    "MSM": "Mount St. Mary's",
    "NDSU": "North Dakota St.",
    "SHSU": "Sam Houston",
    "UCSB": "UC Santa Barbara",
    "JMU": "James Madison",
    "ACU": "Abilene Christian",
    "CLT": "Charlotte",
    "SJSU": "San Jose St.",
    "USD": "San Diego",
    "LBSU": "Long Beach St.",
    "UNF": "North Florida",
    "UNM": "New Mexico",
    "ODU": "Old Dominion",
    "TTU": "Texas Tech",
    "PV": "Prairie View",
    "SDST": "South Dakota St.",
    "AAMU": "Alabama A&M",
    "SDSU": "San Diego St.",
    "TXSO": "Texas Southern",
    "NMSU": "New Mexico St.",
    "SMC": "Saint Mary's (CA)",
    "NCAT": "N.C. A&T",
    "HCU": "Houston Christian",
    "STBK": "Stony Brook",
    "UNO": "New Orleans",
    "CCU": "Coastal Carolina",
    "SJU": "St. John's (NY)",
    "APSU": "Austin Peay",
    "FAMU": "Florida A&M",
    "BGSU": "Bowling Green",
    "SLU": "Saint Louis",
    "HPU": "High Point",
    "WVU": "West Virginia",
    "HC": "Holy Cross",
    "UVU": "Utah Valley",
    "SCU": "Santa Clara",
    "W&M": "William & Mary",
    "STMN": "St. Thomas (MN)",
    "MSST": "Mississippi St.",
    "MIZ": "Missouri",
    "UGA": "Georgia",
    "UVA": "Virginia",
    "VT": "Virginia Tech",
    "FSU": "Florida St.",
    "OKST": "Oklahoma St.",
    "OSU": "Ohio St.",
    "MSU": "Michigan St.",
    "NU": "Northwestern",
    "PSU": "Penn St.",
    "ORST": "Oregon St.",
    "WSU": "Washington St.",
    "ASU": "Arizona St.",
    "ALST": "Alabama St.",
    "ARST": "Arkansas St.",
    "BCU": "Bethune-Cookman",
    "CBU": "California Baptist",
    "CHSO": "Charleston So.",
    "CMU": "Central Mich.",
    "EMU": "Eastern Mich.",
    "FAU": "Fla. Atlantic",
    "GASO": "Ga. Southern",
    "GAST": "Georgia St.",
    "GTWN": "Georgetown",
    "ILST": "Illinois St.",
    "INST": "Indiana St.",
    "JAX": "Jacksonville",
    "JVST": "Jacksonville St.",
    "KSU": "Kansas St.",
    "KU": "Kansas",
    "MD": "Maryland",
    "MRSH": "Marshall",
    "MTSU": "Middle Tenn.",
    "NKU": "Northern Ky.",
    "SELA": "Southeastern La.",
    "SEMO": "Southeast Mo. St.",
    "TULN": "Tulane",
    "TXST": "Texas St.",
    "UCD": "UC Davis",
    "UCI": "UC Irvine",
    "UCR": "UC Riverside",
    "WCU": "Western Caro.",
    "WIU": "Western Ill.",
    "WKU": "Western Ky.",
    "WMU": "Western Mich.",
    "WRST": "Wright St.",
    "YSU": "Youngstown St.",
    "USF": "South Fla.",
    "USM": "Southern Miss.",
    "UTM": "UT Martin",
    "GWEB": "Gardner-Webb",
    "MRMK": "Merrimack",
    "MOST": "Missouri St.",
    "QUC": "Queens (NC)",
    "USI": "Southern Ind.",
    "CARK": "Central Ark.",
    "UNA": "North Ala.",
    "CCSU": "Central Conn. St.",
    "UNCO": "Northern Colo.",
    "EIU": "Eastern Ill.",
    "M-OH": "Miami (OH)",
    "UML": "UMass Lowell",
    "MRST": "Marist",
    "SPU": "Saint Peter's",
    "EKU": "Eastern Ky.",
    "TNTC": "Tennessee Tech",
    "JKST": "Jackson St.",
    "ALCN": "Alcorn",
    "DSU": "Delaware St.",
    "MVSU": "Mississippi Val.",
    "UAPB": "Ark.-Pine Bluff",
    "BRWN": "Brown",
    "CIT": "The Citadel",
    "CP": "Cal Poly",
    "CSUB": "CSU Bakersfield",
    "CSUF": "Cal St. Fullerton",
    "UTU": "Utah Tech",
    "PFW": "Purdue Fort Wayne",
    "NWST": "Northwestern St.",
    "BEL": "Belmont",
    "BELL": "Bellarmine",
    "FOR": "Fordham",
    "LOU": "Louisville",
    "ND": "Notre Dame",
    "OHIO": "Ohio St.",
    "ILL": "Illinois",
    "MISS": "Ole Miss",
    "UCSD": "UC San Diego",
    "GMU": "George Mason",
    "COFC": "College of Charleston",
    "AMCC": "A&M-Corpus Christi",
    "USA": "South Alabama",
    "UNCG": "UNC Greensboro",
    "UNCA": "UNC Asheville",
    "NCCU": "North Carolina Central",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def normalize(s):
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def roster_name_index(rosters):
    """Build a mapping of normalized roster team name → first roster entry's team info."""
    name_map = {}
    for player in rosters:
        key = normalize(player.get("team_name", ""))
        if key and key not in name_map:
            name_map[key] = {
                "team_name": player["team_name"],
                "conference": player.get("conference", ""),
                "team_id": player.get("team_id"),
            }
    return name_map


def get_fg_team_names(fg_data):
    """Extract unique (team_abb, team_name) pairs from FG data."""
    teams = set()
    for player in fg_data:
        abb = player.get("team_abb") or player.get("fg_team_abb", "")
        name = player.get("team_name") or player.get("fg_school", "")
        if abb:
            teams.add((abb.upper(), name))
    return sorted(teams)


def main():
    # Load data
    rosters = load_json(ROSTER_PATH)
    fg_hitters = load_json(FG_HITTERS_PATH)
    fg_pitchers = load_json(FG_PITCHERS_PATH)
    fg_all = fg_hitters + fg_pitchers

    roster_names = roster_name_index(rosters)
    fg_teams = get_fg_team_names(fg_all)

    print(f"Loaded {len(rosters)} roster players, {len(roster_names)} unique team names")
    print(f"Loaded {len(fg_teams)} FG teams")

    # Build crosswalk
    crosswalk = {}
    unmatched = {}

    for fg_abb, fg_full_name in fg_teams:
        norm_abb = fg_abb.strip().upper()

        # 1. Manual override
        if norm_abb in MANUAL_OVERRIDES:
            target = MANUAL_OVERRIDES[norm_abb]
            target_norm = normalize(target)
            if target_norm in roster_names:
                info = roster_names[target_norm]
                crosswalk[norm_abb] = {
                    "fg_abb": norm_abb,
                    "fg_full_name": fg_full_name,
                    "roster_name": info["team_name"],
                    "conference": info["conference"],
                    "team_id": info["team_id"],
                    "match_method": "manual",
                }
                continue
            else:
                print(f"  WARNING: Manual override '{target}' not found in roster names")

        # 2. abb_in_name: fg_abb appears in roster team name
        matched = False
        for rn_key, rn_info in roster_names.items():
            if norm_abb.lower() in rn_key:
                crosswalk[norm_abb] = {
                    "fg_abb": norm_abb,
                    "fg_full_name": fg_full_name,
                    "roster_name": rn_info["team_name"],
                    "conference": rn_info["conference"],
                    "team_id": rn_info["team_id"],
                    "match_method": "abb_in_name",
                    "match_score": f"abb={norm_abb.lower()} in {rn_key}",
                }
                matched = True
                break
        if matched:
            continue

        # 3. name_overlap: compare normalized FG full name vs roster name
        best_overlap = 0
        best_match = None
        norm_fg = set(normalize(fg_full_name).split())
        for rn_key, rn_info in roster_names.items():
            norm_rn = set(rn_key.split())
            overlap = len(norm_fg & norm_rn)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = rn_info

        if best_overlap >= 2 and best_match:
            crosswalk[norm_abb] = {
                "fg_abb": norm_abb,
                "fg_full_name": fg_full_name,
                "roster_name": best_match["team_name"],
                "conference": best_match["conference"],
                "team_id": best_match["team_id"],
                "match_method": "name_overlap",
                "match_score": f"common_words={norm_fg & set(normalize(best_match['team_name']).split())}",
            }
        else:
            unmatched[norm_abb] = fg_full_name

    # Summary
    summary = {
        "total_fg_teams": len(fg_teams),
        "matched": len(crosswalk),
        "unmatched": len(unmatched),
        "match_rate": round(len(crosswalk) / len(fg_teams) * 100, 1) if fg_teams else 0,
    }

    print(f"\n=== Crosswalk Summary ===")
    print(f"  Total teams: {summary['total_fg_teams']}")
    print(f"  Matched: {summary['matched']} ({summary['match_rate']}%)")
    print(f"  Unmatched: {summary['unmatched']}")
    if unmatched:
        print(f"  Unmatched teams: {list(unmatched.keys())}")

    # Write output
    output = {
        "crosswalk": crosswalk,
        "unmatched": unmatched,
        "summary": summary,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote crosswalk to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
