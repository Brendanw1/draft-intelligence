#!/usr/bin/env python3
"""bootstrap_team_mapping.py — Generate team mapping from DuckDB.
Derives readable school names from team code abbreviations where possible.
Usage: python3 scripts/bootstrap_team_mapping.py [--output configs/team_mapping_all_teams.csv]
"""
import csv, os, sys, argparse
from pathlib import Path
import duckdb

DB_PATH = os.path.expanduser("~/baseball/db/baseball.duckdb")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "configs" / "team_mapping_all_teams.csv"

# Common NCAA abbreviation → full school name
SCHOOL_NAMES = {
    "ABI": "Abilene Christian", "AIR": "Air Force", "AKR": "Akron",
    "ALA": "Alabama", "ALB": "Albany", "ALC": "Alcorn State",
    "APP": "Appalachian State", "ARI": "Arizona", "ARK": "Arkansas",
    "ARL": "UT Arlington", "ARM": "Army", "ASU": "Arizona State",
    "AUB": "Auburn", "AUS": "Austin Peay", "BAK": "Bakersfield",
    "BAL": "Ball State", "BAY": "Baylor", "BEL": "Belmont",
    "BET": "Bethune-Cookman", "BIN": "Binghamton", "BOS": "Boston College",
    "BOW": "Bowling Green", "BRA": "Bradley", "BRO": "Brown",
    "BRY": "Bryant", "BUC": "Bucknell", "BUT": "Butler",
    "BYU": "BYU", "CAL": "California", "CAM": "Campbell",
    "CAN": "Canisius", "CEN": "Central Michigan", "CHA": "Charleston",
    "CIT": "The Citadel", "CLE": "Clemson", "CLT": "Charlotte",
    "COA": "Coastal Carolina", "COL": "Columbia", "CON": "Connecticut",
    "COR": "Cornell", "CRE": "Creighton", "CSF": "Cal State Fullerton",
    "CSN": "Cal State Northridge", "CSU": "Charleston Southern", "DAL": "Dallas Baptist",
    "DAR": "Dartmouth", "DAV": "Davidson", "DAY": "Dayton",
    "DEL": "Delaware", "DEP": "DePaul", "DET": "Detroit Mercy",
    "DIX": "Dixie State", "DUK": "Duke", "EAS": "East Carolina",
    "EAS2": "Eastern Illinois", "EAS3": "Eastern Kentucky", "EAS4": "Eastern Michigan",
    "ELO": "Elon", "ETS": "East Tennessee State", "EVA": "Evansville",
    "FAI": "Fairfield", "FAU": "Florida Atlantic", "FGC": "Florida Gulf Coast",
    "FIU": "FIU", "FLA": "Florida", "FOR": "Fordham",
    "FRE": "Fresno State", "FUR": "Furman", "GEO": "Georgetown",
    "GEO2": "George Mason", "GEO3": "George Washington", "GEO4": "Georgia State",
    "GIT": "Georgia Tech", "GON": "Gonzaga", "GRA": "Grand Canyon",
    "HAR": "Harvard", "HAW": "Hawaii", "HOF": "Hofstra",
    "HOL": "Holy Cross", "HOU": "Houston", "HOU2": "Houston Christian",
    "ILL": "Illinois", "ILS": "Illinois State", "IND": "Indiana",
    "INS": "Indiana State", "ION": "Iona", "IOW": "Iowa",
    "JAC": "Jacksonville", "JAM": "James Madison", "KAN": "Kansas",
    "KAN2": "Kansas State", "KEN": "Kentucky", "KEN2": "Kennesaw State",
    "LAM": "Lamar", "LAS": "La Salle", "LEH": "Lehigh",
    "LIB": "Liberty", "LIP": "Lipscomb", "LON": "Long Island",
    "LON2": "Long Beach State", "LOU": "Louisville", "LOU2": "Louisiana",
    "LOU3": "Louisiana Tech", "LOU4": "UL Monroe", "LOY": "Loyola Marymount",
    "LSU": "LSU", "MAN": "Manhattan", "MAR": "Marshall",
    "MAR2": "Marist", "MAS": "Massachusetts", "MCN": "McNeese",
    "MD": "Maryland", "MEM": "Memphis", "MER": "Mercer",
    "MIA": "Miami", "MIA2": "Miami (OH)", "MIC": "Michigan",
    "MIS": "Missouri", "MIN": "Minnesota", "MIS2": "Missouri State",
    "MON": "Monmouth", "MOR": "Morehead State", "MSU": "Mississippi State",
    "MTN": "Mount St. Mary's", "MUR": "Murray State", "NAV": "Navy",
    "NC": "North Carolina", "NC2": "NC State", "NC3": "UNC Greensboro",
    "NC4": "UNC Wilmington", "ND": "Notre Dame", "ND2": "North Dakota State",
    "NEB": "Nebraska", "NEV": "Nevada", "NEW": "New Mexico",
    "NEW2": "New Mexico State", "NIA": "Niagara", "NIC": "Nicholls",
    "NOR": "Norfolk State", "NOR2": "North Florida", "NOR3": "Northern Colorado",
    "NOR4": "Northern Kentucky", "NOR5": "Northwestern", "NOR6": "Northwestern State",
    "OAK": "Oakland", "OHI": "Ohio", "OHI2": "Ohio State",
    "OKL": "Oklahoma", "OKL2": "Oklahoma State", "OLD": "Old Dominion",
    "OLE": "Ole Miss", "OMA": "Omaha", "ORA": "Oral Roberts",
    "ORE": "Oregon", "ORE2": "Oregon State", "PAC": "Pacific",
    "PEN": "Penn", "PEN2": "Penn State", "PEP": "Pepperdine",
    "PIT": "Pittsburgh", "POR": "Portland", "PRA": "Prairie View A&M",
    "PRE": "Presbyterian", "PRI": "Princeton", "PRO": "Providence",
    "PUR": "Purdue", "QUE": "Queens", "RAD": "Radford",
    "RIC": "Richmond", "RID": "Rider", "RUT": "Rutgers",
    "SAC": "Sacramento State", "SAM": "Samford", "SAM2": "Sam Houston",
    "SAN": "San Diego", "SAN2": "San Diego State", "SAN3": "San Francisco",
    "SAN4": "San Jose State", "SAN5": "Santa Clara", "SC": "South Carolina",
    "SC2": "USC Upstate", "SEA": "Seattle U", "SEL": "Southeastern Louisiana",
    "SEM": "Southeast Missouri", "SET": "Seton Hall", "SFA": "Stephen F. Austin",
    "SIE": "Siena", "SIU": "SIU Edwardsville", "SJU": "St. John's",
    "SOU": "Southern", "SOU2": "Southern Illinois", "SOU3": "Southern Miss",
    "SOU4": "South Florida", "SOU5": "South Alabama", "SPU": "Saint Peter's",
    "STA": "Stanford", "STE": "Stetson", "STB": "Stony Brook",
    "STL": "Saint Louis", "STO": "Stonehill", "STT": "St. Thomas",
    "SYR": "Syracuse", "TAR": "Tarleton State", "TCU": "TCU",
    "TEM": "Temple", "TEN": "Tennessee", "TEN2": "Tennessee Tech",
    "TEX": "Texas", "TEX2": "Texas A&M", "TEX3": "Texas State",
    "TEX4": "Texas Tech", "TEX5": "UT Rio Grande Valley", "TEX6": "Texas A&M-Corpus Christi",
    "TOL": "Toledo", "TOW": "Towson", "TRO": "Troy",
    "TUL": "Tulane", "TUL2": "Tulsa", "UAB": "UAB",
    "UCF": "UCF", "UCI": "UC Irvine", "UCL": "UCLA",
    "UCR": "UC Riverside", "UCS": "UC Santa Barbara", "UCS2": "UC San Diego",
    "UCS3": "USC", "UDA": "UC Davis", "UHA": "Hartford",
    "UMA": "UMass Lowell", "UMB": "UMBC", "UNC": "UNC Asheville",
    "UNF": "North Florida", "UNH": "New Hampshire", "UNL": "UNLV",
    "UNM": "New Mexico", "UNO": "New Orleans", "UNR": "Nevada",
    "URI": "Rhode Island", "USA": "South Alabama", "USD": "South Dakota",
    "USF": "South Florida", "USM": "Southern Miss", "UTA": "Utah",
    "UTA2": "Utah Valley", "UTA3": "Utah Tech", "UTE": "UTEP",
    "UTS": "UTSA", "VAL": "Valparaiso", "VAN": "Vanderbilt",
    "VCU": "VCU", "VIL": "Villanova", "VIR": "Virginia",
    "VMI": "VMI", "VTP": "Virginia Tech", "WAG": "Wagner",
    "WAK": "Wake Forest", "WAS": "Washington", "WAS2": "Washington State",
    "WEB": "Weber State", "WES": "West Virginia", "WES2": "Western Carolina",
    "WES3": "Western Kentucky", "WES4": "Western Michigan", "WIC": "Wichita State",
    "WIL": "William & Mary", "WIN": "Winthrop", "WIS": "Wisconsin",
    "WOF": "Wofford", "WRI": "Wright State", "XAV": "Xavier",
    "YAL": "Yale", "YOU": "Youngstown State",
}

def code_to_school(code):
    """Derive readable school name from team code like 'ALA_ANM'."""
    parts = code.split("_")
    if not parts:
        return code
    prefix = parts[0].upper()
    if prefix in SCHOOL_NAMES:
        return SCHOOL_NAMES[prefix]
    # Fallback: expand common patterns
    fallbacks = {
        "NC": "North Carolina", "SC": "South Carolina",
        "TEX": "Texas", "CAL": "California", "FLA": "Florida",
        "VIR": "Virginia", "OHI": "Ohio", "MIS": "Mississippi",
        "LOU": "Louisiana", "ARK": "Arkansas", "ALA": "Alabama",
        "GEO": "Georgia", "TEN": "Tennessee", "KEN": "Kentucky",
        "IND": "Indiana", "ILL": "Illinois", "IOW": "Iowa",
        "KAN": "Kansas", "ORE": "Oregon", "WAS": "Washington",
        "ARI": "Arizona", "UTA": "Utah", "NEV": "Nevada",
        "OKL": "Oklahoma", "NEB": "Nebraska", "MIN": "Minnesota",
        "WIS": "Wisconsin", "MIC": "Michigan", "PEN": "Pennsylvania",
        "NEW": "New Mexico", "HAW": "Hawaii", "MAS": "Massachusetts",
        "CON": "Connecticut", "RHO": "Rhode Island", "MAI": "Maine",
        "DEL": "Delaware", "MAR": "Maryland", "VER": "Vermont",
    }
    for abbr, full in fallbacks.items():
        if code.startswith(abbr + "_"):
            return full
    return code.replace("_", " ").title()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output CSV path")
    parser.add_argument("db_path", nargs="?", default=None,
                        help="SQLite database path (legacy positional, ignored — uses DuckDB)")
    parser.add_argument("output_path_pos", nargs="?", default=None,
                        help="Output CSV path (legacy positional)")
    args = parser.parse_args()

    # Legacy positional support: last positional arg overrides --output
    if args.output_path_pos:
        output_path = Path(args.output_path_pos)
    else:
        output_path = Path(args.output)

    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT DISTINCT PitcherTeam as team_code FROM pitches
        WHERE PitcherTeam IS NOT NULL AND TRIM(PitcherTeam) != ''
        UNION
        SELECT DISTINCT BatterTeam FROM pitches
        WHERE BatterTeam IS NOT NULL AND TRIM(BatterTeam) != ''
        ORDER BY team_code
    """).fetchdf()
    con.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing defaults for conference info
    existing = {}
    default_path = Path(__file__).resolve().parents[1] / "configs" / "default_team_mapping.csv"
    if default_path.exists():
        with default_path.open() as f:
            for row in csv.DictReader(f):
                existing[row["team_code"]] = row

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["team_code", "school_name", "conference"])
        writer.writeheader()
        for code in df["team_code"]:
            default = existing.get(code, {})
            school = default.get("school_name", "") or code_to_school(code)
            conf = default.get("conference", "")
            writer.writerow({"team_code": code, "school_name": school, "conference": conf})

    # Stats
    mapped = sum(1 for code in df["team_code"] if code.split("_")[0].upper() in SCHOOL_NAMES or code in existing)
    print(f"Wrote {len(df)} team codes → {output_path}")
    print(f"  {mapped} mapped to school names, {len(df) - mapped} using derived names")
    print(f"  Conference data: {sum(1 for v in existing.values() if v.get('conference',''))} teams")


if __name__ == "__main__":
    main()
