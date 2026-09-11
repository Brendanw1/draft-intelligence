#!/usr/bin/env python3
"""
evaluate_2026_prospective.py — True prospective evaluation of the MLB Draft Model.
V2 with improved matching.

Matches model projections (trained on ≤2025 data) against actual 2026 draft results.
Generates accuracy metrics and updates the prospective report.
"""

import json
import math
import re
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

BASE = '/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model'

# ── Load data ──────────────────────────────────────────────────────

print("Loading data...")

with open(f'{BASE}/data/draft/draft_2026.json') as f:
    draft_picks = json.load(f)

with open(f'{BASE}/data/training/projections_2026_enriched.json') as f:
    projections = json.load(f)

with open(f'{BASE}/data/draft/draft_all_picks.json') as f:
    all_picks = json.load(f)

# ── Build crosswalk: team_abb -> school_name ─────────────────────

def normalize_school(name):
    """Normalize school name for matching."""
    if not name:
        return ""
    name = name.strip().upper()
    name = re.sub(r'\bUNIVERSITY\b', 'U', name)
    name = re.sub(r'\bSTATE\b', 'ST', name)
    name = re.sub(r'\bOF\b', '', name)
    name = re.sub(r'\bTHE\b', '', name)
    name = re.sub(r'\bAT\b', '', name)
    name = re.sub(r'\b-\b', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.strip()

def normalize_name(name):
    """Normalize player name for matching."""
    if not name:
        return ""
    name = name.strip().lower()
    name = re.sub(r'[^a-z\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# Build comprehensive team_abb -> school_name crosswalk
team_abb_to_school = {
    # Auto-generated from common patterns
    'UCLA': 'UCLA', 'UCSB': 'UC Santa Barbara', 'UCI': 'UC Irvine',
    'UCR': 'UC Riverside', 'UCSD': 'UC San Diego',
    'LSU': 'LSU', 'ARK': 'Arkansas', 'TEX': 'Texas', 'TAMU': 'Texas A&M',
    'TA&M': 'Texas A&M', 'ATM': 'Texas A&M',
    'MISS': 'Mississippi', 'MSST': 'Mississippi State',
    'MIZ': 'Missouri', 'MIZZ': 'Missouri',
    'AUB': 'Auburn', 'ALA': 'Alabama', 'UGA': 'Georgia',
    'SCAR': 'South Carolina', 'FLA': 'Florida', 'FSU': 'Florida State',
    'MIA': 'Miami', 'UNC': 'North Carolina', 'NCSU': 'North Carolina State',
    'NCST': 'North Carolina State',
    'CLEM': 'Clemson', 'VT': 'Virginia Tech', 'VATECH': 'Virginia Tech',
    'UVA': 'Virginia', 'DUKE': 'Duke', 'WAKE': 'Wake Forest',
    'GT': 'Georgia Tech', 'GATECH': 'Georgia Tech',
    'LOU': 'Louisville', 'UK': 'Kentucky', 'TENN': 'Tennessee',
    'VANDY': 'Vanderbilt', 'USC': 'Southern California',
    'SOCAL': 'Southern California', 'STAN': 'Stanford',
    'ORE': 'Oregon', 'OSU': 'Oregon State', 'OREST': 'Oregon State',
    'WSU': 'Washington State', 'WASH': 'Washington',
    'CAL': 'California', 'ARIZ': 'Arizona', 'ASU': 'Arizona State',
    'TCU': 'Texas Christian', 'TTU': 'Texas Tech', 'TXTECH': 'Texas Tech',
    'BAY': 'Baylor', 'KSU': 'Kansas State', 'KU': 'Kansas',
    'OKLA': 'Oklahoma', 'OU': 'Oklahoma', 'OKST': 'Oklahoma State',
    'OSU': 'Oklahoma State',  # careful conflict with Oregon State
    'WVU': 'West Virginia',
    'ILL': 'Illinois', 'NW': 'Northwestern', 'NU': 'Northwestern',
    'PUR': 'Purdue', 'IND': 'Indiana', 'IU': 'Indiana',
    'MICH': 'Michigan', 'UMICH': 'Michigan', 'MSU': 'Michigan State',
    'IOWA': 'Iowa', 'NEB': 'Nebraska', 'MINN': 'Minnesota',
    'WISC': 'Wisconsin', 'RUT': 'Rutgers', 'MD': 'Maryland',
    'PENN': 'Penn State', 'PSU': 'Penn State', 'BC': 'Boston College',
    'ND': 'Notre Dame', 'NDAME': 'Notre Dame', 'CUSE': 'Syracuse',
    'PITT': 'Pittsburgh', 'ULM': 'Louisiana Monroe',
    'ULL': 'Louisiana', 'USA': 'South Alabama',
    'USM': 'Southern Mississippi', 'GASO': 'Georgia Southern',
    'TROY': 'Troy', 'TXST': 'Texas State',
    'ECU': 'East Carolina', 'UCF': 'Central Florida',
    'USF': 'South Florida', 'MEM': 'Memphis', 'TULN': 'Tulane',
    'HOU': 'Houston', 'CIN': 'Cincinnati', 'UCONN': 'Connecticut',
    'WICH': 'Wichita State', 'WSU': 'Wichita State',
    'DBU': 'Dallas Baptist', 'GCU': 'Grand Canyon',
    'UNM': 'New Mexico', 'NMSU': 'New Mexico State',
    'RICE': 'Rice', 'CHAR': 'Charlotte', 'ODU': 'Old Dominion',
    'JMU': 'James Madison', 'LIB': 'Liberty',
    'VCU': 'Virginia Commonwealth', 'CAMP': 'Campbell',
    'NE': 'Northeastern', 'NU': 'Northeastern',
    'HOF': 'Hofstra', 'DEL': 'Delaware',
    'W&M': 'William & Mary', 'RICH': 'Richmond',
    'SHU': 'Seton Hall', 'DAY': 'Dayton',
    'SLU': 'Saint Louis', 'URI': 'Rhode Island',
    'UMASS': 'Massachusetts', 'BRY': 'Bryant',
    'MERC': 'Mercer', 'SAM': 'Samford',
    'HPU': 'High Point', 'WIN': 'Winthrop',
    'JAX': 'Jacksonville', 'UNF': 'North Florida',
    'KSU': 'Kennesaw State', 'ETSU': 'East Tennessee State',
    'BEL': 'Belmont', 'MUR': 'Murray State',
    'SIU': 'Southern Illinois', 'SIUE': 'Southern Illinois Edwardsville',
    'INDST': 'Indiana State', 'INST': 'Indiana State',
    'ILST': 'Illinois State', 'ILLST': 'Illinois State',
    'MOSU': 'Missouri State', 'MOST': 'Missouri State',
    'DRKE': 'Drake', 'EVAN': 'Evansville', 'VALP': 'Valparaiso',
    'BRAD': 'Bradley', 'NIU': 'Northern Illinois',
    'BALL': 'Ball State', 'WMU': 'Western Michigan',
    'CMU': 'Central Michigan', 'EMU': 'Eastern Michigan',
    'TOL': 'Toledo', 'BGSU': 'Bowling Green',
    'MIAOH': 'Miami (Ohio)', 'AKR': 'Akron', 'KENT': 'Kent State',
    'UAB': 'Alabama Birmingham', 'MTSU': 'Middle Tennessee',
    'FAU': 'Florida Atlantic', 'FIU': 'Florida International',
    'WKU': 'Western Kentucky', 'LATECH': 'Louisiana Tech',
    'LT': 'Louisiana Tech', 'UNT': 'North Texas',
    'UTEP': 'Texas El Paso', 'SDSU': 'San Diego State',
    'FRES': 'Fresno State', 'SJSU': 'San Jose State',
    'UNLV': 'Nevada Las Vegas', 'NEV': 'Nevada',
    'HAW': "Hawai'i", 'CSUN': 'Cal State Northridge',
    'LBSU': 'Long Beach State', 'CSUF': 'Cal State Fullerton',
    'CP': 'Cal Poly', 'SLO': 'Cal Poly',
    'USD': 'San Diego', 'PEP': 'Pepperdine', 'LMU': 'Loyola Marymount',
    'BYU': 'Brigham Young', 'CREI': 'Creighton', 'XU': 'Xavier',
    'XAV': 'Xavier', 'GONZ': 'Gonzaga', 'PORT': 'Portland',
    'SANFRAN': 'San Francisco', 'USF': 'San Francisco',
    'SCU': 'Santa Clara', 'STMA': "Saint Mary's",
    'PAC': 'Pacific', 'PEAY': 'Austin Peay',
    'APSU': 'Austin Peay', 'TNST': 'Tennessee State',
    'TNTC': 'Tennessee Tech', 'EKY': 'Eastern Kentucky',
    'JSU': 'Jacksonville State', 'JVST': 'Jacksonville State',
    'NICH': 'Nicholls', 'NWST': 'Northwestern State',
    'MCN': 'McNeese', 'SELA': 'Southeastern Louisiana',
    'LAMAR': 'Lamar', 'DAL': 'Dallas Baptist',
    'MRSH': 'Marshall', 'UNCA': 'North Carolina Asheville',
    'WEBB': 'Gardner-Webb', 'GWU': 'Gardner-Webb',
    'RAD': 'Radford', 'LONG': 'Longwood',
    'CSU': 'Charleston Southern', 'PRES': 'Presbyterian',
    'USCU': 'South Carolina Upstate',
    'BING': 'Binghamton', 'ALB': 'Albany',
    'SB': 'Stony Brook', 'STONY': 'Stony Brook',
    'CCSU': 'Central Connecticut', 'WAG': 'Wagner',
    'FDU': 'Fairleigh Dickinson', 'LIU': 'Long Island',
    'SAC': 'Sacred Heart', 'FUR': 'Furman',
    'WOF': 'Wofford', 'GSU': 'Georgia State',
    'ARST': 'Arkansas State', 'LA': 'Louisiana',
    'UL': 'Louisiana',  # Louisiana not Louisville in context
    'TUL': 'Tulane', 'UH': 'Houston',
    'UC': 'Cincinnati', 'UConn': 'Connecticut',
    'WICHST': 'Wichita State', 'DREX': 'Drexel',
    'LAS': "La Salle", 'STJO': "Saint Joseph's",
    'SPFD': 'Seton Hall', 'MASS': 'Massachusetts',
    'UML': 'Massachusetts Lowell', 'MAIN': 'Maine',
    'UNH': 'New Hampshire', 'UVM': 'Vermont',
    'TOWS': 'Towson', 'UD': 'Delaware',
    'WM': 'William & Mary', 'BU': 'Baylor',
    'KSTATE': 'Kansas State', 'UIUC': 'Illinois',
    'PU': 'Purdue', 'OHST': 'Ohio State',
    'UNL': 'Nebraska', 'UM': 'Minnesota',
    'UW': 'Washington',  # careful: could be Washington or Wisconsin
    'RU': 'Rutgers', 'UMD': 'Maryland',
    'SU': 'Stanford',  # could also be Syracuse
    'UP': 'Pittsburgh',
    'USOUTHM': 'Southern Mississippi',
    'TXST': 'Texas State',
    'CCU': 'Coastal Carolina',
    'UAPB': 'Arkansas Pine Bluff',
    'PV': 'Prairie View',
    'TSU': 'Texas Southern',
    'JKST': 'Jackson State',
    'ALST': 'Alabama State',
    'ALCN': 'Alcorn State',
    'GRAM': 'Grambling',
    'SOU': 'Southern',
    'COOK': 'Bethune-Cookman',
    'FAMU': 'Florida A&M',
    'NCCU': 'North Carolina Central',
    'NCAT': 'North Carolina A&T',
    'SCST': 'South Carolina State',
    'DSU': 'Delaware State',
    'ORU': 'Oral Roberts',
    'NEB': 'Nebraska',
    'UNO': 'New Orleans',
    'NSU': 'Norfolk State',
    'COPP': 'Coppin State',
    'CSB': 'Cal State Bakersfield',
    'UCD': 'UC Davis',
    'UOP': 'Pacific',
    'SMC': "Saint Mary's",
    'GU': 'Gonzaga',
    'CU': 'Creighton',
    'M-OH': 'Miami (Ohio)',
    'BSU': 'Ball State',
    'BGS': 'Bowling Green',
    'KSU': 'Kent State',
    'MT': 'Middle Tennessee',
    'UE': 'Evansville',
    'INST': 'Indiana State',
    'LBS': 'Long Beach State',
    'CSF': 'Cal State Fullerton',
    'UH': "Hawai'i",  # conflict, but contextually for Hawaii
    'UTRGV': 'Texas Rio Grande Valley',
    'TAMUCC': 'Texas A&M Corpus Christi',
    'AMCC': 'Texas A&M Corpus Christi',
    'HCU': 'Houston Christian',
    'SFA': 'Stephen F. Austin',
    'NWLA': 'Northwestern State',
    'MCNEESE': 'McNeese',
    'UIW': 'Incarnate Word',
    'UNI': 'Northern Iowa',
    'DAV': 'Davidson',
    'CIT': 'The Citadel',
    'VMI': 'Virginia Military Institute',
    'CHSO': 'Charleston Southern',
    'MOR': 'Morehead State',
    'WVST': 'West Virginia State',
    'SAL': 'South Alabama',
    'GSU': 'Georgia State',
    'TROY': 'Troy',
    'APP': 'Appalachian State',
    'LR': 'Little Rock',
    'UTA': 'Texas Arlington',
    'SFA': 'Stephen F. Austin',
    'NWST': 'Northwestern State',
    'UNO': 'New Orleans',
    'HBU': 'Houston Baptist',
    'NICH': 'Nicholls',
    'GRAM': 'Grambling',
    'UAPB': 'Arkansas Pine Bluff',
    'TXSO': 'Texas Southern',
    'ALCN': 'Alcorn State',
    'JKST': 'Jackson State',
    'ALST': 'Alabama State',
    'ALCN': 'Alcorn State',
    'WEBB': 'Gardner-Webb',
    'UNCG': 'North Carolina Greensboro',
    'UNCW': 'North Carolina Wilmington',
    'ELON': 'Elon',
    'COFC': 'College of Charleston',
    'TOWS': 'Towson',
    'NE': 'Northeastern',
    'HOF': 'Hofstra',
    'MONM': 'Monmouth',
    'SHU': 'Seton Hall',
    'RID': 'Rider',
    'NIAG': 'Niagara',
    'CAN': 'Canisius',
    'MAN': 'Manhattan',
    'IONA': 'Iona',
    'QUIN': 'Quinnipiac',
    'FAIR': 'Fairfield',
    'SAC': 'Sacred Heart',
    'MSM': 'Mount Saint Mary',
    'L-MD': 'Loyola Maryland',
    'NAVY': 'Navy',
    'ARMY': 'Army',
    'AF': 'Air Force',
    'USCGA': 'Coast Guard',
    'RPI': 'Rensselaer',
    'CLRK': 'Clark Atlanta',
    'BEN': 'Benedict',
    'LEM': 'Le Moyne',
    'STONE': 'Stonehill',
    'MER': 'Merrimack',
    'LIP': 'Lipscomb',
    'BELL': 'Bellarmine',
    'QNC': 'Queens',
    'EKY': 'Eastern Kentucky',
    'CARK': 'Central Arkansas',
    'MORE': 'Morehead State',
}

# ── Match projections to draft picks ──────────────────────────────

# Index projections
proj_by_name = {}  # (normalized_name, normalized_team_abb) -> projection
proj_by_name_only = defaultdict(list)  # normalized_name -> [projections]
proj_by_mlbamid = {}

for p in projections:
    pname = p.get('player_name', '')
    pteam = p.get('team_abb', '')
    nname = normalize_name(pname)
    nteam = normalize_school(pteam)
    
    key = (nname, nteam)
    if key not in proj_by_name:
        proj_by_name[key] = p
    
    if nname:
        proj_by_name_only[nname].append(p)
    
    mid = p.get('xMLBAMID')
    if mid is not None and str(mid).strip():
        proj_by_mlbamid[str(mid)] = p

print(f"Indexed {len(proj_by_name)} unique (name, school) combos")
print(f"Indexed {len(proj_by_name_only)} unique names")
print(f"Indexed {len(proj_by_mlbamid)} MLBAM IDs")

# Match
matched = []
unmatched = []

for pick in draft_picks:
    person_id = str(pick.get('person_id', ''))
    full_name = pick.get('full_name', '')
    school = pick.get('school_name', '') or ''
    pick_number = pick.get('pick_number', 0)
    pick_round = pick.get('pick_round', 0)
    pos_abbr = pick.get('position_abbr', '')
    school_class = pick.get('school_class', '') or ''
    
    is_hs = 'HS' in school_class
    is_college = 'YR' in school_class or 'GR' in school_class
    
    projection = None
    match_type = None
    
    # Strategy 1: By MLBAMID
    if person_id and person_id in proj_by_mlbamid:
        projection = proj_by_mlbamid[person_id]
        match_type = 'mlbamid'
    
    # Strategy 2: By normalized name + school
    if projection is None:
        nname = normalize_name(full_name)
        nschool = normalize_school(school)
        key = (nname, nschool)
        
        if key in proj_by_name:
            projection = proj_by_name[key]
            match_type = 'name_school'
        else:
            # Try via crosswalk
            nschool_via_crosswalk = None
            for ta, sn in team_abb_to_school.items():
                sn_norm = normalize_school(sn)
                if sn_norm == nschool or sn_norm in nschool or nschool in sn_norm:
                    alt_key = (nname, normalize_school(ta))
                    if alt_key in proj_by_name:
                        projection = proj_by_name[alt_key]
                        match_type = 'crosswalk'
                        break
    
    # Strategy 3: Name-only with school disambiguation
    if projection is None:
        nname = normalize_name(full_name)
        candidates = proj_by_name_only.get(nname, [])
        if len(candidates) == 1:
            projection = candidates[0]
            match_type = 'name_only'
        elif len(candidates) > 1:
            # Pick the one with most similar school name
            nschool = normalize_school(school)
            best_score = 0
            best_cand = None
            for c in candidates:
                c_team = normalize_school(c.get('team_abb', ''))
                # Count overlapping school name characters
                overlap = len(set(nschool) & set(c_team))
                if overlap > best_score:
                    best_score = overlap
                    best_cand = c
            if best_cand and best_score > 0:
                projection = best_cand
                match_type = 'fuzzy_name_school'
    
    # Strategy 4: Name-only regardless (last resort for college players)
    if projection is None and is_college:
        nname = normalize_name(full_name)
        candidates = proj_by_name_only.get(nname, [])
        if candidates:
            # Take the first one
            projection = candidates[0]
            match_type = 'name_only_last_resort'
    
    if projection is not None:
        proj_pick = projection.get('projected_pick')
        if proj_pick is None:
            proj_pick = 500
        
        delta = pick_number - proj_pick
        
        matched.append({
            'pick': pick,
            'projection': projection,
            'delta': delta,
            'match_type': match_type,
            'is_college': is_college,
            'is_hs': is_hs,
            'actual_pick': pick_number,
            'proj_pick': proj_pick,
            'proj_round': projection.get('projected_round', 0),
            'player_type': projection.get('player_type', 'unknown'),
        })
    else:
        unmatched.append({
            'pick': pick,
            'is_college': is_college,
            'is_hs': is_hs,
        })

# ── Compute metrics ──────────────────────────────────────────────

total_picks = len(draft_picks)
total_college = sum(1 for r in draft_picks if 'YR' in (r.get('school_class','') or '') or 'GR' in (r.get('school_class','') or ''))
total_hs = sum(1 for r in draft_picks if 'HS' in (r.get('school_class','') or ''))

college_matched = [m for m in matched if m['is_college']]
college_unmatched = [m for m in unmatched if m['is_college']]

print(f"\nTotal picks: {total_picks} (College: {total_college}, HS: {total_hs})")
print(f"College matched: {len(college_matched)}/{total_college}")
print(f"College unmatched: {len(college_unmatched)}")

match_types = defaultdict(int)
for m in college_matched:
    match_types[m['match_type']] += 1
print(f"Match types: {dict(match_types)}")

if college_matched:
    deltas = [m['delta'] for m in college_matched]
    abs_deltas = [abs(d) for d in deltas]
    
    mae = float(np.mean(abs_deltas))
    medae = float(np.median(abs_deltas))
    rmse = float(np.sqrt(np.mean(np.array(deltas)**2)))
    
    # By player type
    hit_mae = float(np.mean([abs(m['delta']) for m in college_matched if m.get('player_type') == 'hitter']))
    pit_mae = float(np.mean([abs(m['delta']) for m in college_matched if m.get('player_type') == 'pitcher']))
    hit_count = len([m for m in college_matched if m.get('player_type') == 'hitter'])
    pit_count = len([m for m in college_matched if m.get('player_type') == 'pitcher'])
    
    # Direction
    higher = sum(1 for m in college_matched if m['delta'] > 0)
    lower = sum(1 for m in college_matched if m['delta'] < 0)
    exact = sum(1 for m in college_matched if m['delta'] == 0)
    
    # Within bands
    bands = {50: 0, 75: 0, 100: 0, 110: 0, 150: 0, 200: 0}
    for m in college_matched:
        ad = abs(m['delta'])
        for b in bands:
            if ad <= b:
                bands[b] += 1
    
    print(f"\nMAE: {mae:.1f}")
    print(f"Median AE: {medae:.1f}")
    print(f"RMSE: {rmse:.1f}")
    print(f"Hitter MAE: {hit_mae:.1f} ({hit_count})")
    print(f"Pitcher MAE: {pit_mae:.1f} ({pit_count})")
    for b in sorted(bands.keys()):
        print(f"Within ±{b}: {bands[b]}/{len(college_matched)} ({bands[b]/len(college_matched)*100:.1f}%)")
    
    # By round
    round_groups = defaultdict(list)
    for m in college_matched:
        round_groups[m['pick']['pick_round']].append(m)
    
    round_table = []
    for rnd in sorted(round_groups.keys()):
        group = round_groups[rnd]
        grp_mae = np.mean([abs(m['delta']) for m in group])
        in_range = sum(1 for m in group if abs(m['delta']) <= 110)
        higher_r = sum(1 for m in group if m['delta'] > 110)
        lower_r = sum(1 for m in group if m['delta'] < -110)
        hit_rate = in_range / len(group) * 100
        round_table.append({
            'round': rnd, 'count': len(group), 'in_range': in_range,
            'higher': higher_r, 'lower': lower_r, 'hit_rate': hit_rate, 'mae': grp_mae,
        })
    
    # Spearman
    actual_ranks = [m['actual_pick'] for m in college_matched]
    predicted_ranks = [m['proj_pick'] for m in college_matched]
    spearman_corr, spearman_p = spearmanr(actual_ranks, predicted_ranks)
    
    print(f"\nSpearman ρ: {spearman_corr:.3f} (p={spearman_p:.6f})")
    print(f"Within ±110: {bands[110]}/{len(college_matched)} ({bands[110]/len(college_matched)*100:.0f}%)")
    print(f"Higher: {higher} ({higher/len(college_matched)*100:.0f}%)")
    print(f"Lower: {lower} ({lower/len(college_matched)*100:.0f}%)")

# ── Build detailed output ────────────────────────────────────────

matched_details = []
for m in college_matched:
    pick = m['pick']
    matched_details.append({
        'pick': pick['pick_number'],
        'round': pick['pick_round'],
        'player': pick['full_name'],
        'school': pick['school_name'],
        'school_abb': m['projection'].get('team_abb', ''),
        'pos': pick.get('position_abbr', ''),
        'proj_pick': round(m['proj_pick'], 1),
        'proj_round': m['proj_round'],
        'delta': round(m['delta'], 1),
        'match_type': m['match_type'],
        'player_type': m['player_type'],
        'grade': m['projection'].get('value_grade', 'unknown'),
    })

output = {
    'timestamp': '2026-07-16',
    'total_college': total_college,
    'total_hs': total_hs,
    'total_matched': len(college_matched),
    'total_unmatched': len(college_unmatched),
    'mae': mae,
    'medae': medae,
    'rmse': rmse,
    'hitter_mae': hit_mae,
    'pitcher_mae': pit_mae,
    'hitter_count': hit_count,
    'pitcher_count': pit_count,
    'within_110': bands[110],
    'within_110_pct': bands[110]/len(college_matched)*100,
    'within_50': bands[50],
    'within_50_pct': bands[50]/len(college_matched)*100,
    'within_100': bands[100],
    'within_100_pct': bands[100]/len(college_matched)*100,
    'within_150': bands[150],
    'within_150_pct': bands[150]/len(college_matched)*100,
    'within_200': bands[200],
    'within_200_pct': bands[200]/len(college_matched)*100,
    'higher': higher,
    'higher_pct': higher/len(college_matched)*100,
    'lower': lower,
    'lower_pct': lower/len(college_matched)*100,
    'exact': exact,
    'spearman_r': spearman_corr,
    'spearman_p': spearman_p,
    'rounds': round_table,
    'match_types': dict(match_types),
    'matched_details': sorted(matched_details, key=lambda x: x['pick']),
    'unmatched_schools': sorted(list(set(
        m['pick'].get('school_name', '') for m in unmatched if m['is_college']
    ))),
    'unmatched_count_by_school': dict(sorted(
        [(s, sum(1 for m in unmatched if m['is_college'] and m['pick'].get('school_name', '') == s))
         for s in set(m['pick'].get('school_name', '') for m in unmatched if m['is_college'])],
        key=lambda x: -x[1]
    )[:30]),
}

with open(f'{BASE}/analysis/eval_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved to analysis/eval_results.json")
print(f"Done.")
