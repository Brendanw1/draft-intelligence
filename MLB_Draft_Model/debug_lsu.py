import json, re, sys
from difflib import SequenceMatcher

def normalize_name(name):
    if not name: return ""
    name = name.lower().strip()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"[^a-z\-\s]", "", name)
    return name

rosters = json.load(open("data/rosters/d1_rosters_2026.json"))
crosswalk = json.load(open("data/rosters/fg_to_roster_crosswalk.json"))
lsu_info = crosswalk.get("crosswalk", {}).get("LSU", {})
roster_team = lsu_info.get("roster_name", "")
team_key = normalize_name(roster_team)
print(f"LSU team_key: {team_key}")

norm_jake = normalize_name("Jake Brown")
print(f"norm_jake: '{norm_jake}'")

for p in rosters:
    tk = normalize_name(p.get("team_name", ""))
    pk = normalize_name(p.get("player_name", "") or p.get("full_name", ""))
    if tk == team_key and "brown" in pk:
        score = SequenceMatcher(None, norm_jake, pk).ratio()
        print(f"  pk='{pk}' score={score:.3f} h={p.get('height')} name={p.get('player_name')}")

print(f"\nTotal LSU roster entries: {sum(1 for p in rosters if normalize_name(p.get('team_name','')) == team_key)}")
