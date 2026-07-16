#!/usr/bin/env python3
"""Question 2: Check hitter data field coverage — raw vs enriched."""
import json

base = "/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model/data/training"

with open(f"{base}/projections_2026.json") as f:
    raw = json.load(f)
with open(f"{base}/projections_2026_enriched.json") as f:
    enr = json.load(f)

print("=== Raw projections keys ===")
print(sorted(raw[0].keys()))
print()
print("=== Enriched projections keys ===")
print(sorted(enr[0].keys()))
print()

# Check by position
for label, records in [("RAW HITTERS", [r for r in raw if r.get("player_type")=="hitter"]),
                        ("RAW PITCHERS", [r for r in raw if r.get("player_type")!="hitter"]),
                        ("ENR HITTERS", [r for r in enr if r.get("player_type")=="hitter"]),
                        ("ENR PITCHERS", [r for r in enr if r.get("player_type")!="hitter"])]:
    print(f"\n=== {label} ({len(records)} records) ===")
    for key in ["college_wOBA", "college_wRC_plus", "college_BB_pct", "college_K_pct",
                 "height_inches", "position", "conference_tier", "draftability_score",
                 "mlb_probability", "composite_score", "projected_pick", "age"]:
        filled = sum(1 for r in records if r.get(key) is not None and str(r.get(key,"")).strip() not in ("","None","0","0.0"))
        pct = 100 * filled / len(records)
        print(f"  {key:<22s} {pct:5.1f}% ({filled}/{len(records)})")

# Check FG training set — where did height/conference come from?
print("\n\n=== FG Training Set — source of features ===")
with open(f"{base}/fg_training_set.json") as f:
    fg = json.load(f)
h = [r for r in fg if r.get("player_type")=="hitter"]
p = [r for r in fg if r.get("player_type")=="pitcher"]
print(f"Hitters: {len(h)}, Pitchers: {len(p)}")
for feat in ["height", "weight", "bmi", "conference_tier", "conference", "draftability_score"]:
    h_filled = sum(1 for r in h if r.get(feat) is not None and str(r.get(feat,"")).strip() not in ("","None","0","0.0"))
    p_filled = sum(1 for r in p if r.get(feat) is not None and str(r.get(feat,"")).strip() not in ("","None","0","0.0"))
    print(f"  {feat:<20s} Hitters: {100*h_filled/len(h):5.1f}% ({h_filled}/{len(h)})  Pitchers: {100*p_filled/len(p):5.1f}% ({p_filled}/{len(p)})")

# Feature correlation with draft pick
print("\n\n=== Top features correlating with draft_pick ===")
h_features = ["fg_wOBA", "fg_wRC_plus", "fg_BB_pct", "fg_K_pct", "fg_AVG", "fg_SLG",
              "fg_ISO", "fg_BABIP", "height", "bmi", "conference_tier", "fg_Age"]
# Show feature range for hitters
for feat in h_features + ["draft_pick"]:
    vals = [float(r[feat]) for r in h if r.get(feat) is not None and str(r.get(feat,"")).strip() not in ("","None")]
    if vals:
        print(f"  {feat:<20s} min={min(vals):.2f}  max={max(vals):.2f}  mean={sum(vals)/len(vals):.2f}  n={len(vals)}")
