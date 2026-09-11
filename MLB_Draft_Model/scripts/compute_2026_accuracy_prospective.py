#!/usr/bin/env python3
"""
compute_2026_accuracy_prospective.py — Compute accuracy of prospective 2026 predictions
against actual 2026 draft outcomes. Generates analysis/2026_draft_accuracy_prospective.md.

This runs AFTER inference has been done on the filtered (no-2026-in-training) model.
"""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1]

# ── Load data ─────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def safe_float(v):
    if v is None: return None
    try: return float(v)
    except: return None

# 1. Enriched projections (model outputs for 2026 prospects)
enriched_path = BASE / "data" / "training" / "projections_2026_enriched.json"
enriched = load_json(enriched_path)
print(f"Loaded enriched projections: {len(enriched)}")

# 2. Actual 2026 draft results
draft_2026 = load_json(BASE / "data" / "draft" / "draft_2026.json")
print(f"Loaded 2026 draft picks: {len(draft_2026)}")

# ── Build lookup: normalised name|school -> actual pick info ─────────

# School_class filtering for college-only
COLLEGE_CLASSES = {"4YR JR", "4YR SR", "4YR SO", "4YR GR", "4YR 5S", "3YR SO", "3YR JR", "3YR SR"}

def normalize(s: str) -> str:
    """Normalize a string for matching."""
    if not s: return ""
    return s.strip().lower().replace("'", "").replace(".", "").replace("-", " ")

def build_actual_2026_lookup(draft_records):
    """Build lookup dict: normalized(name|school_abb) -> draft record."""
    lookup = {}
    for p in draft_records:
        name = normalize(p.get("full_name", ""))
        school = normalize(p.get("school_name", ""))
        school_class = p.get("school_class", "")
        
        # Only include college players
        if school_class not in COLLEGE_CLASSES:
            continue
        
        # Also try without "university" suffix for matching
        school_alt = school.replace("university", "").strip()
        
        key = f"{name}|{school}"
        if key not in lookup:
            lookup[key] = p
        
        # Also index by school_alt
        if school_alt:
            key_alt = f"{name}|{school_alt}"
            if key_alt not in lookup:
                lookup[key_alt] = p
    
    return lookup

actual_lookup = build_actual_2026_lookup(draft_2026)
print(f"College draft picks (actual): {len(set(p.get('pick_number') for p in draft_2026 if p.get('school_class') in COLLEGE_CLASSES))}")

# Count college picks
college_picks = [p for p in draft_2026 if p.get("school_class") in COLLEGE_CLASSES]
print(f"College picks: {len(college_picks)}")
print(f"All picks (incl HS): {len(draft_2026)}")

# ── Match enriched projections to actual draft picks ──────────────

PICK_ROUND_BOUNDARIES = [
    40, 70, 107, 137, 167, 197, 227, 257, 287, 317,
    347, 377, 407, 437, 467, 497, 527, 557, 587, 617,
]

def pick_to_round(pick):
    if pick is None: return None
    for i, boundary in enumerate(PICK_ROUND_BOUNDARIES):
        if pick <= boundary:
            return max(1, min(20, i + 1))
    return 20

# MAE for range band (from Tier 1 training on filtered data)
HITTER_MAE = 108.6  # Will be updated by value from training output
PITCHER_MAE = 111.6

matches = []  # (projection, actual_pick, delta)
unmatched_drafted = []
unmatched_projected = []

for proj in enriched:
    name = normalize(proj.get("player_name", ""))
    team = normalize(proj.get("team_abb", ""))
    school = normalize(proj.get("team_name", ""))
    ptype = proj.get("player_type", "hitter")
    
    # Try matching by name|team_abb first, then name|school
    key = f"{name}|{team}"
    actual = actual_lookup.get(key)
    
    if not actual and school:
        key2 = f"{name}|{school}"
        actual = actual_lookup.get(key2)
    
    if not actual:
        # Also try just school without "University"
        school_alt = school.replace("university", "").strip()
        if school_alt:
            key3 = f"{name}|{school_alt}"
            actual = actual_lookup.get(key3)
    
    if actual:
        actual_pick = actual.get("pick_number")
        proj_pick = safe_float(proj.get("projected_pick"))
        
        if proj_pick and actual_pick:
            delta = proj_pick - actual_pick
            mae = HITTER_MAE if ptype == "hitter" else PITCHER_MAE
            
            # Determine status
            if abs(delta) <= mae:
                status = "✓"
            elif delta > 0:
                status = "⬆ HIGHER"
            else:
                status = "⬇ LOWER"
            
            matches.append({
                "player": proj.get("player_name", ""),
                "school": proj.get("team_name", ""),
                "pos": actual.get("position_abbr", ""),
                "pick": actual_pick,
                "round": actual.get("pick_round", pick_to_round(actual_pick)),
                "proj_pick": round(proj_pick, 1),
                "proj_round": proj.get("projected_round", pick_to_round(proj_pick)),
                "delta": round(delta),
                "status": status,
                "grade": proj.get("value_grade", "low"),
                "ptype": ptype,
                "mlb_prob": safe_float(proj.get("mlb_prob_isotonic")) or safe_float(proj.get("mlb_probability")) or 0,
                "composite": safe_float(proj.get("composite_score")) or 0,
            })
        else:
            unmatched_drafted.append({"player": proj.get("player_name", ""), "school": team, "pick": actual_pick})
    else:
        unmatched_projected.append(proj)

# Find drafted players that weren't in the projection set
all_matched_keys = set()
for proj in enriched:
    name = normalize(proj.get("player_name", ""))
    team = normalize(proj.get("team_abb", ""))
    key = f"{name}|{team}"
    if key in actual_lookup:
        all_matched_keys.add(key)
    else:
        school = normalize(proj.get("team_name", ""))
        key2 = f"{name}|{school}"
        if key2 in actual_lookup:
            all_matched_keys.add(key2)

drafted_not_in_model = []
for p in college_picks:
    name = normalize(p.get("full_name", ""))
    school = normalize(p.get("school_name", ""))
    key = f"{name}|{school}" if school else f"{name}|"
    if key not in all_matched_keys:
        drafted_not_in_model.append(p)

# ── Compute overall statistics ──────────────────────────────────

total_college_drafted = len(college_picks)
total_matched = len(matches)
total_unmatched = len(drafted_not_in_model)

within_range = sum(1 for m in matches if m["status"] == "✓")
drafted_higher = sum(1 for m in matches if m["status"] == "⬆ HIGHER")
drafted_lower = sum(1 for m in matches if m["status"] == "⬇ LOWER")

# Top-line stats
print(f"\n{'='*60}")
print("ACCURACY REPORT (Prospective — No 2026 in Training)")
print(f"{'='*60}")
print(f"Total college players drafted: {total_college_drafted}")
print(f"Matched to model projections: {total_matched} ({100*total_matched/total_college_drafted:.1f}%)")
print(f"Unmatched (not in model): {total_unmatched}")
print(f"\nRange accuracy:")
print(f"  Within projected range: {within_range} ({100*within_range/total_matched:.1f}%)")
print(f"  Drafted HIGHER than projected: {drafted_higher} ({100*drafted_higher/total_matched:.1f}%)")
print(f"  Drafted LOWER than projected: {drafted_lower} ({100*drafted_lower/total_matched:.1f}%)")

# Mean absolute delta
mean_delta = sum(abs(m["delta"]) for m in matches) / len(matches) if matches else 0
median_delta = sorted([abs(m["delta"]) for m in matches])[len(matches)//2] if matches else 0
print(f"\n  Mean |Δ|: {mean_delta:.1f}")
print(f"  Median |Δ|: {median_delta:.1f}")

# ── Accuracy by round ──────────────────────────────────────────

round_stats = defaultdict(lambda: {"total": 0, "in_range": 0, "higher": 0, "lower": 0, "deltas": []})
for m in matches:
    rnd = m["round"]
    if rnd is None:
        rnd = pick_to_round(m["pick"])
    round_stats[rnd]["total"] += 1
    round_stats[rnd]["deltas"].append(m["delta"])
    if m["status"] == "✓":
        round_stats[rnd]["in_range"] += 1
    elif m["status"] == "⬆ HIGHER":
        round_stats[rnd]["higher"] += 1
    elif m["status"] == "⬇ LOWER":
        round_stats[rnd]["lower"] += 1

# ── Generate Report ────────────────────────────────────────────

def fmt_pct(v, total):
    if total == 0: return "0%"
    return f"{100*v//total}%"

lines = []

lines.append(f"# 2026 MLB Draft — PROSPECTIVE Model Accuracy Report")
lines.append(f"")
lines.append(f"**Generated**: Training data filtered to exclude 2026 outcomes. Models retrained")
lines.append(f"from scratch on ≤2025 data only. This is a TRUE prospective test — the model")
lines.append(f"had never seen any 2026 draft outcomes during training.")
lines.append(f"")
lines.append(f"## Summary")
lines.append(f"")
lines.append(f"| Metric | Value |")
lines.append(f"|--------|-------|")
lines.append(f"| Total college players drafted | {total_college_drafted} |")
lines.append(f"| Matched to model projections | {total_matched} ({100*total_matched//total_college_drafted}%) |")
lines.append(f"| Unmatched (not in FanGraphs/model) | {total_unmatched} |")
lines.append(f"| Mean absolute delta (|Δ|) | {mean_delta:.1f} picks |")
lines.append(f"| Median absolute delta | {median_delta:.1f} picks |")

# MAE breakdown by type
hitter_matches = [m for m in matches if m["ptype"] == "hitter"]
pitcher_matches = [m for m in matches if m["ptype"] == "pitcher"]
if hitter_matches:
    h_mae = sum(abs(m["delta"]) for m in hitter_matches) / len(hitter_matches)
    lines.append(f"| Hitter MAE | {h_mae:.1f} picks ({len(hitter_matches)} players) |")
if pitcher_matches:
    p_mae = sum(abs(m["delta"]) for m in pitcher_matches) / len(pitcher_matches)
    lines.append(f"| Pitcher MAE | {p_mae:.1f} picks ({len(pitcher_matches)} players) |")

lines.append(f"")
lines.append(f"### Accuracy (within MAE band)")
lines.append(f"")
lines.append(f"| Metric | Count | % |")
lines.append(f"|--------|-------|---|")
lines.append(f"| ✓ Within projected range | {within_range} | {100*within_range//total_matched}% |")
lines.append(f"| ⬆ Drafted HIGHER than projected | {drafted_higher} | {100*drafted_higher//total_matched}% |")
lines.append(f"| ⬇ Drafted LOWER than projected | {drafted_lower} | {100*drafted_lower//total_matched}% |")

lines.append(f"")
lines.append(f"### Accuracy by Round")
lines.append(f"")
lines.append(f"| Round | Picks | In Range | Higher | Lower | Hit Rate | Mean |Δ| |")
lines.append(f"|-------|-------|----------|--------|-------|----------|---------|")
for rnd in sorted(round_stats.keys()):
    rs = round_stats[rnd]
    mean_abs = sum(abs(d) for d in rs["deltas"]) / len(rs["deltas"]) if rs["deltas"] else 0
    hit_rate = 100 * rs["in_range"] // rs["total"] if rs["total"] > 0 else 0
    lines.append(f"| {rnd} | {rs['total']} | {rs['in_range']} | {rs['higher']} | {rs['lower']} | {hit_rate}% | {mean_abs:.0f} |")

lines.append(f"")
lines.append(f"### Comparison: Prospective vs Original (with 2026 leakage)")
lines.append(f"")
lines.append(f"| Metric | Original (with 2026) | Prospective (without 2026) | Change |")
lines.append(f"|--------|---------------------|---------------------------|--------|")
# We'd need the original accuracy numbers, but let's note this
lines.append(f"| Within range | (see analysis/2026_draft_accuracy.md) | {100*within_range//total_matched}% | — |")
lines.append(f"| Mean |Δ| | (see analysis/2026_draft_accuracy.md) | {mean_delta:.1f} | — |")
lines.append(f"")
lines.append(f"*Note: The original report included 2026 outcomes in training data, which artificially")
lines.append(f"inflates accuracy. This prospective report is the honest assessment.*")
lines.append(f"")

# ── Detailed tables ─────────────────────────────────────────────

# Round 1 table
lines.append(f"---")
lines.append(f"")
lines.append(f"## Round 1 (Picks 1-40)")
lines.append(f"")
lines.append(f"| Pick | Player | Pos | School | Proj Pick | Proj Rnd | Δ | Status | Grade |")
lines.append(f"|------|--------|-----|--------|-----------|----------|---|--------|-------|")
r1_matches = sorted([m for m in matches if m["round"] == 1 or (m["round"] is None and m["pick"] <= 40)], key=lambda x: x["pick"])
for m in r1_matches:
    delta_str = f"{m['delta']:+d}"
    lines.append(f"| {m['pick']} | {m['player']} | {m['pos']} | {m['school']} | {m['proj_pick']} | {m['proj_round']} | {delta_str} | {m['status']} | {m['grade']} |")

# Rounds 2-5 table
lines.append(f"")
lines.append(f"## Rounds 2-5 (Picks 41-170)")
lines.append(f"")
lines.append(f"| Pick | Player | Pos | School | Proj Pick | Δ | Status | Grade |")
lines.append(f"|------|--------|-----|--------|-----------|---|--------|-------|")
r2_5_matches = sorted([m for m in matches if m["round"] in (2, 3, 4, 5)], key=lambda x: x["pick"])
for m in r2_5_matches:
    delta_str = f"{m['delta']:+d}"
    lines.append(f"| {m['pick']} | {m['player']} | {m['pos']} | {m['school']} | {m['proj_pick']} | {delta_str} | {m['status']} | {m['grade']} |")

# Rounds 6-10 — misses only
lines.append(f"")
lines.append(f"## Rounds 6-10 — Misses Only")
lines.append(f"")
lines.append(f"| Pick | Player | Pos | School | Proj Pick | Δ | Status | Grade |")
lines.append(f"|------|--------|-----|--------|-----------|---|--------|-------|")
r6_10_misses = sorted([m for m in matches if m["round"] in (6, 7, 8, 9, 10) and m["status"] != "✓"], key=lambda x: x["pick"])
for m in r6_10_misses:
    delta_str = f"{m['delta']:+d}"
    lines.append(f"| {m['pick']} | {m['player']} | {m['pos']} | {m['school']} | {m['proj_pick']} | {delta_str} | {m['status']} | {m['grade']} |")

# Rounds 11-20 — misses only
lines.append(f"")
lines.append(f"## Rounds 11-20 — Misses Only")
lines.append(f"")
lines.append(f"| Pick | Player | Pos | School | Proj Pick | Δ | Status | Grade |")
lines.append(f"|------|--------|-----|--------|-----------|---|--------|-------|")
r11_20_misses = sorted([m for m in matches if m["round"] in range(11, 21) and m["status"] != "✓"], key=lambda x: x["pick"])
for m in r11_20_misses:
    delta_str = f"{m['delta']:+d}"
    lines.append(f"| {m['pick']} | {m['player']} | {m['pos']} | {m['school']} | {m['proj_pick']} | {delta_str} | {m['status']} | {m['grade']} |")

# Biggest misses
lines.append(f"")
lines.append(f"## Biggest Misses (|Δ| > 150)")
lines.append(f"")
lines.append(f"| Pick | Player | Pos | School | Proj Pick | Δ | Direction | Grade |")
lines.append(f"|------|--------|-----|--------|-----------|---|-----------|-------|")
big_misses = sorted([m for m in matches if abs(m["delta"]) > 150], key=lambda x: abs(x["delta"]), reverse=True)
for m in big_misses:
    direction = "Drafted HIGHER (model undervalued)" if m["delta"] > 0 else "Drafted LOWER (model overvalued)"
    delta_str = f"{m['delta']:+d}"
    lines.append(f"| {m['pick']} | {m['player']} | {m['pos']} | {m['school']} | {m['proj_pick']} | {delta_str} | {direction} | {m['grade']} |")

# Best predictions
lines.append(f"")
lines.append(f"## Best Predictions (|Δ| ≤ 10, Rounds 1-5)")
lines.append(f"")
lines.append(f"| Pick | Player | Pos | School | Proj Pick | Δ | Grade |")
lines.append(f"|------|--------|-----|--------|-----------|---|-------|")
best_predictions = sorted([m for m in matches if abs(m["delta"]) <= 10 and m["round"] in range(1, 6)], key=lambda x: x["pick"])
for m in best_predictions:
    delta_str = f"{m['delta']:+d}"
    lines.append(f"| {m['pick']} | {m['player']} | {m['pos']} | {m['school']} | {m['proj_pick']} | {delta_str} | {m['grade']} |")

# Unmatched players
lines.append(f"")
lines.append(f"## Unmatched Players (Not in Model)")
lines.append(f"")
lines.append(f"These college players were drafted but did not appear in the model's")
lines.append(f"FanGraphs-derived projection set (likely JUCO/D2/NAIA, insufficient")
lines.append(f"stats, or name mismatch):")
lines.append(f"")
lines.append(f"| Pick | Player | School |")
lines.append(f"|------|--------|--------|")
for p in sorted(drafted_not_in_model, key=lambda x: x.get("pick_number", 999)):
    lines.append(f"| {p.get('pick_number')} | {p.get('full_name')} | {p.get('school_name')} |")

# Methodology
lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## Methodology Notes (Prospective Validation)")
lines.append(f"")
lines.append(f"- **Training cutoff**: All training data from 2021-2025 only. 2026 outcomes were")
lines.append(f"  excluded from the training set, making this a true prospective test.")
lines.append(f"- **Projected Pick**: From Tier 1 XGBoost regressor trained on FanGraphs stats + conf_strength + conference-adjusted stats")
lines.append(f"- **Range Band**: ±MAE (Mean Absolute Error) from cross-validation — {HITTER_MAE} picks for hitters, {PITCHER_MAE} for pitchers")
lines.append(f"- **Delta (Δ)**: Projected Pick minus Actual Pick. Positive = model thought they'd go later (drafted higher than expected). Negative = model thought they'd go earlier (drafted lower than expected).")
lines.append(f"- **Status**: ✓ = within MAE band. ⬆ HIGHER = drafted earlier than model's range. ⬇ LOWER = drafted later than model's range.")
lines.append(f"- **Grade**: Model's composite value grade (elite/high/medium/low) combining draft position + MLB probability projections.")
lines.append(f"- **College filter**: Excluded all high school picks (HS SR). Included 4-year, JUCO, and graduate students.")
lines.append(f"- **Player matching**: Name-normalized matching by (player_name, team_abb) cross-referenced against (full_name, school_name).")

lines.append(f"")
lines.append(f"### Interpretation")
lines.append(f"")
lines.append(f"This prospective test shows the model's true out-of-sample accuracy on unseen data.")
lines.append(f"The MAE band reflects the inherent noise in predicting a process heavily influenced")
lines.append(f"by team-specific needs, bonus pool allocation, and subjective evaluation.")
lines.append(f"")
lines.append(f"**Key questions this answers:**")
lines.append(f"- Does the model overfit to year-specific patterns?")
lines.append(f"- How much does 2026 leakage inflate the reported accuracy?")
lines.append(f"- Are the model's confidence calibrations reliable for truly unseen drafts?")

# Save report
report_path = BASE / "analysis" / "2026_draft_accuracy_prospective.md"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as f:
    f.write("\n".join(lines))

print(f"\nReport saved: {report_path}")
print(f"Report lines: {len(lines)}")
print(f"\nDone.")
