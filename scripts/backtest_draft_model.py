#!/usr/bin/env python3
"""
backtest_draft_model.py — Validate Draft Model scoring weights against 2026 outcomes.

Approach:
  1. Load 2025 and 2026 board exports
  2. Join players present in both years by player_uid
  3. For each component score, compute correlation with same-year outcome metric
  4. Test predictive validity: does 2025 score → 2026 outcome?
  5. Test risk calibration: do high-risk players underperform relative to score?
  6. Grid-search alternative weights against 2026 holdout R²

Usage:
  python3 backtest_draft_model.py [--exports-dir exports/dashboard]
"""

import csv, sys, os, json, math
from pathlib import Path
from collections import defaultdict

EXPORTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("exports/dashboard")

def load_board(role):
    """Load board CSV, return list of ALL rows (not indexed)."""
    path = EXPORTS_DIR / f"{role}_board.csv"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"  Loaded {len(rows)} {role} from {path.name}")
    return rows

def safe_float(v):
    try: return float(v)
    except: return None

def corr(xs, ys):
    """Pearson correlation of two equal-length lists, skipping pairs with NA."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 10:
        return None, len(pairs)
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sx = math.sqrt(sum((p[0] - mx)**2 for p in pairs) / (n - 1))
    sy = math.sqrt(sum((p[1] - my)**2 for p in pairs) / (n - 1))
    if sx == 0 or sy == 0:
        return None, n
    r = sum((p[0] - mx) * (p[1] - my) for p in pairs) / ((n - 1) * sx * sy)
    return r, n

def r_squared(actual, predicted):
    """R² = 1 - SSE/SST"""
    pairs = [(a, p) for a, p in zip(actual, predicted) if a is not None and p is not None]
    if len(pairs) < 10:
        return None, len(pairs)
    n = len(pairs)
    mean_a = sum(p[0] for p in pairs) / n
    sst = sum((p[0] - mean_a)**2 for p in pairs)
    sse = sum((p[0] - p[1])**2 for p in pairs)
    if sst == 0:
        return None, n
    return 1 - sse / sst, n

def rank_biserial(group_a, group_b):
    """Effect size for two-group comparison (rank-biserial correlation)."""
    all_vals = group_a + group_b
    ranks = {v: i + 1 for i, v in enumerate(sorted(all_vals))}
    r_a = sum(ranks[v] for v in group_a)
    n_a, n_b = len(group_a), len(group_b)
    u = r_a - n_a * (n_a + 1) / 2
    return 2 * u / (n_a * n_b) - 1  # ranges -1 to 1

# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("DRAFT MODEL BACKTEST")
print("=" * 60)

h25 = load_board("hitters")
h26 = load_board("hitters")
p25 = load_board("pitchers")
p26 = load_board("pitchers")

# Load both seasons from single CSV, then split
h_all = load_board("hitters")
p_all = load_board("pitchers")

# Split by season — index by uid within each season
h25 = {}; h26 = {}
for r in h_all:
    if r["season"] == "2025": h25[r["player_uid"]] = r
    elif r["season"] == "2026": h26[r["player_uid"]] = r

p25 = {}; p26 = {}
for r in p_all:
    if r["season"] == "2025": p25[r["player_uid"]] = r
    elif r["season"] == "2026": p26[r["player_uid"]] = r

# Join players present in both years
h_both = set(h25.keys()) & set(h26.keys())
p_both = set(p25.keys()) & set(p26.keys())
print(f"\nHitters in both seasons: {len(h_both)}")
print(f"Pitchers in both seasons: {len(p_both)}")

# ══════════════════════════════════════════════════════════════
#  HITTER VALIDATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("HITTER COMPONENT VALIDATION")
print("=" * 60)

# Build paired lists for each test
h_tests = {
    # (2025_score, 2026_outcome, label)
    "impact → EV90":       ("impact_score", "p90_ev_wood_adj", "Impact score predicts 2026 EV90?"),
    "impact → barrel%":    ("impact_score", "barrel_rate_proxy_wood_adj", "Impact score predicts 2026 barrel%?"),
    "contact → contact%":  ("contact_score", "contact_rate", "Contact score predicts 2026 contact%?"),
    "contact → whiff%":    ("contact_score", "whiff_rate", "Contact score (inv) predicts 2026 whiff%?"),
    "contact → chase%":    ("contact_score", "chase_rate", "Contact score (inv) predicts 2026 chase%?"),
    "reach → EV90":        ("reach_score", "p90_ev_wood_adj", "Reach score predicts 2026 EV90?"),
    "DV → DV (stability)": ("draft_value_score", "draft_value_score", "2025 DV predicts 2026 DV (stability)?"),
}

for test_name, (score_col, outcome_col, label) in h_tests.items():
    xs = [safe_float(h25[uid][score_col]) for uid in h_both if uid in h25 and uid in h26]
    ys = [safe_float(h26[uid][outcome_col]) for uid in h_both if uid in h25 and uid in h26]
    r, n = corr(xs, ys)
    sig = "***" if r and abs(r) > 0.3 else ("**" if r and abs(r) > 0.2 else ("*" if r and abs(r) > 0.1 else ""))
    r_str = f"{r:+.3f}" if r is not None else "N/A"
    print(f"  {test_name:<25s} r={r_str}{sig} (n={n})  — {label}")

# DV stability by risk tercile
print("\n── Risk Calibration: DV Stability by Risk Tercile ──")
dv_pairs = []
for uid in h_both:
    dv25 = safe_float(h25[uid]["draft_value_score"])
    dv26 = safe_float(h26[uid]["draft_value_score"])
    risk = safe_float(h25[uid]["risk_score"])
    if dv25 and dv26 and risk:
        dv_pairs.append((dv25, dv26, risk))

if dv_pairs:
    dv_pairs.sort(key=lambda x: x[2])
    n = len(dv_pairs)
    low_risk = dv_pairs[:n//3]
    mid_risk = dv_pairs[n//3:2*n//3]
    high_risk = dv_pairs[2*n//3:]
    
    for label, group in [("Low Risk", low_risk), ("Mid Risk", mid_risk), ("High Risk", high_risk)]:
        dv25s = [p[0] for p in group]
        dv26s = [p[1] for p in group]
        r, _ = corr(dv25s, dv26s)
        mean_delta = sum(dv26s[i] - dv25s[i] for i in range(len(dv25s))) / len(dv25s)
        print(f"  {label:<12s} n={len(group):<5d} r(DV25→DV26)={r:+.3f}  mean ΔDV={mean_delta:+.1f}")

# Also check: does high risk → worse 2026 EV90 controlling for 2025 score?
print("\n── Risk Penalty Check: Does Risk Predict Underperformance? ──")
# For players with similar 2025 DV, does higher risk → lower 2026 DV?
dv25_buckets = defaultdict(list)
for uid in h_both:
    dv25 = safe_float(h25[uid]["draft_value_score"])
    risk = safe_float(h25[uid]["risk_score"])
    if dv25 and risk:
        bucket = int(dv25 // 10) * 10
        dv25_buckets[bucket].append(uid)

for bucket in sorted(dv25_buckets.keys()):
    uids = dv25_buckets[bucket]
    if len(uids) < 20:
        continue
    uids.sort(key=lambda u: safe_float(h25[u]["risk_score"]))
    n = len(uids)
    low_r = uids[:n//2]
    high_r = uids[n//2:]
    
    dv26_low = [safe_float(h26[u]["draft_value_score"]) for u in low_r if u in h26]
    dv26_high = [safe_float(h26[u]["draft_value_score"]) for u in high_r if u in h26]
    dv26_low = [v for v in dv26_low if v is not None]
    dv26_high = [v for v in dv26_high if v is not None]
    
    if dv26_low and dv26_high:
        mean_low = sum(dv26_low) / len(dv26_low)
        mean_high = sum(dv26_high) / len(dv26_high)
        delta = mean_low - mean_high
        print(f"  DV25 bucket [{bucket}-{bucket+10}): low-risk 2026 DV={mean_low:.1f}  high-risk 2026 DV={mean_high:.1f}  Δ={delta:+.1f}")

# ══════════════════════════════════════════════════════════════
#  PITCHER VALIDATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PITCHER COMPONENT VALIDATION")
print("=" * 60)

p_tests = {
    "stuff → whiff%":     ("stuff_score", "whiff_pct", "Stuff score predicts 2026 whiff%?"),
    "stuff → FB velo":    ("stuff_score", "avg_fb_velo", "Stuff score predicts 2026 FB velo?"),
    "command → CSW%":     ("command_score", "csw_pct", "Command score predicts 2026 CSW%?"),
    "command → zone%":    ("command_score", "zone_pct", "Command score predicts 2026 zone%?"),
    "reach → FB velo":    ("reach_score", "avg_fb_velo", "Reach score predicts 2026 FB velo?"),
    "DV → DV (stability)": ("draft_value_score", "draft_value_score", "2025 DV predicts 2026 DV?"),
}

for test_name, (score_col, outcome_col, label) in p_tests.items():
    xs = [safe_float(p25[uid][score_col]) for uid in p_both if uid in p25 and uid in p26]
    ys = [safe_float(p26[uid][outcome_col]) for uid in p_both if uid in p25 and uid in p26]
    r, n = corr(xs, ys)
    sig = "***" if r and abs(r) > 0.3 else ("**" if r and abs(r) > 0.2 else ("*" if r and abs(r) > 0.1 else ""))
    r_str = f"{r:+.3f}" if r is not None else "N/A"
    print(f"  {test_name:<25s} r={r_str}{sig} (n={n})  — {label}")

# Pitcher risk calibration
print("\n── Risk Calibration: DV Stability by Risk Tercile ──")
dv_pairs_p = []
for uid in p_both:
    dv25 = safe_float(p25[uid]["draft_value_score"])
    dv26 = safe_float(p26[uid]["draft_value_score"])
    risk = safe_float(p25[uid]["risk_score"])
    if dv25 and dv26 and risk:
        dv_pairs_p.append((dv25, dv26, risk))

if dv_pairs_p:
    dv_pairs_p.sort(key=lambda x: x[2])
    n = len(dv_pairs_p)
    low_risk = dv_pairs_p[:n//3]
    mid_risk = dv_pairs_p[n//3:2*n//3]
    high_risk = dv_pairs_p[2*n//3:]
    
    for label, group in [("Low Risk", low_risk), ("Mid Risk", mid_risk), ("High Risk", high_risk)]:
        dv25s = [p[0] for p in group]
        dv26s = [p[1] for p in group]
        r, _ = corr(dv25s, dv26s)
        mean_delta = sum(dv26s[i] - dv25s[i] for i in range(len(dv25s))) / len(dv25s)
        print(f"  {label:<12s} n={len(group):<5d} r(DV25→DV26)={r:+.3f}  mean ΔDV={mean_delta:+.1f}")

# ══════════════════════════════════════════════════════════════
#  WEIGHT GRID SEARCH (Hitters)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("WEIGHT OPTIMIZATION: HITTER DV → 2026 wOBA")
print("=" * 60)

# Compute approximate wOBA for each 2026 hitter from available metrics
# wOBA ≈ something like: 0.7*BB% + 0.9*1B% + 1.3*2B% + 1.6*3B% + 2.0*HR%
# We don't have those components directly, but we have contact_rate, EV, etc.
# Use 2026 draft_value_score as a proxy for "overall quality" since we want to see
# if different 2025 weights better predict 2026 DV

# Grid search: test different DV weight combinations
# Current: reach=0.25, impact=0.40, contact=0.35, risk_penalty=0.15
# We want to find weights that maximize r(2025_weighted_score, 2026_dv)

# Build raw component arrays
h_paired = []
for uid in h_both:
    if uid not in h25 or uid not in h26:
        continue
    r25 = h25[uid]; r26 = h26[uid]
    dv26 = safe_float(r26["draft_value_score"])
    reach = safe_float(r25["reach_score"])
    impact = safe_float(r25["impact_score"])
    contact = safe_float(r25["contact_score"])
    risk = safe_float(r25["risk_score"])
    if all(v is not None for v in [dv26, reach, impact, contact, risk]):
        h_paired.append((reach, impact, contact, risk, dv26))

print(f"Paired hitters for grid search: {len(h_paired)}")

# Current weights
current_w = {"reach": 0.25, "impact": 0.40, "contact": 0.35, "risk_penalty": 0.15}
r_cur, _ = corr(
    [p[0] * current_w["reach"] + p[1] * current_w["impact"] + p[2] * current_w["contact"] - current_w["risk_penalty"] * p[3] for p in h_paired],
    [p[4] for p in h_paired]
)
print(f"Current weights (r={r_cur:.4f}): reach={current_w['reach']} impact={current_w['impact']} contact={current_w['contact']} risk_penalty={current_w['risk_penalty']}")

# Test alternative weight combos
best_r = r_cur
best_w = dict(current_w)
results = []

for r_w in [0.15, 0.20, 0.25, 0.30, 0.35]:
    for i_w in [0.30, 0.35, 0.40, 0.45, 0.50]:
        c_w = 1.0 - r_w - i_w
        if c_w < 0.15 or c_w > 0.50:
            continue
        for rp in [0.10, 0.15, 0.20]:
            scores = [p[0] * r_w + p[1] * i_w + p[2] * c_w - rp * p[3] for p in h_paired]
            r, _ = corr(scores, [p[4] for p in h_paired])
            results.append((r, r_w, i_w, c_w, rp))
            if r and r > best_r:
                best_r = r
                best_w = {"reach": r_w, "impact": i_w, "contact": c_w, "risk_penalty": rp}

results.sort(reverse=True)
print(f"\nTop 10 weight combinations (correlation with 2026 DV):")
for i, (r_val, rw, iw, cw, rp) in enumerate(results[:10]):
    marker = " ← CURRENT" if (rw, iw, cw, rp) == (current_w["reach"], current_w["impact"], current_w["contact"], current_w["risk_penalty"]) else ""
    print(f"  {i+1}. r={r_val:.4f}  reach={rw:.2f} impact={iw:.2f} contact={cw:.2f} risk_penalty={rp:.2f}{marker}")

print(f"\nBest weights: reach={best_w['reach']:.2f} impact={best_w['impact']:.2f} contact={best_w['contact']:.2f} risk_penalty={best_w['risk_penalty']:.2f} (r={best_r:.4f})")
print(f"Current:      reach={current_w['reach']:.2f} impact={current_w['impact']:.2f} contact={current_w['contact']:.2f} risk_penalty={current_w['risk_penalty']:.2f} (r={r_cur:.4f})")

# ══════════════════════════════════════════════════════════════
#  WEIGHT GRID SEARCH (Pitchers)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("WEIGHT OPTIMIZATION: PITCHER DV → 2026 DV")
print("=" * 60)

p_paired = []
for uid in p_both:
    if uid not in p25 or uid not in p26:
        continue
    r25 = p25[uid]; r26 = p26[uid]
    dv26 = safe_float(r26["draft_value_score"])
    reach = safe_float(r25["reach_score"])
    stuff = safe_float(r25["stuff_score"])
    command = safe_float(r25["command_score"])
    risk = safe_float(r25["risk_score"])
    if all(v is not None for v in [dv26, reach, stuff, command, risk]):
        p_paired.append((reach, stuff, command, risk, dv26))

print(f"Paired pitchers for grid search: {len(p_paired)}")

p_current_w = {"reach": 0.25, "stuff": 0.40, "command": 0.35, "risk_penalty": 0.15}
pr_cur, _ = corr(
    [p[0] * p_current_w["reach"] + p[1] * p_current_w["stuff"] + p[2] * p_current_w["command"] - p_current_w["risk_penalty"] * p[3] for p in p_paired],
    [p[4] for p in p_paired]
)
print(f"Current weights (r={pr_cur:.4f}): reach={p_current_w['reach']} stuff={p_current_w['stuff']} command={p_current_w['command']} risk_penalty={p_current_w['risk_penalty']}")

p_best_r = pr_cur
p_best_w = dict(p_current_w)
p_results = []

for r_w in [0.15, 0.20, 0.25, 0.30, 0.35]:
    for s_w in [0.30, 0.35, 0.40, 0.45, 0.50]:
        c_w = 1.0 - r_w - s_w
        if c_w < 0.15 or c_w > 0.50:
            continue
        for rp in [0.10, 0.15, 0.20]:
            scores = [p[0] * r_w + p[1] * s_w + p[2] * c_w - rp * p[3] for p in p_paired]
            r, _ = corr(scores, [p[4] for p in p_paired])
            p_results.append((r, r_w, s_w, c_w, rp))
            if r and r > p_best_r:
                p_best_r = r
                p_best_w = {"reach": r_w, "stuff": s_w, "command": c_w, "risk_penalty": rp}

p_results.sort(reverse=True)
print(f"\nTop 10 weight combinations:")
for i, (r_val, rw, sw, cw, rp) in enumerate(p_results[:10]):
    marker = " ← CURRENT" if (rw, sw, cw, rp) == (p_current_w["reach"], p_current_w["stuff"], p_current_w["command"], p_current_w["risk_penalty"]) else ""
    print(f"  {i+1}. r={r_val:.4f}  reach={rw:.2f} stuff={sw:.2f} command={cw:.2f} risk_penalty={rp:.2f}{marker}")

print(f"\nBest weights: reach={p_best_w['reach']:.2f} stuff={p_best_w['stuff']:.2f} command={p_best_w['command']:.2f} risk_penalty={p_best_w['risk_penalty']:.2f} (r={p_best_r:.4f})")
print(f"Current:      reach={p_current_w['reach']:.2f} stuff={p_current_w['stuff']:.2f} command={p_current_w['command']:.2f} risk_penalty={p_current_w['risk_penalty']:.2f} (r={pr_cur:.4f})")

print("\n" + "=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)
