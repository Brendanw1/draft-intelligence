"""Verify a sample of matched and unmatched players."""
import json

BASE = '/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model'
results = json.load(open(f'{BASE}/analysis/eval_results.json'))

# Show the first 30 matched details
print("=== FIRST 30 MATCHES ===")
for m in results['matched_details'][:30]:
    print(f"  P{m['pick']:3d} | {m['player']:25s} | {m['school']:30s} | {m['school_abb']:6s} | "
          f"Proj={m['proj_pick']:5.0f} | Act={m['pick']:3d} | Δ={m['delta']:+.0f} | "
          f"{m['match_type']:20s} | {m['player_type']:8s} | {m['grade']:8s}")

# Show unmatched college players
print(f"\n=== UNMATCHED COLLEGE PLAYERS (sample) ===")
draft = json.load(open(f'{BASE}/data/draft/draft_2026.json'))
matched_picks = {m['pick'] for m in results['matched_details']}
unmatched_count = 0
for r in draft:
    school_class = r.get('school_class', '') or ''
    if 'HS' in school_class:
        continue
    if r.get('pick_number') not in matched_picks:
        if unmatched_count < 20:
            print(f"  P{r['pick_number']:3d} | {r['full_name']:25s} | {r['school_name']:30s} | {r.get('school_class',''):10s}")
        unmatched_count += 1
print(f"Total unmatched college: {unmatched_count}")
