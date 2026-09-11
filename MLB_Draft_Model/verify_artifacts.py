"""Verify all output artifacts."""
import json, os

BASE = '/Users/brendanwaterval/Projects/vt_baseball/MLB_Draft_Model'

# Check eval results
results = json.load(open(f'{BASE}/analysis/eval_results.json'))
print(f'eval_results.json: {len(results)} keys, {len(results["matched_details"])} matches')
print(f'  MAE: {results["mae"]:.1f}')
print(f'  Spearman: {results["spearman_r"]:.3f}')
print(f'  Within 110: {results["within_110"]}/{results["total_matched"]} ({results["within_110_pct"]:.0f}%)')

# Check report file
report_path = f'{BASE}/analysis/2026_draft_accuracy_prospective.md'
size = os.path.getsize(report_path)
with open(report_path) as f:
    lines = f.readlines()
print(f'\nReport: {size} bytes, {len(lines)} lines')
print(f'  First line: {lines[0].strip()}')

# Verify enriched projections exist and have data
proj = json.load(open(f'{BASE}/data/training/projections_2026_enriched.json'))
print(f'\nProjections: {len(proj)} records')
print(f'  Sample: {proj[0]["player_name"]} ({proj[0]["team_abb"]}) -> pick {proj[0]["projected_pick"]:.0f}')

print('\n✓ All artifacts verified.')
