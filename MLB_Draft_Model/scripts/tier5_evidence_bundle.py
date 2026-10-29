#!/usr/bin/env python3
"""Evidence bundle for Tier 5 WAR Value Pipeline go/no-go decision."""
import json, pickle, subprocess, sys
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[1]
ARTIFACTS = BASE / 'models/artifacts_full'
DOCS = BASE / 'docs'

print('=' * 70)
print('TIER 5 EVIDENCE BUNDLE')
print('=' * 70)
print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print()

# 1. Model Artifacts Inventory
print('[1] MODEL ARTIFACTS')
print('-' * 70)
artifacts = {}
for pt in ['hitter', 'pitcher']:
    for stage in ['hurdle', 'value']:
        pkl = ARTIFACTS / f'tier5_{stage}_{pt}.pkl'
        feat = ARTIFACTS / f'tier5_{stage}_{pt}_features.json'
        if pkl.exists():
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            with open(feat) as f:
                m = json.load(f)
            artifacts[f'{stage}_{pt}'] = {
                'pkl_size': pkl.stat().st_size,
                'features': len(d.get('features', m.get('features', []))),
                'gate_r2': d.get('gate', {}).get('r2', 'N/A'),
                'gate_auc': d.get('gate', {}).get('auc', 'N/A'),
            }
            print(f'  tier5_{stage}_{pt}: {pkl.stat().st_size:,}B, {len(d.get("features", []))} features')
            if 'gate' in d:
                print(f'    Gate: R²={d["gate"].get("r2", "N/A")}, AUC={d["gate"].get("auc", "N/A")}')
print()

# 2. Validation Gates Summary
print('[2] VALIDATION GATES')
print('-' * 70)
validate_script = BASE / 'scripts/validate_tier5.py'
if validate_script.exists():
    result = subprocess.run(
        [sys.executable, str(validate_script), '--verify'],
        capture_output=True, text=True, cwd=str(BASE),
        env={'PYTHONPATH': str(BASE / 'src')}
    )
    # Count PASS/WARN/FAIL from stdout and stderr
    output = result.stdout + result.stderr
    pass_count = output.count('PASS')
    warn_count = output.count('WARN')
    fail_count = output.count('FAIL')
    if pass_count == 0 and warn_count == 0 and fail_count == 0:
        print('  WARNING: Could not parse validation output')
        if result.returncode != 0:
            print(f'  Script exited with code {result.returncode}')
        if result.stderr:
            print(f'  Stderr: {result.stderr[:200]}')
    print(f'  Total gates: {pass_count + warn_count + fail_count}')
    print(f'  PASS: {pass_count}, WARN: {warn_count}, FAIL: {fail_count}')
    if fail_count > 0:
        print('  CRITICAL: Validation failures detected')
print()

# 3. Honesty Report Status
print('[3] HONESTY REPORT')
print('-' * 70)
honesty_report = DOCS / 'tier5_honesty_report.md'
if honesty_report.exists():
    content = honesty_report.read_text()
    if 'NOT READY' in content:
        print('  Status: NOT READY FOR PRODUCTION')
    elif 'PASS' in content:
        print('  Status: READY FOR PRODUCTION')
    else:
        print('  Status: UNKNOWN')
    print(f'  Size: {honesty_report.stat().st_size:,}B')
else:
    print('  Status: NOT GENERATED')
print()

# 4. Backtest Results
print('[4] BACKTEST RESULTS (2015-2020 Cohort)')
print('-' * 70)
backtest_report = DOCS / 'tier5_backtest_report.md'
if backtest_report.exists():
    content = backtest_report.read_text()
    # Extract key metrics (format: "- **R²**: -0.1132")
    for line in content.split('\n'):
        if '**R²**:' in line or '**RMSE**:' in line or '**Correlation**:' in line:
            print(f'  {line.strip().lstrip("- ")}')
    if 'Kill criteria FAILED' in content:
        print('  Kill criteria: FAILED')
    elif 'Kill criteria PASSED' in content:
        print('  Kill criteria: PASSED')
else:
    print('  Status: NOT GENERATED')
print()

# 5. Frontend Integration
print('[5] FRONTEND INTEGRATION')
print('-' * 70)
players_index = BASE / 'web/public/data/players_index.json'
if players_index.exists():
    with open(players_index) as f:
        players = json.load(f)
    tier5_count = sum(1 for p in players if p.get('tier5_hurdle_prob') is not None)
    print(f'  Total players: {len(players):,}')
    print(f'  Players with Tier 5 predictions: {tier5_count:,} ({100*tier5_count/len(players):.1f}%)')
else:
    print('  Status: Frontend data not generated')

manifest = BASE / 'web/public/data/models_manifest.json'
if manifest.exists():
    with open(manifest) as f:
        m = json.load(f)
    tier5_models = [k for k in m.keys() if 'tier5' in k]
    print(f'  Tier 5 models in manifest: {len(tier5_models)}')
    for model_key in tier5_models:
        gate_status = m[model_key].get('gate_status', 'unknown')
        print(f'    {model_key}: {gate_status}')
print()

# 6. Go/No-Go Decision
print('[6] GO/NO-GO DECISION')
print('-' * 70)
kill_criteria_met = False
for key, data in artifacts.items():
    r2 = data.get('gate_r2', -999)
    if isinstance(r2, (int, float)) and r2 >= 0.0:
        kill_criteria_met = True
        break

if kill_criteria_met:
    print('  Decision: GO — Kill criteria met (R² ≥ 0.0)')
    print('  Tier 5 predictions are production-ready.')
else:
    print('  Decision: NO-GO — Kill criteria not met (R² < 0.0)')
    print('  Tier 5 predictions are for research/exploration only.')
    print()
    print('  Recommendations:')
    print('  - Increase training set size (currently ~300 records)')
    print('  - Add more predictive features (advanced metrics, biomechanics)')
    print('  - Extend temporal window (currently 2021-2023)')
    print('  - Consider ensemble methods or different model architectures')
print()

# 7. Evidence Bundle Summary
print('=' * 70)
print('EVIDENCE BUNDLE SUMMARY')
print('=' * 70)
print(f'  Artifacts: {len(artifacts)} models')
print(f'  Validation: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL')
print(f'  Honesty Report: {"Generated" if honesty_report.exists() else "Missing"}')
print(f'  Backtest: {"Generated" if backtest_report.exists() else "Missing"}')
print(f'  Frontend: {tier5_count:,} players with Tier 5 predictions')
print(f'  Kill Criteria: {"MET" if kill_criteria_met else "NOT MET"}')
print()
print('Bundle saved to: docs/tier5_evidence_bundle.txt')

# Save bundle to file
bundle_path = DOCS / 'tier5_evidence_bundle.txt'
with open(bundle_path, 'w') as f:
    f.write(f'Tier 5 Evidence Bundle\n')
    f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'Kill Criteria: {"MET" if kill_criteria_met else "NOT MET"}\n')
    f.write(f'Go/No-Go: {"GO" if kill_criteria_met else "NO-GO"}\n')
    f.write(f'Artifacts: {len(artifacts)} models\n')
    f.write(f'Validation: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL\n')
    f.write(f'Frontend Coverage: {tier5_count:,} players\n')
