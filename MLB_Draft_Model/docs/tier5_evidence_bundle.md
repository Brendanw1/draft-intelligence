# Tier 5 WAR Value Pipeline — Evidence Bundle

**Generated**: August 21, 2026
**Branch**: `feat/baseballgoat-trackman-3d-lab`
**Pipeline**: T7–T16 complete, Final Wave pending

---

## 1. Pipeline Summary

| Stage | Script | Status | Key Metric |
|-------|--------|--------|------------|
| T7: Phase A (LightGBM) | `scripts/train_tier5_value.py` | ✅ Trained | Hitters R²=-0.32, Pitchers R²=-0.61 |
| T8: Validation | `scripts/validate_tier5.py` | ✅ 8/8 gates PASS | All training + WAR data checks pass |
| T8: Honesty Report | `scripts/tier5_honesty_report.py` | ✅ Generated | Kill criteria triggered (R² < 0.0) |
| T9: Phase B (Optuna) | `scripts/train_tier5_value_phaseB.py` | ✅ Gated skip | Phase A failed kill criteria → Phase B skipped |
| T10: Frontend Export | `scripts/export_frontend_data.py` | ✅ 10,734 players | 8 model cards in manifest |
| T11: WAR Leaderboard | `web/app/war-leaderboard/page.tsx` | ✅ Live | `/war-leaderboard` route |
| T12: Dossier WAR Card | `web/components/player/PlayerDossier.tsx` | ✅ Live | Hurdle prob + expected WAR + CI |
| T13: Methodology | `web/app/methodology/page.tsx` | ✅ Updated | Five-tier pipeline documented |
| T14: Backtest | `scripts/backtest_tier5.py` | ✅ Complete | 2023 cohort, 369 records |
| T15: Docs | STATUS.md, README.md, AGENTS.md | ✅ Updated | Five-tier architecture |
| T16: Verify All | `scripts/verify_all.py` | ✅ 7/7 sections PASS | All gates pass |

---

## 2. Model Artifacts

| Artifact | Size | Features | Gate Status |
|----------|------|----------|-------------|
| `tier5_hurdle_hitter.pkl` | 1.1 KB | 10 | AUC ~0.65 |
| `tier5_hurdle_pitcher.pkl` | 1.1 KB | 10 | AUC ~0.65 |
| `tier5_value_hitter.pkl` | 197 KB | 10 | R² = -0.32 (FAIL) |
| `tier5_value_pitcher.pkl` | 245 KB | 10 | R² = -0.61 (FAIL) |
| `tier5_hurdle_calibration.json` | 2.2 KB | — | Decile calibration curves |
| `tier5_value_calibration.json` | 4.2 KB | — | Decile calibration curves |
| `tier5_*_features.json` (×4) | ~2 KB each | 10 | Feature lists per role |

---

## 3. Backtest Results (2023 Cohort, 369 records)

| Role | N | RMSE | R² | Correlation | Hurdle Acc |
|------|---|------|-----|-------------|------------|
| Hitter | 160 | 2.86 | -0.11 | 0.19 | 72.7% |
| Pitcher | 209 | 0.98 | -0.36 | 0.05 | 61.1% |

**Kill criteria**: R² ≥ 0.0 → **FAIL** for both roles.

**Interpretation**: The value regression is worse than predicting the median WAR. This is expected with ~300 training records, right-censored heldout (2023 players haven't accumulated full career WAR), and a zero-inflated target distribution. The hurdle model shows modest discrimination (AUC ~0.65, accuracy ~65-73%) but the value regression needs more data.

---

## 4. Validation Gates (8/8 PASS)

| Gate | Check | Result |
|------|-------|--------|
| Training V1 | Hurdle rate sanity (2-10%) | 2.94% ✅ |
| Training V2 | Round-WAR negative correlation | -0.078 ✅ |
| Training V3 | Temporal leakage | 0 leakage ✅ |
| Training V4 | WAR distribution skew | median=0.04 ✅ |
| War V1 | BRef match rate | 100% ✅ |
| War V2 | Field completeness | 0 missing ✅ |
| War V3 | Phantom WAR check | 0 impossible ✅ |
| War V4 | Median WAR ≤ 0 | median=0.0 ✅ |

---

## 5. Frontend Integration

- **WAR Leaderboard** (`/war-leaderboard`): Sortable table with type filter, hurdle gate toggle, color-coded hurdle% and WAR values
- **Player Dossier WAR Card**: Shows hurdle probability bar, expected WAR ± CI, top-3 features, honesty note
- **Model Manifest**: 8 model cards (4 Tier 5: hurdle/value × hitter/pitcher) with gate_status and gate_r2
- **Export Pipeline**: All 10,734 players have `tier5_hurdle_prob`, `tier5_expected_war`, `tier5_confidence` fields

---

## 6. Go/No-Go Recommendation

**Status**: ⚠️ NOT READY FOR PRODUCTION

**Rationale**:
- Value regression fails kill criteria (R² < 0.0) for both roles
- Hurdle model shows modest discrimination but not production-grade
- Training set is small (~300 records with positive WAR)
- 2023 heldout is right-censored (players haven't accumulated full career WAR)

**Path to Production**:
1. Wait for more draft classes to accumulate WAR (2024, 2025 cohorts)
2. Re-run Phase A with larger training set
3. If R² ≥ 0.0, proceed to Phase B (Optuna tuning)
4. Re-run backtest on fully observed cohort
5. Re-evaluate kill criteria

**What IS production-ready**:
- Hurdle model architecture and training pipeline
- Validation gates and honesty report framework
- Frontend integration (leaderboard, dossier card)
- Export pipeline with Tier 5 fields
- Backtest infrastructure

---

## 7. File Inventory

```
scripts/
├── train_tier5_value.py          # Phase A LightGBM
├── train_tier5_value_phaseB.py   # Phase B Optuna (gated)
├── validate_tier5.py             # 8 verification gates
├── tier5_honesty_report.py       # Markdown report generator
├── backtest_tier5.py             # E2E backtest on 2023 cohort
├── export_frontend_data.py       # Updated with Tier 5 fields
└── verify_all.py                 # Extended with Tier 5 checks

models/artifacts_full/
├── tier5_hurdle_hitter.pkl
├── tier5_hurdle_pitcher.pkl
├── tier5_value_hitter.pkl
├── tier5_value_pitcher.pkl
├── tier5_hurdle_calibration.json
├── tier5_value_calibration.json
└── tier5_*_features.json (×4)

docs/
├── tier5_honesty_report.md
└── tier5_backtest_report.md

web/
├── app/war-leaderboard/page.tsx
├── components/player/PlayerDossier.tsx (WAR card added)
├── lib/types.ts (Tier 5 fields added)
├── lib/format.ts (fmtWAR added)
└── app/methodology/page.tsx (Tier 5 section added)

Makefile                           # CI entry point
```
