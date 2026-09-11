# MLB Draft Model — Project Status

**Last Updated**: August 21, 2026

## Current State — Five-Tier Pipeline (Tier 5 in Development)

| Component | Status | Detail |
|-----------|--------|--------|
| MLB Draft API scraper | ✅ Live | 9,300+ picks (2015–2026), 100% person_id coverage |
| FanGraphs D1 stat pipeline | ✅ Live | 10,734 players (2021–2026), hitters + pitchers |
| Tier 1 — Round Regressor (XGBoost) | ✅ Live | Predicts pick number, outputs round band + confidence |
| Tier 2 — MLB Probability (XGBoost + Platt) | ✅ Live | Full-population classifier, 56,910 undrafted negatives |
| Tier 3 — MLB Arrival (Elastic Net prior-offset) | ✅ Live | P(MLB debut\|drafted), AUC 0.79, round-anchored prior |
| Tier 4 — Comp Database (MiLB-enriched) | ✅ Live | 1,524 comp pool, 2021–2024 only, with peak level + MLB flags |
| Tier 5 — WAR Value (LightGBM hurdle + value) | ✅ Pipeline Complete | Two-stage model: hurdle (P(WAR>0)) + value regression (E[WAR]). Kill criteria not yet met (R² < 0.0 on 2023 heldout, expected with ~300 train records). Hurdle AUC ~0.65, value R² negative. Full pipeline: Phase A → validation gates → honesty report → Phase B (gated) → frontend export → leaderboard → dossier card → backtest. |
| Conference adjustment (conf_strength) | ✅ Live | Continuous draft-rate ratio, replaces old 4-tier category |
| Nearest-neighbor comps (MiLB-enriched) | ✅ Live | 1,524 comp pool, 2021–2024 only, with peak level + MLB flags |
| Next.js 15 static frontend | ✅ Live | vt-draft-intelligence.vercel.app |
| Player shard data (64 files, 38 MB) | ✅ Synced | R2 bucket + git (for build-time) |
| Design system v2 | ✅ Live | Precision-instrument tokens, signal gradient, dark/light theme |
| Model Lab + Audit pages | ✅ Live | Calibration reliability ladders, backtest curves, feature importances |
| WAR Value Leaderboard | ✅ Live | `/war-leaderboard` page with hurdle gate toggle, sortable table |
| Player Dossier WAR Card | ✅ Live | Tier 5 hurdle prob bar, expected WAR + CI, top-3 features |

## Code Coverage

| Frontend | Status |
|----------|--------|
| TypeScript type check (`tsc --noEmit`) | ✅ Passes — zero errors |
| Test suite | ❌ Not configured — no test runner in web/package.json |

| Python | Status |
|--------|--------|
| `pytest` | ✅ 82 passed, 0 errors |
| Fix applied | `pythonpath = ["src"]` added to `[tool.pytest.ini_options]`. The `mlb_draft_dashboard` package was never moved — it is at `src/mlb_draft_dashboard` and declared in `pyproject.toml`; it was simply not on the import path. No `pip install -e .` needed. |
| Removed | `tests/test_bootstrap_team_mapping.py` — shelled out to `scripts/bootstrap_team_mapping.py`, which does not exist in this repo (it lives in the parent `vt_baseball/scripts/`). `build_team_crosswalk.py` is not a replacement: no CLI args, reads JSON rosters rather than a SQLite `VTData` table, and is currently untested. |

## Key Metrics (held-out test set)

| Metric | Value |
|--------|-------|
| Tier 1 backtest MAE | ~110 picks |
| Tier 2 AUC (hitters) | 0.994 |
| Tier 2 AUC (pitchers) | 0.989 |
| Tier 3 AUC (arrival) | 0.79 |
| Calibration error | <3% avg absolute (Platt-scaled) |
| Comp database | 1,524 records, 0 bad entries (all 2021–2024) |

## Data Assets (in git)

- `data/draft/` — MLB draft picks (2015–2026, per-year + consolidated)
- `data/fangraphs/` — FanGraphs D1 leaderboard exports (2021–2026)
- `data/milb/` — MiLB outcome scrapes (2021–2025)
- `data/rosters/` — NCAA D1 rosters + crosswalks
- `data/training/` — Training sets, projections, tier inputs
- `web/public/data/` — Frontend JSON bundle (shards, index, classes, manifest)
- `configs/` — Team mappings
- `exports/` — Dashboard-ready Parquet/CSV exports
- `scripts/` — All pipeline scripts (export, inference, training)

## Known Gaps

- `scripts/build_team_crosswalk.py` has no test coverage
- No frontend test suite configured
- STATUS.md now lives in git — keep in sync with README.md after model updates
