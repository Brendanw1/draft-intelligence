# MLB Draft Model — Project Status

**Last Updated**: July 16, 2026

## Current State — Three-Tier Pipeline (Live)

| Component | Status | Detail |
|-----------|--------|--------|
| MLB Draft API scraper | ✅ Live | 9,300+ picks (2015–2026), 100% person_id coverage |
| FanGraphs D1 stat pipeline | ✅ Live | 10,734 players (2021–2026), hitters + pitchers |
| Tier 1 — Round Regressor (XGBoost) | ✅ Live | Predicts pick number, outputs round band + confidence |
| Tier 2 — MLB Probability (XGBoost + Platt) | ✅ Live | Full-population classifier, 56,910 undrafted negatives |
| Tier 3 — MLB Arrival (Elastic Net prior-offset) | ✅ Live | P(MLB debut\|drafted), AUC 0.79, round-anchored prior |
| Conference adjustment (conf_strength) | ✅ Live | Continuous draft-rate ratio, replaces old 4-tier category |
| Nearest-neighbor comps (MiLB-enriched) | ✅ Live | 1,524 comp pool, 2021–2024 only, with peak level + MLB flags |
| Next.js 15 static frontend | ✅ Live | vt-draft-intelligence.vercel.app |
| Player shard data (64 files, 38 MB) | ✅ Synced | R2 bucket + git (for build-time) |
| Design system v2 | ✅ Live | Precision-instrument tokens, signal gradient, dark/light theme |
| Model Lab + Audit pages | ✅ Live | Calibration reliability ladders, backtest curves, feature importances |

## Code Coverage

| Frontend | Status |
|----------|--------|
| TypeScript type check (`tsc --noEmit`) | ✅ Passes — zero errors |
| Test suite | ❌ Not configured — no test runner in web/package.json |

| Python | Status |
|--------|--------|
| `pytest` (6 test files) | ⚠️ 2/6 pass, 4/6 broken — import errors (`mlb_draft_dashboard` package not installed) |
| Root cause | Tests written for old `mlb_draft_dashboard` package structure; model code migrated to `scripts/` |
| Fix needed | Install project in dev mode (`pip install -e .`) or migrate tests to match current layout |

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

- Python tests reference obsolete `mlb_draft_dashboard` package — need updating
- No frontend test suite configured
- STATUS.md now lives in git — keep in sync with README.md after model updates
