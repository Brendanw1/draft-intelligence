# MLB Draft Model — Project Status

**Last Updated**: October 29, 2026

## Current State — Tiers 1–4 live, Tier 5 withdrawn

| Component | Status | Detail |
|-----------|--------|--------|
| MLB Draft API scraper | ✅ Live | 9,300+ picks (2015–2026), 100% person_id coverage |
| FanGraphs D1 stat pipeline | ✅ Live | 10,734 players (2021–2026), hitters + pitchers |
| Tier 1 — Round Regressor (XGBoost) | ✅ Live | Projected pick if drafted (trained on draftees); forward MAE 117 / 115 (2024-25) |
| Tier 2 — P(pick ≤ 315 within 3 drafts) (XGBoost + OOF Platt) | ✅ Live (rebuilt 2026-10-29) | Full-population classifier; 2024 class re-read: AUC 0.857 / 0.843, ECE 1.4% / 0.6% (hitters / pitchers); board sums to 714 picks over 3 drafts |
| Tier 3 — MLB Arrival (logistic regression, round-rate logit offset) | ✅ Live | P(debut within 3 seasons \| drafted); 2021-22 classes, nested CV AUC 0.70 / 0.75, below the round rate alone (0.72 / 0.77): draft round carries it |
| Tier 4 — Comp Database (MiLB-enriched) | ✅ Live | 1,524 comp pool, 2021–2024 only, with peak level + MLB flags |
| Tier 5 — WAR Value | ⛔ Withdrawn 2026-10-29 | Failed its backtest: 2021 class out of fold, R² +0.03 hitters / −0.03 pitchers (docs/tier5_backtest_report.md). Not exported. |
| Conference adjustment (conf_strength) | ✅ Live | Continuous draft-rate ratio, replaces old 4-tier category |
| Nearest-neighbor comps (MiLB-enriched) | ✅ Live | 1,524 comp pool, 2021–2024 only, with peak level + MLB flags |
| Next.js 15 static frontend | ✅ Live | vt-draft-intelligence.vercel.app |
| Player shard data (64 files, 38 MB) | ✅ Synced | R2 bucket + git (for build-time) |
| Design system v2 | ✅ Live | Precision-instrument tokens, signal gradient, dark/light theme |
| Model Lab + Audit pages | ✅ Live | Calibration reliability ladders, backtest curves, feature importances |
| WAR Value Leaderboard | ⛔ Withdrawn | `/war-leaderboard` shows a "withdrawn: failed its backtest" notice |
| Player Dossier WAR Card | ⛔ Withdrawn | Hidden (no Tier 5 data exported) |

## Code Coverage

| Frontend | Status |
|----------|--------|
| TypeScript type check (`tsc --noEmit`) | ✅ Passes — zero errors |
| Test suite | ❌ Not configured — no test runner in web/package.json |

| Python | Status |
|--------|--------|
| `pytest` | ✅ see `tests/` (includes `test_leak_fixes.py`) |
| Fix applied | `pythonpath = ["src"]` added to `[tool.pytest.ini_options]`. The `mlb_draft_dashboard` package was never moved — it is at `src/mlb_draft_dashboard` and declared in `pyproject.toml`; it was simply not on the import path. No `pip install -e .` needed. |
| Removed | `tests/test_bootstrap_team_mapping.py` — shelled out to `scripts/bootstrap_team_mapping.py`, which does not exist in this repo (it lives in the parent `vt_baseball/scripts/`). `build_team_crosswalk.py` is not a replacement: no CLI args, reads JSON rosters rather than a SQLite `VTData` table, and is currently untested. |

## Key Metrics (split-audit fixes of 2026-10-29; see URGENT-FIXES.md and README)

| Metric | Value |
|--------|-------|
| Tier 1 MAE, forward by draft year (2024 / 2025) | 120.7 / 114.2 hitters, 113.2 / 117.2 pitchers (if drafted) |
| Tier 2 AUC (hitters), 2024 class re-read | 0.857 |
| Tier 2 AUC (pitchers), 2024 class re-read | 0.843 |
| Tier 3 AUC (debut within 3 seasons), nested CV, 2021-22 classes | 0.70 hitters / 0.75 pitchers (round rate alone: 0.72 / 0.77) |
| Tier 2 ECE, 2024 class re-read (OOF Platt) | 1.4% hitters / 0.6% pitchers |
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
