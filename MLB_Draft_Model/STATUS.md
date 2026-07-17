# MLB Draft Model — Project Status

**Last Updated**: July 14, 2026

## Current State

| Component | Status | Detail |
|-----------|--------|--------|
| MLB Draft API scraper | ✅ Complete | 9,308 picks (2015–2025), 100% person_id coverage |
| TrackMan college data | ✅ Pipeline exists | 8,184 players, 351 metrics each |
| Team mapping (313 codes) | ✅ Complete | Verified against 308+ D1 schools |
| Joined training set | ✅ Built | 608 players (271 H, 337 P) with features + labels |
| Draft prediction model (v1) | 🟡 Next step | XGBoost on training set |
| MiLB outcome data | 🔴 Not started | Need pybaseball or Stathead |
| Historical college stats (pre-2024) | 🔴 Not started | Need baseballr (R) pipeline |
| Streamlit dashboard integration | 🟡 Planned | After v1 model trains |

## Data Assets

- `data/draft/draft_all_picks.json` — 9,308 picks with full flattening
- `data/draft/draft_college_picks.json` — ~2,386 college-only picks from 2020–2025
- `data/training/training_set.json` — 608 joined player records
- `data/training/training_set.csv` — CSV version (351 feature columns)

## Architecture

See `AGENTS.md` for full architecture and build plan.
