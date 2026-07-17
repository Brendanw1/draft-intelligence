# MLB Draft Model — Ground Floor & Execution Plan

**Status**: Framework built, data scraped, training set assembled, architecture designed.
**What's running**: Draft data for 9,308 picks (2015–2025), joined training set of 608 players with TrackMan metrics + draft outcomes, team mapping covering 313 D1 codes.

---

## 1. Data Inventory — What We Have

### 1.1 MLB Draft Data — SCRAPED ✅

| Source | Status | Records | Key Fields |
|--------|--------|---------|-----------|
| MLB Stats API `/api/v1/draft/{year}` | ✅ 2015–2025 | 9,308 picks | person_id, full_name, school, pick_round, pick_number, signing_bonus, position, bats/throws, height/weight, birth_date, team, mlb_debut_date |
| College-only subset | ✅ 2020–2025 | ~2,386 | School_class (JR/SR/SO/GR), school_name |
| MLB player IDs | ✅ 100% coverage | 9,308 | person_id — the universal join key to MiLB/MLB stats |

**Stored at**: `MLB_Draft_Model/data/draft/` — `draft_all_picks.json`, `draft_college_picks.json`, per-year files.

**Coverage by year**:

| Year | Picks | College | HS | JC | Signed | Bonus Total |
|------|-------|---------|----|-----|--------|-------------|
| 2015 | 1,214 | 0* | 0* | 0* | 0* | $0 |
| 2016 | 1,216 | 0* | 0* | 0* | 0* | $0 |
| 2017 | 1,215 | 0* | 0* | 0* | 936 | $287M |
| 2018 | 1,214 | 1* | 300 | 103 | 863 | $285M |
| 2019 | 1,217 | 0* | 0* | 0* | 906 | $309M |
| 2020 | 160 | 108 | 47 | 5 | 160 | $238M |
| 2021 | 612 | 447 | 115 | 47 | 566 | $292M |
| 2022 | 616 | 455 | 118 | 41 | 563 | $314M |
| 2023 | 614 | 445 | 124 | 44 | 564 | $350M |
| 2024 | 615 | 474 | 113 | 26 | 570 | $374M |
| 2025 | 615 | 456 | 124 | 31 | 576 | $391M |

*\* school_class field unreliable before 2020 (API schema change). Data still valid for bonus/team/position fields.*

### 1.2 TrackMan College Data — PIPELINE EXISTS ✅

| Dataset | Records | Description |
|---------|---------|-------------|
| `all_hitters.json` | 4,074 | 100+ metrics: wOBA, EV, barrel%, chase%, whiff%, decision scores |
| `all_pitchers.json` | 4,110 | 100+ metrics: velo, IVB, HB, spin, extension, CSW%, Stuff+, Location+ |
| `pitch_type_board.json` | ~28K rows | Per-pitch Stuff+, Location+, weapon scores |
| `fastball_anchor.json` | ~5,300 | FB profiles: VAA, arm angle, extension |
| `composite_rankings.json` | 12,493 | Raw Index, Model Index, OFP, PV |
| `all_rosters.json` | 11,784 | Class year, position, height, hometown |
| `defense_scores.json` | ~10K | Position-adjusted defense |
| Team mapping | 313 codes → full names | 308+ D1 schools |

### 1.3 Joined Training Set — BUILT ✅

**608 players** with both TrackMan metrics AND draft outcomes:

| | Count | % |
|---|---|---|
| Hitters | 271 | 45% |
| Pitchers | 337 | 55% |
| High confidence (name + school match) | 372 | 61% |
| Medium confidence (name-only) | 236 | 39% |
| Has signing bonus | 457 | 75% |
| Has class year | 144 | 24% |
| Has position | 143 | 24% |
| Draft 2025 | 412 | 68% |

**Feature space**: 351 numeric metrics per player (EV, velo, chase%, whiff%, contact%, etc.) + draft labels (round, pick, bonus, team, position).

**Stored at**: `MLB_Draft_Model/data/training/training_set.json` and `.csv`

**Draft round distribution**: Even across R1–R20 (21–45 picks per round).

---

## 2. Data Sources — What Remains to Acquire

### 2.1 Minor League Performance Data 🟡 (The Key Gap)

This is the **outcome variable** for a true draft projection model. Without it, we can only predict draft position (which is itself a proxy, not the ground truth).

| Source | Method | Status | Verdict |
|--------|--------|--------|---------|
| MLB Stats API `people/{id}/stats` | `yearByYear` endpoint | 🟡 Tested | Returns MLB stats only. MiLB data not exposed through this endpoint. |
| `sports/{sportId}/players` | Lists players at each MiLB level | ✅ Works | Gives roster data but not stats. |
| FanGraphs MiLB leaders | Web/CSV export | 🔴 Blocked | Cloudflare bot detection on browser. Direct HTTP returns HTML page, not CSV. |
| **pybaseball** | Python library wrapping FG/BR | 🟡 Untested | `milb_batting_stats()`, `milb_pitching_stats()` — the standard Python interface. Needs install + test. |
| The Baseball Cube | API with MiLB data | 🟡 Untested | Has per-player MiLB stats, accessible via their API. |
| **Stathead** | Paid subscription | 🟡 $8/mo | Clean CSV exports for all MiLB seasons. Most reliable option. |
| Baseball-Reference MiLB | Per-player pages | 🔴 Rate-limited | Scrapable at low volume but impractical for 2,000+ players. |

### 2.2 Historical College Stats 🟡 (For Multi-Year Training)

Our TrackMan data only covers ~2024–2026. For a robust model, we need college stats for players drafted in 2015–2022.

| Source | Method | Status | Verdict |
|--------|--------|--------|---------|
| **baseballr** (R package) | `ncaa_scrape()` functions | 🟡 Untested | The gold standard. Returns individual player stats by team/year. Time-consuming but complete. |
| WarrenNolan.com | Web scrape | 🔴 404 | Team stats exist but individual player stats may be behind different URLs. |
| NCAA.org stats | Web scrape | 🔴 Complex | Needs team-by-team scraping, fragile HTML structure. |
| Chadwick Player ID crosswalk | `baseballr::chadwick_player_lu()` | 🟡 Untested | Links player name/school to MLBAM ID. Key for cross-referencing college → pro. |
| D1Baseball | Paywall | 🔴 | Behind subscription. |

### 2.3 Draft Scouting Grades 🟡 (Optional)

| Source | Method | Status |
|--------|--------|--------|
| MLB.com pre-draft rankings | Already in API as `rank` | ✅ Included |
| MLB Pipeline grades | Web scrape | 🔴 Available but needs browser automation |
| FanGraphs scouting grades | Web scrape | 🔴 Behind Cloudflare |

---

## 3. Architecture — Current State

```
                          ┌─────────────────────┐
                          │  MLB Stats API       │
                          │  /api/v1/draft/{year} │
                          └──────────┬──────────┘
                                     │ 9,308 picks
                                     ▼
                          ┌─────────────────────┐
                          │  Draft Data Store    │
                          │  data/draft/         │  ✅ Complete
                          │  person_id, school,  │
                          │  round, pick, bonus  │
                          └──────────┬──────────┘
                                     │
              ┌──────────────────────┤
              │                      │
              ▼                      ▼
  ┌──────────────────────┐  ┌──────────────────────┐
  │  TrackMan Pipeline   │  │  Join: NameKey       │
  │  PS server/data/2025 │──▶  + school normalizer │
  │  8,184 players       │  │  608 matched         │  ✅ Complete
  │  351 metrics each    │  └──────────┬───────────┘
  └──────────────────────┘            │
                                      ▼
                          ┌──────────────────────┐
                          │  Training Set        │
                          │  data/training/      │  ✅ Complete
                          │  features + labels   │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  XGBoost Model (v1)  │
                          │  Pick/Round/Bonus    │  🟡 Next Step
                          │  prediction          │
                          └──────────────────────┘
```

---

## 4. Model Strategy — Two-Tier Approach

### Tier 1: Draft Position Model (NOW — n=608)

Predict draft pick number / round from TrackMan college metrics.

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Target | `pick_number` (regression) | Continuous, captures within-round ordering |
| Alternatively | `pick_round` (ordinal) | Coarser but less noise from bonus-pool shenanigans |
| Algorithm | XGBoost Regressor | Handles 351 sparse features, gives feature importance, regularizes well on small n |
| Validation | 5-fold cross-validation | n=608 is small enough that a holdout set would starve training. CV gives realistic error estimate. |
| Features | Top 50 by importance | Start with all 351, prune to keep model interpretable |
| Metric | RMSE (picks), MAE (rounds) | Pick error of ±30 is usable; ±15 is excellent |
| Baseline | Mean prediction | Compare against naive "predict middle-of-round" |

**Projected performance**: With 351 features and 608 rows, we're in high-dimensional regime (p > n/2). XGBoost's built-in regularization (`lambda`, `alpha`, `subsample`) is critical. Expect feature importance to converge on 15–25 dominant metrics (velo, EV, chase rate, whiff rate, barrel rate, PV score).

### Tier 2: MiLB Outcome Model (FUTURE — needs more data)

Predict actual pro performance (wOBA, ERA, level reached) rather than draft position.

Requires: MiLB stats for training labels. See Section 2.1.

---

## 5. Data Flow for Model Training

```python
# The training pipeline (next to build):
#
# 1. Load training_set.csv
# 2. Select feature columns (drop IDs, labels, strings)
# 3. Impute missing values (median, or player-type median)
# 4. Scale features (standard or robust scaler)
# 5. Train XGBoost with:
#    - n_estimators=500, early_stopping=50
#    - max_depth=4 (shallow — n=608)
#    - learning_rate=0.05
#    - subsample=0.7, colsample_bytree=0.5
#    - reg_lambda=2.0, reg_alpha=1.0
# 6. Evaluate: 5-fold CV, RMSE, MAE
# 7. Feature importance analysis
# 8. Save model artifact
# 9. Apply to ALL 8,184 portal players → generate draft projections
```

---

## 6. Immediate Next Steps (Priority Order)

### 🔴 High — Execute This Week

1. **Train v1 model** on the training set (608 rows). XGBoost is already installed in the venv. Should take <5 minutes to run.
2. **Generate draft projections** for all 8,184 portal prospects using the trained model.
3. **Update the Streamlit dashboard** to show model predictions alongside existing scores.

### 🟡 Medium — Needs Investigation

4. **Test pybaseball for MiLB stats** — `pip install pybaseball` and check `milb_batting_stats()`. If it works, this replaces all manual MiLB data acquisition.
5. **Get class_year from draft API's schoolClass field** — the draft data has `school_class` (JR/SR/SO). Map this to a standardized class year for the 246 draft-matched players who don't have it from roster join.
6. **Evaluate name-only match quality** — spot-check 20 name-only matches to confirm they're correct (most should be school-format differences).

### 🔵 Longer Term

7. **Historical college stats via baseballr** — run `baseballr::ncaa_scrape()` for 2015–2024 teams to get box-score stats for multi-year training.
8. **Chadwick crosswalk** — link college player names to MLBAM IDs for direct MiLB stat pulls.
9. **MiLB stats acquisition** — once pybaseball is verified, pull MiLB career stats for all 608 training players.
10. **Retrain Tier 2 model** with MiLB outcome as target.

---

## 7. Key Decisions Made

| Decision | Choice | Why |
|----------|--------|-----|
| Target variable (v1) | Draft pick number | Clean signal, 100% coverage for signed players, directly useful |
| Feature source | TrackMan pipeline | 351 metrics per player, already cleaned and normalized |
| Join key | NameKey + school normalizer | Reuse from portal-scout; 61% achieve school-level match |
| Draft data source | MLB Stats API (free, no auth) | 100% person_id coverage, no rate limiting at our volume |
| MiLB data (future) | pybaseball → FanGraphs | Standard Python interface, handles scraping complexity |
| Historical college data (future) | baseballr (R) → Parquet | Most reliable NCAA stats interface |
| Model algorithm | XGBoost | Industry standard for tabular sports data, handles sparse high-dim |
| Team mapping | 313 codes → full names | Already built and tested for portal-scout |

## 8. Files & Where Everything Lives

```
MLB_Draft_Model/
├── data/
│   ├── draft/
│   │   ├── draft_all_picks.json        # 9,308 picks across all years
│   │   ├── draft_college_picks.json    # ~2,386 college-only picks
│   │   ├── draft_2024.json etc.        # Per-year files
│   └── training/
│       ├── training_set.json           # 608 joined player records
│       └── training_set.csv            # CSV version for modeling
├── scripts/
│   ├── mlb_draft_data.py               # Draft scraper ✅
│   ├── build_training_set.py           # TrackMan × Draft join ✅
│   ├── train_draft_model.py            # XGBoost training 🟡 TO BUILD
│   └── [future] get_milb_stats.py      # MiLB acquisition 🟡
├── models/
│   └── artifacts/                      # Saved model files 🟡
├── data/exports/dashboard/             # Existing TrackMan exports
├── app/streamlit_app.py                # Existing dashboard
└── src/mlb_draft_dashboard/            # Dashboard logic
```

```
portal-scout/server/data/2025/
├── all_hitters.json                   # 4,074 hitters ← FEATURES
├── all_pitchers.json                   # 4,110 pitchers ← FEATURES
├── all_rosters.json                    # 11,784 roster records
├── composite_rankings.json            # 12,493 ranked
├── draft_board.json, draft_board2.json # Draft boards
├── drafted.json, drafted_merged.json   # Draft picks
└── team_names.json                     # Team code → school
```

---

*Built by the VT Baseball Analytics team. Draft data from MLB Stats API (statsapi.mlb.com). College data from TrackMan pitch tracking systems via the portal-scout pipeline.*
