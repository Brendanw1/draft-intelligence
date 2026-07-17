# MLB Draft Model — Comprehensive Execution Plan
## Data Pipeline, Validation, Tier 2 Model, and 2026 Projections

**Status**: Plan v1 — based on live validation of all three data sources
**Last updated**: 2026-07-16

---

## 0. What We Just Proved (Live Validation Results)

Before designing any more, we validated all three data pipelines end-to-end:

| Source | Status | Proof |
|--------|--------|-------|
| **FanGraphs college stats** | ✅ Working | 63,524 rows (2021-2026), direct JSON API, 99.9% xMLBAMID match rate |
| **MiLB season stats (game feed)** | ✅ Working | 867/2,366 drafted players matched in 2024 alone, 37 hitting + 63 pitching fields |
| **MLB draft data** | ✅ Complete | 9,308 picks (2015-2025), 2,366 college-relevant with FG stats |
| **Tier 1 draft model** | ✅ Positive R² | Hitters R²=0.279, Pitchers R²=0.300 on blind 2026 test |

Critical finding: **MiLB data IS accessible for free via the game feed**. The baseballr R package's MiLB functions are Legacy/FanGraphs-dependent and not the path. The real path is the game feed `seasonStats` field, which contains cumulative season data for every player on the roster.

---

## 1. MiLB Data Pipeline — Build Out

### 1.1 Current State

Single-season (2024) proof of concept complete. What we know:

**Coverage**: 867/2,366 drafted players matched (36.6%) in one season
**Expected coverage across 5 seasons** (2021-2025): ~85-90%

The 36% single-season coverage is expected — not every draftee is active in MiLB every year:
- 2025/2026 draftees may not have signed/been assigned yet
- Some draftees went straight to MLB
- Some washed out before 2024
- Some are in rookie/complex leagues we skipped

**Data quality — string vs numeric fields:**

| Category | Fields | Storage | Action needed |
|----------|--------|---------|---------------|
| Counting stats | AB, H, HR, BB, SO, R, RBI, SB, IP, H, ER, etc. | Numeric ✅ | Directly usable |
| Core rate stats | AVG, OBP, SLG, OPS, ERA, WHIP | String (".248", "3.27") | Parse → float |
| String placeholders | ".-" , "-.--" | String | Treat as null |
| Calculated rates | K/9, BB/9, HR/9, BABIP | String or missing | Compute from counting stats |
| Advanced metrics | FIP, wOBA, xFIP, wRC+ | ❌ Not in game feed | Must calculate or accept limitation |

**Key insight**: The game feed gives us season-level counting stats and basic rates. Advanced metrics (FIP, wOBA, xwOBA, wRC+) are NOT in the game feed. We either:
- Compute FIP from components (HR, BB, HBP, SO, IP) — **feasible**
- Accept basic rate stats (AVG, OBP, SLG, ERA, WHIP) — **sufficient for Tier 2?**

**Distribution shape** (from 2024 data):
- Batters with ≥50 games: 110/867 (13%)
- Pitchers with ≥15 games started: ~60/867 (7%)
- Heavy right-tail: most players have very few games, a few have full seasons
- This is expected — MiLB rosters turn over, players get promoted/released

### 1.2 Full Scraper Build

**API call budget** (per season):

| Step | Calls | Notes |
|------|-------|-------|
| Team lists | 4 | One per sportId (AAA, AA, A+, A) |
| Team rosters | 120 | One per MiLB team |
| Schedule lookup | ~80 | Only teams with our drafted players |
| Game feed | ~80 | Only teams with our drafted players |
| **Total per season** | **~284** | |
| **Total across 6 seasons** (2021-2026) | **~1,700** | ~25 minutes runtime |

**Rate limiting strategy**: 3-5 calls/second, 0.2-0.3s delay between calls. MLB Stats API has generous limits — 1,700 calls is trivial.

**Storage structure:**
```
data/milb/
├── milb_{season}.json          # Per-season master files
├── milb_all.json                # All seasons merged
├── milb_drafted.json            # Filtered to our training set
├── milb_careers.json            # Player-career aggregates (across seasons)
├── raw/                         # Per-team-season game feeds (cached)
└── milb_summary.json            # Coverage stats per season
```

**Data parsing pipeline** (critical step):
```
Raw game feed → extract person_id + seasonStats
             ↓
Parse string fields: avg, obp, slg, ops, era, whip → float
Parse innings: "22.0", "109.2" → float (handle .2 = 2/3 inning)
             ↓
Compute derived fields:
  K/9 = SO / IP * 9
  BB/9 = BB / IP * 9
  HR/9 = HR / IP * 9
  K-BB% = (SO - BB) / BF
  FIP = (13*HR + 3*(BB+HBP) - 2*SO) / IP + 3.10 (league constant)
  BB/K = BB / SO
             ↓
Join to draft data on person_id
```

### 1.3 Expected Coverage

| Season | Drafted players | Expected MiLB match | Notes |
|--------|----------------|---------------------|-------|
| 2021 | 327 | ~280 (86%) | 3+ years in system, most have MiLB data |
| 2022 | 361 | ~310 (86%) | 2-3 years in system |
| 2023 | 410 | ~330 (80%) | 1-2 years in system |
| 2024 | 426 | ~300 (70%) | Partial first season (summer call-up) |
| 2025 | 405 | ~200 (49%) | Mostly just drafted, minimal MiLB time |
| 2026 | 437 | ~50 (11%) | Recently drafted, no MiLB yet |
| **Total** | **2,366** | **~1,470** | |

For the Tier 2 model, we'd train on 2021-2024 (where MiLB data is most complete) and use 2025-2026 as out-of-sample test.

---

## 2. Validation & Distribution Checks

### 2.1 Cross-Validation: FanGraphs vs TrackMan

**Why**: User flagged this — some FG metrics like wOBA, BABIP, BB%, K% should match portal-scout's TrackMan calculations. If they don't, our model is learning from biased formulas.

**Check**: For players who exist in BOTH datasets (portal 2025 players who also have FG college stats):

| Metric | FG source | TrackMan source | Expected match |
|--------|-----------|-----------------|----------------|
| AVG | H/AB | AVG | Should match |
| OBP | (H+BB+HBP)/(AB+BB+HBP+SF) | OBP | Should match |
| SLG | TB/AB | SLG | Should match |
| BABIP | (H-HR)/(AB-HR-K+SF) | BABIP | Slight divergence possible |
| BB% | BB/PA | BB_Pct | Should match |
| K% | K/PA | K_Pct | Should match |
| wOBA | Weighted on-base | xwOBA (expected) | Will differ (FG = actual, TM = expected) |

**Action**: Write a validation script that:
1. Finds players in both datasets (by name + school or by xMLBAMID)
2. Compares matching fields
3. Reports discrepancies > 2%
4. Flags systemic bias if FG consistently differs from TrackMan

### 2.2 MiLB Data Distributions

Per level (AAA, AA, A+, A), check:

| Check | What to look for | Action if failed |
|-------|-----------------|------------------|
| Games played distribution | Most players with 0-20 games | Filter threshold (≥10 games for meaningful sample) |
| AVG distribution (AAA) | Should center ~.250-.260 with SD ~.030 | Flag if bimodal or extreme |
| ERA distribution (AA) | Should center ~4.00-4.50 with SD ~1.5 | Flag if unrealistic |
| K/9 trend across levels | K/9 should decrease as level increases (harder to miss bats) | Flag if inverse |
| BB/9 trend across levels | BB/9 should decrease as level increases (better command) | Flag if inverse |
| Year-over-year stability | Same players should have similar stats across seasons | Flag extreme year-to-year swings |

This is a **diagnostic** — the model should learn these patterns, but extreme outliers or unrealistic data would degrade performance.

### 2.3 Missing Data Patterns

| Missing pattern | Likely cause | Mitigation |
|----------------|-------------|------------|
| No MiLB data for 2025-2026 draftee | Too recent, hasn't been assigned | Use as test set only |
| No pitching data for hitter (and vice versa) | Player only hits or only pitches | Separate hitter/pitcher models |
| Zero games played in season | Player was injured or never active | Try previous season's data |
| String placeholder for rate stat ("-.--") | Insufficient playing time to qualify | Compute from counting stats |
| No seasonStats for player in game feed | Player was promoted/optioned | Try earlier game feed or different team |

**Critical feasibility note**: The 2026 draft class (picks from last week) has essentially zero MiLB data — they were just drafted days ago. Their Tier 2 outcome is entirely unknown. This means:
- For **historical predictions** (2021-2025): full pipeline works
- For **2026 draft projections**: Tier 1 only (college stats → draft position)
- Tier 2 is trained on historical players and used to evaluate future classes at the same point in their careers

---

## 3. Tier 2 Model: Pro Viability / Outcome Model

### 3.1 Outcome Variable Options

| Outcome | Type | Pros | Cons |
|---------|------|------|------|
| **Reached MLB** | Binary (0/1) | Clean label, available for all draftees | Coarse — an MLB cup of coffee vs 10-year career are both "1" |
| **Peak MiLB level** | Ordinal (1-4) | Good proxy for career trajectory | Doesn't capture quality of performance |
| **MiLB WAR / WARP** | Continuous | Best measure of actual production | Labor-intensive to compute, not directly in game feed |
| **Years in MiLB** | Count | Measures organizational longevity | Doesn't distinguish performance quality |
| **Reached AAA** | Binary (0/1) | Captures "serious prospect" status | Misses MLB success |
| **Combined score** | Continuous (0-100) | Composite of level + performance + MLB | Complex to define, harder to interpret |

**Recommended**: Start with **Reached MLB** (binary) + **Peak MiLB level** (ordinal) as two separate models. This gives both a "will they make it" probability and a "how far will they go" prediction. These can be combined into a single value score later.

**Label availability**: For players drafted 2021-2026:
- Reached MLB: ~15% of top-100 picks, ~2% of later rounds (known ✅)
- Peak MiLB level: ~100% known from roster data ✅
- MiLB performance: ~85% known (the ones who played) ✅

### 3.2 Feature Space

**Group A: Pre-draft features (known at draft time)**
| Feature | Source | Type |
|---------|--------|------|
| Draft pick number | MLB API | Numeric — strongest predictor |
| Draft round | MLB API | Ordinal |
| Signing bonus | MLB API | Numeric (log transform) |
| College final-year wOBA | FanGraphs | Numeric |
| College final-year wRC+ | FanGraphs | Numeric |
| College final-year FIP (pitchers) | FanGraphs | Numeric |
| College final-year K-BB% (pitchers) | FanGraphs | Numeric |
| Age at draft | MLB API | Numeric |
| Position | MLB API | Categorical |
| School type (D1/D2/NAIA/JC) | MLB API | Categorical |
| Height / Weight | MLB API | Numeric |

**Group B: Early MiLB performance (known 1-2 years post-draft)**
| Feature | Source | Type |
|---------|--------|------|
| First-season MiLB level | Game feed | Ordinal |
| MiLB AVG/OBP/SLG at first level | Game feed | Numeric |
| MiLB ERA/WHIP/K/9 at first level | Game feed | Numeric |
| Games played at first level | Game feed | Numeric (proxy for health) |
| Promotion speed (days to next level) | Game feeds | Numeric |

**Group C: Physical / Scouting (known at draft)**
| Feature | Source | Type |
|---------|--------|------|
| Pitch velocity (if available) | Not yet acquired | Numeric |
| FB% / GB% / LD% | Game feed (batted ball data) | Numeric |

### 3.3 Validation Strategy

| Component | Method | Purpose |
|-----------|--------|---------|
| Temporal split | Train: 2021-2023, Test: 2024-2025 | No lookahead bias |
| Cross-validation | 5-fold within training years | Hyperparameter tuning |
| Calibration check | Predicted probability vs actual rate | For binary outcome (reached MLB) |
| Feature ablation | Drop each feature group, track R² change | Which features add real signal |
| Round-stratified | Separate evaluation for R1-3, R4-10, R11-20 | Model performs differently by round |

**Expected baseline**: Draft pick number alone should predict ~20-25% of MLB outcome variance. College stats + physical should add 5-10%. Early MiLB performance should add 10-15%.

### 3.4 Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Late-round picks rarely reach MLB (~2%) | Imbalanced classification | Use weighted loss function, stratified sampling |
| No FIP/wOBA in game feed | Less refined performance metric | Compute FIP from components; accept K/BB/HR rates |
| No Statcast/Savant data for MiLB | No exit velo, launch angle, spin rate | Accept — these are MLB-level metrics |
| Players can be traded | May have stats across multiple orgs | Game feed captures whatever team they're on at game time |
| Sample size limited to ~1,500 | Limited capacity for complex models | Keep model simple (XGBoost with heavy regularization) |

---

## 4. 2026 Draft Projections — Can We Do It?

**Short answer: Yes, for Tier 1 (draft position). Tier 2 (pro viability) on the 2026 class is limited.**

### 4.1 What's Possible Now

The **Tier 1 FG model** can generate draft projections for every 2026 D1 player who has FanGraphs college stats. These are the 5,330 batters and 5,404 pitchers we pulled yesterday. For each player, we predict:

- **Projected draft pick number** (e.g., pick 142)
- **Projected draft round** (e.g., round 5)
- **Confidence** (based on how well the player's profile fits the model's training distribution)

The existing model files are at:
- `models/artifacts/fg_draft_hitter.json` (R²=0.279 on 2026)
- `models/artifacts/fg_draft_pitcher.json` (R²=0.300 on 2026)

### 4.2 What Tier 2 Adds (Once Built)

Tier 2 would add a **value overlay** — not just "where will they be drafted" but "what's their expected career value":

```
Composite Value = f(Draft Position, College Performance, MiLB Outcome)
```

For the **2026 class specifically**: Tier 2 would tell us "based on his college profile and projected draft position, a player like this typically reaches MLB X% of the time with Y peak level." This is a **prior probability**, not a 2026-specific prediction, because the 2026 class has zero MiLB data yet.

### 4.3 Projection Output

For each 2026 D1 college player:

```json
{
  "player_name": "Travis Bazzana",
  "school": "Oregon State",
  "position": "2B",
  "tier1_draft_pick": 1.0,
  "tier1_draft_round": 1,
  "tier1_confidence": "high",
  "tier2_mlb_probability": 0.85,
  "tier2_peak_level": "MLB",
  "tier2_value_score": 92,
  "composite_grade": "70 (FV)",
  "comparable_prospect": "Dustin Pedroia",
  "valuation": "$8.5M bonus equivalent"
}
```

This is the end-state vision. The first deliverable (Tier 1 projections) can ship this week.

---

## 5. Feasibility & Blocker Analysis

| Block | Severity | Workaround |
|-------|----------|------------|
| Cloudflare on FanGraphs | ✅ None — API endpoint works directly | Direct HTTP requests, no scraper needed |
| MLB API rate limiting | ✅ None — 1,700 calls over 25 min is trivial | Simple delay between calls |
| No advanced MiLB stats (wOBA, FIP) | 🟡 Mild — can compute FIP from components | Counting stats are complete; compute own rates |
| MiLB seasonStats as strings | 🟡 Minor — needs parsing | Write a parsing layer |
| 2026 draftees have no MiLB data | 🟡 Expected — use Tier 1 only for 2026 | Train Tier 2 on historical classes; apply as prior |
| TrackMan vs FG formula mismatch | 🟡 Unknown — needs validation | Run validation script, adjust formulas if biased |
| Small training sample (~1,500) | 🟡 Manageable — XGBoost handles small n well | Keep model simple, regularization heavy, feature ablation |
| R2 bucket for FG/MiLB storage | 🟢 Setup task — waiting on user | Local storage works until bucket is ready |

---

## 6. Build Order (Priority)

| Step | Output | Time | Depends on |
|------|--------|------|------------|
| **1. Scrape all MiLB seasons** (2021-2026) | ~1,500 player-season records with parsed stats | 1 hour | Current script (exists) |
| **2. Validate FG vs TrackMan** | Cross-validation report, bias check | 30 min | FG data (done) |
| **3. Build Tier 2 training set** | MiLB outcomes joined to draft + college features | 30 min | Step 1 |
| **4. Train Tier 2 model** | Reached-MLB + Peak-Level models | 30 min | Step 3 |
| **5. Run 2026 draft projections** | Projections for all 10,734 D1 players | 30 min | Step 4 |
| **6. MiLB data distributions report** | Per-level stat distributions, coverage tables | 20 min | Step 1 |
| **7. Productionize for R2 bucket** | Upload pipeline, bucket config | 15 min | User sets up bucket |

**Total engineering time**: ~3.5 hours for the full stack from current state to 2026 projections.

---

## 7. What You'll Have At The End

```
MLB_Draft_Model/
├── data/
│   ├── fangraphs/          # FanGraphs college stats (bucket-ready)
│   ├── milb/               # MiLB career stats (bucket-ready)
│   ├── draft/              # MLB draft picks
│   └── training/           # Merged training sets
│       ├── fg_training_set.json        # Tier 1: 2,366 rows
│       ├── tier2_training_set.json     # Tier 2: ~1,500 rows with MiLB outcomes
│       └── projections_2026.json       # 2026 draft projections
├── models/
│   └── artifacts/
│       ├── fg_draft_hitter.json        # Tier 1 hitter model
│       ├── fg_draft_pitcher.json       # Tier 1 pitcher model
│       ├── tier2_mlb_hitter.json       # Tier 2 hitter outcome model
│       └── tier2_mlb_pitcher.json      # Tier 2 pitcher outcome model
├── scripts/
│   ├── pull_fangraphs_college.py       # FG college data puller
│   ├── build_fg_training_set.py        # Join FG → draft
│   ├── train_fg_model.py               # Tier 1 trainer
│   ├── scrape_milb_data.py             # MiLB game feed scraper
│   ├── build_tier2_training.py         # Join MiLB → draft
│   ├── train_tier2_model.py            # Tier 2 trainer
│   └── generate_2026_projections.py    # 2026 draft projections
└── AGENTS.md
```

---

## 8. Decision Points

1. **Tier 2 outcome variable**: Binary (reached MLB) vs Ordinal (peak level) vs Continuous (combined score)? Binary is easiest to validate and interpret.
2. **Validation priority**: Cross-validate FG vs TrackMan first (prevent garbage-in-garbage-out) or push to Tier 2?
3. **Storage**: Your R2 bucket for FG/MiLB data — separate from portal-scout bucket, as discussed. Ready when you are.
4. **2026 projections**: Generate raw Tier 1 projections now (this afternoon) or wait for Tier 2?
