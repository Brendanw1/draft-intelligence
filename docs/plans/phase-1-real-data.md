# MLB Draft Model — Phase 1 Implementation Plan

> **For Hermes:** Implement task-by-task, commit after each checkpoint.

**Goal:** Replace demo data with the user's real 3M-row TrackMan Parquet dataset, build proper feature engineering (not heuristics), and validate the Streamlit dashboard loads real players with meaningful draft-relevant scores.

**Architecture:** The R export script (`R/export_dashboard_data.R`) currently reads a demo SQLite database and computes component scores (Reach, Impact, Contact, Stuff, Command, Risk) using percentile-bin heuristics. Phase 1 rewrites the data ingestion to read the user's Parquet files via the Python bridge (`pq_bridge.py`), replaces heuristic scoring with statistically-grounded feature engineering, and outputs the same 13 Parquet tables the dashboard expects. The Streamlit dashboard code does NOT change — only the data feeding it changes.

**Tech Stack:** R (dplyr, mgcv), Python/pyarrow bridge, Parquet I/O, Streamlit dashboard (unchanged)

**Data:** `~/baseball/parquet/` — 3.07M rows, 332 teams, 2025 (full) + 2026 (partial through Apr 26)

---

## Checkpoint 1: Audit Current Pipeline

### Task 1.1: Map the export schema end-to-end

**Objective:** Understand exactly what columns the dashboard expects and what the R script currently outputs.

**Files:**
- Read: `MLB_Draft_Model/src/mlb_draft_dashboard/contracts.py` (REQUIRED_COLUMNS)
- Read: `MLB_Draft_Model/R/export_dashboard_data.R` (full file, trace output sections)
- Read: `MLB_Draft_Model/exports/dashboard/` (inspect demo Parquet files)

**Step 1:** Extract REQUIRED_COLUMNS from contracts.py

Run:
```python
import sys; sys.path.insert(0, '~/baseball/MLB_Draft_Model/src')
from mlb_draft_dashboard.contracts import REQUIRED_COLUMNS
for table, cols in REQUIRED_COLUMNS.items():
    print(f"{table}: {cols}")
```

**Step 2:** Trace every output write in export_dashboard_data.R — map each `write_parquet()` or `write_csv()` call to its corresponding REQUIRED_COLUMNS entry. Flag any columns present in contracts but missing from the R export (or vice versa).

**Step 3:** Open one demo Parquet file (e.g., `hitters_board.parquet`) and verify column names match the contracts.

**Verification:** A markdown table mapping: R output table → contracts table name → all columns present? → any gaps.

---

### Task 1.2: Identify the heuristic scoring functions

**Objective:** Find every percentile-bin, z-score, or hardcoded-threshold scoring formula in the R script that needs to be replaced with proper feature engineering.

**Files:**
- Read: `MLB_Draft_Model/R/export_dashboard_data.R` (search for `percent_rank`, `ntile`, `scale_0_100`, `case_when`)

**Step 1:** Search for scoring patterns:
```r
grep -n "percent_rank\|ntile\|scale_0_100\|case_when.*score\|quantile" R/export_dashboard_data.R
```

**Step 2:** Document each scoring function: what it computes, what inputs it uses, what the current formula is, and what a proper replacement would look like.

**Step 3:** Categorize into:
- **Keep as-is**: Simple aggregations (e.g., `avg_fb_velo = mean(RelSpeed)` — this is correct)
- **Replace with stats**: Heuristic scores that should become percentile ranks vs league (e.g., `impact_score = ntile(ev90, 100)`)
- **Replace with model**: Scores that should eventually come from a trained model (e.g., `risk_score` — currently a formula, should be a model predicting injury/bust probability)

**Verification:** A markdown table: function name → current formula → category → replacement strategy.

---

## Checkpoint 2: Wire Real Data

### Task 2.1: Create a Parquet-to-R data loader for the draft model

**Objective:** The R export script currently opens a SQLite database directly. Replace the `dbConnect(RSQLite::SQLite(), ...)` calls with the existing Parquet bridge (`load_baseball.R`).

**Files:**
- Create: `MLB_Draft_Model/R/load_draft_data.R`
- Modify: `MLB_Draft_Model/R/export_dashboard_data.R` (data loading section)

**Step 1:** Create `load_draft_data.R` that sources the bridge and provides:
```r
source("~/baseball/load_baseball.R")

load_draft_pitchers <- function(min_pitches = 100) {
  # Load ALL pitchers from both 2025 and 2026 parquet master tables
  # Returns a single dataframe with season column
}

load_draft_batters <- function(min_pa = 50) {
  # Same, for batters
}
```

**Step 2:** Handle the cross-year loading pattern (the bridge reads one file at a time — need `bind_rows()` across both years).

**Step 3:** Verify the loader returns data:
```r
source("R/load_draft_data.R")
pitchers <- load_draft_pitchers(min_pitches = 100)
print(paste("Pitchers:", nrow(pitchers)))
# Expected: >5000 pitchers
```

**Verification:** Script runs without error, returns 3M+ row dataframe with season column.

---

### Task 2.2: Adapt the export script to use real data

**Objective:** Replace the demo SQLite path and demo-specific filters with the real Parquet loader.

**Files:**
- Modify: `MLB_Draft_Model/R/export_dashboard_data.R` (top ~100 lines — config, DB connection, data loading)

**Step 1:** Replace hardcoded DB path with Parquet loader calls.

**Step 2:** Remove demo-only filters (the demo script filters to a small subset of teams). The real script should process ALL available players.

**Step 3:** Add a `--season` CLI flag to control which seasons to export (default: all).

**Step 4:** Add sample-size floors:
- Pitchers: minimum 100 pitches total across seasons
- Hitters: minimum 50 plate appearances
- These replace the demo's curated small-sample picks

**Verification:** Run with `--season=2025` on a small sample, confirm the output Parquet files are generated with real player names (not demo data).

---

## Checkpoint 3: Build Proper Features

### Task 3.1: Build league-wide percentile baselines

**Objective:** Replace hardcoded thresholds with data-driven percentile ranks computed from the full dataset.

**Files:**
- Create: `MLB_Draft_Model/R/build_baselines.R`
- Modify: `MLB_Draft_Model/R/export_dashboard_data.R` (scoring section)

**Step 1:** `build_baselines.R` computes league-wide percentiles for every key metric, grouped by role (pitcher/hitter) and optionally by conference tier:
```r
compute_baselines <- function(df, role) {
  # For each metric, compute 10th, 25th, 50th, 75th, 90th percentiles
  # Save as JSON for reproducibility
}
```

**Step 2:** Generate baselines once, save to `exports/baselines/`. The export script loads them rather than recomputing.

**Step 3:** Replace `scale_0_100()` calls with `percent_rank()` against the stored baselines. A hitter with EV90 at the 92nd percentile gets `impact_score = 92`.

**Step 4:** For multi-component scores (e.g., `draft_value_score`), use a weighted average of percentile components rather than the current arbitrary formula.

**Verification:** Run baselines, run export, spot-check 3 players — their scores should reflect where they rank in the full population, not arbitrary thresholds.

---

### Task 3.2: Implement proper component score definitions

**Objective:** Define what each component score actually measures and implement it correctly.

**Files:**
- Create: `MLB_Draft_Model/R/component_scores.R`
- Modify: `MLB_Draft_Model/R/export_dashboard_data.R` (replace inline scoring)

**Component definitions:**

| Score | What it measures | Formula |
|---|---|---|
| **Reach** | Physical projectability (frame, velo ceiling, EV ceiling) | Weighted blend: FB velo percentile (0.4) + max EV percentile (0.3) + age-adjusted projection (0.3) |
| **Impact** (hitters) | Game power potential | EV90 percentile (0.5) + barrel proxy percentile (0.3) + max EV percentile (0.2) |
| **Contact** (hitters) | Hit tool quality | (1 − Whiff%) percentile (0.4) + Contact% percentile (0.3) + Chase% inverse percentile (0.3) |
| **Stuff** (pitchers) | Pitch quality independent of location | IVB percentile (0.3) + HB percentile (0.2) + velo percentile (0.3) + spin percentile (0.2) |
| **Command** (pitchers) | Ability to locate and generate strikes | Zone% percentile (0.4) + CSW% percentile (0.3) + (1 − BB%) percentile (0.3) |
| **Risk** | Data sparsity, one-season-only flag, injury flags | Composite: data completeness (0.4) + single-season penalty (0.3) + trend negativity penalty (0.3) |

**Step 1:** Implement `compute_hitter_scores(df, baselines)` and `compute_pitcher_scores(df, baselines)`.

**Step 2:** Each returns a dataframe with `player_uid`, `reach_score`, `impact_score`/`stuff_score`, `contact_score`/`command_score`, `risk_score`, `draft_value_score`.

**Step 3:** `draft_value_score = weighted.mean(c(reach, impact/stuff, contact/command), c(0.25, 0.40, 0.35)) − risk_penalty`.

**Verification:** Run on a single player, print the component breakdown. Verify each score is 0-100, risk_score is lower for multi-year players with complete data, and the draft_value_score ordering passes a sanity check (top 10 should include known draft prospects).

---

### Task 3.3: Build trend features

**Objective:** Multi-year players should get trend deltas (velocity gain, EV progression) that feed into reach and risk scores.

**Files:**
- Modify: `MLB_Draft_Model/R/export_dashboard_data.R` (player_trends section)

**Step 1:** For players appearing in both 2025 and 2026:
- Pitchers: compute ΔFB velo, ΔIVB, ΔCSW%
- Hitters: compute ΔEV90, ΔContact%, ΔChase%

**Step 2:** Add `trend_delta` to the board exports. Positive trend = bonus in `reach_score`. Negative trend = penalty in `risk_score`.

**Step 3:** Generate `player_trends` table with per-season metric values for the trend chart in the dashboard.

**Verification:** A multi-year player shows trend lines in the dashboard. A single-season player shows a note about limited history.

---

## Checkpoint 4: Validate End-to-End

### Task 4.1: Run full export and verify dashboard loads

**Objective:** Generate real exports and confirm the Streamlit dashboard opens, displays real players, and all tabs work.

**Files:**
- Run: `MLB_Draft_Model/R/export_dashboard_data.R` (full pipeline)
- Run: `MLB_Draft_Model/app/streamlit_app.py` (dashboard)

**Step 1:** Run the full export:
```bash
cd ~/baseball/MLB_Draft_Model
Rscript R/export_dashboard_data.R --db_path="" --output_dir=exports/dashboard
```

**Step 2:** Verify output files exist and have reasonable row counts:
```python
import pandas as pd
h = pd.read_parquet('exports/dashboard/hitters_board.parquet')
p = pd.read_parquet('exports/dashboard/pitchers_board.parquet')
print(f"Hitters: {len(h)}, Pitchers: {len(p)}")
print(h[['player_name','reach_score','impact_score','contact_score']].head())
```

**Step 3:** Launch the dashboard:
```bash
cd ~/baseball/MLB_Draft_Model
PYTHONPATH=src streamlit run app/streamlit_app.py
```

**Step 4:** Verify: boards load, filters work, player detail shows score summary and EV/LA scatter, trends render, notes save. Spot-check 5 random players — do the scores make sense?

**Verification:** Dashboard loads without errors, all 5 pages functional, scores are in 0-100 range, top-ranked players are recognizable names.

---

### Task 4.2: Run the existing test suite against real data

**Objective:** Ensure the validation contracts still pass with real data.

**Files:**
- Run: `MLB_Draft_Model/tests/`

**Step 1:** Run tests:
```bash
cd ~/baseball/MLB_Draft_Model
PYTHONPATH=src pytest tests/ -v
```

**Step 2:** Fix any contract violations (column name mismatches, missing required columns, type errors).

**Step 3:** Add a new test `test_real_data_loads` that verifies the Parquet exports can be loaded by data_access.py without error.

**Verification:** All tests pass, including the new real-data test.

---

## Phase 1 Completion Criteria

- [ ] Export script reads real Parquet data (not demo SQLite)
- [ ] All 13 Parquet tables generated with real player names and scores
- [ ] Component scores are percentile-ranked against league baselines (not heuristic bins)
- [ ] Multi-year trend deltas computed for returning players
- [ ] Dashboard loads without errors, all pages functional
- [ ] All existing tests pass + new real-data test
- [ ] Top 10 board rankings pass a gut-check (recognizable draft prospects at the top)
