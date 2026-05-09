# MLB Draft Model — Phase 1 Completion Plan

> **For Hermes:** Implement task-by-task, commit after each checkpoint.

**Goal:** Close the 4 remaining gaps between the current pipeline and the original Phase 1 spec: trend integration, sample-size floors, `--season` CLI, and the `test_real_data_loads` test.

**Architecture:** The pipeline is already wired end-to-end. Real data flows from DuckDB → R export script → 9 CSV tables → Streamlit dashboard. All 11 tests pass. This plan adds the missing pieces without restructuring.

**Status:** Checkpoints 1-4 are functionally complete. 4 specific items remain.

**Tech Stack:** R (dplyr, jsonlite), Python/DuckDB bridge, Streamlit dashboard (unchanged)

---

## Priority 1: Trend Integration → Reach/Risk Scores

**What's broken:** `trend_delta` is computed per player (line 344/432 of `export_dashboard_data.R`) but never fed into `reach_score` or `risk_score`. A player gaining 2 mph on his fastball from 2025→2026 gets the same reach_score as someone losing 2 mph. The plan explicitly says "Positive trend = bonus in reach_score. Negative trend = penalty in risk_score."

**Where the code lives:** `R/component_scores.R` — `compute_hitter_scores()` and `compute_pitcher_scores()`. The trend column passes through in the export dataframes but is ignored by the scoring functions.

### Task 1.1: Add trend bonus to hitter reach_score

**Objective:** Hitters with positive EV90 trend get a reach_score bonus. Negative EV90 trend feeds into risk_score penalty.

**Files:**
- Modify: `MLB_Draft_Model/R/component_scores.R` (~lines 135-165, `compute_hitter_scores`)

**Step 1: Define the trend adjustment logic**

The `trend_delta` field is `p90_ev_wood_adj(current) - p90_ev_wood_adj(prior)`. A positive number means the hitter is gaining exit velocity year-over-year — that's projectability.

Guardrails (defined BEFORE writing code):
- **Cap:** Trend bonus maxes at +10 points, trend penalty maxes at +10 risk points. This keeps trends from dominating the base scores.
- **Threshold:** Only apply if `abs(trend_delta) > 0.5` mph — noise floor so tiny fluctuations don't trigger rewards/penalties.
- **Scaling:** Linear mapping: 0 mph Δ → 0 points, 4+ mph Δ → +10 reach bonus. -4- mph Δ → +10 risk penalty. Between: proportional.
- **Single-season players:** `trend_delta` is 0 for single-year players (coalesced to 0 at line 361/449). They get no bonus or penalty — correct behavior, no change needed.
- **Missing data:** If `trend_delta` is NA, treat as 0 (no bonus, no penalty).

**Step 2: Modify `compute_hitter_scores()`**

In the `rowwise() |> mutate()` block (after line 163), add:

```r
# Trend adjustment (after existing scores are computed)
trend_delta_val = coalesce(trend_delta, 0),
trend_bonus = if (abs(trend_delta_val) <= 0.5) 0 else {
  if (trend_delta_val > 0) pmin(10, trend_delta_val / 4 * 10) else 0
},
trend_penalty = if (abs(trend_delta_val) <= 0.5) 0 else {
  if (trend_delta_val < 0) pmin(10, abs(trend_delta_val) / 4 * 10) else 0
},
reach_score = pmin(100, reach_score + trend_bonus),
risk_score_raw = risk_score_raw + trend_penalty,
risk_score = pmin(100, risk_score_raw)
```

**Step 3: Verify**

Run on a multi-year player with a positive trend:
```r
# In R, after running the export:
h <- read_csv("exports/dashboard/hitters_board.csv")
multi_year <- h |> filter(!one_season_only_flag, abs(trend_delta) > 1)
head(multi_year |> select(player_name, trend_delta, reach_score))
```

Expected: Players with large positive trend_delta have reach_score boosted above what pure EV percentile would give them. Players with large negative trend_delta have elevated risk_score.

**Guardrail check:** No reach_score exceeds 100. No risk_score exceeds 100. Single-year players are unchanged.

---

### Task 1.2: Add trend bonus to pitcher reach_score

**Objective:** Pitchers with positive FB velo trend get a reach_score bonus. Negative velo trend feeds into risk_score.

**Files:**
- Modify: `MLB_Draft_Model/R/component_scores.R` (~lines 226-245, `compute_pitcher_scores`)

**Step 1: Same guardrails as hitters**

Same caps (+10 reach bonus, +10 risk penalty), same noise floor (0.5 mph), same linear scaling.

**Step 2: Modify `compute_pitcher_scores()`**

In the `rowwise() |> mutate()` block (after line 243), add the same trend block as hitters but referencing the pitcher's `trend_delta` column (which is ΔFB velo, computed at line 432 of export script).

```r
trend_delta_val = coalesce(trend_delta, 0),
trend_bonus = if (abs(trend_delta_val) <= 0.5) 0 else {
  if (trend_delta_val > 0) pmin(10, trend_delta_val / 4 * 10) else 0
},
trend_penalty = if (abs(trend_delta_val) <= 0.5) 0 else {
  if (trend_delta_val < 0) pmin(10, abs(trend_delta_val) / 4 * 10) else 0
},
reach_score = pmin(100, reach_score + trend_bonus),
risk_score_raw = risk_score_raw + trend_penalty,
risk_score = pmin(100, risk_score_raw)
```

**Step 3: Verify**

```r
p <- read_csv("exports/dashboard/pitchers_board.csv")
multi_year <- p |> filter(!one_season_only_flag, abs(trend_delta) > 1)
head(multi_year |> select(player_name, trend_delta, reach_score, risk_score))
```

**Guardrail check:** Same as hitters — no score exceeds 100, single-year players unchanged.

---

### Task 1.3: Run full export and verify trend changes are visible

**Step 1:** Run the export pipeline:
```bash
cd ~/baseball/MLB_Draft_Model
rm -f exports/dashboard/*.csv
Rscript R/export_dashboard_data.R
```

**Step 2:** Verify the exports generated:
```bash
ls -la exports/dashboard/
```

**Step 3:** Spot-check 3 multi-year players to confirm trend delta is affecting scores:
```r
h <- read_csv("exports/dashboard/hitters_board.csv")
# Find multi-year hitters with non-zero trend
h |> filter(!one_season_only_flag, trend_delta != 0) |>
  select(player_name, trend_delta, reach_score, risk_score) |>
  head(10)
```

Expected: Players with +3 mph trend have visibly higher reach_score than similar-EV players with flat trends.

**Verification:** Trend delta appears as a real factor in scores, not just a data column. Multi-year players get differentiated from single-year players on projectability.

---

## Priority 2: Sample-Size Floors in Export

**What's broken:** The export script loads ALL players from DuckDB, then filters by season later. Players with 8 pitches or 3 PA appear on the draft board with noisy, unreliable scores. The baselines script applies `--min-pitches 100` / `--min-pa 50`, but the main export does not.

**Where the code lives:** `R/export_dashboard_data.R` — no minimum sample filter exists anywhere in the hitter or pitcher aggregation pipeline.

### Task 2.1: Add minimum sample filters before board export

**Objective:** Before writing `hitters_board` and `pitchers_board`, filter out players below minimum sample thresholds. These players still get scored (so their data flows through) but don't appear on the draft board.

**Files:**
- Modify: `MLB_Draft_Model/R/export_dashboard_data.R` (~line 365, before the scoring block for hitters; ~line 452, before the scoring block for pitchers)

**Step 1: Define thresholds**

Guardrails:
- **Pitchers:** minimum 50 pitches across all seasons. This is lower than the baselines threshold (100) because a pitcher with 75 pitches is still worth scoring — just with higher risk.
- **Hitters:** minimum 25 plate appearances across all seasons. Same logic — lower than baselines threshold (50).
- **Apply AFTER aggregation, BEFORE scoring.** The aggregation step computes per-player-season metrics with all data. The filter removes players who don't meet the cumulative threshold.
- **Don't filter the detail tables.** `hitter_bbe_detail` and `pitcher_pitchtype_detail` should still include all players for completeness.

**Step 2: Add filter to current_hitters pipeline**

After `current_hitters` is constructed (before line 365 scoring block), add:

```r
# ── Apply sample-size floor before scoring ──
current_hitters <- current_hitters |>
  filter(plate_events >= 25)
message(sprintf("Hitters after 25 PA floor: %d", nrow(current_hitters)))
```

**Step 3: Add filter to current_pitchers pipeline**

After `current_pitchers` is constructed (before line 452 scoring block), add:

```r
# ── Apply sample-size floor before scoring ──
current_pitchers <- current_pitchers |>
  filter(pitch_count >= 50)
message(sprintf("Pitchers after 50 pitch floor: %d", nrow(current_pitchers)))
```

**Step 4: Run export and verify**

```bash
cd ~/baseball/MLB_Draft_Model
rm -f exports/dashboard/*.csv
Rscript R/export_dashboard_data.R
```

```r
h <- read_csv("exports/dashboard/hitters_board.csv")
p <- read_csv("exports/dashboard/pitchers_board.csv")
cat("Hitters:", nrow(h), " — min plate_events:", min(h$plate_events), "\n")
cat("Pitchers:", nrow(p), " — min pitch_count:", min(p$pitch_count), "\n")
```

Expected: `min(h$plate_events) >= 25`, `min(p$pitch_count) >= 50`.

**Guardrail check:** Board counts should drop from ~12K to a more reasonable number. The lowest-PA players (who had the noisiest scores) should be gone. Detail tables (`hitter_bbe_detail`, `pitcher_pitchtype_detail`) should still have all players.

---

## Priority 3: `--season` CLI Flag

**What's broken:** The Python bridge script (`export_draft_source.py`) supports `--season`, but the R export script never passes it through. You can't export only 2025 players.

**Where the code lives:** `R/export_dashboard_data.R` ~line 234-237 (bridge command construction). The `args` parser already exists. Just needs to pass `--season` through.

### Task 3.1: Wire `--season` from R script to Python bridge

**Objective:** Running `Rscript R/export_dashboard_data.R --season=2025` exports only 2025 players.

**Files:**
- Modify: `MLB_Draft_Model/R/export_dashboard_data.R` (line 234-237)

**Step 1: Read `--season` from args**

The `parse_args()` function already handles `--key=value` syntax. Just check for it:

```r
season_arg <- args[["season"]]  # NULL if not provided
```

**Step 2: Pass to bridge command**

Replace the bridge command construction (lines 234-237):

```r
season_flag <- if (!is.null(season_arg)) paste("--season", season_arg) else ""
bridge_cmd <- paste("python3", shQuote(bridge_script), limit_arg, season_flag)
```

**Step 3: Verify**

```bash
cd ~/baseball/MLB_Draft_Model
Rscript R/export_dashboard_data.R --season=2025
```

```r
h <- read_csv("exports/dashboard/hitters_board.csv")
cat("Seasons in export:", unique(h$season), "\n")
cat("Rows:", nrow(h), "\n")
```

Expected: Only `2025` appears in `unique(h$season)`. Row counts are lower than the full export.

**Guardrail check:** Default (no `--season`) still works and exports all seasons.

---

## Priority 4: `test_real_data_loads`

**What's broken:** The Phase 1 plan specifies a test that verifies real Parquet/CSV exports can be loaded by `data_access.py` without error. It was never written.

**Where the code lives:** `tests/` directory. `data_access.py` already has `load_dashboard_bundle()`.

### Task 4.1: Write `test_real_data_loads`

**Objective:** A pytest test that loads the exported dashboard CSV files using `data_access.load_dashboard_bundle()` and verifies all 9 tables have data.

**Files:**
- Create: `MLB_Draft_Model/tests/test_real_data_loads.py`

**Step 1: Write the test**

```python
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_draft_dashboard.data_access import load_dashboard_bundle
from mlb_draft_dashboard.config import EXPORTS_DIR


def test_load_dashboard_bundle_has_all_tables():
    """All 9 export tables should load without error and contain data."""
    bundle = load_dashboard_bundle(EXPORTS_DIR)

    expected_tables = {
        "hitters_board",
        "pitchers_board",
        "player_trends",
        "hitter_bbe_detail",
        "pitcher_pitchtype_detail",
        "benchmarks_acc_sec",
        "explanations",
        "diagnostics",
        "qa",
    }

    missing = expected_tables - set(bundle.keys())
    assert not missing, f"Missing tables: {missing}"

    for table_name, df in bundle.items():
        assert not df.empty, f"Table '{table_name}' is empty"


def test_hitters_board_has_required_columns():
    """Hitters board must have all columns from REQUIRED_COLUMNS contract."""
    from mlb_draft_dashboard.contracts import REQUIRED_COLUMNS

    bundle = load_dashboard_bundle(EXPORTS_DIR)
    df = bundle["hitters_board"]

    required = REQUIRED_COLUMNS["hitters_board"]
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_pitchers_board_has_required_columns():
    """Pitchers board must have all columns from REQUIRED_COLUMNS contract."""
    from mlb_draft_dashboard.contracts import REQUIRED_COLUMNS

    bundle = load_dashboard_bundle(EXPORTS_DIR)
    df = bundle["pitchers_board"]

    required = REQUIRED_COLUMNS["pitchers_board"]
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"Missing columns: {missing}"
```

**Step 2: Run the test**

```bash
cd ~/baseball/MLB_Draft_Model
PYTHONPATH=src pytest tests/test_real_data_loads.py -v
```

Expected: 3 tests pass.

**Step 3: Run full suite**

```bash
cd ~/baseball/MLB_Draft_Model
PYTHONPATH=src pytest tests/ -v
```

Expected: 14 tests pass (11 existing + 3 new).

**Guardrail check:** Tests must pass even if exports are CSV (not Parquet). `data_access.read_export()` already handles CSV fallback via `resolve_export_path`.

---

## Checkpoint 5: Final End-to-End Verification

After all 4 priorities are complete:

### Task 5.1: Clean export + full test run

```bash
cd ~/baseball/MLB_Draft_Model
rm -f exports/dashboard/*.csv
Rscript R/export_dashboard_data.R
PYTHONPATH=src pytest tests/ -v
```

**Verification:** Export completes without errors. All 14 tests pass.

### Task 5.2: Dashboard sanity check

```bash
cd ~/baseball/MLB_Draft_Model
PYTHONPATH=src streamlit run app/streamlit_app.py
```

**Verification checklist:**
- [ ] Hitters board loads with real player names and scores
- [ ] Pitchers board loads with real player names and scores
- [ ] Multi-year players have trend_delta ≠ 0 and differentiated reach/risk scores
- [ ] No players with <25 PA (hitters) or <50 pitches (pitchers)
- [ ] Filters work (conference, search, sort)
- [ ] Player detail shows EV/LA scatter, trends, pitch type breakdown
- [ ] Notes save and persist

### Task 5.3: `--season=2025` export

```bash
Rscript R/export_dashboard_data.R --season=2025
```

**Verification:** Only 2025 data in boards. Export succeeds without error.

---

## Phase 1 Completion Criteria (Updated)

- [x] Export script reads real DuckDB data (not demo SQLite)
- [x] All 9 CSV tables generated with real player names and scores
- [x] Component scores use population-percentile ranking (not heuristic bins)
- [ ] Multi-year trend deltas feed into reach_score (bonus) and risk_score (penalty)
- [ ] Sample-size floors applied (25 PA hitters, 50 pitch pitchers)
- [ ] `--season` CLI flag functional
- [x] Dashboard loads without errors, all pages functional
- [ ] All tests pass including `test_real_data_loads` (14 total)
- [ ] Top 10 board rankings pass gut-check (recognizable draft prospects at the top)

---

## Guardrails Summary

| Guardrail | Check |
|---|---|
| No reach_score > 100 from trend bonus | Verify in `component_scores.R` — `pmin(100, ...)` |
| No risk_score > 100 from trend penalty | Verify in `component_scores.R` — `pmin(100, ...)` |
| Single-year players unchanged by trends | `trend_delta` is 0 for them → bonus/penalty = 0 |
| Noise floor: |Δ| < 0.5 mph → no adjustment | Guard in trend logic |
| Sample floors don't affect detail tables | Filters only on `current_hitters`/`current_pitchers` |
| `--season` absent = all seasons (backward compat) | Check `is.null(season_arg)` |
| Tests pass with CSV exports | `resolve_export_path` already handles `.csv` fallback |
| Trend bonus cap at +10 | `pmin(10, ...)` in trend logic |
