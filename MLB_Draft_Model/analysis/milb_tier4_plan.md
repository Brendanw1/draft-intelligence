# MiLB Integration — Execution Plan

## Design Rationale

### The two-model architecture

**Tier 3 (existing):** Predicts P(MLB debut | drafted) from college stats + round prior + NN rate.
  - Use case: Pre-draft — "will this college draftee reach MLB?"
  - Trained on: 2021-2023 draftees with MLB debut outcomes
  - No MiLB data (doesn't exist on draft day)

**Tier 4 (new):** Same binary target (MLB debut) but adds first-year MiLB stats as features.
  - Use case: Post-first-season — "now that we've seen their rookie year, update their odds"
  - Trained on: 2021-2022 draftees with college + 1 year MiLB data
  - Adds: milb_year1_wOBA/FIP, milb_year1_level

**MiLB Outcome Model (new):** Predicts continuous MiLB performance from college stats only.
  - Use case: Pre-draft — "what MiLB performance should we expect from this college profile?"
  - Target: peak wOBA (hitters) or FIP (pitchers) in years 2-3 pro
  - Features: College stats only (same as Tier 1)
  - Trained on: 2021-2022 draftees with 3+ years MiLB history

### Why separate Tier 3 and Tier 4?

Tier 3 works on ALL drafted players (n=1,262). Tier 4 only works on players who have completed a MiLB season (n~1,000). Different populations, different use cases. Side-by-side comparison shows how much MiLB year-1 data adds.

---

## Phase 1: Tier 4 (7 steps, ~4 hours)

### Step 1 — Build MiLB extended training set
**Script:** `scripts/build_milb_training.py`

```
Inputs:
  - data/milb/milb_{2021..2025}.json
  - data/training/expanded_training_set.json

Logic:
  1. Load all 5 milb_{year}.json files into a person_id -> [season_data] index
  2. For each person_id, sort by season (ascending)
  3. Compute per-player:
     - year_1_stats: first season's batting/pitching + level
     - year_2_3_peak: best wOBA (hitter) or best FIP (pitcher) among seasons 2-3
     - highest_level_by_year_3: max level reached within first 3 seasons
     - progression_flag: level increased from year 1 to year 2 to year 3? (up/stable/down)
  4. Join with expanded_training_set.json on person_id
  5. Only keep signed draftees (draft_pick > 0 and not None)
  6. Only keep 2021-2022 draftees for training, 2023 for validation
  7. Apply minimums: 50 PA (hitters), 20 IP (pitchers) in MiLB year 1
  8. Compute derived MiLB stats:
     - For hitters: milb_year1_wOBA, milb_year1_avg, milb_year1_ops
     - For pitchers: milb_year1_FIP, milb_year1_K9, milb_year1_BB9
  9. Add college features from expanded_training_set (same as Tier 3 features)

Validation checks:
  [V1] person_id join rate >= 90% of signed draftees
  [V2] Year distribution: 2021 ~45%, 2022 ~40%, 2023 ~15%
  [V3] Hitter/pitcher split roughly balanced (~50/50)
  [V4] No null targets in year_2_3_peak (every row must have a value)
  [V5] min(PA) >= 50 for hitters, min(IP) >= 20 for pitchers

Output: data/training/milb_extended_training.json
```

### Step 2 — Validate the training set
**Script:** `scripts/validate_milb_training.py`

```
Logic:
  1. Load milb_extended_training.json
  2. Print:
     - Total records, by draft year, by player type
     - Year 1 level distribution (A/A+/AA/AAA)
     - Peak wOBA distribution (hitters) — mean, std, range
     - Peak FIP distribution (pitchers) — mean, std, range
     - Correlation: college wOBA vs MiLB peak wOBA (should be positive)
     - Correlation: draft round vs MiLB peak wOBA (should be negative — earlier = better)
  3. Flag if correlations are in the wrong direction
  4. Print sample of 5 random players showing all computed fields

Validation checks:
  [V1] Correlation(college_wOBA, milb_peak_wOBA) > 0.15
  [V2] Correlation(draft_round, milb_peak_wOBA) < -0.10 (earlier round = better)
  [V3] Mean peak wOBA ~.320 (MiLB average)
  [V4] Mean peak FIP ~4.50 (MiLB average)

Output: Console report only
```

### Step 3 — Train Tier 4 model (Tier 3 + MiLB features)
**Script:** `scripts/train_tier4_milb_arrival.py`

```
Logic:
  1. Load milb_extended_training.json
  2. Filter to players with college + year-1 MiLB data
  3. Split: train on 2021-2022, validate on 2023
  4. Build feature matrix:
     SAME FEATURES as Tier 3:
       - Age, conf_strength, wOBA_adj, ERA_adj, height_inches, bmi
       - round_logit_prior, nn_mlb_rate
     PLUS:
       - milb_year1_wOBA (hitter) or milb_year1_FIP (pitcher)
       - milb_year1_level (1=A, 2=A+, 3=AA, 4=AAA)
       - milb_year1_games (games played — proxy for health/opportunity)
  5. Train Elastic Net logistic regression (same hyperparams as Tier 3)
  6. Compare against baseline Tier 3 trained on same subset:
     - AUC improvement
     - Brier score improvement
     - Feature coefficients (new MiLB features should be top 5)
  7. Save model artifacts to models/artifacts_full/tier4_*.pkl

Validation checks:
  [V1] Tier 4 AUC > Tier 3 AUC on same validation set
  [V2] milb_year1_wOBA/FIP coefficient has expected sign (higher wOBA = positive, higher FIP = negative)
  [V3] Model converges (solver saga, max_iter=2000)

Output: models/artifacts_full/tier4_mlb_hitter.pkl, tier4_mlb_pitcher.pkl, tier4_features_*.json
```

### Step 4 — Comparison report
**Script:** `scripts/compare_tier3_vs_tier4.py`

```
Logic:
  1. Load Tier 3 model + Tier 4 model
  2. For each, compute on 2023 validation set:
     - AUC
     - Brier score
     - Calibration curve (ECE)
     - Top 5 features by coefficient magnitude
  3. Print comparison table
  4. Save to analysis/tier3_vs_tier4_comparison.md

Validation checks:
  [V1] Both models compute successfully
  [V2] Metrics match between runs (deterministic)

Output: analysis/tier3_vs_tier4_comparison.md
```

### Step 5 — Train standalone MiLB Outcome Model
**Script:** `scripts/train_milb_outcome_model.py`

```
Logic:
  1. Load milb_extended_training.json
  2. Filter to 2021-2022 draftees (training) + 2023 (validation)
  3. Target: milb_peak_wOBA (hitters) / milb_peak_FIP (pitchers) in years 2-3
  4. Features (college stats only — same as Tier 1):
     - Age, conf_strength
     - wOBA_adj, OPS_adj, BB_pct_adj, K_pct_adj (hitters)
     - ERA_adj, FIP_adj, K_per_nine_adj, BB_per_nine_adj (pitchers)
     - height_inches, bmi
  5. Additional feature: draft_round (as a control — "given where they were picked")
  6. Model: Elastic Net regression (NOT logistic — continuous target)
     - alpha=1.0, l1_ratio=0.3, max_iter=2000
  7. Cross-validation: 5-fold on 2021-2022
  8. Metrics: R², RMSE, MAE
  9. Baseline: predict at training set mean
  10. Save model artifacts

Validation checks:
  [V1] R² > 0.05 (modest but real signal)
  [V2] Model MAE < baseline MAE
  [V3] Feature coefficients have expected signs
  [V4] No extreme predictions (predicted wOBA between .200 and .450)

Output: models/artifacts_full/milb_outcome_hitter.pkl, milb_outcome_pitcher.pkl, milb_outcome_features_*.json
```

### Step 6 — Inference on 2026 prospects
**Script:** Integrate into `scripts/infer_2026.py`

```
Changes needed:
  1. After Tier 1-3 inference, load Tier 4 and MiLB Outcome models
  2. For each 2026 prospect (10,734 players), compute:
     - Tier 4: Not applicable for pre-draft (no MiLB year-1 data yet)
       -> Store as null for now
     - MiLB Outcome: Predict peak wOBA (hitters) / FIP (pitchers)
       -> Add fields: projected_milb_woba, projected_milb_fip, projected_milb_peak_level
  3. Save updated enriched projections

Note: Tier 4 will be populated after the 2027 draft when 2026 draftees have MiLB year-1 data.
The MiLB Outcome model works pre-draft since it uses college stats only.
```

### Step 7 — MCP Export update
**Script:** `scripts/export_frontend_data.py`

```
Changes:
  1. Read new fields from enriched projections: projected_milb_woba, projected_milb_fip
  2. Add to player detail records
  3. Add to model manifest (new model cards for Tier 4 and MiLB Outcome)
  4. Re-export to web/public/data/
```

---

## Phase 2: MiLB Outcome Model Standalone (scoping)

See `analysis/milb_integration_scope.md` for the full scope document.

The key addition beyond Phase 1:
- Frontend integration: show projected MiLB wOBA in player dossier
- Comparison view: "college stats → expected MiLB performance" next to "college stats → draft slot"
- Returning player application: filter the 10,734 scored 2026 prospects to undrafted players with high projected MiLB wOBA → these are the breakout candidates

---

## Summary of validation checks

| Step | Check | What to do if it fails |
|------|-------|-----------------------|
| Step 1 V1 | Join rate < 90% | Debug person_id mismatch between milb data and draft data |
| Step 1 V2 | Year dist skewed | Check milb file completeness by year |
| Step 1 V4 | Null targets | Check if 2023 draftees have years 2-3 data (they shouldn't — expected) |
| Step 2 V1 | wOBA correlation < 0.15 | System issue: college stats don't predict MiLB performance. Proceed but document. |
| Step 3 V1 | Tier 4 AUC ≤ Tier 3 AUC | MiLB year-1 data doesn't add signal. Document finding. |
| Step 5 V1 | R² < 0.05 | Model can't predict MiLB outcomes from college stats. Acceptable finding. |
| Step 5 V3 | Wrong coefficient signs | Check for data leakage or feature construction errors |
| Step 5 V4 | Predictions out of range | Clamp predictions to realistic bounds |

---

## Dispatch order

Each script in this plan is designed to be dispatched as a subagent with:
1. The full script specification above
2. The prior step's output file path
3. Its own validation checks
4. Clear success/failure criteria

Step 1 (build training set) must finish before Steps 3-5. Steps 3, 4, 5 can run in parallel after Step 1.
