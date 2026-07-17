# MiLB Integration Scope — Tier 3 Enhancement

**Status:** Scoping document  
**Data ready:** Yes — `data/milb/milb_{2021..2025}.json` already contains 37 batting + 63 pitching columns per player per year, joined by person_id

---

## Current Tier 3 Model

```
Outcome:    P(MLB debut | drafted) — binary (yes/no)
Features:   Age, conf_strength, wOBA_adj, ERA_adj, height_inches, bmi,
            round_logit_prior, nn_mlb_rate
Algorithm:  Elastic Net logistic regression
n_train:    ~1,262 (549 H + 713 P, 2021-2023 draftees)
AUC:        ~0.785
```

The model works but is capped by the binary target. Most players who get drafted never debut (~85% don't). A continuous target would give the model more signal to learn from.

---

## Proposed Multi-Target Tier 3

### Target 1: Peak MiLB wOBA (hitters) / FIP (pitchers) within first 3 pro years

The most useful upgrade. Instead of "did they debut?" (rare event), predict a continuous performance metric that every drafted player has:

- **Hitters:** Peak MiLB wOBA in years 2–3 post-draft
- **Pitchers:** Best MiLB FIP (or lowest WHIP) in years 2–3 post-draft
- **Year 1 excluded** — most players spend Year 1 in rookie/A-ball adjusting to pro ball
- **Why 3 years:** ~60% of signed draftees have 3+ years of MiLB data available

### Target 2: Highest level reached by year 3 post-draft (ordinal)

Ordinal classification (A=1, A+=2, AA=3, AAA=4, MLB=5). This is what scout grades track — "will this player advance?" It correlates strongly with eventual MLB debut but is a richer signal:

- A player who reaches AAA by year 3 is on track for an MLB shot
- A player still in A-ball by year 3 is org filler
- The model can learn which college stat profiles predict upward mobility

### Target 3 (stretch): MiLB K-BB% (pitchers) / wRC+ (hitters) peak

Advanced MiLB metrics that carry better signal than raw AVG/ERA:

- **Hitter:** Peak MiLB wRC+ (controls for park/league)
- **Pitcher:** Peak MiLB K-BB% (most stable MiLB pitching metric)

---

## Data Sources & Join

### Existing data (ready to use)

```
data/milb/milb_{year}.json  ──person_id──▶  data/training/expanded_training_set.json
Level: A/A+/AA/AAA
Batting: 37 cols (avg, obp, slg, ops, babip, hr, so, bb, sb, ...)
Pitching: 63 cols (era, whip, k/9, bb/9, hr/9, k-bb%, ...)
```

Total unique signed draftees with MiLB data: **2,164** across 2021-2025

### Coverage by cohort

| Draft Year | Signed | With MiLB data | 3+ years of data |
|------------|--------|---------------|------------------|
| 2021 | 566 | 545 (96%) | 466 (82%) |
| 2022 | 563 | 522 (93%) | 338 (60%) |
| 2023 | 564 | 478 (85%) | 87 (15%) |
| 2024 | 570 | 451 (79%) | 0 |
| 2025 | 576 | 228 (40%) | 0 |

**Implication for training:** Train on 2021-2022 draftees (have 3+ years of MiLB history), validate on 2023, hold out 2024-2025 as right-censored.

---

## Feature Engineering Plan

### New input features (for the Elastic Net model)

| Feature | Source | Rationale |
|---------|--------|-----------|
| `milb_season_count` | milb JSON count | How many MiLB seasons does this player have? Proxy for team investment |
| `milb_year1_wOBA` or `milb_year1_FIP` | First-year MiLB stats | Year 1 performance predicts year 3 outcomes |
| `milb_year1_level` | A/A+/AA/AAA as ordinal | Which level did they start at? |
| `milb_year1_to_year2_delta_wOBA` | Year-over-year change | Improvement rate in pro ball |

### New outcome targets

| Target | Type | Range | Available for |
|--------|------|-------|---------------|
| `milb_peak_wOBA` | Regression | ~.200 - .450 | All drafted hitters with MiLB data |
| `milb_peak_FIP` | Regression | ~2.0 - 8.0 | All drafted pitchers with MiLB data |
| `milb_highest_level` | Ordinal | 1-5 | All drafted players |
| `milb_vs_draft_value` | Regression | wOBA above/below expectation for their draft slot | All |

### Right-censoring rules

- **2021 draftees:** Full 5-year MiLB history available → use as ground truth
- **2022 draftees:** 4 years available → reliable
- **2023 draftees:** 2-3 years available → useful but right-censored
- **2024-2025 draftees:** Too recent → exclude from training, use only as inference validation
- **Training set:** 2021-2022 (1,088 players) — sufficient for an Elastic Net with ~10 features

---

## Pipeline Changes

### New script: `scripts/build_milb_training.py`

```
1. Load milb_{2021..2025}.json → index by person_id
2. For each player, compute:
   a. Peak wOBA/FIP in years 2-3 post-draft
   b. Highest level reached by year 3
   c. Year 1 MiLB stats (for input features)
   d. Year-over-year deltas
3. Join with expanded_training_set.json on person_id
4. Apply minimum thresholds (≥50 PA for hitters, ≥20 IP for pitchers)
5. Apply year filters (train on 2021-2022 only)
6. Write to data/training/milb_extended_training.json
```

### Modified script: `scripts/train_tier3_mlb_arrival.py`

Changes needed:
1. Add `milb_year1_wOBA`, `milb_year1_FIP`, `milb_year1_level` to T3_FEATURES
2. Add new target option: `--target peak_woba` or `--target highest_level`
3. Add new feature: `milb_progression_speed` (if multi-year data available)
4. Retain backward compatibility with current binary debut target

### New script: `scripts/train_milb_outcome_model.py`

Standalone model that predicts MiLB performance instead of MLB debut:
- Same architecture (Elastic Net + round prior)
- Continuous target (wOBA or FIP)
- Different feature set (no round_logit_prior, more MiLB-specific features)
- Useful as a standalone tool: "what MiLB performance should we expect from this college profile?"

---

## Effort Estimate

| Task | Hours | Deliverable |
|------|-------|-------------|
| Read & understand existing milb data structure | 0.5 | Familiarity |
| Write `build_milb_training.py` — join, filter, compute targets | 1.5 | `data/training/milb_extended_training.json` |
| Modify `train_tier3_mlb_arrival.py` — add features + targets | 1.5 | Updated Tier 3 model |
| Write `train_milb_outcome_model.py` — standalone MiLB pred model | 2.0 | New model + metadata |
| Run forward pass — predict MiLB outcomes for 2026 prospects | 1.0 | `projections_2026_enriched.json` updated |
| Update frontend / export to expose MiLB projections | 1.0 | Web data export |
| Write validation report (cross-validate MiLB predictions) | 1.0 | `analysis/milb_outcome_validation.md` |
| **Total** | **8.5** | |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| One-level-per-year is too coarse (promoted players get single level) | High | Medium | Accept limitation — peak level is still useful. Note it in docs. |
| 2021-2022 sample (1,088) too small for XGBoost | Medium | Medium | Use simpler model (Elastic Net keeps working). Don't need XGBoost for this. |
| MiLB stats quality varies by org (some orgs promote aggressively) | Low | Low | Random noise across 30 teams. Not systematic. |
| Data pipeline fails midway (wrong column name, missing data) | Medium | Low | `build_milb_training.py` can validate coverage before proceeding. |

---

## Decision Points

1. **Which target?** Peak wOBA/FIP is the best single target. Highest-level is a good secondary. Recommend both in a multi-output model.
2. **Which players to train on?** 2021-2022 only (sufficient MiLB history). 2023 for validation only. 2024-2025 excluded.
3. **Feature set?** Add year-1 MiLB performance + draft round. Don't over-engineer.
4. **Model class?** Elastic Net (current architecture) — works well with small n, interpretable coefficients, handles correlated features.

---

## Recommendation

**Yes, this is worth doing.** The existing MiLB data is underutilized — it's already there but not wired into any model. The effort (8.5 hours total, ~2.5 to MVP) is low relative to the signal gain. The continuous targets (peak wOBA/FIP) would give Tier 3 twice the predictive surface of the current binary debut target, and the model would answer a fundamentally different question: "what MiLB performance should we expect?" instead of just "will they debut or not?"

The returning-college-player application also works directly: a 2026 returning player with high projected MiLB wOBA (even if undrafted) is worth tracking as a transfer target or breakout candidate for 2027.
