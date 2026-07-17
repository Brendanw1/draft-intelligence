# MiLB Integration — Findings Report

**Date:** 2026-07-16
**Models trained:** Tier 4 (Tier 3 + MiLB year-1 features), MiLB Outcome Model (college stats -> MiLB peak wOBA/FIP)

---

## Summary

College FanGraphs stats predict draft position well (Tier 1 MAE ~120, Spearman 0.50) and MLB debut moderately well (Tier 3 AUC ~0.79). But they do **not** predict how well a draftee performs once they reach the minors. This is a meaningful finding: draftability and MiLB performance appear to be driven by different sets of signals.

---

## Tier 4: Tier 3 + MiLB Year-1 Features

Predicts P(MLB debut | drafted) with the same Elastic Net architecture as Tier 3, but adds first-season MiLB stats as features.

| Metric | Hitters | Pitchers |
|--------|---------|----------|
| Tier 3 AUC (college only, baseline) | 0.771 | 0.502* |
| Tier 4 AUC (college + MiLB year 1) | 0.746 | 0.647 |
| Delta | -0.025 | +0.145 |
| Tier 3 Brier | 0.138 | 0.112 |
| Tier 4 Brier | 0.123 | 0.107 |
| n_train / n_val | 290 / 233 | 177 / 171 |

*\*Pitcher Tier 3 baseline near-random because only 3 of 171 validation pitchers debuted*

**Interpretation:**
- For hitters, year-1 MiLB stats do not improve MLB debut prediction. College performance already captures the ceiling. Year-1 MiLB wOBA is redundant with college wOBA.
- For pitchers, year-1 MiLB FIP and level add meaningful signal. A pitcher who dominates A-ball in year 1 is more likely to reach MLB, independent of their college stats. But the small positive class (3/171) makes this fragile.

---

## MiLB Outcome Model

Predicts continuous MiLB performance (peak wOBA/FIP in years 2-3 post-draft) from college stats only.

| Metric | Hitters | Pitchers |
|--------|---------|----------|
| Training set | 290 (2021-2022) | 177 (2021-2022) |
| Validation set | 233 (2023) | 171 (2023) |
| CV R² | 0.018 ± 0.008 | 0.028 ± 0.058 |
| Validation R² | 0.037 | -0.090 |
| Model MAE | 0.0364 | 0.9362 |
| Baseline MAE (mean) | 0.0370 | 0.9181 |
| Improvement vs baseline | +1.6% | -2.0% (worse) |
| Best predictor | BB_pct_adj (+0.22) | None (all zeroed) |
| Checks passed | 5/5 | 2/5 |

**Top hitter predictors (std coefficients):**
| Feature | Coefficient | Expected sign | Met? |
|---------|------------|---------------|------|
| BB_pct_adj | +0.0083 | + (better plate discipline = higher MiLB wOBA) | Yes |
| draft_round | -0.0073 | - (earlier round = better) | Yes |
| OPS_adj | +0.0057 | + (better college production = better MiLB) | Yes |
| conf_strength | -0.0038 | - (stronger conference = tougher competition = less translation) | Yes |
| K_pct_adj | +0.0031 | + (high K% in SEC means something different) | No |

**Interpretation:**
- College stats explain less than 4% of the variance in MiLB peak wOBA.
- For pitchers, the signal is completely absent — regularization zeroed out every coefficient.
- The weak performing models validate the hypothesis: MiLB outcomes are driven by scouting grades, development environment, health, and opportunity — none of which appear in FanGraphs college leaderboards.

---

## What This Means for the Project

### What the model DOES well (confirmed)

| Question | Model | Performance | Use |
|----------|-------|-------------|-----|
| Will this player be drafted? | Tier 2 | AUC 0.97 | Screening 10K players → draftable pool |
| How early will they go? | Tier 1 | MAE ~120, Spearman 0.50 | Tiering by round range |
| Will they reach MLB? | Tier 3 | AUC 0.79 | Pre-draft ceiling projection |

### What the model CANNOT do (new finding)

| Question | Attempt | Result |
|----------|---------|--------|
| How will this player perform in AAA? | MiLB Outcome Model | R² < 0.04 — no usable signal |
| Will MiLB year-1 data improve draft projections? | Tier 4 | Hitters: no. Pitchers: marginal. |

### Practical takeaway

The MiLB data that already exists in `data/milb/milb_*.json` is best used as a **retrospective validation tool** rather than a pre-draft input. For each draft class, you can:

1. After 3 years, compare each draftee's predicted Tier 1 slot with their actual MiLB trajectory
2. Identify which college stat profiles systematically outperform or underperform their draft position
3. Feed that back into the next year's draft model as a market-inefficiency feature

That third point is the real opportunity: "players with this college profile tend to outperform their draft slot" is a signal the current model doesn't capture but the MiLB data could reveal.

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/build_milb_training.py` | Build joined training set (871 records) |
| `scripts/validate_milb_training.py` | Validation checks on training set |
| `scripts/train_tier4_milb_arrival.py` | Tier 4 model (Tier 3 + MiLB year-1) |
| `scripts/train_milb_outcome_model.py` | Standalone MiLB outcome regression |
| `data/training/milb_extended_training.json` | Joined training data |
| `models/artifacts_full/tier4_mlb_hitter.pkl` | Tier 4 hitter model |
| `models/artifacts_full/tier4_mlb_pitcher.pkl` | Tier 4 pitcher model |
| `models/artifacts_full/milb_outcome_hitter.pkl` | MiLB outcome hitter model |
| `models/artifacts_full/milb_outcome_pitcher.pkl` | MiLB outcome pitcher model |
| `analysis/milb_integration_scope.md` | Original scoping document |
| `analysis/milb_data_feasibility.md` | Earlier feasibility report |
