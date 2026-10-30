# Tier 5 WAR Value — Honesty Report

**Generated:** 2026-10-30 04:06 UTC

---

## Cohort Description

| Metric | Value |
|--------|-------|
| Total records | 1124 |
| Train (draft_year=2021) | 376 |
| Heldout (draft_year=2022-2023) | 748 |
| Positive (WAR > 2.0) | 33 (2.94%) |

**Right-censoring note:** The heldout set (2022-2023 draftees) has only 3-4 years of MLB data. Players may accumulate more WAR in future seasons. This biases the model toward underestimating true 5-year WAR.

---

## Hurdle Classifier Performance

Predicts P(war_years_1_through_5 > 2.0) = 'meaningfully productive'

### Hitters

| Metric | Value |
|--------|-------|
| Heldout AUC | 0.9324 |
| Heldout Brier | 0.3848 |
| Bootstrap 95% CI | [0.883, 0.971] |
| Gate (AUC ≥ 0.6) | PASS |

### Pitchers

| Metric | Value |
|--------|-------|
| Heldout AUC | 0.6915 |
| Heldout Brier | 0.1714 |
| Bootstrap 95% CI | [0.514, 0.860] |
| Gate (AUC ≥ 0.6) | PASS |

---

## Value Regression Performance

Predicts signed_log_war (sign(war) * log1p(|war|))

### Hitters

| Metric | Value |
|--------|-------|
| Heldout RMSE | 0.8136 |
| Heldout MAE | 0.6894 |
| Heldout R² | 0.0774 |
| Bootstrap 95% CI | [0.716, 0.917] |
| Baseline (median) RMSE | 0.8715 |
| Gate (R² ≥ 0.0) | PASS |

### Pitchers

| Metric | Value |
|--------|-------|
| Heldout RMSE | 0.7507 |
| Heldout MAE | 0.5643 |
| Heldout R² | -0.1142 |
| Bootstrap 95% CI | [0.619, 0.873] |
| Baseline (median) RMSE | 0.7567 |
| Gate (R² ≥ 0.0) | WARN |

---

## Kill Criteria

**Thresholds:** AUC < 0.6 OR R² < 0.0

**STATUS: NOT READY FOR PRODUCTION**

Kill criteria triggered:

  * pitcher value R² -0.1142 < 0.0

The model performs worse than a naive baseline (predicting the median WAR for all players). Do not deploy until these issues are resolved.

---

## Honesty Notes

1. **Small sample size:** Only ~300 training records (draft year 2021). This limits the model's ability to generalize.

2. **Right-censoring:** The heldout set (2022-2023) has incomplete WAR data. Players may accumulate more WAR in future seasons.

3. **Zero-inflated target:** Most drafted players produce 0 WAR. The signed_log_war transform handles this, but the model may still struggle with the extreme imbalance.

4. **Feature limitations:** Only 10 features per role (Tier 3 features). TrackMan data was not available for most players in the training set.

5. **Interpretation:** A 'meaningfully productive' prediction (hurdle prob > 0.5) means the player is likely to accumulate >2.0 WAR over 5 years. This is roughly equivalent to a league-average player.

---

*This report is automatically generated. Do not edit manually.*