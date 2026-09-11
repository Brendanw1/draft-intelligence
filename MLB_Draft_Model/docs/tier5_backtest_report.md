# Tier 5 Backtest Report (2023 Cohort)

## Overview
- **Cohort**: 2023 draft class (heldout from training)
- **Models**: Tier 5 Phase A (LightGBM)
- **Kill criteria**: R² ≥ 0.0
- **Status**: FAIL

## Performance Metrics

### Hitter
- **N**: 160
- **RMSE**: 2.8577 (baseline: 2.9739)
- **MAE**: 1.5998
- **R²**: -0.1132 (baseline: -0.2056)
- **Correlation**: 0.1889
- **Hurdle accuracy**: 72.73%

### Pitcher
- **N**: 209
- **RMSE**: 0.9761 (baseline: 0.8718)
- **MAE**: 0.6655
- **R²**: -0.3616 (baseline: -0.0862)
- **Correlation**: 0.0524
- **Hurdle accuracy**: 61.11%

## Warnings

- hitter: R² = -0.1132 < 0.0 (kill criterion failed)
- pitcher: R² = -0.3616 < 0.0 (kill criterion failed)

## Interpretation

This backtest uses fully observed WAR outcomes (no right-censoring),
providing the most honest evaluation of Tier 5 predictive power.

If R² < 0.0, the model performs worse than predicting the median WAR.
This is expected with small training sets (~300 records) and high WAR variance.

The hurdle model (positive WAR probability) is typically more reliable
than the value regression (expected WAR magnitude).
