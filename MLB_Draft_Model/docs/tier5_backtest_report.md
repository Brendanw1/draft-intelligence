# Tier 5 Backtest Report (matured classes, out of fold)

- **Cohort**: draft classes [2021] (all 5 post-draft seasons complete through 2026)
- **Model**: P(reach MLB) x E[WAR yrs 1-5 | reached], player-grouped 5-fold OOF, prior and NN rate rebuilt per fold
- **Kill criteria**: R² ≥ 0.0 and RMSE below the median baseline
- **Status**: FAIL (Tier 5 stays off the site until PASS)

## Hitter
- n: 158
- n_reached: 34
- n_war_gt_2: 9
- r2: 0.0274
- r2_median_baseline: -0.0587
- rmse: 1.0228
- rmse_median_baseline: 1.0671
- rmse_mean_baseline: 1.0371
- spearman: 0.1006
- reach_auc: 0.6518
- reach_brier: 0.1697
- reach_brier_base_rate: 0.1689
- value_r2_given_reach: -0.0199
- war_gt_2_auc: 0.7532

## Pitcher
- n: 218
- n_reached: 68
- n_war_gt_2: 9
- r2: -0.0323
- r2_median_baseline: -0.0320
- rmse: 1.2251
- rmse_median_baseline: 1.2249
- rmse_mean_baseline: 1.2057
- spearman: 0.0058
- reach_auc: 0.6840
- reach_brier: 0.2041
- reach_brier_base_rate: 0.2146
- value_r2_given_reach: -0.0898
- war_gt_2_auc: 0.5986

## Warnings

- pitcher: R² = -0.0323 < 0.0 (kill criterion failed)
- pitcher: RMSE 1.225 not below median baseline 1.225
