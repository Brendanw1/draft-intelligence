# Draft Intelligence — Rolling Year-Out Backtest

## Summary

This report evaluates the Tier 1 draft pick prediction model using a rolling year-out backtest methodology. For each holdout year (2022–2026), the model is trained exclusively on data from prior years (draft_year ≤ Y−1), then used to predict draft positions for the holdout year. Predictions are compared to actual draft outcomes and three naive baselines: (1) predict every draftee at the historical mean pick, (2) predict at the conference-level historical mean, and (3) predict at the round-level historical average (upper-bound reference, since it uses the actual round assignment).

## Performance by Holdout Year

| Holdout | n_train | n_test | Model MAE | ±110 Hit% | Spearman ρ | Naive MAE | Conf MAE | Round Avg MAE |
|---------|---------|--------|-----------|-----------|------------|-----------|----------|---------------|
| 2022 | 327 | 361 | 130.7 | 47% | 0.3996 | 146.5 | 149.8 | 16.1 |
| 2023 | 688 | 410 | 118.2 | 54% | 0.4856 | 140.2 | 144.2 | 14.2 |
| 2024 | 1098 | 426 | 114.8 | 55% | 0.5255 | 144.1 | 143.6 | 13.5 |
| 2025 | 1524 | 405 | 116.7 | 55% | 0.4955 | 142.0 | 140.0 | 18.1 |
| 2026 | 1929 | 437 | 118.4 | 53% | 0.526 | 146.3 | 144.9 | 15.1 |

**Key observations:**

- **Average Model MAE**: 119.8 picks across 5 holdout years
- **Average Naive Mean MAE**: 143.8 picks (model is 17% better)
- **Average Spearman ρ**: 0.486
- **Average ±110 capture rate**: 53%

## Baseline Comparison

Three naive baselines provide context for model performance:

1. **Naive Mean**: Every draftee predicted at the training set's mean draft pick.
2. **Conference Mean**: Each draftee predicted at their conference's historical mean pick.
3. **Round Average (reference)**: Each draftee predicted at their round's historical average. This uses the actual round assignment and is an upper-bound reference, not a fair comparison.

| Holdout | Model MAE | Naive Mean | Conf Mean | Round Avg | Model vs Naive Δ |
|---------|-----------|------------|-----------|-----------|------------------|
| 2022 | 130.7 | 146.5 | 149.8 | 16.1 | +15.8 |
| 2023 | 118.2 | 140.2 | 144.2 | 14.2 | +22.0 |
| 2024 | 114.8 | 144.1 | 143.6 | 13.5 | +29.3 |
| 2025 | 116.7 | 142.0 | 140.0 | 18.1 | +25.3 |
| 2026 | 118.4 | 146.3 | 144.9 | 15.1 | +27.9 |

## How the Band Performs

A key operational question: if we use this model to identify which college players will be drafted within a specific range, what capture rate do different bands achieve?

| Holdout | n | ±50 | ±75 | ±110 | ±150 | ±200 |
|---------|---|-----|-----|------|------|------|
| 2022 | 361 | 73 (20%) | 119 (33%) | 171 (47%) | 221 (61%) | 275 (76%) |
| 2023 | 410 | 98 (24%) | 159 (39%) | 222 (54%) | 276 (67%) | 335 (82%) |
| 2024 | 426 | 113 (27%) | 161 (38%) | 235 (55%) | 298 (70%) | 359 (84%) |
| 2025 | 405 | 98 (24%) | 146 (36%) | 223 (55%) | 285 (70%) | 341 (84%) |
| 2026 | 437 | 106 (24%) | 153 (35%) | 232 (53%) | 307 (70%) | 365 (84%) |

**Cumulative across all holdout years:**

| Band | Total Captured | Capture Rate |
|------|----------------|--------------|
| ±50 | 488/2039 | 24% |
| ±75 | 738/2039 | 36% |
| ±110 | 1083/2039 | 53% |
| ±150 | 1387/2039 | 68% |
| ±200 | 1675/2039 | 82% |

- **80th percentile error**: 191 picks — 80% of predictions are within ±191 picks of actual.
- **90th percentile error**: 241 picks — 90% of predictions are within ±241 picks of actual.
- **To capture 90% of draftees**, you would need a band of approximately ±241 picks.

## Learning Curve

Does model accuracy improve as the training set grows from 1 year of data (2021 → predict 2022) to 5 years (2021–2025 → predict 2026)?

| Holdout | Train Years | n_train | Model MAE | Naive MAE | Spearman ρ | ±110% |
|---------|-------------|---------|-----------|-----------|------------|-------|
| 2022 | 1 | 327 | 130.7 | 146.5 | 0.3996 | 47% |
| 2023 | 2 | 688 | 118.2 | 140.2 | 0.4856 | 54% |
| 2024 | 3 | 1098 | 114.8 | 144.1 | 0.5255 | 55% |
| 2025 | 4 | 1524 | 116.7 | 142.0 | 0.4955 | 55% |
| 2026 | 5 | 1929 | 118.4 | 146.3 | 0.526 | 53% |


**Trend**: Model MAE goes from 124.4 (first 2 years) to 117.6 (last 2 years), improving by 6.9 picks.

## Position Breakdown

| Holdout | Hitters n | Hitter MAE | Hitter ±110% | Pitchers n | Pitcher MAE | Pitcher ±110% |
|---------|-----------|------------|--------------|------------|-------------|---------------|
| 2022 | 156 | 142.5 | 43% | 205 | 121.7 | 51% |
| 2023 | 180 | 128.5 | 51% | 230 | 110.1 | 57% |
| 2024 | 178 | 118.1 | 58% | 248 | 112.4 | 53% |
| 2025 | 173 | 115.6 | 58% | 232 | 117.5 | 53% |
| 2026 | 181 | 127.9 | 49% | 256 | 111.7 | 56% |

**Hitters** average MAE: 126.5 over 868 predictions. **Pitchers** average MAE: 114.7 over 1171 predictions.
The model performs better for pitchers than hitters across the backtest period.

## 2026 Prospective Evaluation

The model trained on 2021–2025 data (n=1929) predicts the 2026 draft class with the following performance against actual picks (n=437 matched draftees):

- **MAE**: 118.4 picks
- **Spearman ρ**: 0.526
- **Within ±110**: 53%
- **Within ±75**: 35%

Compared to the naive mean baseline (146.3 MAE) and conference-mean baseline (144.9 MAE), the model provides a meaningful improvement.

- **Hitters** (n=181): MAE=127.9, ±110 capture=49%
- **Pitchers** (n=256): MAE=111.7, ±110 capture=56%

## Honest Assessment

### What the model is good for

1. **Broad draft-range identification**: The model can identify which broad tier a player will be drafted in (top 100, rounds 3–5, rounds 6–10, day 3). The ±110-band capture rate averages 53% across all years, meaning roughly half of college draftees have their actual pick within about 3.5 rounds (±110 picks) of the prediction.
2. **Conference-strength calibration**: By adjusting raw stats for conference quality and including strength × stat interactions, the model correctly discounts inflated production in weak conferences and boosts strong performance in elite conferences.
3. **Rank ordering**: The Spearman ρ of 0.486 indicates solid rank-order agreement between predicted and actual draft order.

### What the model is NOT good for

1. **Precise pick prediction**: Individual pick predictions should NOT be taken literally. The average error is 120 picks, and the 90th percentile error is ~241 picks. This model cannot tell you whether a player will go at pick 117 vs. 134.
2. **Between-round distinctions near boundaries**: Players projected near round boundaries are the hardest to pin down — a player projected at pick 115 might go at 95 or 140 depending on team need, bonus pool dynamics, and signability.
3. **Undrafted player identification**: Tier 1 is trained only on drafted players (selection bias). It cannot distinguish between a player who will go undrafted and one who will be a late-round pick. Tier 2 (draftability classifier) addresses this separately.

### Tier-based confidence

| Tier | Picks | Expected MAE | Recommendation |
|------|-------|-------------|----------------|
| 1st/2nd round | 1–75 | Low–moderate | Good for ranking, less so for exact pick |
| Rounds 3–5 | 76–175 | Moderate | Best use case — broad range identification |
| Rounds 6–10 | 176–315 | Moderate–high | Useful for identifying late-round depth |
| Day 3 | 316+ | High | Expect large variance; use for direction only |

### Data limitations

- **Uniform college stats**: The model uses FanGraphs college statistics, which are uniform across all players. It does not include scouting grades, exit velocity, or TrackMan metrics.
- **Conference coverage**: Some smaller conferences have very few drafted players in the training set, making conference-adjusted stats less reliable for those players.
- **Signability effects**: The model does not account for signability, which can cause players to fall significantly in the draft (e.g., a draft-eligible sophomore with high bonus demands).
- **Sample size**: With ~300–400 drafted players per year in the training set, the model is limited by the available data. As more years of data accumulate, accuracy should improve.

## Methodology

- **Model**: XGBoost regressor (500 trees, max_depth=6, lr=0.05, with early stopping)
- **Features**: ~55+ college stats (AVG, OPS, wOBA, K%, BB%, ISO, etc.) + height, BMI, conference strength + conference-adjusted stats + strength×stat interactions
- **Training**: Separate models for hitters and pitchers, trained on `fg_training_set.json`
- **Conference adjustment**: Stats are adjusted by subtracting the conference-season average, then multiplied by conference strength
- **Lookahead prevention**: Conference stats are filtered to only include seasons ≤ Y−1 for each holdout year
- **Naive baselines**: Mean pick, conference-mean pick, and round-average pick computed from the same training set used for model training
