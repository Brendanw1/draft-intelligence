# Draft Model Methodology

## Overview

Two-tier XGBoost draft projection system for college baseball players, trained on 6,274 player-seasons (2,600 unique players) from FanGraphs college data (2021-2026) linked to MLB draft outcomes.

## Tier 1: Draft Position Model

**Task**: Predict MLB draft pick number (regression, 1-616) from college performance stats + physical profile + context.

**Algorithm**: XGBoost Regressor  
- 500 trees, max depth 6, learning rate 0.05  
- Column/row subsampling for regularization  
- Early stopping on validation

**Features**: 37 (hitters) / 36 (pitchers)  
- College offensive/pitching stats from FG (AVG, OBP, SLG, wOBA, K%, BB%, etc.)  
- **Height** — inches (from MLB combine measurements, 100% coverage)  
- **BMI** — body mass index (computed from height/weight, 100% coverage)  
- **Conference tier** — ordinal 1-4 based on conference strength  
- **Age** — age during season (strongest single predictor)

**Validation**:  
- **Player-grouped 5-fold CV**: Players are split into folds, not records. Prevents the model from seeing a player's 2023 stats when predicting their 2024 stats. This is the most rigorous evaluation method.  
- **Year-out holdout**: Train on 2021-2024, test on 2025-2026 seasons (unseen competitive years).

**Results**:
| Group | Records | Players | CV R² | Year-out R² |
|-------|--------:|--------:|:-----:|:-----------:|
| Hitters | 2,711 | 1,189 | 0.177 | 0.307 |
| Pitchers | 3,563 | 1,608 | 0.138 | 0.240 |

## Tier 2: MLB Probability Model

**Task**: Predict probability of being drafted in top 10 rounds (binary classification).

**Algorithm**: XGBoost Classifier + Platt (sigmoid) / Isotonic calibration  
- 500 trees, max depth 4  
- Scale pos weight for class imbalance  
- Two calibration methods for well-calibrated probabilities

**Results**:
| Group | Records | Players | CV AUC |
|-------|--------:|--------:|:-----:|
| Hitters | 2,711 | 1,189 | 0.661 |
| Pitchers | 3,563 | 1,608 | 0.634 |

## New Features (Added July 2026)

### Conference Tier
**What**: Ordinal ranking of college baseball conferences by historical draft strength.  
- Tier 1: SEC, ACC, Big 12 (highest draft rate per player)  
- Tier 2: Big Ten, Pac-12, Big East, WCC, Sun Belt, American, MWC  
- Tier 3: CUSA, MAC, SoCon, MVC, A-10, Ivy, CAA, ASUN, Southland, Big South, Big West, Horizon, WAC  
- Tier 4: Patriot, America East, NEC, MAAC, Summit, SWAC, OVC, MEAC, Big Sky  

**Why**: Conference strength directly impacts draft position — a .900 OPS in the SEC is worth more than a 1.000 OPS in the Patriot League. SEC/ACC/Big 12 players benefit from higher quality of competition, more pro scouts attending games, and stronger development infrastructure.

**Impact**: Ranks #2 for hitters and #3 for pitchers in Tier 1.

### Height 
**What**: Player height in inches, from MLB combine measurements.

**Why**: Height is a well-documented predictor of pitcher success (longer lever = more velocity projection) and correlates with power for hitters. MLB front offices consistently draft taller pitchers earlier, controlling for performance.

**Impact**: Ranks #11 for pitchers in Tier 1, #7 for pitchers in Tier 2.

### BMI
**What**: Weight(kg) / height(m)² — a measure of body composition.

**Why**: Replaces raw weight as a feature. BMI captures build more accurately than weight alone (a 6'7" 230 lb pitcher has a different body type than a 5'11" 230 lb catcher). Both too-low and too-high BMI correlate with injury risk.

**Impact**: Ranks #7 for hitters in Tier 1 and Tier 2.

### Position
**What**: Defensive position from NCAA roster data, available for 89% of players via roster enrichment.

**Why**: Positional scarcity is a well-known draft factor — up-the-middle defenders (C, SS, CF) are drafted earlier than corner players with equivalent bats. The draft model incorporates this naturally through the statistical relationship between position and draft pick.

## Training Data Expansion

**Before**: 2,366 records (1 season per drafted player, only their draft year)  
**After**: 6,274 records (all available college seasons per drafted player)

**Methodology**: For each drafted player with FG college data, we extract ALL seasons (not just their draft year). A player drafted as a junior contributes their freshman, sophomore, and junior seasons as separate training examples with the same draft outcome.

**Why this is valid**:
1. **More data = more stable models**: Multi-season records help the model learn which stats are stable signals vs. noisy one-year flukes  
2. **Captures development**: A player who hit .300 as a freshman and improved to .350 as a junior tells a different story than a one-year wonder  
3. **Industry standard**: MLB front office models consistently use multi-year college data (per public reporting from Fangraphs, The Athletic)  
4. **Player-grouped CV prevents leakage**: The model never sees any season of a player when predicting that player's other seasons

## Statistical Rigour

### Cross-Validation Strategy
- **GroupKFold by person_id**: Critical — standard random CV would leak information because the same player appears multiple times. Grouped CV ensures each player's seasons are entirely in one fold.
- **5 folds** with stratification where possible
- Reported metric is mean ± std across folds

### Year-Out Testing
- Models trained on 2021-2024 seasons, tested on 2025-2026
- Tests generalization to unseen competitive years (rule changes, talent pool shifts)
- More rigorous than random splits because it tests temporal generalization

### Feature Stability
- Feature importance is checked across CV folds
- New features (conference_tier, height, bmi) show consistent rankings across folds
- No feature has zero importance across all folds (no dead features)

## Data Sources

| Source | Content | Years | Records |
|--------|---------|:-----:|:-------:|
| FanGraphs D1 College (drafted) | Season stats for drafted players | 2021-2026 | 6,283 |
| MLB Stats API | Draft outcomes (pick, round, bonus) | 2015-2026 | 9,921 |
| NCAA via collegebaseball | Rosters (height, position, conference) | 2026 | 11,784 |
| FanGraphs D1 (all) | All D1 player stats | 2021-2026 | 63,524 |

## Known Limitations

1. **Data range**: FG college data starts at 2021. Pre-2021 draft picks (2015-2020) can't be included unless we find an alternative source for college stats.
2. **Conference assignment**: Based on 2026 rosters. Some schools changed conferences — a 2021 player might be assigned the wrong conference tier.
3. **No high school data**: The model covers only college players (D1 transfers, JUCO). High school draftees are excluded.
4. **Draft target is a proxy**: Draft pick number is itself a prediction of future value, not actual MLB production. A player drafted at 1.1 (Paul Skenes) is more likely to succeed than one drafted at pick 300, but the correlation isn't perfect.
5. **No TrackMan data integration**: The current model uses only FG college stats. TrackMan metrics (EV, velo, spin, etc.) could add signal but aren't available for historical players.

## Future Work

1. **Historical data expansion**: Scrape NCAA stats for 2013-2020 via `collegebaseball::ncaa_stats()` once API access is restored  
2. **TrackMan integration**: Build a joint model using FG stats + TrackMan metrics for players who have both  
3. **MiLB outcome validation**: Replace "draft pick" target with actual MLB production (WAR, games played) for drafted players who have reached the majors  
4. **Position-specific modelling**: Separate models for each position group (C, INF, OF, RHP, LHP)  
5. **Updated conference tiers**: Conference strength changes over time — tiers should be updated annually
