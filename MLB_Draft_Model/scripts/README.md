# MLB Draft Model Pipeline Scripts

## Pipeline Order

### 1. Scrape Rosters
```bash
Rscript MLB_Draft_Model/scripts/scrape_d1_rosters.R [year]
```
Scrapes NCAA D1 rosters via `collegebaseball::ncaa_rosters()`. Gets height, position, class, bats/throws, conference for all D1 players.

### 2. Build Team Crosswalk
```bash
python3 MLB_Draft_Model/scripts/build_team_crosswalk.py
```
Matches FanGraphs team abbreviations to NCAA roster team names using manual overrides + fuzzy matching. Needed because FG and NCAA use different team naming conventions.

### 3. Enrich Projections
```bash
python3 MLB_Draft_Model/scripts/enrich_projections.py
```
Joins roster bio data (height, position, class, conference) into FG projections by matching on team name + player name. Computes BMI and conference tier.

### 4. Train Tier 1 Model
```bash
python3 MLB_Draft_Model/scripts/train_fg_model.py [--output-dir models/artifacts]
```
Trains XGBoost regressor predicting draft pick number from FG college stats + height + BMI + conference_tier. Separate models for hitters and pitchers.

### 5. Train Tier 2 Model
```bash
python3 MLB_Draft_Model/scripts/train_tier2_model.py [--output-dir models/artifacts]
```
Trains XGBoost classifier predicting MLB success probability from FG college stats + height + BMI + conference_tier. Includes Platt and isotonic calibration.

## Data Sources
- All data files live in `MLB_Draft_Model/data/`
- FG college stats: `data/fangraphs/`
- Draft data: `data/draft/`
- Rosters: `data/rosters/`
- Training sets: `data/training/`

## Feature Summary (July 2026)
| Feature | Tier 1 Hitter | Tier 1 Pitcher | Tier 2 Hitter | Tier 2 Pitcher |
|---------|:---:|:---:|:---:|:---:|
| `conference_tier` | #7 | #12 | #36 | #9 |
| `height_inches` | #22 | #20 | #33 | #7 |
| `bmi` | #34 | #25 | #22 | #22 |
