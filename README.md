# VT Draft Intelligence

Draft projection model + dashboard for Hokies Baseball Analytics.

## Model Pipeline
- **Tier 1**: XGBoost regressor predicting draft pick number from college stats + height/BMI/conference
- **Tier 2**: XGBoost classifier predicting draft probability (trained on full D1 population, 63K player-seasons)
- Train/test split: player-grouped cross-validation + year-out validation

## Dashboard
Next.js dashboard visualizing draft projections, player comparisons, and model diagnostics.

## Quick Start
```bash
# Model pipeline
cd MLB_Draft_Model && python3 scripts/train_expanded_models.py

# Dashboard
npm run dev
```
