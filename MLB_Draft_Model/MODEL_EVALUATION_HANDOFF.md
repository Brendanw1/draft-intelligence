# Draft Intelligence — Model Evaluation Handoff

**Project:** MLB Draft Intelligence (`vt-draft-intelligence.vercel.app`)
**Repo:** `Brendanw1/vt-draft-intelligence` (monorepo root is `vt_baseball/`)
**App root:** `MLB_Draft_Model/web/`
**Data pipeline:** `MLB_Draft_Model/scripts/`
**Contact:** Brendan

---

## Goals for This Engagement

1. **Critically evaluate each tier's performance** — assess whether the current model choices are sound given the training data constraints (small n, sparse features).
2. **Try alternatives** — systematically test different model families, feature sets, and validation strategies. The current implementation was built iteratively and has no baseline comparisons.
3. **Produce a clear verdict** — for each tier, recommend: keep, retrain with changes, or replace entirely. If replace, show the alternative's performance numbers.
4. **Document methodology** — what was tried, what worked, what didn't, and why. The output should be actionable for a frontend developer: "display this number, not that one."

---

## Architecture Overview

Three-tier structure applied per player type (hitter/pitcher), trained independently:

```
           ┌──────────────────────────────────────────────────┐
           │           2026 Prospect Pool (10,734)             │
           │  (FanGraphs D1 stats + conference + physical)     │
           └─────────────────────┬────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │   Tier 1       │ │   Tier 2       │ │   Tier 3       │
     │  Projected     │ │  P(drafted     │ │  P(MLB |       │
     │  Pick Number   │ │  top 10 rds)   │ │  drafted)      │
     │                │ │                │ │                │
     │ XGBoost Reg    │ │ XGBoost Class  │ │ Elastic Net    │
     │ n=608          │ │ n=608          │ │ (prior-offset) │
     │ 37 FG features │ │ 37+ FG feats   │ │ n=1,524        │
     │ CV MAE ~108    │ │ CV AUC ~0.88   │ │ CV AUC ~0.79   │
     └────────────────┘ └────────────────┘ └────────────────┘
```

**Key constraint:** All tiers share the same core training data — ~608 players with both TrackMan college metrics AND draft outcomes (2015–2025 drafts). This is the "bottleneck" dataset. An expanded 2,366-player set (FanGraphs stats only, no TrackMan) is used for Tier 3 and comps.

---

## Data Sources

| File | Path | Rows | Used By |
|------|------|------|---------|
| Training set | `data/training/training_set.json` | 608 | Tier 1, Tier 2 |
| Historical FG | `data/training/fg_training_set.json` | 2,366 | Tier 3, comps |
| 2026 projections | `data/training/projections_2026_enriched.json` | 10,734 | Inference |
| Raw FG 2026 | `data/fangraphs/raw/batters_2026.json` | 5,330 | Export |
| Raw FG 2026 | `data/fangraphs/raw/pitchers_2026.json` | 5,404 | Export |
| Historical raw FG | `data/fangraphs/raw/batters_{year}.json` | 2021–2025 | Export (seasons) |
| Historical raw FG | `data/fangraphs/raw/pitchers_{year}.json` | 2021–2025 | Export (seasons) |
| Draft picks | `data/draft/draft_all_picks.json` | 9,921 | Comp enrichment |
| MiLB outcomes | `data/milb/milb_{year}.json` | 2021–2025 | Comp enrichment |
| Conference strength | `data/conference_strength.json` | — | Feature engineering |

### Training Set Details

**608 joined players** (TrackMan + draft outcomes):

- 271 hitters, 337 pitchers
- All have FG stats AND draft outcomes (round, pick, bonus, team)
- ~372 high-confidence matches (name + school match), ~236 name-only
- Draft pick range: ~1–620 (first 20 rounds)
- Years covered: 2015–2025 (but the joined set skews heavily 2024–2025 since TrackMan data is recent)

**Known gaps:**
- No MiLB outcome data joined into training set (peak level, MLB arrival)
- No player age at draft time in the 608-player set (some columns exist but are sparse)
- TrackMan metrics come from a single season (~2024–2025), not multi-year
- Conference strength is a `conf_strength` float, replacing an earlier broken categorical `conference_tier`

---

## Current Model Details

### Tier 1: Projected Pick (XGBoost Regressor)

| | Hitter | Pitcher |
|---|---|---|
| **Algorithm** | XGBoost Regressor (500 trees, max_depth=6) | XGBoost Regressor (500 trees, max_depth=6) |
| **Target** | `pick_number` (integer 1–620) | `pick_number` (integer 1–620) |
| **Features** | 37 FG stats (AVG, OBP, SLG, wOBA, BB%, K%, ISO, HR, SB, etc.) | 37 FG stats (ERA, FIP, WHIP, K%, BB%, K/9, BB/9, IP, GS, etc.) |
| **Training size** | ~271 | ~337 |
| **CV MAE** | ~108 picks (~3.5 rounds) | ~112 picks (~3.6 rounds) |
| **Holdout MAE (2024)** | ±112 picks (site disclaimer) | ±112 picks (site disclaimer) |
| **Calibration** | Isotonic regression per type | Isotonic regression per type |
| **Artifacts** | `models/artifacts/fg_draft_hitter.json` | `models/artifacts/fg_draft_pitcher.json` |

**How it works:** XGBoost is trained on the 608-player set with FG college stats as features. Predictions are isotonically calibrated against a holdout year. The point estimate is raw tree output; the band is ±CV MAE.

**Concerns to evaluate:**
- n=271 / n=337 is very small for 37 features and 500 trees. Risk of overfitting despite XGBoost regularization.
- The FG stats are from the player's draft-eligible season. But college stats across years aren't directly comparable (different competition, different park factors, different bats). No year-adjustment or opponent-adjustment is applied.
- No conference strength, age, or physical metrics in Tier 1 (these are only in Tier 2). Should they be?

### Tier 2: P(drafted top 10 rounds) — XGBoost Classifier

| | Hitter | Pitcher |
|---|---|---|
| **Algorithm** | XGBoost Classifier (scale_pos_weight≈6.8) | XGBoost Classifier |
| **Target** | Binary: drafted in top 10 rounds (1) or not (0) | Same |
| **Features** | FG stats + age + height + BMI + conference_strength | FG stats + age + height + BMI + conference_strength |
| **Training size** | ~271 | ~337 |
| **CV AUC** | ~0.88 (site claims) | ~0.88 (site claims) |
| **Artifacts** | `models/artifacts/tier2_hitter.json` | `models/artifacts/tier2_pitcher.json` |

**How it works:** Same feature set as Tier 1 plus physical/conference features. XGBoost with class-weighting to handle imbalance (most players are NOT drafted). Output is Platt-scaled to produce a well-calibrated probability.

### Tier 3: P(MLB | drafted) — Elastic Net (Prior-Offset)

| | Hitter | Pitcher |
|---|---|---|
| **Algorithm** | Elastic Net (prior-offset logistic regression) | Elastic Net (prior-offset logistic regression) |
| **Target** | Binary: reached MLB (1) or not (0) | Same |
| **Base rate prior** | Historical round-specific MLB debut rate (logit) | Same |
| **Features** | FG stats + conf_strength + `nn_mlb_rate` | FG stats + conf_strength + `nn_mlb_rate` |
| **Training size** | 1,524 (FG-only, 2021–2024) | 1,524 (FG-only, 2021–2024) |
| **CV AUC** | ~0.79 | ~0.79 |
| **Artifacts** | `models/artifacts_full/tier3_mlb_hitter.{pkl,json}` | `models/artifacts_full/tier3_mlb_pitcher.{pkl,json}` |

**How it works:** Uses the 2,366-player historical FG dataset (filtered to 1,524 by removing 2025+). The prior-offset approach: instead of directly predicting P(MLB), the model predicts the deviation from the historical round-by-round MLB debut rate. The prior is `round_logit_prior` — the empirical logit of MLB arrival for each draft round (e.g., Round 1 ≈ 61%, Round 20 ≈ 3%). The model then adjusts this baseline based on the player's stats.

**Additional Tier 3 component:** Nearest-neighbor comps. For each 2026 prospect, the 5 most stat-similar historical drafted players are found (Euclidean distance on normalized stat vectors). The comp's `nn_mlb_rate` (fraction of comps who reached MLB) is used as a feature in the Elastic Net AND shown as an independent signal in the UI.

### Conference Strength

The `conf_strength` feature replaced a broken categorical `conference_tier`. It's computed by:

1. For each D1 conference, calculate the average draft pick of its drafted players over the historical period
2. Normalize to a 0–100 scale (higher = stronger conference — players get drafted earlier)
3. Used as a continuous feature in Tier 2 and Tier 3

**Artifact:** `models/artifacts_full/conference_strength.json`

---

## Training Pipeline (Scripts)

```
train_tier1_model.py          → trains XGBoost regressor (fg_draft_hitter/pitcher)
train_fg_model.py             → same logic, alternative formulation
train_tier2_model.py          → trains XGBoost classifier (tier2_hitter/pitcher)
train_tier2_full.py           → tier2 with expanded features
train_tier3_mlb_arrival.py    → trains Elastic Net prior-offset model
compute_conference_strength.py → computes conf_strength from draft data
infer_2026.py                 → runs all tiers on 2026 prospects
calibrate_probs.py            → Platt / Isotonic calibration
build_calibration_curve.py    → calibration curve data for UI
export_frontend_data.py       → writes web/public/data/ and syncs to R2
```

**Execution order for a full refresh:**
```bash
# 1. Train each tier
python3 scripts/train_tier1_model.py
python3 scripts/train_tier2_model.py
python3 scripts/train_tier3_mlb_arrival.py

# 2. Calibrate
python3 scripts/calibrate_probs.py

# 3. Infer on 2026 class
python3 scripts/infer_2026.py

# 4. Export for frontend
python3 scripts/export_frontend_data.py

# 5. Sync to R2
rclone copy web/public/data/ r2:mlbdraftcol/data/ --progress

# 6. Git push → Vercel auto-deploy
git push
```

---

## What to Evaluate

### High Priority

1. **Tier 1 overfitting risk** — With 37 features and only 271/337 training examples, what does learning curve analysis show? Try: reduced feature set (top 10 by importance), simpler model (ridge/lasso), leave-one-year-out CV.

2. **Tier 2 vs Tier 1 as a classifier** — If we threshold Tier 1's predicted pick at round 10, does that outperform the dedicated classifier? Is Tier 2 providing independent signal or just replicating Tier 1?

3. **Tier 3 prior-offset validation** — The prior-offset approach assumes round-specific base rates are stable. Validate: does the model actually beat a constant "round X baseline" predictor? How sensitive is AUC to the prior?

4. **Feature importance stability** — Do top features change significantly across bootstrap iterations? If feature ranks are unstable, the model is fitting noise.

5. **Calibration quality** — The site uses Platt scaling (Tier 2) and Isotonic regression (Tier 1). Are these actually improving calibration over raw XGBoost outputs? Show reliability diagrams.

### Medium Priority

6. **Alternative model families** — Try: Random Forest, LightGBM, linear models with regularization. Does XGBoost actually outperform simpler alternatives on this small-n problem?

7. **Conference strength impact** — How much does `conf_strength` actually move predictions? Is it doing real work or just encoding school name? Try: ablation (remove conf_strength, compare AUC/MAE).

8. **Data leakage check** — The 608 training set includes 2025 draftees whose FG stats come from the same season as the 2026 prospects. Is there any year-over-year information leakage?

9. **Age-adjusted features** — Age is currently a raw feature. Should stats be age-adjusted (e.g., a 22-year-old putting up a .400 wOBA vs. a 19-year-old doing the same)?

### Lower Priority

10. **Nearest-neighbor comp distance metric** — Currently Euclidean distance on 10 normalized stats. Try: Mahalanobis distance (accounts for feature correlation), cosine similarity, weighted dimensions.

11. **Ensemble of similarity search + model** — Instead of using `nn_mlb_rate` as a feature, try a two-stage approach where the model is trained on residuals after comp-group adjustment.

12. **Poisson / negative binomial for pick number** — Draft picks are count-like but bounded 1–620. Would a count model be more appropriate than treating it as unbounded regression?

---

## What's Displayed on the Site

| Frontend Field | Data Source | Tier |
|---|---|---|
| Projected Pick | `key_stats` on index + shard | Tier 1 |
| Projected Round | `pick_to_round(proj_pick)` | Tier 1 |
| Pick Band (±MAE) | `pick_band` | Tier 1 |
| MLB % (calibrated) | `mlb_p` (Platt) or `mlb_p_iso` (Isotonic) | Tier 2 |
| MLB % (raw model) | `mlb_p_raw` | Tier 2 |
| Value Grade | `composite_score` → elite/high/medium/low | Tier 2 + Tier 3 composite |
| MLB Arrival % | `mlb_arrival` | Tier 3 |
| NN MLB Rate | `nn_mlb_rate` | Tier 3 (comps) |
| Similar Players | `comps[].name, .pick, .peak_level, .reached_mlb` | Comp DB |
| Percentile bars | `pctl` : `{so: 99, fip: 98, ...}` | Export-computed |
| Season stats | `seasons[].ERA, .FIP, .WHIP, .K_pct, ...` | Raw FG |
| Key stats (board) | `key_stats` from FG 2026 raw data | Raw FG |

**Not yet displayed on frontend:**
- Tier 3 model cards in `models_manifest.json`
- `mlb_arrival` and `nn_mlb_rate` in `BoardTable.tsx` and `PlayerDossier.tsx`
- Calibration curves in model lab
- Conference strength value per player

---

## Constraints

- **No API keys or secrets** — all data is from public sources (FanGraphs leaderboards, MLB Stats API, NCAA rosters)
- **No external model hosting** — models are pickled/joblib'd and run locally via `infer_2026.py`
- **Python 3.12+** environment with standard ML stack (scikit-learn, xgboost, numpy, pandas)
- **The 608-player training set is the bottleneck** — this is hard to grow without more TrackMan data or reliable name-matching
- **Frontend is static Next.js** — no server-side inference; all model outputs are pre-computed and pushed to R2

---

## Repository Layout (relevant paths)

```
MLB_Draft_Model/
├── data/
│   ├── training/           # Training sets + projections
│   ├── fangraphs/raw/      # Raw FanGraphs leaderboards
│   ├── draft/              # MLB draft picks
│   └── milb/               # MiLB outcome data
├── models/
│   ├── artifacts/          # Production models + calibrators
│   └── artifacts_full/     # Additional models (Tier 3, conference)
├── scripts/
│   ├── train_*.py          # Training scripts
│   ├── infer_2026.py       # Inference on 2026 class
│   └── export_frontend_data.py  # Data pipeline to web
├── web/
│   ├── app/                # Next.js pages
│   ├── components/         # React components
│   └── lib/                # Utility functions
└── README.md               # Project overview
```

---

## Deliverables

The ideal output is a `MODEL_EVALUATION.md` in the repo root containing:

1. **Executive summary** — one-paragraph verdict per tier (keep / retrain / replace)
2. **Methodology** — what was tested, hyperparameters, CV strategy
3. **Results tables** — MAE/AUC comparisons across model families for each tier
4. **Feature analysis** — top features, stability, ablation results
5. **Calibration assessment** — reliability diagrams as code-generated ASCII or instructions to generate
6. **Recommendations** — specific code changes to make, with expected impact
7. **Further work** — what wasn't tried that should be
