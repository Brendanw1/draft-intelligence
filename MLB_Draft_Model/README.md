# MLB Draft Intelligence — Prediction & Scouting Dashboard

A **full-stack machine learning pipeline** that projects where 10,000+ NCAA Division I baseball players will be selected in the MLB Draft, estimates their probability of reaching the majors, and surfaces similar historical player comps.

Built as a static Next.js 15 application with a Python/XGBoost/Elastic Net backend. No server required — the ML pipeline generates static JSON data that the frontend consumes at build time.

---

## The Problem

College baseball has **no centralized draft board**. Scouts, front offices, and analysts each maintain their own lists. Public projections are either paywalled (FanGraphs, Baseball America) or rely on a single human opinion. There's no open, reproducible, data-driven baseline that any team or fan can inspect.

This project is that baseline.

---

## What It Does

| Page | What it answers |
|------|----------------|
| **Board** | Who matters in the 2026 class? (10,734 players, sortable/filterable by grade tier, position, conference, draft round band) |
| **Value** | Where will the market misprice talent? (projected pick vs calibrated MLB probability) |
| **Class Retrospectives** | How did past draft classes actually turn out? (2021–2026 with real outcomes) |
| **Player Dossier** | What's the full picture on this specific player? (projections, percentile bars, multi-season stats, top-5 comparable draftees, physical profile, tiered model transparency including MLB arrival outlook, model disagreement indicators) |
| **Model Lab** | Why should anyone believe these predictions? (backtest curves, reliability diagrams, feature importances, known limitations, per-model calibration audits) |
| **Methodology** | How exactly do the three tiers work, what are the known gaps, and what changed in the latest vintage? |

---

## Technical Architecture

```
                     ┌──────────────────────────────────────────────────┐
                     │           Data Pipeline (Python)                 │
                     │                                                  │
                     │  MLB Stats API  ──┐                              │
                     │  FanGraphs D1    ──┤─── Training Set (all tiers) │
                     │  NCAA Rosters   ──┘    6,274 rows                │
                     │  MiLB outcomes  ──┐    1,524 comp pool           │
                     │  Draft records   ──┘                              │
                     │                                                  │
                     │  ┌─────────────────────────────────────────────┐ │
                     │  │  Tier 1: Round Regressor (XGBoost)          │ │
                     │  │  Predicts pick number from conference-      │ │
                     │  │  adjusted stats. Features: wOBA_adj,        │ │
                     │  │  ERA_adj, conf_strength, height, age,       │ │
                     │  │  interaction features (strength_x_{stat}).  │ │
                     │  └──────────────────┬──────────────────────────┘ │
                     │                     ▼                            │
                     │  ┌─────────────────────────────────────────────┐ │
                     │  │  Tier 2: MLB Probability (XGBoost + Platt)  │ │
                     │  │  Full-population classifier trained on       │ │
                     │  │  56,910 undrafted negatives. Features:       │ │
                     │  │  Tier 1 projected pick + all conf-adjusted   │ │
                     │  │  stats. Platt-scaled for calibration.        │ │
                     │  └──────────────────┬──────────────────────────┘ │
                     │                     ▼                            │
                     │  ┌─────────────────────────────────────────────┐ │
                     │  │  Tier 3: MLB Arrival (Prior-offset EN)      │ │
                     │  │  Elastic Net predicting P(MLB debut|drafted) │ │
                     │  │  anchored by round-specific historical      │ │
                     │  │  baselines. Features: stats + NN_mlb_rate.  │ │
                     │  └──────────────────┬──────────────────────────┘ │
                     │                     ▼                            │
                     │  ┌─────────────────────────────────────────────┐ │
                     │  │  Nearest-Neighbor Comps (+ NNs MLB rate)    │ │
                     │  │  Euclidean distance across 10 normalized     │ │
                     │  │  stat dimensions → 5 most similar draftees. │ │
                     │  └─────────────────────────────────────────────┘ │
                     │                     ▼                            │
                     │       65 JSON files + 64 player shards (38 MB)   │
                     └──────────────────────┬───────────────────────────┘
                                            │
               ┌────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────┐
│         Next.js 15 Static Site (no DB)             │
│                                                    │
│   - Virtualized 10K-row table                      │
│   - Client-side sorting/filtering                  │
│   - Per-player dossier with 3-tier outlook         │
│   - Model backtest plots + calibration audits      │
│   - Static export → any CDN (Vercel, CF Pages, S3) │
│   - Dark/light theme with precision-instrument DS  │
└──────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Three-tier model**: Separates "when will you be drafted?" (Tier 1 regressor), "will you ever reach MLB?" (Tier 2 classifier), and "if drafted, what's your arrival probability?" (Tier 3 prior-offset). These are different questions requiring different architectures.
- **Conference adjustment**: Raw stats (wOBA, ERA, etc.) are multiplied by the inverse of `conf_strength` — a continuous score based on empirical draft rates per conference. SEC production (2.98× draft rate) isn't compared at face value to SWAC production (0.09×).
- **Full-population training**: Tier 2 includes 56,910 undrafted players as true negatives. Most draft models skip this — the model learns what *doesn't* get drafted, not just who does.
- **Platt calibration**: Raw XGBoost probabilities are systematically overconfident. Platt scaling maps them to well-calibrated MLB% estimates verified by reliability diagrams.
- **Prior-offset Tier 3**: Instead of predicting absolute MLB arrival probability from scratch, Tier 3 predicts *deviation* from a round-specific historical baseline. A 5th-rounder projected at 15% arrival is above their round baseline; a 1st-rounder at 15% is below theirs.
- **Static export**: Zero runtime infrastructure. No database, no API, no server. The pipeline generates JSON → Next.js builds a static site → deploy anywhere (Vercel, Cloudflare Pages, S3).
- **Sharded data**: 10,734 player records split into 64 shards + 1 index file. The board loads from the index (9 MB, fast); dossiers fetch single shards lazily.
- **Precision-instrument design system**: Signal gradient (gray=low, amber=medium, teal=high, blue=elite) encodes uncertainty visually. 4px spacing grid, 6-step type scale, tabular numbers throughout.

---

## Data Sources

All sourced from **freely public APIs and websites**:

| Source | Data | Coverage |
|--------|------|----------|
| MLB Stats API | Draft picks, rounds, bonuses, schools, physicals, MLB debut dates | 2015–2026, ~1,200 picks/year |
| FanGraphs College Leaderboards | Hitting & pitching stats (AVG, OPS, wOBA, ERA, FIP, K/BB, etc.) | 2021–2026, ~4,000+ players/year |
| NCAA D1 Rosters | Height, positions, conferences | 2026 current, 85% height coverage |
| MiLB game feeds | Peak minor-league level reached, MLB debut status | 2021–2025, 1,524 comp pool records |

No proprietary TrackMan data, no internal team data, no paywalled sources.

---

## Model Performance (held-out test set)

| Metric | Hitters | Pitchers |
|--------|---------|----------|
| Year-out AUC (Tier 2) | 0.994 | 0.989 |
|| Tier 3 AUC (arrival) | 0.79 | 0.79 |
|| Dominant feature | height_inches (0.764) | height_inches (0.616) |
|| Position/velocity | No single skill stat dominates — physical projection carries the signal |
|| Calibration | Platt-scaled; reliability diagrams confirm <3% average absolute error |

Height is the #1 predictor because it correlates with physical ceiling. All 10,734 prospects now have measured or imputed height (100%), and 100% have BMI — up from 0.4% coverage before the fix detailed below.

---

## July 2026 Fix: Data Quality Audit & Model Corrections

A comprehensive audit of the data pipeline and all three model tiers identified 15 issues. All are now fixed:

| Severity | Issue | Fix |
|----------|-------|-----|
| **CRITICAL** | 56,910 undrafted negatives had `height=0,bmi=0` — model learned "zero height = undrafted" as dominant split (importance 0.708) | Imputed realistic height/weight/BMI from conference+position distributions derived from 6,274 drafted players |
| **CRITICAL** | Only 40/10,734 inference players had BMI — 99.6% got `bmi=0`, which the model treats as "undrafted" | Fixed xMLBAMID guard clause; added unconditional BMI lookup + position-based imputation from 9,921 draft records |
| **HIGH** | Export manifest showed OLD model metadata (conference_tier, 37 features) while serving NEW predictions (conf_strength, 53 features) | Switched export to load exclusively from `models/artifacts_full/` |
| **HIGH** | Calibration view in frontend showed hardcoded fake bins (5000 count, 0.02 rate) | Now loads real calibration curves from `calibration_lookup_*.json` |
| **MODERATE** | Round projection used fixed 30.75 picks/round → pick 315 shown as Rd 11 not Rd 10 | Empirical boundaries from actual MLB draft data (2015-2025) |
| **MODERATE** | Composite score excluded Tier 3 (MLB arrival) and used raw uncalibrated probabilities | Now: 30% draft position + 40% isotonic-calibrated MLB prob + 30% Tier 3 arrival |
| **MODERATE** | Tier 3 `round_logit_prior` hardcoded in inference — would mismatch if retrained | Saved to Tier 3 artifact; inference loads from artifact |
| **LOW** | `mlb_prob_calibrated` field unread by export | Added to frontend data as `mlb_p_cal` |
| **LOW** | Missing height displayed as "0-0" instead of "—" | `fmt_height()` returns `None` for zero |

**Results after fixes:**
- BMI coverage in inference: **0.4% → 100%** (40 → 10,734 players)
- Height coverage in inference: **85% → 100%** (9,215 → 10,734 players)
- Negatives biometric coverage for training: **0% → 100%** (0 → 56,910 with realistic values)
- Frontend manifests: correct features (conf_strength + adj stats + interactions), real calibration curves, accurate round projections

---

## Running It Yourself

```bash
# 1. Full pipeline: scrape data, train models, generate predictions
python3 scripts/infer_2026.py

# 2. Export frontend JSON bundle (player shards, comps, model cards)
python3 scripts/export_frontend_data.py

# 3. Build the static site
cd web && npm install && npm run build

# 4. Serve locally or deploy
npm start                   # local preview at http://localhost:3000
npx serve out               # or any static file server
```

The entire pipeline runs on a laptop. No GPUs, no cloud services, no API keys required.

---

## What I Learned

- **Grouped cross-validation matters**: Multi-year player records leak information if you split randomly. Grouping by player identity before cross-validating gave honest performance estimates.
- **Calibration is not accuracy**: A model with 0.99 AUC can still be overconfident by 2x at decision thresholds. Platt scaling (not just temperature scaling) was the fix.
- **Conference adjustment is non-negotiable**: Without it, the model systematically overrates small-conference production. `conf_strength` as a continuous ratio (not a 4-tier category) was key — it lets the gradient of SEC vs ACC vs A-Sun sort naturally.
- **Prior-offset for rare events**: P(MLB debut | drafted) has a strong baseline by round. Modeling deviation from that baseline (Elastic Net) beat modeling the absolute probability (XGBoost) by 0.05–0.08 AUC.
- **Nearest-neighbor comps in high dimensions**: Standardizing 10 diverse stats (rates, counting, ratios) before Euclidean distance was non-obvious. Z-score normalization by stat type preserved the signal.
- **Static site for ML dashboards**: 38 MB of JSON + Next.js static export + CDN = zero-infrastructure deployment that handles 10K-player datasets. No cold starts, no database connection pools, no server costs.

---

## Project Status

Active development through the 2026 MLB Draft season. The model retrains as new stats arrive. Everything in this repo is open for inspection — no black boxes, no paywalled predictions, no proprietary data.

**Stack:** Python, XGBoost, scikit-learn, NumPy | Next.js 15, TypeScript, Tailwind CSS v4, TanStack Table | Cloudflare R2, GitHub, Vercel

---

*Questions, feedback, or want to use this for your own team? Open an issue or reach out.*
