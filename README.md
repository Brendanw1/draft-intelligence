# MLB Draft Intelligence — Prediction & Scouting Dashboard

A **full-stack machine learning pipeline** that projects where 10,000+ NCAA Division I baseball players will be selected in the MLB Draft, estimates their probability of being drafted in the top 10 rounds, and calculates their MLB arrival outlook if drafted. Surfaces similar historical player comps with interactive pick-axis visualization.

Built as a static Next.js 15 application with a Python/XGBoost/Elastic Net backend. No server required — the ML pipeline generates static JSON data that the frontend consumes at build time and serves from Cloudflare R2.

---

## The Problem

College baseball has **no centralized draft board**. Scouts, front offices, and analysts each maintain their own lists. Public projections are either paywalled (FanGraphs, Baseball America) or rely on a single human opinion. There's no open, reproducible, data-driven baseline that any team or fan can inspect.

This project is that baseline.

---

## What It Does

| Page | What it answers |
|------|----------------|
| **Board** | Who matters in the 2026 class? (10,734 players, sortable/filterable by grade tier, position, conference, draft round band) |
| **Value** | Where will the market misprice talent? (projected pick vs calibrated top-10-round probability) |
| **Class Retrospectives** | How did past draft classes actually turn out? (2021–2026 with real outcomes) |
| **Player Dossier** | What's the full picture on this specific player? (projections, percentile bars, multi-season stats, top-10 comparable draftees with interactive pick-axis dot plot, physical profile, tiered model transparency including MLB arrival outlook, model disagreement indicators) |
| **Model Lab** | Why should anyone believe these predictions? (backtest curves, reliability diagrams, feature importances, known limitations, per-model calibration audits) |
| **Methodology** | How exactly do the three tiers work, what are the known gaps, and what changed in the latest vintage? |

---

## Architecture

Three-tier pipeline: **XGBoost regressor** (Tier 1 → projected draft pick) → **XGBoost classifier** with Platt calibration (Tier 2 → top-10-round probability) → **prior-offset Elastic Net** (Tier 3 → MLB arrival probability if drafted). Nearest-neighbor comps provide context. All fed into a static Next.js 15 site with no runtime infrastructure.

**Key design decisions:**

- **Three-tier architecture**: Separates "when will you be drafted?" (Tier 1 regressor), "will you go in the top 10 rounds?" (Tier 2 classifier), and "if drafted, what's your arrival probability?" (Tier 3 prior-offset). These are different questions requiring different architectures.
- **Top-10-round target**: Tier 2 predicts top-10-round draft status (pick ≤315), not generic "MLB probability." The top-10-round cutoff is where draft value crystallizes — players drafted after round 10 have dramatically lower signing rates and career ceilings. Calibrated via Platt scaling with reliability diagram verification.
- **Conference adjustment**: Raw stats (wOBA, ERA, etc.) are multiplied by the inverse of `conf_strength` — a continuous score based on empirical draft rates per conference. SEC production (2.98× draft rate) isn't compared at face value to SWAC production (0.09×).
- **Full-population training**: Tier 2 includes 56,910 undrafted players as true negatives with biometrically imputed height/weight/BMI. Most draft models skip this — the model learns what *doesn't* get drafted, not just who does.
- **Platt + Isotonic calibration**: Raw XGBoost probabilities are systematically overconfident (~2.3×). Platt scaling maps them to well-calibrated probabilities; isotonic calibration provides a secondary cross-check.
- **Prior-offset Tier 3**: Instead of predicting absolute MLB arrival probability from scratch, Tier 3 predicts *deviation* from a round-specific historical baseline. A 5th-rounder projected at 15% arrival is above their round baseline; a 1st-rounder at 15% is below theirs.
- **Interactive NN comps**: Euclidean nearest-neighbor search across 10 standardized stat dimensions yields 10 comparable draftees per player. An interactive pick-axis dot plot links hovered comps to their draft position with bidirectional row↔dot highlighting.
- **Composite score**: Weighted combination of all three tiers: 30% draft position (Tier 1) + 40% calibrated top-10-round probability (Tier 2) + 30% MLB arrival (Tier 3), scaled 0–100.
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
| Tier 2 year-out AUC | 0.97 | 0.97 |
| Tier 3 AUC (arrival) | 0.79 | 0.79 |
| Tier 1 year-out MAE | ±111 picks | ±108 picks |
| Top features | conf_strength, wOBA_adj, K_pct_adj, age | conf_strength, FIP_adj, K-BB%, velo proxy |
| Calibration | Platt-scaled; reliability diagrams confirm <3% average absolute error | Same |

After the July 2026 data quality audit, biometric features (height, BMI) dropped from artificial dominance (importance 0.71) to realistic proportional weight (~0.05–0.10), allowing conference-adjusted performance stats to carry the predictive signal as intended. All 10,734 prospects now have measured or imputed height and BMI (100% coverage).

---

## 2026 Draft — Prospective Validation

The model's 2026 projections can be compared against actual draft outcomes with a clean temporal split. **Models were trained on ≤2025 data only** — no 2026 outcomes were used during training. The numbers below reflect true out-of-sample predictive accuracy.

| Metric | Value |
|--------|-------|
| Prospectively matchable draftees | 212 of 474 (44.7%) |
| Mean absolute pick error | **115.7 picks** |
| Median absolute pick error | 100.0 picks |
| Within projected pick range (±~110 picks) | **54%** |
| Drafted higher than projected | 19% |
| Drafted lower than projected | 25% |

The lower match rate (44.7% vs the retrospective 93%) is a function of the matching methodology — enriched projections use abbreviated school codes (`LSU`) while draft records use full school names (`Louisiana State University`). A better crosswalk would improve this. The true prospective MAE of ±116 picks validates the ±110-pick backtest estimate. Full report in [`analysis/2026_draft_accuracy_prospective.md`](analysis/2026_draft_accuracy_prospective.md).

Full round-by-round breakdown, biggest misses, best predictions, and unmatched player analysis in the retrospective [`analysis/2026_draft_accuracy.md`](analysis/2026_draft_accuracy.md).

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
| **MODERATE** | Composite score used wrong weights and excluded Tier 3 | Now: 30% draft position + 40% calibrated top-10-round prob + 30% Tier 3 arrival |
| **MODERATE** | Tier 3 `round_logit_prior` hardcoded in inference — would mismatch if retrained | Saved to Tier 3 artifact; inference loads from artifact |
| **LOW** | `mlb_prob_calibrated` field unread by export | Added to frontend data |
| **LOW** | Missing height displayed as "0-0" instead of "—" | `fmt_height()` returns `None` for zero |

**Results after fixes:**
- BMI coverage in inference: **0.4% → 100%** (40 → 10,734 players)
- Height coverage in inference: **85% → 100%** (9,215 → 10,734 players)
- Negatives biometric coverage for training: **0% → 100%** (0 → 56,910 with realistic values)
- Frontend manifests: correct features (conf_strength + adj stats + interactions), real calibration curves, accurate round projections

---

## What I Learned

- **Grouped cross-validation matters**: Multi-year player records leak information if you split randomly. Grouping by player identity before cross-validating gave honest performance estimates.
- **Calibration is not accuracy**: A model with 0.99 AUC can still be overconfident by 2x at decision thresholds. Platt scaling (not just temperature scaling) was the fix.
- **Conference adjustment is non-negotiable**: Without it, the model systematically overrates small-conference production. `conf_strength` as a continuous ratio (not a 4-tier category) was key — it lets the gradient of SEC vs ACC vs A-Sun sort naturally.
- **Biometric imputation prevents artifact learning**: Zero-imputed missing data lets tree-based models learn "missingness = negative outcome" as a perfect split. Stochastic imputation from conference+position distributions eliminates this artifact while preserving signal.
- **Prior-offset for rare events**: P(MLB debut | drafted) has a strong baseline by round. Modeling deviation from that baseline (Elastic Net) beat modeling the absolute probability (XGBoost) by 0.05–0.08 AUC.
- **Interactive NN comps**: Euclidean nearest-neighbor search benefits from z-score normalization across diverse stat types (rates, counting, ratios). The pick-axis dot plot makes comps scannable — hover linking between dots and list rows ties spatial position to player identity without cluttering the SVG.
- **Static site for ML dashboards**: 38 MB of JSON + Next.js static export + CDN = zero-infrastructure deployment that handles 10K-player datasets. No cold starts, no database connection pools, no server costs.

---

## Project Status

Active development through the 2026 MLB Draft season. The model retrains as new stats arrive. Everything in this repo is open for inspection — no black boxes, no paywalled predictions, no proprietary data.

**Stack:** Python, XGBoost, scikit-learn, NumPy | Next.js 15, TypeScript, Tailwind CSS v4, TanStack Table | Cloudflare R2, GitHub, Vercel

---

*Questions, feedback, or want to use this for your own team? Open an issue or reach out.*
