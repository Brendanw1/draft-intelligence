# MLB Draft Intelligence — Prediction & Scouting Dashboard

A **full-stack machine learning pipeline** that projects where 10,000+ NCAA Division I baseball players will be selected in the MLB Draft, estimates their probability of being drafted in the top 315 picks (about 10 rounds) within three drafts, and calculates their MLB arrival outlook if drafted. Surfaces similar historical player comps with interactive pick-axis visualization.

Built as a static Next.js 15 application with a Python/XGBoost/scikit-learn backend. No server required — the ML pipeline generates static JSON data that the frontend consumes at build time and serves from Cloudflare R2.

---

## The Problem

College baseball has **no centralized draft board**. Scouts, front offices, and analysts each maintain their own lists. Public projections are either paywalled (FanGraphs, Baseball America) or rely on a single human opinion. There's no open, reproducible, data-driven baseline that any team or fan can inspect.

This project is that baseline.

---

## What It Does

| Page | What it answers |
|------|----------------|
| **Board** | Who matters in the 2026 class? (10,734 players, sortable/filterable by grade tier, position, conference, draft round band) |
| **Value** | Where will the market misprice talent? (projected pick if drafted vs calibrated top-315 probability) |
| **Class Retrospectives** | How did past draft classes actually turn out? (2021–2026 with real outcomes) |
| **Player Dossier** | What's the full picture on this specific player? (projections, percentile bars, multi-season stats, top-10 comparable draftees with interactive pick-axis dot plot, physical profile, tiered model transparency including MLB arrival outlook, model disagreement indicators) |
| **Model Lab** | Why should anyone believe these predictions? (backtest curves, reliability diagrams, feature importances, known limitations, per-model calibration audits) |
| **Methodology** | How exactly does each tier work, what are the known gaps, and what changed in the latest vintage? |

---

## Architecture

Pipeline on the site: **XGBoost regressor** (Tier 1 → projected draft pick, if drafted) → **XGBoost classifier** with Platt calibration fit on out-of-fold scores (Tier 2 → P(pick ≤ 315 within three drafts)) → **logistic regression with a draft-round logit offset** (Tier 3 → P(MLB debut within 3 seasons) if drafted) → **Nearest-neighbor comps** (Tier 4 → similar historical players). A career-WAR tier (Tier 5) was built and withdrawn after failing its backtest. All fed into a static Next.js 15 site with no runtime infrastructure.

**Key design decisions:**

- **Tiered architecture**: Separates "when will you be drafted?" (Tier 1), "will you go in the top 315 within three drafts?" (Tier 2), "if drafted, will you debut within 3 seasons?" (Tier 3) and "who are comparable players?" (Tier 4). Each tier answers a distinct question with appropriate modeling.
- **Top-315 target, fixed horizon**: Tier 2 predicts a pick of 315 or better (about 10 rounds) in this season's draft or the next two, not generic "MLB probability." The fixed window means every training season is equally observed. Calibrated on out-of-fold scores and checked on the 2024 class (see below).
- **Conference adjustment**: Raw stats (wOBA, ERA, etc.) are multiplied by the inverse of `conf_strength` — a continuous score based on empirical draft rates per conference. SEC production (2.98× draft rate) isn't compared at face value to SWAC production (0.09×).
- **Full-population training**: Tier 2 trains on every D1 season with a FanGraphs line, 2021–2024, drafted or not, freshmen included (41,725 player-seasons). Age, height and BMI are left out because the sources record them far more completely for drafted players than for undrafted ones. Most negatives are easy to separate, so the AUC below still flatters the ranking of real prospects.
- **Calibration on out-of-fold scores**: Raw XGBoost probabilities are systematically overconfident (~2.3×). The displayed probability is a quantile map fit on player-grouped out-of-fold scores; Platt and isotonic maps, fit on forward-chained out-of-fold scores, are cross-checks.
- **Prior-offset Tier 3**: Tier 3 predicts *deviation* from the draft round's historical debut rate, passed as a fixed logit offset that is computed out of fold. A 5th-rounder projected at 15% arrival is above their round baseline; a 1st-rounder at 15% is below theirs. Honest result: the college stats add no ranking power beyond the round (see below).
- **Interactive NN comps**: Euclidean nearest-neighbor search across 10 standardized stat dimensions yields 10 comparable draftees per player. An interactive pick-axis dot plot links hovered comps to their draft position with bidirectional row↔dot highlighting.
- **Composite score**: Weighted combination of all tiers: 30% draft position (Tier 1) + 40% calibrated top-315 probability (Tier 2) + 30% MLB arrival (Tier 3).
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

## Model Performance — Rolling Year-Out Backtest

The model is tested using a **rolling year-out backtest**: for each holdout year (2022–2026), models are trained exclusively on data from prior years (draft_year ≤ Y−1), then compared against actual outcomes. Three naive baselines provide context:

1. **Naive mean**: predict every draftee at the training set's historical mean pick
2. **Conference mean**: predict at the conference-level historical average
3. **Round average (reference)**: predict at the round-level average (using actual round — not a fair comparison, included as an upper bound)

| Holdout | n_train | Model MAE | Naive MAE | Conf MAE | ±110% | Spearman ρ |
|---------|---------|-----------|-----------|----------|-------|------------|
| 2022    | 327     | 130.7     | 146.5     | 149.8    | 47%   | 0.400      |
| 2023    | 688     | 118.2     | 140.2     | 144.2    | 54%   | 0.486      |
| 2024    | 1098    | 114.8     | 144.1     | 143.6    | 55%   | 0.526      |
| 2025    | 1524    | 116.7     | 142.0     | 140.0    | 55%   | 0.496      |
| 2026    | 1929    | 118.4     | 146.3     | 144.9    | 53%   | 0.526      |
| **Avg** | **913** | **119.8** | **143.8** | **144.5**| **53%** | **0.487** |

The model beats the naive mean by **17%** consistently across all five holdout years. The ±110-pick range band achieves a 53% capture rate — matching the backtest estimate. The Spearman rank correlation of ~0.50 confirms meaningful rank ordering. Performance plateaus after ~3 years of training data (2024 forward), suggesting the limiting factor is feature quality, not sample size. Full breakdown in [`analysis/draft_intelligence_backtest.md`](analysis/draft_intelligence_backtest.md).

### Tier 2 and Tier 3

| Metric | Hitters | Pitchers |
|--------|---------|----------|
| Tier 1 projected pick (if drafted), MAE forward by draft year: 2024 / 2025 | 120.7 / 114.2 | 113.2 / 117.2 |
| Tier 1 baseline (predict the training mean pick), MAE 2024 / 2025 | 145.3 / 145.1 | 139.2 / 137.7 |
| Tier 2 top-315-within-3-drafts AUC, 2024 class (re-read) | 0.857 | 0.843 |
| Tier 2 Brier / ECE on the 2024 class (Platt on player-grouped out-of-fold scores) | 0.044 / 1.4% | 0.049 / 0.6% |
| Tier 3 debut within 3 seasons given drafted, AUC (nested out-of-fold CV, 2021–22 classes) | 0.698 | 0.747 |
| Tier 3 draft-round rate alone, AUC (same folds) | 0.723 | 0.774 |

- **Tier 2 was rebuilt on 2026-10-29** after a split audit. The old 0.977 / 0.966 came mostly from shortcuts in how the
  training set was built, not from the stats: (1) negatives with Age < 20 were dropped while drafted players kept their
  freshman seasons, and FanGraphs has Age for every drafted player but for under half of the undrafted, so "young" or
  "no age" told the model the answer; (2) drafted players carry measured whole-inch heights from the draft record while
  undrafted rows carried random imputed heights (87% non-integer), so "whole-inch height" told it too, and on the 2026
  board, where every height is a roster whole inch, the probabilities ran 2-3x hot; (3) the label was "eventually
  drafted in the top 315", so recent seasons were censored; (4) test players' earlier rows were dropped from training;
  (5) the site showed a quantile curve that was never scored out of sample; (6) conf_strength used every season's draft
  outcomes. Now: the label is a fixed horizon (pick ≤ 315 in this season's draft or the next two), seasons 2021–2024
  only (the ones whose three drafts have happened), freshmen kept in both classes, no Age / height / BMI, conf_strength
  refit on each fit's training rows, and the site shows the Platt output that was scored.
- **The 2024 number is a re-read, not a clean holdout.** The 2024 class was read on several earlier specs, and twice on
  2026-10-29 (once before the height shortcut was found: AUC 0.950 / 0.941). The holdout model trains only on 2021
  seasons, the only ones whose labels were complete before the 2024 draft. The next clean read is the 2025 class after
  the 2027 draft.
- **Board check:** the board's `mlb_p` sums to 714 expected top-315 picks over three drafts (about 238 a year; roughly
  250 four-year college players go in the top 315 each year). It summed to 1,696 before. Freshmen in the board's top 100:
  15 before, 3 now; players with no Age in FanGraphs: 0 before (mean `mlb_p` 0.000), 2 now (mean 0.029).
- Tier 1 is trained on drafted players only, so `proj_pick` is the pick **if drafted**. Its MAE is now reported forward by
  draft year with conf_strength refit on labels complete before that draft; the old ~110 came from a random 80/20 split.
- Tier 3 is conditional on being drafted, so it is not a population-level arrival rate. The label is a debut by the end of the third season after the draft. Debut dates were pulled on 2026-07-14, so 2026 counts as incomplete and only the 2021–2022 classes train (360 hitters, 55 debuts; 473 pitchers, 64 debuts). **Draft round carries it**: the round rate alone ranks players better than the full model, so read Tier 3 as the round's rate, nudged.

---

## 2026 Draft — Prospective Validation

The model's 2026 projections can be compared against actual draft outcomes with a clean temporal split. **Models were trained on ≤2025 data only** — no 2026 outcomes were used during training. The numbers below reflect true out-of-sample predictive accuracy.

| Metric | Value |
|--------|-------|
| Prospectively matchable draftees | 454 of 474 (96%) |
| Mean absolute pick error | **116.4 picks** |
| Median absolute pick error | 101.4 picks |
| Spearman rank correlation | 0.559 (p < 0.001) |
| Within projected pick range (±~110 picks) | **54%** |
| Within ±150 picks | 69% |
| Within ±200 picks | 84% |
| Drafted higher than projected (model undervalued) | 45% |
| Drafted lower than projected (model overvalued) | 55% |

The prospective MAE of **116.4 picks** validates the ±110 backtest estimate used throughout the UI. The Spearman ρ of 0.56 (p < 0.001) shows meaningful rank ordering. The 96% match rate was achieved by combining person_id joins with fuzzy name+school matching. Full report in [`analysis/2026_draft_accuracy_prospective.md`](analysis/2026_draft_accuracy_prospective.md).

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

## Tier 5 (expected career WAR): withdrawn, failed its backtest

Tier 5 is off the site (no WAR leaderboard, no WAR card). Scored as P(reach MLB) × E[WAR | reached], out of fold on the 2021 class (the only class with five pro seasons complete), it reached R² +0.027 for hitters and −0.032 for pitchers, no better than predicting the median for pitchers ([`docs/tier5_backtest_report.md`](docs/tier5_backtest_report.md)). The earlier gate numbers below, from the older two-stage build, also failed:

| Gate | Hitters | Pitchers |
|------|---------|----------|
| Value R² vs. predicting the median (must be ≥ 0) | −0.319 | −0.611 |
| Hurdle held-out AUC [95% CI] | 0.932 [0.883, 0.971] | 0.692 [0.514, 0.860] |

Why it fails: one usable draft class (2021) with 33 positives in 1,124 players, right-censored careers, and inputs that encode draft slot, so the hitter AUC is mostly the round prior. Full report in [`docs/tier5_honesty_report.md`](docs/tier5_honesty_report.md). It stays experimental until a larger, uncensored cohort passes the gates.

---

## What I Learned

- **Grouped cross-validation matters**: Multi-year player records leak information if you split randomly. Grouping by player identity before cross-validating gave honest performance estimates.
- **Calibration is not accuracy**: A model with 0.97 AUC can still be overconfident by 2x at decision thresholds. Platt scaling (not just temperature scaling) was the fix.
- **Conference adjustment is non-negotiable**: Without it, the model systematically overrates small-conference production. `conf_strength` as a continuous ratio (not a 4-tier category) was key — it lets the gradient of SEC vs ACC vs A-Sun sort naturally.
- **Biometric imputation prevents artifact learning**: Zero-imputed missing data lets tree-based models learn "missingness = negative outcome" as a perfect split. Stochastic imputation from conference+position distributions eliminates this artifact while preserving signal.
- **A leaky prior flatters everything**: P(MLB debut | drafted) has a strong baseline by round. Once that baseline was computed out of fold (it had included each player's own outcome), the stats on top of it stopped adding ranking power. Check the prior-only model before crediting the features.
- **Interactive NN comps**: Euclidean nearest-neighbor search benefits from z-score normalization across diverse stat types (rates, counting, ratios). The pick-axis dot plot makes comps scannable — hover linking between dots and list rows ties spatial position to player identity without cluttering the SVG.
- **Static site for ML dashboards**: 38 MB of JSON + Next.js static export + CDN = zero-infrastructure deployment that handles 10K-player datasets. No cold starts, no database connection pools, no server costs.

---

## Project Status

Active development through the 2026 MLB Draft season. The model retrains as new stats arrive. Everything in this repo is open for inspection — no black boxes, no paywalled predictions, no proprietary data.

**Stack:** Python, XGBoost, scikit-learn, NumPy | Next.js 15, TypeScript, Tailwind CSS v4, TanStack Table | Cloudflare R2, GitHub, Vercel

---

*Questions, feedback, or want to use this for your own team? Open an issue or reach out.*
