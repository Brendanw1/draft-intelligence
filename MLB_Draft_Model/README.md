# VT Draft Intelligence — MLB Draft Prediction & Scouting Dashboard

A **full-stack machine learning pipeline** that projects where 10,000+ NCAA Division I baseball players will be selected in the MLB Draft, calibrates their probability of reaching the majors, and surfaces similar historical player comps.

Built as a static Next.js 15 application with a Python/XGBoost backend. No server required — the ML pipeline generates static JSON data that the frontend consumes at build time.

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
| **Player Dossier** | What's the full picture on this specific player? (projections, percentile bars, multi-season stats, top-5 comparable draftees, physical profile, model transparency) |
| **Model Cards** | Why should anyone believe these predictions? (backtest curves, reliability diagrams, feature importances, known limitations) |

---

## Technical Architecture

```
                     ┌─────────────────────────────────────┐
                     │        Data Pipeline (Python)        │
                     │                                     │
                     │  MLB Stats API  ──┐                  │
                     │  FanGraphs D1    ──┤─── Training Set │
                     │  NCAA Rosters   ──┘    6,274 rows   │
                     │                                     │
                     │  ┌──────────────────────────────┐   │
                     │  │  Tier 1: Round Regressor     │   │
                     │  │  (XGBoost — predict draft    │   │
                     │  │   pick number from stats)    │   │
                     │  └──────────────┬───────────────┘   │
                     │                 ▼                   │
                     │  ┌──────────────────────────────┐   │
                     │  │  Tier 2: MLB Probability     │   │
                     │  │  (XGBoost — full population, │   │
                     │  │   56,910 undrafted negatives │   │
                     │  │   + Platt calibration)       │   │
                     │  └──────────────┬───────────────┘   │
                     │                 ▼                   │
                     │  ┌──────────────────────────────┐   │
                     │  │  Nearest-Neighbor Comps      │   │
                     │  │  (Euclidean distance across  │   │
                     │  │   10 stat dimensions → 5     │   │
                     │  │   most similar draftees)     │   │
                     │  └──────────────┬───────────────┘   │
                     │                 ▼                   │
                     │       65 JSON files (25 MB)         │
                     └──────────────────┬──────────────────┘
                                        │
                   ┌────────────────────┘
                   ▼
┌──────────────────────────────────────┐
│   Next.js 15 Static Site (no DB)     │
│                                      │
│   - Virtualized 10K-row table        │
│   - Client-side sorting/filtering    │
│   - Per-player dossier pages         │
│   - Model backtest plots             │
│   - Static export → any CDN          │
└──────────────────────────────────────┘
```

**Key design decisions:**

- **Two-tier model**: Separates "when will you be drafted?" (Tier 1 regressor) from "will you ever reach MLB?" (Tier 2 classifier). These are different questions requiring different architectures.
- **Full-population training**: Tier 2 includes 56,910 undrafted players as true negatives. Most draft models skip this — the model learns what *doesn't* get drafted, not just who does.
- **Platt calibration**: Raw XGBoost probabilities are systematically overconfident. Platt scaling maps them to well-calibrated MLB% estimates verified by reliability diagrams.
- **Static export**: Zero runtime infrastructure. No database, no API, no server. The pipeline generates JSON → Next.js builds a static site → deploy anywhere (Vercel, Cloudflare Pages, S3).
- **Sharded data**: 10,734 player records split into 64 shards + 1 index file. The board loads from the index (9 MB, fast); dossiers fetch single shards lazily.

---

## Data Sources

All sourced from **freely public APIs and websites**:

| Source | Data | Coverage |
|--------|------|----------|
| MLB Stats API | Draft picks, rounds, bonuses, schools, physicals | 2015–2026, ~1,200 picks/year |
| FanGraphs College Leaderboards | Hitting & pitching stats (AVG, OPS, wOBA, ERA, FIP, K/BB, etc.) | 2021–2026, ~4,000+ players/year |
| NCAA D1 Rosters | Height, positions, conferences | 2026 current, 85% height coverage |

No proprietary TrackMan data, no internal team data, no paywalled sources.

---

## Model Performance (withheld test set)

| Metric | Hitters | Pitchers |
|--------|---------|----------|
| Year-out AUC (Tier 2) | 0.994 | 0.989 |
| Dominant feature | height_inches (0.764) | height_inches (0.616) |
| Position/velocity | No single skill stat dominates — physical projection carries the signal |

Height is the #1 predictor because it correlates with physical ceiling, and it's available for 85% of players (unlike weight or arm speed from public data).

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
- **Nearest-neighbor comps in high dimensions**: Standardizing 10 diverse stats (rates, counting, ratios) before Euclidean distance was non-obvious. Z-score normalization by stat type preserved the signal.
- **Static site for ML dashboards**: 25 MB of JSON + Next.js static export + CDN = zero-infrastructure deployment that handles 10K-player datasets. No cold starts, no database connection pools, no server costs.

---

## Project Status

Active development through the 2026 MLB Draft season. The model retrains as new stats arrive. Everything in this repo is open for inspection — no black boxes, no paywalled predictions, no proprietary data.

**Stack:** Python, XGBoost, scikit-learn, NumPy | Next.js 15, TypeScript, Tailwind CSS, TanStack Table | GitHub, Vercel

---

*Questions, feedback, or want to use this for your own team? Open an issue or reach out.*
