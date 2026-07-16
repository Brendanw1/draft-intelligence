# VT Draft Intelligence

An **ML-powered MLB draft projection dashboard** — predicts where 10,000+ NCAA Division I baseball players will be drafted, estimates their probability of reaching the majors, and surfaces comparable historical players. All from public data.

---

## The Problem

College baseball lacks a centralized, data-driven draft board. Scouts and analysts maintain private lists. Public projections are either paywalled or rely on subjective rankings. There's no open, reproducible baseline that any team or fan can inspect, challenge, or build on.

This project builds that baseline.

## What It Does

| View | Answers |
|------|---------|
| **Board** | Who should I be watching? (10,734 players, sortable by grade tier, position, conference, draft round) |
| **Value** | Where will the market misprice talent? (projected pick vs calibrated MLB probability) |
| **Class Retrospectives** | How did past draft classes turn out? (2021–2026 with real outcomes) |
| **Player Dossier** | What's the full picture on this player? (predictions, percentile bars, multi-season stats, 5 nearest historical comps, physical profile, model transparency) |
| **Model Cards** | Why should I believe these predictions? (backtest curves, reliability diagrams, feature importances, known limitations) |

## Architecture

```
Public APIs & data ──→ Three-Tier Model ──→ JSON (25 MB) ──→ Next.js static site ──→ CDN
```

**Tier 1 — Draft Position (XGBoost Regressor)**
Predicts draft pick number from college stats + **conf_strength** (continuous, replaces broken 4-tier system) + conference-adjusted stats + interaction features.

**Tier 2 — Draft Probability (XGBoost Classifier)**
Predicts P(drafted in top 10 rounds) from same features. Retrained with conf_strength, conference-adjusted stats, and interaction features to properly discount low-conference stat inflation.

**Tier 3 — MLB Arrival (Elastic Net + Nearest Neighbors)**
Predicts P(reaches MLB | drafted) using:
- Elastic Net logistic regression with L1/L2 regularization
- **Round logit prior**: empirical MLB debut rate per draft round as a statistical baseline
- **Nearest-neighbor MLB rate**: proportion of 20 most similar drafted players who reached MLB
- Trained on 2021-2023 drafted players with verified MLB debut dates from the MLB Stats API

**No server, no database, no API.** The entire pipeline generates static JSON files; the frontend is a Next.js static export deployable anywhere (Vercel, Cloudflare Pages, S3).

## Data Sources — All Public

| Source | Data | Coverage |
|--------|------|----------|
| [MLB Stats API](https://statsapi.mlb.com/docs/) | Draft picks, rounds, bonuses, schools, physicals | 2015–2026 |
| [FanGraphs College Leaderboards](https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=college) | Hitting/pitching stats (AVG, OPS, wOBA, ERA, FIP, K/BB, etc.) | 2021–2026 |
| NCAA D1 Rosters | Height, positions, conferences | 2026 current |

No proprietary TrackMan data, no team-internal data, no paywalled sources.

## Model Performance

| | Hitters | Pitchers |
|---|---|---|
| Year-out AUC (Tier 2) | 0.994 | 0.989 |
| Dominant feature | height_inches (0.764) | height_inches (0.616) |

Height is the #1 predictor (available for 85% of players) because it correlates with physical ceiling. No single skill stat dominates — the model learns that projection matters more than any one stat line.

## Quick Start

```bash
# Full pipeline (laptop, no GPU needed)
python3 MLB_Draft_Model/scripts/infer_2026.py
python3 MLB_Draft_Model/scripts/export_frontend_data.py

# Build & serve
cd MLB_Draft_Model/web && npm install && npm run build && npm start
```

## Repo Structure

| Directory | Contents |
|-----------|----------|
| `MLB_Draft_Model/` | **The draft project** — all pipeline scripts, models, training data, and frontend |
| `MatchupApp/` | R/Shiny pitch-similarity matchup tool (separate project) |
| `BatTracking/` | R package for bat-tracking analytics (separate project) |
| `Apollo/` | R/Plumber + Next.js war-room API (separate project) |

The other directories are standalone baseball analytics projects in the same monorepo. Only `MLB_Draft_Model/` is relevant to this README.

---

*Everything in this repository uses freely public data. Built as an open, reproducible baseline for MLB draft analysis.*
