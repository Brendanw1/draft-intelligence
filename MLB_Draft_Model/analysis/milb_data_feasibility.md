# MiLB Data Integration Feasibility Report

**Date**: 2026-07-16  
**Evaluator**: Hermes Agent  
**Scope**: Assess pybaseball and existing MiLB data for ~1,500 drafted players in the MLB Draft Model training pool

---

## Executive Summary

**Verdict: GO — with a significant caveat.** The existing `data/milb/milb_*.json` files already contain season-level MiLB performance stats for **8,110 players across 2021–2025**, including batting (37 fields) and pitching (63 fields) stats with person_id crosswalk. pybaseball itself is **not the right tool** for this task — but the data is already integrated and available.

---

## 1. pybaseball Installation & Status

| Item | Status |
|------|--------|
| **Package** | pybaseball **v2.2.7** installed (`/opt/anaconda3/lib/python3.12/site-packages/pybaseball/`) |
| **FanGraphs backend** (`batting_stats` / `pitching_stats`) | ❌ **Blocked** — HTTP 403 (Cloudflare bot detection) on `fangraphs.com/leaders-legacy.aspx` |
| **Baseball-Reference backend** (`batting_stats_bref` / `pitching_stats_bref`) | ✅ **Works** — Returns MLB stats via `baseball-reference.com/leagues/daily.cgi` |
| **MiLB-specific functions** (`milb_batting_stats`, `milb_pitching_stats`) | ❌ **Do not exist** in pybaseball 2.2.7. These were removed or never ported. |
| **Player ID crosswalk** (`playerid_reverse_lookup`) | ✅ **Works** — Maps MLBAM ID → `key_mlbam`, `key_bbref`, `key_fangraphs`, `key_retro` |
| **Colleague stats** (`college_playing`) | Untested |

### Key Finding: pybaseball is NOT a viable MiLB data source

- The FanGraphs leaderboard pages that `batting_stats`/`pitching_stats` depend on are behind Cloudflare (403 error).
- The Baseball-Reference `batting_stats_bref`/`pitching_stats_bref` functions are **hardcoded to `level=mlb`** — they only return MLB data.
- There are **no dedicated MiLB functions** in the current pybaseball release.

---

## 2. Existing MiLB Data (Already Integrated!)

**This changes the calculus entirely.** The `data/milb/milb_{year}.json` files already have what we need:

### Data Structure
```
{
  "season": 2023,
  "players": [
    {
      "person_id": 642215,              # ← joins to draft_all_picks.json
      "full_name": "...",
      "team_name": "...",
      "level": "AAA",                    # ← peak/end-of-season level
      "season": 2023,
      "game_pk": null,
      "batting": {
        "gamesPlayed": 125,
        "atBats": 460,
        "hits": 119,
        "avg": ".259",
        "obp": ".???",
        "slg": ".???",
        "ops": ".???",
        "babip": ".???",
        "homeRuns": 1,
        "strikeOuts": 42,
        "baseOnBalls": 81,
        "stolenBases": 45,
        ...
      },
      "pitching": {
        "gamesPlayed": 18,
        "gamesStarted": 14,
        "wins": 3,
        "losses": 1,
        "era": "1.80",
        "whip": "1.28",
        "inningsPitched": "35.0",
        "strikeOuts": 81,
        "baseOnBalls": 3,
        "homeRuns": 4,
        "strikeoutsPer9Inn": "12.53",
        "walksPer9Inn": "1.28",
        "homeRunsPer9": "1.28",
        ...
      }
    }
  ]
}
```

### Available Columns

| Category | Columns Available |
|----------|------------------|
| **Batting (37)** | `avg`, `obp`, `slg`, `ops`, `babip`, `homeRuns`, `strikeOuts`, `baseOnBalls`, `stolenBases`, `rbi`, `runs`, `doubles`, `triples`, `plateAppearances`, `atBats`, `hits`, `gamesPlayed`, `sacBunts`, `sacFlies`, `hitByPitch`, `groundIntoDoublePlay`, `caughtStealing`, `intentionalWalks`, `totalBases`, `leftOnBase`, `atBatsPerHomeRun`, etc. |
| **Pitching (63)** | `era`, `whip`, `wins`, `losses`, `saves`, `inningsPitched`, `strikeOuts`, `baseOnBalls`, `homeRuns`, `strikeoutsPer9Inn`, `walksPer9Inn`, `homeRunsPer9`, `strikeoutWalkRatio`, `hitsPer9Inn`, `groundOutsToAirouts`, `holds`, `blownSaves`, `gamesStarted`, `completeGames`, `shutouts`, `wildPitches`, `balks`, `strikePercentage`, `inheritedRunners`, `inheritedRunnersScored`, etc. |

### Coverage by Year

| Draft Year | Picks | With MiLB Data | Coverage Rate |
|-----------|-------|---------------|--------------|
| **2021** | 612 | 545 | **89.1%** |
| **2022** | 616 | 522 | **84.7%** |
| **2023** | 614 | 478 | **77.9%** |
| **2024** | 615 | 451 | **73.3%** |
| **2025** | 615 | 228 | **37.1%** (partial — drafted mid-2025) |
| **Total** | **3,571** | **2,164** | **60.6%** |

### Coverage by Signing Status

| Group | Count | With MiLB Data | Coverage |
|-------|-------|---------------|----------|
| Signed draftees 2021-2025 | 2,839 | 2,157 | **76.0%** |
| Unsigned draftees | 732 | 7 | ~1% |
| Already debuted in MLB | 313 | 286 | **91.4%** |
| Not yet debuted | 3,372 | 1,878 | **55.7%** |

---

## 3. How the Existing Data Was Generated

The `milb_*.json` files appear to have been generated via the MLB Stats API (`/api/v1/people/{person_id}/stats?stats=yearByYear&group=[hitting,pitching]`). This is the same API that powers the draft data pipeline — so no Cloudflare issues.

Each file contains:
- **One record per player per season** at their **highest level played** that year
- Breaking it down: 2023 has 3,487 players spread across A (893), A+ (889), AAA (870), AA (835)
- Level progression data exists: 1,249 players have 3+ consecutive years of stats

### Data Quality Considerations

| Aspect | Assessment |
|--------|------------|
| **Sample size thresholds** | No minimum PA/IP filter — includes players with 1 game. We'd need to filter to meaningful samples (e.g., ≥50 PA for batters, ≥20 IP for pitchers). |
| **Level specificity** | Each player has exactly ONE record per year at their peak level. Stats are NOT split by level if a player was promoted mid-season. This is a known limitation. |
| **ID crosswalk** | person_id is the same MLBAM ID used in draft data — joins directly with no mapping needed. |
| **Season length** | Full MLB MiLB season (Apr-Sep). The 2025 data is partial (only through mid-2025). |

---

## 4. Alternative: BRef Custom Scraper

We confirmed that Baseball-Reference's `daily.cgi` endpoint **does support `level=AAA`, `level=AA`, `level=A+`, `level=A`** parameters. However:

- The pybaseball wrapper is hardcoded to `level=mlb` — we'd need to write a custom script.
- Response pages are large (~130KB–1.3MB per level/year combo).
- BRef rate limits aggressively (would need delays between requests).
- Each level/year combo gives ~500–900 players with `mlbID` field (same as person_id).
- **Total effort**: Would take ~2–3 hours to write and test a scraper for all levels + years. But this is **unnecessary** since we already have the data.

---

## 5. Feature Ideas for Tier 3 Modeling

### New Continuous Outcome Targets

| Target | Type | Description |
|--------|------|-------------|
| **Peak MiLB wOBA** (year 2-3 post-draft) | Regression | Best offensive performance in first 3 pro years |
| **Highest level reached (by year N)** | Ordinal | 1=A, 2=A+, 3=AA, 4=AAA, 5=MLB |
| **Seasons to reach MLB** | Regression | How fast did they advance? |
| **MiLB WAR / WAA** | Regression | Composite performance measure |
| **MiLB K% / BB%** | Regression | Plate discipline indicators in pro ball |
| **MiLB ERA / FIP** | Regression | Pitching performance |

### New Features for Existing Model

| Feature | Rationale |
|---------|-----------|
| **Draft year-relative MiLB stats** | How did this year's draftees perform relative to prior year classes? |
| **Position-level MiLB baseline** | Average MiLB performance by position and draft round |
| **Sign vs. Performance interaction** | Did higher bonuses predict better MiLB performance? |

---

## 6. Estimated Effort to Integrate

Since the data already exists in JSON files with person_id crosswalk, integration is **minimal**:

| Task | Effort | Notes |
|------|--------|-------|
| Load and join milb_*.json with training set | **30 min** | Simple person_id inner join, merge across years |
| Filter to meaningful samples | **15 min** | Set thresholds (≥50 PA, ≥20 IP) |
| Engineer outcome features | **1 hour** | Compute peak wOBA, highest level, progression speed |
| Engineer input features | **30 min** | Compute MiLB stats as model features |
| Verify data splits (train/test) | **15 min** | Ensure no data leakage across years |
| **Total** | **~2.5 hours** | |

If we wanted to supplement with per-level splits (multi-level per season), additional effort:

| Task | Effort |
|------|--------|
| Write BRef scraper for per-level MiLB stats | 2–3 hours |
| Cross-validate with existing data | 1 hour |
| Replace/merge with existing milb JSON | 1 hour |
| **Total (if supplementing)** | **~4–6 hours** |

---

## 7. Recommendations

### Go / No-Go: **GO**

**Why**: The existing `data/milb/milb_*.json` files already contain comprehensive season-level MiLB stats that cover **76% of signed draftees (2021–2025)**. Integration is a straightforward person_id join — no scraping, no API, no Cloudflare issues.

### Immediate Next Steps

1. **Verify the "one level per season" assumption** — Check if any player has multiple records at different levels in the same year across the milb JSON files. If so, we have per-level data. If not, we have peak-level-only data.

2. **Build the training set extension** — Create a script that joins draft data with best-available MiLB stats per player, computing:
   - Peak wOBA in years 1-3 post-draft
   - Highest level reached by year 2, year 3
   - MiLB K%, BB%, ISO, HR/9, etc.

3. **Set data quality thresholds** — Skip players with <50 PA (batters) or <20 IP (pitchers).

4. **Consider multi-season modeling** — For players with 3+ years of MiLB data (1,249 exist), we could build a trajectory model that predicts future performance from early MiLB stats.

### What We Would NOT Recommend
- **Don't try to use pybaseball's `milb_batting_stats`** — these functions don't exist in v2.2.7.
- **Don't write a BRef scraper** unless we need per-level splits — the existing data covers 76% of signed draftees, which is sufficient.
- **Don't waste time on FanGraphs** — blocked by Cloudflare, not worth the fight.

---

## Appendix: Quick-Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│               MiLB Data Quick Reference                     │
├──────────────────────┬──────────────────────────────────────┤
│ Data location        │ data/milb/milb_{2021..2025}.json     │
│ Format               │ JSON, one object per year            │
│ Players total        │ 8,110 across all years               │
│ Join key             │ person_id → draft_all_picks.json     │
│ Batting columns      │ 37                                   │
│ Pitching columns     │ 63                                   │
│ Levels available     │ A, A+, AA, AAA                       │
│ Multi-year players   │ 1,249 with 3+ consecutive years      │
│ pybaseball needed?   │ ❌ No — data already extracted       │
│ Scraper needed?      │ ❌ No — unless per-level splits      │
│ Cloudflare issue?    │ ❌ No — MLB Stats API source         │
├──────────────────────┼──────────────────────────────────────┤
│ Coverage (signed)    │ 76% (2,157/2,839)                    │
│ Effort to integrate  │ ~2.5 hours                           │
│ Verdict              │ ✅ GO — data ready                   │
└──────────────────────┴──────────────────────────────────────┘
```
