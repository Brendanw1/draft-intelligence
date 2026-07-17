# Statistical and Strategic Evaluation
## MLB Draft Model — Formula Audit, Domain Validation, and Next Steps

---

## 1. Formula Audit — Every Calculation We're Making

### 1.1 FanGraphs College Stats (Tier 1 Features)

These come pre-computed from FanGraphs' API. We're not calculating them ourselves, but we need to know what they mean and whether they're appropriate for this use case.

| Stat | What It Measures | Baseball Context Check | Concern |
|------|------------------|----------------------|---------|
| **wOBA** | Weighted On-Base Average — total offensive value per PA | Standard. MLB weights used for college — acceptable approximation since run values don't vary wildly by level | Low |
| **wRC+** | Weighted Runs Created Plus — park/league-adjusted, 100=average | FanGraphs normalizes within the college D1 population. **What's the baseline population?** All D1 vs just one conference? If it's all D1, a .400 wOBA at Vanderbilt vs .400 wOBA at Morehead State get the same wRC+ — this loses conference quality signal | **Medium — missing conference adjustment** |
| **BABIP** | Batting Average on Balls in Play | Standard formula. College BABIP can be noisy due to variable defensive quality across D1 | Low for within-population ranking |
| **BB%** | Walk rate | Standard | Low |
| **K%** | Strikeout rate | Standard | Low |
| **ISO** | Isolated Power (SLG - AVG) | Standard | Low |
| **Spd** | Speed Score (FanGraphs multi-factor) | Proprietary FanGraphs formula. Uses SB, CS, triples, etc. Acceptable | Low |
| **FIP** (pitchers) | Fielding Independent Pitching | Standard: (13*HR + 3*(BB+HBP) - 2*SO) / IP + constant | Low — computed correctly |

**Note**: The FG API returns `xMLBAMID` which maps to MLB's `person.id`. We confirmed 99.9% match rate against our draft data. This is the cleanest join key possible.

### 1.2 MiLB Stats (Tier 2 Features)

These we compute ourselves from raw game feed data. Every formula needs verification.

**Ingested from game feed (raw JSON):**
```python
# From the game feed boxscore → seasonStats
batting: { gamesPlayed, atBats, hits, homeRuns, doubles, triples, baseOnBalls,
           strikeOuts, avg (string), obp (string), slg (string), ops (string),
           babip (string), totalBases, stolenBases, caughtStealing, ... }

pitching: { gamesPitched, gamesStarted, inningsPitched (string), hits, 
            earnedRuns, homeRuns, baseOnBalls, strikeOuts, battersFaced,
            era (string), whip (string), ... }
```

**Our parsing layer:**

| Formula | Code | Correct? |
|---------|------|----------|
| **AVG parse** | `parse_pct(".248") → 0.248` | ✅ Correct |
| **ERA parse** | `parse_pct("3.27") → 3.27` | ✅ Correct — `>1.0` check distinguishes ERA from AVG |
| **IP parse** | `"109.2" → 109 + 2/3 = 109.667` | ✅ Correct — MLB convention: .1=.333, .2=.667 |
| **K/9** | `SO / IP * 9` | ✅ Standard formula |
| **BB/9** | `BB / IP * 9` | ✅ Standard formula |
| **HR/9** | `HR / IP * 9` | ✅ Standard formula |
| **K-BB%** | `(SO - BB) / BF` | ✅ Standard formula |
| **K/BB** | `SO / BB` | ✅ Standard formula |
| **FIP** | `(13*HR + 3*(BB+HBP) - 2*SO) / IP + 3.10` | ✅ Standard formula. Constant of 3.10 is league-average FIP — we use a fixed constant. Minor impact since we're comparing within-population |
| **BB%** | `BB / PA` (from batting), `BB / BF` (from pitching) | ✅ Correct context-dependent |
| **K%** | `SO / PA` (batting), `SO / BF` (pitching) | ✅ Correct |

**Issue found**: The `pitchesThrown` field is only present for 421/867 players (48%). This is because the game feed only tracks pitch counts for pitchers who actually threw pitches in that specific game, not for all players on the roster. We don't use this in our models, but it's worth noting.

### 1.3 FIP Constant Concern

**The problem**: FIP includes a league-average constant (`FIP = (13*HR + 3*(BB+HBP) - 2*SO) / IP + C`). The constant C varies by level:
- MLB: ~3.10
- AAA: ~4.10-4.30 (higher run environment)
- AA: ~3.80-4.00
- A+: ~3.70-3.90

**Impact**: We use C=3.10 for all levels. This means FIP values are systematically offset — a pitcher with a 4.50 FIP in AAA might actually be league-average, but our calculation shows them as 1.40 runs above average. 

**Mitigation**: This doesn't affect the model much because XGBoost learns relative differences, not absolute values. A pitcher who's 0.5 FIP better than their level peers will be 0.5 better regardless of the constant. But absolute FIP numbers in our reporting are misleading.

**Fix**: Add per-level FIP constants or at minimum note this in documentation.

### 1.4 Innings Pitched Edge Cases

The game feed stores IP as a string like "109.2" (109 full innings + 2/3). Our parsing:
```python
parts = ip_str.split(".")
whole = int(parts[0])
frac = int(parts[1]) if len(parts) > 1 else 0
ip = whole + frac / 3
```

**Edge cases tested:**
- "0.0" → 0.0 ✅
- "1.0" → 1.0 ✅
- "109.2" → 109.667 ✅
- "0.1" → 0.333 ✅ (one out)
- "0.2" → 0.667 ✅ (two outs)
- Missing/null → None ✅ (our code handles this)

No issues found.

---

## 2. Baseball Domain Knowledge — Do The Results Make Sense?

### 2.1 Batting Stat Distributions by Level

```
Level   AVG    OBP    SLG    OPS    BB%    K%
AAA    .253   .340   .416   .756   10.6%  23.5%
AA     .237   .322   .376   .698    9.8%  25.1%
A+     .230   .317   .356   .673    9.9%  26.1%
A      .233   .332   .348   .680   11.2%  25.5%
```

**Domain check:**
- ✅ AVG decreases from AAA to AA to A+ — expected (harder competition at higher levels)
- ✅ K% increases from AAA to A+ — expected (less experienced hitters at lower levels)
- ✅ SLG drops from AAA to A+ — expected (less power at lower levels)
- ⚠️ A-ball having slightly higher AVG than A+ (.233 vs .230) — plausible (less advanced pitching at A-ball can offset the hitting quality)
- ⚠️ A-ball having highest BB% (11.2%) — contrarian but plausible (A-ball pitchers have less command)

**Verdict**: Distributions pass the smell test. The level gradients are in the right direction.

### 2.2 Pitching Stat Distributions by Level

```
Level   ERA    WHIP
AAA    4.93   1.48
AA     4.63   1.43
A+     4.76   1.44
A      4.85   1.49
```

**Domain check:**
- ⚠️ AAA ERA (4.93) is higher than AA (4.63) — is this normal?
  - **Yes**, this is well-documented. AAA is hitter-friendly because:
    1. AAA uses the same baseball as MLB (since 2021 "deaden the ball" changes at lower levels, lower-level MiLB balls are different)
    2. AAA has more "AAAA" veterans who can hit but can't make a 26-man roster
    3. Many AAA parks are in hitter-friendly environments (PCL)
    4. AA is actually where the best prospects are — the true future MLB talent on the mound
  - This is widely known in baseball analytics: **AA is often a better predictor of MLB success than AAA**.

### 2.3 Model Feature Importance — Baseball Sense Check

**Tier 1 (Draft Position — FG stats):**

Hitters top features: Age, wRC, PA, BB, HR, BB%, BB/K
- ✅ **Plate discipline dominates** — BB% and BB/K being top features matches MLB scouting wisdom. Teams value college hitters who control the zone.
- ✅ **Age** being #1 makes sense — younger draftees = more projection
- ✅ **Power counts** (HR) being important tracks
- ❓ **wRC over wOBA** — wRC is counting, wOBA is rate. Having the counting stat higher is slightly suspicious but could just mean teams value production volume

Pitchers top features: SO, Age, GS, K-BB%, K%, FIP, WHIP
- ✅ **Strikeouts dominate** — SO and K-BB% being top features is absolutely correct. Teams chase strikeout stuff.
- ✅ **Age important** for pitchers too
- ✅ **FIP and WHIP** being secondary is correct — they capture command and contact quality

**Tier 2 (Reached MLB — Draft + FG + MiLB):**

Hitters top features: draft_pick, first_milb_ops, weight, draft_bonus, draft_round
- ✅ **draft_pick #1** — obviously. Where you're drafted is the single best predictor of whether you reach MLB
- ✅ **first_milb_ops** — immediate MiLB performance matters. .800 OPS in first MiLB stint → good sign
- ✅ **weight** — bigger hitters tend to project better (power potential)
- ❓ **weight > college stats** — slightly suspicious. This might be a small-sample artifact

Pitchers top features: draft_pick, draft_round, fg_SHO (shutouts), draft_bonus
- ⚠️ **fg_SHO as #3** — Shutouts are a noisy, rare event for college pitchers. This is concerning — it suggests the model is latching onto a sparse signal. A pitcher who threw a shutout in college might have faced weak competition. This needs investigation.
- ✅ **draft_pick/round** — correct
- ✅ **fg_IP, fg_GS** — workload matters for pitcher development

### 2.4 MLB Rate — Is 12.3% Reasonable?

Of the 1,549 players in the Tier 2 training set:
- 190 reached MLB (12.3%)
- This is across 2021-2024 draftees

Is this realistic?

**Checking**: In a typical draft, ~600 players are selected. Of those 600, roughly:
- R1-2 (60 players): ~80% reach MLB ≈ 48
- R3-5 (90 players): ~30% reach MLB ≈ 27
- R6-10 (150 players): ~15% reach MLB ≈ 22
- R11-20 (300 players): ~5% reach MLB ≈ 15
- Total from one draft class: ~112 / 600 = **18.7%**

But our training set excludes HS draftees (who have lower MLB rates) and only includes college draftees who were matched to MiLB data. College draftees have higher MLB rates than HS draftees.

So 12.3% across all rounds from 2021-2024 classes seems **slightly low but reasonable** — the data includes recent draftees (2024) who haven't had time to reach MLB yet. If we only look at 2021-2022 (who've had 4-5 years), the rate would be higher.

### 2.5 Model Performance — What's Real vs Noise

**Tier 2 Hitters:**
- Test ROC-AUC: 0.93 — sounds great, but...
- Test Precision: 0.267 — only 27% of "will reach MLB" predictions were correct
- This means: if the model says "this hitter will reach MLB", there's a 73% chance they won't

Is this useful? Yes — because:
1. **Recall is 0.80** — the model catches 80% of future MLB players
2. The alternative (scouting) also has high false-positive rates
3. The model provides a **ranking**, not a definitive label. Rank-ordering prospects by MLB probability is valuable even if the absolute probabilities are noisy
4. For draft decisions, you want the model to say: "this late-round guy has a similar probability to those early-round picks" — that's actionable

**Tier 2 Pitchers:**
- Test set has only 1 positive label (reached MLB) — statistically meaningless for evaluation
- CV ROC-AUC 0.769 ± 0.073 is the most reliable metric
- This is modest but meaningful

---

## 3. Data Leakage Audit — Are We Cheating?

### 3.1 Temporal Leaks

| Potential Leak | Status | Explanation |
|----------------|--------|-------------|
| Tier 1: Future draft data in training | ✅ Clean | Train: 2021-2025, Test: 2026. No 2026 data seen during training |
| Tier 1: 2026 FG stats used in training | ✅ Clean | We pulled 2026 FG data but it's only used for projections, not training |
| Tier 2: MiLB stats from same year as draft | ⚠️ **Needs audit** | For a 2022 draftee, their "first_milb" stats might come from their post-draft 2022 season. The model sees: "drafted in 2022 → first_milb_ops in 2022 → reached MLB by 2026". The draft year IS the same as the first MiLB year for summer draftees. This is correct — you know their first MiLB stats within 2-3 months of drafting them. But it's not a "pre-draft" feature. |
| Tier 2: 2026 MiLB data (current season) | ✅ Not used | We only scraped through 2025 for training. 2026 is out of sample. |
| FG season selection: using draft-year stats | ✅ **But needs thought** | If a player is drafted in 2022 but their last college season was 2022 (spring), we use 2022 stats. This is correct — those stats existed at draft time. But if a player was drafted as a sophomore and their 2022 stats are partial (due to injury, etc.), using 2021 instead might be better. We use "latest season ≤ draft year" which is the safest approach. |

### 3.2 Selection Bias

| Bias | Impact | Mitigation |
|------|--------|------------|
| **Only players with xMLBAMID in FG data** | We can only analyze players who appear in both FG and draft data. Players who played at non-D1 schools or who never got an MLBAM ID are excluded. | Accept — these are fringe prospects anyway |
| **Only players matched to MiLB rosters** | The 1,549 players in Tier 2 are the ones who actually signed and were assigned to an affiliate. Players who chose not to sign, or who were immediately released, are excluded. | This means our Tier 2 model only applies to players who actually started a MiLB career. For 2026 projections, we're assuming the player will sign. |
| **Survivorship bias in MiLB data** | To appear in our MiLB data, a player must have been good enough to stay on a roster long enough to appear in a late-season game. Players released mid-season are undercounted. | This inflates the apparent MLB rate slightly. Mitigation: note that our model's baseline assumes a signed/rostered player. |

### 3.3 Target Leakage Check

**Tier 2 target = "Reached MLB" — is it measured correctly?**

We check via `people/{id}` API → `mlbDebutDate` field. This returns the DATE of first MLB appearance.

**False negatives risk**: A player who reached MLB but was never assigned an `mlbDebutDate` in the API. We tested this: for known MLB players (Travis Bazzana, Charlie Condon, JJ Wetherholt), the API correctly returns a debut date. For MiLB-only players, it correctly returns null.

**False positives risk**: The API could return a date for a player who played in spring training but not the regular season. MLB considers spring training games as MLB appearances for debut tracking. This could slightly inflate our MLB rate. Acceptable.

---

## 4. Model Soundness Assessment

### 4.1 Tier 1 (Draft Position Regression)

| Check | Result | Notes |
|------|--------|-------|
| Overfitting? | ✅ No | CV R² ≈ Test R² (0.258 vs 0.279 hitters, 0.283 vs 0.300 pitchers) |
| Temporal stability? | ✅ Yes | Performance holds across train (2021-2025) and test (2026) |
| Feature count vs sample | ✅ Good | 34 features, 998 hitters (29:1 ratio), 33 features, 1,368 pitchers (41:1 ratio) |
| Residual pattern? | ⚠️ Not checked | Should check: does the model underpredict for SEC/ACC players? Overpredict for small-conference players? |
| Pick distribution? | ✅ Reasonable | Range 30-648, median 424 — covers the actual draft range (R1-20) |

**Verdict**: Tier 1 is sound. The 0.28-0.30 R² is modest but expected from college stats alone. Adding conference would likely boost it to 0.35-0.40.

### 4.2 Tier 2 (MLB Probability Classification)

| Check | Result | Notes |
|------|--------|-------|
| Class imbalance handled? | ✅ Yes | `scale_pos_weight` = ~5:1 ratio |
| Temporal stability? | ⚠️ Moderate | CV=0.643 vs Val=0.770 for hitters — 0.13 gap suggests some year-specific patterns |
| Calibration? | ⚠️ Not checked | Need to bin predictions and compare to actual rates |
| Feature importance stable? | ⚠️ Not checked | Need to verify features are consistent across CV folds |
| Pitcher test set tiny | ⚠️ Limited | Only 1 positive in test set — pitcher evaluation unreliable |

**Verdict**: Tier 2 is useful but needs calibration checks. The binary model works better for hitters than pitchers (more data). The precision (0.27) is low but expected for this problem.

---

## 5. Backtesting and Verification Plan

### 5.1 Historical Backtest (Most Important)

**Method**: Run the model as if it's draft day 2021, using only data available at that time.

```
For each draft year Y in [2021, 2022, 2023, 2024]:
  1. Train model on all data from years < Y (strictly pre-Y)  
  2. For each player drafted in year Y:
     a. Feed their college stats (from season Y or earlier) into the model
     b. Record predicted draft position
     c. Record predicted MLB probability
  3. Compare predictions to actual outcomes:
     a. Predicted pick vs actual pick → MAE, rank correlation
     b. Predicted MLB prob vs actual (by 2026) → calibration curve
```

This is the gold standard for validation. We can do this right now.

### 5.2 Prospect Rank Overlap

Compare our top-100 projected prospects against:
- **MLB Pipeline Top 200 Draft Prospects** (if available)
- **FanGraphs College Draft Rankings**
- **Baseball America Draft Top 500**
- **D1Baseball Top 100**

A high Spearman correlation (ρ > 0.5) with any of these would validate our model's ranking.

### 5.3 Calibration Curve

For the Tier 2 binary model:
1. Sort all players by predicted MLB probability
2. Bin them into deciles (0-10%, 10-20%, ..., 90-100%)
3. For each bin, calculate the actual MLB rate
4. Plot: predicted vs actual
5. A diagonal line = perfect calibration

**Expected result**: The model will be overconfident at the high end (predicting 80% but only 50% actually reach MLB). This is normal for imbalanced classification.

### 5.4 Cross-Source Validation (Step 2 — FG vs TrackMan)

For players in both datasets:
1. Find players with FG college stats AND portal-scout TrackMan stats
2. Compare matching metrics side by side:

| Metric | Expected Match | If different, why? |
|--------|---------------|-------------------|
| AVG | Exact | Different rounding |
| OBP | Exact | Different HBP/SF inclusion |
| SLG | Exact | Different TB calculation |
| **wOBA** | **Will differ** | FG uses actual outcomes, TM uses expected (xwOBA) |
| **BABIP** | **May differ** | Different filter for "balls in play" |
| BB% | Exact | Same formula |
| K% | Exact | Same formula |

### 5.5 Year-Over-Year Stability

For players with multiple college seasons:
1. Take each player's FG stats from consecutive years (2023 and 2024)
2. Compute year-to-year correlation for each stat
3. Expected: r > 0.6 for stable metrics (BB%, K%), r > 0.4 for noisy metrics (BABIP, HR)

Low correlations would suggest our single-season stats are unreliable.

---

## 6. Next Steps — Implementation Roadmap

### Phase 1: Validate (Now)

| Task | Time | Deliverable |
|------|------|-------------|
| Historical backtest 2021-2024 | 1 hr | MAE per year, rank correlation |
| Calibration curve for Tier 2 | 30 min | Predicted vs actual plot |
| FG vs TrackMan cross-validation | 30 min | Side-by-side comparison report |
| Year-over-year stability | 30 min | Year-to-year correlations per stat |

### Phase 2: Enhance (This Week)

| Task | Time | Impact |
|------|------|--------|
| Add conference as a feature | 30 min | Expected +0.05 R² for Tier 1 |
| Add class year (FR/SO/JR/SR) as feature | 30 min | Expected +0.03 R² for Tier 1 |
| Tier 2 ordinal model (peak level) | 1 hr | More nuanced outcome than binary |
| Per-level FIP constants | 15 min | Correct absolute FIP values |
| Conference-level stat adjustments | 1 hr | wRC+ by conference context |

### Phase 3: Productize (Next Week)

| Task | Time | Deliverable |
|------|------|-------------|
| R2 bucket setup (you) | 15 min | Data in cloud, not local |
| S3 sync scripts | 30 min | Automated upload |
| API service for projections | 2 hr | Queryable endpoint |
| React dashboard | 3-4 hr | Visual storytelling (see Section 7) |

---

## 7. Visual Storytelling — Advanced React Site

The goal: build a site that makes complex data immediately useful for scouts, coaches, and front office. **Not a dashboard. A story.**

### Core Design Principle

> Every page should answer one question a baseball person would actually ask.

Not "here's a table of stats" but "who should I draft in round 5?"

### Page Concepts

#### 7.1 The Draft Board — "Who's Left?"

A living draft board that updates as picks are made. The key visual:

```
┌──────────────────────────────────────────────────────┐
│  2026 MLB DRAFT BOARD                  R1  │  R2  │ R3 │
├──────────────────────────────────────────────────────┤
│ Grayden Harris   │ Tomas Valincius │ Wes Mendes   ...│
│ [P94 / 68% MLB]  │ [P57 / 56% MLB] │ [P56/53%MLB] │
│ ████████████░░░░ │ ██████████░░░░░ │ ████████░░░░ │
│ USM - RHP        │ MS State - RHP  │ FSU - RHP    │
├──────────────────────────────────────────────────────┤
│ BEST AVAILABLE            │ TEAM NEEDS               │
│ ┌──────────────────┐      │ ┌──────────────────┐     │
│ │ Rk Player    Val │      │ │ SP: 3 needed     │     │
│ │ 1. Harris   68%  │      │ │ C:  1 needed     │     │
│ │ 2. Knowles  69%  │      │ │ OF: 2 needed     │     │
│ │ 3. Melton   80%  │      │ └──────────────────┘     │
│ └──────────────────┘      └──────────────────┘     │
└──────────────────────────────────────────────────────┘
```

**Story**: "Here's who should be drafted next, and why."

#### 7.2 Player Card — "Should We Draft This Guy?"

Single-player deep dive. The key insight is **why** the model thinks what it thinks.

```
┌─────────────────────────────────────────────────────────┐
│  Payton Knowles          Seattle U            OF │ R │ L│
├─────────────────────────────────────────────────────────┤
│  Projected: P197 (R7)     MLB Probability: 69%          │
│                                                         │
│  ╔══════════════════════════════════════════╗            │
│  ║     PROFILE RADAR                        ║            │
│  ║          Power                          ║            │
│  ║         ██████                          ║            │
│  ║       ██████████                        ║            │
│  ║      ████████████  Contact              ║            │
│  ║      ████████████  ████████             ║            │
│  ║       ██████████  ██████████            ║            │
│  ║         ██████   ████████████           ║            │
│  ║    Speed          ██████████            ║            │
│  ║    █████████       ████████  Discipline ║            │
│  ╚══════════════════════════════════════════╝            │
│                                                         │
│  WHAT DRIVES THIS PROJECTION:                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │ ▲ College wRC+ (228) — Top 1% of D1               │ │
│  │ ▲ BB/K (0.96) — Elite plate discipline             │ │
│  │ ▼ Competition: Seattle U (WAC) — No power conf     │ │
│  │ ▼ Age (23) — Older for a draft prospect            │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  COMPARABLE SUCCESS STORIES:                            │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Similar profile → reached MLB in 38% of cases     │ │
│  │ Most similar: [Player A] (R5, reached MLB 2024)  │ │
│  │              [Player B] (R8, did not reach MLB)  │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Story**: "Here's what the model sees, what it likes, and what it worries about."

#### 7.3 The Value Map — "Where's the Value?"

Scatter plot showing the entire draft-eligible population:

```
MLB Prob  ↑                                            ╭──╮
  80%    │                                       ★★★  │High│
         │                                    ★★      │Val │
  60%    │                               ★★★         ╰──╯
         │                          ★★
  40%    │                    ★★★★
         │              ★★★★★★              ╭──╮
  20%    │        ★★★★★★★                   │Ov-│
         │  ★★★★★★★★★★                       │er-│
   0%    ├──────────────────────────────────│val│──
         └────┴────┴────┴────┴────┴────┴────╰──╯
         R1    R3    R5    R7    R9    R11   R13
               Projected Draft Pick →
```

The "Value Zone" (upper-right quadrant) shows players projected for late rounds with high MLB probability — these are the steals.

**Story**: "Here's where you find draft bargains."

#### 7.4 Conference/Position Heatmap — "Where Does Talent Come From?"

```
        SP    RP    C     1B    2B    SS    3B    OF
SEC     ███   ██    ██    ██    ██    ███   ██    ███
ACC     ██    █     █     █     █     ██    █     ██
B1G     █     █     █     █     █     █     █     █
P12     ██    █     █     ██    █     ██    █     ██
WCC     █     █     █     █     █     █     █     █
AAC     █     █     █     █     █     █     █     █
```

Cell value = average MLB probability for that conference+position combo.

**Story**: "Where to find the best talent at each position."

#### 7.5 Draft Simulator — "What If?"

Let users make picks and see how the board changes.

```
ROUND 5, PICK 142 — YOU'RE ON THE CLOCK

Best available: Grayden Harris (68% MLB, P94 projected)

If you draft Harris:
  → Other teams in R5 will target: Valincius (56%), Mendes (53%)
  → Your next pick (R7, P198): Melton (80% MLB) may still be available
  → Position need: You already drafted 2 SPs. Harris is another SP.

Other options:
  ┌─────────────────────────────────────────────────────┐
  │ Player                │ Pick │ MLB% │ Position need │
  ├─────────────────────────────────────────────────────┤
  │ Grayden Harris (USM)  │ P94  │ 68%  │ ⚠️ 3rd SP    │
  │ Payton Knowles (SeaU) │ P197 │ 69%  │ ✅ Need OF   │
  │ Dylan Melton (UMBC)   │ P306 │ 80%  │ ✅ Need OF   │
  └─────────────────────────────────────────────────────┘
```

**Story**: "Here's how your decisions change your draft outcome."

### Technical Architecture

```
Frontend: Next.js 14 (App Router) + Tailwind + Recharts/D3
  ├── /draft-board        → Live draft board (Server-Side Rendered)
  ├── /players/[id]       → Player card (SSR + ISR, revalidate daily)
  ├── /value-map          → Scatter plot (Client-side interactive)
  ├── /heatmap            → Conference/position grid (Client-side)
  ├── /simulator          → Draft simulator (Client-side, stateful)
  └── /api/               → API routes
      ├── /projections    → Query projections (SQLite via better-sqlite3)
      ├── /players/:id    → Single player with comparables
      └── /simulate       → "What if" scenario engine

Data Layer: SQLite (via better-sqlite3) or DuckDB
  Projections in data/training/projections_2026.json
  Tier 2 training set for comparables lookup
  FG college stats for player profiles

Deployment: Vercel (Same as portal-scout)
  Separate project from portal-scout
  Different subdomain (e.g., draft.portalscout.app)
```

---

## 8. Summary — What Would I Fix?

### Critical (Fixes Before Trusting Results)

1. **fg_SHO as #3 pitcher feature** — Shutouts are rare events. Investigate whether the model is latching onto a sparse signal. Feature ablation test: remove fg_SHO, retrain, check if R² drops.

2. **Calibration curve** — Generate it. If the model says 70% but reality is 30%, we need to recalibrate (Platt scaling or isotonic regression).

3. **2024 MiLB data inconsistency** — The 2024 scrape captured only 867 players (drafted-only) while other years had 3,400+. This means the 2024 tier2 training data is less complete.

### Important (Should Address)

4. **Conference signal** — Missing entirely from both models. This is the #1 missing feature.

5. **Class year (FR/SO/JR/SR)** — A .400 OBP as a freshman is way more impressive than as a senior.

6. **Per-level FIP constants** — Our FIP values are systematically wrong in absolute terms (correct in relative terms).

### Nice to Have (Enhancements)

7. **Ordinal Tier 2 model** — Predict peak level (MLB/AAA/AA/A+/A) instead of just binary

8. **HS draftee coverage** — Our model only covers college draftees. HS draftees need a completely different approach (no FG college stats).

9. **Pitch-level MiLB data** — The game feed also contains pitch-by-pitch data which could give us Stuff+ approximations for MiLB pitchers.
