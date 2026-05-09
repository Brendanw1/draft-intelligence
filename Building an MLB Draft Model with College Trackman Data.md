# Building an MLB Draft Model with College Trackman Data

## Executive Summary

An MLB draft model powered by college Trackman data can be built by integrating pitch-level and batted-ball biomechanics from your 2024–2026 college datasets with publicly available MLB Statcast data. The core insight driving this approach is that raw Trackman physics — velocity, induced vertical break (IVB), horizontal break, spin rate, VAA, release height, extension — are **level-agnostic**: they do not systematically change when a player advances to the pro level. What does change is the quality of competition, park environments, and the metal-vs-wood bat effect. A good model exploits this asymmetry by leaning on the physical metrics that translate and correcting for the contextual factors that do not.

***

## Part 1: What You Have — Understanding the Trackman Dataset

### What Trackman Captures

At the college level, Trackman records ~63 data points per pitch. The most draft-relevant signals fall into two categories:[^1]

**Pitching Metrics**
- Release speed, extension, release height (RelZ), release side
- Spin rate (RPM), spin axis, spin efficiency
- Induced Vertical Break (IVB), Horizontal Break (HB)
- Vertical Approach Angle (VAA), Horizontal Approach Angle
- Plate height (PlaLocHei), plate side (PlaLocSide)
- Pitch result (called strike, swinging strike, ball in play, etc.)

**Hitting/Batted Ball Metrics**
- Exit velocity (EV), launch angle (LA), hit direction
- Contact depth, contact side/height
- Estimated distance

With 2024, 2025, and 2026 data, you have multi-year longitudinal coverage on players who may appear in all three seasons, enabling trend analysis (velocity development, spin changes, EV progression) — a meaningful edge over single-season snapshots.[^2]

### The Metal Bat Problem

The most important calibration issue for hitters: college players swing aluminum bats. Exit velocities in college average roughly 87–95 mph in game settings, but when hitters transition to wood bats in the minors, EV drops approximately 2.5–3.2 mph on average. This is not a trivial correction — the difference between average (~89 mph) and elite (95+ mph) MLB exit velocity is on the order of 5–6 mph. So a college hitter averaging 94 mph with aluminum might project to 91–92 mph with wood, which is above-average pro territory. Build this adjustment explicitly into your feature pipeline for all EV-based features. The Cape Cod League, which uses wood bats, provides the cleanest college-level exit velocity data that translates most directly to pro environments.[^3][^4][^5]

***

## Part 2: Data Normalization — The Critical Pipeline Steps

Before modeling, the raw Trackman data must pass through three normalization layers. Skipping any one of them risks major bias in your model.

### Step 1: Park Factors

College baseball park factors vary wildly — far more than MLB parks. The most pitcher-friendly D1 venue grades out at a run factor of 75 (Evans Diamond, Cal-Berkeley), while the most hitter-friendly (Allen Field, Morehead State) sits at 139. By contrast, the entire MLB range runs from 92 (Seattle) to 114 (Colorado). More than half of all D1 venues have a bigger effect on home runs — positive or negative — than any MLB ballpark.[^6]

If you don't park-adjust, a player posting a .900 OPS at Morehead State looks far worse in context than one posting an .820 OPS at Evans Diamond. College Splits (the data provider behind FanGraphs' new college data section launched in March 2025) computes multi-year, regressed D1 park factors and shows OPS swings of nearly 100 points even at top-tier programs. Use their framework or build your own:[^7][^6]

- Use **multi-year factors** (single-year park factors for 25–30 game samples are extremely noisy)[^6]
- Apply **component park factors** separately for runs, HR, singles, walks, and K (the walk factor alone ranges from 80 to 135 across D1)[^6]
- Account for **home game percentage** — strong warm-climate programs (like Arizona) play 57% of games at home, distorting raw park factor estimates[^6]
- For pitching, park adjustments matter less since VAA, velo, break, and spin don't change with park environment — but ERA and strikeout rate do

**For Trackman-specific metrics (EV, spin rate, VAA):** These are largely park-neutral and require no park adjustment. Apply park adjustments only to outcome-based metrics (slash lines, K%, BB%).

### Step 2: Strength of Schedule / Conference Adjustments

A pitcher posting 0.87 ERA in the Big East faces fundamentally different competition than one posting the same ERA in the SEC. FanGraphs' new college data (launched March 2025) provides wRC+, ERA-, and FIP- that are **conference-adjusted but not park-adjusted** — giving you a ready-made baseline. You can augment this with a custom SOS layer:[^7]

- Compute each team's **opponent quality** using a weighted blend of opponents' park-adjusted ERA- and OPS+[^8]
- Consider a **Bayesian shrinkage** approach: regress player-level metrics toward conference means, with the degree of shrinkage inversely proportional to sample size
- Use **Baseball Prospectus's historical approach** of applying a multiplier for competition level — their research showed incorporating park factors and SOS raised correlation between college and minor-league performance from ~0.26 R to ~0.37 R[^9]

### Step 3: Sample Size Weighting

College seasons are short (~55–60 games). A hitter might have 35 batted ball events (BBE) against Trackman-equipped opponents. Small samples require:[^3]
- Regressing Trackman metrics to position or conference means
- Weighting multi-year data more heavily — a 2024+2025+2026 combined profile is far more reliable than 2026 alone
- Flagging players where Trackman coverage is sparse (not all opponents have Trackman units) — one 2020 draft analysis found several games with no Trackman data even for top-5 picks[^3]

***

## Part 3: Feature Engineering — What Actually Translates

### For Pitchers

The key insight is that **stuff-based metrics are level-agnostic while outcome-based metrics are not**. VAA doesn't change when a pitcher advances to the pros; wOBA against and wRC+ do. Build two feature categories accordingly:[^10]

**Physical/Translatable Features (high weight)**
| Feature | Why It Matters | Notes |
|---------|----------------|-------|
| Release speed | Foundation of any arsenal | MLB avg FB ~93.5 mph; college top prospects ~94–98 mph[^11] |
| Induced Vertical Break (IVB) | Rides fastball; swing-and-miss generator | Chase Burns (2024 #2 pick): 20.2" IVB; elite threshold ~18+"[^12] |
| Horizontal Break (HB) | Arm-side or glove-side movement | Component of pitch design; level-agnostic[^13] |
| Vertical Approach Angle (VAA) | Flat VAA in upper zone → massive whiff rate | VAA is location-dependent; normalize to zone-specific averages[^10] |
| Release height (RelZ) | Major driver of VAA; lower arm slot = flatter perceived trajectory | College avg ~5.5–6.5 ft; level-agnostic[^10] |
| Extension | Effective distance to plate; related to perceived velo | Level-agnostic[^14] |
| Spin rate | Pitch-specific predictive value | College avg FB: 2148 rpm; MLB avg: 2256 rpm[^15] |
| Spin axis | Determines movement direction; affects pitch design potential | Level-agnostic[^2] |
| Arsenal depth | Number of distinct, effective pitch shapes | Pitch diversity increases ceiling[^16] |

**Outcome-Based Features (lower weight, use with caution)**
- Whiff rate, CSW% (called strike + whiff %): informative but context-dependent
- K% and BB%: adjust for conference and park using ERA-/FIP- values
- ERA: park/SOS adjust heavily, or use FIP-/xFIP- instead

**Derived Features to Build**
- **Stuff+ score** (college-level): Train an XGBoost model on your Trackman data, predicting run value per pitch from velo, IVB, HB, spin rate, VAA, and release point. This collapses the pitch arsenal into a single number — a college-calibrated equivalent of FanGraphs' Stuff+.[^17][^18]
- **Movement diversity score**: Spread of pitch shapes across the movement plot (how much separation exists between pitches)
- **VAA constant**: Use a regression model (release height, IVB, velo) to predict VAA at a standardized location, enabling fair comparison across pitchers regardless of zone tendencies[^10]

### For Hitters

**Physical/Translatable Features (high weight)**
| Feature | Notes |
|---------|-------|
| Exit velocity (EV) | Apply the 2.5–3.2 mph wood bat discount[^3]; percentile rank within college class |
| EV percentile (90th) | Max EV more predictive than mean EV for power projection |
| Launch angle distribution | Tight LD/FB distribution (0–25°) is favorable; normalize for park HR factor[^3] |
| xBA / xSLG from EV+LA | Construct expected stats from Trackman using speed-angle coefficients[^3] |
| Barrel rate (EV 98+, LA 26–30°) | College equivalent of MLB barrel definition, adjusted for wood bat EV discount |
| Hard-hit rate (EV > avg + SD) | Relative to class average, not absolute |

**Outcome-Based Features (park/conference adjust)**
- OPS, SLG, ISO: park-adjust and conference-adjust before using
- BB%/K%: FanGraphs college data provides conference-adjusted wRC+[^7]
- Pull% / Oppo%: Batted ball spray tendencies from Trackman hit direction data

**Derived Features**
- **xwOBA from Trackman**: Use EV + LA → wOBA value (similar to Statcast's xwOBA) built on your own BBE distribution
- **Contact quality score**: Combine hard-hit rate, barrel rate, and tight LA distribution into a composite

***

## Part 4: Incorporating Public MLB Statcast Data

### The Bridging Strategy

The fundamental challenge: your college Trackman data and MLB Statcast data cover different player populations. The bridge is **players who went from college in your dataset to affiliated minor league or MLB play**. This linkage gives you training labels.

**Step 1: Build a player ID crosswalk.** The Chadwick Bureau maintains a public people.csv (accessible via the `baseballr` R package or GitHub) linking names across MLB MLBAM IDs, FanGraphs IDs, Baseball Reference IDs, and college rosters. TrackMan's internal player database also links to official college roster IDs and MLB player IDs. Match your Trackman player entries (first name, last name, school, year) to the Chadwick crosswalk.[^19][^20]

**Step 2: Pull MLB/MiLB outcome data.** Use `pybaseball` (Python) to pull Statcast pitch-level and aggregated data from Baseball Savant for any players in your dataset who subsequently reached affiliated ball. The public Statcast dataset covers all MLB pitches with full Trackman metrics (velo, IVB, HB, spin rate, extension, release height, VAA at the plate). Note that **VAA is not publicly available in MLB Statcast CSV exports**, but you can compute it from the physics parameters (vz0, ay, release height) if needed.[^21][^22][^23][^10]

**Step 3: Define your target variable.** Options:
- **Draft slot** (round/pick number): Easy to compute but reflects organizational biases and market inefficiencies — not necessarily true talent
- **Minor league performance** at A/AA level (park-adjusted wRC+ for hitters, FIP- for pitchers): More signal, but requires 2+ year lag and suffers from promotion selection bias
- **Career WAR at ages 24–27**: The cleanest target but requires 5+ year lag; suitable for training on historical data (2018–2022 draftees with 2024 MLB data)
- **Probability of reaching MLB** (binary): Directly addresses organizational risk; class-imbalanced (roughly 80% of draft picks never reach the majors)[^24]

The recommended approach for your timeframe is a **two-stage model**: (1) Predict probability of reaching MLB or AA within 3 years (classification), and (2) among those who reach, predict performance level (regression on park-adjusted minor league stats). This avoids conflating "will he develop?" with "how good will he be?".

**Step 4: Historical training data.** You can augment 2024–2026 data with historical Trackman data for past draftees where available. BaseballCloud and D1Baseball have partnership data. For earlier draft classes (2019–2023), Baseball America's historical Trackman tables provide velocity, spin, IVB, HB, VAA, and release metrics that can be manually assembled into a training set.[^11][^25][^12]

### Using Statcast for Comparables (Comps)

Even without a perfect historical training set, Statcast enables a **nearest-neighbor comps approach**: for each college prospect, find the closest matching MLB or MiLB pitcher/hitter in Statcast based on pitch shape similarity. Baseball America publicly uses this methodology — for the 2024 draft, Chase Burns' Statcast fastball profile closely matched Dylan Cease's (velo, IVB, HB, spin, release height). A vector similarity search (cosine or Euclidean distance) across your feature space can systematically generate comps for every prospect in your dataset.[^12]

This has two uses:
1. **Comp quality as a feature**: The career outcome of the comp pool (avg WAR of 5 most similar MLB pitchers) becomes a predictive feature itself
2. **Interpretability**: Scouts and front offices respond better to "his pitch profiles look like Corbin Burnes at age 21" than to a black-box probability score

***

## Part 5: Model Architecture

### Recommended Stack

Given your background in R and Python and the structured tabular nature of the data, **XGBoost or LightGBM gradient boosting** is the strongest choice. The reasons:[^18][^26]

- Handles non-linear feature interactions (the interaction between release height and IVB on VAA is non-linear) without manual engineering
- Robust to missing data — critical when Trackman coverage is incomplete for some players[^2]
- Feature importance outputs translate naturally to scouting report language
- Outperforms random forests and neural networks on structured tabular data of this size

For the classification stage (reach-MLB probability), use XGBoost with `scale_pos_weight` to handle the ~80% negative class rate. For the regression stage (performance level), use LightGBM with early stopping on a validation split.[^24]

### Feature Set Summary

```
Pitcher features (per player-season):
  - Physical: avg_velo, max_velo, avg_ivb, avg_hb, spin_rate, 
              release_height, extension, vaa_constant, spin_axis
  - Derived: stuff_plus_college, movement_diversity, arsenal_depth
  - Adjusted outcomes: k_pct_adj, bb_pct_adj, fip_minus
  - Multi-year: velo_trend (2024→2026), spin_trend, bb_trend
  - Context: conference_tier, opponent_avg_stuff

Hitter features (per player-season):
  - Physical: avg_ev_wood_adj, p90_ev_wood_adj, barrel_rate_adj,
              hard_hit_pct, la_distribution_tightness
  - Derived: xba_trackman, xslg_trackman, xwoba_trackman
  - Adjusted outcomes: wrc_plus_conf_adj, bb_pct, k_pct
  - Multi-year: ev_trend, la_trend
  - Context: bat_handedness, primary_position, conference_tier
```

### Cross-Validation Strategy

Use **leave-one-draft-class-out cross-validation** (also called temporal CV): train on 2024 and 2025 draftees, validate on 2026 draft outcomes when available. This prevents data leakage from players who appear in multiple seasons and realistically simulates deployment — you're always predicting a future draft class from past data.

### Handling the Long Lag Problem

The biggest practical obstacle: players drafted in 2025 and 2026 won't have meaningful MLB outcomes until 2028–2030 at the earliest. To train the model now, you need historical draftees as ground truth. Pull Trackman data (from BaseballCloud, D1Baseball/TrackMan partnership, or manual Baseball America tables) for the 2019–2023 draft classes, where you now have 3–5 years of minor league outcomes to use as labels. Then apply the trained model to your 2025–2026 prospects.

***

## Part 6: Validation and Output Design

### Validation Metrics

- **For classification (reach-MLB probability):** AUC-ROC and precision-recall AUC (preferred over accuracy due to class imbalance)[^24]
- **For regression (performance level):** RMSE and Spearman rank correlation against actual outcomes; rank correlation is more useful operationally since you care about relative player ordering, not absolute WAR prediction
- **Residual analysis by conference:** Check whether the model systematically over- or undervalues players from a particular conference — a sign of incomplete SOS adjustment

### Output: What the Model Should Produce

The model should output, per player:
1. **Reach-MLB probability score** (0–1)
2. **Projected performance tier** (elite, above-average, average, org depth) if they reach
3. **Draft value composite** (blended score combining both outputs, scaled 0–100)
4. **Comparable MLB player** (nearest neighbor in feature space)
5. **Key strengths and flags** (SHAP value outputs translated to readable text: "above-average VAA, elite EV but below-average IVB")

SHAP (SHapley Additive Explanations) values are particularly useful here because they decompose each prediction into per-feature contributions — the closest public analog to how scouts think about "what's working for this guy".[^13][^27]

### Presentation Layer

Given your experience with R Shiny and Python Streamlit, the natural output is an interactive prospect dashboard:
- Player cards with movement plots (scatter of IVB vs HB for each pitch type)
- EV/LA scatter vs. class average
- Trend lines across 2024–2026 for key metrics
- Comp player profile with Statcast career arc
- Model score with SHAP waterfall chart

***

## Part 7: Known Limitations and Honest Caveats

| Challenge | Mitigation |
|-----------|------------|
| Trackman coverage gaps (not all opponents have units) | Flag sparse-data players; weight multi-year data more heavily[^4] |
| Metal bat EV discount is variable (not exactly 2.5–3.2 mph for every hitter) | Use range estimate; validate on Cape Cod hitters who have wood bat data[^3] |
| Small sample sizes in college (35–200 pitches with Trackman) | Bayesian priors; regress to conference/position mean; report uncertainty intervals |
| VAA not in public MLB Statcast exports | Reconstruct from physics params (vz0, ay, release height) or limit comp analysis to IVB/HB/velo[^10] |
| Selection bias in training labels | Players drafted in Round 1 get better development; non-draftees disappear from the outcome data entirely[^28] |
| College Trackman data accuracy | College data is "more noisy than pro data"[^12]; Trackman spin efficiency measurement has known limitations vs. Hawkeye[^29] |
| Defense and baserunning | Trackman doesn't measure fielding or sprint speed — consider supplementing with Blast Motion or running time data where available |

The most important structural limitation is **survival bias in the target variable**: if you train only on players who were drafted, you're not accounting for the undrafted players in your dataset who might have projected well but went undiscovered. Including undrafted players from your Trackman dataset as "negative examples" (with a 0 outcome) in the classification stage is methodologically cleaner.[^24]

***

## Practical Next Steps

1. **Data audit**: Inventory your 2024–2026 Trackman files for completeness — which games had units, how many BBEs per hitter, pitch counts per pitcher, coverage by conference
2. **Build the player crosswalk**: Match Trackman player IDs to Chadwick Bureau (accessible via `baseballr::chadwick_player_lu()`), then pull Statcast data for any historically drafted players
3. **Compute park factors**: Use the College Splits methodology (multi-year, regressed, component-level) or directly use FanGraphs' new conference-adjusted stats (launched March 2025)[^7]
4. **Build your college Stuff+ model first**: Train XGBoost on your Trackman pitch-level data predicting run value per pitch (use pitch result + contact quality as proxies if you lack a direct run value metric)
5. **Assemble historical training labels**: Draft class 2019–2023, pull their minor league performance from Baseball Reference/MiLB.com, join to any Trackman data available for those players
6. **Fit draft model**: Two-stage XGBoost (classification then regression), temporal CV, SHAP outputs
7. **Build Shiny/Streamlit dashboard** for interactive prospect exploration and board-building

---

## References

1. [How Data Visualization Aids in Player Development - YouTube](https://www.youtube.com/watch?v=dbTVVQ5lCMc) - This weeks video is a general introduction to some of the ways you can begin deciphering your trackm...

2. [Using Trackman Data to Evaluate Pitching - John Creel | Substack](https://therightspot.substack.com/p/using-trackman-data-to-evaluate-pitching) - Trackman technology allows scouts and coaches to analyze pitch movement, velocity profiles, and even...

3. [What Goes Into an MLB Draft Model: Batted Ball Profiles - Magnus](https://www.seemagnus.com/blog-posts-test/what-goes-into-a-mlb-draft-model-batted-ball-profiles)

4. [From spin rate stars to exit velo kings: TrackMan reveals MLB draft's ...](https://www.espn.com/mlb/insider/story/_/id/29284103/from-spin-rate-stars-exit-velo-kings-trackman-reveals-mlb-draft-metric-standouts) - Who spins the best curve of this year's draft prospects? Which hitter is crushing balls hardest? Her...

5. [Understanding Average Exit Velo by Age - WIN Reality](https://winreality.com/blog/exit-velo-by-age/) - Skill #1: Increased Bat-to-Ball Skills = Increased Average Exit Velo. Several factors influence exit...

6. [Making Sense of Division One Park Factors - College Splits Research](https://collegesplits.substack.com/p/making-sense-of-division-one-park) - Park adjustments are one thing, but what about strength of schedule? Even programs within the same c...

7. [We've Got College Data! - FanGraphs Baseball](https://blogs.fangraphs.com/weve-got-college-data/) - Division I data is updated daily and is available going back to 2021. wRC+, ERA-, and FIP- are confe...

8. [CORRECTED: Strength of Schedule Visualized by ERA-/OPS+thru Games 4/6/25](https://www.reddit.com/r/collegebaseball/comments/1jtp513/corrected_strength_of_schedule_visualized_by/) - CORRECTED: Strength of Schedule Visualized by ERA-/OPS+thru Games 4/6/25

9. [Looking Ahead: Translating College Performance | Baseball Prospectus](https://www.baseballprospectus.com/news/article/2787/looking-ahead-translating-college-performance/) - You know that insurance commercial where the guy sleepily mumbles that he's going to skip class befo...

10. [What Goes Into An MLB Draft Model: Vertical Approach Angle - Magnus](https://www.seemagnus.com/blog-posts-test/what-goes-into-an-mlb-draft-model-vertical-approach-angle)

11. [Analyzing Fastball Shapes Of Top 2024 College Pitchers](https://www.baseballamerica.com/stories/mlb-draft-prospects-analyzing-fastball-shapes-of-top-2024-college-pitchers/) - Examining the fastballs that stand out among the top college arms in the 2024 MLB Draft class.

12. [Finding Pro Pitching Comps For Top 2024 MLB Draft College Pitchers](https://www.baseballamerica.com/stories/finding-pro-pitching-comps-for-top-2024-mlb-draft-college-pitchers/) - With more data available than ever, we draw similarities between draft arms and pro pitchers, includ...

13. [How MLB Scouts Interpret TrackMan Numbers and How They ...](https://www.chriswest.tech/article/mlb_scouts_trackman_interpretation_and_draft_decisions) - TrackMan data now drives MLB scouting, shaping draft value and trade decisions. Pitch metrics like s...

14. [Trackman Portable B1: Key Metrics for Pitching and Hitting](https://www.trackman.com/baseball/Portable-B1/what-we-track) - Our patented technology lets you track what your eyes can't see. Measure full ball trajectory and sp...

15. [Top Prospect Games II: TrackMan Pitching Data](https://www.prepbaseballreport.com/news/PA/Top-Prospect-Games-II-TrackMan-Pitching-Data-0839241765) - Today we look at some of the data collected on the pitchers from our TrackMan portable unit at this ...

16. [Draft Profile: Tommy Mace - Magnus](https://www.seemagnus.com/blog-posts-test/draft-profile-tommy-mace) - Tommy Mace, hailing from Florida. He was recently selected by the Cleveland Indians in the second ro...

17. [Stuff+ in Collegiate Baseball - Normal CornBelters](https://cornbeltersbaseball.com/stuff-in-collegiate-baseball/)

18. [Introducing My Stuff+ Model - Adam Salorio on Substack](https://adamsalorio.substack.com/p/introducing-my-stuff-model-volume) - Using velocity, spin, and movement characteristics to measure pitch effectiveness.

19. [database of playerIDs and names between fangraphs and statcast](https://www.reddit.com/r/Sabermetrics/comments/107nx69/database_of_playerids_and_names_between_fangraphs/) - Does anyone know of a CSV or excel file that lists the MLB statcast player Id #, their name, and the...

20. [B1 Unit | Player Management System - Baseball](https://support.trackmanbaseball.com/hc/en-us/articles/5089811420699-B1-Unit-Player-Management-System) - Lastly, official players are linked to their official playerID (MLB, NPB, college roster ID) which i...

21. [Statcast Search CSV Documentation | baseballsavant.com](https://baseballsavant.mlb.com/csv-docs) - Statcast Search CSV Documentation. This is the documentation for the Statcast Search CSV data downlo...

22. [pybaseball/README.md at master - GitHub](https://github.com/jldbc/pybaseball/blob/master/README.md) - Statcast: Pull advanced metrics from Major League Baseball's Statcast system. Statcast data include ...

23. [jldbc/pybaseball: Pull current and historical baseball ... - GitHub](https://github.com/jldbc/pybaseball) - Pull current and historical baseball statistics using Python (Statcast, Baseball Reference, FanGraph...

24. [Predicting Future MLB Players Using Scouting Reports](https://arxiv.org/pdf/1910.12622.pdf)

25. [D1Baseball Renews Analytics Partnership With TrackMan](https://d1baseball.com/stories/d1baseball-renews-analytics-partnership-with-trackman/) - “Trackman data powers player development and scouting efforts across college baseball, while also su...

26. [Predicting MLB Player Value & Team Wins with Machine Learning](https://github.com/eric8395/baseball-analytics) - Predicting MLB player salaries and team wins with machine learning regression models. - GitHub - eri...

27. [How MLB Scouts Interpret TrackMan Numbers and How They Shape ...www.chriswest.tech › projects › mlb_scouts_trackman_interpretation_and_...](https://www.chriswest.tech/projects/mlb_scouts_trackman_interpretation_and_draft_decisions) - TrackMan data now drives MLB scouting, shaping draft value and trade decisions. Pitch metrics like s...

28. [Interpreting Coefficients for MLB Linear Regression Models](https://samirthanedar.github.io/2020/04/20/MLB-Linear-Regression/) - I scraped minor league stats and MLB ABs from baseball-reference.com and used OLS, Ridge, and Lasso ...

29. [Magnus Models and Constant Acceleration Assumptions: Post 46](https://baseballaero.com/2020/02/08/magnus-models-and-constant-acceleration-assumptions-post-46/) - The measurement system, Trackman, does not make a direct measurement of the spin that contributes to...

