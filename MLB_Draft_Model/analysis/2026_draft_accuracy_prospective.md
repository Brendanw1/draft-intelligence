# 2026 MLB Draft — PROSPECTIVE Model Accuracy Report

**Generated**: Training data filtered to exclude 2026 outcomes. Models retrained
from scratch on ≤2025 data only. This is a TRUE prospective test — the model
had never seen any 2026 draft outcomes during training.

**Model versions**:
- Tier 1 (draft position): XGBoost regression on 2,797 historical draftees (1189 H / 1608 P)
- Tier 2 (draftability): XGBoost on 51,674 player-season records (26324 H / 25350 P)
- Tier 3 (MLB arrival): Logistic regression + NN on 1,262 historical draftees (549 H / 713 P)
- All trained on ≤2025 data only — 2026 records excluded from training set

## Summary

| Metric | Value |
|--------|-------|
| Total college players drafted | 474 |
| Matched to model projections | 454 (96%) |
| Unmatched (not in FanGraphs/model) | 20 |
| Mean absolute delta (\|Δ\|) | 116.4 picks |
| Median absolute delta | 101.4 picks |
| RMSE | 140.7 picks |
| Hitter MAE | 126.1 picks (186 players) |
| Pitcher MAE | 109.7 picks (268 players) |
| Spearman rank correlation | 0.559 (p < 0.001) |

### Accuracy within MAE bands

| Band | Count | % |
|------|-------|---|
| ✓ Within ±50 picks | 109 | 24% |
| ✓ Within ±75 picks | 160 | 35% |
| ✓ Within ±100 picks | 218 | 48% |
| ✓ Within ±110 picks | 245 | 54% |
| ✓ Within ±150 picks | 311 | 69% |
| ✓ Within ±200 picks | 383 | 84% |

### Directional Accuracy

| Direction | Count | % |
|-----------|-------|---|
| ⬆ Drafted HIGHER than projected (model undervalued) | 206 | 45% |
| ⬇ Drafted LOWER than projected (model overvalued) | 248 | 55% |
| ✓ Exactly right | 0 | 0% |

The model shows a slight tendency to overvalue players (55% were drafted lower
than projected), meaning the model's predicted picks are systematically earlier
than actual draft position. This is consistent with the model training on
historical data where draft value was distributed differently.

### Accuracy by Round

| Round | Picks | In Range | Higher | Lower | Hit Rate | Mean \|Δ\| |
|-------|-------|----------|--------|-------|----------|-----------|
| 0 (Comp) | 16 | 6 | 4 | 6 | 38% | 126 |
| 1 | 18 | 7 | 5 | 6 | 39% | 143 |
| 2 | 16 | 9 | 2 | 5 | 56% | 111 |
| 3 | 18 | 7 | 4 | 7 | 39% | 117 |
| 4 | 18 | 6 | 7 | 5 | 33% | 133 |
| 5 | 24 | 16 | 5 | 3 | 67% | 90 |
| 6 | 24 | 14 | 7 | 3 | 58% | 105 |
| 7 | 23 | 15 | 5 | 3 | 65% | 94 |
| 8 | 26 | 19 | 5 | 2 | 73% | 88 |
| 9 | 24 | 18 | 3 | 3 | 75% | 79 |
| 10 | 25 | 23 | 1 | 1 | 92% | 62 |
| 11 | 18 | 12 | 2 | 4 | 67% | 79 |
| 12 | 20 | 15 | 3 | 2 | 75% | 78 |
| 13 | 20 | 16 | 1 | 3 | 80% | 75 |
| 14 | 23 | 17 | 3 | 3 | 74% | 86 |
| 15 | 19 | 9 | 3 | 7 | 47% | 133 |
| 16 | 23 | 12 | 5 | 6 | 52% | 118 |
| 17 | 23 | 9 | 2 | 12 | 39% | 144 |
| 18 | 18 | 6 | 5 | 7 | 33% | 170 |
| 19 | 18 | 1 | 5 | 12 | 6% | 185 |
| 20 | 19 | 1 | 5 | 13 | 5% | 263 |

Observations:
- Accuracy peaks in Rounds 8-14 (70-92% hit rates), where the model's mid-round
  projections are most reliable
- Early rounds (1-4) show poor hit rates (33-56%) because the model systematically
  undervalues elite talent (projects them much later than they're drafted)
- Late rounds (18-20) have near-zero hit rates — the model's projections cluster
  around pick 200-300 while actual picks extend to 615
- Comp rounds (Round 0) are particularly hard to predict since they're tied to
  specific team compensation scenarios

---

## Comparison: Prospective vs Retrospective (with 2026 leakage)

| Metric | Retrospective (with 2026) | Prospective (without 2026) | Change |
|--------|--------------------------|---------------------------|--------|
| Matched | 452 (93%) | 454 (96%) | +2 |
| Mean \|Δ\| | ~43 | 116.4 | +73.4 |
| Hit Rate (within band) | 94.9% | 54.0% | -40.9pp |
| Round 1 hit rate | 95% | 39% | -56pp |
| Round 10 hit rate | 93% | 92% | -1pp |
| Spearman ρ | ~0.95 | 0.559 | -0.391 |

Key takeaways:
1. **The retrospective numbers were dramatically overoptimistic** — when 2026
   outcomes were included in training, the model appeared nearly perfect (95%
   hit rate). The true prospective accuracy is 54%.
2. **The prospective numbers are honest and meaningful** — MAE of 116 picks
   is a substantial error (~4 rounds), but the model still captures meaningful
   rank-order signal (Spearman ρ = 0.559).
3. **Mid-round accuracy (R8-14) is genuinely good** — 70-92% hit rates suggest
   the model is particularly useful for identifying which mid-tier college
   players will get drafted.
4. **Early-round projections are the weakest** — the model struggles to predict
   which college players will be drafted in the top 50 picks, likely because
   high school players (not in the model's scope) dominate the top rounds.

---

## Matching Methodology and Limitations

### Matching Strategy

1. **MLBAMID (primary)**: Matched 450/454 players via MLB Stats API person_id
   → xMLBAMID in enriched projections. This is exact and reliable.
2. **Name + School crosswalk**: 1 player matched via school abbreviation mapping
3. **Name-only fallback**: 3 players matched by name alone (no MLBAMID match)

### Match Rate by School

The 20 unmatched college players are primarily from:
- Small/Non-D1 schools not covered by FanGraphs college stats (Heartland CC,
  Seton Hill, Central Missouri, Olivet, Park University-Gilbert, etc.)
- D1 schools where the player's name variant differs from our data (e.g.,
  "Christopher Hacopian" vs "Chris Hacopian" at Texas A&M)

### Limitations

1. **College-only model**: The draft model only covers college (D1) players.
   High school players (127 in 2026) and JC transfers are not predicted.
2. **FanGraphs coverage gap**: ~20 college draftees were not in FanGraphs
   college stats database, primarily from non-D1 or small-conference programs.
3. **Name variant sensitivity**: Some mismatches occur due to name variants
   (Chris/Christopher, Cam/Camden, etc.) despite fuzzy matching.
4. **Team abbreviation handling**: School names in draft data (full names like
   "Louisiana State University") differ from projections (abbreviations like
   "LSU"), requiring a comprehensive crosswalk.
5. **2026 draft is one observation**: Single-year evaluation has high variance.
   Accumulating multiple years of prospective validation is needed.

---

## Detailed Results

### Round 1 (Picks 1-40)

| Pick | Player | Pos | School | Proj Pick | Δ | Grade |
|------|--------|-----|--------|-----------|---|-------|
| 1 | Roch Cholowsky | SS | UCLA | 112.7 | -112 | elite |
| 3 | Vahn Lackey | C | Georgia Tech | 53.4 | -50 | elite |
| 4 | Jackson Flora | P | UC Santa Barbara | 160.1 | -156 | elite |
| 5 | Derek Curiel | OF | LSU | 291.2 | -286 | high |
| 6 | Zion Rose | OF | Louisville | 220.2 | -214 | elite |
| 8 | Drew Burress | OF | Georgia Tech | 61.9 | -54 | elite |
| 9 | AJ Gracia | OF | Virginia | 155.4 | -146 | elite |
| 10 | Tyler Bell | SS | Kentucky | 357.7 | -348 | high |
| 15 | Ryder Helfrick | C | Arkansas | 110.2 | -95 | elite |
| 17 | Logan Hughes | OF | Texas Tech | 37.2 | -20 | elite |
| 18 | Justin Lebron | SS | Alabama | 239.2 | -221 | high |
| 19 | Liam Peterson | P | Florida | 47.3 | -28 | elite |
| 20 | Jake Schaffner | SS | North Carolina | 224.7 | -205 | elite |
| 22 | Cameron Flukey | P | Coastal Carolina | 331.4 | -309 | high |
| 23 | Cade Townsend | P | Mississippi | 158.4 | -135 | elite |
| 24 | Ace Reese | 3B | Mississippi State | 175.8 | -152 | elite |
| 26 | Carter Beck | OF | Indiana State | 156.0 | -130 | elite |
| 28 | Jack Radel | P | Notre Dame | 145.1 | -117 | elite |
| 30 | Taylor Rabe | P | Mississippi | 87.6 | -58 | elite |
| 32 | Tegan Kuhns | P | Tennessee | 86.1 | -54 | elite |
| 35 | Hunter Dietz | P | Arkansas | 22.2 | +14 | elite |
| 37 | Daniel Jackson | OF | Georgia | 141.9 | -105 | elite |
| 38 | Logan Reddemann | P | UCLA | 150.8 | -113 | elite |
| 39 | Cole Carlon | P | Arizona State | 15.7 | +23 | elite |
| 42 | Chase Brunson | OF | TCU | 297.3 | -255 | high |
| 43 | Carson Tinney | C | Texas | 60.1 | -17 | elite |
| 45 | Jarren Advincula | 2B | Georgia Tech | 203.9 | -159 | elite |
| 46 | Ty Head | OF | North Carolina State | 193.2 | -147 | elite |
| 47 | Mason Edwards | P | USC | 63.2 | -16 | elite |
| 49 | Ben Blair | P | Liberty | 121.8 | -73 | elite |

### Rounds 2-5 (Picks 41-170)

| Pick | Player | Pos | School | Proj Pick | Δ | Grade |
|------|--------|-----|--------|-----------|---|-------|
| 52 | Ethan Kleinschmit | P | Oregon State | 101.2 | -49 | elite |
| 51 | Chris Rembert | 2B | Auburn | 339.9 | +289 | medium |
| 53 | Carson Kerce | SS | Georgia Tech | 123.5 | +71 | elite |
| 57 | Wes Mendes | P | Florida State | 29.1 | -28 | elite |
| 58 | Eric Becker | SS | Virginia | 99.9 | +42 | elite |
| 61 | Tyson LeBlanc | SS | Kansas | 68.7 | +8 | elite |
| 62 | Caden Sorrell | OF | Texas A&M | 131.2 | +69 | elite |
| 64 | Caden Bogenpohl | OF | Missouri State | 155.2 | +91 | high |
| 65 | Jake Brown | OF | LSU | 165.6 | +101 | high |
| 66 | Sawyer Strosnider | OF | TCU | 137.5 | +72 | elite |
| 72 | Dawson Montesa | P | West Virginia | 280.6 | +209 | high |
| 75 | Myles Bailey | 1B | Florida State | 229.2 | +154 | high |
| 76 | Jack Natili | C | Cincinnati | 182.7 | +107 | high |
| 77 | Joey Volchko | P | Georgia | 56.5 | -20 | elite |
| 80 | Jason DeCaro | P | North Carolina | 165.3 | +85 | elite |
| 81 | Gavin Grahovac | 3B | Texas A&M | 118.3 | +37 | elite |
| 82 | Dominic Voegele | P | Kansas | 122.7 | +41 | elite |
| 83 | Jacob Dudan | P | North Carolina State | 236.7 | +154 | high |
| 86 | Caden Ferraro | OF | Texas Tech | 216.0 | +130 | high |
| 88 | Brayden Dowd | OF | Florida State | 92.1 | +4 | elite |
| 90 | Peyton Bonds | OF | Rutgers | 130.2 | +40 | elite |
| 91 | Maxx Yehl | P | West Virginia | 261.4 | +170 | high |
| 92 | Aiden Robbins | OF | Texas | 103.1 | +11 | elite |
| 94 | Tyner Horn | P | Nebraska | 107.7 | +14 | elite |
| 95 | Tyler Head | C | LSU | 96.1 | +1 | elite |
| 96 | Landon Victorian | P | Louisiana | 49.3 | -47 | elite |
| 98 | Jaxon Millet | P | LSU | 59.0 | -39 | elite |
| 100 | Ethan Hurst | OF | Georgia | 91.2 | -9 | elite |
| 105 | Eric Segura | P | Oregon State | 202.3 | +97 | high |
| 107 | Tommy LaPour | P | TCU | 389.1 | +282 | medium |
| 108 | Simon Baumgardt | 2B | Southern California | 135.4 | +27 | elite |
| 111 | Roman Martin | SS | UCLA | 264.5 | +154 | medium |
| 114 | Dee Kennedy | SS | Kansas State | 155.4 | +41 | elite |
| 118 | Carlos Martinez | P | Hofstra | 156.4 | +38 | elite |
| 122 | Ethan Norby | P | East Carolina | 138.6 | +17 | elite |
| 123 | Kade Lewis | 3B | Wake Forest | 256.8 | +134 | high |
| 124 | Robbie Lavey | C | George Washington | 237.6 | +114 | high |
| 126 | Dylan Marionneaux | P | Northwestern State | 309.9 | +184 | medium |
| 128 | Deven Sheerin | P | LSU | 283.9 | +156 | high |

### Rounds 6-10 (Picks 171-310)

| Pick | Player | Pos | School | Proj Pick | Δ | Grade |
|------|--------|-----|--------|-----------|---|-------|
| 140 | Ryan Marohn | P | North Carolina State | 182.6 | +43 | elite |
| 146 | Cal Randall | P | UCLA | 222.6 | +77 | high |
| 147 | Trey Beard | P | Florida State | 97.1 | -50 | elite |
| 150 | Luke Nixon | 2B | North Carolina State | 264.7 | +115 | high |
| 156 | Lucas Davenport | P | Baylor | 330.0 | +174 | high |
| 157 | James Guyette | P | Kansas State | 201.8 | +45 | high |
| 158 | Declan Dahl | P | Louisiana Tech | 139.8 | -18 | elite |
| 161 | Will Gasparino | OF | UCLA | 184.6 | +24 | high |
| 163 | Aidan Knaak | P | Clemson | 246.5 | +84 | high |
| 164 | Nolan Higgins | P | Michigan State | 373.4 | +209 | medium |
| 180 | Justin LeGuernic | P | Clemson | 388.0 | +208 | medium |
| 182 | Michael Addari | P | Illinois State | 381.3 | +199 | medium |
| 183 | Duncan Marsten | P | Wake Forest | 326.9 | +144 | high |
| 195 | Clay Burdette | OF | Xavier | 337.9 | +143 | medium |
| 204 | Derek Schaefer | P | Arizona State | 388.7 | +185 | medium |
| 208 | Beau Bryans | P | Jacksonville State | 381.8 | +174 | medium |
| 210 | Aidan Keenan | P | Stanford | 433.4 | +223 | medium |
| 218 | Michael Harpster | P | East Tennessee State | 394.1 | +176 | low |
| 225 | Jayson Jones | 3B | Wichita State | 397.8 | +173 | medium |
| 228 | Alex Overbay | P | Arizona State | 374.6 | +147 | medium |
| 231 | Aidan Weaver | P | Duke | 362.6 | +132 | high |
| 248 | Luke Pettitte | TWP | Dallas Baptist | 362.5 | +114 | medium |
| 257 | JT Raab | P | Georgetown | 401.6 | +145 | medium |
| 277 | Chase Meyer | P | West Virginia | 395.6 | +119 | low |
| 306 | Jack Turner | P | New Mexico State | 454.9 | +149 | medium |

### Rounds 11-20 — Misses Only (|Δ| > 110)

| Pick | Player | Pos | School | Proj Pick | Δ | Grade |
|------|--------|-----|--------|-----------|---|-------|
| 348 | Rohan Lettow | P | San Diego State | 200.0 | -148 | high |
| 354 | Drew Horn | P | Middle Tennessee State | 472.7 | +119 | low |
| 361 | Owen Nowak | OF | Middle Tennessee State | 202.7 | -158 | high |
| 373 | Gavin Van Kempen | P | East Carolina | 215.8 | -157 | high |
| 379 | Jake Long | OF | Utah | 259.4 | -120 | high |
| 399 | Brayden Bakes | OF | Illinois State | 211.6 | -187 | high |
| 425 | Ty Brachbill | P | High Point | 301.6 | -123 | medium |
| 430 | Chris Diaz | P | Florida Gulf Coast | 297.6 | -132 | medium |
| 434 | Ryan Niedzwiedz | 1B | Southern Illinois Edwardsville | 275.9 | -158 | medium |
| 454 | Chase Frey | P | Grand Canyon | 327.8 | -126 | high |
| 455 | Ryan Kucherak | SS | Northwestern | 293.9 | -161 | medium |
| 464 | Sam Larson | P | Tulane | 283.4 | -181 | medium |
| 465 | Darin Horn | P | Coastal Carolina | 291.4 | -174 | medium |
| 468 | Grant Govel | P | Southern California | 236.7 | -231 | high |
| 478 | Dalton Wentz | 3B | Wake Forest | 172.3 | -306 | high |
| 484 | Colton Coates | SS | Louisiana Tech | 287.8 | -196 | high |
| 485 | Albert Roblez | P | Oregon State | 306.1 | -179 | high |
| 486 | Michael Lane | P | Delaware State | 335.2 | -151 | low |
| 487 | Ashton Pocol | P | Florida Gulf Coast | 371.0 | -116 | high |
| 489 | Matt Quintanar | C | Texas Tech | 360.5 | -128 | medium |
| 492 | Carson Cormier | P | Illinois State | 368.5 | -124 | medium |
| 494 | Josh Swink | P | Liberty | 334.7 | -159 | high |
| 501 | Javier Gorostola | 3B | Florida Gulf Coast | 290.0 | -211 | high |
| 508 | Tanner Mally | OF | Western Michigan | 252.8 | -255 | high |
| 511 | Ben Tryon | IF | Dallas Baptist | 342.6 | -168 | medium |
| 512 | Jack Lausch | OF | Northwestern | 357.7 | -154 | medium |
| 514 | Alex Kranzler | P | Vanderbilt | 284.5 | -230 | high |
| 523 | Camden Wimbish | P | Campbell | 381.0 | -142 | medium |
| 526 | Avery Ortiz | IF | Oklahoma State | 261.3 | -265 | low |
| 527 | Colter McAnelly | P | Utah | 338.6 | -188 | medium |
| 531 | Bear Madliak | C | Kansas State | 251.6 | -279 | low |
| 532 | Ethan Stade | P | Bowling Green | 337.4 | -195 | medium |
| 537 | Max Kaufer | C | Wichita State | 396.8 | -140 | medium |
| 543 | Parker Dillhoff | P | Nevada Las Vegas | 308.8 | -234 | medium |
| 554 | Cort MacDonald | OF | Stanford | 397.2 | -157 | low |
| 557 | PJ Moutzouridis | SS | Arizona State | 301.7 | -255 | medium |
| 558 | Andrew Duncan | OF | Wright State | 360.6 | -197 | high |
| 568 | Mikey Bell | 3B | Gonzaga | 364.8 | -203 | medium |
| 573 | Zac Cowan | P | LSU | 396.2 | -177 | medium |
| 576 | Cade Rusch | P | Bellarmine | 452.3 | -124 | medium |
| 577 | Luke Guth | P | Vanderbilt | 409.2 | -168 | medium |
| 578 | Tyce Armstrong | 1B | Baylor | 349.9 | -228 | high |
| 580 | Michael Petite | OF | Virginia Commonwealth | 309.2 | -271 | medium |
| 583 | Luke Bard | C | Houston Christian | 460.0 | -123 | low |
| 585 | Connor Fennell | P | Vanderbilt | 174.5 | -410 | elite |
| 587 | Michael Barnett | P | UCLA | 402.8 | -184 | medium |
| 589 | Jake Jackson | OF | San Diego State | 257.0 | -332 | medium |
| 594 | Kollin Ritchie | OF | Oklahoma State | 142.9 | -451 | elite |
| 596 | Andrew Wertz | P | Northeastern | 401.7 | -194 | low |
| 597 | Cody Airington | P | Austin Peay | 440.6 | -156 | low |
| 601 | Mick Uebelhor | P | Western Kentucky | 255.3 | -346 | high |
| 604 | Aiden VanDeHatert | P | Dallas Baptist | 364.7 | -239 | medium |
| 608 | Dean Toigo | OF | Arizona State | 364.9 | -243 | low |
| 609 | Justin Lee | P | UCLA | 345.3 | -264 | medium |
| 610 | Connor Shouse | 3B | Texas Tech | 264.7 | -345 | medium |

---

## Biggest Misses (|Δ| > 250)

| Pick | Player | Pos | School | Proj Pick | Δ | Direction | Grade |
|------|--------|-----|--------|-----------|---|-----------|-------|
| 594 | Kollin Ritchie | OF | Oklahoma State | 142.9 | -451 | Drafted LOWER (model overvalued) | elite |
| 585 | Connor Fennell | P | Vanderbilt | 174.5 | -410 | Drafted LOWER (model overvalued) | elite |
| 601 | Mick Uebelhor | P | Western Kentucky | 255.3 | -346 | Drafted LOWER (model overvalued) | high |
| 610 | Connor Shouse | 3B | Texas Tech | 264.7 | -345 | Drafted LOWER (model overvalued) | medium |
| 589 | Jake Jackson | OF | San Diego State | 257.0 | -332 | Drafted LOWER (model overvalued) | medium |
| 22 | Cameron Flukey | P | Coastal Carolina | 331.4 | +309 | Drafted HIGHER (model undervalued) | high |
| 478 | Dalton Wentz | 3B | Wake Forest | 172.3 | -306 | Drafted LOWER (model overvalued) | high |
| 51 | Chris Rembert | 2B | Auburn | 339.9 | +289 | Drafted HIGHER (model undervalued) | medium |
| 5 | Derek Curiel | OF | LSU | 291.2 | +286 | Drafted HIGHER (model undervalued) | high |
| 107 | Tommy LaPour | P | TCU | 389.1 | +282 | Drafted HIGHER (model undervalued) | medium |
| 531 | Bear Madliak | C | Kansas State | 251.6 | -279 | Drafted LOWER (model overvalued) | low |
| 580 | Michael Petite | OF | Virginia Commonwealth | 309.2 | -271 | Drafted LOWER (model overvalued) | medium |
| 526 | Avery Ortiz | IF | Oklahoma State | 261.3 | -265 | Drafted LOWER (model overvalued) | low |
| 609 | Justin Lee | P | UCLA | 345.3 | -264 | Drafted LOWER (model overvalued) | medium |
| 508 | Tanner Mally | OF | Western Michigan | 252.8 | -255 | Drafted LOWER (model overvalued) | high |
| 557 | PJ Moutzouridis | SS | Arizona State | 301.7 | -255 | Drafted LOWER (model overvalued) | medium |
| 42 | Chase Brunson | OF | TCU | 297.3 | +255 | Drafted HIGHER (model undervalued) | high |
| 104 | Ben Davis | P | Mississippi State | 343.6 | +240 | Drafted HIGHER (model undervalued) | high |
| 610 | Connor Shouse | 3B | Texas Tech | 378.4 | -232 | Drafted LOWER (model overvalued) | medium |
| 468 | Grant Govel | P | Southern California | 236.7 | -231 | Drafted LOWER (model overvalued) | high |

---

## Best Predictions (|Δ| ≤ 10, Rounds 1-5)

| Pick | Player | Pos | School | Proj Pick | Δ | Grade |
|------|--------|-----|--------|-----------|---|-------|
| 88 | Brayden Dowd | OF | Florida State | 92.1 | +4 | elite |
| 95 | Tyler Head | C | LSU | 96.1 | +1 | elite |
| 100 | Ethan Hurst | OF | Georgia | 91.2 | -9 | elite |
| 61 | Tyson LeBlanc | SS | Kansas | 68.7 | +8 | elite |
| 92 | Aiden Robbins | OF | Texas | 103.1 | +11 | elite |

---

## Unmatched Players (Not in Model)

These 20 college players were drafted but did not appear in the model's
FanGraphs-derived projection set (likely JUCO/D2/NAIA, insufficient
stats, or name mismatch):

| Pick | Player | School |
|------|--------|--------|
| 11 | Chris Hacopian | Texas A&M |
| 27 | Carson Wiggins | Arkansas |
| 69 | Evan Dempsey | Florida Gulf Coast |
| 87 | Cam Kozeal | Arkansas |
| 94 | Tyner Horn | Nebraska |
| 97 | Ryan Lynch | North Carolina |
| 127 | Paul Gutierrez-Contreras II | Cal State Fullerton |
| 142 | Jimmy Anderson | Heartland CC |
| 149 | Michael Anderson Jr. | Penn State |
| 175 | Owen Henne | Seton Hill |
| 192 | Ryan Oshinskie | Brown |
| 194 | Jack Scott | Central Missouri |
| 211 | Bryan Carney | University of Olivet |
| 223 | Charlie West | Connecticut |
| 256 | Cashel Dugger | UCLA |
| 276 | Kenneth Ward | Park University-Gilbert |
| 282 | Joey Urban | U Southern Mississippi |
| 285 | Matthew Bucciero | Fairfield |
| 290 | Carlos Sanchez | LSU Shreveport |
| 295 | Zach Peters | Virginia Commonwealth |

Note: Some of these players (like Chris Hacopian, Cam Kozeal, Tyner Horn)
are in the projection database under different name variants or team
abbreviations that weren't matched by our crosswalk. They represent
matching false negatives rather than model coverage gaps.

---

## Key Takeaways

1. **Genuine predictive signal**: The model achieves Spearman ρ = 0.559 with
   actual draft position, confirming it captures meaningful information about
   which college players will be drafted and approximately where.

2. **Systematic undervaluation of top talent**: The model consistently projects
   elite players (grades "elite"/"high") much later than they're actually drafted.
   This is because the model learned from historical data where highly-touted
   high school players (excluded from the college-only model) dominate the top
   rounds.

3. **Mid-round sweet spot**: The model performs best in rounds 8-14 (70-92% hit
   rates), where college performance metrics are most predictive of draft
   position and there's less noise from high school prospects and bonus-pool
   maneuvering.

4. **Honest vs inflated accuracy**: The prospective MAE (116.4 picks) is ~3x
   larger than the retrospective MAE (~43 picks with leakage), highlighting
   why true prospective validation is essential. The retrospective numbers
   gave a dangerously misleading picture of model performance.

5. **Model calibration needs work**: The "elite" and "high" grade designations
   don't reliably map to early-round draft positions. Players graded "elite"
   were drafted anywhere from pick 1 (Roch Cholowsky) to pick 594 (Kollin
   Ritchie), indicating poor calibration of the grade tiers.

6. **Next steps**: 
   - Incorporate high school player projections to improve early-round accuracy
   - Improve calibration of draft-grade tiers
   - Expand coverage to include smaller-conference and non-D1 programs
   - Run multi-year prospective validation as future drafts occur
