# URGENT: model bugs to fix (found 2026-10-29)

**Why urgent:** this project is on the owner's portfolio and the site is public (draft-intelligence.vercel.app).
They were found in a code review done for the GOAT-Pro prospect model (goat-pro `docs/prospect-model.md`, "MLB_Draft_Model
code review"). The scripts were read, not run, so confirm each bug before fixing it. Re-run the backtests and Model Lab numbers
after each fix, and update the Methodology page.

## Status (2026-10-29)

Code fixed and re-run. New artifacts are in `models/artifacts_fixed/` (same filenames as `artifacts_full/`, which is untouched).
Nothing is exported or deployed. A fixed leak lowers the numbers; these are the honest ones.

| # | file | bug (corrected where the review was wrong) | status | before → after |
|---|---|---|---|---|
| 1 | `scripts/train_tier3_mlb_arrival.py` | Confirmed (old L222-266): the round prior was computed on all players, own label included, and fed as an ordinary feature (fitted coefficient 1.00). Inputs unscaled. | **fixed**: prior out of fold (cross-fit inside each CV fold), shrunk to the pooled rate, passed as a true logit offset (`OffsetLogit`, coefficient fixed at 1); inputs standardised; L2 penalty (sklearn has no offset, so not elastic net). | Same label, same folds, old method vs fixed: AUC hitters 0.830 → 0.747, pitchers 0.843 → 0.770; Brier 0.095 → 0.103, 0.083 → 0.092. Site claims 0.79. **The round prior alone scores AUC 0.777 / 0.813**: the stats add nothing beyond draft round. |
| 2 | same | Confirmed with a correction: 2024+ draftees were not training rows, but they were in the round prior and the NN pool labelled 0 (old L223, L260-275), and the 2022-23 classes were themselves part-censored for "ever debuted". NN rate was built on the full pool before CV. | **fixed**: NN rate rebuilt per fold from non-scored rows only; label is now a fixed horizon (debut by end of draft_year+3), so 2021-23 are equally observed; 2024+ dropped from training, prior and NN pool. | Included in #1. Label change means Tier 3 now reads "P(MLB within 3 seasons)". |
| 3 | `scripts/train_tier2_full.py` | Confirmed (old L249-274, year-out stub L228-234). Players had no person_id among negatives, grouping used `hash(name)`. | **fixed**: Platt and isotonic fit on forward-chained OOF scores (season s scored by a model fit on earlier seasons, its players removed); scored on the 2024 class held out (players removed from training). Reliability bins saved in `tier2_full_features_{pt}.json` → `holdout`. | Hitters, 2024 holdout: AUC 0.994 (site, in-sample) → 0.977; Brier: live calibrator in-sample 0.012, old method out of sample 0.032, fixed 0.024; ECE 0.017 → 0.039 (old method) → 0.023. Pitchers: AUC 0.989 → 0.966; Brier 0.017 / 0.035 / 0.032; ECE 0.020 / 0.035 / 0.038. |
| 4 | `scripts/backtest_tier5.py` | Confirmed, plus: it read a `career_war` column that does not exist (raw WAR always 0). Correction: a 2015-20 backtest is impossible, those classes have WAR but no college stats. The only matured class (5 seasons done) is 2021. | **fixed**: cohort = matured classes only (`draft_year+5 <= 2026`), scored out of fold since 2021 is also the training class; no debut = 0 WAR, debuted without a WAR record = missing (dropped, none today); hurdle scored on its own label; no try/except. | Old (2023 class, censored): R² hitters -0.19, pitchers -0.88. Fixed (2021, OOF): R² +0.027 / -0.032; reach AUC 0.65 / 0.68; WAR>2 ranking AUC 0.75 / 0.60. **Gate FAILS for pitchers.** |
| 5 | `scripts/train_tier5_hurdle.py`, `train_tier5_value.py` | Confirmed: WAR>2 has 33 positives (18 in the 2021 training class); the stages were never combined; heldout 2022-23 is censored. Also: both used the stored `round_logit_prior`/`nn_mlb_rate`, which carry the #1 leak. | **partial**: P(reach MLB) × E[WAR \| reached], with per-fold prior/NN, is built and evaluated in `backtest_tier5.py` (matured class only). The two stage trainers are unchanged and still write the old Phase A artifacts that `export_frontend_data.py` puts on the WAR leaderboard and dossier card (marked "low confidence", not gated off). | See #4. Tier 5 must stay off: recommend dropping it from the export. |

Checks: `tests/test_leak_fixes.py` (one test per bug, each fails if its leak returns). Full suite: 86 passed.
Shared code: `src/mlb_draft_dashboard/leakfree.py` (pickled classes, so run inference/export with `PYTHONPATH=src`).

### To ship (owner)
1. Review and commit by path: the three scripts above, `src/mlb_draft_dashboard/leakfree.py`, `tests/test_leak_fixes.py`,
   `docs/tier5_backtest_report.md`, this file.
2. Switch artifacts: copy the 14 files in `models/artifacts_fixed/` over `models/artifacts_full/`.
3. Rebuild `calibration_curve_*`/`calibration_lookup_*` (`scripts/build_calibration_curve.py`), which the Model Lab reads.
4. `PYTHONPATH=src python scripts/infer_2026.py`, then `scripts/export_frontend_data.py`; drop Tier 5 from the export first.
5. Update Model Lab / Methodology / README / STATUS numbers: Tier 2 AUC 0.977 / 0.966 and ECE 2.3% / 3.8% on the 2024 class;
   Tier 3 AUC 0.75 / 0.77, label "within 3 seasons", and say it does not beat draft round alone; Tier 5 off (gate fails).
6. Deploy (`scripts/upload_r2.sh`, Vercel).

## Split audit fixes (2026-10-29, evening)

From the split audit (Draft Tier 1/2/3 rows). Re-trained, re-inferred and re-exported; not pushed, not uploaded.

| tier | fix | before → after |
|---|---|---|
| 2 | Same eligibility for both classes (freshmen kept); Age dropped (FanGraphs has it for all drafted, <half of undrafted) | see below |
| 2 | Height/BMI dropped: drafted = measured whole inches, undrafted = random imputed (87% non-integer); board ran 2-3x hot | found while fixing; not in the audit |
| 2 | Label = pick ≤ 315 within this season's draft or the next two; seasons 2021-2024 only; 2026 draftees back in | positive rate by season 6.7 / 6.5 / 6.4 / 6.2% (was 7.6 → 2.6%) |
| 2 | No player-based row drops; holdout trains on seasons with labels complete before the test draft (2021 for 2024) | |
| 2 | Site `mlb_p` = OOF Platt (the scored calibrator), not the quantile curve | board sum 1,696 → 714 over 3 drafts (~238/yr) |
| 2 | conf_strength fit on training rows only (`leakfree.conf_strength_table`) | |
| 2 | 2024 class AUC / ECE (re-read) | 0.977 / 0.966, 2.3% / 3.8% → 0.857 / 0.843, 1.4% / 0.6% |
| 1 | Forward-by-draft-year MAE, conf_strength refit before each test draft; `proj_pick` labelled "if drafted" | ~110 (random split: 111.0 / 107.8) → 117.4 / 115.2 |
| 3 | `LAST_SEASON = 2025` (debuts pulled 2026-07-14): classes 2021-22 | AUC 0.75 / 0.77 → 0.70 / 0.75 (round alone 0.72 / 0.77) |

Holdout honesty: the 2024 Tier 2 class has been read on several specs, twice today (first read today, before the height
fix: 0.950 / 0.941). Treat 0.857 / 0.843 as post hoc. The next clean read is the 2025 class after the 2027 draft.
Not done: Tier 4 comp display fixes (audit row "suspect"); `calibration_curve_*` / `calibration_lookup_*` and
`calibrate_probs.py` are no longer read by the export and describe the old model.

## Original list

| # | file | bug | fix |
|---|---|---|---|
| 1 | `scripts/train_tier3_mlb_arrival.py` | The round "prior" is an ordinary feature computed on ALL rows, so each player's own label leaks into his prior. The "prior offset" in the README is not an offset. | Compute the prior inside each training fold (out of fold), and pass it as a true offset (logit offset, not a feature). Scale inputs before the elastic net. |
| 2 | `scripts/train_tier3_mlb_arrival.py` | The neighbour rate feature leaks across CV folds. Also, 2024+ draftees are labelled 0 (no debut) although they are censored. | Rebuild the neighbour rate per fold. Drop censored classes from training, or label them with censoring (survival/competing-risk). |
| 3 | `scripts/train_tier2_full.py` | `CalibratedClassifierCV(XGB, sigmoid, cv=5)` is fit on all rows with stratified (not player-grouped) folds. The calibrated output is never scored on held-out data, and the year-out check is a stub. | Fit Platt on out-of-fold predictions from time-ordered, player-grouped folds. Report Brier, ECE and a reliability diagram on a later draft class. |
| 4 | `scripts/backtest_tier5.py` | It claims a 2015–20 cohort but reads the censored 2023 class. Missing WAR becomes 0. The hurdle is thresholded at a different label than it was trained on, and errors are swallowed (try/except). | Read the 2015–20 cohort only. Treat missing WAR as missing, not 0. Use the training label's threshold. Let errors raise. |
| 5 | `scripts/train_tier5_hurdle.py`, `train_tier5_value.py` | The hurdle (WAR > 2) has only 33 positives, and the value model is never combined with it. The held-out 2022–23 WAR is censored. | Combine them as P(reach) × E[value given reach], evaluate only on matured classes, and keep Tier 5 gated off until it passes. |

Also (lower priority): the Tier 4 comps use equal-weight Euclidean distance on z-scores. Correlated rate stats count several
times, volume stats are included, and missing values are set to 0. Use level- and age-adjusted skill z, drop volume, and use a
Mahalanobis or PCA distance.

### Shipped locally (2026-10-29, owner approved)
Steps 2-5 done: `artifacts_fixed/*` copied over `artifacts_full/` (backup: `models/artifacts_prefix_2026-10-29/`,
untracked, rollback = copy it back), calibration curves rebuilt, Tier 5 dropped from the export and the site (WAR page shows
"withdrawn: failed its backtest", dossier card hidden), inference + export re-run, Model Lab / Methodology / README / STATUS updated.
Two inference fixes found on the way: `infer_2026.py` keyed FanGraphs rows by name|team, so a two-way player's pitching row
overwrote his hitting row (now keyed by type); `OffsetLogit.predict_proba` clips standardised inputs to ±4 SD (1-PA lines put
round-18 picks at P 0.99). Not done: R2 upload (no credentials in the shell; the site does not read R2) and the Vercel deploy
(live = git deploy of `main`, see the commit message).
