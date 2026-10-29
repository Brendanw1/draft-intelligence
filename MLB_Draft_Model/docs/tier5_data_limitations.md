# Tier 5 WAR Value Pipeline — Data Limitations

**Generated**: August 21, 2026
**Branch**: `feat/baseballgoat-trackman-3d-lab`

---

## 1. Training Set Limitations

### 1.1 WAR Data Sparsity

| Metric | Value |
|--------|-------|
| Total records | 1,124 |
| Records with WAR | 216 (19.2%) |
| Records without WAR | 908 (80.8%) |

**Impact**: The model trains on only 19% of available drafted players. The remaining 80% have null WAR values because:
- Players who haven't debuted in MLB yet (no WAR accumulation)
- WAR scraping pipeline incomplete for some draft classes
- Right-censoring bias (recent draftees have fewer years of WAR data)

**Mitigation**: Future work should:
1. Complete WAR ground truth for all drafted players (2015-2023)
2. Impute WAR=0 for players who haven't debuted after 5 years
3. Rebuild training set with full WAR coverage

### 1.2 Small Effective Sample Size

| Split | Total Records | Records with WAR | Effective N |
|-------|---------------|------------------|-------------|
| Train (2021) | 376 | 102 | 27.1% |
| Heldout (2022-2023) | 748 | 114 | 15.2% |

**Impact**: With only ~100 training records per role, the model cannot learn robust patterns. High variance in WAR outcomes (range: -1.65 to 12.03) requires large sample sizes to generalize.

**Mitigation**: Accumulate more draft classes (2024, 2025, 2026) to increase training set size. Target: 500+ records per role with WAR data.

### 1.3 Right-Censoring Bias

| Draft Year | WAR Null Rate | Years of WAR Data |
|------------|---------------|-------------------|
| 2021 | 72.9% | 5 years (2021-2025) |
| 2022 | 80.5% | 4 years (2022-2025) |
| 2023 | 89.2% | 3 years (2023-2025) |

**Impact**: The heldout set (2022-2023) has incomplete WAR data. Players may accumulate more WAR in future seasons. This biases the model toward underestimating true 5-year WAR.

**Example**: A 2023 draftee with 0 WAR through 2025 may accumulate 5+ WAR by 2028, but the model sees them as "0 WAR" in training.

**Mitigation**:
- Wait for more draft classes to accumulate full 5-year WAR
- Use survival analysis techniques to handle right-censoring
- Exclude recent draft years from training until fully observed

---

## 2. Feature Limitations

### 2.1 No TrackMan Data

**Current features** (10 per role):
- Conference-adjusted stats (wOBA_adj, ERA_adj, etc.)
- Biometrics (height, BMI)
- Round prior (round_logit_prior)
- Nearest-neighbor MLB rate (nn_mlb_rate)

**Missing features**:
- Exit velocity, launch angle, barrel rate (hitters)
- Spin rate, induced vertical break, horizontal break (pitchers)
- Swing decisions, chase rate, contact quality

**Impact**: TrackMan features are strong predictors of MLB success. Their absence limits model accuracy, especially for:
- Distinguishing "lucky" vs "skilled" performance
- Identifying underlying talent vs surface-level stats
- Projecting development trajectory

**Root cause**: TrackMan data not available for most players in the training set (2021-2023 draft classes). Only available for players at schools with TrackMan systems.

**Mitigation**:
- Integrate TrackMan data as it becomes available (2024+ draft classes)
- Use proxy features (e.g., wood bat EV adjustment) where TrackMan unavailable
- Build separate models for TrackMan-equipped vs non-equipped schools

### 2.2 Biometric Imputation

| Metric | Coverage |
|--------|----------|
| Height | 100% (imputed) |
| BMI | 100% (imputed) |

**Impact**: 85% of height and 99.6% of BMI were imputed from conference+position distributions. While this prevents artifact learning (zero height = undrafted), imputed values add noise and reduce predictive power.

**Mitigation**:
- Source complete biometric data from MLB combine, pro days, or team records
- Use imputation uncertainty as a feature (e.g., "height_imputed" flag)

---

## 3. Target Variable Limitations

### 3.1 Zero-Inflated WAR Distribution

| Statistic | Value |
|-----------|-------|
| Median WAR | 0.0 |
| Mean WAR | 0.77 |
| Positive WAR rate | ~50% (where observed) |
| WAR > 2.0 rate | 2.94% |

**Impact**: Most drafted players produce 0 WAR over 5 years. The `signed_log_war` transform handles zero-inflation, but the extreme imbalance makes value regression difficult.

**Example**: Predicting whether a player will have WAR=0 vs WAR=0.1 is nearly impossible with current features.

**Mitigation**:
- Hurdle model (P(WAR>0)) is more appropriate than direct WAR regression
- Consider ordinal targets (WAR bands: 0, 0-1, 1-2, 2-5, 5+) instead of continuous
- Focus on "meaningfully productive" threshold (WAR > 2.0) rather than exact WAR

### 3.2 WAR Definition Ambiguity

**Current definition**: `war_years_1_through_5` — sum of WAR over first 5 MLB seasons

**Issues**:
- Doesn't account for players who debut late (e.g., 2021 draftee debuts in 2024)
- Doesn't distinguish between "0 WAR because injured" vs "0 WAR because untalented"
- Doesn't capture peak performance (a player with 10 WAR in year 1, then injured, is more valuable than 2 WAR spread over 5 years)

**Mitigation**:
- Use "WAR within 5 years of debut" instead of "WAR within 5 years of draft"
- Add injury/availability features
- Consider peak WAR or career WAR (no time limit) as alternative targets

---

## 4. Backtest Limitations

### 4.1 Cohort Size

| Metric | Value |
|--------|-------|
| Backtest cohort (2023) | 369 records |
| Records with WAR | 40 (10.8%) |
| Effective backtest N | 40 |

**Impact**: The backtest report claims "369 records" but only 40 have actual WAR data. This is an extremely small sample for evaluating model performance.

**Mitigation**:
- Clarify documentation: "369 records, 40 with observed WAR"
- Use 2015-2020 cohort for backtest (fully observed, larger sample)
- Report confidence intervals for all metrics

### 4.2 Temporal Leakage Risk

**Current split**: Train=2021, Heldout=2022-2023

**Risk**: If features include data from after the draft (e.g., minor league stats, TrackMan data from post-draft showcases), this introduces temporal leakage.

**Mitigation**:
- Verify all features are computed from pre-draft data only
- Add temporal leakage check to validation gates
- Document feature computation windows

---

## 5. Model Performance Limitations

### 5.1 Kill Criteria Failure

| Model | Metric | Threshold | Actual | Status |
|-------|--------|-----------|--------|--------|
| Value Hitter | R2 | >= 0.0 | -0.32 | FAIL |
| Value Pitcher | R2 | >= 0.0 | -0.61 | FAIL |
| Hurdle Hitter | AUC | >= 0.6 | 0.93 | PASS |
| Hurdle Pitcher | AUC | >= 0.6 | 0.69 | PASS |

**Impact**: The value regression performs worse than predicting the median WAR for all players. This means:
- Model predictions are less accurate than a naive baseline
- Deploying the model would provide worse guidance than "everyone is average"
- Hurdle model shows modest discrimination but not production-grade

**Root cause**: Small sample size + high variance + right-censoring + missing TrackMan features

**Mitigation**:
- Do not deploy value regression until R2 >= 0.0
- Deploy hurdle model only (with clear disclaimer about AUC 0.65-0.93)
- Focus on data collection (more draft classes, TrackMan integration)

### 5.2 Calibration Uncertainty

**Current calibration**: Decile calibration curves saved but not validated on independent data

**Risk**: Calibration may be overfit to the small training set. Real-world probabilities may differ from model outputs.

**Mitigation**:
- Validate calibration on 2024-2025 draft classes (when WAR data available)
- Use wider confidence intervals in frontend display
- Add "calibration uncertainty" note to UI

---

## 6. Recommendations

### 6.1 Immediate (Block Production Deployment)

1. **Complete WAR ground truth**: Scrape WAR for all drafted players (2015-2023)
2. **Impute WAR=0 for non-debuted players**: After 5 years, if no MLB debut, WAR=0
3. **Rebuild training set**: Target 500+ records per role with WAR data
4. **Re-run Phase A**: With larger, complete dataset
5. **Re-evaluate kill criteria**: R2 >= 0.0 required for production

### 6.2 Short-Term (Improve Model Quality)

1. **Integrate TrackMan data**: As available for 2024+ draft classes
2. **Extend temporal window**: Include 2024, 2025 draft classes (when WAR available)
3. **Add feature engineering**:
   - Wood bat EV adjustment (already in pipeline)
   - Park factors
   - Strength of schedule
4. **Improve target definition**:
   - Use "WAR within 5 years of debut" instead of draft
   - Consider ordinal targets (WAR bands)

### 6.3 Long-Term (Production-Grade Model)

1. **Survival analysis**: Handle right-censoring properly
2. **Ensemble methods**: Combine multiple model architectures
3. **Bayesian approach**: Quantify prediction uncertainty
4. **Continuous retraining**: As new draft classes accumulate WAR
5. **A/B testing**: Compare model predictions vs scout assessments

---

## 7. Conclusion

The Tier 5 WAR Value Pipeline is **architecturally sound** but **data-limited**. The two-stage approach (hurdle + value) is appropriate for zero-inflated WAR distributions. Feature engineering is well-designed given available data.

**The core blocker is data quantity and quality:**
- Only 19% of records have WAR data
- Small effective sample (~100 per role)
- Right-censoring bias in recent draft classes
- No TrackMan features

**Path to production:**
1. Complete WAR ground truth (2015-2023)
2. Accumulate more draft classes (2024-2026)
3. Integrate TrackMan data
4. Re-train with larger dataset
5. Re-evaluate kill criteria

**What IS production-ready:**
- Hurdle model architecture and training pipeline
- Validation gates and honesty report framework
- Frontend integration (leaderboard, dossier card)
- Export pipeline with Tier 5 fields
- Backtest infrastructure

**Recommendation**: Deploy hurdle model as "research feature" with clear disclaimer. Hold value regression until kill criteria met.
