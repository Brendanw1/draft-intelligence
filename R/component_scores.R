# component_scores.R — Domain-weighted draft component scoring
#
# Replaces heuristic rowMeans(cbind(scale_0_100(...))) with:
#   1. Population percentiles (via pre-computed league baselines)
#   2. Domain-informed component weights
#   3. Proper directional handling (lower-is-better metrics inverted)
#
# Source after build_baselines.R has been run (or baselines JSON exists).
#
# Functions:
#   load_draft_baselines(path)          → list(pitchers, hitters)
#   score_percentile(value, baseline)   → 0-100 (higher = better percentile)
#   score_inverse(value, baseline)      → 0-100 (lower = better percentile)  
#   compute_hitter_scores(df, bl)       → df with score columns
#   compute_pitcher_scores(df, bl)      → df with score columns

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
})

# ── Load baselines from JSON ──
load_draft_baselines <- function(path = "exports/baselines/league_baselines.json") {
  if (!file.exists(path)) {
    stop("League baselines not found at: ", normalizePath(path, mustWork = FALSE),
         "\nRun: Rscript R/build_baselines.R")
  }
  bl <- fromJSON(path, simplifyVector = FALSE)
  message(sprintf("Loaded baselines: %s (generated %s, %d pitcher metrics, %d hitter metrics)",
                  path, bl$generated,
                  length(bl$pitchers) - 1,  # subtract _population
                  length(bl$hitters) - 1))
  bl
}

# ── Map a raw value to population percentile (0-100) ──
# Uses linear interpolation between stored percentiles.
# Higher raw value → higher percentile (use for velo, EV, contact%, etc.)
score_percentile <- function(value, baseline) {
  if (is.null(baseline) || is.na(value) || is.null(value)) return(NA_real_)
  
  # Extract percentile reference points
  pcts <- c(1, 5, 10, 25, 50, 75, 90, 95, 99)
  refs <- c(baseline$p01, baseline$p05, baseline$p10,
            baseline$p25, baseline$p50, baseline$p75,
            baseline$p90, baseline$p95, baseline$p99)
  
  # Remove NAs
  valid <- !is.na(refs)
  if (sum(valid) < 2) return(NA_real_)
  pcts <- pcts[valid]
  refs <- refs[valid]
  
  # Interpolate
  if (value <= refs[1]) return(pcts[1])
  if (value >= refs[length(refs)]) return(pcts[length(pcts)])
  
  # Find bracket
  for (i in seq_len(length(refs) - 1)) {
    if (value >= refs[i] && value <= refs[i + 1]) {
      frac <- (value - refs[i]) / (refs[i + 1] - refs[i])
      return(pcts[i] + frac * (pcts[i + 1] - pcts[i]))
    }
  }
  return(NA_real_)
}

# ── Inverse percentile (lower raw value → higher score) ──
score_inverse <- function(value, baseline) {
  p <- score_percentile(value, baseline)
  if (is.na(p)) return(NA_real_)
  100 - p
}

# ── Weighted score helper with NA tolerance ──
weighted_score <- function(scores, weights) {
  # scores: named list/vector of component scores (0-100, NA allowed)
  # weights: named numeric vector
  common <- intersect(names(scores), names(weights))
  if (length(common) == 0) return(NA_real_)
  
  w <- weights[common]
  s <- unlist(scores[common])
  
  # Remove NA components, renormalize weights
  valid <- !is.na(s)
  if (sum(valid) == 0) return(NA_real_)
  
  w <- w[valid] / sum(w[valid])
  s <- s[valid]
  
  sum(w * s)
}

# ══════════════════════════════════════════════════════════════
#  HITTER SCORES
# ══════════════════════════════════════════════════════════════

compute_hitter_scores <- function(df, baselines) {
  # df must have: p90_ev_wood_adj, avg_ev_wood_adj, barrel_rate_proxy_wood_adj,
  #               contact_rate, whiff_rate, chase_rate, bbe_count, event_rows,
  #               data_completeness_score, one_season_only_flag
  # baselines: from load_draft_baselines()$hitters
  
  bl <- baselines$hitters
  
  if (nrow(df) == 0) {
    message("compute_hitter_scores: empty input, returning zero-row dataframe")
    return(df[0, , drop = FALSE])
  }
  
  # Pre-compute per-player percentiles
  df <- df |>
    rowwise() |>
    mutate(
      # Impact components (higher = better)
      pctl_ev90      = score_percentile(p90_ev_wood_adj, bl$p90_ev_wood_adj),
      pctl_avg_ev    = score_percentile(avg_ev_wood_adj, bl$avg_ev_wood_adj),
      pctl_barrel    = score_percentile(barrel_rate_proxy_wood_adj, bl$barrel_rate_proxy_wood_adj),
      
      # Contact components (higher = better)
      pctl_contact   = score_percentile(contact_rate, bl$contact_rate),
      pctl_whiff_inv = score_inverse(whiff_rate, bl$whiff_rate),
      pctl_chase_inv = score_inverse(chase_rate, bl$chase_rate),
      
      # Reach / projectability (EV potential + volume)
      pctl_volume    = score_percentile(bbe_count, bl$bbe_count),
      pctl_sample    = score_percentile(event_rows, bl$event_rows),
      
      # Data quality
      pctl_complete  = score_percentile(data_completeness_score, bl$data_completeness)
    ) |>
    ungroup()
  
  # Component weights (domain-informed)
  IMPACT_WEIGHTS  <- c(pctl_ev90 = 0.50, pctl_avg_ev = 0.20, pctl_barrel = 0.30)
  CONTACT_WEIGHTS <- c(pctl_contact = 0.40, pctl_whiff_inv = 0.30, pctl_chase_inv = 0.30)
  REACH_WEIGHTS   <- c(pctl_ev90 = 0.50, pctl_avg_ev = 0.25, pctl_barrel = 0.25)
  
  # Compute composite scores row-by-row
  df <- df |>
    rowwise() |>
    mutate(
      impact_score = weighted_score(
        list(pctl_ev90 = pctl_ev90, pctl_avg_ev = pctl_avg_ev, pctl_barrel = pctl_barrel),
        IMPACT_WEIGHTS),
      contact_score = weighted_score(
        list(pctl_contact = pctl_contact, pctl_whiff_inv = pctl_whiff_inv, pctl_chase_inv = pctl_chase_inv),
        CONTACT_WEIGHTS),
      reach_score = weighted_score(
        list(pctl_ev90 = pctl_ev90, pctl_avg_ev = pctl_avg_ev, pctl_barrel = pctl_barrel),
        REACH_WEIGHTS),
      # Risk: penalties for incomplete data, single-season, low sample, and missing EV
      risk_score_raw = (100 - coalesce(pctl_complete, 50)) * 0.35 +
        (if (one_season_only_flag) 30 else 0) * 0.25 +
        (if (bbe_count < 10 && !is.na(bbe_count)) 70 else if (bbe_count < 25 && !is.na(bbe_count)) 40 else 0) * 0.20 +
        (if (is.na(pctl_ev90) || is.na(pctl_barrel)) 60 else 0) * 0.20,
      # Trend adjustment: positive EV90 trend → reach bonus, negative → risk penalty
      trend_delta_val = coalesce(trend_delta, 0),
      trend_bonus = if (abs(trend_delta_val) <= 0.5) 0 else {
        if (trend_delta_val > 0) pmin(10, trend_delta_val / 4 * 10) else 0
      },
      trend_penalty = if (abs(trend_delta_val) <= 0.5) 0 else {
        if (trend_delta_val < 0) pmin(10, abs(trend_delta_val) / 4 * 10) else 0
      },
      reach_score = pmin(100, reach_score + trend_bonus),
      risk_score_raw = risk_score_raw + trend_penalty,
      risk_score = pmin(100, risk_score_raw),
      # Draft value base (may be NA if EV data missing)
      dv_raw = weighted_score(
        list(reach = reach_score, impact = impact_score, contact = contact_score),
        c(reach = 0.25, impact = 0.50, contact = 0.25)),
      draft_value_score = if (is.na(reach_score) && is.na(impact_score)) {
        pmax(0, pmin(40, coalesce(contact_score, 30) - 0.25 * risk_score))
      } else {
        pmax(0, pmin(100, coalesce(dv_raw, 50) - 0.10 * risk_score))
      }
    ) |>
    ungroup()
  
  df
}

# ══════════════════════════════════════════════════════════════
#  PITCHER SCORES
# ══════════════════════════════════════════════════════════════

compute_pitcher_scores <- function(df, baselines) {
  # df must have: avg_fb_velo, max_fb_velo, avg_ivb, avg_hb, extension, avg_spin,
  #               whiff_pct, csw_pct, zone_pct, hard_hit_allowed_pct,
  #               pitch_count, arsenal_count,
  #               data_completeness_score, one_season_only_flag
  # baselines: from load_draft_baselines()$pitchers
  
  bl <- baselines$pitchers
  
  if (nrow(df) == 0) {
    message("compute_pitcher_scores: empty input, returning zero-row dataframe")
    return(df[0, , drop = FALSE])
  }
  
  df <- df |>
    rowwise() |>
    mutate(
      # Stuff components (higher = better)
      pctl_fb_velo   = score_percentile(avg_fb_velo, bl$avg_fb_velo),
      pctl_max_velo  = score_percentile(max_fb_velo, bl$max_fb_velo),
      pctl_ivb       = score_percentile(avg_ivb, bl$avg_ivb),
      pctl_hb        = score_percentile(abs(avg_hb), bl$avg_hb),  # abs HB — movement in either direction
      pctl_spin      = score_percentile(avg_spin, bl$avg_spin),
      
      # Command components (higher = better)
      pctl_zone      = score_percentile(zone_pct, bl$zone_pct),
      pctl_csw       = score_percentile(csw_pct, bl$csw_pct),
      pctl_hh_inv    = score_inverse(hard_hit_allowed_pct, bl$hard_hit_allowed_pct),
      
      # Whiff (straddles stuff + command — used in reach)
      pctl_whiff     = score_percentile(whiff_pct, bl$whiff_pct),
      
      # Reach components (physical projectability)
      pctl_ext       = score_percentile(extension, bl$extension),
      pctl_arsenal   = score_percentile(arsenal_count, bl$arsenal_depth),
      pctl_volume    = score_percentile(pitch_count, bl$pitch_count),
      
      # Data quality
      pctl_complete  = score_percentile(data_completeness_score, bl$data_completeness)
    ) |>
    ungroup()
  
  # Component weights
  STUFF_WEIGHTS    <- c(pctl_ivb = 0.30, pctl_hb = 0.20, pctl_fb_velo = 0.30, pctl_spin = 0.20)
  COMMAND_WEIGHTS  <- c(pctl_zone = 0.40, pctl_csw = 0.30, pctl_hh_inv = 0.30)
  REACH_WEIGHTS    <- c(pctl_fb_velo = 0.35, pctl_max_velo = 0.20, pctl_ext = 0.15,
                        pctl_arsenal = 0.15, pctl_whiff = 0.15)
  
  df <- df |>
    rowwise() |>
    mutate(
      stuff_score = weighted_score(
        list(pctl_ivb = pctl_ivb, pctl_hb = pctl_hb,
             pctl_fb_velo = pctl_fb_velo, pctl_spin = pctl_spin),
        STUFF_WEIGHTS),
      command_score = weighted_score(
        list(pctl_zone = pctl_zone, pctl_csw = pctl_csw, pctl_hh_inv = pctl_hh_inv),
        COMMAND_WEIGHTS),
      reach_score = weighted_score(
        list(pctl_fb_velo = pctl_fb_velo, pctl_max_velo = pctl_max_velo,
             pctl_ext = pctl_ext, pctl_arsenal = pctl_arsenal, pctl_whiff = pctl_whiff),
        REACH_WEIGHTS),
      risk_score_raw = (100 - coalesce(pctl_complete, 50)) * 0.35 +
        (if (one_season_only_flag) 30 else 0) * 0.25 +
        (if (pitch_count < 50 && !is.na(pitch_count)) 70 else if (pitch_count < 150 && !is.na(pitch_count)) 40 else 0) * 0.20 +
        (if (is.na(pctl_fb_velo) || is.na(pctl_ivb)) 60 else 0) * 0.20,
      # Trend adjustment: positive FB velo trend → reach bonus, negative → risk penalty
      trend_delta_val = coalesce(trend_delta, 0),
      trend_bonus = if (abs(trend_delta_val) <= 0.5) 0 else {
        if (trend_delta_val > 0) pmin(10, trend_delta_val / 4 * 10) else 0
      },
      trend_penalty = if (abs(trend_delta_val) <= 0.5) 0 else {
        if (trend_delta_val < 0) pmin(10, abs(trend_delta_val) / 4 * 10) else 0
      },
      reach_score = pmin(100, reach_score + trend_bonus),
      risk_score_raw = risk_score_raw + trend_penalty,
      risk_score = pmin(100, risk_score_raw),
      # Draft value base
      dv_raw = weighted_score(
        list(reach = reach_score, stuff = stuff_score, command = command_score),
        c(reach = 0.35, stuff = 0.50, command = 0.15)),
      draft_value_score = if (is.na(stuff_score) || is.na(reach_score)) {
        pmax(0, pmin(40, coalesce(command_score, 30) - 0.25 * risk_score))
      } else {
        pmax(0, pmin(100, coalesce(dv_raw, 50) - 0.10 * risk_score))
      }
    ) |>
    ungroup()
  
  df
}
