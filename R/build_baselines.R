#!/usr/bin/env Rscript
#
# build_baselines.R — Compute league-wide percentile distributions for draft metrics.
# Calls DuckDB via Python bridge (one call per role), computes P1–P99 + mean/SD,
# saves as JSON. Each metric gets its full population distribution.
#
# Usage:
#   Rscript build_baselines.R [--output exports/baselines/league_baselines.json]
#

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(jsonlite)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || identical(x, "")) y else x

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  values <- list()
  index <- 1
  while (index <= length(args)) {
    arg <- args[[index]]
    if (!startsWith(arg, "--")) { index <- index + 1; next }
    parts <- strsplit(arg, "=", fixed = TRUE)[[1]]
    key <- sub("^--", "", parts[1])
    if (length(parts) > 1) {
      values[[key]] <- paste(parts[-1], collapse = "=")
    } else if (index < length(args) && !startsWith(args[[index + 1]], "--")) {
      values[[key]] <- args[[index + 1]]; index <- index + 1
    } else {
      values[[key]] <- TRUE
    }
    index <- index + 1
  }
  values
}

safe_pct <- function(num, den) {
  ifelse(den > 0, 100 * num / den, NA_real_)
}

metric_distribution <- function(values, metric_name) {
  if (all(is.na(values))) {
    return(list(metric = metric_name, n = 0L, mean = NA_real_, sd = NA_real_,
                p01 = NA_real_, p05 = NA_real_, p10 = NA_real_,
                p25 = NA_real_, p50 = NA_real_, p75 = NA_real_,
                p90 = NA_real_, p95 = NA_real_, p99 = NA_real_))
  }
  q <- quantile(values, probs = c(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
                na.rm = TRUE)
  list(metric = metric_name, n = sum(!is.na(values)),
       mean = mean(values, na.rm = TRUE), sd = sd(values, na.rm = TRUE),
       p01 = unname(q[1]), p05 = unname(q[2]), p10 = unname(q[3]),
       p25 = unname(q[4]), p50 = unname(q[5]), p75 = unname(q[6]),
       p90 = unname(q[7]), p95 = unname(q[8]), p99 = unname(q[9]))
}

compute_baselines_for <- function(df, metrics) {
  results <- list()
  for (m in names(metrics)) {
    col_name <- metrics[[m]]
    if (col_name %in% names(df)) {
      results[[m]] <- metric_distribution(df[[col_name]], m)
    }
  }
  results[["_population"]] <- list(
    metric = "_population",
    n_players = nrow(df),
    n_seasons = length(unique(df$season))
  )
  results
}

args <- parse_args()
output_path <- args[["output"]] %||% "exports/baselines/league_baselines.json"

script_dir <- dirname(normalizePath(sub("^--file=", "",
  grep("^--file=", commandArgs(), value = TRUE)[[1]])))
bridge_script <- file.path(script_dir, "..", "scripts", "compute_league_baselines.py")

if (!file.exists(bridge_script)) {
  stop("Baselines bridge not found: ", normalizePath(bridge_script))
}

# ── Load pitcher aggregates ──
message("Loading pitcher-season aggregates from DuckDB...")
pitcher_raw <- read.csv(
  pipe(paste("python3", shQuote(bridge_script), "--role pitchers --min-pitches 100")),
  stringsAsFactors = FALSE, check.names = FALSE
)
message(sprintf("  %d pitcher-seasons loaded", nrow(pitcher_raw)))

# ── Load hitter aggregates ──
message("Loading hitter-season aggregates from DuckDB...")
hitter_raw <- read.csv(
  pipe(paste("python3", shQuote(bridge_script), "--role hitters --min-pa 50")),
  stringsAsFactors = FALSE, check.names = FALSE
)
message(sprintf("  %d hitter-seasons loaded", nrow(hitter_raw)))

# ── Compute derived rates ──
pitcher_df <- pitcher_raw |>
  mutate(
    whiff_pct = safe_pct(whiff_count, swing_count),
    csw_pct = safe_pct(csw_count, pitch_count),
    zone_pct = safe_pct(zone_count, pitch_count),
    hard_hit_allowed_pct = safe_pct(hard_hit_allowed_count, bbe_count),
    data_completeness = pmax(0, 100 - (missing_critical_count / pmax(pitch_count, 1)) * 100)
  )

hitter_df <- hitter_raw |>
  mutate(
    contact_rate = safe_pct(contact_count, swing_count),
    whiff_rate = safe_pct(whiff_count, swing_count),
    chase_rate = safe_pct(chase_count, out_of_zone_count),
    barrel_rate_proxy_wood_adj = safe_pct(barrel_proxy_count, bbe_count),
    data_completeness = pmax(0, 100 - (missing_critical_count / pmax(event_rows, 1)) * 100)
  )

# ── Pitcher baselines ──
pitcher_metrics <- list(
  avg_fb_velo = "avg_fb_velo",
  max_fb_velo = "max_fb_velo",
  avg_ivb = "avg_ivb",
  avg_hb = "avg_hb",
  extension = "extension",
  avg_spin = "avg_spin",
  whiff_pct = "whiff_pct",
  csw_pct = "csw_pct",
  zone_pct = "zone_pct",
  hard_hit_allowed_pct = "hard_hit_allowed_pct",
  pitch_count = "pitch_count",
  arsenal_depth = "arsenal_depth",
  data_completeness = "data_completeness"
)

pitcher_baselines <- compute_baselines_for(pitcher_df, pitcher_metrics)

# ── Hitter baselines ──
hitter_metrics <- list(
  p90_ev_wood_adj = "p90_ev_wood_adj",
  avg_ev_wood_adj = "avg_ev_wood_adj",
  barrel_rate_proxy_wood_adj = "barrel_rate_proxy_wood_adj",
  contact_rate = "contact_rate",
  whiff_rate = "whiff_rate",
  chase_rate = "chase_rate",
  bbe_count = "bbe_count",
  event_rows = "event_rows",
  data_completeness = "data_completeness"
)

hitter_baselines <- compute_baselines_for(hitter_df, hitter_metrics)

# ── Assemble and save ──
baselines <- list(
  generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  source = "DuckDB → per-player-season aggregates → population percentiles",
  pitchers = pitcher_baselines,
  hitters = hitter_baselines
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_json(baselines, output_path, auto_unbox = TRUE, pretty = TRUE, digits = 8)
message(sprintf("Saved baselines: %s", normalizePath(output_path)))

# ── Summary ──
cat("\n── Pitcher baselines ──\n")
for (m in c("avg_fb_velo", "whiff_pct", "avg_ivb", "csw_pct")) {
  b <- pitcher_baselines[[m]]
  if (!is.null(b)) {
    cat(sprintf("  %-20s  mean=%.1f  sd=%.2f  p50=%.1f  p90=%.1f  (n=%d)\n",
                m, b$mean, b$sd, b$p50, b$p90, b$n))
  }
}

cat("\n── Hitter baselines ──\n")
for (m in c("p90_ev_wood_adj", "contact_rate", "whiff_rate", "chase_rate")) {
  b <- hitter_baselines[[m]]
  if (!is.null(b)) {
    cat(sprintf("  %-25s  mean=%.1f  sd=%.2f  p50=%.1f  p90=%.1f  (n=%d)\n",
                m, b$mean, b$sd, b$p50, b$p90, b$n))
  }
}
