suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(readr)
  library(purrr)
})

# ── Source component scoring engine ──
script_dir <- dirname(normalizePath(sub("^--file=", "",
  grep("^--file=", commandArgs(), value = TRUE)[[1]])))
component_scores_path <- file.path(script_dir, "component_scores.R")
if (file.exists(component_scores_path)) {
  source(component_scores_path)
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || identical(x, "")) y else x
}

clean_names_base <- function(names_vec) {
  out <- gsub("([a-z0-9])([A-Z])", "\\1_\\2", names_vec, perl = TRUE)
  out <- gsub("[^A-Za-z0-9]+", "_", out)
  out <- tolower(out)
  out <- gsub("_+", "_", out)
  gsub("^_|_$", "", out)
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  values <- list()
  index <- 1
  while (index <= length(args)) {
    arg <- args[[index]]
    if (!startsWith(arg, "--")) {
      index <- index + 1
      next
    }

    parts <- strsplit(arg, "=", fixed = TRUE)[[1]]
    key <- sub("^--", "", parts[1])

    if (length(parts) > 1) {
      value <- paste(parts[-1], collapse = "=")
      values[[key]] <- value
      index <- index + 1
      next
    }

    if (index < length(args) && !startsWith(args[[index + 1]], "--")) {
      values[[key]] <- args[[index + 1]]
      index <- index + 2
    } else {
      values[[key]] <- TRUE
      index <- index + 1
    }
  }
  values
}

safe_pct <- function(num, den) {
  ifelse(den > 0, 100 * num / den, NA_real_)
}

scale_0_100 <- function(x, descending = FALSE) {
  if (all(is.na(x))) {
    return(rep(NA_real_, length(x)))
  }
  filled <- ifelse(is.na(x), median(x, na.rm = TRUE), x)
  if (descending) {
    filled <- -filled
  }
  if (length(unique(filled)) == 1) {
    return(rep(50, length(filled)))
  }
  dplyr::percent_rank(filled) * 100
}

first_non_missing <- function(x) {
  out <- x[!is.na(x) & x != ""]
  if (length(out) == 0) NA_character_ else out[[1]]
}

canonical_pitch_type <- function(x) {
  case_when(
    x %in% c("Fastball", "FourSeamFastBall", "TwoSeamFastBall") ~ "Fastball",
    x %in% c("Sinker") ~ "Sinker",
    x %in% c("Slider") ~ "Slider",
    x %in% c("Sweeper") ~ "Sweeper",
    x %in% c("Curveball", "Knuckle Curve", "Slurve") ~ "Curveball",
    x %in% c("ChangeUp") ~ "ChangeUp",
    x %in% c("Cutter") ~ "Cutter",
    x %in% c("Splitter") ~ "Splitter",
    TRUE ~ NA_character_
  )
}

load_team_mapping <- function(path) {
  script_flag <- grep("^--file=", commandArgs(), value = TRUE)
  script_dir <- if (length(script_flag) > 0) dirname(normalizePath(sub("^--file=", "", script_flag[[1]]))) else getwd()
  default_path <- file.path(script_dir, "..", "configs", "team_mapping_all_teams.csv")
  required <- c("team_code", "school_name", "conference")

  default_mapping <- if (file.exists(default_path)) {
    mapping <- read_csv(default_path, show_col_types = FALSE)
    names(mapping) <- clean_names_base(names(mapping))
    mapping |> select(all_of(required)) |> distinct()
  } else {
    tibble(team_code = character(), school_name = character(), conference = character())
  }

  if (is.null(path) || !file.exists(path)) {
    return(default_mapping)
  }

  user_mapping <- read_csv(path, show_col_types = FALSE)
  names(user_mapping) <- clean_names_base(names(user_mapping))
  missing <- setdiff(required, names(user_mapping))
  if (length(missing) > 0) {
    stop("Team mapping is missing required columns: ", paste(missing, collapse = ", "))
  }

  bind_rows(
    default_mapping,
    user_mapping |> select(all_of(required))
  ) |>
    arrange(team_code) |>
    group_by(team_code) |>
    summarise(
      school_name = dplyr::last(na_if(school_name, "")),
      conference = dplyr::last(na_if(conference, "")),
      .groups = "drop"
    ) |>
    mutate(
      school_name = coalesce(school_name, ""),
      conference = coalesce(conference, "")
    )
}

load_player_metadata <- function(path) {
  if (is.null(path) || !file.exists(path)) {
    return(tibble(role = character(), player_uid = character(), class_year = character()))
  }
  metadata <- read_csv(path, show_col_types = FALSE)
  names(metadata) <- clean_names_base(names(metadata))
  required <- c("role", "player_uid", "class_year")
  missing <- setdiff(required, names(metadata))
  if (length(missing) > 0) {
    stop("Player metadata is missing required columns: ", paste(missing, collapse = ", "))
  }
  metadata |> select(all_of(required)) |> distinct()
}

write_export_table <- function(frame, output_dir, filename_base, arrow_available) {
  if (arrow_available) {
    arrow::write_parquet(frame, file.path(output_dir, paste0(filename_base, ".parquet")))
  } else {
    readr::write_csv(frame, file.path(output_dir, paste0(filename_base, ".csv")))
  }
}

build_pa_summary <- function(data) {
  data |>
    arrange(game_id, inning, top_bottom, paof_inning, pitchof_pa, pitch_no) |>
    group_by(season, batter_uid, batter, batter_team, batter_side, pitcher_uid, pa_key) |>
    summarise(
      pa_result = dplyr::last(play_result),
      korbb_last = dplyr::last(kor_bb),
      pitch_count_pa = n(),
      .groups = "drop"
    )
}

metric_lookup <- list(
  hitters = c(
    reach_score = "Reach Score",
    impact_score = "Impact Score",
    contact_score = "Contact Score",
    risk_score = "Risk Score",
    p90_ev_wood_adj = "P90 EV (Wood Adj)",
    avg_ev_wood_adj = "Avg EV (Wood Adj)",
    barrel_rate_proxy_wood_adj = "Barrel Proxy",
    contact_rate = "Contact Rate",
    whiff_rate = "Whiff Rate",
    chase_rate = "Chase Rate"
  ),
  pitchers = c(
    reach_score = "Reach Score",
    stuff_score = "Stuff Score",
    command_score = "Command Score",
    risk_score = "Risk Score",
    avg_fb_velo = "Avg FB Velo",
    max_fb_velo = "Max FB Velo",
    avg_ivb = "Avg IVB",
    avg_hb = "Avg HB",
    extension = "Extension",
    csw_pct = "CSW%",
    whiff_pct = "Whiff%",
    zone_pct = "Zone%"
  )
)

required_source_columns <- c(
  "player_type", "season", "player_uid", "player_name", "team_code", "bats", "throws"
)

args <- parse_args()
output_dir <- args[["output"]] %||% "exports/dashboard"
team_mapping_path <- args[["team-mapping"]] %||% NA_character_
player_metadata_path <- args[["player-metadata"]] %||% NA_character_
export_ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
arrow_available <- requireNamespace("arrow", quietly = TRUE)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (!arrow_available) {
  message("Package `arrow` is not installed in R. Falling back to CSV dashboard exports.")
}

# ── Data source: DuckDB via Python bridge (file-based, not pipe — avoids OOM) ──
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]])))
bridge_script <- file.path(script_dir, "..", "scripts", "export_draft_source_agg.py")

if (!file.exists(bridge_script)) {
  stop(sprintf("Python bridge not found: %s — pipeline cannot run", bridge_script))
}

message("Loading data from DuckDB...")
limit_arg <- if (!is.null(args[["limit"]])) paste("--limit", args[["limit"]]) else ""
season_arg <- if (!is.null(args[["season"]])) paste("--season", args[["season"]]) else ""
temp_csv <- tempfile(fileext = ".csv")
py_args <- c(shQuote(bridge_script), limit_arg, season_arg, "--output", shQuote(temp_csv))
py_args <- py_args[nzchar(py_args)]
system2("python3", py_args)
raw <- data.table::fread(temp_csv, stringsAsFactors = FALSE, check.names = FALSE, showProgress = FALSE)
raw <- as.data.frame(raw)  # convert data.table to data.frame for dplyr compatibility
unlink(temp_csv)

if (nrow(raw) == 0) {
  stop("Source data is empty — DuckDB may need to be rebuilt (python3 db/build_db.py)")
}

message(sprintf("Loaded %s rows, %d columns", format(nrow(raw), big.mark = ","), ncol(raw)))

# Column names already lowercase_snake_case from the Python bridge
missing_source_columns <- setdiff(required_source_columns, names(raw))
if (length(missing_source_columns) > 0) {
  stop("Source data is missing required columns: ", paste(missing_source_columns, collapse = ", "))
}
mapping <- load_team_mapping(team_mapping_path)
player_metadata <- load_player_metadata(player_metadata_path)

# ── Adaptor: convert pre-aggregated bridge output to expected format ──
# The aggregated bridge outputs one row per (player_type, season, player_uid)
# with all counts/aggregations pre-computed in SQL. We compute rates and proceed.

message("Processing pre-aggregated data...")

raw_hitters <- raw |> filter(player_type == "hitter")
raw_pitchers <- raw |> filter(player_type == "pitcher")

# --- Hitters: compute rates and build hitter_season ---
hitter_season <- raw_hitters |>
  mutate(
    barrel_rate_proxy_wood_adj = safe_pct(barrel_num, bbe_denom),
    chase_rate = safe_pct(chase_count, out_of_zone_count),
    whiff_rate = safe_pct(whiff_count, swing_count),
    contact_rate = safe_pct(contact_count, swing_count),
    avg_ev_wood_adj = if_else(is.nan(avg_ev_wood_adj), NA_real_, avg_ev_wood_adj),
    p90_ev_wood_adj = if_else(is.nan(p90_ev_wood_adj), NA_real_, p90_ev_wood_adj)
  ) |>
  left_join(
    mapping |> rename(team_code = team_code),
    by = "team_code"
  ) |>
  mutate(
    school_name = coalesce(school_name, team_code),
    conference = coalesce(conference, "Unknown"),
    batter_uid = player_uid,
    batter = player_name,
    batter_team = team_code,
    batter_side = bats,
    batter_school_name = school_name,
    batter_conference = conference
  )

# --- Pitchers: build pitcher_season ---
pitcher_season <- raw_pitchers |>
  left_join(
    mapping |> rename(team_code = team_code),
    by = "team_code"
  ) |>
  mutate(
    school_name = coalesce(school_name, team_code),
    conference = coalesce(conference, "Unknown"),
    pitcher_uid = player_uid,
    pitcher = player_name,
    pitcher_team = team_code,
    pitcher_throws = throws,
    pitcher_school_name = school_name,
    pitcher_conference = conference,
    extension = avg_extension,
    avg_spin = avg_spin_rate
  )

# --- Compute trends ---
hitter_history <- hitter_season |>
  group_by(batter_uid) |>
  arrange(season, .by_group = TRUE) |>
  mutate(
    trend_delta = p90_ev_wood_adj - lag(p90_ev_wood_adj),
    one_season_only_flag = n_distinct(season) == 1
  ) |>
  ungroup()

pitcher_history <- pitcher_season |>
  group_by(pitcher_uid) |>
  arrange(season, .by_group = TRUE) |>
  mutate(
    trend_delta = avg_fb_velo - lag(avg_fb_velo),
    one_season_only_flag = n_distinct(season) == 1
  ) |>
  ungroup()

current_season <- max(raw$season, na.rm = TRUE)

# --- Build current_hitters ---
current_hitters <- hitter_history |>
  mutate(
    data_completeness_score = pmax(0, 100 - (missing_critical_count / pmax(event_rows, 1)) * 100),
    player_uid = batter_uid,
    player_name = batter,
    school_name = batter_school_name,
    conference = batter_conference,
    bats = batter_side,
    throws = "Unknown",
    class_year = "Unknown",
    trend_delta = coalesce(trend_delta, 0),
    export_ts = export_ts
  )

# ── Apply sample-size floor before scoring ──
current_hitters <- current_hitters |>
  filter(plate_events >= 25)
message(sprintf("Hitters after 25 PA floor: %d", nrow(current_hitters)))

# ── Replace heuristic scores with population-percentile engine ──
baselines_path <- file.path(dirname(script_dir), "exports", "baselines", "league_baselines.json")
if (exists("load_draft_baselines") && file.exists(baselines_path)) {
  bl <- load_draft_baselines(baselines_path)
  current_hitters <- compute_hitter_scores(current_hitters, bl)
  message("Hitter scores: population-percentile engine (", nrow(current_hitters), " players)")
} else {
  message("WARNING: Baselines not found at ", baselines_path, " — using heuristic fallback")
  current_hitters <- current_hitters |>
    rowwise() |>
    mutate(
      impact_score = mean(c(scale_0_100(p90_ev_wood_adj), scale_0_100(avg_ev_wood_adj),
                            scale_0_100(barrel_rate_proxy_wood_adj)), na.rm = TRUE),
      contact_score = mean(c(scale_0_100(contact_rate), scale_0_100(100 - whiff_rate),
                             scale_0_100(100 - chase_rate)), na.rm = TRUE),
      reach_score = mean(c(impact_score, contact_score, scale_0_100(plate_events),
                           scale_0_100(bbe_count)), na.rm = TRUE),
      risk_score = mean(c(scale_0_100(whiff_rate), scale_0_100(chase_rate),
                          scale_0_100(100 - data_completeness_score),
                          scale_0_100(if_else(one_season_only_flag, 100, 0))), na.rm = TRUE),
      draft_value_score = pmax(0, pmin(100, 0.4 * reach_score + 0.35 * impact_score +
                                       0.25 * contact_score - 0.15 * risk_score))
    ) |>
    ungroup()
}

# --- Build current_pitchers ---
current_pitchers <- pitcher_history |>
  mutate(
    data_completeness_score = pmax(0, 100 - (missing_critical_count / pmax(pitch_count, 1)) * 100),
    player_uid = pitcher_uid,
    player_name = pitcher,
    school_name = pitcher_school_name,
    conference = pitcher_conference,
    throws = pitcher_throws,
    class_year = "Unknown",
    trend_delta = coalesce(trend_delta, 0),
    export_ts = export_ts
  )

# ── Apply sample-size floor before scoring ──
current_pitchers <- current_pitchers |>
  filter(pitch_count >= 50)
message(sprintf("Pitchers after 50 pitch floor: %d", nrow(current_pitchers)))

# ── Replace heuristic scores with population-percentile engine ──
if (exists("load_draft_baselines") && exists("bl") && !is.null(bl$pitchers)) {
  current_pitchers <- compute_pitcher_scores(current_pitchers, bl)
  message("Pitcher scores: population-percentile engine (", nrow(current_pitchers), " players)")
} else {
  message("WARNING: Pitcher baselines not available — using heuristic fallback")
  current_pitchers <- current_pitchers |>
    rowwise() |>
    mutate(
      stuff_score = mean(c(scale_0_100(avg_fb_velo), scale_0_100(max_fb_velo),
                           scale_0_100(avg_ivb), scale_0_100(whiff_pct)), na.rm = TRUE),
      command_score = mean(c(scale_0_100(zone_pct), scale_0_100(csw_pct),
                             scale_0_100(100 - hard_hit_allowed_pct)), na.rm = TRUE),
      reach_score = mean(c(stuff_score, command_score, scale_0_100(arsenal_count),
                           scale_0_100(pitch_count)), na.rm = TRUE),
      risk_score = mean(c(scale_0_100(hard_hit_allowed_pct), scale_0_100(100 - zone_pct),
                          scale_0_100(100 - data_completeness_score),
                          scale_0_100(if_else(one_season_only_flag, 100, 0))), na.rm = TRUE),
      draft_value_score = pmax(0, pmin(100, 0.35 * reach_score + 0.40 * stuff_score +
                                       0.25 * command_score - 0.15 * risk_score))
    ) |>
    ungroup()
}

# --- Unmapped teams ---
unmapped_batter_teams <- current_hitters |>
  distinct(team_code, school_name, conference) |>
  filter(conference == "Unknown")

unmapped_pitcher_teams <- current_pitchers |>
  distinct(team_code, school_name, conference) |>
  filter(conference == "Unknown")

# --- Detail tables ---
hitter_trends <- hitter_history |>
  semi_join(current_hitters |> select(player_uid), by = c("batter_uid" = "player_uid")) |>
  transmute(
    player_uid = batter_uid,
    player_name = batter,
    role = "hitters",
    season,
    `P90 EV (Wood Adj)` = p90_ev_wood_adj,
    `Contact Rate` = contact_rate,
    `Chase Rate` = chase_rate
  ) |>
  pivot_longer(cols = -c(player_uid, player_name, role, season), names_to = "metric_label", values_to = "metric_value") |>
  mutate(metric_key = case_when(
    metric_label == "P90 EV (Wood Adj)" ~ "p90_ev_wood_adj",
    metric_label == "Contact Rate" ~ "contact_rate",
    TRUE ~ "chase_rate"
  )) |>
  select(player_uid, player_name, role, season, metric_key, metric_label, metric_value)

pitcher_trends <- pitcher_history |>
  semi_join(current_pitchers |> select(player_uid), by = c("pitcher_uid" = "player_uid")) |>
  transmute(
    player_uid = pitcher_uid,
    player_name = pitcher,
    role = "pitchers",
    season,
    `Avg FB Velo` = avg_fb_velo,
    `CSW%` = csw_pct,
    `Zone%` = zone_pct
  ) |>
  pivot_longer(cols = -c(player_uid, player_name, role, season), names_to = "metric_label", values_to = "metric_value") |>
  mutate(metric_key = case_when(
    metric_label == "Avg FB Velo" ~ "avg_fb_velo",
    metric_label == "CSW%" ~ "csw_pct",
    TRUE ~ "zone_pct"
  )) |>
  select(player_uid, player_name, role, season, metric_key, metric_label, metric_value)

player_trends <- bind_rows(hitter_trends, pitcher_trends)

# Stub detail tables (require raw pitch data — not available from aggregated bridge)
hitter_bbe_detail <- tibble(
  player_uid = character(), player_name = character(), season = integer(),
  exit_speed = numeric(), angle = numeric(), direction = numeric()
)

pitcher_pitchtype_detail <- tibble(
  player_uid = character(), player_name = character(), season = integer(),
  pitch_type = character(), usage_pct = numeric(), avg_velo = numeric(),
  avg_ivb = numeric(), avg_hb = numeric(), extension = numeric(),
  rel_height = numeric(), rel_side = numeric(), zone_pct = numeric(),
  whiff_pct = numeric(), csw_pct = numeric(), hard_hit_allowed_pct = numeric()
)

if (nrow(player_metadata) > 0) {
  current_hitters <- current_hitters |>
    left_join(player_metadata |> filter(role == "hitter") |> select(player_uid, class_year), by = "player_uid", suffix = c("", "_meta")) |>
    mutate(class_year = coalesce(class_year_meta, class_year)) |>
    select(-class_year_meta)

  current_pitchers <- current_pitchers |>
    left_join(player_metadata |> filter(role == "pitcher") |> select(player_uid, class_year), by = "player_uid", suffix = c("", "_meta")) |>
    mutate(class_year = coalesce(class_year_meta, class_year)) |>
    select(-class_year_meta)
}

if (nrow(current_hitters) > 0) {
  current_hitters <- current_hitters |>
    arrange(desc(draft_value_score), desc(impact_score), desc(contact_score), player_name) |>
    mutate(production_rank = row_number()) |>
    select(
      player_uid, player_name, team_code, school_name, conference, season, bats, throws, class_year,
      production_rank, draft_value_score, reach_score, impact_score, contact_score, risk_score,
      plate_events, bbe_count, p90_ev_wood_adj, avg_ev_wood_adj, barrel_rate_proxy_wood_adj,
      contact_rate, whiff_rate, chase_rate, trend_delta, data_completeness_score,
      one_season_only_flag, missing_critical_count, export_ts
    )
} else {
  message("WARNING: Zero hitters after filtering — exporting empty hitters_board")
  current_hitters <- current_hitters[, character(0), drop = FALSE]
}

if (nrow(current_pitchers) > 0) {
  current_pitchers <- current_pitchers |>
    arrange(desc(draft_value_score), desc(stuff_score), desc(command_score), player_name) |>
    mutate(production_rank = row_number()) |>
    select(
    player_uid, player_name, team_code, school_name, conference, season, throws, class_year,
    production_rank, draft_value_score, reach_score, stuff_score, command_score, risk_score,
    pitch_count, avg_fb_velo, max_fb_velo, avg_ivb, avg_hb, extension, avg_spin, csw_pct,
    whiff_pct, zone_pct, arsenal_count, trend_delta, data_completeness_score,
    one_season_only_flag, missing_critical_count, export_ts
  )
} else {
  message("WARNING: Zero pitchers after filtering — exporting empty pitchers_board")
  current_pitchers <- current_pitchers[, character(0), drop = FALSE]
}

build_benchmarks <- function(board_df, role_name) {
  if (!"conference" %in% names(board_df)) {
    return(tibble(season = integer(), role = character(), metric_key = character(), benchmark_scope = character(), benchmark_value = double(), benchmark_label = character()))
  }
  scoped <- board_df |>
    filter(conference %in% c("ACC", "SEC"))
  if (nrow(scoped) == 0) {
    return(tibble(season = integer(), role = character(), metric_key = character(), benchmark_scope = character(), benchmark_value = double(), benchmark_label = character()))
  }
  metrics <- metric_lookup[[role_name]]
  map_dfr(names(metrics), function(metric_key) {
    scoped |>
      group_by(season) |>
      summarise(
        role = role_name,
        metric_key = metric_key,
        benchmark_scope = "ACC_SEC",
        benchmark_value = mean(.data[[metric_key]], na.rm = TRUE),
        benchmark_label = metrics[[metric_key]],
        .groups = "drop"
      )
  })
}

benchmarks_acc_sec <- bind_rows(
  build_benchmarks(current_hitters, "hitters"),
  build_benchmarks(current_pitchers, "pitchers")
)

build_explanations <- function(board_df, role_name) {
  if (nrow(board_df) == 0) {
    return(tibble(
      player_uid = character(), role = character(), sample_size_text = character(),
      data_completeness_score = double(), match_confidence = double(),
      positive_driver_1 = character(), positive_driver_2 = character(),
      negative_driver_1 = character(), negative_driver_2 = character(), warning_text = character()
    ))
  }

  if (role_name == "hitters") {
    board_df |>
      mutate(
        sample_size_text = paste0(plate_events, " plate events / ", bbe_count, " BBE"),
        match_confidence = 0.95,
        positive_driver_1 = paste0("Impact score ", round(impact_score, 1)),
        positive_driver_2 = paste0("P90 EV wood-adj ", round(p90_ev_wood_adj, 1)),
        negative_driver_1 = paste0("Whiff rate ", round(whiff_rate, 1), "%"),
        negative_driver_2 = paste0("Chase rate ", round(chase_rate, 1), "%"),
        warning_text = case_when(
          one_season_only_flag ~ "One-season-only warning: no prior season trend context available.",
          data_completeness_score < 85 ~ "Data completeness warning: review missing critical fields before trusting the board score.",
          TRUE ~ "No critical reliability warnings."
        ),
        role = "hitters"
      ) |>
      select(player_uid, role, sample_size_text, data_completeness_score, match_confidence,
             positive_driver_1, positive_driver_2, negative_driver_1, negative_driver_2, warning_text)
  } else {
    board_df |>
      mutate(
        sample_size_text = paste0(pitch_count, " pitches / arsenal count ", arsenal_count),
        match_confidence = 0.95,
        positive_driver_1 = paste0("Stuff score ", round(stuff_score, 1)),
        positive_driver_2 = paste0("Avg FB velo ", round(avg_fb_velo, 1)),
        negative_driver_1 = paste0("Risk score ", round(risk_score, 1)),
        negative_driver_2 = paste0("Zone rate ", round(zone_pct, 1), "%"),
        warning_text = case_when(
          one_season_only_flag ~ "One-season-only warning: no prior season trend context available.",
          data_completeness_score < 85 ~ "Data completeness warning: review missing critical fields before trusting the board score.",
          TRUE ~ "No critical reliability warnings."
        ),
        role = "pitchers"
      ) |>
      select(player_uid, role, sample_size_text, data_completeness_score, match_confidence,
             positive_driver_1, positive_driver_2, negative_driver_1, negative_driver_2, warning_text)
  }
}

explanations <- bind_rows(
  build_explanations(current_hitters, "hitters"),
  build_explanations(current_pitchers, "pitchers")
)

build_diagnostics <- function(board_df, role_name, score_columns) {
  if (nrow(board_df) == 0) {
    return(tibble(role = character(), record_type = character(), section = character(), label = character()))
  }
  distributions <- map_dfr(score_columns, function(column_name) {
    values <- board_df[[column_name]]
    bins <- seq(0, 100, by = 20)
    counts <- hist(values, breaks = bins, plot = FALSE)$counts
    tibble(
      role = role_name,
      record_type = "distribution",
      section = "score_distribution",
      label = column_name,
      bucket_label = paste0(head(bins, -1), "-", tail(bins, -1)),
      bucket_start = head(bins, -1),
      bucket_end = tail(bins, -1),
      value_num = counts,
      value_text = NA_character_,
      player_uid = NA_character_,
      player_name = NA_character_
    )
  })
  metrics <- tibble(
    role = role_name,
    record_type = "metric",
    section = "availability",
    label = "Model diagnostics status",
    value_num = NA_real_,
    value_text = "Deterministic component-score export; calibration/ROC can be added once trained model outputs are available.",
    player_uid = NA_character_,
    player_name = NA_character_
  )
  averages <- tibble(
    role = role_name,
    record_type = "metric",
    section = "component_average",
    label = score_columns,
    value_num = map_dbl(score_columns, ~ mean(board_df[[.x]], na.rm = TRUE)),
    value_text = NA_character_,
    player_uid = NA_character_,
    player_name = NA_character_
  )
  high_risk <- board_df |>
    arrange(desc(risk_score), desc(draft_value_score)) |>
    slice_head(n = 3) |>
    transmute(
      role = role_name,
      record_type = "example",
      section = "high_score_high_risk_examples",
      label = "High-risk profile",
      value_num = risk_score,
      value_text = paste0("Draft value ", round(draft_value_score, 1)),
      player_uid,
      player_name
    )
  bind_rows(metrics, averages, distributions, high_risk)
}

diagnostics <- bind_rows(
  build_diagnostics(current_hitters, "hitters", c("draft_value_score", "reach_score", "impact_score", "contact_score", "risk_score")),
  build_diagnostics(current_pitchers, "pitchers", c("draft_value_score", "reach_score", "stuff_score", "command_score", "risk_score"))
)

qa <- bind_rows(
  tibble(record_type = "metric", section = "freshness", label = "Export Timestamp", value_num = NA_real_, value_text = export_ts, role = NA_character_, player_uid = NA_character_, player_name = NA_character_),
  tibble(record_type = "metric", section = "coverage", label = "Current Season", value_num = current_season, value_text = NA_character_, role = NA_character_, player_uid = NA_character_, player_name = NA_character_),
  tibble(record_type = "metric", section = "coverage", label = "Hitters on Board", value_num = nrow(current_hitters), value_text = NA_character_, role = "hitters", player_uid = NA_character_, player_name = NA_character_),
  tibble(record_type = "metric", section = "coverage", label = "Pitchers on Board", value_num = nrow(current_pitchers), value_text = NA_character_, role = "pitchers", player_uid = NA_character_, player_name = NA_character_),
  tibble(record_type = "metric", section = "coverage", label = "ACC/SEC Benchmarks Available", value_num = if_else(nrow(benchmarks_acc_sec) > 0, 1, 0), value_text = if_else(nrow(benchmarks_acc_sec) > 0, "Yes", "No"), role = NA_character_, player_uid = NA_character_, player_name = NA_character_),
  tibble(record_type = "metric", section = "mapping", label = "Unmapped Batter Teams", value_num = nrow(unmapped_batter_teams), value_text = NA_character_, role = NA_character_, player_uid = NA_character_, player_name = NA_character_),
  tibble(record_type = "metric", section = "mapping", label = "Unmapped Pitcher Teams", value_num = nrow(unmapped_pitcher_teams), value_text = NA_character_, role = NA_character_, player_uid = NA_character_, player_name = NA_character_)
)

if (nrow(unmapped_batter_teams) > 0) {
  qa <- bind_rows(
    qa,
    unmapped_batter_teams |>
      transmute(
        record_type = "detail",
        section = "mapping",
        label = "Unmapped Batter Team",
        value_num = NA_real_,
        value_text = team_code,
        role = "hitters",
        player_uid = NA_character_,
        player_name = school_name
      )
  )
}

if (nrow(unmapped_pitcher_teams) > 0) {
  qa <- bind_rows(
    qa,
    unmapped_pitcher_teams |>
      transmute(
        record_type = "detail",
        section = "mapping",
        label = "Unmapped Pitcher Team",
        value_num = NA_real_,
        value_text = team_code,
        role = "pitchers",
        player_uid = NA_character_,
        player_name = school_name
      )
  )
}

write_export_table(current_hitters, output_dir, "hitters_board", arrow_available)
write_export_table(current_pitchers, output_dir, "pitchers_board", arrow_available)
write_export_table(player_trends, output_dir, "player_trends", arrow_available)
write_export_table(hitter_bbe_detail, output_dir, "hitter_bbe_detail", arrow_available)
write_export_table(pitcher_pitchtype_detail, output_dir, "pitcher_pitchtype_detail", arrow_available)
write_export_table(benchmarks_acc_sec, output_dir, "benchmarks_acc_sec", arrow_available)
write_export_table(explanations, output_dir, "explanations", arrow_available)
write_export_table(diagnostics, output_dir, "diagnostics", arrow_available)
write_export_table(qa, output_dir, "qa", arrow_available)

message("Wrote dashboard exports to: ", normalizePath(output_dir))
