#!/usr/bin/env Rscript
#
# assess_fg_coverage.R — Check what years of FG D1 data are available
# via the collegebaseball package, and how they map to our draft data.

library(collegebaseball)
library(jsonlite)

years_to_check <- c(2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)

results <- list()

for (yr in years_to_check) {
  cat(sprintf("Checking %d...", yr))

  batters <- tryCatch({
    df <- fg_d1_batters(season = yr, q = "n")
    nrow(df)
  }, error = function(e) NA)

  cat(sprintf(" batters=%s", ifelse(is.na(batters), "FAIL", as.character(batters))))

  pitchers <- tryCatch({
    df <- fg_d1_pitchers(season = yr, q = "n")
    nrow(df)
  }, error = function(e) NA)

  cat(sprintf(" pitchers=%s\n", ifelse(is.na(pitchers), "FAIL", as.character(pitchers))))

  results[[as.character(yr)]] <- list(
    year = yr,
    fg_batters = ifelse(is.na(batters), 0, batters),
    fg_pitchers = ifelse(is.na(pitchers), 0, pitchers)
  )
}

# Check ncaa_stats for a single team across years
cat("\n=== NCAA Stats Coverage (Virginia Tech, batting) ===\n")
teams <- ncaa_teams(years = 2025, divisions = 1)
vt_id <- subset(teams, grepl("Virginia Tech", team_name, ignore.case = TRUE))$team_id[1]
cat(sprintf("VT team_id: %s\n", vt_id))

for (yr in c(2013, 2015, 2018, 2020, 2022, 2025)) {
  n_players <- tryCatch({
    df <- ncaa_stats(team_id = vt_id, year = yr, type = "batting")
    nrow(df)
  }, error = function(e) NA)
  cat(sprintf("  %d: %s batters\n", yr, ifelse(is.na(n_players), "FAIL", as.character(n_players))))
}

# Save results
write_json(results, "data/fangraphs/fg_coverage_check.json", pretty = TRUE)
cat("\nDone. Results saved.\n")
