#!/usr/bin/env Rscript
#
# scrape_d1_rosters.R — Scrape NCAA Division I baseball rosters
#
# Uses the collegebaseball R package (robert-frey/collegebaseball) to
# scrape all D1 team rosters for a given season, extracting player bio
# data: height, position, class year, bats/throws, hometown, conference.
#
# Output: data/rosters/d1_rosters_{year}.json
#
# Usage:
#   Rscript scripts/scrape_d1_rosters.R [year]
#   Default year: 2026

library(jsonlite)
library(dplyr)
library(collegebaseball)

year <- ifelse(length(commandArgs(trailingOnly = TRUE)) >= 1,
               as.integer(commandArgs(trailingOnly = TRUE)[1]),
               2026)

message("Scraping D1 baseball rosters for ", year, "...")

# Get all D1 teams for the given year
teams <- ncaa_teams(year = year, division = 1)
message("Found ", nrow(teams), " D1 teams")

# Scrape roster for each team
all_rosters <- list()
failed_teams <- c()

for (i in seq_len(nrow(teams))) {
  team <- teams[i, ]
  team_name <- team$school
  team_id <- team$team_id
  conference <- team$conference

  tryCatch({
    roster <- ncaa_roster(team_id = team_id, year = year)
    if (nrow(roster) > 0) {
      roster$team_name <- team_name
      roster$team_id <- team_id
      roster$conference <- conference
      roster$year <- year
      all_rosters[[length(all_rosters) + 1]] <- roster
    }
    if (i %% 25 == 0) {
      message(sprintf("  Processed %d/%d teams (%d players so far)...",
                      i, nrow(teams), sum(sapply(all_rosters, nrow))))
    }
  }, error = function(e) {
    failed_teams <<- c(failed_teams, team_name)
    message(sprintf("  Failed: %s (ID: %d) — %s", team_name, team_id,
                    conditionMessage(e)))
  })
}

if (length(all_rosters) == 0) {
  stop("No rosters scraped successfully")
}

rosters_df <- bind_rows(all_rosters)
message(sprintf("\nScraped %d players from %d teams", nrow(rosters_df), length(all_rosters)))
if (length(failed_teams) > 0) {
  message(sprintf("Failed teams (%d): %s", length(failed_teams),
                  paste(failed_teams, collapse = ", ")))
}

# Write output
output_dir <- "data/rosters"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
output_path <- file.path(output_dir, sprintf("d1_rosters_%d.json", year))

write_json(rosters_df, output_path, pretty = TRUE)
message(sprintf("Wrote %d players to %s", nrow(rosters_df), output_path))
