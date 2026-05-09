#!/usr/bin/env python3
"""
export_draft_source.py — Pre-aggregate TrackMan data in DuckDB and output player-level CSV.
Avoids piping 3M raw rows through a temp file — aggregation happens in SQL, output is ~10K rows.

Usage:
  python3 export_draft_source.py [--season 2025] [--limit N]
"""

import duckdb
import os
import sys
import argparse

DB_PATH = os.path.expanduser("~/baseball/db/baseball.duckdb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, help="Filter to specific season")
    parser.add_argument("--limit", type=int, help="Max raw pitch rows (for testing)")
    parser.add_argument("--output", type=str, help="Write to file instead of stdout")
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH, read_only=True)

    # Build the full query — all aggregation happens in SQL
    # This produces one row per (season, batter_uid) + one row per (season, pitcher_uid)
    where_conditions = []
    if args.season:
        where_conditions.append(f"season = {args.season}")

    where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
    limit_clause = f"LIMIT {args.limit}" if args.limit else ""

    query = f"""
    WITH raw AS (
        SELECT *
        FROM pitches
        {where_clause}
        {f'USING SAMPLE {args.limit} ROWS' if args.limit else ''}
    ),
    derived AS (
        SELECT
            season,
            -- Batter identifiers
            BatterId AS batter_id,
            Batter AS batter,
            BatterTeam AS batter_team,
            BatterSide AS batter_side,
            CASE WHEN BatterId IS NOT NULL AND BatterId != ''
                 THEN 'h_' || BatterId
                 ELSE 'h_' || BatterTeam || '_' || Batter
            END AS batter_uid,
            -- Pitcher identifiers
            PitcherId AS pitcher_id,
            Pitcher AS pitcher,
            PitcherTeam AS pitcher_team,
            PitcherThrows AS pitcher_throws,
            CASE WHEN PitcherId IS NOT NULL AND PitcherId != ''
                 THEN 'p_' || PitcherId
                 ELSE 'p_' || PitcherTeam || '_' || Pitcher
            END AS pitcher_uid,
            -- Derived indicators
            CASE WHEN PitchCall IN ('StrikeSwinging', 'InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable')
                 THEN 1 ELSE 0 END AS did_swing,
            CASE WHEN PitchCall = 'StrikeSwinging' THEN 1 ELSE 0 END AS whiff,
            CASE WHEN PitchCall = 'StrikeCalled' THEN 1 ELSE 0 END AS called_strike,
            CASE WHEN PitchCall = 'InPlay' THEN 1 ELSE 0 END AS ball_in_play,
            CASE WHEN PlateLocSide BETWEEN -0.83 AND 0.83
                  AND PlateLocHeight BETWEEN 1.5 AND 3.5
                 THEN 1 ELSE 0 END AS in_zone,
            CASE WHEN ExitSpeed - 2.8 >= 95 AND PitchCall = 'InPlay' THEN 1 ELSE 0 END AS hard_hit,
            CASE WHEN ExitSpeed - 2.8 >= 98 AND Angle BETWEEN 26 AND 30 AND PitchCall = 'InPlay'
                 THEN 1 ELSE 0 END AS barrel_proxy,
            ExitSpeed - 2.8 AS wood_ev,
            CASE WHEN PitchCall IN ('StrikeCalled', 'StrikeSwinging') THEN 1 ELSE 0 END AS csw_event,
            CASE WHEN PitchCall IN ('StrikeSwinging', 'InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable')
                  AND NOT (PlateLocSide BETWEEN -0.83 AND 0.83
                           AND PlateLocHeight BETWEEN 1.5 AND 3.5)
                 THEN 1 ELSE 0 END AS chase,
            CASE WHEN PitchCall IN ('StrikeSwinging', 'InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable')
                  AND PitchCall != 'StrikeSwinging'
                 THEN 1 ELSE 0 END AS contact,
            -- Pitcher metrics
            RelSpeed AS rel_speed,
            SpinRate AS spin_rate,
            InducedVertBreak AS induced_vert_break,
            HorzBreak AS horz_break,
            Extension AS extension,
            -- Pitcher flags
            hard_hit AS hard_hit_allowed,
            -- Raw values for pitch type
            TaggedPitchType AS tagged_pitch_type,
            -- PA identifier
            GameID || '|' || Inning || '|' ||
                CASE WHEN BatterTeam = HomeTeam THEN 'bottom' ELSE 'top' END || '|' ||
                PAofInning || '|' ||
                CASE WHEN BatterId IS NOT NULL AND BatterId != ''
                     THEN 'h_' || BatterId
                     ELSE 'h_' || BatterTeam || '_' || Batter
                END AS pa_key,
            -- Missing critical data flag
            CASE WHEN ExitSpeed IS NULL OR Angle IS NULL
                  OR PlateLocHeight IS NULL OR PlateLocSide IS NULL
                 THEN 1 ELSE 0 END AS missing_critical,
            -- Plate appearance end (for K/BB tracking)
            KorBB AS kor_bb,
            PitchofPA AS pitchof_pa
        FROM raw
    ),
    -- HITTER aggregations (by season + batter_uid)
    hitter_agg AS (
        SELECT
            season,
            batter_uid,
            MAX(batter) AS batter,
            MAX(batter_team) AS batter_team,
            MAX(batter_side) AS batter_side,
            SUM(did_swing) AS swing_count,
            SUM(1 - in_zone) AS out_of_zone_count,
            SUM(chase) AS chase_count,
            SUM(whiff) AS whiff_count,
            SUM(contact) AS contact_count,
            SUM(ball_in_play) AS bbe_count,
            AVG(CASE WHEN ball_in_play = 1 THEN wood_ev END) AS avg_ev_wood_adj,
            QUANTILE(CASE WHEN ball_in_play = 1 THEN wood_ev END, 0.9) AS p90_ev_wood_adj,
            SUM(barrel_proxy) AS barrel_count,
            SUM(missing_critical) AS missing_critical_count,
            COUNT(*) AS event_rows,
            -- For derived rates (computed downstream)
            SUM(CASE WHEN ball_in_play = 1 THEN 1 ELSE 0 END) AS bbe_denom,
            SUM(CASE WHEN ball_in_play = 1 THEN barrel_proxy ELSE 0 END) AS barrel_num
        FROM derived
        WHERE batter_uid IS NOT NULL
        GROUP BY season, batter_uid
    ),
    -- PLATE APPEARANCES (for PA-level stats per hitter)
    pa_hitter AS (
        SELECT
            season,
            batter_uid,
            COUNT(*) AS plate_events,
            SUM(CASE WHEN kor_bb = 'Walk' AND pitchof_pa = (
                SELECT MAX(p2.pitchof_pa) FROM derived p2
                WHERE p2.pa_key = derived.pa_key
            ) THEN 1 ELSE 0 END) AS walks,
            SUM(CASE WHEN kor_bb = 'Strikeout' AND pitchof_pa = (
                SELECT MAX(p2.pitchof_pa) FROM derived p2
                WHERE p2.pa_key = derived.pa_key
            ) THEN 1 ELSE 0 END) AS strikeouts
        FROM derived
        WHERE batter_uid IS NOT NULL
        GROUP BY season, batter_uid
    ),
    -- PITCHER aggregations (by season + pitcher_uid)
    pitcher_agg AS (
        SELECT
            season,
            pitcher_uid,
            MAX(pitcher) AS pitcher,
            MAX(pitcher_team) AS pitcher_team,
            MAX(pitcher_throws) AS pitcher_throws,
            COUNT(*) AS pitch_count,
            AVG(CASE WHEN tagged_pitch_type IN ('Fastball', 'Sinker') THEN rel_speed END) AS avg_fb_velo,
            MAX(rel_speed) AS max_fb_velo,
            AVG(induced_vert_break) AS avg_ivb,
            AVG(horz_break) AS avg_hb,
            AVG(extension) AS extension,
            AVG(spin_rate) AS avg_spin_rate,
            SUM(whiff) * 100.0 / NULLIF(SUM(did_swing), 0) AS whiff_pct,
            SUM(csw_event) * 100.0 / NULLIF(COUNT(*), 0) AS csw_pct,
            SUM(in_zone) * 100.0 / NULLIF(COUNT(*), 0) AS zone_pct,
            SUM(hard_hit_allowed) * 100.0 / NULLIF(SUM(ball_in_play), 0) AS hard_hit_allowed_pct,
            COUNT(DISTINCT tagged_pitch_type) AS arsenal_count,
            SUM(missing_critical) AS missing_critical_count,
            -- Single-season flag requires multi-season data — computed in R
            0 AS one_season_only_flag
        FROM derived
        WHERE pitcher_uid IS NOT NULL
        GROUP BY season, pitcher_uid
    )
    -- Output union: hitters first, then pitchers
    SELECT
        'hitter' AS player_type,
        season,
        batter_uid AS player_uid,
        batter AS player_name,
        batter_team AS team_code,
        batter_side AS bats,
        NULL AS throws,
        swing_count,
        out_of_zone_count,
        chase_count,
        whiff_count,
        contact_count,
        bbe_count,
        avg_ev_wood_adj,
        p90_ev_wood_adj,
        barrel_count,
        bbe_denom,
        barrel_num,
        missing_critical_count,
        event_rows,
        h.plate_events,
        h.walks,
        h.strikeouts,
        -- Pitcher columns (NULL for hitters)
        NULL AS pitch_count,
        NULL AS avg_fb_velo,
        NULL AS max_fb_velo,
        NULL AS avg_ivb,
        NULL AS avg_hb,
        NULL AS avg_extension,
        NULL AS avg_spin_rate,
        NULL AS whiff_pct,
        NULL AS csw_pct,
        NULL AS zone_pct,
        NULL AS hard_hit_allowed_pct,
        NULL AS arsenal_count,
        NULL AS pitcher_throws,
        NULL AS pitcher_team,
        NULL AS pitcher_name,
        0 AS one_season_only_flag
    FROM hitter_agg ha
    LEFT JOIN pa_hitter h USING (season, batter_uid)
    UNION ALL
    SELECT
        'pitcher' AS player_type,
        season,
        pitcher_uid AS player_uid,
        pitcher AS player_name,
        pitcher_team AS team_code,
        NULL AS bats,
        pitcher_throws AS throws,
        NULL AS swing_count,
        NULL AS out_of_zone_count,
        NULL AS chase_count,
        NULL AS whiff_count,
        NULL AS contact_count,
        NULL AS bbe_count,
        NULL AS avg_ev_wood_adj,
        NULL AS p90_ev_wood_adj,
        NULL AS barrel_count,
        NULL AS bbe_denom,
        NULL AS barrel_num,
        missing_critical_count,
        NULL AS event_rows,
        NULL AS plate_events,
        NULL AS walks,
        NULL AS strikeouts,
        pitch_count,
        avg_fb_velo,
        max_fb_velo,
        avg_ivb,
        avg_hb,
        extension AS avg_extension,
        avg_spin_rate,
        whiff_pct,
        csw_pct,
        zone_pct,
        hard_hit_allowed_pct,
        arsenal_count,
        pitcher_throws,
        pitcher_team,
        pitcher AS pitcher_name,
        one_season_only_flag
    FROM pitcher_agg
    """

    output = args.output if args.output else '/dev/stdout'
    con.execute(f"COPY ({query}) TO '{output}' (HEADER, DELIMITER ',', QUOTE '\"')")
    con.close()

    if args.output:
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"Wrote {args.output} ({size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
