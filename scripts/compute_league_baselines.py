#!/usr/bin/env python3
"""
compute_league_baselines.py — Per-player-season aggregates for MLB Draft Model baselines.
Produces ONE consistent CSV schema for both pitchers and hitters.

Usage:
  python3 compute_league_baselines.py --role pitchers [--min-pitches 100]
  python3 compute_league_baselines.py --role hitters  [--min-pa 50]
Output: CSV (stdout)
"""

import duckdb, os, sys, argparse

DB_PATH = os.path.expanduser("~/baseball/db/baseball.duckdb")

def pitcher_baselines(con, min_pitches):
    query = f"""
    WITH fb AS (
        SELECT season, "Pitcher", COALESCE("PitcherId", '') AS pitcher_id,
               "PitcherTeam", "PitcherThrows",
               "RelSpeed", "InducedVertBreak", "HorzBreak", "Extension", "SpinRate"
        FROM pitches
        WHERE "Pitcher" IS NOT NULL
          AND "TaggedPitchType" IN ('Fastball','FourSeamFastBall','TwoSeamFastBall','Sinker')
    ),
    fb_agg AS (
        SELECT season, "Pitcher", pitcher_id, "PitcherTeam", "PitcherThrows",
               AVG("RelSpeed") AS avg_fb_velo, MAX("RelSpeed") AS max_fb_velo,
               AVG("InducedVertBreak") AS avg_ivb, AVG("HorzBreak") AS avg_hb,
               AVG("Extension") AS extension, AVG("SpinRate") AS avg_spin
        FROM fb GROUP BY season, "Pitcher", pitcher_id, "PitcherTeam", "PitcherThrows"
    )
    SELECT
        p.season, p."Pitcher" AS player_name, p."PitcherTeam" AS team_code,
        p."PitcherThrows" AS throws,
        CONCAT('p_', CASE WHEN p.pitcher_id != '' THEN p.pitcher_id
                   ELSE CONCAT(p."PitcherTeam", '_', p."Pitcher") END) AS player_uid,
        'pitchers' AS role,
        p.pitch_count, p.swing_count, p.whiff_count, p.csw_count, p.zone_count,
        p.bbe_count, p.hard_hit_allowed_count, p.missing_critical_count,
        p.pitch_types_seen AS arsenal_depth,
        f.avg_fb_velo, f.max_fb_velo, f.avg_ivb, f.avg_hb, f.extension, f.avg_spin,
        -- Hitter columns (NULL for pitchers)
        NULL::INTEGER AS out_of_zone_count, NULL::INTEGER AS chase_count,
        NULL::INTEGER AS contact_count, NULL::DOUBLE AS avg_ev_wood_adj,
        NULL::DOUBLE AS p90_ev_wood_adj, NULL::INTEGER AS barrel_proxy_count,
        NULL::STRING AS bats
    FROM (
        SELECT season, "Pitcher", COALESCE("PitcherId", '') AS pitcher_id,
               "PitcherTeam", "PitcherThrows",
               COUNT(*) AS pitch_count,
               SUM(CASE WHEN "PitchCall" IN ('StrikeSwinging','InPlay','FoulBallFieldable','FoulBallNotFieldable') THEN 1 ELSE 0 END) AS swing_count,
               SUM(CASE WHEN "PitchCall" = 'StrikeSwinging' THEN 1 ELSE 0 END) AS whiff_count,
               SUM(CASE WHEN "PitchCall" IN ('StrikeCalled','StrikeSwinging') THEN 1 ELSE 0 END) AS csw_count,
               SUM(CASE WHEN "PlateLocSide" BETWEEN -0.83 AND 0.83 AND "PlateLocHeight" BETWEEN 1.5 AND 3.5 THEN 1 ELSE 0 END) AS zone_count,
               SUM(CASE WHEN "PitchCall" = 'InPlay' THEN 1 ELSE 0 END) AS bbe_count,
               SUM(CASE WHEN "PitchCall" = 'InPlay' AND "ExitSpeed" >= 95 THEN 1 ELSE 0 END) AS hard_hit_allowed_count,
               SUM(CASE WHEN "RelSpeed" IS NULL OR "InducedVertBreak" IS NULL OR "HorzBreak" IS NULL OR "Extension" IS NULL THEN 1 ELSE 0 END) AS missing_critical_count,
               COUNT(DISTINCT "TaggedPitchType") AS pitch_types_seen
        FROM pitches
        WHERE "Pitcher" IS NOT NULL AND "TaggedPitchType" IS NOT NULL
        GROUP BY season, "Pitcher", pitcher_id, "PitcherTeam", "PitcherThrows"
        HAVING COUNT(*) >= {min_pitches}
    ) p
    LEFT JOIN fb_agg f ON p.season=f.season AND p."Pitcher"=f."Pitcher" AND p.pitcher_id=f.pitcher_id
    """
    return con.execute(query).fetchdf()

def hitter_baselines(con, min_pa):
    query = f"""
    SELECT
        season, "Batter" AS player_name, "BatterTeam" AS team_code,
        "BatterSide" AS bats,
        CONCAT('h_', COALESCE("BatterId", CONCAT("BatterTeam", '_', "Batter"))) AS player_uid,
        'hitters' AS role,
        -- Pitcher columns (NULL for hitters)
        NULL::INTEGER AS pitch_count, NULL::INTEGER AS csw_count,
        NULL::INTEGER AS zone_count, NULL::INTEGER AS hard_hit_allowed_count,
        NULL::INTEGER AS arsenal_depth,
        NULL::DOUBLE AS avg_fb_velo, NULL::DOUBLE AS max_fb_velo,
        NULL::DOUBLE AS avg_ivb, NULL::DOUBLE AS avg_hb,
        NULL::DOUBLE AS extension, NULL::DOUBLE AS avg_spin,
        NULL::STRING AS throws,
        -- Hitter columns
        COUNT(*) AS event_rows,
        SUM(CASE WHEN "PitchCall" IN ('StrikeSwinging','InPlay','FoulBallFieldable','FoulBallNotFieldable') THEN 1 ELSE 0 END) AS swing_count,
        SUM(CASE WHEN "PitchCall" = 'StrikeSwinging' THEN 1 ELSE 0 END) AS whiff_count,
        SUM(CASE WHEN "PlateLocSide" NOT BETWEEN -0.83 AND 0.83 OR "PlateLocHeight" NOT BETWEEN 1.5 AND 3.5 THEN 1 ELSE 0 END) AS out_of_zone_count,
        SUM(CASE WHEN "PitchCall" IN ('StrikeSwinging','InPlay','FoulBallFieldable','FoulBallNotFieldable')
                  AND ("PlateLocSide" NOT BETWEEN -0.83 AND 0.83 OR "PlateLocHeight" NOT BETWEEN 1.5 AND 3.5) THEN 1 ELSE 0 END) AS chase_count,
        SUM(CASE WHEN "PitchCall" IN ('StrikeSwinging','InPlay','FoulBallFieldable','FoulBallNotFieldable')
                  AND "PitchCall" != 'StrikeSwinging' THEN 1 ELSE 0 END) AS contact_count,
        SUM(CASE WHEN "PitchCall" = 'InPlay' THEN 1 ELSE 0 END) AS bbe_count,
        AVG(CASE WHEN "PitchCall" = 'InPlay' THEN GREATEST("ExitSpeed" - 2.8, 0) END) AS avg_ev_wood_adj,
        QUANTILE_CONT(CASE WHEN "PitchCall" = 'InPlay' THEN GREATEST("ExitSpeed" - 2.8, 0) END, 0.9) AS p90_ev_wood_adj,
        SUM(CASE WHEN "PitchCall" = 'InPlay' AND "ExitSpeed" >= 98 AND "Angle" BETWEEN 26 AND 30 THEN 1 ELSE 0 END) AS barrel_proxy_count,
        SUM(CASE WHEN "ExitSpeed" IS NULL OR "Angle" IS NULL OR "PlateLocHeight" IS NULL OR "PlateLocSide" IS NULL THEN 1 ELSE 0 END) AS missing_critical_count
    FROM pitches
    WHERE "Batter" IS NOT NULL
    GROUP BY season, "Batter", "BatterTeam", "BatterSide", "BatterId"
    HAVING COUNT(*) >= {min_pa}
    """
    return con.execute(query).fetchdf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["pitchers", "hitters"])
    parser.add_argument("--min-pitches", type=int, default=100)
    parser.add_argument("--min-pa", type=int, default=50)
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH, read_only=True)

    if args.role == "pitchers":
        df = pitcher_baselines(con, args.min_pitches)
    else:
        df = hitter_baselines(con, args.min_pa)

    df.to_csv(sys.stdout, index=False)
    con.close()

    print(f"Exported {len(df)} {args.role}", file=sys.stderr)

if __name__ == "__main__":
    main()
