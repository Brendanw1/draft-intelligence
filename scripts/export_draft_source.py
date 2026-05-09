#!/usr/bin/env python3
"""
export_draft_source.py — Extract TrackMan data from DuckDB for the MLB Draft Model R pipeline.
Harmonizes column names to lowercase_snake_case (matching what export_dashboard_data.R expects).
Pipes CSV to stdout so R can read it via readr::read_csv(pipe(...)).

Usage:
  python3 export_draft_source.py [--season 2025] [--min-pitches 100]
"""
import duckdb, os, sys, argparse

DB_PATH = os.path.expanduser("~/baseball/db/baseball.duckdb")

# Column mapping: Parquet/DuckDB → R script expected names (lowercase_snake_case)
COLUMN_MAP = {
    "Date": "date",
    "PitchNo": "pitch_no",
    "PAofInning": "paof_inning",
    "PitchofPA": "pitchof_pa",
    "Pitcher": "pitcher",
    "PitcherId": "pitcher_id",
    "PitcherThrows": "pitcher_throws",
    "PitcherTeam": "pitcher_team",
    "Batter": "batter",
    "BatterId": "batter_id",
    "BatterSide": "batter_side",
    "BatterTeam": "batter_team",
    "Inning": "inning",
    "Outs": "outs",
    "Balls": "balls",
    "Strikes": "strikes",
    "TaggedPitchType": "tagged_pitch_type",
    "PitchCall": "pitch_call",
    "KorBB": "kor_bb",
    "TaggedHitType": "tagged_hit_type",
    "PlayResult": "play_result",
    "RelSpeed": "rel_speed",
    "SpinRate": "spin_rate",
    "RelHeight": "rel_height",
    "RelSide": "rel_side",
    "Extension": "extension",
    "InducedVertBreak": "induced_vert_break",
    "HorzBreak": "horz_break",
    "PlateLocHeight": "plate_loc_height",
    "PlateLocSide": "plate_loc_side",
    "VertApprAngle": "vert_appr_angle",
    "HorzApprAngle": "horz_appr_angle",
    "ExitSpeed": "exit_speed",
    "Angle": "angle",
    "Direction": "direction",
    "Distance": "distance",
    "HangTime": "hang_time",
    "GameID": "game_id",
    "PitchUID": "pitch_uid",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "RunsScored": "runs_scored",
    "OutsOnPlay": "outs_on_play",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, help="Filter to specific season")
    parser.add_argument("--min-pitches", type=int, default=0, help="Min pitches per pitcher to include")
    parser.add_argument("--limit", type=int, help="Max rows (for testing)")
    parser.add_argument("--output", type=str, help="Write to file instead of stdout")
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH, read_only=True)

    # Build SELECT with column aliases
    selects = []
    available = set()
    cols_info = con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='pitches'").fetchall()
    for (col,) in cols_info:
        available.add(col)
    
    # Map TrackMan columns to lowercase_snake_case
    for duck_col, r_name in COLUMN_MAP.items():
        if duck_col in available:
            selects.append(f'"{duck_col}" AS {r_name}')
        else:
            selects.append(f"NULL AS {r_name}")
    
    # Derive top_bottom
    if "BatterTeam" in available and "HomeTeam" in available:
        selects.append(f'CASE WHEN "BatterTeam" = "HomeTeam" THEN \'bottom\' ELSE \'top\' END AS top_bottom')
    else:
        selects.append("NULL AS top_bottom")
    
    # Bearing — not in TrackMan Parquet
    selects.append("NULL AS bearing")

    has_season = "season" in available
    
    where = []
    if args.season and has_season:
        where.append(f"season = {args.season}")
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""
    limit_clause = f"LIMIT {args.limit}" if args.limit else ""
    order_clause = "ORDER BY RANDOM()" if args.limit else ""

    query = f"""
        SELECT {', '.join(selects)}
        FROM pitches
        {where_clause}
        {order_clause}
        {limit_clause}
    """

    output = args.output if args.output else '/dev/stdout'
    con.execute(f"COPY ({query}) TO '{output}' (HEADER, DELIMITER ',')")
    con.close()
    
    if args.output:
        import os
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"Wrote {args.output} ({size_mb:.1f} MB)", file=sys.stderr)

if __name__ == "__main__":
    main()
