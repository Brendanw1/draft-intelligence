import sqlite3
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/create_representative_sample.py <source_db> <output_db> [table_name] [rows_per_team]",
            file=sys.stderr,
        )
        return 1

    source_db = Path(sys.argv[1])
    output_db = Path(sys.argv[2])
    table_name = sys.argv[3] if len(sys.argv) > 3 else "VTData"
    rows_per_team = int(sys.argv[4]) if len(sys.argv) > 4 else 250

    source = sqlite3.connect(source_db)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(output_db)

    try:
        columns = source.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        if not columns:
            raise RuntimeError(f"Table `{table_name}` not found in {source_db}")

        column_defs = ", ".join(f'"{col["name"]}" {col["type"] or "TEXT"}' for col in columns)
        target.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        target.execute(f'CREATE TABLE "{table_name}" ({column_defs})')

        teams = [
            row[0]
            for row in source.execute(
                f"""
                SELECT DISTINCT team_code
                FROM (
                    SELECT BatterTeam AS team_code FROM {table_name}
                    UNION
                    SELECT PitcherTeam AS team_code FROM {table_name}
                )
                WHERE team_code IS NOT NULL AND TRIM(team_code) <> ''
                ORDER BY team_code
                """
            ).fetchall()
        ]

        selected_rowids = set()
        for team_code in teams:
            rows = source.execute(
                f"""
                SELECT rowid
                FROM {table_name}
                WHERE BatterTeam = ? OR PitcherTeam = ?
                ORDER BY COALESCE(Date, ''), COALESCE(GameID, ''), COALESCE(PAofInning, 0), COALESCE(PitchofPA, 0), COALESCE(PitchNo, 0)
                LIMIT ?
                """,
                (team_code, team_code, rows_per_team),
            ).fetchall()
            selected_rowids.update(row[0] for row in rows)

        if not selected_rowids:
            raise RuntimeError("No rows were selected for the sample database.")

        placeholders = ",".join("?" for _ in selected_rowids)
        selected_columns = ", ".join(f'"{col["name"]}"' for col in columns)
        rows = source.execute(
            f'SELECT {selected_columns} FROM "{table_name}" WHERE rowid IN ({placeholders})',
            tuple(sorted(selected_rowids)),
        ).fetchall()

        insert_placeholders = ",".join("?" for _ in columns)
        target.executemany(
            f'INSERT INTO "{table_name}" VALUES ({insert_placeholders})',
            [tuple(row[col["name"]] for col in columns) for row in rows],
        )
        target.commit()

        selected_team_count = target.execute(
            f"""
            SELECT COUNT(DISTINCT team_code)
            FROM (
                SELECT BatterTeam AS team_code FROM {table_name}
                UNION
                SELECT PitcherTeam AS team_code FROM {table_name}
            )
            WHERE team_code IS NOT NULL AND TRIM(team_code) <> ''
            """
        ).fetchone()[0]

        print(
            f"Created {output_db} with {len(rows)} rows covering {selected_team_count} teams from {source_db}"
        )
        return 0
    finally:
        source.close()
        target.close()


if __name__ == "__main__":
    raise SystemExit(main())
