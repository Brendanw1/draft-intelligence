import sqlite3
import subprocess
import sys
from pathlib import Path


def test_representative_sample_covers_all_teams(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    output_db = tmp_path / "sample.db"

    connection = sqlite3.connect(source_db)
    try:
        connection.execute(
            """
            CREATE TABLE VTData (
                Date TEXT,
                GameID TEXT,
                PAofInning INTEGER,
                PitchofPA INTEGER,
                PitchNo INTEGER,
                BatterTeam TEXT,
                PitcherTeam TEXT
            )
            """
        )
        rows = []
        for index, team in enumerate(["AAA", "BBB", "CCC"], start=1):
            for pitch_no in range(1, 6):
                rows.append((f"2026-02-0{index}", f"G{index}", 1, 1, pitch_no, team, "AAA"))
        connection.executemany("INSERT INTO VTData VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        connection.commit()
    finally:
        connection.close()

    result = subprocess.run(
        [sys.executable, "scripts/create_representative_sample.py", str(source_db), str(output_db), "VTData", "2"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "covering 3 teams" in result.stdout

    connection = sqlite3.connect(output_db)
    try:
        team_count = connection.execute(
            """
            SELECT COUNT(DISTINCT team_code)
            FROM (
                SELECT BatterTeam AS team_code FROM VTData
                UNION
                SELECT PitcherTeam AS team_code FROM VTData
            )
            """
        ).fetchone()[0]
        row_count = connection.execute("SELECT COUNT(*) FROM VTData").fetchone()[0]
    finally:
        connection.close()

    assert team_count == 3
    assert row_count > 0
