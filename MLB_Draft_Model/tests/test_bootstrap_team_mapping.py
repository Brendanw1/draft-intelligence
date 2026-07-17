import csv
import sqlite3
import subprocess
import sys
from pathlib import Path


def test_bootstrap_team_mapping_from_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "sample.db"
    output_csv = tmp_path / "team_mapping.csv"

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("CREATE TABLE VTData (BatterTeam TEXT, PitcherTeam TEXT)")
        connection.executemany(
            "INSERT INTO VTData (BatterTeam, PitcherTeam) VALUES (?, ?)",
            [("VIR_TEC", "FLO_DAR"), ("Wal_Sen", "VIR_TEC")],
        )
        connection.commit()
    finally:
        connection.close()

    result = subprocess.run(
        [sys.executable, "scripts/bootstrap_team_mapping.py", str(db_path), str(output_csv)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Wrote 3 team codes" in result.stdout
    with output_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["team_code"] for row in rows] == ["FLO_DAR", "VIR_TEC", "Wal_Sen"]
    flo_dar = next(row for row in rows if row["team_code"] == "FLO_DAR")
    wal_sen = next(row for row in rows if row["team_code"] == "Wal_Sen")
    assert flo_dar["school_name"] == "Florida"
    assert flo_dar["conference"] == "SEC"
    assert wal_sen["school_name"] == "Walters State"
    assert wal_sen["conference"] == "JUCO"
