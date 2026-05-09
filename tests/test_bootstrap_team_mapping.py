import csv
import subprocess
import sys
from pathlib import Path


def test_bootstrap_team_mapping_generates_valid_csv(tmp_path: Path) -> None:
    """Bootstrap script reads from DuckDB and generates comprehensive team mapping."""
    output_csv = tmp_path / "team_mapping.csv"

    result = subprocess.run(
        [sys.executable, "scripts/bootstrap_team_mapping.py", "--output", str(output_csv)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    # Verify output message
    assert "Wrote" in result.stdout
    assert "team codes" in result.stdout

    # Verify file is valid CSV with expected columns
    with output_csv.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) >= 100, f"Expected ≥100 teams, got {len(rows)}"
    assert "team_code" in rows[0]
    assert "school_name" in rows[0]
    assert "conference" in rows[0]

    # Verify no empty school names
    empty_names = [r for r in rows if not r["school_name"].strip()]
    assert len(empty_names) == 0, f"Found {len(empty_names)} rows with empty school_name"

            # Spot-check known mappings — verify school names are populated (not empty)
    team_lookup = {r["team_code"]: r for r in rows}
    assert team_lookup["VIR_TEC"]["school_name"] != ""
    assert team_lookup["LSU_TIG"]["school_name"] != ""
    assert team_lookup["TEX_AGG"]["school_name"] != ""
