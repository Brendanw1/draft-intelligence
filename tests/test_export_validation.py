from pathlib import Path

import pandas as pd

from mlb_draft_dashboard.export_validation import validate_bundle, validate_exports_dir
from mlb_draft_dashboard.sample_data import write_demo_exports


def test_validate_demo_bundle_has_no_errors(tmp_path: Path) -> None:
    write_demo_exports(tmp_path)
    issues = validate_exports_dir(tmp_path)
    assert [issue for issue in issues if issue.level == "error"] == []


def test_validate_bundle_catches_duplicate_player_ids(tmp_path: Path) -> None:
    write_demo_exports(tmp_path)
    hitters_path = tmp_path / "hitters_board.parquet"
    hitters = pd.read_parquet(hitters_path)
    duplicate = pd.concat([hitters, hitters.iloc[[0]]], ignore_index=True)
    duplicate.to_parquet(hitters_path, index=False)

    issues = validate_exports_dir(tmp_path)
    messages = [issue.message for issue in issues]
    assert any("Duplicate player_uid rows" in message for message in messages)
