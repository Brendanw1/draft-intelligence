from pathlib import Path

from mlb_draft_dashboard.config import EXPORT_FILES
from mlb_draft_dashboard.contracts import REQUIRED_COLUMNS, missing_columns, missing_export_files
from mlb_draft_dashboard.data_access import load_dashboard_bundle
from mlb_draft_dashboard.sample_data import write_demo_exports


def test_demo_exports_satisfy_required_contracts(tmp_path: Path) -> None:
    write_demo_exports(tmp_path)
    bundle = load_dashboard_bundle(tmp_path)

    assert missing_export_files(tmp_path) == []
    for table_name, required_columns in REQUIRED_COLUMNS.items():
        frame = bundle[table_name]
        assert frame is not None
        assert missing_columns(frame.columns, required_columns) == []


def test_missing_export_detection(tmp_path: Path) -> None:
    write_demo_exports(tmp_path)
    (tmp_path / EXPORT_FILES["qa"]).unlink()
    assert "qa" in missing_export_files(tmp_path)


def test_csv_fallback_is_detected(tmp_path: Path) -> None:
    write_demo_exports(tmp_path)
    hitters_parquet = tmp_path / EXPORT_FILES["hitters_board"]
    hitters = load_dashboard_bundle(tmp_path)["hitters_board"]
    hitters_parquet.unlink()
    hitters.to_csv(tmp_path / "hitters_board.csv", index=False)
    assert "hitters_board" not in missing_export_files(tmp_path)
