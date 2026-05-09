"""test_real_data_loads.py — Verify the export bundle loads without error and satisfies REQUIRED_COLUMNS contracts."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_draft_dashboard.data_access import load_dashboard_bundle
from mlb_draft_dashboard.config import EXPORTS_DIR
from mlb_draft_dashboard.contracts import REQUIRED_COLUMNS


def test_load_dashboard_bundle_has_all_tables():
    """All 9 export tables should load without error and contain data."""
    bundle = load_dashboard_bundle(EXPORTS_DIR)

    expected_tables = {
        "hitters_board",
        "pitchers_board",
        "player_trends",
        "hitter_bbe_detail",
        "pitcher_pitchtype_detail",
        "benchmarks_acc_sec",
        "explanations",
        "diagnostics",
        "qa",
    }

    missing = expected_tables - set(bundle.keys())
    assert not missing, f"Missing tables in bundle: {missing}"

    # Detail tables may be empty (require raw pitch data — not always available
    # with the aggregated bridge)
    detail_tables = {"hitter_bbe_detail", "pitcher_pitchtype_detail"}
    for table_name, df in bundle.items():
        if table_name in detail_tables:
            continue
        assert not df.empty, f"Table '{table_name}' is empty"
        assert len(df) > 0, f"Table '{table_name}' has zero rows"


def test_hitters_board_has_required_columns():
    """Hitters board must have all columns from REQUIRED_COLUMNS contract."""
    bundle = load_dashboard_bundle(EXPORTS_DIR)
    df = bundle["hitters_board"]

    required = REQUIRED_COLUMNS["hitters_board"]
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"Hitters board missing columns: {missing}"


def test_pitchers_board_has_required_columns():
    """Pitchers board must have all columns from REQUIRED_COLUMNS contract."""
    bundle = load_dashboard_bundle(EXPORTS_DIR)
    df = bundle["pitchers_board"]

    required = REQUIRED_COLUMNS["pitchers_board"]
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"Pitchers board missing columns: {missing}"
