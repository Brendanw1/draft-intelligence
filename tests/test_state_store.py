from pathlib import Path

from mlb_draft_dashboard.state_store import StateStore


def test_player_notes_round_trip(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "app_state.sqlite")
    store.save_player_note("h_001", True, "Sleeper", "Loose, athletic swing.")

    notes = store.get_player_notes()
    assert notes.shape[0] == 1
    assert notes.loc[0, "player_uid"] == "h_001"
    assert notes.loc[0, "role_fit"] == "Sleeper"


def test_saved_views_round_trip(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "app_state.sqlite")
    store.save_view(
        "hitters",
        "Aggressive upside",
        {"reach": 20, "impact": 50, "contact": 20, "risk": 10},
        {"conference": ["ACC"], "favorites_only": True},
    )

    views = store.list_views("hitters")
    payload = store.get_view("hitters", "Aggressive upside")

    assert views == ["Aggressive upside"]
    assert payload is not None
    assert payload["weights"]["impact"] == 50
    assert payload["filters"]["favorites_only"] is True
