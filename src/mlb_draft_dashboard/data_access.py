from __future__ import annotations

from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd

from .config import EXPORT_FILES
from .contracts import resolve_export_path


def read_export(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".csv":
        return pd.read_csv(path)
    with duckdb.connect(database=":memory:") as connection:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def load_dashboard_bundle(exports_dir: Path) -> Dict[str, pd.DataFrame]:
    bundle: Dict[str, pd.DataFrame] = {}
    for key, filename in EXPORT_FILES.items():
        resolved_path = resolve_export_path(exports_dir, filename)
        bundle[key] = read_export(resolved_path) if resolved_path is not None else pd.DataFrame()
    return bundle


def merge_notes(board_df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
    if board_df.empty:
        return board_df.copy()
    merged = board_df.copy()
    if notes_df.empty:
        merged["is_favorite"] = False
        merged["role_fit"] = "Unassigned"
        merged["note_text"] = ""
        merged["has_note"] = False
        return merged

    merged = merged.merge(notes_df, on="player_uid", how="left")
    merged["is_favorite"] = merged["is_favorite"].fillna(False).astype(bool)
    merged["role_fit"] = merged["role_fit"].fillna("Unassigned")
    merged["note_text"] = merged["note_text"].fillna("")
    merged["has_note"] = merged["note_text"].str.strip().ne("")
    return merged
