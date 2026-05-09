from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class StateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS player_notes (
                    player_uid TEXT PRIMARY KEY,
                    is_favorite INTEGER NOT NULL DEFAULT 0,
                    role_fit TEXT NOT NULL DEFAULT 'Unassigned',
                    note_text TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS saved_views (
                    board_type TEXT NOT NULL,
                    view_name TEXT NOT NULL,
                    weights_json TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (board_type, view_name)
                )
                """
            )

    def get_player_notes(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query("SELECT * FROM player_notes", connection)

    def save_player_note(
        self,
        player_uid: str,
        is_favorite: bool,
        role_fit: str,
        note_text: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO player_notes (player_uid, is_favorite, role_fit, note_text, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(player_uid) DO UPDATE SET
                    is_favorite = excluded.is_favorite,
                    role_fit = excluded.role_fit,
                    note_text = excluded.note_text,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (player_uid, int(is_favorite), role_fit, note_text),
            )

    def save_view(
        self,
        board_type: str,
        view_name: str,
        weights: Dict[str, float],
        filters: Dict[str, object],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO saved_views (board_type, view_name, weights_json, filters_json, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(board_type, view_name) DO UPDATE SET
                    weights_json = excluded.weights_json,
                    filters_json = excluded.filters_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (board_type, view_name, json.dumps(weights), json.dumps(filters)),
            )

    def list_views(self, board_type: str) -> List[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT view_name FROM saved_views WHERE board_type = ? ORDER BY updated_at DESC, view_name ASC",
                (board_type,),
            ).fetchall()
        return [row["view_name"] for row in rows]

    def get_view(self, board_type: str, view_name: str) -> Optional[Dict[str, object]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT board_type, view_name, weights_json, filters_json, updated_at
                FROM saved_views
                WHERE board_type = ? AND view_name = ?
                """,
                (board_type, view_name),
            ).fetchone()
        if row is None:
            return None
        return {
            "board_type": row["board_type"],
            "view_name": row["view_name"],
            "weights": json.loads(row["weights_json"]),
            "filters": json.loads(row["filters_json"]),
            "updated_at": row["updated_at"],
        }
