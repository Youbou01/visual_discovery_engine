"""Small SQLite repository for search history, likes, and relevance feedback."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FeedbackRepository:
    """Persist user interactions locally; replace with managed storage for production."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('like', 'dislike')),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_user_action
                    ON feedback(user_id, action);

                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    query_mode TEXT NOT NULL,
                    query_summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    result_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_history_user_time
                    ON search_history(user_id, created_at DESC);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record_feedback(self, user_id: str, image_path: str, action: str) -> None:
        if action not in {"like", "dislike"}:
            raise ValueError("action must be 'like' or 'dislike'.")
        with self._connect() as connection:
            # A unique latest action per user/image avoids piling up repeated clicks.
            connection.execute(
                "DELETE FROM feedback WHERE user_id = ? AND image_path = ?",
                (user_id, image_path),
            )
            connection.execute(
                "INSERT INTO feedback(user_id, image_path, action, created_at) VALUES (?, ?, ?, ?)",
                (user_id, image_path, action, self._now()),
            )

    def liked_paths(self, user_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT image_path FROM feedback WHERE user_id = ? AND action = 'like'",
                (user_id,),
            ).fetchall()
        return [row["image_path"] for row in rows]

    def disliked_paths(self, user_id: str) -> list[str]:
        """Return paths explicitly marked not relevant by this local profile."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT image_path FROM feedback WHERE user_id = ? AND action = 'dislike'",
                (user_id,),
            ).fetchall()
        return [row["image_path"] for row in rows]

    def record_search(
        self, user_id: str, query_mode: str, query_summary: str, result_count: int
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO search_history
                   (user_id, query_mode, query_summary, created_at, result_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, query_mode, query_summary[:500], self._now(), int(result_count)),
            )

    def recent_searches(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT query_mode, query_summary, created_at, result_count
                   FROM search_history WHERE user_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (user_id, max(1, min(limit, 100))),
            ).fetchall()
        return [dict(row) for row in rows]
