"""SQLite persistence layer: schema, connections, and row helpers."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from src.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    message    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id  INTEGER NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    action     TEXT NOT NULL,
    reason     TEXT NOT NULL,
    confidence REAL NOT NULL,
    sources    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tickets_user ON tickets(user_id);
CREATE INDEX IF NOT EXISTS idx_decisions_ticket ON decisions(ticket_id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Transactional connection: commits on success, rolls back on error."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# --- users -------------------------------------------------------------

def create_user(email: str, password_hash: str) -> dict[str, Any]:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, password_hash, utc_now()),
        )
        return dict(
            conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        )


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


# --- tickets + decisions ----------------------------------------------

def create_ticket_with_decision(
    user_id: int,
    message: str,
    action: str,
    reason: str,
    confidence: float,
    sources: list[str],
) -> dict[str, Any]:
    """Persist the ticket and its decision in a single transaction."""
    now = utc_now()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO tickets (user_id, message, created_at) VALUES (?, ?, ?)",
            (user_id, message, now),
        )
        ticket_id = cur.lastrowid
        conn.execute(
            """INSERT INTO decisions (ticket_id, action, reason, confidence, sources, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticket_id, action, reason, confidence, json.dumps(sources), now),
        )
    return get_ticket_for_user(ticket_id, user_id)  # type: ignore[return-value]


def list_tickets_for_user(user_id: int) -> list[dict[str, Any]]:
    """Every read is scoped by user_id -- ownership is enforced in SQL."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT t.id, t.message, t.created_at,
                      d.action, d.reason, d.confidence, d.sources
               FROM tickets t
               LEFT JOIN decisions d ON d.ticket_id = t.id
               WHERE t.user_id = ?
               ORDER BY t.id DESC""",
            (user_id,),
        ).fetchall()
    return [_row_to_ticket(row) for row in rows]


def get_ticket_for_user(ticket_id: int, user_id: int) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT t.id, t.message, t.created_at,
                      d.action, d.reason, d.confidence, d.sources
               FROM tickets t
               LEFT JOIN decisions d ON d.ticket_id = t.id
               WHERE t.id = ? AND t.user_id = ?""",
            (ticket_id, user_id),
        ).fetchone()
    return _row_to_ticket(row) if row else None


def _row_to_ticket(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "message": row["message"],
        "created_at": row["created_at"],
        "action": row["action"],
        "reason": row["reason"],
        "confidence": row["confidence"],
        "sources": json.loads(row["sources"]) if row["sources"] else [],
    }
