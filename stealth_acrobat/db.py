import sqlite3
import threading
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests.db')

_write_lock = threading.Lock()


def _get_conn():
    """Create a new connection per call (thread-safe pattern for SQLite)."""
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the table if it doesn't exist. Called once at app startup."""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_submissions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                submitted_at TEXT NOT NULL,
                raw_text     TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_test(raw_text):
    """Insert a test submission. Thread-safe via _write_lock."""
    now = (datetime.utcnow() + timedelta(hours=7)).isoformat(sep=' ', timespec='seconds')
    with _write_lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO test_submissions (submitted_at, raw_text) VALUES (?, ?)",
                (now, raw_text)
            )
            conn.commit()
        finally:
            conn.close()


def get_all_tests():
    """Return all test submissions ordered by date. No lock needed (WAL allows concurrent reads)."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, submitted_at, raw_text FROM test_submissions ORDER BY id ASC"
        ).fetchall()
        return [(row['id'], row['submitted_at'], row['raw_text']) for row in rows]
    finally:
        conn.close()
