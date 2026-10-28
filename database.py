"""
Database Management Module for DeafMate.
Uses Python's built-in sqlite3 for zero-configuration, robust local SQL persistence.
"""

from pathlib import Path
import sqlite3
from typing import Generator

DB_FILE = Path(__file__).resolve().parent / "deafmate.db"


def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with dict-like row access, timeout lock protection, and foreign keys enabled."""
    conn = sqlite3.connect(str(DB_FILE), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            is_verified INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Index for fast email lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")

    # 2. Translation History table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            sentence TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'Sign to Voice',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)

    # Index for fast history queries per user
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON translation_history(user_id);")

    conn.commit()
    conn.close()
    print(f"[DATABASE] Initialized SQLite database successfully at: {DB_FILE}")


if __name__ == "__main__":
    init_db()
