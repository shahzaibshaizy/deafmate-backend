"""
Translation History Persistence Module for DeafMate.
Stores, retrieves, and manages user sign-to-voice and voice-to-text translation logs in SQLite.
"""

from typing import Dict, List, Optional
from database import get_db_connection


def save_history(user_id: int, sentence: str, mode: str = "Sign to Voice") -> Dict:
    """Saves a completed translation record to the database."""
    clean_sentence = sentence.strip()
    if not clean_sentence:
        raise ValueError("Cannot save empty sentence.")

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO translation_history (user_id, sentence, mode)
            VALUES (?, ?, ?);
        """, (user_id, clean_sentence, mode))
        history_id = cursor.lastrowid
        conn.commit()

        row = cursor.execute("""
            SELECT id, user_id, sentence, mode, created_at
            FROM translation_history
            WHERE id = ?;
        """, (history_id,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def get_user_history(user_id: int, limit: int = 100) -> List[Dict]:
    """Retrieves translation records for a given user ordered from newest to oldest."""
    conn = get_db_connection()
    try:
        rows = conn.execute("""
            SELECT id, user_id, sentence, mode, created_at
            FROM translation_history
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?;
        """, (user_id, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_history_item(user_id: int, history_id: int) -> bool:
    """Deletes a specific history record owned by the authenticated user."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM translation_history
            WHERE id = ? AND user_id = ?;
        """, (history_id, user_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def clear_user_history(user_id: int) -> int:
    """Deletes all history records for a user."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM translation_history WHERE user_id = ?;", (user_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
