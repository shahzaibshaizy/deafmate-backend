"""
Authentication & JWT Management Module for DeafMate.
Implements PBKDF2-HMAC-SHA256 password hashing and PyJWT bearer token issuance.
"""

from datetime import datetime, timedelta
import hashlib
import hmac
import os
import secrets
from typing import Dict, Optional

import jwt

from database import get_db_connection

# JWT Configuration
JWT_SECRET = os.environ.get("JWT_SECRET", "deafmate-fyp-secure-key-2026-production-asl")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = 30


# -------------------------------------------------------------
# Password Cryptography (PBKDF2-HMAC-SHA256)
# -------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hashes a password with a cryptographically secure 16-byte random salt."""
    salt = secrets.token_bytes(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{salt.hex()}${pwd_hash.hex()}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """Verifies a plain password against the stored salt$hash string using constant-time comparison."""
    try:
        salt_hex, hash_hex = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
        computed_hash = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(computed_hash, expected_hash)
    except Exception:
        return False


# -------------------------------------------------------------
# JWT Token Operations
# -------------------------------------------------------------
def create_access_token(user_id: int, email: str, name: str) -> str:
    """Creates a signed JWT access token valid for 30 days."""
    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRATION_DAYS)
    payload = {
        "sub": str(user_id),
        "email": email,
        "name": name,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict]:
    """Decodes and validates a signed JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None


# -------------------------------------------------------------
# User Database Operations
# -------------------------------------------------------------
def get_user_by_email(email: str) -> Optional[Dict]:
    """Finds a user by email."""
    conn = get_db_connection()
    try:
        user = conn.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE;", (email.strip(),)).fetchone()
        return dict(user) if user else None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Finds a user by their numeric ID."""
    conn = get_db_connection()
    try:
        user = conn.execute("SELECT id, name, email, is_verified, created_at FROM users WHERE id = ?;", (user_id,)).fetchone()
        return dict(user) if user else None
    finally:
        conn.close()


def create_user(name: str, email: str, password: str) -> Dict:
    """Registers a new user record in the database."""
    clean_email = email.strip().lower()
    clean_name = name.strip()
    pwd_hash = hash_password(password)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, is_verified)
            VALUES (?, ?, ?, 0);
        """, (clean_name, clean_email, pwd_hash))
        user_id = cursor.lastrowid
        conn.commit()
        return {
            "id": user_id,
            "name": clean_name,
            "email": clean_email,
            "is_verified": False
        }
    finally:
        conn.close()


def mark_user_verified(email: str) -> bool:
    """Marks a user's account as email-verified."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET is_verified = 1 WHERE email = ? COLLATE NOCASE;", (email.strip(),))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """Authenticates a user by email and password."""
    user = get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "is_verified": bool(user["is_verified"])
    }
