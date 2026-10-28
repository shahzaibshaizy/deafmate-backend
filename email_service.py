"""
Email & OTP Verification Service for DeafMate.
Uses Gmail SMTP with an automatic terminal console fallback for 100% reliable FYP presentations.
"""

from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from pathlib import Path
import secrets
import smtplib
from typing import Dict, Optional, Tuple

# Try loading optional .env file if present
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# Configuration
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")
SMTP_APP_PASSWORD = os.environ.get("SMTP_APP_PASSWORD", "")

# In-memory OTP storage: email -> {"code": "123456", "expires_at": datetime}
_otp_store: Dict[str, Dict] = {}

# Master dev bypass code for safe university presentation defense
MASTER_BYPASS_CODE = "000000"


def generate_otp(email: str, expiry_minutes: int = 10) -> str:
    """Generates a secure 6-digit OTP code and records its expiry time."""
    code = f"{secrets.randbelow(900000) + 100000}"
    expires_at = datetime.utcnow() + timedelta(minutes=expiry_minutes)
    
    clean_email = email.strip().lower()
    _otp_store[clean_email] = {
        "code": code,
        "expires_at": expires_at
    }

    # Print prominently to server terminal
    print("\n" + "=" * 60)
    print(f" [EMAIL SERVICE] 6-Digit OTP for '{clean_email}': {code}")
    print(f" [EMAIL SERVICE] Valid for {expiry_minutes} minutes. (Master Bypass: {MASTER_BYPASS_CODE})")
    print("=" * 60 + "\n")

    return code


def send_otp_email(to_email: str, code: str, user_name: str = "User") -> Tuple[bool, str]:
    """
    Sends the 6-digit OTP to the user via SMTP.
    Returns (success: bool, message: str).
    """
    clean_email = to_email.strip().lower()

    if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        msg = "SMTP credentials not configured. Code was logged to server terminal."
        print(f"[EMAIL SERVICE] Notice: {msg}")
        return True, msg

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"{code} is your DeafMate verification code"
        msg["From"] = f"DeafMate App <{SMTP_EMAIL}>"
        msg["To"] = clean_email

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #F8FAFC; margin: 0; padding: 24px; }}
            .card {{ max-width: 480px; margin: 0 auto; background: #FFFFFF; border-radius: 16px; border: 1px solid #E2E8F0; padding: 32px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
            .brand {{ font-size: 24px; font-weight: 800; color: #16A34A; margin-bottom: 8px; }}
            .code-box {{ background-color: #DCFCE7; border-radius: 12px; padding: 18px; text-align: center; margin: 24px 0; }}
            .code {{ font-size: 36px; font-weight: 900; letter-spacing: 8px; color: #16A34A; }}
            .subtext {{ font-size: 13px; color: #64748B; line-height: 1.5; }}
          </style>
        </head>
        <body>
          <div class="card">
            <div class="brand">DeafMate</div>
            <p style="font-size: 16px; color: #1E293B; margin-top: 0;">Hello <strong>{user_name}</strong>,</p>
            <p style="font-size: 14px; color: #475569; line-height: 1.5;">
              Welcome to DeafMate! Please use the following 6-digit verification code to complete your registration:
            </p>
            <div class="code-box">
              <span class="code">{code}</span>
            </div>
            <p class="subtext">
              This code will expire in 10 minutes. If you did not request this, please ignore this email.
            </p>
          </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            server.sendmail(SMTP_EMAIL, clean_email, msg.as_string())

        print(f"[EMAIL SERVICE] Successfully dispatched real SMTP email to: {clean_email}")
        return True, "Email sent successfully."

    except Exception as e:
        error_msg = f"SMTP dispatch failed ({str(e)}). Code logged to console for development."
        print(f"[EMAIL SERVICE] Warning: {error_msg}")
        return False, error_msg


def verify_otp(email: str, input_code: str) -> Tuple[bool, str]:
    """
    Validates the provided OTP.
    Returns (is_valid: bool, reason: str).
    """
    clean_email = email.strip().lower()
    clean_code = input_code.strip()

    # Presentation fail-safe bypass
    if clean_code == MASTER_BYPASS_CODE:
        print(f"[EMAIL SERVICE] Master bypass code accepted for: {clean_email}")
        return True, "Master code verified."

    record = _otp_store.get(clean_email)
    if not record:
        return False, "No verification code was requested for this email."

    if datetime.utcnow() > record["expires_at"]:
        del _otp_store[clean_email]
        return False, "Verification code has expired. Please request a new one."

    if record["code"] != clean_code:
        return False, "Incorrect verification code."

    # Code is valid; consume it
    del _otp_store[clean_email]
    return True, "Code verified successfully."
