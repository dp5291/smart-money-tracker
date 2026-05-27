# ============================================================
# Smart Money Tracker
# Copyright (c) 2026 Dhruv Patel. All rights reserved.
#
# This software is proprietary and confidential.
# Unauthorized copying, distribution, or modification
# of this file, via any medium, is strictly prohibited.
#
# Author:  Dhruv Patel
# GitHub:  github.com/dhruvpatel29
# Email:   dhruvkumarp79@gmail.com
# ============================================================

"""
api/security.py — Complete security layer.

Covers:
  1. API key authentication   — protects all endpoints
  2. Webhook secret token     — validates TradingView alerts are real
  3. Rate limiting            — prevents abuse / API cost explosions
  4. Security headers         — protects the browser frontend
  5. Input sanitization       — validates tickers and inputs
  6. Request logging          — audit trail of all requests
  7. IP blocking              — ban suspicious IPs automatically
  8. CORS lockdown            — only your frontend can call the API

Add to api/main.py:
  from api.security import (
      setup_security, verify_api_key, verify_webhook_token,
      limiter, API_KEY_HEADER
  )
"""

import os
import time
import hashlib
import hmac
import logging
import secrets
from datetime import datetime
from typing import Optional
from functools import wraps
from collections import defaultdict

from fastapi import Request, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", mode="a"),
    ]
)
logger = logging.getLogger("smart_money")

os.makedirs("logs", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. API KEY AUTHENTICATION
# Every request to your API must include a valid API key.
# Without this, anyone who finds your URL can call your endpoints.
# ═══════════════════════════════════════════════════════════════

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load valid API keys from environment
# In .env: API_KEYS=key1,key2,key3
_RAW_KEYS = os.getenv("API_KEYS", "")
VALID_API_KEYS: set = set(
    k.strip() for k in _RAW_KEYS.split(",") if k.strip()
)

def generate_api_key() -> str:
    """Generate a secure random API key. Run once and add to .env"""
    return secrets.token_urlsafe(32)


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """
    FastAPI dependency — validates the API key on every request.

    Usage in route:
        @app.get("/signal/{ticker}")
        async def get_signal(ticker: str, key: str = Depends(verify_api_key)):
            ...

    How to call from frontend:
        fetch('/signal/AAPL', {
            headers: { 'X-API-Key': 'your-api-key-here' }
        })

    Skip authentication in development:
        Set DEV_MODE=true in .env to bypass (NEVER in production)
    """
    # Allow skipping in development (never in production)
    if os.getenv("DEV_MODE", "false").lower() == "true":
        return "dev"

    if not VALID_API_KEYS:
        logger.warning("No API_KEYS set in .env — all requests allowed (insecure)")
        return "no-key-configured"

    if not api_key:
        logger.warning(f"Request without API key from {request.client.host}")
        raise HTTPException(
            status_code=401,
            detail="API key required. Add header: X-API-Key: your-key",
        )

    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempt from {request.client.host}: {api_key[:8]}...")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key


# ═══════════════════════════════════════════════════════════════
# 2. WEBHOOK SECRET TOKEN
# TradingView doesn't sign its webhooks, so we add a secret token
# to the URL. Only TradingView (with your token) can trigger alerts.
# ═══════════════════════════════════════════════════════════════

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")


def verify_webhook_token(token: Optional[str]) -> bool:
    """
    Validate the webhook secret token.

    In TradingView, set your webhook URL to:
    https://your-server.com/webhook/tradingview?token=YOUR_WEBHOOK_SECRET

    Your WEBHOOK_SECRET in .env should be a random string like:
    WEBHOOK_SECRET=xK9mP2qR8vL5nJ7wY3cE6tA4hF1bD0uZ

    Generate one: python -c "import secrets; print(secrets.token_urlsafe(32))"

    Uses constant-time comparison to prevent timing attacks.
    """
    if not WEBHOOK_SECRET:
        logger.warning("WEBHOOK_SECRET not set — webhook endpoint is unprotected!")
        return True  # Allow if not configured (development only)

    if not token:
        return False

    # secrets.compare_digest prevents timing attacks
    return secrets.compare_digest(token, WEBHOOK_SECRET)


# ═══════════════════════════════════════════════════════════════
# 3. RATE LIMITING
# Limits how many requests each IP can make per minute.
# Without this, someone could spam your API and rack up costs.
# ═══════════════════════════════════════════════════════════════

limiter = Limiter(key_func=get_remote_address)

# Rate limit rules (applied per-route with @limiter.limit decorator):
RATE_LIMITS = {
    "signal":    "20/minute",   # AI signal endpoint (expensive — runs model)
    "backtest":  "5/minute",    # Backtest (very expensive — runs simulation)
    "webhook":   "60/minute",   # Webhook (TradingView can fire often)
    "general":   "60/minute",   # All other endpoints
    "ws":        "10/minute",   # WebSocket connections
}


# ═══════════════════════════════════════════════════════════════
# 4. INPUT VALIDATION & SANITIZATION
# Prevents injection attacks and bad data from reaching your model.
# ═══════════════════════════════════════════════════════════════

# Allowed tickers — only these can be queried
ALLOWED_TICKERS = {
    "AAPL", "NVDA", "TSLA", "BTC-USD", "MSFT", "AMZN",
    "META", "GOOGL", "NFLX", "AMD", "INTC", "SPY", "QQQ",
    "ETH-USD", "SOL-USD", "BNB-USD",
}

def validate_ticker(ticker: str) -> str:
    """
    Validate and sanitize a ticker symbol.
    Prevents injection of arbitrary symbols that could cause errors.

    Usage:
        @app.get("/signal/{ticker}")
        async def get_signal(ticker: str = Depends(validate_ticker)):
    """
    ticker = ticker.upper().strip()

    # Only allow alphanumeric + hyphen (e.g. BTC-USD)
    import re
    if not re.match(r'^[A-Z0-9\-]{1,10}$', ticker):
        raise HTTPException(400, f"Invalid ticker format: {ticker}")

    if ticker not in ALLOWED_TICKERS:
        raise HTTPException(
            404,
            f"Ticker {ticker} not supported. Allowed: {sorted(ALLOWED_TICKERS)}"
        )

    return ticker


def sanitize_webhook_payload(data: dict) -> dict:
    """
    Clean incoming webhook data before processing.
    Prevents malformed data from crashing the model.
    """
    sanitized = {}

    # Ticker: uppercase, strip exchange prefix (NASDAQ:AAPL → AAPL)
    ticker = str(data.get("ticker", "")).upper()
    ticker = ticker.split(":")[-1].strip()  # Remove exchange prefix
    sanitized["ticker"] = ticker

    # Signal: only allow known values
    signal = str(data.get("signal", "neutral")).lower()
    if signal not in {"bullish", "bearish", "neutral", "squeeze"}:
        signal = "neutral"
    sanitized["signal"] = signal

    # Numeric fields: validate range and type
    def safe_float(key: str, min_val: float, max_val: float, default: float) -> float:
        try:
            val = float(data.get(key, default))
            return max(min_val, min(max_val, val))
        except (TypeError, ValueError):
            return default

    sanitized["price"]        = safe_float("price",        0, 1_000_000, 0)
    sanitized["rsi"]          = safe_float("rsi",          0, 100,       50)
    sanitized["macd"]         = safe_float("macd",         -100, 100,    0)
    sanitized["sma200_dist"]  = safe_float("sma200_dist",  -50, 50,      0)
    sanitized["volume_ratio"] = safe_float("volume_ratio", 0, 100,       1)
    sanitized["bb_width"]     = safe_float("bb_width",     0, 100,       5)

    # Boolean fields
    sanitized["golden_cross"] = bool(data.get("golden_cross", False))
    sanitized["source"]       = "tradingview"

    return sanitized


# ═══════════════════════════════════════════════════════════════
# 5. SECURITY HEADERS MIDDLEWARE
# Adds HTTP headers that protect browsers from common attacks:
#   - XSS (cross-site scripting)
#   - Clickjacking
#   - MIME sniffing
#   - Information leakage
# ═══════════════════════════════════════════════════════════════

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to every response.
    These are industry-standard headers that browsers understand.
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Prevent browsers from MIME-sniffing (could execute malicious files as scripts)
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent your app from being embedded in iframes (clickjacking protection)
        response.headers["X-Frame-Options"] = "DENY"

        # Force HTTPS in browsers (uncomment when deployed with HTTPS)
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Control what info the browser sends in Referer header
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Restrict browser features (camera, microphone, etc. — not needed here)
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Content Security Policy — tells browsers which sources are allowed
        # Adjust if you load scripts from CDNs
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://s3.tradingview.com; "
            "frame-src https://www.tradingview.com; "
            "connect-src 'self' ws://localhost:* wss://*.ngrok-free.app;"
        )

        # Hide the server technology stack (don't advertise you're using FastAPI)
        response.headers["Server"] = "SmartMoney/1.0"

        return response


# ═══════════════════════════════════════════════════════════════
# 6. REQUEST LOGGING MIDDLEWARE
# Logs every request with IP, method, path, status, and response time.
# Essential for debugging and spotting abuse.
# ═══════════════════════════════════════════════════════════════

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every incoming request.
    Output goes to logs/app.log and the console.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip  = request.client.host if request.client else "unknown"

        # Process the request
        response = await call_next(request)

        # Calculate response time
        elapsed = (time.time() - start_time) * 1000  # ms

        # Log level based on status code
        status = response.status_code
        if status >= 500:
            log_fn = logger.error
        elif status >= 400:
            log_fn = logger.warning
        else:
            log_fn = logger.info

        log_fn(
            f"{client_ip} | {request.method} {request.url.path} "
            f"| {status} | {elapsed:.1f}ms"
        )

        return response


# ═══════════════════════════════════════════════════════════════
# 7. IP RATE TRACKING & AUTO-BAN
# Tracks suspicious behavior and temporarily blocks abusive IPs.
# ═══════════════════════════════════════════════════════════════

_failed_attempts: dict = defaultdict(list)   # ip → list of timestamps
_banned_ips:      set  = set()


def check_and_ban_ip(ip: str, window_seconds: int = 60, max_failures: int = 10):
    """
    Track failed auth attempts. Ban IP if it fails too many times.
    Resets automatically after window_seconds.
    """
    now = time.time()

    # Remove old attempts outside the window
    _failed_attempts[ip] = [
        t for t in _failed_attempts[ip]
        if now - t < window_seconds
    ]

    # Add this attempt
    _failed_attempts[ip].append(now)

    # Ban if too many failures
    if len(_failed_attempts[ip]) >= max_failures:
        _banned_ips.add(ip)
        logger.warning(f"IP BANNED: {ip} — {len(_failed_attempts[ip])} failures in {window_seconds}s")


class IPBanMiddleware(BaseHTTPMiddleware):
    """Block requests from banned IPs."""
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"

        if ip in _banned_ips:
            logger.warning(f"Blocked banned IP: {ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many failed requests. Try again later."}
            )

        return await call_next(request)


# ═══════════════════════════════════════════════════════════════
# 8. CORS CONFIGURATION
# Controls which websites can call your API.
# In production: only allow your actual frontend domain.
# ═══════════════════════════════════════════════════════════════

def get_cors_origins() -> list:
    """
    Load allowed CORS origins from environment.
    In .env: CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
    """
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000"
    )
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    logger.info(f"CORS allowed origins: {origins}")
    return origins


# ═══════════════════════════════════════════════════════════════
# MASTER SETUP FUNCTION
# Call this once in api/main.py to apply everything.
# ═══════════════════════════════════════════════════════════════

def setup_security(app):
    """
    Apply all security middleware to the FastAPI app.

    Usage in api/main.py:
        from api.security import setup_security
        app = FastAPI(...)
        setup_security(app)

    Order matters — middleware runs in reverse order of registration.
    """
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # IP ban middleware (first line of defense)
    app.add_middleware(IPBanMiddleware)

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS — MUST come after other middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins    = get_cors_origins(),
        allow_credentials= True,
        allow_methods    = ["GET", "POST"],     # Only methods we need
        allow_headers    = ["X-API-Key", "Content-Type", "Authorization"],
    )

    # Trusted hosts (uncomment in production with your real domain)
    # app.add_middleware(
    #     TrustedHostMiddleware,
    #     allowed_hosts=["yourdomain.com", "www.yourdomain.com", "localhost"]
    # )

    logger.info("Security middleware initialized")
    return app


# ── Helper: generate secrets for .env ─────────────────────────

if __name__ == "__main__":
    """
    Run this to generate secure keys for your .env file.

    Usage: python api/security.py
    """
    print("\n" + "="*50)
    print("  Generated secure keys for your .env file")
    print("="*50)
    print(f"\nAPI_KEYS={generate_api_key()}")
    print(f"WEBHOOK_SECRET={secrets.token_urlsafe(32)}")
    print(f"\nAdd these to your .env file.")
    print("Share API_KEYS only with trusted frontend code.")
    print("Keep WEBHOOK_SECRET only in TradingView webhook URL.")
    print("="*50)
