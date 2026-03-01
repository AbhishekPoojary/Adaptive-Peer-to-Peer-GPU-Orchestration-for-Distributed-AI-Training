"""
auth/dependencies.py
---------------------
FastAPI security dependencies for API key authentication.

Usage:
    from auth.dependencies import require_api_key

    @app.post("/submit_job", dependencies=[Depends(require_api_key)])
    def submit_job(...): ...

The API key is read from the X-API-Key request header and validated
against the ORCHESTRATOR_API_KEY environment variable (via settings).
"""

import secrets
import logging
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from utils.settings import get_settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.

    Returns the key on success; raises HTTP 401 on failure.
    Uses constant-time comparison to prevent timing attacks.
    """
    settings = get_settings()
    expected = settings.orchestrator_api_key

    if not api_key:
        logger.warning("Request rejected: missing X-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not secrets.compare_digest(api_key.encode(), expected.encode()):
        logger.warning("Request rejected: invalid API key (first 4 chars: %s...)", api_key[:4])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key
