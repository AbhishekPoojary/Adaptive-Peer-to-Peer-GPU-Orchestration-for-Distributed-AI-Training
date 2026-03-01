"""
utils/middleware.py
--------------------
ASGI middleware for the orchestrator.

RequestLoggingMiddleware
  - Generates a unique X-Request-ID for every request (or uses the one supplied by the caller).
  - Logs: method, path, status code, and wall-clock duration.
  - Injects X-Request-ID into the response headers so callers can correlate logs.
"""

import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("orchestrator.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every inbound HTTP request with timing and request ID."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        response: Response = await call_next(request)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "%s %s → %d  (%.1f ms)  rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        response.headers["X-Request-ID"] = request_id
        return response
