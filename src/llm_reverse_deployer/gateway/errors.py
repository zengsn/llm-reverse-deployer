"""Gateway HTTP error type.

All Gateway-side failures are surfaced as OpenAI-style error objects:

    {"error": {"type": "...", "message": "..."}}
"""

from __future__ import annotations


class GatewayError(Exception):
    def __init__(self, status_code: int, error_type: str, message: str) -> None:
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        super().__init__(message)


def status_for_error(error_type: str) -> int:
    """Map a tunnel error type to an HTTP status code."""
    return {
        "model_not_found": 404,
        "invalid_request": 400,
        "request_too_large": 413,
        "worker_unavailable": 503,
        "backend_error": 502,
    }.get(error_type, 502)
