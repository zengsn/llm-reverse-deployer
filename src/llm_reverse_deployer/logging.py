"""Logging setup.

Sensitive values (API keys, worker tokens, vLLM API keys) must never appear in
logs. All log statements in the codebase avoid them by construction; keep it
that way.
"""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
    )
