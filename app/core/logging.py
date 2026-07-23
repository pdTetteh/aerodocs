"""Structured logging configured with the Python standard library."""

import json
import logging
from datetime import UTC, datetime
from typing import TextIO

from app.core.config import LogLevel

AERODOCS_LOGGER_NAME = "aerodocs"


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=UTC,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class AeroDocsJsonHandler(logging.StreamHandler[TextIO]):
    """JSON stream handler owned by AeroDocs logging configuration."""

    aerodocs_owned = True


def configure_logging(log_level: LogLevel) -> None:
    """Configure the AeroDocs logger without changing root logging state."""
    logger = logging.getLogger(AERODOCS_LOGGER_NAME)
    owned_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, AeroDocsJsonHandler) and handler.aerodocs_owned
    ]

    if owned_handlers:
        handler = owned_handlers[0]
        for duplicate in owned_handlers[1:]:
            logger.removeHandler(duplicate)
            duplicate.close()
    else:
        handler = AeroDocsJsonHandler()
        logger.addHandler(handler)

    handler.setFormatter(JsonFormatter())
    handler.setLevel(log_level)
    logger.setLevel(log_level)
    logger.propagate = False
