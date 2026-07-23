"""Tests for structured logging."""

import json
import logging
from collections.abc import Iterator
from io import StringIO

import pytest

from app.core.config import Settings
from app.core.logging import (
    AERODOCS_LOGGER_NAME,
    AeroDocsJsonHandler,
    JsonFormatter,
    configure_logging,
)


@pytest.fixture
def preserved_logging_state() -> Iterator[None]:
    """Restore all logger state modified by a test."""
    root_logger = logging.getLogger()
    aerodocs_logger = logging.getLogger(AERODOCS_LOGGER_NAME)
    root_state = (
        list(root_logger.handlers),
        list(root_logger.filters),
        root_logger.level,
        root_logger.propagate,
        root_logger.disabled,
    )
    aerodocs_state = (
        list(aerodocs_logger.handlers),
        list(aerodocs_logger.filters),
        aerodocs_logger.level,
        aerodocs_logger.propagate,
        aerodocs_logger.disabled,
    )
    handler_states = {
        handler: (handler.level, handler.formatter, list(handler.filters))
        for handler in {*root_state[0], *aerodocs_state[0]}
    }

    try:
        yield
    finally:
        original_handlers = {*root_state[0], *aerodocs_state[0]}
        for logger in (root_logger, aerodocs_logger):
            for handler in logger.handlers:
                if handler not in original_handlers:
                    handler.close()

        root_logger.handlers[:] = root_state[0]
        root_logger.filters[:] = root_state[1]
        root_logger.setLevel(root_state[2])
        root_logger.propagate = root_state[3]
        root_logger.disabled = root_state[4]

        aerodocs_logger.handlers[:] = aerodocs_state[0]
        aerodocs_logger.filters[:] = aerodocs_state[1]
        aerodocs_logger.setLevel(aerodocs_state[2])
        aerodocs_logger.propagate = aerodocs_state[3]
        aerodocs_logger.disabled = aerodocs_state[4]

        for handler, state in handler_states.items():
            handler.setLevel(state[0])
            handler.setFormatter(state[1])
            handler.filters[:] = state[2]


def test_json_formatter_emits_structured_record() -> None:
    record = logging.LogRecord(
        name="aerodocs.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="health check",
        args=(),
        exc_info=None,
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "aerodocs.test"
    assert payload["message"] == "health check"
    assert isinstance(payload["timestamp"], str)


def test_configure_logging_is_non_destructive_and_idempotent(
    preserved_logging_state: None,
) -> None:
    root_logger = logging.getLogger()
    unrelated_handler = logging.StreamHandler(StringIO())
    root_logger.addHandler(unrelated_handler)

    configure_logging("INFO")
    configure_logging("DEBUG")

    aerodocs_logger = logging.getLogger(AERODOCS_LOGGER_NAME)
    owned_handlers = [
        handler
        for handler in aerodocs_logger.handlers
        if isinstance(handler, AeroDocsJsonHandler)
    ]
    assert unrelated_handler in root_logger.handlers
    assert len(owned_handlers) == 1
    assert aerodocs_logger.level == logging.DEBUG
    assert owned_handlers[0].level == logging.DEBUG
    assert aerodocs_logger.propagate is False


def test_repeated_create_app_calls_do_not_duplicate_logging_handlers(
    preserved_logging_state: None,
    default_settings: Settings,
) -> None:
    from app.factory import create_app

    create_app(default_settings)
    create_app(default_settings)

    aerodocs_logger = logging.getLogger(AERODOCS_LOGGER_NAME)
    owned_handlers = [
        handler
        for handler in aerodocs_logger.handlers
        if isinstance(handler, AeroDocsJsonHandler)
    ]
    assert len(owned_handlers) == 1
