"""MCP-specific validation composed from shared AeroDocs rules."""

import re
from urllib.parse import unquote_to_bytes

from app.core.identifiers import is_valid_document_id

INVALID_URI_CHARACTERS = frozenset("/\\\u2044\u2215\uff0f\u29f8\ufe68\uff3c")
PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")


class InvalidMcpArgumentsError(ValueError):
    """Raised for safe, expected MCP argument validation failures."""


class McpResultTooLargeError(ValueError):
    """Raised when a serialized MCP tool result exceeds its configured bound."""


def validate_pagination(
    offset: int,
    limit: int | None,
    *,
    default_limit: int,
    maximum_limit: int,
) -> tuple[int, int]:
    """Resolve and validate MCP pagination without silent clamping."""
    resolved = default_limit if limit is None else limit
    if offset < 0 or not 1 <= resolved <= maximum_limit:
        raise InvalidMcpArgumentsError
    return offset, resolved


def validate_document_id(value: str) -> str:
    """Validate one already-decoded opaque document identifier."""
    if not is_valid_document_id(value):
        raise InvalidMcpArgumentsError
    return value


def decode_document_resource_id(value: str) -> str:
    """Strictly percent-decode one resource-template identifier once."""
    index = 0
    while index < len(value):
        if value[index] == "%":
            match = PERCENT_ESCAPE.match(value, index)
            if match is None:
                raise InvalidMcpArgumentsError
            index = match.end()
        else:
            index += 1
    try:
        decoded = unquote_to_bytes(value).decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise InvalidMcpArgumentsError from error
    if (
        "%" in decoded
        or any(character in INVALID_URI_CHARACTERS for character in decoded)
        or decoded in {".", ".."}
        or not is_valid_document_id(decoded)
    ):
        raise InvalidMcpArgumentsError
    return decoded
