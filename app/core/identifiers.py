"""Shared validation for opaque AeroDocs identifiers."""

import re

DOCUMENT_ID_PATTERN = re.compile(r"doc_[0-9a-f]{64}")


def is_valid_document_id(value: str) -> bool:
    """Return whether a value is one complete canonical document identifier."""
    return DOCUMENT_ID_PATTERN.fullmatch(value) is not None
