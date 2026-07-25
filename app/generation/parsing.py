"""Strict duplicate-aware parsing of untrusted model output."""

import json

from pydantic import TypeAdapter, ValidationError

from app.generation.errors import GenerationOutputInvalidError
from app.generation.models import GenerationDraft

_DRAFT_ADAPTER: TypeAdapter[GenerationDraft] = TypeAdapter(GenerationDraft)


def parse_generation_draft(raw: str) -> GenerationDraft:
    """Parse exactly one JSON object and reject duplicate keys at every level."""
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(value, dict):
            raise GenerationOutputInvalidError
        return _DRAFT_ADAPTER.validate_python(value)
    except GenerationOutputInvalidError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
        raise GenerationOutputInvalidError from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GenerationOutputInvalidError
        result[key] = value
    return result
