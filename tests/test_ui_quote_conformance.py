"""Cross-language grounding-quotation conformance vectors."""

import json
from pathlib import Path

from app.generation.service import quote_matches_evidence


def test_python_grounding_matches_shared_browser_vectors() -> None:
    fixture = Path(__file__).parent / "fixtures" / "quote_matching.json"
    vectors = json.loads(fixture.read_text(encoding="utf-8"))

    assert vectors
    for vector in vectors:
        assert (
            quote_matches_evidence(vector["quote"], vector["evidence"])
            is vector["expected"]
        ), vector["id"]
