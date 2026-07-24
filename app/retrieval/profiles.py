"""Canonical deterministic retrieval identities."""

import hashlib
import json

from app.retrieval.models import ModelIdentity, SourceSummary


def canonical_hash(prefix: str, value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def vector_profile_id(
    *,
    index_format_version: str,
    model: ModelIdentity,
    normalization: bool,
    index_type: str,
    metric: str,
) -> str:
    return canonical_hash(
        "vpr",
        {
            "index_format_version": index_format_version,
            "model": model.model_dump(mode="json"),
            "normalization": normalization,
            "index_type": index_type,
            "metric": metric,
        },
    )


def corpus_fingerprint(sources: list[SourceSummary]) -> str:
    return canonical_hash(
        "cor",
        [source.model_dump(mode="json") for source in sources],
    )


def index_profile_id(vector_id: str, corpus_id: str) -> str:
    return canonical_hash(
        "ipr",
        {"vector_profile_id": vector_id, "corpus_fingerprint": corpus_id},
    )
