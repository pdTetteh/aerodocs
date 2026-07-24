"""Expected retrieval and indexing failures."""


class RetrievalError(Exception):
    """Base exception carrying a safe retrieval API error."""

    code = "retrieval_error"
    status_code = 500
    public_message = "The retrieval request could not be completed."


class IndexNotReadyError(RetrievalError):
    code = "index_not_ready"
    status_code = 404
    public_message = "No active retrieval index is available."


class IndexBuildInProgressError(RetrievalError):
    code = "index_build_in_progress"
    status_code = 409
    public_message = "Another index build is already in progress."


class EmptyCorpusError(RetrievalError):
    code = "empty_corpus"
    status_code = 422
    public_message = "No extractable document chunks are available to index."


class CorruptSourceDocumentError(RetrievalError):
    code = "corrupt_source_document"
    public_message = "A source document failed integrity validation."


class IndexIntegrityError(RetrievalError):
    code = "index_integrity_error"
    public_message = "The retrieval index failed integrity validation."


class IncompatibleIndexProfileError(RetrievalError):
    code = "incompatible_index_profile"
    status_code = 409
    public_message = "The active index is incompatible with current configuration."


class StaleIndexError(RetrievalError):
    code = "stale_index"
    status_code = 409
    public_message = "The document corpus changed; rebuild the retrieval index."


class EmbeddingModelUnavailableError(RetrievalError):
    code = "embedding_model_unavailable"
    status_code = 503
    public_message = "The configured local embedding model is unavailable."


class EmbeddingShapeError(RetrievalError):
    code = "embedding_shape_error"
    public_message = "The embedding provider returned an invalid shape."


class EmbeddingDimensionError(RetrievalError):
    code = "embedding_dimension_error"
    public_message = "The embedding dimension is incompatible with the index."


class NonFiniteEmbeddingError(RetrievalError):
    code = "non_finite_embedding"
    public_message = "The embedding provider returned non-finite values."


class IndexResourceLimitError(RetrievalError):
    code = "index_resource_limit"
    status_code = 413
    public_message = "The corpus exceeds configured index build limits."


class InvalidSearchRequestError(RetrievalError):
    code = "invalid_search_request"
    status_code = 422
    public_message = "The search request is invalid."


class FilteredDocumentNotFoundError(RetrievalError):
    code = "filtered_document_not_found"
    status_code = 404
    public_message = "A requested document is not present in the active index."
