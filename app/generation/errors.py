"""Expected grounded-generation failures."""


class GenerationError(Exception):
    """Base exception carrying a safe generation API error."""

    code = "generation_error"
    status_code = 500
    public_message = "The grounded answer request could not be completed."


class LlmModelUnavailableError(GenerationError):
    code = "llm_model_unavailable"
    status_code = 503
    public_message = "The configured local generation model is unavailable."


class GenerationFailedError(GenerationError):
    code = "generation_failed"
    status_code = 502
    public_message = "The local model could not generate an answer."


class GenerationOutputInvalidError(GenerationError):
    code = "generation_output_invalid"
    status_code = 502
    public_message = "The local model returned an invalid structured answer."


class GenerationTimeoutError(GenerationError):
    code = "generation_timeout"
    status_code = 504
    public_message = "The local generation request exceeded its time limit."

    def __init__(
        self,
        stage: str = "generation",
    ) -> None:
        super().__init__()
        self.stage = stage


class AnswerResourceLimitError(GenerationError):
    code = "answer_resource_limit"
    status_code = 413
    public_message = "The answer request exceeds configured resource limits."


class InvalidAnswerRequestError(GenerationError):
    code = "invalid_answer_request"
    status_code = 422
    public_message = "The answer request is invalid."


class AuditPersistenceError(GenerationError):
    code = "audit_persistence_failed"
    status_code = 503
    public_message = "The generation audit record could not be stored safely."
