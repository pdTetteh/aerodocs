"""Expected work-order failures with safe public messages."""


class WorkOrderError(Exception):
    """Base exception carrying a safe work-order API error."""

    code = "work_order_error"
    status_code = 500
    public_message = "The work-order request could not be completed."


class WorkOrderNotFoundError(WorkOrderError):
    code = "work_order_not_found"
    status_code = 404
    public_message = "The requested work-order draft was not found."


class InvalidDefectReportError(WorkOrderError):
    code = "invalid_defect_report"
    status_code = 422
    public_message = "The defect report is empty or invalid."


class WorkOrderAlreadyReviewedError(WorkOrderError):
    code = "work_order_already_reviewed"
    status_code = 409
    public_message = "This work-order draft has already been reviewed by a human."


class AcknowledgementRequiredError(WorkOrderError):
    code = "acknowledgement_required"
    status_code = 422
    public_message = (
        "Confirming a draft requires explicit acknowledgement that a qualified "
        "human reviewed it against the controlling documentation."
    )


class ReviewerIdentityError(WorkOrderError):
    code = "reviewer_identity_invalid"
    status_code = 422
    public_message = (
        "A reviewer name is required when authentication is disabled and is not "
        "accepted when the reviewer is identified by a verified token."
    )


class SeparateReviewerRequiredError(WorkOrderError):
    code = "separate_reviewer_required"
    status_code = 403
    public_message = "The reviewer must be a different person from the drafter."


class WorkOrderIntegrityError(WorkOrderError):
    code = "work_order_integrity_failure"
    status_code = 500
    public_message = "The stored work-order draft failed integrity validation."


class WorkOrderPersistenceError(WorkOrderError):
    code = "work_order_storage_unavailable"
    status_code = 503
    public_message = "The work-order draft could not be stored or read."


class WorkOrderResourceLimitError(WorkOrderError):
    code = "work_order_resource_limit"
    status_code = 413
    public_message = "The work-order store exceeds its configured listing limit."
