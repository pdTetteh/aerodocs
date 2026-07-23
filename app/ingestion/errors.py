"""Expected document-ingestion failures."""


class DocumentError(Exception):
    """Base exception carrying a safe public API error."""

    code = "document_error"
    status_code = 400
    public_message = "The document request could not be completed."


class InvalidFilenameError(DocumentError):
    """Raised when an upload filename is missing or unsafe."""

    code = "invalid_filename"
    public_message = "The uploaded file must have a safe, non-empty filename."


class UnsupportedMediaTypeError(DocumentError):
    """Raised when an upload is not declared as a PDF."""

    code = "unsupported_media_type"
    status_code = 415
    public_message = "The uploaded file must use the application/pdf media type."


class EmptyUploadError(DocumentError):
    """Raised when no file bytes were uploaded."""

    code = "empty_pdf"
    public_message = "The uploaded PDF is empty."


class UploadTooLargeError(DocumentError):
    """Raised when an upload exceeds the configured byte limit."""

    code = "file_too_large"
    status_code = 413
    public_message = "The uploaded PDF exceeds the configured size limit."


class PdfExpansionLimitError(DocumentError):
    """Raised when PDF pages or extracted text exceed safe limits."""

    code = "pdf_expansion_limit"
    status_code = 413
    public_message = "The PDF exceeds configured extraction safety limits."


class InvalidPdfSignatureError(DocumentError):
    """Raised when an upload does not begin with the PDF signature."""

    code = "invalid_pdf_signature"
    public_message = "The uploaded content does not have a valid PDF signature."


class InvalidPdfError(DocumentError):
    """Raised when PyMuPDF cannot parse a PDF safely."""

    code = "invalid_pdf"
    public_message = "The uploaded content is not a structurally valid PDF."


class EncryptedPdfError(DocumentError):
    """Raised when a PDF is encrypted or password protected."""

    code = "encrypted_pdf"
    public_message = "Encrypted or password-protected PDFs are not supported."


class ZeroPagePdfError(DocumentError):
    """Raised when a PDF contains no pages."""

    code = "zero_page_pdf"
    public_message = "The uploaded PDF must contain at least one page."


class DocumentNotFoundError(DocumentError):
    """Raised when a document identifier is unknown."""

    code = "document_not_found"
    status_code = 404
    public_message = "The requested document was not found."


class StorageError(DocumentError):
    """Raised when stored data cannot be published or trusted."""

    code = "document_storage_error"
    status_code = 500
    public_message = "The document could not be stored or read safely."


class ProcessingConfigurationConflictError(DocumentError):
    """Raised when source bytes exist under an incompatible processing profile."""

    code = "processing_configuration_conflict"
    status_code = 409
    public_message = (
        "The document already exists with a different processing configuration."
    )

    def __init__(self, stored_configuration: dict[str, object]) -> None:
        super().__init__()
        self.public_details = {
            "stored_processing_configuration": stored_configuration,
        }


class InternalDocumentError(DocumentError):
    """Raised after an unexpected ingestion failure has been logged."""

    code = "document_processing_failed"
    status_code = 500
    public_message = "The document could not be processed."
