"""Closed authentication, principal, and permission models."""

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class Role(StrEnum):
    """Closed AeroDocs HTTP roles."""

    EVIDENCE_VIEWER = "evidence_viewer"
    DOCUMENT_OPERATOR = "document_operator"


class CredentialSource(StrEnum):
    """Credential sources implemented in Phase 8A."""

    DIRECT_BEARER = "direct_bearer"


class Principal(BaseModel):
    """Identity and effective roles derived from a verified access token."""

    model_config = ConfigDict(frozen=True)

    subject: str
    roles: frozenset[Role]
    credential_source: CredentialSource = CredentialSource.DIRECT_BEARER

    def can_view_evidence(self) -> bool:
        """Return whether this principal has evidence-viewer permissions."""
        return bool(self.roles & {Role.EVIDENCE_VIEWER, Role.DOCUMENT_OPERATOR})

    def can_operate_documents(self) -> bool:
        """Return whether this principal has document-operator permissions."""
        return Role.DOCUMENT_OPERATOR in self.roles


class AccessTokenValidator(Protocol):
    """Injected access-token validation boundary."""

    async def validate(self, token: str) -> Principal: ...


class InvalidCredentialsError(Exception):
    """Signal a fixed, non-disclosing credential rejection."""


class IdentityValidationUnavailableError(Exception):
    """Signal that identity verification cannot currently be completed."""
