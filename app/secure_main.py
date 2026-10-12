"""Uvicorn entry point for the OIDC-only Phase 8A review application.

Unlike ``app.main``, importing this module fails unless complete, non-placeholder
OIDC settings are present, so a deployment built on it fails closed instead of
silently serving without authentication.
"""

from app.core.config import Settings
from app.factory import create_phase_8a_secure_app

app = create_phase_8a_secure_app(Settings())
