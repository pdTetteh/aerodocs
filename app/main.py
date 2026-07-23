"""Uvicorn entry point for AeroDocs."""

from app.factory import create_app

app = create_app()
