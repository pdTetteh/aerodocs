"""Stdio process configuration and runner."""

import logging
import os

from app.core.config import Settings
from app.core.logging import configure_logging
from app.mcp_server.server import create_mcp_server

LOGGER = logging.getLogger("aerodocs.mcp.stdio")


def configure_stdio_process(settings: Settings) -> None:
    """Keep protocol stdout isolated before lazy third-party imports occur."""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    configure_logging(settings.log_level)


def main() -> int:
    """Run the local AeroDocs MCP server over stdio only."""
    try:
        settings = Settings()
        configure_stdio_process(settings)
        server = create_mcp_server(settings)
        server.run(transport=settings.mcp_transport)
    except KeyboardInterrupt:
        return 0
    except Exception:
        LOGGER.exception("AeroDocs MCP server stopped unexpectedly")
        return 1
    return 0
