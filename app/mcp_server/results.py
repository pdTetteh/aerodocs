"""Exact MCP tool-result construction and serialization."""

from __future__ import annotations

import json

from mcp.types import CallToolResult, TextContent


def serialize_tool_result(result: CallToolResult) -> str:
    """Serialize a complete tool result exactly as MCP stdio serializes models."""
    return result.model_dump_json(by_alias=True, exclude_none=True)


def safe_tool_error(code: str, message: str) -> CallToolResult:
    """Build a bounded tool error without including rejected result content."""
    return safe_tool_error_payload({"error": {"code": code, "message": message}})


def safe_tool_error_payload(payload: dict[str, object]) -> CallToolResult:
    """Build a tool error from an adapter-controlled public payload."""
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        isError=True,
    )


RESULT_TOO_LARGE_ERROR = safe_tool_error(
    "mcp_result_too_large",
    "The MCP result exceeds its configured limit.",
)
MINIMUM_SAFE_TOOL_RESULT_CHARACTERS = len(serialize_tool_result(RESULT_TOO_LARGE_ERROR))
