from enum import StrEnum


class GraphFeature(StrEnum):
    """Features supported by a registered graph."""

    BACKGROUND = "background"
    FILE_INPUTS = "file_inputs"
    INTERRUPTS = "interrupts"
    MCP_TOOLS = "mcp_tools"
