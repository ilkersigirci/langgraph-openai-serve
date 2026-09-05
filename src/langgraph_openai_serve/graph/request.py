"""Protocol-neutral inputs exposed to graph adapters."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import JsonValue


@dataclass(frozen=True, slots=True)
class ClientFunctionTool:
    """A client-supplied function available to the graph."""

    name: str
    description: str | None
    parameters: Mapping[str, JsonValue] | None
    strict: bool | None


@dataclass(frozen=True, slots=True)
class NamedFunctionToolChoice:
    """Require one named client-supplied function."""

    name: str


ClientToolChoice: TypeAlias = (
    Literal["none", "auto", "required"] | NamedFunctionToolChoice
)


@dataclass(frozen=True, slots=True)
class GraphRequest:
    """Request data shared by protocol decoders and graph execution."""

    model: str
    metadata: Mapping[str, str]
    user: str | None
    tools: tuple[ClientFunctionTool, ...]
    tool_choice: ClientToolChoice | None
    parallel_tool_calls: bool | None


__all__ = [
    "ClientFunctionTool",
    "ClientToolChoice",
    "GraphRequest",
    "NamedFunctionToolChoice",
]
