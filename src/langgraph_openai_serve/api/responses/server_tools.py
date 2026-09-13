"""Translate LGOS-executed tools into native Responses output items."""

from collections.abc import Collection, Iterator, Sequence
from typing import TypeAlias

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import UpdatesStreamPart
from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseCustomToolCallOutputItem,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_function_web_search import ActionSearch

from langgraph_openai_serve.api.responses.service import (
    UnsupportedResponsesOutputError,
    response_function_calls,
)

ServerToolItem: TypeAlias = (
    ResponseCustomToolCall
    | ResponseCustomToolCallOutputItem
    | ResponseFunctionWebSearch
)
_ServerCall: TypeAlias = ResponseCustomToolCall | ResponseFunctionWebSearch


class ServerToolTracker:
    """Correlate root-graph server-tool calls with their ToolMessages."""

    def __init__(self, selected: Collection[str]) -> None:
        self._selected = frozenset(selected)
        self._pending: dict[str, _ServerCall] = {}
        self._completed: set[str] = set()

    def items(self, event: UpdatesStreamPart) -> Iterator[ServerToolItem]:
        """
        Yield public tool items represented by one root graph update.

        Yields:
            Selected calls and graph-produced results.

        """
        if event["ns"]:
            return
        for update in event["data"].values():
            if not isinstance(update, dict):
                continue
            messages = update.get("messages", ())
            if isinstance(messages, BaseMessage):
                messages = (messages,)
            if not isinstance(messages, Sequence):
                continue
            for message in messages:
                if isinstance(message, AIMessage):
                    yield from self._tool_calls(message)
                elif isinstance(message, ToolMessage):
                    item = self._tool_result(message)
                    if item is not None:
                        yield item

    def ensure_complete(self) -> None:
        """Reject a response whose selected call has no graph-produced result."""
        if self._pending:
            msg = "Server tool output contains a call without its executed result."
            raise UnsupportedResponsesOutputError(msg)

    def client_function_calls(
        self, message: AIMessage
    ) -> list[ResponseFunctionToolCall]:
        """Return final function calls that belong to the client."""
        self.ensure_complete()
        calls = response_function_calls(message)
        if any(call.name in self._selected for call in calls):
            msg = "Server tool output contains a call without its executed result."
            raise UnsupportedResponsesOutputError(msg)
        return calls

    def _tool_calls(self, message: AIMessage) -> Iterator[ResponseCustomToolCall]:
        for tool_call in message.tool_calls:
            name = tool_call.get("name")
            if name not in self._selected:
                continue
            call_id = tool_call.get("id")
            arguments = tool_call.get("args")
            if not isinstance(call_id, str) or not call_id:
                msg = "Server tool calls must include a non-empty id."
                raise UnsupportedResponsesOutputError(msg)
            if name == "web_search":
                query = arguments.get("query") if isinstance(arguments, dict) else None
                if not isinstance(query, str) or not query.strip():
                    msg = "Server web_search calls must include a non-empty query."
                    raise UnsupportedResponsesOutputError(msg)
                call = ResponseFunctionWebSearch(
                    id=f"ws_{call_id}",
                    type="web_search_call",
                    action=ActionSearch(type="search", query=query.strip()),
                    status="in_progress",
                )
            else:
                tool_input = (
                    arguments.get("__arg1") if isinstance(arguments, dict) else None
                )
                if not isinstance(tool_input, str):
                    msg = "Server custom tool calls must include string input."
                    raise UnsupportedResponsesOutputError(msg)
                call = ResponseCustomToolCall(
                    id=f"ctc_{call_id}",
                    type="custom_tool_call",
                    status="completed",
                    call_id=call_id,
                    name=name,
                    input=tool_input,
                )
            if call_id in self._pending or call_id in self._completed:
                msg = "Server tool call IDs must be unique within a graph run."
                raise UnsupportedResponsesOutputError(msg)
            self._pending[call_id] = call
            if isinstance(call, ResponseCustomToolCall):
                yield call

    def _tool_result(self, message: ToolMessage) -> ServerToolItem | None:
        call_id = message.tool_call_id
        call = self._pending.pop(call_id, None)
        if call is None:
            if call_id in self._completed:
                msg = "Server tool call IDs must be unique within a graph run."
                raise UnsupportedResponsesOutputError(msg)
            return None
        self._completed.add(call_id)
        if isinstance(call, ResponseCustomToolCall):
            return ResponseCustomToolCallOutputItem(
                id=f"ctco_{call_id}",
                type="custom_tool_call_output",
                call_id=call_id,
                output=_custom_output(message),
                status="completed",
            )
        return call.model_copy(
            update={"status": "failed" if message.status == "error" else "completed"}
        )


def _custom_output(message: ToolMessage) -> str:
    output = message.content
    if isinstance(output, list) and len(output) == 1:
        block = output[0]
        if isinstance(block, dict) and block.get("type") == "custom_tool_call_output":
            output = block.get("output")
    if not isinstance(output, str):
        msg = "Server custom tool output must be a string."
        raise UnsupportedResponsesOutputError(msg)
    return output


__all__ = ["ServerToolItem", "ServerToolTracker"]
