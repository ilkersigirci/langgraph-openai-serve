"""Translate LGOS-executed tools into native Responses output items."""

from collections.abc import Collection, Iterator, Sequence
from typing import TypeAlias, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import UpdatesStreamPart
from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseCustomToolCallItem,
    ResponseCustomToolCallOutputItem,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_function_web_search import ActionSearch

from langgraph_openai_serve.api.responses.schemas import (
    ResponseCreateRequest,
    ResponseCustomToolCallInput,
    ResponseWebSearchCallInput,
)
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
_Item = TypeVar("_Item")


def update_messages(event: UpdatesStreamPart) -> Iterator[BaseMessage]:
    """
    Yield messages written by nodes in one native LangGraph update.

    Yields:
        Messages carried by the update.

    """
    for update in event["data"].values():
        if not isinstance(update, dict):
            continue
        messages = update.get("messages", ())
        if isinstance(messages, BaseMessage):
            yield messages
        elif isinstance(messages, Sequence):
            yield from (
                message for message in messages if isinstance(message, BaseMessage)
            )


class ServerToolTracker:
    """Correlate selected custom and web-search calls with graph results."""

    def __init__(
        self, request: ResponseCreateRequest, selected: Collection[str]
    ) -> None:
        self._selected = frozenset(selected)
        history = request.input if isinstance(request.input, list) else ()
        self._history_call_ids = {
            item.call_id
            for item in history
            if isinstance(item, ResponseCustomToolCallInput)
        }
        self._history_item_ids = {
            item.id for item in history if isinstance(item, ResponseWebSearchCallInput)
        }
        self._calls: dict[str, _ServerCall] = {}
        self._results: dict[str, ServerToolItem] = {}

    def items(self, message: BaseMessage) -> Iterator[ServerToolItem]:
        """
        Yield new public tool items represented by a completed message.

        Yields:
            Previously unseen calls and graph-produced results.

        """
        if isinstance(message, AIMessage):
            yield from self._custom_calls(message)
            if "web_search" in self._selected:
                self._local_search_calls(message)
                yield from self._provider_search(message)
        elif isinstance(message, ToolMessage):
            item = self._tool_result(message)
            if item is not None:
                yield item

    def ensure_complete(self) -> None:
        """Reject a response whose selected call has no graph-produced result."""
        if self._calls.keys() - self._results.keys():
            msg = "Server tool output contains a call without its executed result."
            raise UnsupportedResponsesOutputError(msg)

    def client_function_calls(
        self, message: AIMessage
    ) -> list[ResponseFunctionToolCall]:
        """Return final function calls that belong to the client."""
        self.ensure_complete()
        calls = response_function_calls(message)
        custom_call_ids = (
            {
                block.get("call_id")
                for block in message.content
                if isinstance(block, dict) and block.get("type") == "custom_tool_call"
            }
            if isinstance(message.content, list)
            else set()
        )
        calls = [call for call in calls if call.call_id not in custom_call_ids]
        if any(call.name in self._selected for call in calls):
            msg = "Server tool output contains a call without its executed result."
            raise UnsupportedResponsesOutputError(msg)
        return calls

    def _custom_calls(self, message: AIMessage) -> Iterator[ResponseCustomToolCall]:
        if not isinstance(message.content, list):
            return
        for block in message.content:
            if not isinstance(block, dict) or block.get("type") != "custom_tool_call":
                continue
            if block.get("name") not in self._selected:
                continue
            call = _custom_call(block)
            if call.call_id in self._history_call_ids:
                continue
            if _remember_once(self._calls, call.call_id, call):
                yield call

    def _local_search_calls(self, message: AIMessage) -> None:
        for tool_call in message.tool_calls:
            if tool_call.get("name") != "web_search":
                continue
            call_id = tool_call.get("id")
            arguments = tool_call.get("args")
            query = arguments.get("query") if isinstance(arguments, dict) else None
            if not isinstance(call_id, str) or not call_id:
                msg = "Server web_search calls must include a non-empty id."
                raise UnsupportedResponsesOutputError(msg)
            if not isinstance(query, str) or not query.strip():
                msg = "Server web_search calls must include a non-empty query."
                raise UnsupportedResponsesOutputError(msg)
            call = ResponseFunctionWebSearch(
                id=f"ws_{call_id}",
                type="web_search_call",
                action=ActionSearch(type="search", query=query.strip()),
                status="in_progress",
            )
            if call.id not in self._history_item_ids:
                _remember_once(self._calls, call_id, call)

    def _provider_search(self, message: AIMessage) -> Iterator[ServerToolItem]:
        for block in message.content_blocks:
            if block["type"] == "server_tool_call" and block["name"] == "web_search":
                call = ResponseFunctionWebSearch.model_validate(
                    {
                        "id": block["id"],
                        "type": "web_search_call",
                        "action": block["args"],
                        "status": "in_progress",
                    }
                )
                if call.id not in self._history_item_ids:
                    _remember_once(self._calls, call.id, call)
            elif block["type"] == "server_tool_result":
                call_id = block["tool_call_id"]
                if not isinstance(self._calls.get(call_id), ResponseFunctionWebSearch):
                    continue
                status = block.get("status")
                if status not in {"success", "error"}:
                    msg = "Server web_search results must include a terminal status."
                    raise UnsupportedResponsesOutputError(msg)
                item = self._search_result(call_id, failed=status == "error")
                if item is not None:
                    yield item

    def _tool_result(self, message: ToolMessage) -> ServerToolItem | None:
        call_id = message.tool_call_id
        call = self._calls.get(call_id)
        if isinstance(call, ResponseCustomToolCall):
            item = ResponseCustomToolCallOutputItem(
                id=f"ctco_{call_id}",
                type="custom_tool_call_output",
                call_id=call_id,
                output=_custom_output(message),
                status="completed",
            )
            return item if _remember_once(self._results, call_id, item) else None
        return self._search_result(call_id, failed=message.status == "error")

    def _search_result(
        self, call_id: str, *, failed: bool
    ) -> ResponseFunctionWebSearch | None:
        call = self._calls.get(call_id)
        if not isinstance(call, ResponseFunctionWebSearch):
            return None
        item = call.model_copy(update={"status": "failed" if failed else "completed"})
        return item if _remember_once(self._results, call_id, item) else None


def _custom_call(block: dict) -> ResponseCustomToolCall:
    """Validate a LangChain custom call after dropping chunk-only fields."""
    item = ResponseCustomToolCallItem.model_validate(
        {
            key: value
            for key, value in block.items()
            if key in ResponseCustomToolCallItem.model_fields
        }
    )
    # ResponseOutputItem uses the base SDK model, whose extra fields include the
    # status required on a full output item.
    return ResponseCustomToolCall.model_validate(item.model_dump())


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


def _remember_once(items: dict[str, _Item], key: str, item: _Item) -> bool:
    """Deduplicate parent/subgraph updates, rejecting conflicting activity."""
    previous = items.get(key)
    if previous is None:
        items[key] = item
        return True
    if previous != item:
        msg = "Server tool updates disagree for the same call."
        raise UnsupportedResponsesOutputError(msg)
    return False


__all__ = ["ServerToolItem", "ServerToolTracker", "update_messages"]
