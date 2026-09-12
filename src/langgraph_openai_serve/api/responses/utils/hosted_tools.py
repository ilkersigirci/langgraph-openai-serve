"""Translate server-executed LangChain tools into Responses output items."""

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import UpdatesStreamPart
from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseCustomToolCallItem,
    ResponseCustomToolCallOutputItem,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_function_web_search import Action
from pydantic import TypeAdapter

from langgraph_openai_serve.api.responses.schemas import (
    ResponseCreateRequest,
    ResponseCustomToolCallInput,
    ResponseWebSearchCallInput,
)
from langgraph_openai_serve.api.responses.service import (
    UnsupportedResponsesOutputError,
)

HostedToolItem: TypeAlias = (
    ResponseCustomToolCall
    | ResponseCustomToolCallOutputItem
    | ResponseFunctionWebSearch
)
_HostedResult: TypeAlias = ResponseCustomToolCallOutputItem | ResponseFunctionWebSearch
_Item = TypeVar("_Item")
_WEB_SEARCH_ACTION_ADAPTER = TypeAdapter(Action)


def update_messages(event: UpdatesStreamPart) -> Iterator[BaseMessage]:
    """
    Read messages written by nodes in one native LangGraph update.

    Yields:
        Each message carried by the update.

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


@dataclass(frozen=True, slots=True)
class _WebSearchCall:
    item_id: str
    action: Action


_HostedCall: TypeAlias = ResponseCustomToolCall | _WebSearchCall


class HostedToolTracker:
    """Correlate selected agent calls with results executed by the graph."""

    def __init__(
        self,
        request: ResponseCreateRequest,
        selected: Collection[str],
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
        self._calls: dict[str, _HostedCall] = {}
        self._results: dict[str, _HostedResult] = {}

    def items(self, message: BaseMessage) -> Iterator[HostedToolItem]:
        """
        Read new public items represented by one agent message.

        Yields:
            Completed public hosted-tool items.

        """
        if isinstance(message, AIMessage):
            yield from self._custom_calls(message)
            yield from self._web_items(message)
        elif isinstance(message, ToolMessage):
            item = self._result(message)
            if item is not None:
                yield item

    def ensure_complete(self) -> None:
        """Reject a response that exposes a call the server did not finish."""
        if self._calls.keys() - self._results.keys():
            msg = "Hosted output contains a call without its server-executed result."
            raise UnsupportedResponsesOutputError(msg)

    def _custom_calls(self, message: AIMessage) -> Iterator[HostedToolItem]:
        """
        Read selected native custom calls from one assistant message.

        Yields:
            Each previously unseen custom call.

        """
        if not isinstance(message.content, list):
            return
        for block in message.content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "custom_tool_call":
                continue
            if block.get("name") not in self._selected:
                continue
            call = _custom_call(block)
            if call.call_id in self._history_call_ids:
                continue
            if _remember_once(self._calls, call.call_id, call):
                yield call

    def _web_items(self, message: AIMessage) -> Iterator[ResponseFunctionWebSearch]:
        if "web_search" not in self._selected:
            return
        self._remember_local_web_calls(message)
        for block in message.content_blocks:
            if block.get("type") == "server_tool_call":
                self._remember_provider_web_call(block)
            elif block.get("type") == "server_tool_result":
                item = self._provider_web_result(block)
                if item is not None:
                    yield item

    def _remember_local_web_calls(self, message: AIMessage) -> None:
        for call in message.tool_calls:
            if call.get("name") != "web_search":
                continue
            call_id = call.get("id")
            arguments = call.get("args")
            query = arguments.get("query") if isinstance(arguments, dict) else None
            if not isinstance(call_id, str) or not call_id:
                msg = "Hosted web_search calls must include a non-empty id."
                raise RuntimeError(msg)
            if not isinstance(query, str) or not query.strip():
                msg = "Hosted web_search calls must include a non-empty query."
                raise RuntimeError(msg)
            search = _WebSearchCall(
                item_id=f"ws_{call_id}",
                action=_WEB_SEARCH_ACTION_ADAPTER.validate_python(
                    {"type": "search", "query": query.strip()}
                ),
            )
            if search.item_id in self._history_item_ids:
                continue
            _remember_once(self._calls, call_id, search)

    def _remember_provider_web_call(self, block: Mapping[str, object]) -> None:
        if block.get("name") != "web_search":
            return
        call_id = block.get("id")
        action = block.get("args")
        if not isinstance(call_id, str) or not call_id:
            msg = "Hosted provider web_search calls must include a non-empty id."
            raise RuntimeError(msg)
        if not isinstance(action, dict):
            msg = "Hosted provider web_search calls must include an action."
            raise TypeError(msg)
        search = _WebSearchCall(
            item_id=call_id,
            action=_WEB_SEARCH_ACTION_ADAPTER.validate_python(action),
        )
        if search.item_id in self._history_item_ids:
            return
        _remember_once(self._calls, call_id, search)

    def _provider_web_result(
        self, block: Mapping[str, object]
    ) -> ResponseFunctionWebSearch | None:
        call_id = block.get("tool_call_id")
        if not isinstance(call_id, str):
            return None
        call = self._calls.get(call_id)
        if not isinstance(call, _WebSearchCall):
            return None
        status = block.get("status")
        if status not in {"success", "error"}:
            msg = "Hosted provider web_search results must include a valid status."
            raise RuntimeError(msg)
        item = _web_search_result(call, failed=status == "error")
        return item if _remember_once(self._results, call_id, item) else None

    def _result(self, message: ToolMessage) -> HostedToolItem | None:
        call_id = message.tool_call_id
        call = self._calls.get(call_id)
        if isinstance(call, ResponseCustomToolCall):
            output = _custom_output(message)
            item = ResponseCustomToolCallOutputItem(
                id=f"ctco_{call_id}",
                type="custom_tool_call_output",
                call_id=call_id,
                output=output,
                status="completed",
            )
        elif isinstance(call, _WebSearchCall):
            item = _web_search_result(call, failed=message.status == "error")
        else:
            return None
        return item if _remember_once(self._results, call_id, item) else None


def _custom_call(block: dict) -> ResponseCustomToolCall:
    """Drop LangChain chunk metadata before SDK validation."""
    item = ResponseCustomToolCallItem.model_validate(
        {
            key: value
            for key, value in block.items()
            if key in ResponseCustomToolCallItem.model_fields
        }
    )
    # ResponseOutputItem names the base class. Store ``status`` as a base-model
    # extra so FastAPI's response-model serialization does not drop it.
    return ResponseCustomToolCall.model_validate(item.model_dump())


def _custom_output(message: ToolMessage) -> str:
    output = message.content
    if isinstance(output, list) and len(output) == 1:
        part = output[0]
        if isinstance(part, dict) and part.get("type") == "custom_tool_call_output":
            output = part.get("output")
    if not isinstance(output, str):
        msg = "Hosted custom tool output must be a string."
        raise TypeError(msg)
    return output


def _web_search_result(
    call: _WebSearchCall, *, failed: bool
) -> ResponseFunctionWebSearch:
    return ResponseFunctionWebSearch(
        id=call.item_id,
        type="web_search_call",
        action=call.action,
        status="failed" if failed else "completed",
    )


def _remember_once(items: dict[str, _Item], key: str, item: _Item) -> bool:
    """Store a new item, ignore a replay, and reject conflicting updates."""
    previous = items.get(key)
    if previous is None:
        items[key] = item
        return True
    if previous != item:
        msg = "Hosted tool updates disagree for the same call."
        raise RuntimeError(msg)
    return False


__all__ = [
    "HostedToolItem",
    "HostedToolTracker",
    "update_messages",
]
