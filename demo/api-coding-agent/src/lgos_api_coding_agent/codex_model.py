"""Translate typed Codex notifications into LangChain's native message stream."""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
)
from langchain_core.messages import AIMessageChunk, BaseMessage, UsageMetadata
from langchain_core.messages.ai import add_usage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langgraph.config import get_stream_writer
from langgraph_openai_serve import GraphError, InvalidRequestError, status_event
from openai_codex.generated.v2_all import (
    AgentMessageDeltaNotification,
    AgentMessageThreadItem,
    CommandExecutionThreadItem,
    FileChangeThreadItem,
    ItemCompletedNotification,
    ItemStartedNotification,
    MessagePhase,
    ThreadItem,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
    TurnStartedNotification,
    TurnStatus,
)
from openai_codex.models import Notification
from pydantic import Field

from lgos_api_coding_agent.codex_runtime import CodexTurn


def conversation_prompt(messages: list[BaseMessage]) -> str:
    """Keep caller history explicitly labelled instead of promoting it to instructions."""
    transcript: list[dict[str, str]] = []
    roles = {"human": "user", "ai": "assistant", "system": "system"}
    for message in messages:
        if message.type not in roles or any(
            block["type"] != "text" for block in message.content_blocks
        ):
            msg = "The Codex adapter accepts text conversation history only."
            raise InvalidRequestError(
                msg,
                param="input",
                code="unsupported_input",
            )
        transcript.append({"role": roles[message.type], "content": message.text})
    return "Conversation transcript (JSON):\n" + json.dumps(
        transcript, ensure_ascii=False
    )


class CodexChatModel(BaseChatModel):
    """An async chat model solely to participate in LangGraph messages streaming."""

    event_source: Callable[[CodexTurn], AsyncGenerator[Notification | str, None]] = (
        Field(exclude=True)
    )
    model_name: str

    @property
    def _llm_type(self) -> str:
        return "codex"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        msg = "The Codex adapter requires asynchronous graph execution."
        raise NotImplementedError(msg)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        thread_name: str | None = None,
        **kwargs: object,
    ) -> ChatResult:
        return await agenerate_from_stream(
            self._astream(
                messages,
                stop=stop,
                run_manager=run_manager,
                thread_name=thread_name,
                **kwargs,
            )
        )

    async def _astream(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements] - Keep ordered SDK notification handling in one streaming state machine.
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # ruff: ignore[unused-method-argument] - Preserve the LangChain override signature.
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # ruff: ignore[unused-method-argument] - Preserve the LangChain override signature.
        *,
        thread_name: str | None = None,
        **kwargs: object,  # ruff: ignore[unused-method-argument] - Preserve the LangChain override signature.
    ) -> AsyncIterator[ChatGenerationChunk]:
        transcript = conversation_prompt(messages)
        writer = get_stream_writer()
        writer(status_event("Waiting for the workspace"))
        # Streamed text of every answer message. Commentary is never an entry.
        answers: dict[str, str] = {}
        usage: UsageMetadata | None = None
        completed = False

        def answer(item_id: str, text: str) -> ChatGenerationChunk:
            # An upstream without message phases sends every message as an answer.
            streamed = answers.get(item_id, "")
            separator = "\n\n" if any(answers.values()) and not streamed else ""
            answers[item_id] = streamed + text
            return ChatGenerationChunk(message=AIMessageChunk(content=separator + text))

        stream = self.event_source(
            CodexTurn(
                transcript=transcript,
                latest=messages[-1].text,
                thread_name=thread_name,
                continues=any(message.type == "ai" for message in messages),
            )
        )
        try:
            async for event in stream:
                if isinstance(event, str):
                    writer(status_event(event))
                    continue
                match event.payload:
                    case TurnStartedNotification():
                        writer(status_event("Codex is working in the workspace"))
                    case ItemStartedNotification(
                        item=ThreadItem(root=AgentMessageThreadItem() as item)
                    ):
                        if item.phase is not MessagePhase.commentary:
                            answers[item.id] = ""
                    case AgentMessageDeltaNotification(item_id=item_id, delta=delta):
                        # Deltas have no phase. An unknown ID is not a phase-less
                        # answer: wait for its authoritative completed item instead.
                        if item_id in answers:
                            yield answer(item_id, delta)
                    case ItemCompletedNotification(
                        item=ThreadItem(root=AgentMessageThreadItem() as item)
                    ):
                        streamed = answers.get(item.id)
                        if item.phase is MessagePhase.commentary:
                            if streamed:
                                msg = "Codex changed an answer's message phase."
                                raise GraphError(msg)
                            writer(status_event(item.text))
                        elif streamed and streamed != item.text:
                            msg = (
                                "Codex completed text differs from its streamed answer."
                            )
                            raise GraphError(msg)
                        elif not streamed and item.text:
                            yield answer(item.id, item.text)
                    case ItemStartedNotification(
                        item=ThreadItem(root=CommandExecutionThreadItem() as item)
                    ):
                        writer(status_event(f"Running: {item.command}"))
                    case ItemCompletedNotification(
                        item=ThreadItem(root=CommandExecutionThreadItem())
                    ):
                        writer(status_event("Shell command finished"))
                    case ItemStartedNotification(
                        item=ThreadItem(root=FileChangeThreadItem())
                    ):
                        writer(status_event("Editing files"))
                    case ItemCompletedNotification(
                        item=ThreadItem(root=FileChangeThreadItem())
                    ):
                        writer(status_event("File edit finished"))
                    case ThreadTokenUsageUpdatedNotification(token_usage=token_usage):
                        # `last` is one model call. `total` also covers the
                        # earlier requests of a resumed thread.
                        last = token_usage.last
                        usage = add_usage(
                            usage,
                            UsageMetadata(
                                input_tokens=last.input_tokens,
                                output_tokens=last.output_tokens,
                                total_tokens=last.total_tokens,
                                input_token_details={
                                    "cache_read": last.cached_input_tokens
                                },
                                output_token_details={
                                    "reasoning": last.reasoning_output_tokens
                                },
                            ),
                        )
                    case TurnCompletedNotification(turn=turn):
                        if turn.status is not TurnStatus.completed:
                            reason = (
                                turn.error.message if turn.error else turn.status.value
                            )
                            msg = f"Codex did not complete the turn: {reason}"
                            raise GraphError(msg)
                        completed = True
            if not completed or not any(answers.values()):
                msg = "Codex ended without a completed answer."
                raise GraphError(msg)
            if usage is not None:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata={"model_name": self.model_name},
                        usage_metadata=usage,
                    )
                )
        finally:
            await stream.aclose()
