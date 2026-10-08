"""File-input graph backed by the central demo Files service."""

from collections.abc import Mapping, Sequence
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.content import (
    ContentBlock,
    create_text_block,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import GraphConfig, GraphFeature
from openai import AsyncOpenAI
from pydantic import BaseModel

from lgos_demo_api.core.settings import settings
from lgos_demo_api.utils.file_inputs import load_file_block
from lgos_demo_api.utils.models import responses_model

DEFAULT_PROMPT = "Describe the attached file."
INSTRUCTIONS = "Answer the user's request using the attached files."


class FileInputState(BaseModel):
    """Conversation state consumed by the file-input graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def _latest_human_message(messages: Sequence[BaseMessage]) -> HumanMessage | None:
    return next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        None,
    )


def _file_ids(message: HumanMessage) -> list[str]:
    """Read protocol-neutral normalized file parts from the latest user message."""
    if not isinstance(message.content, list):
        return []

    file_ids: list[str] = []
    for part in message.content:
        if not isinstance(part, Mapping) or part.get("type") != "file":
            continue
        file = part.get("file")
        file_id = file.get("file_id") if isinstance(file, Mapping) else None
        if isinstance(file_id, str) and file_id:
            file_ids.append(file_id)
    return file_ids


async def process_files(state: FileInputState) -> dict[str, list[AIMessage]]:
    """Resolve attached file IDs and send their bytes to OpenAI Responses."""
    message = _latest_human_message(state.messages)
    if message is None:
        return {"messages": [AIMessage(content="Attach a file and try again.")]}

    file_ids = _file_ids(message)
    if not file_ids:
        return {"messages": [AIMessage(content="Attach a file and try again.")]}

    prompt = message.text.strip() or DEFAULT_PROMPT
    input_content: list[ContentBlock] = [create_text_block(text=prompt)]

    async with AsyncOpenAI(
        base_url=settings.FILES_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
    ) as files_client:
        input_content.extend(
            [await load_file_block(files_client, file_id) for file_id in file_ids]
        )

    model = responses_model()
    response = await model.ainvoke(
        [
            SystemMessage(content=INSTRUCTIONS),
            HumanMessage(content_blocks=input_content),
        ]
    )
    return {"messages": [response]}


workflow = StateGraph(FileInputState)
workflow.add_node("process_files", process_files)
workflow.add_edge("process_files", END)
workflow.set_entry_point("process_files")

file_input_graph = workflow.compile()

file_input_graph_config = GraphConfig(
    graph=file_input_graph,
    description="Analyzes attached files with the OpenAI Responses API.",
    features={GraphFeature.FILE_INPUTS},
)

__all__ = ["file_input_graph", "file_input_graph_config"]
