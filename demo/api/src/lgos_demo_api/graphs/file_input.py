"""File-input graph backed by the central demo Files service."""

from collections.abc import Mapping, Sequence
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import GraphConfig, GraphFeature
from openai import AsyncOpenAI
from pydantic import BaseModel

from lgos_demo_api.core.settings import settings
from lgos_demo_api.utils.file_inputs import resolve_file_inputs
from lgos_demo_api.utils.models import responses_model

DEFAULT_PROMPT = "Describe the attached file."
INSTRUCTIONS = "Answer the user's request using the attached files."


class FileInputState(BaseModel):
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


def _attachments(message: HumanMessage) -> list[str | dict]:
    """Return the message's files and images: its parts other than text."""
    if not isinstance(message.content, list):
        return []
    return [
        part
        for part in message.content
        if isinstance(part, Mapping) and part.get("type") != "text"
    ]


async def process_files(state: FileInputState) -> dict[str, list[AIMessage]]:
    """Resolve attached file IDs and send their bytes to OpenAI Responses."""
    message = _latest_human_message(state.messages)
    attachments = _attachments(message) if message is not None else []
    if message is None or not attachments:
        return {"messages": [AIMessage(content="Attach a file and try again.")]}

    async with AsyncOpenAI(
        base_url=settings.FILES_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
    ) as files_client:
        # Images given by URL are already model-ready and stay as they are.
        (resolved,) = await resolve_file_inputs(
            [HumanMessage(content=attachments)], files_client
        )

    model = responses_model()
    response = await model.ainvoke(
        [
            SystemMessage(content=INSTRUCTIONS),
            HumanMessage(
                content=[
                    {"type": "text", "text": message.text.strip() or DEFAULT_PROMPT},
                    *resolved.content,
                ]
            ),
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
