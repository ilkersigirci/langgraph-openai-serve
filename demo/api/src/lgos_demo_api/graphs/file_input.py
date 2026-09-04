"""File-input graph backed by the central demo Files service."""

from base64 import b64encode
from collections.abc import Mapping, Sequence
from mimetypes import guess_type
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import GraphConfig, GraphFeature
from openai import AsyncOpenAI
from openai.types.responses import ResponseInputContentParam
from pydantic import BaseModel

from lgos_demo_api.settings import settings

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
    """Read native Chat Completions file parts from the latest user message."""
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


def _data_url(content_type: str, content: bytes) -> str:
    encoded = b64encode(content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


async def process_files(state: FileInputState) -> dict[str, list[AIMessage]]:
    """Resolve attached file IDs and send their bytes to OpenAI Responses."""
    message = _latest_human_message(state.messages)
    if message is None:
        return {"messages": [AIMessage(content="Attach a file and try again.")]}

    file_ids = _file_ids(message)
    if not file_ids:
        return {"messages": [AIMessage(content="Attach a file and try again.")]}

    prompt = message.text.strip() or DEFAULT_PROMPT
    input_content: list[ResponseInputContentParam] = [
        {"type": "input_text", "text": prompt}
    ]

    async with AsyncOpenAI(
        base_url=settings.FILES_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
    ) as files_client:
        for file_id in file_ids:
            metadata = await files_client.files.retrieve(file_id)
            download = await files_client.files.content(file_id)
            content_type = download.response.headers.get(
                "content-type", "application/octet-stream"
            ).partition(";")[0]
            if content_type == "application/octet-stream":
                content_type = guess_type(metadata.filename)[0] or content_type
            data_url = _data_url(content_type, await download.aread())
            if content_type.startswith("image/"):
                input_content.append(
                    {"type": "input_image", "detail": "auto", "image_url": data_url}
                )
            else:
                input_content.append(
                    {
                        "type": "input_file",
                        "filename": metadata.filename,
                        "file_data": data_url,
                    }
                )

    async with AsyncOpenAI(
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
    ) as model_client:
        response = await model_client.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=INSTRUCTIONS,
            input=[{"role": "user", "content": input_content}],
        )

    return {"messages": [AIMessage(content=response.output_text)]}


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
