"""Display generated LGOS files as persisted Chainlit elements."""

import chainlit as cl
from chainlit_utils.chat.history import mark_model_context_excluded
from chainlit_utils.openai.tools import function_call_output
from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall
from plotly import io as pio
from pydantic import BaseModel, ConfigDict, Field

from lgos_chainlit.clients import files_request

DISPLAY_FILE_TOOL_NAME = "display_file"
PLOTLY_MEDIA_TYPE = "application/vnd.plotly.v1+json"


class DisplayFileArguments(BaseModel):
    """Arguments for the demo's client-owned file display function."""

    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    media_type: str = Field(pattern=r"^(?:image/|application/vnd\.plotly\.v1\+json$)")
    title: str = Field(min_length=1)
    alt: str = Field(min_length=1)


DISPLAY_FILE_TOOL: FunctionToolParam = {
    "type": "function",
    "name": DISPLAY_FILE_TOOL_NAME,
    "description": "Display a file stored in the configured OpenAI Files API.",
    "strict": True,
    "parameters": DisplayFileArguments.model_json_schema(),
}


async def display_file(call: ResponseFunctionToolCall) -> dict[str, object]:
    """Download and persist a native image or interactive Plotly element."""
    if call.name != DISPLAY_FILE_TOOL_NAME:
        raise ValueError(f"Unsupported client function: {call.name}")
    try:
        arguments = DisplayFileArguments.model_validate_json(call.arguments)
    except ValueError as exc:
        raise ValueError("The display_file call contains invalid arguments.") from exc

    client, provider = files_request()
    download = await client.files.content(
        arguments.file_id, extra_query={"provider": provider}
    )
    content = await download.aread()
    if arguments.media_type == PLOTLY_MEDIA_TYPE:
        element = cl.Plotly(
            name=arguments.filename,
            figure=pio.from_json(content.decode()),
            display="inline",
        )
    else:
        element = cl.Image(
            name=arguments.filename,
            content=content,
            mime=arguments.media_type,
            display="inline",
        )
    message = cl.Message(content=arguments.title, elements=[element])
    mark_model_context_excluded(message)
    await message.send()
    return function_call_output(call, '{"displayed":true}')
