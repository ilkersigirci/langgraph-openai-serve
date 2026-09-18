"""Apply LGOS file capabilities to reusable Chainlit file helpers."""

from typing import Any

from chainlit.config import ChainlitConfigOverrides
from chainlit_utils.openai.files import (
    file_upload_overrides as chainlit_file_upload_overrides,
)
from chainlit_utils.openai.files import (
    with_response_file_parts as with_openai_response_file_parts,
)
from openai.types import Model

from lgos_chainlit.clients import files_request
from lgos_chainlit.lgos_protocol import GraphFeature, model_supports


def file_upload_overrides(model: Model | None) -> ChainlitConfigOverrides:
    """Enable attachments only for graphs that advertise file inputs."""
    return chainlit_file_upload_overrides(
        model is not None and model_supports(model, GraphFeature.FILE_INPUTS)
    )


async def with_response_file_parts(
    input_items: list[dict[str, Any]],
    message: object,
) -> list[dict[str, Any]]:
    """Upload attachments through the configured gateway Files API."""
    client, provider = files_request()
    return await with_openai_response_file_parts(
        input_items,
        message,
        client=client,
        extra_query={"provider": provider},
    )
