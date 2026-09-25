"""Report state, settings, and simulated work shared by the background demos."""

from collections.abc import Sequence
from typing import Annotated

from anyio import sleep
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph_openai_serve import ClientSettings
from pydantic import BaseModel, Field


class BackgroundReportState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class BackgroundReportSettings(ClientSettings):
    delay_seconds: int = Field(
        default=5,
        ge=0,
        le=300,
        title="Delay (seconds)",
        description="Time each report step takes, leaving room to poll or cancel it.",
    )


async def wait_for_report(settings: BackgroundReportSettings | None) -> None:
    await sleep((settings or BackgroundReportSettings()).delay_seconds)


async def prepare_report(
    request: str, settings: BackgroundReportSettings | None
) -> str:
    await wait_for_report(settings)
    return f"Background report for: {request}"
