"""A client-requested time tool executed inside LGOS."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph_openai_serve import GraphConfig, GraphRequest
from pydantic import BaseModel

from lgos_demo_api.settings import settings


@tool
async def get_current_time(timezone: str) -> str:
    """Get the current time in an IANA timezone, such as Europe/Istanbul."""
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return (
            f"Unknown timezone: {timezone}. Use an IANA name such as Europe/Istanbul."
        )
    return f"{timezone}: {datetime.now(zone).isoformat(timespec='seconds')}"


class HostedToolState(BaseModel):
    messages: list[BaseMessage]
    time_requested: bool = False


def request_to_input(
    request: GraphRequest, messages: list[BaseMessage]
) -> HostedToolState:
    return HostedToolState(
        messages=messages,
        time_requested=request.tool_choice != "none"
        and "lgos_current_time" in request.hosted_tools,
    )


async def answer(state: HostedToolState) -> dict[str, list[AIMessage]]:
    agent = create_agent(
        model=ChatOpenAI(
            model=settings.OPENAI_MODEL,
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
        ),
        tools=[get_current_time] if state.time_requested else [],
        system_prompt=(
            "Help the user check the current time in different timezones. "
            "Use get_current_time for current times; never guess them. "
            "If the tool is unavailable, ask the client to enable lgos_current_time. "
            "Keep answers concise and include the timezone and UTC offset."
        ),
    )
    result = await agent.ainvoke({"messages": state.messages})
    return {"messages": [result["messages"][-1]]}


workflow = StateGraph(HostedToolState)
workflow.add_node("answer", answer)
workflow.add_edge(START, "answer")
workflow.add_edge("answer", END)
hosted_tool_graph = workflow.compile()
hosted_tool_graph_config = GraphConfig(
    graph=hosted_tool_graph,
    description="Checks current times with a client-requested tool executed on LGOS.",
    hosted_tools={"lgos_current_time"},
    request_to_input=request_to_input,
    streamable_node_names=["answer"],
)
