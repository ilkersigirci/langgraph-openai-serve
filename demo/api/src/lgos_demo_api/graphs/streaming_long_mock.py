"""Deterministic long stream for checking streaming and Stop in the demo UIs."""

from typing import Annotated, Sequence

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import GraphConfig
from pydantic import BaseModel

SENTENCES = "".join(
    f"{number}. This is sentence {number} of 100.\n" for number in range(1, 101)
)
# The fake model streams one character per delay, about 17 seconds in total.
CHARACTER_DELAY_SECONDS = 0.005


class StreamingLongMockState(BaseModel):
    """Messages the client supplied for this request."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


async def stream_sentences(
    state: StreamingLongMockState,
) -> dict[str, list[AIMessage]]:
    """Stream the sentences, first quoting the end of the previous answer."""
    answer = SENTENCES
    previous = next(
        (
            message
            for message in reversed(state.messages)
            if isinstance(message, AIMessage)
        ),
        None,
    )
    if previous is not None:
        # Shows the history the client sent back, such as a stopped answer.
        last_line = previous.text.rstrip().rpartition("\n")[2]
        answer = f'Previous answer ended with: "{last_line}"\n\n{answer}'
    model = FakeListChatModel(responses=[answer], sleep=CHARACTER_DELAY_SECONDS)
    return {"messages": [await model.ainvoke(state.messages)]}


streaming_long_mock_graph = (
    StateGraph(StreamingLongMockState)
    .add_node("stream_sentences", stream_sentences)
    .add_edge(START, "stream_sentences")
    .add_edge("stream_sentences", END)
    .compile()
)

streaming_long_mock_graph_config = GraphConfig(
    graph=streaming_long_mock_graph,
    description=(
        "Slowly streams 100 numbered sentences for checking streaming and Stop, "
        "first quoting the end of the previous answer it received."
    ),
)

__all__ = ["streaming_long_mock_graph", "streaming_long_mock_graph_config"]
