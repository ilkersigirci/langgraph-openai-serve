"""Simple LangGraph agent implementation.

This module defines a simple LangGraph agent that interfaces directly with an LLM model.
It creates a straightforward workflow where a single node generates responses to user messages.

Examples:
    >>> from langgraph_openai.utils.simple_graph import app
    >>> result = await app.ainvoke({"messages": messages})
    >>> print(result["messages"][-1].content)

The module contains the following components:
- `AgentState` - TypedDict defining the state schema for the graph.
- `generate(state)` - Function that processes messages and generates responses.
- `workflow` - The StateGraph instance defining the workflow.
- `app` - The compiled workflow application ready for invocation.
"""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Type definition for the agent state.

    This TypedDict defines the structure of the state that flows through the graph.
    It uses the add_messages annotation to properly handle message accumulation.

    Attributes:
        messages: A sequence of BaseMessage objects annotated with add_messages.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


async def generate(state: AgentState):
    """Generate a response to the latest message in the state.

    This function extracts the latest message, creates a prompt with it,
    runs it through an LLM, and returns the response as an AIMessage.

    Args:
        state: The current state containing the message history.

    Returns:
        A dict with a messages key containing the AI's response.
    """
    question = state["messages"][-1].content

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant called api-template. Chat with the user with friendly tone",
            ),
            ("human", "{question}"),
        ]
    )

    model = ChatOpenAI(streaming=True, temperature=0.05)
    chain = prompt | model | StrOutputParser()

    response = await chain.ainvoke({"question": question})

    return {
        "messages": [AIMessage(content=response)],
    }


# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("generate", generate)
workflow.add_edge("generate", END)
workflow.set_entry_point("generate")

# Compile the workflow for execution
app = workflow.compile()
