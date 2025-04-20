"""Simple LangGraph agent implementation.

This module defines a simple LangGraph agent that interfaces directly with an LLM model.
It creates a straightforward workflow where a single node generates responses to user messages.

Examples:
    >>> from langgraph_openai.utils.simple_graph import app
    >>> result = await app.ainvoke({"messages": messages})
    >>> print(result["messages"][-1].content)

The module contains the following components:
- `AgentState` - Pydantic BaseModel defining the state schema for the graph.
- `generate(state)` - Function that processes messages and generates responses.
- `workflow` - The StateGraph instance defining the workflow.
- `app` - The compiled workflow application ready for invocation.
"""

from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

# from langgraph.prebuilt import create_react_agent


class AgentState(BaseModel):
    """Type definition for the agent state.

    This BaseModel defines the structure of the state that flows through the graph.
    It uses the add_messages annotation to properly handle message accumulation.

    Attributes:
        messages: A sequence of BaseMessage objects annotated with add_messages.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


class SimpleConfigSchema(BaseModel):
    """Configurable fields that are taken from the user"""

    use_history: bool = False


async def generate(state: AgentState, config: SimpleConfigSchema) -> dict:
    """Generate a response to the latest message in the state.

    This function extracts the latest message, creates a prompt with it,
    runs it through an LLM, and returns the response as an AIMessage.

    Args:
        state: The current state containing the message history.

    Returns:
        A dict with a messages key containing the AI's response.
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)

    system_message = (
        "system",
        "You are a helpful assistant called Langgraph Openai Serve. Chat with the user with friendly tone",
    )

    if config["configurable"]["use_history"] is False:
        question = state.messages[-1].content

        prompt = ChatPromptTemplate.from_messages(
            [system_message, ("human", "{question}")]
        )

        chain = prompt | model | StrOutputParser()
        response = await chain.ainvoke({"question": question})
    else:
        messages = state.messages
        prompt = ChatPromptTemplate.from_messages([system_message, *messages])
        chain = prompt | model | StrOutputParser()
        response = await chain.ainvoke({})

    return {
        "messages": [AIMessage(content=response)],
    }


# Define the workflow graph
workflow = StateGraph(AgentState, config_schema=SimpleConfigSchema)
workflow.add_node("generate", generate)
workflow.add_edge("generate", END)
workflow.set_entry_point("generate")

# Compile the workflow for execution
app = workflow.compile()
