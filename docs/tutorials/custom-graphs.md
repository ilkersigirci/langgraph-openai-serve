# Creating Custom LangGraph Workflows

This tutorial explains how to create custom LangGraph workflows and expose them via the OpenAI-compatible API.

## Creating a Basic LangGraph Workflow

LangGraph allows you to create complex workflow graphs for orchestrating LLM calls and other operations. Let's create a simple example:

```python
from typing import Annotated, Sequence
from pydantic import BaseModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Define the state schema for our graph
class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define the node function that processes messages
async def generate(state: AgentState):
    """Generate a response to the user's message."""
    # Use the messages in the state to create a prompt for the LLM
    messages = state.messages

    # Create a simple LLM chain
    llm = ChatOpenAI(temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    # Get the last message from the user
    last_message = messages[-1]
    response = await chain.ainvoke({"input": last_message.content})

    # Return the updated state with the new AI message
    return {
        "messages": [AIMessage(content=response)]
    }

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("generate", generate)
workflow.add_edge("generate", END)
workflow.set_entry_point("generate")

# Compile the workflow for execution
custom_graph = workflow.compile()
```

## Exposing Your Custom Graph

After creating your custom graph, you can expose it through the OpenAI-compatible API:

```python
from langgraph_openai_serve import LangchainOpenaiApiServe, GraphRegistry, GraphConfig

# Assume custom_graph is your compiled LangGraph instance
# from my_custom_graph_module import custom_graph

# Create a GraphRegistry
graph_registry = GraphRegistry(
    registry={
        "my-custom-graph": GraphConfig(graph=custom_graph, streamable_node_names=["generate"]),
    }
)

# Create a server instance with your custom graph
graph_serve = LangchainOpenaiApiServe(
    graphs=graph_registry,
)

# Bind the OpenAI-compatible endpoints
graph_serve.bind_openai_chat_completion(prefix="/v1")

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(graph_serve.app, host="0.0.0.0", port=8000)
```

## Creating More Advanced Graphs

For more advanced use cases, you can create graphs with multiple nodes and complex logic:

```python
from typing import Annotated, Sequence, TypedDict
from pydantic import BaseModel, Field
import operator
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Define a more complex state
class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    should_search: bool = False
    search_results: str = ""

# Define node functions
async def router(state: AgentState):
    """Route to the appropriate tool based on the user's query."""
    query = state.messages[-1].content.lower()

    if "search" in query or "find" in query or "look up" in query:
        return {"should_search": True}
    else:
        return {"should_search": False}

async def search_tool(state: AgentState):
    """Simulate a search operation."""
    query = state.messages[-1].content
    # In a real implementation, you would call a search API here
    search_results = f"Found the following information about '{query}': This is simulated search data."
    return {"search_results": search_results}

async def generate_response(state: AgentState):
    """Generate a response based on messages and any search results."""
    llm = ChatOpenAI(temperature=0.7)

    messages = [
        HumanMessage(content="You are a helpful assistant with search capabilities.")
    ]

    # Add all the conversation history
    messages.extend(state.messages)

    # If we have search results, add them
    if state.search_results:
        messages.append(HumanMessage(content=f"Search results: {state.search_results}\nPlease use this information in your response."))

    # Generate a response
    ai_response = await llm.ainvoke(messages)

    return {"messages": [AIMessage(content=ai_response.content)]}

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", router)
workflow.add_node("search", search_tool)
workflow.add_node("generate", generate_response)

# Add conditional edges
workflow.add_conditional_edges(
    "router",
    {
        True: "search",
        False: "generate"
    },
    key=operator.itemgetter("should_search")
)

workflow.add_edge("search", "generate")
workflow.add_edge("generate", END)

workflow.set_entry_point("router")

# Compile the workflow
advanced_graph = workflow.compile()
```

## Best Practices for Graph Creation

When creating graphs for use with LangGraph OpenAI Serve, consider the following best practices:

1. **State Management**: Design your state schema carefully to include all necessary information.
2. **Error Handling**: Add error handling to your node functions to prevent crashes.
3. **Naming Conventions**: Use clear, descriptive names for graphs and nodes.
4. **Streaming Support**: For better user experience, design your graph to support streaming responses when possible. Configure which nodes should stream by setting the `streamable_node_names` list in the `GraphConfig` when registering your graph.
5. **Documentation**: Document what each graph does to make it easier for API users to choose the right model.

## Next Steps

Once you've created your custom graphs, you might want to:

- [Connect with OpenAI clients](openai-clients.md) to interact with your graphs
- Learn about [deploying with Docker](../how-to-guides/docker.md) for production use
- Explore how to [add authentication](../how-to-guides/authentication.md) to your API
