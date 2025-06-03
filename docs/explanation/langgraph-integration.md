# Integration with LangGraph

This document explains how LangGraph OpenAI Serve integrates with LangGraph to execute workflows through an OpenAI-compatible API.

## LangGraph Overview

[LangGraph](https://github.com/langchain-ai/langgraph) is a library for building stateful, multi-actor applications with LLMs. It provides a way to create complex workflows where different components (or "nodes") can interact in a structured manner.

Key concepts in LangGraph:
- **StateGraph**: A graph where nodes modify a shared state
- **Nodes**: Functions that process the state and return updates
- **Edges**: Connections between nodes that define the flow of execution
- **Compiled Graphs**: Executable workflows created by compiling a graph

## How LangGraph OpenAI Serve Uses LangGraph

LangGraph OpenAI Serve acts as a bridge between the OpenAI API interface and your LangGraph workflows. Here's how it integrates with LangGraph:

### 1. Graph Registration

LangGraph workflows are registered with a unique name that will be used as the "model" name in the OpenAI API:

```python
from langgraph_openai_serve import LangchainOpenaiApiServe, GraphRegistry, GraphConfig

# Assume simple_graph and advanced_graph are compiled LangGraph workflows
# from your_graphs import simple_graph, advanced_graph

# Create a GraphRegistry
graph_registry = GraphRegistry(
    registry={
        "simple_graph": GraphConfig(graph=simple_graph, streamable_node_names=["generate"]),  # A compiled LangGraph workflow
        "advanced_graph": GraphConfig(graph=advanced_graph, streamable_node_names=["generate"]  # Another compiled LangGraph workflow
    }
)

graph_serve = LangchainOpenaiApiServe(
    graphs=graph_registry
)
```

### 2. Message Conversion

When a request comes in through the OpenAI API, the messages need to be converted from OpenAI format to LangChain format:

```python
def convert_to_lc_messages(messages: list[ChatCompletionRequestMessage]) -> list[BaseMessage]:
    """Convert OpenAI API messages to LangChain messages."""
    lc_messages = []
    for message in messages:
        if message.role == "user":
            lc_messages.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            lc_messages.append(AIMessage(content=message.content))
        elif message.role == "system":
            lc_messages.append(SystemMessage(content=message.content))
        # Handle more message types as needed
    return lc_messages
```

### 3. Graph Execution

The LangGraph workflow is executed using the converted messages:

#### Non-Streaming Execution

For regular (non-streaming) requests, the graph is executed using `.ainvoke()`:

```python
# Convert OpenAI messages to LangChain messages
lc_messages = convert_to_lc_messages(messages)

# Run the graph with the messages
result = await graph.ainvoke({"messages": lc_messages})

# Extract the response from the result
response = result["messages"][-1].content if result["messages"] else ""
```

#### Streaming Execution

For streaming requests, the graph is executed using `.astream_events()`:

```python
# Get streamable node names from the graph configuration
streamable_node_names = graph_config.streamable_node_names
inputs = {"messages": lc_messages}

async for event in graph.astream_events(inputs, version="v2"):
    event_kind = event["event"]
    langgraph_node = event["metadata"].get("langgraph_node", None)

    if event_kind == "on_chat_model_stream":
        if langgraph_node not in streamable_node_names:
            continue

        ai_message_chunk: AIMessageChunk = event["data"]["chunk"]
        ai_message_content = ai_message_chunk.content
        if ai_message_content:
            yield f"{ai_message_content}", {"tokens": 1}
```

### 4. Response Formatting

After the graph is executed, the response is formatted according to the OpenAI API schema:

```python
def format_completion_response(
    model: str,
    response_content: str,
    token_usage: dict[str, int]
) -> ChatCompletion:
    """Format a response according to the OpenAI API schema."""
    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    role="assistant",
                    content=response_content
                ),
                finish_reason="stop"
            )
        ],
        usage=ChatCompletionUsage(**token_usage)
    )
```

## State Management

LangGraph's core feature is state management, which allows for complex, multi-turn interactions. LangGraph OpenAI Serve preserves this capability by providing the messages as part of the state:

```python
# Define the state schema for our graph
class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define a node that processes messages
async def generate(state: AgentState):
    """Generate a response to the user's message."""
    messages = state.messages

    # Process messages and generate a response
    # ...

    return {
        "messages": [AIMessage(content=response)]
    }
```

## Default Simple Graph

LangGraph OpenAI Serve includes a default simple graph for users who want to get started quickly:

```python
# Define the state schema
class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define the generate function
async def generate(state: AgentState):
    # Use the messages in the state to create a prompt for the LLM
    messages = state.messages

    # Create a simple LLM chain
    llm = ChatOpenAI(temperature=0.7)
    response = await llm.ainvoke(messages)

    # Return the updated state with the new AI message
    return {"messages": [response]}

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("generate", generate)
workflow.add_edge("generate", END)
workflow.set_entry_point("generate")

# Compile the workflow
simple_graph = workflow.compile()
```

## Supporting Advanced LangGraph Features

LangGraph OpenAI Serve supports various advanced LangGraph features:

### 1. Multi-node Graphs

You can register complex graphs with multiple nodes:

```python
# Create a workflow with multiple nodes
workflow = StateGraph(AgentState)
workflow.add_node("parse", parse_query)
workflow.add_node("search", search_documents)
workflow.add_node("generate", generate_response)
workflow.add_conditional_edges("parse", ......)
workflow.add_edge("search", "generate")
workflow.add_edge("generate", END)
workflow.set_entry_point("parse")

advanced_graph = workflow.compile()
```

### 2. Streaming from Specific Nodes

LangGraph OpenAI Serve allows streaming content from specific nodes within your graph. This is configured using the `streamable_node_names` attribute in the `GraphConfig` when registering your graph. Only events originating from nodes listed in `streamable_node_names` will be streamed back to the client.

```python
from langgraph_openai_serve import GraphConfig, GraphRegistry, LangchainOpenaiApiServe

# Assume 'my_streaming_node_graph' has a node named 'streamer'
graph_config = GraphConfig(
    graph=my_streaming_node_graph,
    streamable_node_names=["streamer"] # Specify which node(s) can stream
)

registry = GraphRegistry(registry={"streaming_model": graph_config})

graph_serve = LangchainOpenaiApiServe(graphs=registry)
# ... rest of the server setup
```

### 3. Tool/Function Calling

OpenAI's function calling can be integrated with LangGraph's tool usage:

```python
# Define tools
def calculator(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

# Graph that uses tools
async def agent_with_tools(state: AgentState):
    # Process messages and execute tools
    # ...
    pass
```

## Considerations for LangGraph Integration

When integrating LangGraph workflows with the OpenAI API, consider the following:

1. **State Management**: Design your state schema to handle conversation history effectively.
2. **Streaming Support**: For better user experience, design your nodes to support streaming.
3. **Error Handling**: Add robust error handling to your nodes to prevent crashes.
4. **Parameter Mapping**: Consider how OpenAI API parameters like `temperature` and `max_tokens` should map to your graph.

## Next Steps

- Read about [OpenAI API compatibility](openai-compatibility.md) to understand how the API matches OpenAI's interface
- Learn how to [create custom graphs](../tutorials/custom-graphs.md) for your specific use cases
- Check the [API reference](../reference.md) for detailed endpoint documentation
