# Core Graph Patterns

Four small graphs isolate the basic ways an OpenAI request can drive a
LangGraph. They keep persistence, client events, and interrupts out of the way
so each adapter or streaming behavior is visible on its own.

| Graph | Demonstrates |
| --- | --- |
| `custom-input-output-context` | Custom graph input, output, and typed runtime context |
| `advanced-mcp-tools` | An async graph factory that loads MCP-style tools before building an agent |
| `multi-node-streaming` | Ordered text streamed by more than one graph node |
| `simple-graph` | A real chat model controlled by discoverable runtime settings |

## LangGraph Topology

=== "custom-input-output-context"

    ```mermaid
    graph TD;
		__start__ --> generate;
		generate --> __end__;
    ```

=== "advanced-mcp-tools"

    ```mermaid
    graph TD;
		__start__ --> model;
		model -.-> __end__;
		model -.-> tools;
		tools -.-> model;
    ```

=== "multi-node-streaming"

    ```mermaid
    graph TD;
		__start__ --> write_first_contribution;
		write_first_contribution --> write_second_contribution;
		write_second_contribution --> assemble_answer;
		assemble_answer --> __end__;
    ```

=== "simple-graph"

    ```mermaid
    graph TD;
		__start__ --> generate;
		generate --> __end__;
    ```

## Request Flow

### custom-input-output-context

This deterministic graph demonstrates all three `GraphConfig` adapters around
one typed graph:

1. `request_to_input` takes the final OpenAI message and returns
   `{"question": ...}` instead of the default message-state input.
2. `context_factory` maps OpenAI `user` to immutable `AppContext`, falling back
   to `anonymous` when it is absent.
3. `generate` returns the graph-native `{"answer": ...}` output.
4. `output_to_message` converts that output to the final `AIMessage`.

No chat model or external service is called.

### advanced-mcp-tools

LGOS awaits the registered async graph factory for each request. The factory
loads one mock weather tool, passes it to LangChain `create_agent`, and returns
the compiled model-tools loop. The deterministic fake model calls the tool for
Istanbul and then returns its result as assistant text.

This is an MCP-style lifecycle example, not a network MCP integration. A real
application can replace the mock client with LangChain's
[`MultiServerMCPClient`](https://docs.langchain.com/oss/python/langchain/mcp)
while keeping the async factory boundary.

### multi-node-streaming

The default request adapter supplies OpenAI messages as graph state. Two nodes
run sequentially and stream one deterministic sentence each. Their
`answer_parts` updates use an append reducer; `assemble_answer` joins those
parts into the single final assistant message. Streaming and non-streaming
requests therefore produce identical complete text.

### simple-graph

LGOS validates the model's advertised `SimpleContext` settings before calling
the graph. `generate` adds the system prompt and selected audience, then applies
the history setting before calling the upstream chat model:

- `use_history=false` sends only the latest message.
- `use_history=true` sends every message supplied in the current request.
- `audience` is `general`, `beginner`, or `expert`.

`use_history` does not load or persist conversation history; history exists only
when the client includes it in the current request. See
[Runtime Settings](../../how-to-guides/langgraph-runtime-settings.md) for the
shared discovery and metadata transport.

## State And Output

None of these graphs uses a checkpointer or LangGraph Store, so graph state ends
with the request. All four return standard OpenAI assistant text and emit no
LGOS client events. `multi-node-streaming` and `simple-graph` identify their
answer-producing nodes for incremental text streaming; the other two focus on
adapter and factory behavior rather than token timing.

## Try It

| Model | Prompt | Optional request value |
| --- | --- | --- |
| `custom-input-output-context` | `Show me custom schemas.` | `user="demo-user"` |
| `advanced-mcp-tools` | `What is the weather in Istanbul?` | None |
| `multi-node-streaming` | `Build one answer from two nodes.` | None |
| `simple-graph` | `Explain what this demo does.` | Select an audience in the UI |
