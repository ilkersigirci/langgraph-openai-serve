# Core Graph Patterns

Six small graphs isolate the basic ways an OpenAI request can drive a
LangGraph. They keep persistence, client events, and interrupts out of the way
so each adapter or streaming behavior is visible on its own.

| Graph | Demonstrates |
| --- | --- |
| `custom-input-output-context` | Custom graph input, output, and typed runtime context |
| `advanced-mcp-tools` | An async graph factory that loads MCP-style tools before building an agent |
| `multi-node-streaming` | Ordered text streamed by more than one graph node |
| `response-outcomes` | Native refusal content and incomplete terminal responses |
| `simple-graph` | A real chat model controlled by discoverable runtime settings |
| `simple-graph-external-tools` | A chat model that receives and returns client-owned function tools |

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

=== "response-outcomes"

    ```mermaid
    graph TD;
		__start__ --> respond_with_outcome;
		respond_with_outcome --> __end__;
    ```

=== "simple-graph-external-tools"

    ```mermaid
    graph TD;
		__start__ --> generate;
		generate --> __end__;
    ```

## Request Flow

Each maintained demo UI sends a standard request through LGOS
`/v1/responses`. After LGOS decodes the OpenAI input, each selected graph uses
the graph-specific path below and returns standard assistant output through the
same endpoint.

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

### simple-graph-external-tools

This graph keeps the tool loop on the client side. `request_to_input` carries the
normalized `tools`, `tool_choice`, and `parallel_tool_calls` fields into graph
state. `generate` binds those definitions to the upstream chat model and returns
its `AIMessage`.

The graph does not execute a tool or discard history. Responses clients replay
the returned `function_call` items and append matching `function_call_output`
items. Direct Chat compatibility clients send the returned assistant
`tool_calls` and matching `tool` messages. LGOS normalizes both protocols before
the graph sees them; no Responses-to-Chat gateway is required.

### response-outcomes

This deterministic graph makes two Responses outcomes reproducible without an
upstream model:

- `refusal` returns the model's explanation as a `refusal` content part. The
  message and Response are still `completed`; `output_text` is empty because a
  refusal is not ordinary output text. A stream emits
  `response.refusal.delta`, `response.refusal.done`, then
  `response.completed`.
- `incomplete` returns partial text plus `status="incomplete"` and
  `incomplete_details.reason="max_output_tokens"`. A stream ends with
  `response.incomplete`, not `response.completed`.

Use refusal content when a model declines a request, especially for a safety
reason. Use an incomplete result when generation started but stopped before a
complete answer was available, such as at an output-token limit or content
filter. Transport and application failures belong in the normal HTTP or
`response.failed` paths instead.

OpenAI defines refusal as a distinct assistant-message content type and defines
incomplete details separately on the Response. Current upstream incomplete
reasons include `max_output_tokens`, `max_messages`, `content_filter`, and
`steered`; LGOS currently maps model output-token and content-filter outcomes.
See the official OpenAI [Responses output message schema](https://developers.openai.com/api/reference/resources/responses#response-output-message)
and [incomplete details schema](https://developers.openai.com/api/reference/resources/responses#response-incomplete-details).

The demo synthesizes these outputs only to stay deterministic. A real
model-backed graph should return the provider's final `AIMessage` unchanged so
LGOS can retain its refusal content or incomplete metadata.

## State And Output

None of these graphs uses a checkpointer or LangGraph Store, so graph state ends
with the request. All six return standard OpenAI assistant messages and emit no
LGOS client events. `multi-node-streaming`, `simple-graph`, and the
external-tools graph identify their answer-producing nodes for incremental text
streaming and standard OpenAI function-call output.

## Try It

| Model | Prompt | Optional request value |
| --- | --- | --- |
| `custom-input-output-context` | `Show me custom schemas.` | `user="demo-user"` |
| `advanced-mcp-tools` | `What is the weather in Istanbul?` | None |
| `multi-node-streaming` | `Build one answer from two nodes.` | None |
| `response-outcomes` | `refusal` or `incomplete` | None |
| `simple-graph` | `Explain what this demo does.` | Select an audience in the UI |
| `simple-graph-external-tools` | `Use the supplied function tool when needed.` | Client supplies `tools` |
