# OpenAI API Compatibility

LangGraph OpenAI Serve is an OpenAI-client compatibility layer, not a separate
LangGraph-specific HTTP API. Public chat and model behavior must remain
reachable through the configured OpenAI-compatible base URL used by official
OpenAI SDKs, Chainlit, Open WebUI, and similar clients.

## Contract

- Registered graph names are exposed as OpenAI `model` values.
- Core graph behavior must fit OpenAI-compatible request fields, response
  objects, tool calls, streaming chunks, metadata, or error envelopes.
- Do not require custom payloads, headers, routes, or SSE event shapes for core
  behavior unless the OpenAI client path still works.
- Treat direct HTTP calls such as `curl` as diagnostics; validate compatibility
  through OpenAI client behavior.

The implemented endpoints are listed in [Reference](../reference.md).

## Message And Schema Adaptation

Incoming OpenAI messages are converted to LangChain messages. The default graph
input is `{"messages": langchain_messages}` and the default response text is
`result["messages"][-1].content`.

Graphs with custom LangGraph schemas should use `GraphConfig` adapters rather
than inventing a new HTTP contract. See [Custom Graphs](../tutorials/custom-graphs.md).

## Streaming

Streaming responses use OpenAI-compatible Server-Sent Events. Internally the
runner consumes LangGraph `astream` events with message streaming for text and
update streaming for interrupts. Only chunks from configured
`streamable_node_names` are emitted to clients.

## Citation Ownership And UI Rendering

OpenAI `url_citation` annotations are the canonical citation contract. Their
URL, title, and text span associate a source with the answer. `end_index` is
inclusive, matching OpenAI's last-character convention.

| Layer | Citation behavior |
| --- | --- |
| LGOS API | Returns `message.annotations` for non-streaming responses and `delta.annotations` on the final streaming chunk. It has no Chainlit or Open WebUI source schema. |
| Chainlit demo | Streams assistant content unchanged and relies on Chainlit's Markdown renderer for links and images. It does not consume citation annotations. |
| Open WebUI demo | Streams assistant content unchanged and relies on Open WebUI's Markdown renderer for links and images. For streaming requests, its generic Pipe transparently forwards annotations in an OpenAI-compatible chunk without translating them. |

Portable resource presentation belongs in the assistant text, not in the
annotation object. Graphs may return ordinary Markdown links and images in
`message.content`; Chainlit and Open WebUI receive that content unchanged. When
a graph also emits structured attribution, its `url_citation` remains limited
to its standard URL, title, and text span. Audio and video resources should use
ordinary Markdown links rather than UI-specific players. RAG graphs must
preserve only resource URLs supplied by their retrieved context and must not
invent or rewrite them.

Structured citations remain available to OpenAI clients that need
machine-readable provenance. The `citation-events` demo showcases that optional
contract; the default Chainlit, Open WebUI, and `lgos-rag` paths prefer direct
Markdown and avoid UI-specific citation handling.

The streaming field is a compatibility extension because the published Chat
Completions delta schema does not currently declare annotations. The OpenAI
Python SDK preserves it as extra model data. The Open WebUI Pipe forwards this
extension only when `body["stream"]` is `true`; non-streaming generator results
remain plain text because Open WebUI stringifies yielded dictionaries on that
path.

See the official [OpenAI citation contract](https://developers.openai.com/api/docs/guides/tools-web-search#output-and-citations),
[Chainlit messages](https://docs.chainlit.io/concepts/message), and
[Open WebUI Pipe streaming format](https://docs.openwebui.com/features/extensibility/pipelines/pipes/#streaming-response-format).

## Errors

OpenAI-compatible routes return errors in the OpenAI envelope:

```json
{
  "error": {
    "message": "Graph 'missing' not found in registry.",
    "type": "invalid_request_error",
    "param": "model",
    "code": null
  }
}
```

Route code that knows the OpenAI error metadata should raise
`OpenAIHTTPException` with `openai.types.shared.ErrorObject`. Shared handlers
translate generic FastAPI validation and HTTP errors into the same envelope.

## Tool Calls And Interrupts

Tool definitions are accepted for OpenAI compatibility. Graphs can read them
through the full request in `request_to_input` or load tools independently, as
the mock MCP demo does.

Interrupt-enabled graphs represent human-in-the-loop pauses as an OpenAI tool
call named `langgraph_interrupt` with a versioned JSON argument envelope
containing the thread id, interrupt id, and payload. Clients resume by sending a
follow-up `tool` role message with the matching `tool_call_id` and JSON content
such as `{"resume": "approved"}`.

## Known Differences From OpenAI

- `model` selects a registered LangGraph graph, not an OpenAI-hosted model.
- The supported surface focuses on chat completions, model listing, health, and
  compatible tool-call flows.
- Authentication is not enforced by default.
- Token usage is approximate.
