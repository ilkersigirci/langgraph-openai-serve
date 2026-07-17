# OpenAI API Compatibility

LangGraph OpenAI Serve is an OpenAI-client compatibility layer, not a separate
LangGraph-specific HTTP API. Public chat and model behavior must remain
reachable through the configured OpenAI-compatible base URL.

The same contract lets LGOS run behind OpenAI-compatible intermediaries without
a project-specific inference adapter. Concrete client and gateway behavior is
documented under [Integrations](../integrations/index.md).

## Contract

- Registered graph names are exposed as OpenAI `model` values.
- Core graph behavior must fit OpenAI-compatible request fields, response
  objects, tool calls, streaming chunks, metadata, or error envelopes.
- Do not require custom payloads, headers, routes, or SSE event shapes for core
  behavior unless the OpenAI client path still works.
- Treat direct HTTP calls such as `curl` as diagnostics; validate compatibility
  through OpenAI client behavior.

The implemented endpoints are listed in [Reference](../reference.md).

## Model Feature Discovery

The [OpenAI Model object](https://developers.openai.com/api/reference/resources/models)
has no `metadata` field. LGOS keeps its standard fields unchanged and places
feature and runtime-settings discovery in a namespaced, versioned extension
on the standard model-retrieval response:

```json
{
  "id": "simple-graph",
  "object": "model",
  "created": 1720000000,
  "owned_by": "langgraph-openai-serve",
  "langgraph_openai_serve": {
    "schema_version": 1,
    "features": [],
    "client_settings": {
      "schema_version": 1,
      "json_schema": {
        "type": "object",
        "properties": {
          "use_history": {
            "type": "boolean",
            "default": false
          },
          "audience": {
            "type": "string",
            "enum": ["general", "beginner", "expert"],
            "default": "general"
          }
        },
        "additionalProperties": false
      },
      "defaults": {
        "use_history": false,
        "audience": "general"
      }
    }
  }
}
```

`GraphConfig.features` is the single source of truth: the runner uses it to
enable behavior and `GET /v1/models/{model}` serializes it for discovery.
`GraphConfig.client_settings` is an explicit, allowlisted public Pydantic model;
LGOS never publishes a graph's internal LangGraph context schema automatically.
Additive features do not require an outer schema-version change. The nested
runtime settings descriptor has its own version, and clients must ignore
versions they do not understand.

`GET /v1/models` remains a lightweight list containing only the standard
`id`, `object`, `created`, and `owned_by` fields. A client lists profiles first
and retrieves details only for the selected model. This avoids large list
responses and keeps internal or secret-bearing runtime context out of discovery.

[OpenAI treats added response properties as backward-compatible](https://developers.openai.com/api/reference/overview#backwards-compatibility).
Direct JavaScript clients can read the property normally, and the
[OpenAI Python SDK exposes it through `model_extra`](https://github.com/openai/openai-python#making-customundocumented-requests).
An intermediary may implement its own model catalog or rebuild a retrieved model
from the standard fields and drop extensions. Clients that require detailed
LGOS discovery must use direct model retrieval or a route that forwards the
response unchanged. Request paths must also preserve OpenAI metadata. Concrete
gateway configurations are documented under
[OpenAI-Compatible Proxies](../integrations/openai-proxies.md).

## Runtime Settings

The request keeps each concern in its standard OpenAI location:

| Concern | OpenAI request location |
| --- | --- |
| System instructions | A `system` message |
| Small graph-specific values | One `metadata.langgraph_runtime_settings` string containing a JSON object |
| Graph selection | `model` |
| Thread/checkpoint identity | Existing `metadata.langgraph_thread_id` convention |

Only small graph-specific values belong to `ClientSettings`. A graph may expose
controlled semantic choices such as intended audience, but not arbitrary system
instruction text. Client-authored system instructions remain ordinary graph-input
messages.

OpenAI metadata permits at most 16 string pairs, with keys up to 64 characters
and values up to 512 characters. Public settings consume one pair and checkpoint
identity consumes one more. Clients use `json.dumps()` or `JSON.stringify()` to
encode the complete metadata string and omit values equal to the advertised
defaults. The advertised JSON Schema describes the available settings; LGOS
remains the validation authority. The descriptor's separate `defaults` object
is the authoritative validated baseline; JSON Schema `default` keywords are
annotations and may precede Pydantic field normalization. Native Chat
Completions fields keep their standard semantics. Graphs that need identity,
authorization, database clients, secrets, or other server-owned per-request
context combine `client_settings` with `context_factory(request, settings)`.

### Per-Request Resolution

Every chat completion starts from the registered defaults. Values supplied in
`metadata.langgraph_runtime_settings` replace matching top-level defaults, and LGOS
validates the complete result. The merge is shallow: a supplied nested object
replaces that whole default value rather than recursively merging its keys.

Client settings are not persisted between requests. In particular,
`metadata.langgraph_thread_id` restores checkpoint state but does not restore
runtime context. Clients must resend non-default settings on every request that
needs them, including interrupt-resume requests. A later request that omits
`langgraph_runtime_settings` uses registered defaults again.

Treat a missing or unsupported discovery extension as a normal fallback to server
defaults. See [Configure LangGraph Runtime Settings](../how-to-guides/langgraph-runtime-settings.md)
for the complete author and client flow. Adapter support is summarized under
[Integrations](../integrations/index.md#capabilities).

## Message And Schema Adaptation

Incoming OpenAI messages are converted to LangChain messages. `GraphConfig`
adapters keep custom LangGraph schemas behind that public boundary. See
[LangGraph Integration](langgraph-integration.md#adaptation) and
[Custom Graphs](../tutorials/custom-graphs.md#custom-schemas).

## Streaming

Streaming responses use OpenAI-compatible Server-Sent Events. See
[LangGraph Integration](langgraph-integration.md#runner-behavior) for internal
event handling and [Request Cancellation](langgraph-integration.md#request-cancellation)
for request-scoped disconnect cancellation, proxy behavior, and cooperative
limits.

## Citation Ownership

OpenAI `url_citation` annotations are the canonical citation contract. Their
URL, title, and text span associate a source with the answer. `end_index` is
inclusive, matching OpenAI's last-character convention.

LGOS returns `message.annotations` for non-streaming responses and
`delta.annotations` on the final streaming chunk. It does not define a
UI-specific source schema.

Portable resource presentation belongs in the assistant text, not in the
annotation object. Graphs may return ordinary Markdown links and images in
`message.content`. When a graph also emits structured attribution, its
`url_citation` remains limited to its standard URL, title, and text span. Audio
and video resources should use ordinary Markdown links rather than UI-specific
players. RAG graphs must preserve only resource URLs supplied by their retrieved
context and must not invent or rewrite them.

Structured citations remain available to OpenAI clients that need
machine-readable provenance. The `citation-events` demo showcases that optional
contract.

The streaming field is a compatibility extension because the published Chat
Completions delta schema does not currently declare annotations. The OpenAI
Python SDK preserves it as extra model data.

See the official [OpenAI citation contract](https://developers.openai.com/api/docs/guides/tools-web-search#output-and-citations).

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

Invalid runtime settings return HTTP 400 with
`param: "metadata.langgraph_runtime_settings"`. A missing discovery extension is not an
error; the client simply uses server defaults.

## Tool Calls And Interrupts

Tool definitions are accepted for OpenAI compatibility. Graphs can read them
through the full request in `request_to_input` or load tools independently, as
the mock MCP demo does.

LGOS supports only the modern Chat Completions tool-calling shape: `tools`,
`tool_choice`, assistant `tool_calls`, and `tool` messages with a matching
`tool_call_id`. The deprecated `functions`, singular `function_call`, and
`function` message role are rejected rather than silently ignored. OpenAI marks
the older `functions` and top-level `function_call` parameters as deprecated in
the [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

Interrupt-enabled graphs represent human-in-the-loop pauses as an OpenAI tool
call named `langgraph_interrupt` with a versioned JSON argument envelope
containing the thread id, interrupt id, and payload. Clients resume by sending a
follow-up `tool` role message with the matching `tool_call_id` and JSON content
such as `{"resume": "approved"}`.

## Known Differences From OpenAI

- `model` selects a registered LangGraph graph, not an OpenAI-hosted model.
- The supported surface focuses on chat completions, model listing/retrieval,
  health, and compatible tool-call flows.
- Authentication is not enforced by default.
- Token usage is approximate.
