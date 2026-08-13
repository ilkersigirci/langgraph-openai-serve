# OpenAI API Compatibility

LangGraph OpenAI Serve is an OpenAI-client compatibility layer, not a separate
LangGraph-specific HTTP API. Public chat and model behavior must remain
reachable through the configured OpenAI-compatible base URL.

The same contract lets LGOS run behind OpenAI-compatible intermediaries without
a project-specific inference adapter. Generic gateway requirements are in the
[proxy guide](../how-to-guides/openai-proxies.md); concrete Chainlit, Open WebUI,
and Bifrost implementations belong to the [Demo Stack](../demo/index.md).

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
    "description": "Streams responses with configurable history and audience.",
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

The standard OpenAI Model object has no description field. The required
`GraphConfig.description` is therefore exposed as
`langgraph_openai_serve.description` on both list entries and detailed model
responses. It is API-owned presentation text; clients decide how to render it.

`GraphConfig.features` is the single source of truth: the runner uses it to
enable behavior and `GET /v1/models/{model}` serializes it for discovery.
`GraphConfig.client_settings` is an explicit, allowlisted public Pydantic model;
LGOS never publishes a graph's internal LangGraph context schema automatically.
Additive features do not require an outer schema-version change. The nested
runtime settings descriptor has its own version, and clients must ignore
versions they do not understand.

| Feature | Enabled behavior |
| --- | --- |
| `client_events` | The server may emit opted-in public client-event chunks. |
| `interrupts` | The server supports the checkpointed interrupt/resume flow. |

`GET /v1/models` remains lightweight. Every entry contains the standard `id`,
`object`, `created`, and `owned_by` fields plus a small
`langgraph_openai_serve` object with only `schema_version` and `description`.
Features and client-settings schemas remain detail-only.
Every successful LGOS `GET /v1/models/{model}` response includes the complete
`langgraph_openai_serve` extension, even when its feature list is empty and it
has no client settings. A UI reads catalog descriptions from the list and
retrieves the selected model details through the same configured OpenAI client.
This keeps large schemas out of list responses and keeps internal or
secret-bearing runtime context out of discovery.

[OpenAI treats added response properties as backward-compatible](https://developers.openai.com/api/reference/overview#backwards-compatibility).
Direct JavaScript clients can read the property normally, and the
[OpenAI Python SDK exposes it through `model_extra`](https://github.com/openai/openai-python#making-customundocumented-requests).
An intermediary may rebuild a retrieved model from the standard fields and drop
extensions. For one LGOS deployment, clients must use one direct or
pass-through OpenAI base URL for model listing, model retrieval, and chat
completions. A federating gateway may expose a separate normalized catalog for
provider and model routing, but that catalog is not a source of LGOS
descriptions or capabilities. Clients must obtain those fields again through
the selected provider's direct or pass-through route. Request paths must also
preserve OpenAI metadata and extension-only stream chunks. Concrete gateway
configurations are documented under
[OpenAI-Compatible Proxies](../how-to-guides/openai-proxies.md).

!!! warning "Limited functionality signal"

    A missing description in model listing or missing or invalid
    `langgraph_openai_serve` metadata on model retrieval means the configured
    endpoint is not preserving the LGOS contract. A UI may continue ordinary
    Chat Completions, but it must visibly label the model or chat as **Limited
    functionality** and must not assume runtime settings, client events, or
    interrupts are available. A normalized routing catalog cannot remove this
    requirement.

## Runtime Settings

The request keeps each concern in its standard OpenAI location:

| Concern | OpenAI request location |
| --- | --- |
| System instructions | A `system` message |
| Small graph-specific values | One `metadata.langgraph_runtime_settings` string containing a JSON object |
| Graph selection | `model` |
| Caller-selected interrupt operation ID | Optional `metadata.langgraph_run_id` UUID |

Only small graph-specific values belong to `ClientSettings`. A graph may expose
controlled semantic choices such as intended audience, but not arbitrary system
instruction text. Client-authored system instructions remain ordinary graph-input
messages.

OpenAI metadata permits at most 16 string pairs, with keys up to 64 characters
and values up to 512 characters. Public settings consume one pair; a
caller-selected interrupt run consumes another. Clients use `json.dumps()` or
`JSON.stringify()` to encode the complete settings string and omit values equal
to the advertised defaults. The advertised JSON Schema describes the available
settings; LGOS remains the validation authority. The descriptor's separate
`defaults` object is the authoritative validated baseline; JSON Schema
`default` keywords are annotations and may precede Pydantic field
normalization. Native Chat Completions fields keep their standard semantics.
Graphs that need identity, authorization, database clients, secrets, or other
server-owned per-request context combine `client_settings` with
`context_factory(request, settings)`.

### Per-Request Resolution

Every chat completion starts from the registered defaults. Values supplied in
`metadata.langgraph_runtime_settings` replace matching top-level defaults, and LGOS
validates the complete result. The merge is shallow: a supplied nested object
replaces that whole default value rather than recursively merging its keys.

Client settings are not persisted between requests. The interrupt tool-call
envelope identifies durable state, but it does not restore runtime context.
Clients must resend non-default settings on every request that needs them,
including interrupt-resume requests. A later request that omits
`langgraph_runtime_settings` uses registered defaults again.

When the required extension is missing or unsupported, the client omits runtime
settings and shows the limited-functionality warning described above. See
[Configure LangGraph Runtime Settings](../how-to-guides/langgraph-runtime-settings.md)
for the complete author and client flow. Adapter support is summarized under
[demo client capability matrix](../demo/index.md#client-capabilities).

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

## Client Stream Events

Passive application notifications are an opt-in, namespaced extension on an
otherwise complete `chat.completion.chunk`. The graph must declare
`GraphFeature.CLIENT_EVENTS`, and the client requests v1 events through the
standard Chat Completions metadata field only when model retrieval advertises
`client_events`:

```python
stream = client.chat.completions.create(
    model="research-graph",
    messages=messages,
    stream=True,
    metadata={"langgraph_stream_events": "v1"},
)
```

An event frame has the following data payload:

```json
{
  "id": "chatcmpl-abc",
  "object": "chat.completion.chunk",
  "created": 1784280000,
  "model": "research-graph",
  "choices": [
    {
      "index": 0,
      "delta": {},
      "finish_reason": null
    }
  ],
  "langgraph_openai_serve": {
    "schema_version": 1,
    "event": {
      "type": "progress",
      "namespace": ["research"],
      "data": {
        "stage": "retrieval",
        "completed": 2,
        "total": 5,
        "message": "Searching documents"
      }
    }
  }
}
```

Event chunks reuse the completion ID, creation timestamp, and model. Choice `0`
has an empty delta and a null finish reason; the actual final chunk still uses
`stop` or `tool_calls`, and `[DONE]` is unchanged. Recognized public events are
emitted immediately among text chunks in LangGraph stream order. The namespace
is explicitly authored by the graph so dynamic task IDs and internal subgraph
structure do not become part of the public contract.

!!! note "Proxy compatibility"

    Schema-normalizing proxies may discard extension-only chunks because their
    delta is empty, while continuing to stream assistant text normally. Use a
    documented raw pass-through route when client events are required. See
    [OpenAI-Compatible Proxies](../how-to-guides/openai-proxies.md#client-event-compatibility)
    for verified Bifrost and LiteLLM behavior.

Without the graph feature and exact `v1` opt-in, LGOS emits no event extensions.
Even with both, only explicitly marked event envelopes in the shape produced by
`client_event()` or `status_event()` and revalidated by the server are exposed.
Ordinary LangGraph custom data, malformed events, debug data, and non-JSON
Python objects stay private. The v1 public event types are `status`, `progress`,
and `artifact`.

`status_event()` produces portable data with a user-facing `description` and
the booleans `done` and `hidden`. The graph emits meaningful application status
at the point where it knows what work is happening. LGOS does not infer status
from node names, graph topology, inputs, or results.

Keep standard response semantics separate:

| Graph result | Chat Completions representation |
| --- | --- |
| Assistant text | `delta.content` |
| Interrupt requiring input | `delta.tool_calls` |
| Citation | `delta.annotations` |
| Midstream failure | OpenAI error object |
| Passive status, progress, or artifact notification | `langgraph_openai_serve.event` |

Status updates are deliberately not encoded as `delta.tool_calls`. In OpenAI
[function calling](https://developers.openai.com/api/docs/guides/function-calling),
a tool call asks the client application to execute work and return a matching
tool message. A passive status only describes backend work already in progress.
UI adapters render it with native status components without changing the Chat
Completions tool protocol.

The published
[Chat Completions chunk schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events#chat.completion.chunk)
does not define arbitrary delta event fields. OpenAI's
[compatibility policy](https://developers.openai.com/api/reference/overview#backwards-compatibility)
treats added JSON response or event properties as backward-compatible, and the
[Python SDK preserves undocumented response properties in `model_extra`](https://github.com/openai/openai-python#making-customundocumented-requests).
Consume the events while iterating the stream; an SDK's accumulated final
completion is not the event log.

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
`param: "metadata.langgraph_runtime_settings"`. A proxy-stripped model
extension does not make standard chat invalid, but clients surface it as
limited functionality rather than silently presenting a fully capable model.
Malformed interrupt envelopes, a missing or duplicate tool result, and invalid
caller-supplied run UUIDs return HTTP 400. A structurally complete exchange that
does not match the durable pending set, or is stale or already completed,
returns HTTP 409 with `code: "interrupt_state_conflict"`. A request that cannot
acquire the active run lease returns HTTP 409 with `code: "run_busy"`.

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

Ordinary chats work in any client that implements the supported OpenAI Chat
Completions surface. Interrupt graphs are also OpenAI-wire-compatible, but they
require the client application to implement tool execution or human-approval
semantics. A UI that only renders assistant text cannot complete an interrupt.

### Operation Identity

An initial interrupt request does not require metadata. LGOS generates a UUID
operation ID and returns it inside every resulting tool call. A caller may
instead supply a non-nil UUID in `metadata.langgraph_run_id`; doing so lets it
retry an initial request deterministically if the response is lost. Reusing
that UUID while the run is pending re-emits the durable pending batch without
executing the interrupted nodes again. If the caller lets LGOS generate the UUID
and loses the first response, it has not learned an address for that pending
run; choose the UUID before sending whenever initial-response recovery matters.

Treat a caller-chosen UUID as single-use. LGOS deliberately deletes terminal
checkpoint state and keeps no tombstone, so a later ordinary initial request
with that UUID is indistinguishable from a new operation and can start again.
Only replaying the old assistant/tool resume ledger is fail-closed after
terminal deletion.

The public run UUID is not a UI chat ID. LGOS derives a fixed-length internal
checkpointer key from a server-trusted scope, the registered model, and the
operation, so two models or authenticated tenant scopes do not share state even
when callers use the same UUID. Configure the server scope from trusted request
or authentication state, never caller-controlled metadata or the Chat
Completions `user` field. The default shared scope is appropriate only for a
single-tenant or shared-trust deployment. Conversation history remains
client-owned; the checkpoint contains only the isolated workflow state needed
while this operation is paused.

The authenticated scope must remain stable between the initial request and all
resumes. A request resolved into another scope cannot address the pending
checkpoint, even if it presents the same public run UUID and tool ledger.

### Interrupt Tool Envelope

Every pending LangGraph interrupt becomes an OpenAI function tool call named
`langgraph_interrupt`. Its `arguments` string contains this JSON object:

```json
{
  "run_id": "f654e904-1bd8-4fd6-a8bf-53a49ca25699",
  "state_token": "47ecb7c6f7b9...",
  "payload": {
    "question": "Approve the refund?"
  }
}
```

The UI renders `payload` and otherwise preserves the arguments unchanged.
Treat `run_id`, `state_token`, and the tool-call ID as opaque protocol data. The
tool-call ID is `lg_interrupt_` followed by the LangGraph interrupt ID.

### Canonical Batch Replay

A resume request must end with the complete assistant message returned by LGOS,
including every `langgraph_interrupt` tool call, followed by exactly one `tool`
message for every call. Each result has the matching `tool_call_id` and JSON
content containing a `resume` value:

```json
{
  "model": "interruptible",
  "messages": [
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "lg_interrupt_6f719db61be2b8e875cc775f0f6c86aa",
          "type": "function",
          "function": {
            "name": "langgraph_interrupt",
            "arguments": "{\"run_id\":\"f654e904-1bd8-4fd6-a8bf-53a49ca25699\",\"state_token\":\"47ecb7c6f7b9...\",\"payload\":{\"question\":\"Approve the refund?\"}}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "lg_interrupt_6f719db61be2b8e875cc775f0f6c86aa",
      "content": "{\"resume\":\"approved\"}"
    }
  ]
}
```

Parallel interrupts are one atomic approval batch: the assistant message must
contain all pending calls, and the following messages must answer all of them.
A client must not select one call, mix ordinary tool calls into that exchange,
duplicate a result, or synthesize a partial replay. Streaming clients assemble
all tool-call deltas into the canonical assistant message before presenting the
batch.

The replayed arguments carry the resume operation ID. Metadata is not required
on a resume, but `metadata.langgraph_run_id`, when present, must match the UUID
in every replayed call.

The UI owns persistence of this canonical assistant/tool ledger. It must store
the exact calls before soliciting approval so a reconnect can reproduce the
same resume request. Persisting only rendered prompt text or only the user's
decision is insufficient.

### Durable Validation And Recovery

For an interrupt-enabled run, LGOS uses LangGraph exit durability and holds a
run-scoped coordinator lease while it reads state, validates a resume, and
executes the graph. Same-key contention is rejected instead of queued. Exit
durability stores state when the invocation pauses or finishes without
retaining every intermediate superstep.
LGOS drains the invocation before it exposes interrupt tool calls. It compares
the replayed pending IDs and opaque state token with the durable checkpoint
before passing answers to LangGraph. The replayed display payload is never used
as graph input.
Concurrent work for another operation remains independent; a second request
for the same operation receives HTTP 409.

LGOS preserves checkpoint state only after it produces an interrupt batch for
the client. It deletes the isolated thread after terminal completion and
best-effort after failure or cancellation before a batch. Cleanup failure can
leave an unreachable thread for operators to reap; it never replaces the
original execution error. If the terminal HTTP response is lost,
replaying the old resume returns a safe HTTP 409 and does not re-execute the
completed operation. This is conflict detection, not durable storage of the
terminal response; applications that need result replay must add a
result/idempotency store at their own boundary.

An interrupted node restarts from its beginning when resumed. Any side effect
before `interrupt()` can therefore run again; make it idempotent or move it
after the interrupt. This is a LangGraph execution rule, documented in the
official [interrupt guidance](https://docs.langchain.com/oss/python/langgraph/interrupts#rules-of-interrupts).
Moving work after approval avoids that normal interrupt replay, but it does not
make an external side effect exactly once: a process can still fail after the
effect succeeds and before its task result is durably recorded. Put external
effects in durable tasks and give the downstream operation an idempotency key
when duplicates are unacceptable; LangGraph's
[idempotency guidance](https://docs.langchain.com/oss/python/langgraph/functional-api#idempotency)
describes that remaining crash window. The coordinator prevents overlapping
run execution, not crash-time exactly-once delivery.

Pending runs abandoned by users remain checkpoint data. Production operators
must define an expiry policy that accounts for the maximum approval window and
deletes expired checkpoint threads through the checkpointer; do not treat
ordinary database backups or retention as an active-run cleanup policy. See
LangGraph's [persistence documentation](https://docs.langchain.com/oss/python/langgraph/persistence)
for the underlying checkpoint model.

## Known Differences From OpenAI

- `model` selects a registered LangGraph graph, not an OpenAI-hosted model.
- The supported surface focuses on chat completions, model listing/retrieval,
  health, and compatible tool-call flows.
- Authentication is not enforced by default.
- Token usage is approximate.
