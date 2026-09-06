# OpenAI API Compatibility

LangGraph OpenAI Serve is an OpenAI-client compatibility layer, not a separate
LangGraph-specific HTTP API. Public Responses, Chat Completions, and model
behavior remain reachable through the configured OpenAI-compatible base URL.

The same contract lets LGOS run behind OpenAI-compatible intermediaries without
a project-specific inference adapter. Generic gateway requirements are in the
[proxy guide](../how-to-guides/openai-proxies.md); concrete Chainlit, Open WebUI,
Files, LiteLLM, and Bifrost implementations belong to the
[Demo Stack](../demo/index.md).

## Contract

- Registered graph names are exposed as OpenAI `model` values.
- Core graph behavior must fit OpenAI-compatible request fields, Responses
  items, Chat objects, tool calls, streaming events, metadata, or error
  envelopes.
- Do not require custom payloads, headers, routes, or SSE event shapes for core
  behavior unless the OpenAI client path still works.
- Treat direct HTTP calls such as `curl` as diagnostics; validate compatibility
  through OpenAI client behavior.

The implemented endpoints are listed in [Reference](../reference.md).

## Model Feature Discovery

The [OpenAI Model object](https://developers.openai.com/api/reference/resources/models)
has no `metadata` field. LGOS keeps its standard fields unchanged and places
feature discovery in a namespaced, versioned extension on model-list and
model-retrieval responses. Runtime-settings discovery remains detail-only:

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
enable behavior, while model listing and retrieval serialize it for discovery.
`GraphConfig.client_settings` is an explicit, allowlisted public Pydantic model;
LGOS never publishes a graph's internal LangGraph context schema automatically.
Additive features do not require an outer schema-version change. The nested
runtime settings descriptor has its own version, and clients must ignore
versions they do not understand.

| Feature | Enabled behavior |
| --- | --- |
| `client_events` | Streaming Responses may emit status commentary. Chat Completions ignores client events. |
| `file_inputs` | The graph accepts native file parts and resolves their opaque `file_id` values. |
| `interrupts` | The server supports the checkpointed interrupt/resume flow. |

`GET /v1/models` remains lightweight. Every entry contains the standard `id`,
`object`, `created`, and `owned_by` fields plus a small
`langgraph_openai_serve` object with `schema_version`, `description`, and
`features`. Client-settings schemas remain detail-only.
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
extensions. For one LGOS deployment, a client can use one OpenAI base URL for
model listing, model retrieval, Responses, and Chat Completions; that URL may
be a proxy pass-through such as the demo's LiteLLM `/v1/lgos-a` catalog route. A
federating gateway may expose a normalized catalog for provider and model
routing, but that catalog is not necessarily a source of LGOS descriptions or
capabilities. Standard Responses requests do not depend on the extension. A UI
that offers graph-specific settings or capability controls must retrieve the
selected provider's detail object through a route that preserves it. Concrete
gateway configurations and native Responses requirements are documented under
[OpenAI-Compatible Proxies](../how-to-guides/openai-proxies.md).

!!! warning "Limited functionality signal"

    A missing description in model listing or missing or invalid
    `langgraph_openai_serve` metadata on model retrieval means the configured
    endpoint is not preserving the optional LGOS discovery contract. A UI may
    continue plain Responses text, but it must visibly label the model or chat
    as **Limited functionality** and must not assume runtime settings, file
    inputs, status commentary, or interrupts are available. A normalized
    routing catalog cannot remove this requirement.

## Runtime Settings

The request keeps each concern in its standard OpenAI location:

| Concern | OpenAI request location |
| --- | --- |
| System instructions | Responses `instructions` or an input `system`/`developer` message; a `system` message in Chat |
| Small graph-specific values | One `metadata.langgraph_runtime_settings` string containing a JSON object |
| Graph selection | `model` |
| Caller-selected interrupt operation ID | Optional `metadata.langgraph_run_id` UUID |
| Conversation correlation | Optional `metadata.session_id` string |

Only small graph-specific values belong to `ClientSettings`. A graph may expose
controlled semantic choices such as intended audience, but not arbitrary system
instruction text. Client-authored system instructions remain ordinary graph-input
messages.

OpenAI metadata permits at most 16 string pairs, with keys up to 64 characters
and values up to 512 characters. Public settings consume one pair; a
caller-selected interrupt run or conversation correlation value consumes
another. Clients use `json.dumps()` or `JSON.stringify()` to encode the complete
settings string and omit values equal to the advertised defaults. The
advertised JSON Schema describes the available settings; LGOS remains the
validation authority. The descriptor's separate `defaults` object is the
authoritative validated baseline; JSON Schema `default` keywords are annotations
and may precede Pydantic field normalization. Native OpenAI fields keep their
standard semantics. Graphs that need identity, authorization,
database clients, secrets, or other server-owned per-request context combine
`client_settings` with `context_factory(request, settings)`.

`metadata.session_id` is an optional, UI-neutral correlation value. A client
uses the same stable value for every Responses or Chat Completions request in
one conversation. LGOS maps it to the Langfuse-recognized
`RunnableConfig.metadata.langfuse_session_id`; each request remains a separate
trace, while Langfuse can group those traces in one
[session](https://langfuse.com/docs/observability/features/sessions). It does
not select checkpoint state or cause LGOS to retain conversation history.
Clients targeting Langfuse should use an ASCII value shorter than 200
characters. The value is distinct from the OpenAI `user` field,
`metadata.langgraph_run_id`, and per-request trace or request identifiers.

### Per-Request Resolution

Every graph request starts from the registered defaults. Values supplied in
`metadata.langgraph_runtime_settings` replace matching top-level defaults, and LGOS
validates the complete result. The merge is shallow: a supplied nested object
replaces that whole default value rather than recursively merging its keys.

Client settings are not persisted between requests. The paused Response ID and
interrupt call IDs identify durable state, but they do not restore runtime context.
Clients must resend non-default settings on every request that needs them,
including interrupt-resume requests. A later request that omits
`langgraph_runtime_settings` uses registered defaults again.

When the required extension is missing or unsupported, the client omits runtime
settings and shows the limited-functionality warning described above. See
[Configure LangGraph Runtime Settings](../how-to-guides/langgraph-runtime-settings.md)
for the complete author and client flow. Adapter support is summarized under
[demo client capability matrix](../demo/index.md#client-capabilities).

## Message And Schema Adaptation

Incoming OpenAI messages are converted to LangChain messages. The protocol
decoder also produces the small, protocol-neutral `GraphRequest` received by
`GraphConfig` adapters, keeping custom LangGraph schemas behind the public API
boundary. See
[LangGraph Integration](langgraph-integration.md#adaptation) and
[Custom Graphs](../tutorials/custom-graphs.md#custom-schemas).

Responses `input_file.file_id` content and native Chat Completions file parts
normalize to the same LangChain file content. LGOS does not expose Files routes
or own file storage. A client uploads through an external OpenAI-compatible
Files API and sends the returned `file_id` to a graph; the graph still owns file
interpretation. See [Accept And Display Files](../how-to-guides/file-inputs.md).

## Supported Responses Subset

`POST /v1/responses` implements stateless text, files, function calls, and
streaming over the same graph runner as Chat Completions. It intentionally does
not claim every field in the upstream OpenAI API.

| Request field or item | LGOS behavior |
| --- | --- |
| `model`, `input`, `instructions` | Supported. String input and ordered user, system, developer, and replayed assistant messages become LangChain messages. New `instructions` are rejected on interrupt resumes because a paused invocation cannot consume them. |
| `input_text` | Supported. |
| `input_file.file_id` | Supported and normalized to the existing graph file block. |
| function `tools`, `tool_choice`, `parallel_tool_calls` | Supported for client-owned functions. |
| `function_call` and string-valued `function_call_output` | Supported for ordinary client-tool continuation. Interrupt resumes accept only `function_call_output` items with `previous_response_id`. |
| `metadata`, `user` | Supported and passed through the protocol-neutral graph request boundary. They are not authentication. |
| `stream` | Supported with typed Responses SSE events. |
| `store` | Omitted and false mean false; true is rejected. |
| `text.format.type="text"` | Supported. |
| `tools=[{"type": "custom", "name": "lgos_..."}]` | LGOS hosted-tool selector; selected graph must declare each identifier. |
| `previous_response_id` | Supported for interruptible graphs to resume from an interrupted state. Rejected for non-interruptible graphs. |
| `conversation`, `background: true` | Rejected because LGOS has no Responses conversation store or background lifecycle. |
| `include`, reasoning, generation controls, service tier, stream options, reusable prompts, prompt-cache fields, truncation | Rejected rather than accepted without semantics. |

Unknown request fields also fail validation. Exact errors use the standard
OpenAI envelope and identify the unsupported parameter where it is known.

### Stateless Item Continuation

LGOS generates an opaque Response ID for correlation but does not persist it.
There are no response retrieve, delete, cancel, compact, or input-item routes.
Clients therefore keep an input ledger and resend the items needed by the next
turn instead of using a server-side Conversation.

When continuing a function call, append every item from `response.output`
unchanged and then append a matching `function_call_output`. Replaying complete
SDK items preserves message and call IDs plus assistant `phase`. The current
SDK may serialize optional function-call `caller` and `namespace` fields as
null; LGOS accepts those null values but rejects non-null program or namespace
semantics. This state model follows OpenAI's documented manual item replay while
keeping storage in the client.

An interrupt continuation uses a narrower stateful path. The client sends the
paused Response ID as `previous_response_id` and sends only matching
`function_call_output` items. LGOS uses those opaque IDs to locate and validate
the paused checkpoint; it does not reconstruct ordinary conversation history.

LangGraph checkpoint and Store persistence are separate. A checkpointer keeps
only paused workflow execution; a graph Store keeps explicit application data.
Neither makes a Response ID retrievable or lets LGOS reconstruct a conversation.

### Responses Output

| Graph result | Responses representation |
| --- | --- |
| Final assistant text | Completed message item with `phase="final_answer"` and `output_text` content |
| Visible streaming status | Separate completed message item with `phase="commentary"` |
| Client tool or interrupt | One `function_call` item per call |
| Tool result on the next request | Matching `function_call_output` item |
| URL citation | `url_citation` annotation on `output_text` |
| Provider-reported usage | `usage` on the completed Response |

Function arguments are complete JSON strings. The graph runner does not emit
incremental arguments, so the Responses stream sends one argument delta before
the corresponding done event. IDs and output indices remain stable throughout
the typed event lifecycle.

## Streaming

Streaming responses use OpenAI-compatible Server-Sent Events. See
[LangGraph Integration](langgraph-integration.md#runner-behavior) for internal
event handling and [Request Cancellation](langgraph-integration.md#request-cancellation)
for request-scoped disconnect cancellation, proxy behavior, and cooperative
limits.

LGOS aggregates usage reported by LangChain model calls across the graph run.
Complete Responses include it in `usage`, and a Responses stream carries it on
the terminal `response.completed` object. Chat streams add the standard final
empty-choices usage chunk only when the request sets
`stream_options={"include_usage": true}`. When underlying providers report no
usage, LGOS omits it rather than estimating tokens.

### Assistant Text Parity

The final rendered `AIMessage.text` is the canonical assistant text.
Non-streaming returns it directly. Streaming emits eligible message chunks
immediately and retains them until the final message arrives. It then
concatenates the chunks and compares them with the final text. If no text
streamed, LGOS emits the final text as a fallback; a mismatch instead produces
the protocol's failure sequence rather than a successful terminal event. This
check covers one graph run, not two independent LLM executions. Transient
status events are excluded.

When multiple streamable nodes contribute text, the graph's
`output_to_message` adapter must render their messages in the same order.

## Streaming Status

The graph must declare `GraphFeature.CLIENT_EVENTS` before any public client
event can cross an HTTP route. Ordinary LangGraph custom data, malformed events,
debug values, and non-JSON Python objects stay private. Responses exposes only
validated `status_event()` values. Chat Completions ignores custom events.

### Responses Commentary

A streaming Responses request needs no metadata opt-in. LGOS maps every visible
status description to its own completed assistant message with
`phase="commentary"` and maps the durable answer to a message with
`phase="final_answer"`. A status whose graph-owned `hidden` flag is true is
suppressed. The custom namespace and `done` flag do not leak into the Response;
item completion is a wire lifecycle concept, not graph progress state.

Commentary is transient and streaming-only. Non-streaming execution calls the
graph once for its durable result and does not collect status history. The
OpenAI Python SDK's `Response.output_text` convenience property concatenates
text across both phases, so UIs must select `final_answer` messages for
the transcript and render commentary separately. The maintained Chainlit and
Open WebUI adapters do this. Other clients may ignore `phase` or show all text
as one answer; that is a client presentation limitation, not a reason to add a
custom server event.

### Chat Completions vs Responses Boundary

Complex workflow features—such as streaming status commentary, checkpointed
persistence, and human-in-the-loop interrupts—are exclusively available through
the native Responses API (`/v1/responses`).

The Chat Completions API (`/v1/chat/completions`) provides strict, standard OpenAI
compatibility for simple graphs and tool calling. It streams plain text
`delta.content` chunks and ignores custom streaming events. Interrupt-enabled
models requested via Chat Completions fail fast with HTTP 400 Bad Request
indicating that interrupts require the Responses API.

| Graph result | Responses | Chat Completions |
| --- | --- | --- |
| Assistant text | `final_answer` message | `delta.content` |
| Interrupt requiring input | `function_call` item | Unsupported (HTTP 400) |
| Citation | `output_text.annotations` | message/final-delta annotations |
| Passive status | `commentary` message | Ignored |
| Diagnostic progress or artifact | Ignored | Ignored |
| Midstream failure | `error` then `response.failed` | OpenAI error object |

Status is deliberately not a tool call. In OpenAI
[function calling](https://developers.openai.com/api/docs/guides/function-calling),
a function call asks the client to execute work and return a result. A passive
status describes backend work already in progress.

## Citation Ownership

OpenAI `url_citation` annotations are the canonical citation contract. Their
URL, title, and text span associate a source with the answer. `end_index` is
inclusive, matching OpenAI's last-character convention.

Graphs attach LangChain citation annotations to their final `AIMessage`.
Responses returns them on `output_text.annotations` and emits
`response.output_text.annotation.added` during streaming. Chat Completions
returns them as `message.annotations` and as an extension on the final delta.
LGOS does not define a UI-specific source schema or reconstruct citations from
custom events.

Portable resource presentation belongs in the assistant text, not in the
annotation object. Graphs may return ordinary Markdown links and images in
`message.content`, including visible inline citation markers. Annotations do not
require clients to synthesize marker text. When a graph also emits structured
attribution, its `url_citation` remains limited to its standard URL, title, and
text span. Audio and video resources should use ordinary Markdown links rather
than UI-specific players. RAG graphs must preserve only resource URLs supplied
by their retrieved context and must not invent or rewrite them.

Structured citations remain available to OpenAI clients that need
machine-readable provenance. The `citation-events` demo showcases that optional
contract.

Only the Chat streaming field is a compatibility extension because the
published Chat delta schema does not declare annotations. Responses annotations
and their typed streaming event are standard fields.

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
extension does not make plain text generation invalid, but clients surface it
as limited functionality rather than silently presenting a fully capable
model.
Malformed interrupt envelopes, a missing or duplicate tool result, and invalid
caller-supplied run UUIDs return HTTP 400. A structurally complete exchange that
does not match the durable pending set, or is stale or already completed,
returns HTTP 409 with `code: "interrupt_state_conflict"`. A request that cannot
acquire its interrupt-run lease returns HTTP 409 with `code: "run_busy"`.

## Tool Calls And Interrupts

Tool definitions are accepted for OpenAI compatibility. Graphs can read their
normalized function definitions and choices from `GraphRequest` in
`request_to_input` or load tools independently, as the mock MCP demo does.

Responses accepts flat function tool definitions, named or automatic tool
choice, returned `function_call` items, and matching string-valued
`function_call_output` items. Chat supports the modern nested `tools`,
`tool_choice`, assistant `tool_calls`, and `tool` messages with matching
`tool_call_id` values. The deprecated Chat `functions`, singular
`function_call`, and `function` message role are rejected rather than silently
ignored.

Interrupt graphs require a client application that can collect and submit tool
results. Interrupts and checkpoint resumes are supported exclusively via
the Responses API (`/v1/responses`). Requesting an interrupt-enabled model via
Chat Completions returns HTTP 400 Bad Request.

### Hosted Tools

Responses accepts LGOS hosted-tool selectors matching the OpenAI custom tool shape,
such as `tools=[{"type": "custom", "name": "lgos_current_time"}]`. Function schemas and
execution remain server-owned.

A graph declares its supported identifiers in `GraphConfig.hosted_tools` and
reads the selected identifiers from `GraphRequest.hosted_tools`. Identifiers
must match `lgos_[a-z][a-z0-9_]*`. Unknown or unavailable selectors return HTTP
400 with the offending `tools.N.name` parameter before execution or streaming.
The package validates selection; the graph binds and executes its own tools
using its native agent implementation. Internal calls do not become client-owned
`function_call` items. Responses echoes the selectors as standard `CustomTool`
objects in `tools`.

Responses accepts the selector directly in the standard `tools` parameter:

```python
response = client.responses.create(
    model="hosted-tool",
    input="What time is it in Istanbul?",
    tools=[{"type": "custom", "name": "lgos_current_time"}],
    store=False,
)
```

Because `custom` is a standard OpenAI Responses tool type, proxies such as
Bifrost preserve this selection across normalized `/openai/v1` routes as well
as passthrough routes. See the [hosted-tool demo](../demo/graphs/hosted-tool.md).

### Files And `display_file`

Portable generated files use the standard Files API plus a client-owned
function. The client offers strict `display_file` arguments; the graph uploads
the bytes and returns their `file_id`; the trusted client backend downloads and
persists the file through its native UI; and the client appends a small matching
`function_call_output`. Neither file bytes nor a protected bearer URL are placed
in the transcript. There is no LGOS artifact field or custom chart event. See
[Accept And Display Files](../how-to-guides/file-inputs.md#display-a-graph-generated-file).

### Operation Identity

An initial interrupt request does not require metadata. LGOS generates a UUID
operation ID and embeds it in the paused Response ID. A caller may
instead supply a non-nil UUID in `metadata.langgraph_run_id`; doing so lets it
retry an initial request deterministically if the response is lost. Reusing
that UUID while the run is pending re-emits the durable pending batch without
executing the interrupted nodes again. If the caller lets LGOS generate the UUID
and loses the first response, it has not learned an address for that pending
run; choose the UUID before sending whenever initial-response recovery matters.

Treat a caller-chosen UUID as single-use. LGOS deliberately deletes terminal
checkpoint state and keeps no tombstone, so a later ordinary initial request
with that UUID is indistinguishable from a new operation and can start again.
Only resubmitting the old paused Response ID and call outputs is fail-closed
after terminal deletion.

The public run UUID is not a UI chat ID. LGOS derives a fixed-length internal
checkpointer key from a server-trusted scope, the registered model, and the
operation, so two models or authenticated tenant scopes do not share state even
when callers use the same UUID. Configure the server scope from trusted request
or authentication state, never caller-controlled metadata or the OpenAI `user`
field. The default shared scope is appropriate only for a
single-tenant or shared-trust deployment. Conversation history remains
client-owned; the checkpoint contains only the isolated workflow state needed
while this operation is paused.

The authenticated scope must remain stable between the initial request and all
resumes. A request resolved into another scope cannot address the pending
checkpoint, even if it presents the same public run UUID and continuation IDs.

### Interrupt Tool Envelope

Every pending LangGraph interrupt becomes an OpenAI function tool call named
`langgraph_interrupt`. Its `arguments` string contains the JSON payload directly:

```json
{
  "question": "How should the refund be handled?",
  "choices": ["approve", "reject"],
  "allow_other": true
}
```

Response and call IDs are opaque. The Response ID locates the paused operation;
each call ID binds an interrupt to that exact checkpoint generation. Clients
must persist and return both values unchanged.

### Resuming an Interrupt

Clients can resume using standard OpenAI `previous_response_id`:

```json
{
  "model": "interruptible",
  "previous_response_id": "resp_lg_f654e9041bd84fd6a8bf53a49ca25699_0123456789abcdef0123456789abcdef",
  "input": [
    {
      "type": "function_call_output",
      "call_id": "call_lg_47ecb7c6f7b901230fc4d3119976daae11888d39c973953060b8a849c3d8a5f2_6f719db6-1be2-4b8e-875c-c775f0f6c86a",
      "output": "Verify the delivery address first."
    }
  ],
  "store": false
}
```

Parallel interrupts are one atomic interrupt batch: the resume request must answer
all of them. A client must not select one call, mix ordinary function calls into that
request, duplicate a result, or synthesize a call ID. Streaming clients persist
the terminal Response ID and completed function-call items instead of reconstructing
them from argument deltas.

Metadata is not required on a resume, but `metadata.langgraph_run_id`, when
present, must match the operation encoded by `previous_response_id`.

The UI owns persistence of the paused Response ID and exact calls. It must store
them before soliciting input so a reconnect can reproduce the same resume request.
Persisting only rendered prompt text or only the user's response is insufficient.

### Durable Validation And Recovery

For an interrupt-enabled run, LGOS uses LangGraph exit durability and holds a
run-scoped coordinator lease while it reads state, validates a resume, and
executes the graph. Same-key contention is rejected instead of queued. Exit
durability stores state when the invocation pauses or finishes without
retaining every intermediate superstep.
LGOS drains the invocation before it exposes interrupt tool calls. It compares
the submitted pending IDs and opaque state token with the durable checkpoint
before passing answers to LangGraph. The displayed interrupt payload is not part
of the resume input.
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
Moving work after `interrupt()` avoids replaying it when the node restarts, but
it does not make an external side effect exactly once: a process can still fail
after the effect succeeds and before its task result is durably recorded. Put
external effects in durable tasks and give the downstream operation an
idempotency key when duplicates are unacceptable; LangGraph's
[idempotency guidance](https://docs.langchain.com/oss/python/langgraph/functional-api#idempotency)
describes that remaining crash window. The coordinator prevents overlapping
run execution, not crash-time exactly-once delivery.

Pending runs abandoned by users remain checkpoint data. Production operators
must define an expiry policy that accounts for the maximum response window and
deletes expired checkpoint threads through the checkpointer; do not treat
ordinary database backups or retention as an active-run cleanup policy. See
LangGraph's [persistence documentation](https://docs.langchain.com/oss/python/langgraph/persistence)
for the underlying checkpoint model.

## Known Differences From OpenAI

- `model` selects a registered LangGraph graph, not an OpenAI-hosted model.
- Responses implements the explicit subset above; response storage,
  Conversations, general previous-response chaining, background work,
  OpenAI-hosted tools, structured output, and unconsumed generation controls are
  rejected. `previous_response_id` is reserved for interrupt continuation.
- Chat Completions remains a direct compatibility surface, while maintained
  demo UIs use Responses for every graph.
- The package exposes model listing/retrieval and health, but no Files storage;
  deploy a separate OpenAI-compatible Files service when graphs use file IDs.
- Authentication is not enforced by default.
- Token usage is present only when underlying LangChain model calls report it.
  LGOS aggregates reported usage across the graph run and never estimates
  missing counts.
