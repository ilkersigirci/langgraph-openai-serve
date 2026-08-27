# OpenAI Clients

Configure clients with the server base URL, usually `http://localhost:8000/v1`.
The `api_key` value is sent as `Authorization: Bearer <key>`; the application
from [Get Started](../getting-started.md) does not verify it.

For one LGOS deployment, use one OpenAI client and base URL for model listing,
model retrieval, and chat completions. When a proxy is present, that URL must be
its complete LGOS pass-through route. A gateway that federates multiple LGOS
providers may expose a separate routing catalog; use it only to discover
provider-qualified IDs, then use pass-through for LGOS metadata and chat. The
[Bifrost demo](../demo/bifrost.md) shows that split.

The basic examples below call that application's `echo` model. Examples named
`my-graph`, `my-settings-graph`, or `research-graph` describe capabilities your
registered graph must enable. The [demo graph catalog](../demo/graphs/index.md)
provides runnable models for those advanced behaviors.

## Install A Client

=== "Python"

    ```bash
    pip install openai
    ```

=== "JavaScript"

    ```bash
    npm install openai
    ```

## Chat Completions

!!! warning "Do not expose real API keys in a browser"

    The JavaScript examples enable `dangerouslyAllowBrowser` because the local
    examples use a dummy key. Keep production credentials in server-side code.

=== "Python"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

    response = client.chat.completions.create(
        model="echo",
        messages=[{"role": "user", "content": "Hello from Python"}],
    )

    print(response.choices[0].message.content)
    ```

=== "Python Streaming"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

    stream = client.chat.completions.create(
        model="my-graph",
        messages=[{"role": "user", "content": "Write a short poem about graphs."}],
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="")
    ```

=== "JavaScript"

    ```javascript
    import OpenAI from "openai";

    const openai = new OpenAI({
      baseURL: "http://localhost:8000/v1",
      apiKey: "DUMMY",
      dangerouslyAllowBrowser: true,
    });

    const completion = await openai.chat.completions.create({
      model: "echo",
      messages: [{ role: "user", content: "Hello from JavaScript" }],
    });

    console.log(completion.choices[0].message.content);
    ```

=== "JavaScript Streaming"

    ```javascript
    import OpenAI from "openai";

    const openai = new OpenAI({
      baseURL: "http://localhost:8000/v1",
      apiKey: "DUMMY",
      dangerouslyAllowBrowser: true,
    });

    const stream = await openai.chat.completions.create({
      model: "my-graph",
      messages: [{ role: "user", content: "Write a short poem about graphs." }],
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      process.stdout.write(content);
    }
    ```

## Client Stream Events

First retrieve the model and confirm that
`langgraph_openai_serve.features` contains `client_events`. Then request
explicitly public graph events with the standard metadata field. The Python SDK
keeps the namespaced extension in each chunk's `model_extra`:

```python
model = client.models.retrieve("research-graph")
model_extension = (model.model_extra or {}).get("langgraph_openai_serve")
features = (
    model_extension.get("features") if isinstance(model_extension, dict) else None
)
valid_extension = (
    isinstance(model_extension, dict)
    and model_extension.get("schema_version") == 1
    and isinstance(features, list)
    and all(isinstance(feature, str) for feature in features)
)
if not valid_extension:
    show_limited_functionality_warning()

event_metadata = (
    {"langgraph_stream_events": "v1"}
    if valid_extension and "client_events" in features
    else {}
)

stream = client.chat.completions.create(
    model="research-graph",
    messages=[{"role": "user", "content": "Research this topic."}],
    stream=True,
    metadata=event_metadata,
)

for chunk in stream:
    extension = (chunk.model_extra or {}).get("langgraph_openai_serve")
    if isinstance(extension, dict) and extension.get("schema_version") == 1:
        event = extension.get("event")
        if isinstance(event, dict):
            handle_client_event(event)

    for choice in chunk.choices:
        if choice.delta.content:
            print(choice.delta.content, end="")
```

When using the higher-level streaming helper, inspect its raw `ChunkEvent`:

```python
with client.chat.completions.stream(
    model="research-graph",
    messages=[{"role": "user", "content": "Research this topic."}],
    metadata=event_metadata,
) as stream:
    for item in stream:
        if item.type != "chunk":
            continue
        extension = (item.chunk.model_extra or {}).get(
            "langgraph_openai_serve"
        )
        if isinstance(extension, dict) and extension.get("schema_version") == 1:
            event = extension.get("event")
            if isinstance(event, dict):
                handle_client_event(event)
```

The helper emits a raw chunk event for every Chat Completions chunk. Consume
LGOS events during iteration; do not expect `get_final_completion()` to retain
them. See the OpenAI SDK's
[Chat Completions event reference](https://github.com/openai/openai-python/blob/main/helpers.md#chat-completions-events)
and the LGOS [wire contract](../explanation/openai-compatibility.md#client-stream-events).

Portable `status` events contain a user-facing `description` plus `done` and
`hidden` booleans. Treat them as passive UI updates, and stop the active
indicator when `done` is true. Do not execute them as OpenAI tool calls; the
backend graph owns the work.

## Model Discovery And Runtime Settings

List model summaries and read descriptions from their lightweight LGOS
extensions, then retrieve the selected model to discover its settings. Check
both the LGOS extension version and the nested runtime-settings version. A valid
detail extension without `client_settings` means the model has no public
settings. A missing description or an invalid detail extension means the
configured endpoint is degraded: keep standard chat available, omit extended
behavior, and show **Limited functionality** rather than silently treating the
model as fully capable.

=== "Python"

    ```python
    models = client.models.list()
    selected = next(
        model for model in models.data if model.id == "my-settings-graph"
    )
    summary_extension = (selected.model_extra or {}).get(
        "langgraph_openai_serve"
    )
    description = (
        summary_extension.get("description")
        if isinstance(summary_extension, dict)
        and summary_extension.get("schema_version") == 1
        else None
    )
    print(description)

    model = client.models.retrieve(selected.id)

    extension = (model.model_extra or {}).get("langgraph_openai_serve")
    settings = (
        extension.get("client_settings")
        if isinstance(extension, dict) and extension.get("schema_version") == 1
        else None
    )

    if isinstance(settings, dict) and settings.get("schema_version") == 1:
        print(settings["json_schema"])
        print(settings["defaults"])
    ```

=== "JavaScript"

    ```javascript
    const models = await openai.models.list();
    const selectedModel = models.data.find(
      (model) => model.id === "my-settings-graph",
    );
    if (!selectedModel) throw new Error("my-settings-graph is not registered");
    const summaryExtension = selectedModel.langgraph_openai_serve;
    const description =
      summaryExtension?.schema_version === 1
        ? summaryExtension.description
        : undefined;
    console.log(description);

    const model = await openai.models.retrieve(selectedModel.id);

    const extension = model.langgraph_openai_serve;
    const settings =
      extension?.schema_version === 1 &&
      extension.client_settings?.schema_version === 1
        ? extension.client_settings
        : undefined;

    if (settings) {
      console.log(settings.json_schema);
      console.log(settings.defaults);
    }
    ```

`metadata.langgraph_runtime_settings` must be a JSON-encoded string, produced by
`json.dumps()` or `JSON.stringify()`, rather than a nested metadata object. Send
only values that differ from the discovered defaults; the encoded value must be
512 characters or fewer. See
[Client Request](../how-to-guides/langgraph-runtime-settings.md#client-request)
for the request shape.

Settings apply to one request. Resend non-default values whenever they are
needed, including interrupt-resume requests. Omitting the metadata on a later
request uses server defaults again.

## Interrupt Resume

Interrupt-enabled graphs use OpenAI tool calls. Retrieve the selected model and
check `langgraph_openai_serve.features` for `interrupts` before starting. No
metadata is required for a first request: LGOS generates a UUID operation ID and
places it in the interrupt arguments. Supply your own non-nil UUID in
`metadata.langgraph_run_id` when an initial request might be retried after its
response is lost. Generate a new UUID per operation; terminal checkpoints are
deleted, so LGOS retains no tombstone that could reject a later ordinary initial
request reusing the same UUID.

```python
import json
from uuid import uuid4

run_id = str(uuid4())
metadata = {"langgraph_run_id": run_id}
messages = [
    {"role": "user", "content": "Perform the protected action"}
]

paused = client.chat.completions.create(
    model="interruptible",
    messages=messages,
    metadata=metadata,
)
assistant = paused.choices[0].message
tool_calls = assistant.tool_calls or []
if not tool_calls:
    raise RuntimeError("The graph completed without interrupting")

# Render every pending call to the user, then collect one decision per call.
decisions = {
    tool_call.id: "approved" for tool_call in tool_calls
}

assistant_message = {
    "role": "assistant",
    "content": assistant.content,
    "tool_calls": [
        tool_call.model_dump(mode="json") for tool_call in tool_calls
    ],
}
tool_messages = [
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps({"resume": decisions[tool_call.id]}),
    }
    for tool_call in tool_calls
]

completed = client.chat.completions.create(
    model="interruptible",
    messages=[assistant_message, *tool_messages],
    metadata=metadata,
)
```

The resume request must replay the complete assistant `tool_calls` message,
unchanged, followed by exactly one `tool` message for each call. Answer all
parallel interrupts in one batch; do not resume a subset. For streaming, first
assemble the complete assistant tool-call message from every delta and persist
that canonical message before asking for decisions.

The run ID comes from the tool arguments, making resume metadata optional; if
resent, it must match. Runtime settings remain per-request. Dedicated approval
clients should use `OpenAI(max_retries=0)` (or JavaScript `maxRetries: 0`)
because a lost terminal response is not reconstructable from the deleted
checkpoint.

See [OpenAI compatibility](../explanation/openai-compatibility.md#tool-calls-and-interrupts)
for validation errors, stale-state conflicts, and recovery rules.

## Diagnostics

??? example "Direct HTTP diagnostic"

    Use direct HTTP only to inspect behavior while debugging:

    ```bash
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "echo",
        "messages": [{"role": "user", "content": "Hello from HTTP"}]
      }'
    ```

## Notes

- Use the registered graph name as `model`.
- Set timeouts for long-running graphs.
- Use streaming only for graphs configured to emit streamed chunks.
- Add bearer-token authentication before exposing the API outside trusted
  development environments.
