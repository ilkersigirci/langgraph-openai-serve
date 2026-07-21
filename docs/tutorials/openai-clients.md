# OpenAI Clients

Configure clients with the server base URL, usually `http://localhost:8000/v1`.
The `api_key` value is sent as `Authorization: Bearer <key>`; the application
from [Get Started](../getting-started.md) does not verify it.

The basic examples below call that application's `echo` model. Examples named
`my-graph`, `my-settings-graph`, or `research-graph` describe capabilities your
registered graph must enable. The [demo graph catalog](../demo/graphs.md)
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

Request explicitly public graph events with the standard metadata field. The
Python SDK keeps the namespaced extension in each chunk's `model_extra`:

```python
stream = client.chat.completions.create(
    model="research-graph",
    messages=[{"role": "user", "content": "Research this topic."}],
    stream=True,
    metadata={"langgraph_stream_events": "v1"},
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
    metadata={"langgraph_stream_events": "v1"},
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

## Model Discovery And Runtime Settings

List standard model summaries, then retrieve the selected model to discover its
settings. Check both the LGOS extension version and the nested runtime-settings
version. A missing or unsupported descriptor means the client should use server
defaults.

=== "Python"

    ```python
    models = client.models.list()
    model_id = next(
        model.id for model in models.data if model.id == "my-settings-graph"
    )
    model = client.models.retrieve(model_id)

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
check `langgraph_openai_serve.features` for `interrupts` before starting. Pass
`metadata.langgraph_thread_id`, then resume `langgraph_interrupt` with a matching
`tool` message. The thread ID restores checkpoint state only; include the same
non-default `langgraph_runtime_settings` string on the resume request when the resumed run
needs those settings. See
[OpenAI compatibility](../explanation/openai-compatibility.md#tool-calls-and-interrupts).

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
