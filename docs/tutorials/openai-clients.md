# OpenAI Clients

Configure an ordinary OpenAI SDK with the LGOS base URL, usually
`http://localhost:8000/v1`, and use a registered graph name as `model`. The
`api_key` is sent as a bearer token; the application from
[Get Started](../getting-started.md) does not verify it.

Use Responses for new clients and every maintained demo UI. Chat Completions
remains a direct compatibility surface for clients that cannot use Responses.
An optional proxy must expose a native `/v1/responses` route and preserve the
same typed items and events; raw pass-through is not part of the client design.

The basic examples call the provider-free `echo` graph from the getting-started
application. Names such as `research-graph` and `interruptible` describe
capabilities the registered graph must enable. The
[demo graph catalog](../demo/graphs/index.md) provides runnable examples.

## Install A Client

=== "Python"

    ```bash
    pip install openai
    ```

=== "JavaScript"

    ```bash
    npm install openai
    ```

## Create A Response

LGOS is stateless at the Responses layer. Send `store=False` explicitly so the
request remains portable to other OpenAI-compatible endpoints; LGOS also
defaults an omitted value to false.

!!! warning "Do not expose real API keys in a browser"

    The JavaScript example enables `dangerouslyAllowBrowser` only because the
    local application uses a dummy key. Keep production credentials in
    server-side code.

=== "Python"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

    response = client.responses.create(
        model="echo",
        input="Hello from Python",
        store=False,
    )

    print(response.output_text)
    ```

=== "JavaScript"

    ```javascript
    import OpenAI from "openai";

    const openai = new OpenAI({
      baseURL: "http://localhost:8000/v1",
      apiKey: "DUMMY",
      dangerouslyAllowBrowser: true,
    });

    const response = await openai.responses.create({
      model: "echo",
      input: "Hello from JavaScript",
      store: false,
    });

    console.log(response.output_text);
    ```

## Stream Final Text And Commentary

Responses streams typed lifecycle events rather than Chat chunks. When a graph
declares `GraphFeature.CLIENT_EVENTS`, every visible `status_event()` becomes a
completed assistant message with `phase="commentary"`; no request metadata
opt-in is required. The durable answer uses `phase="final_answer"`.

Track the phase from `response.output_item.added` before handling text deltas:

=== "Python"

    ```python
    phases = {}

    with client.responses.stream(
        model="research-graph",
        input="Research this topic.",
        store=False,
    ) as stream:
        for event in stream:
            if (
                event.type == "response.output_item.added"
                and event.item.type == "message"
            ):
                phases[event.output_index] = event.item.phase
            elif event.type == "response.output_text.delta":
                if phases.get(event.output_index) == "final_answer":
                    print(event.delta, end="", flush=True)
            elif event.type == "response.output_text.done":
                if phases.get(event.output_index) == "commentary":
                    show_status(event.text)

        response = stream.get_final_response()
    ```

=== "JavaScript"

    ```javascript
    const phases = new Map();
    const stream = await openai.responses.create({
      model: "research-graph",
      input: "Research this topic.",
      store: false,
      stream: true,
    });

    for await (const event of stream) {
      if (
        event.type === "response.output_item.added" &&
        event.item.type === "message"
      ) {
        phases.set(event.output_index, event.item.phase);
      } else if (
        event.type === "response.output_text.delta" &&
        phases.get(event.output_index) === "final_answer"
      ) {
        process.stdout.write(event.delta);
      } else if (
        event.type === "response.output_text.done" &&
        phases.get(event.output_index) === "commentary"
      ) {
        showStatus(event.text);
      }
    }
    ```

Commentary is streaming-only and transient. Non-streaming execution does not
collect old status updates. The OpenAI Python SDK's `Response.output_text`
convenience property concatenates all output-text parts, including commentary,
so a UI processing a completed stream must select only message items whose
phase is `final_answer`:

```python
final_text = "".join(
    part.text
    for item in response.output
    if item.type == "message" and item.phase == "final_answer"
    for part in item.content
    if part.type == "output_text"
)
```

The maintained Chainlit and Open WebUI adapters apply this filter and render
commentary through their native status interfaces. A generic client that does
not understand `phase` may display commentary as answer text; keep status text
useful but do not treat that client as a full advanced-UI integration.

## Manage Conversation State

LGOS does not persist Response objects or Conversations. It rejects
`store=True`, `previous_response_id`, `conversation`, and background mode.
Keep an input ledger and resend the items needed by each turn:

```python
input_items = [{"role": "user", "content": "Introduce LangGraph briefly."}]
first = client.responses.create(
    model="echo",
    input=input_items,
    store=False,
)

input_items.extend(item.model_dump(mode="json") for item in first.output)
input_items.append({"role": "user", "content": "Now make it one sentence."})
second = client.responses.create(
    model="echo",
    input=input_items,
    store=False,
)
```

Replay complete SDK output items instead of rebuilding assistant text. This
preserves item IDs, function-call IDs, and assistant `phase`. Keep every earlier
user, system, or developer item that the next turn needs. This is application
conversation state; LangGraph checkpoints remain a separate temporary store for
paused interrupts.

## Continue Function Calls

LGOS accepts the flat Responses function-tool shape. When a graph returns a
`function_call`, execute only a function your client owns, then replay the
complete output and append the matching string-valued result:

```python
import json

tools = [
    {
        "type": "function",
        "name": "lookup_order",
        "description": "Look up an order visible to the signed-in user.",
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    }
]

input_items = [{"role": "user", "content": "Where is order A123?"}]
response = client.responses.create(
    model="my-graph",
    input=input_items,
    tools=tools,
    store=False,
)

input_items.extend(item.model_dump(mode="json") for item in response.output)
for item in response.output:
    if item.type != "function_call":
        continue
    result = lookup_order(**json.loads(item.arguments))
    input_items.append(
        {
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(result),
        }
    )

completed = client.responses.create(
    model="my-graph",
    input=input_items,
    tools=tools,
    store=False,
)
```

The same full-item rule drives the demo's Files-plus-`display_file` chart
contract. The trusted UI downloads the `file_id`, renders or persists it
natively, and returns a small acknowledgment. See
[Accept And Display Files](../how-to-guides/file-inputs.md#display-a-graph-generated-file).

## Resume An Interrupt

An interrupt-enabled graph returns one or more `function_call` items named
`langgraph_interrupt`. Preserve every returned call and answer the whole batch.
No metadata is required for an initial request; use a new UUID in
`metadata.langgraph_run_id` when retrying a lost initial response must address
the same pending operation.

```python
import json
from uuid import uuid4

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="DUMMY",
    max_retries=0,
)
metadata = {"langgraph_run_id": str(uuid4())}

input_items = [
    {"role": "user", "content": "Perform the protected action."}
]
paused = client.responses.create(
    model="interruptible",
    input=input_items,
    metadata=metadata,
    store=False,
)
calls = [item for item in paused.output if item.type == "function_call"]
if not calls:
    raise RuntimeError("The graph completed without interrupting")

input_items.extend(item.model_dump(mode="json") for item in paused.output)
input_items.extend(
    {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": json.dumps({"resume": collect_answer(call)}),
    }
    for call in calls
)

completed = client.responses.create(
    model="interruptible",
    input=input_items,
    metadata=metadata,
    store=False,
)
```

Do not resume a subset, synthesize a new call, or replay only the visible
question. Persist the canonical returned items before asking the user so a
reconnect can reproduce the request. Runtime settings remain per-request and
must be resent. See
[Tool Calls And Interrupts](../explanation/openai-compatibility.md#tool-calls-and-interrupts)
for stale-state conflicts and recovery boundaries.

## Model Discovery And Runtime Settings

Use `client.models.list()` for registered graph IDs. Direct LGOS model objects
also expose the namespaced `langgraph_openai_serve` extension. Retrieve a
selected model to discover its settings descriptor:

```python
model = client.models.retrieve("my-settings-graph")
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

`metadata.langgraph_runtime_settings` is JSON text, not a nested metadata
object. Send only values that differ from the advertised defaults, keep the
encoded value at 512 characters or fewer, and resend it on every request that
needs it. A normalizing proxy may omit this optional extension; plain Responses
still work, but the client must not infer settings or graph features it cannot
discover. See [Runtime Settings](../how-to-guides/langgraph-runtime-settings.md).

## Direct Chat Compatibility

Chat Completions remains available for direct clients that need the familiar
message/choice shape:

```python
completion = client.chat.completions.create(
    model="echo",
    messages=[{"role": "user", "content": "Hello through Chat"}],
)

print(completion.choices[0].message.content)
```

This route shares the same graph runner but has its own protocol adapter. Direct
Chat clients can opt into the namespaced client-event extension (`status`,
`progress`, and `artifact`) with
`metadata.langgraph_stream_events="v1"`; maintained demo UIs do not use that
transport. A schema-normalizing proxy may discard extension-only Chat chunks,
so use standard Responses commentary, function calls, and Files for portable
advanced UI behavior.

## Diagnostics

??? example "Direct HTTP diagnostic"

    Use direct HTTP only to inspect behavior while debugging:

    ```bash
    curl -X POST http://localhost:8000/v1/responses \
      -H "Content-Type: application/json" \
      -d '{
        "model": "echo",
        "input": "Hello from HTTP",
        "store": false
      }'
    ```

Set timeouts for long-running graphs and add bearer-token authentication before
exposing the API outside a trusted development environment. The exact accepted
request fields and explicit exclusions are in the
[compatibility contract](../explanation/openai-compatibility.md#supported-responses-subset).
