# OpenAI Clients

Configure clients with the server base URL, usually `http://localhost:8000/v1`.
The `api_key` value is sent as `Authorization: Bearer <key>`; the default demo
does not verify it.

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
    demo uses a dummy key. Keep production credentials in server-side code.

=== "Python"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

    response = client.chat.completions.create(
        model="custom-input-output-context",
        messages=[{"role": "user", "content": "Show me the custom adapter."}],
        user="demo-user",
    )

    print(response.choices[0].message.content)
    ```

=== "Python Streaming"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")

    stream = client.chat.completions.create(
        model="simple-graph",
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
      model: "custom-input-output-context",
      messages: [{ role: "user", content: "Show me the custom adapter." }],
      user: "demo-user",
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
      model: "simple-graph",
      messages: [{ role: "user", content: "Write a short poem about graphs." }],
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      process.stdout.write(content);
    }
    ```

## Model Discovery And Configuration

List standard model summaries, then retrieve detailed LGOS metadata only for
the selected graph:

=== "Python"

    ```python
    models = client.models.list()
    selected = models.data[0].id
    model = client.models.retrieve(selected)

    extension = (model.model_extra or {}).get("langgraph_openai_serve", {})
    if extension.get("schema_version") == 1:
        print(extension.get("features", []))
        print(extension.get("client_config"))
    ```

=== "JavaScript"

    ```javascript
    const models = await openai.models.list();
    const model = await openai.models.retrieve(models.data[0].id);

    const extension = model.langgraph_openai_serve || {};
    if (extension.schema_version === 1) {
      console.log(extension.features || []);
      console.log(extension.client_config);
    }
    ```

The simple demo carries changed graph settings in one JSON metadata envelope.
System instructions remain ordinary OpenAI messages, independent of discovered
client settings:

```python
import json

response = client.chat.completions.create(
    model="simple-graph",
    messages=[
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Explain LangGraph."},
    ],
    metadata={
        "langgraph_config": json.dumps({"use_history": True}),
    },
)
```

Use the fixed `metadata.langgraph_config` envelope and omit values equal to the
advertised defaults. Native Chat Completions fields such as `temperature` keep
their normal API meaning and are not reused as graph runtime context.

The server validates advertised settings against the retrieved JSON Schema.
Do not send arbitrary nested or large configuration values through metadata.

## Interrupt Resume

Interrupt-enabled graphs use OpenAI tool calls. Retrieve the selected model and
check `langgraph_openai_serve.features` for `interrupts` before starting. Pass
`metadata.langgraph_thread_id`, then resume `langgraph_interrupt` with a matching
`tool` message. See
[OpenAI compatibility](../explanation/openai-compatibility.md#tool-calls-and-interrupts).

## Diagnostics

??? example "Direct HTTP diagnostic"

    Use direct HTTP only to inspect behavior while debugging:

    ```bash
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "custom-input-output-context",
        "messages": [{"role": "user", "content": "Show me the custom adapter."}],
        "user": "demo-user"
      }'
    ```

## Notes

- Use the registered graph name as `model`.
- Set timeouts for long-running graphs.
- Use streaming only for graphs configured to emit streamed chunks.
- Add bearer-token authentication before exposing the API outside trusted
  development environments.
