# Add Your Graphs

`lgos serve` runs the catalog returned by `create_registry` in
`src/{{ cookiecutter.project_slug }}/registry.py`. The server calls it with
`ServerResources`: the checkpointer, store, and run coordinator it owns. The
Hatchet worker calls the same factory, so both processes serve one catalog.

1. Add a graph factory under `src/{{ cookiecutter.project_slug }}/graphs/`.
2. Register its `GraphConfig` in `create_registry` with a unique model ID and description.
3. Add focused tests using a deterministic model or your domain's own fixtures.
4. Call the new model through `examples/responses.py` or any OpenAI-compatible client.

Graphs with a `messages` input/output use LGOS's defaults. For other schemas,
provide `request_to_input` and `output_to_message` adapters on `GraphConfig`.
Keep your business behavior in the graph; LGOS owns request decoding and the
OpenAI response format. When a graph owns clients that must be closed, make
`create_registry` an async context manager that yields the registry.

## Runtime settings

The simple graph's `SimpleContext` allowlists `use_history` and `audience`.
Pass runtime settings through ordinary Responses metadata:

```python
response = client.responses.create(
    model="simple-graph",
    input="Explain this for a beginner.",
    metadata={"lgos_settings": '{"audience":"beginner","use_history":false}'},
    store=False,
)
```

Use the same `ClientSettings` model as the graph's `context_schema` and its
`GraphConfig.client_settings`. Settings are input, not trusted identity.

## Interrupts

`approval` pauses with a LangGraph interrupt and returns a result when resumed.
It is compiled with `resources.checkpointer`; the registry uses
`resources.run_coordinator` so only one process resumes a paused run at a time.

```bash
uv run --locked --env-file .env examples/approval.py
```

!!! important "Resume through the OpenAI contract"

    Preserve the returned Response ID and every `lgos_interrupt` call ID.
    Send `previous_response_id` and matching `function_call_output` items.
    Answer every pending call together.

LangGraph restarts an interrupted node on resume. Place external side effects
after the approval node and make them idempotent.

## Background models

Both graphs declare `GraphFeature.BACKGROUND`, so clients can submit them with
`background=True` and poll:

```bash
uv run --locked --env-file .env examples/background.py
uv run --locked --env-file .env examples/approval.py --background
```

Add the feature to a new graph only when it is safe to run in the worker. See
[custom graphs](https://ilkersigirci.github.io/langgraph-openai-serve/latest/tutorials/custom-graphs/)
for LGOS adapters and feature declarations.
