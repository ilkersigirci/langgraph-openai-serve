# Calling LGOS with OpenAI clients

Point an OpenAI SDK at the LGOS `/v1` base URL and use a registered graph name
as the model. No LGOS-specific request wrapper is needed.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="DUMMY")
response = client.responses.create(
    model="simple-graph",
    input="Hello",
    store=False,
)
print(response.output_text)
```

Set `stream=True` and iterate the returned typed events for token streaming. Model
listing uses `client.models.list()`. Retrieve one selected model with
`client.models.retrieve(model_name)` when a UI needs optional detailed LGOS
feature or runtime-settings metadata. The namespaced extension is preserved by
the OpenAI Python SDK in `model_extra`.

The
[OpenAI client tutorial](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/tutorials/openai-clients.md)
covers Python and JavaScript, stateless replay, streaming commentary, function
calls, errors, model discovery, and the direct Chat compatibility surface.
