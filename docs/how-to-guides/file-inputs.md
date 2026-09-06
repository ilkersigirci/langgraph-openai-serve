# Accept And Display Files

Use the native OpenAI Files API as one logical file boundary. Clients upload
opaque bytes with `purpose="user_data"`, receive a stable `file_id`, and include
that ID in a Responses `input_file` part. LGOS normalizes the part for the
graph; the graph decides how to resolve and interpret it.

Graphs that resolve these IDs declare `GraphFeature.FILE_INPUTS`. Clients can
then offer attachment controls only for graphs that accept them. The feature is
discovery metadata, not permission for LGOS to download arbitrary URLs.

## Deploy One File Boundary

LGOS deliberately does not expose Files routes or hold file-storage
credentials. Deploy one external OpenAI-compatible Files API for the graph
services that share a file namespace. That service may have multiple stateless
replicas over one durable repository.

At a gateway, route standard `/v1/files` operations to this service independently
of the selected graph model. Use a gateway-native Files implementation when it
supplies the lifecycle, authorization, storage, and API semantics you need.
Otherwise deploy the standalone S3-backed [demo Files API](../demo/files-api.md).

## Upload And Reference A File

The Files and graph APIs may have different base URLs:

```python
from openai import OpenAI

files = OpenAI(base_url="https://files.example.com/v1", api_key="...")
graphs = OpenAI(base_url="https://graphs.example.com/v1", api_key="...")

with open("report.pdf", "rb") as file:
    uploaded = files.files.create(file=file, purpose="user_data")

response = graphs.responses.create(
    model="document-graph",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize this file."},
                {"type": "input_file", "file_id": uploaded.id},
            ],
        }
    ],
    store=False,
)

print(response.output_text)
```

LGOS currently accepts only the `input_file.file_id` form. The standard
`file_url` and inline `file_data` forms are rejected: URL fetching needs an
explicit SSRF, redirect, DNS, timeout, and size policy, while inline data needs
an application-defined size limit. `input_image`, audio, and other Responses
content types are also outside the current subset.

The graph receives this existing LangChain content shape:

```python
{
    "type": "file",
    "file": {"file_id": uploaded.id},
}
```

It can call `GET /v1/files/{file_id}/content`, create a short-lived
object-storage URL, or upload the bytes to a downstream model provider. LGOS
does not parse the file or turn its ID into a URL. The direct Chat compatibility
route also preserves native Chat file parts, but maintained demo clients use
Responses `input_file` items.

## Display A Graph-Generated File

Responses has no generic artifact field. A graph that needs the client to show
a generated file returns a client-owned function call and keeps the bytes in
the Files service:

1. The client offers a strict `display_file` function tool.
2. The graph renders the file, uploads it with `purpose="user_data"`, and
   returns a deterministic `function_call` containing `file_id`, filename,
   media type, title, and alt text.
3. The client's trusted backend downloads the bytes with
   `client.files.content(file_id)` and persists or renders them through its
   native UI storage.
4. The client replays the complete returned Response items and appends a
   matching string-valued `function_call_output`, such as
   `{"displayed":true}`.
5. The graph returns the final answer.

Do not put a protected content URL in browser Markdown: a browser image request
cannot attach the Files API bearer credential. Do not use `file_citation`,
`file_path`, code-interpreter output, or a custom SSE event to mean “display
this graph-owned file.” The [persistent plot demo](../demo/graphs/persistent-plot-agent.md)
shows the complete Chainlit and Open WebUI flow. It uploads native Plotly JSON
with media type `application/vnd.plotly.v1+json`; the UIs render it as a Chainlit
`Plotly` element or an Open WebUI HTML embed. `display_file` is an
application-defined function using standard OpenAI function calling, not a
built-in OpenAI rendering tool.

LGOS currently accepts string-valued `function_call_output.output`. The newer
list form containing text, image, or file parts is rejected until an
end-to-end graph and client need it.

## Storage And Security

The Files API is storage, not conversation or graph state. Send the `file_id`
again in every request that needs it. Keep canonical application data in the
application's Store and treat rendered files as derived output.

The demo accepts opaque bytes and only `purpose="user_data"`; it rejects
`expires_after`. Before serving untrusted traffic, add authentication, tenant
isolation, upload limits, retention, malware policy, and availability controls.
Generate bearer URLs only when needed, and never persist a presigned URL as a
`file_id`.

The demo implements upload, list, retrieve, content, and delete operations from
the [OpenAI Files API](https://developers.openai.com/api/reference/resources/files).
