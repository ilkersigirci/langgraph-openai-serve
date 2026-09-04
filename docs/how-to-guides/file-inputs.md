# Accept File Inputs

Use the native OpenAI Files API as one logical file boundary. Clients upload
opaque bytes with `purpose="user_data"`, receive a stable `file_id`, and include
that ID in a native Chat Completions file content part. LGOS preserves the part;
the graph decides how to resolve and interpret it.

Graphs that resolve these IDs declare `GraphFeature.FILE_INPUTS`. Clients can
then offer attachment controls only for graphs that accept them; ordinary
graphs continue to receive standard chat messages without pretending they can
interpret an external file namespace.

## Deploy One File Boundary

LGOS deliberately does not expose Files routes or hold file-storage
credentials. Deploy one external OpenAI-compatible Files API for the graph
services that share a file namespace. That service may have multiple stateless
replicas over the same durable repository.

At a gateway, route `/v1/files` to this service independently of the selected
chat model. File storage credentials and lifecycle policy then remain outside
the graph-serving processes. The same layout stays valid when one LGOS instance
grows into many; only the graph service scales independently.

Use a gateway-native Files implementation when it supplies the lifecycle,
authorization, storage, and API semantics you need. Otherwise deploy the
standalone S3-backed [demo Files API](../demo/files-api.md) project. The graph API
only needs the Files base URL when a graph resolves IDs itself.

## Adapt Storage In The Files Service

The demo keeps its repository protocol inside the standalone project. Replace
its S3 adapter there when an application needs another store; no LGOS package
or graph API change is required. Keep a repository synchronous when its native
client is synchronous because FastAPI runs the Files route functions in a
worker thread.

The demo accepts `purpose="user_data"` and rejects the unsupported
`expires_after` parameter. Before serving untrusted traffic, add the required
authentication, tenant isolation, upload limits, retention, and malware policy.

## Upload And Reference A File

The uploader and graph APIs may have different base URLs when no gateway fronts
them:

```python
from openai import OpenAI

files = OpenAI(base_url="https://files.example.com/v1", api_key="...")
graphs = OpenAI(base_url="https://graphs.example.com/v1", api_key="...")

with open("report.pdf", "rb") as file:
    uploaded = files.files.create(file=file, purpose="user_data")

response = graphs.chat.completions.create(
    model="document-graph",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this file."},
                {"type": "file", "file": {"file_id": uploaded.id}},
            ],
        }
    ],
)
```

LGOS preserves the file part in the LangChain `HumanMessage.content` list. The
graph can use an application-owned resolver to call
`GET /v1/files/{file_id}/content`, create a short-lived object-storage URL, or
upload the bytes to a downstream provider. LGOS does not parse the file or turn
the ID into a URL.

Generate bearer URLs only when needed. Do not persist presigned URLs in messages
or use one as a `file_id`; URLs expire and may leak through logs and traces.

The demo Files API accepts opaque bytes without checking extension or MIME
type. Only `purpose="user_data"` is accepted because the demo does not
implement OpenAI batch, fine-tuning, Assistants, or eval workflows.

The demo rejects the native `expires_after` upload parameter rather than
silently ignoring it. Apply retention in the storage service until an
implementation supports that parameter.

The demo implements upload, list, retrieve, content, and delete operations from the
[OpenAI Files API](https://developers.openai.com/api/reference/resources/files).

The Files API does not turn storage into graph state. The client sends the
`file_id` in each request that needs it.
