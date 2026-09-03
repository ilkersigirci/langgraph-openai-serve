# Run The Files API

The independent `demo/files_api` project exposes an OpenAI-compatible Files API
over S3-compatible storage. It has its own package, lockfile, Dockerfile, image,
settings, and tests. It does not depend on `langgraph-openai-serve`, LangGraph,
or the demo graph API.

## Run The Service

Configure the `DEMO_API_FILES_*` values in `demo/.env`, then choose one mode:

=== "Local process"

    ```bash
    cd demo
    make run-files-local
    ```

=== "Published container"

    ```bash
    cd demo
    make run-files
    ```

The OpenAI base URL is `http://localhost:3006/v1`; health is available at
`http://localhost:3006/health`.

The service supports upload, list, retrieve, content, and delete operations for
`purpose="user_data"`. Chainlit may call it directly or send Files requests
through Bifrost's dedicated `lgos-files` provider. The `file-input` graph uses
the same service to resolve an incoming `file_id`.

## Use A Custom Repository

Implement the structural `FileRepository` protocol in
`lgos_files_api.contracts` and pass it to the public `create_files_app`
factory:

```python
from lgos_files_api import create_files_app
from my_repository import MyFileRepository

app = create_files_app(MyFileRepository())
```

The default command uses `create_s3_app`; a custom deployment supplies its own
ASGI module and storage lifecycle. Neither inheritance nor a dependency on
LGOS or LangGraph is required.

## Ownership Boundary

Only this service receives the `DEMO_API_FILES_AWS_*` credentials and bucket
configuration. LGOS API replicas receive only the Files base URL needed by a
graph that resolves IDs. Chainlit's native element-storage credentials remain
separate.

The demo is a deliberately small reference implementation. Before exposing it
to untrusted clients, add authentication, tenant isolation, upload limits,
retention, malware policy, and the availability controls required by the
deployment. See [Accept File Inputs](../how-to-guides/file-inputs.md) for the
client and graph flow and [Settings and Commands](reference.md#files-api-settings)
for configuration.
