# LGOS Files API

This independent demo project exposes the OpenAI-compatible Files API over an
S3-compatible object store. It does not depend on `langgraph-openai-serve`,
LangGraph, or the demo graph API.

From `demo/`, configure the `DEMO_API_FILES_*` values in `.env`, then run:

```bash
make run-files-local
```

The OpenAI base URL is `http://localhost:3006/v1`. The service supports upload,
list, retrieve, content, and delete for files with `purpose="user_data"`.

## Custom Repositories

Implement the structural `FileRepository` protocol in
`lgos_files_api.contracts`, then pass the implementation to the public app
factory:

```python
from lgos_files_api import create_files_app
from my_repository import MyFileRepository

app = create_files_app(MyFileRepository())
```

No inheritance or LGOS dependency is required. The repository owns its storage
client and logic; the factory owns only the OpenAI-compatible HTTP routes.

This is a small-scale reference implementation. Add authentication, tenant
isolation, upload limits, retention, and malware policy before exposing it to
untrusted clients.
