# Configuration

`src/{{ cookiecutter.project_slug }}/settings.py` holds the graphs' own settings
under the `APP_` prefix. Pydantic loads `.env` for those fields; process
environment variables take precedence.

| Variable | Purpose |
| --- | --- |
| `APP_OPENAI_BASE_URL` | Upstream OpenAI-compatible provider |
| `APP_OPENAI_API_KEY`, `APP_OPENAI_MODEL` | Upstream credentials and model |
| `CLIENT_BASE_URL`, `CLIENT_API_KEY` | Application endpoint and key used by examples |

The server reads `UVICORN_*`, `LGOS_*`, `HATCHET_CLIENT_*`, `LANGFUSE_*`, and
`OTEL_*`. `lgos serve` is Uvicorn's command, so `UVICORN_HOST`, `UVICORN_PORT`,
and every other Uvicorn option apply. The most common LGOS settings are
`LGOS_POSTGRES_URI` for durable interrupts, `LGOS_BACKGROUND` for background
Responses, and `LGOS_ENABLE_LANGFUSE` for tracing. See the
[LGOS server settings](https://ilkersigirci.github.io/langgraph-openai-serve/latest/how-to-guides/server/#settings)
for the full list. `.env.example` lists every variable this project uses.

!!! important "Load the process environment"

    LGOS, Langfuse, Hatchet, and OpenTelemetry read process variables. Just
    loads `.env` before each command; direct runs should use
    `uv run --locked --env-file .env ...`. Docker Compose passes the same file
    to the application containers.

`APP_OPENAI_API_KEY` is an upstream credential, not authentication for this
API. Configure application access at your gateway. See
[deployment](deployment.md).
