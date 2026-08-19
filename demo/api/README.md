# LGOS demo API

Standalone FastAPI application exposing example LangGraph graphs through
`langgraph-openai-serve`.

```bash
cp .env.example .env
uv run --locked --env-file .env lgos-demo-api-setup
uv run --locked --env-file .env lgos-demo-api
```

The API writes JSON logs to stdout. LGOS log records include correlation fields
such as `request_id`, `model`, `stream`, and `run_id` when available. The
optional `docker/compose/otel.yml` deployment overlay also exports the same
standard-library records as native OpenTelemetry logs over OTLP.

Configuration uses the `DEMO_API_` prefix. For example,
`DEMO_API_POSTGRES_URI` selects the checkpoint database.
Set the package-owned `LGOS_ENABLE_LANGFUSE=true` and provide the `LANGFUSE_*`
credentials to enable LGOS's lazy tracing integration for every demo graph.

The `lgos-rag` graph reads a compact Markdown corpus packaged under
`src/lgos_demo_api/corpus`, so source installs, wheels, and images need no
external documentation checkout.
