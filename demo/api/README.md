# LGOS demo API

Standalone FastAPI application exposing example LangGraph graphs through
`langgraph-openai-serve`.

```bash
cp .env.example .env
uv run --locked --env-file .env lgos-demo-api-setup
uv run --locked --env-file .env lgos-demo-api
```

Configuration uses the `DEMO_API_` prefix. For example,
`DEMO_API_POSTGRES_URI` selects the checkpoint database.

The `lgos-rag` graph reads a compact Markdown corpus packaged under
`src/lgos_demo_api/corpus`, so source installs, wheels, and images need no
external documentation checkout.
