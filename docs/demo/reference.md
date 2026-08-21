# Demo Settings And Commands

This reference describes the independently locked projects and Compose stack
under `demo/`. These commands and `DEMO_*` settings are not part of the
`langgraph-openai-serve` package API.

## Projects

| Path | Purpose | Imports LGOS? |
| --- | --- | --- |
| `demo/api` | Example FastAPI and LangGraph application | Yes, from PyPI by default |
| `demo/ui/chainlit_ui` | Persistent OpenAI-protocol client | No |
| `demo/ui/openwebui` | Open WebUI Function sources and sync command | No |
| `demo/docker` | Compose-only Bifrost configuration and service data directories | No |

Each Python project has its own `pyproject.toml`, virtual environment, and
`uv.lock`; `demo/` deliberately is not a uv workspace.

## Common Commands

Run these from `demo/` after copying `.env.example` to `.env`:

| Command | Purpose |
| --- | --- |
| `make run-api` / `make run-api-a` | Run the published `lgos-a` container on port 3004 |
| `make run-api-b` | Run the published `lgos-b` container on port 3005 |
| `make run-chainlit` | Run the published Chainlit container and its dependencies on port 3002 |
| `make run-api-local` / `make run-api-a-local` | Set up checkpoints and run the editable local `lgos-a` process |
| `make run-api-b-local` | Set up checkpoints and run the editable local `lgos-b` process |
| `make run-chainlit-local` | Apply Chainlit migrations and run the local UI process |
| `make sync-openwebui` | Sync the Open WebUI Functions and generated LGOS Workspace Models |
| `make compose` | Run the stack with published project-owned images |
| `make compose-dev` | Build local images; run the API and LGOS packages editable |
| `make compose-otel` | Run published images with the OTEL overlay |
| `make compose-otel-dev` | Build the checkout and run it with the OTEL overlay |
| `make sync` | Synchronize all three projects from their lockfiles |
| `make test` | Test all three projects from their lockfiles |
| `make test-postgres` | Run the end-to-end interrupt test against PostgreSQL on port 3001 |
| `make lint` | Check all three projects with Ruff |
| `make check` | Run tests, lint, formatting checks, and Compose validation |

## Stack Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEMO_IMAGE_TAG` | `latest` | Tag selected for both project-owned demo images |
| `PUID` | `1000` | Host user ID used by Compose services |
| `PGID` | `1000` | Host group ID used by Compose services |

## OpenTelemetry Settings

These settings apply when using `make compose-otel` or
`make compose-otel-dev`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OTEL_COLLECTOR_GATEWAY_ENDPOINT` | required | OTLP/HTTP base URL for the host or platform gateway |
| `OTEL_COLLECTOR_GATEWAY_INSECURE` | `false` | Set the Collector exporter `tls.insecure` flag |
| `OTEL_SERVICE_NAMESPACE` | `lgos` | Namespace default for application and Collector signals |
| `OTEL_DEPLOYMENT_ENVIRONMENT` | `production` | Environment default for application and Collector signals |
| `OTEL_HOST_NAME` | required | Stable host identity added by the local Collector |
| `OTEL_TRACES_SAMPLE_RATE` | `1.0` | Parent-based sampling ratio for Python SDK services |

The OTEL overlay requires both `OTEL_COLLECTOR_GATEWAY_ENDPOINT` and
`OTEL_HOST_NAME`; set them per machine in `.env`. Set
`OTEL_COLLECTOR_GATEWAY_INSECURE=true` only when the gateway intentionally
accepts a non-TLS OTLP/HTTP connection; leave it `false` for the normal HTTPS
gateway endpoint.

## Demo API Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEMO_API_PORT` | `8000` | HTTP port used by the `lgos-demo-api` command |
| `DEMO_API_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Upstream OpenAI-compatible base URL |
| `DEMO_API_OPENAI_API_KEY` | `DUMMY` | Upstream key for provider-backed graphs |
| `DEMO_API_OPENAI_MODEL` | `gpt-5.4-mini` | Upstream generation model |
| `DEMO_API_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model used by `lgos-rag` |
| `DEMO_API_POSTGRES_URI` | `postgresql://lgos:lgos@localhost:3001/lgos` | Checkpoint database |

The API also reads the package-owned `LGOS_OPENAI_API_PREFIX`,
`LGOS_OPENAI_API_DOCS_ENABLED`, and `LGOS_ENABLE_LANGFUSE` settings documented
in the package [Reference](../reference.md#settings).
The demo settings model deliberately supports its own `.env` file for local
development; the installed LGOS package itself reads only process environment
values or explicit constructor arguments.

## Open WebUI Sync Settings

The typed
[settings model](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/demo/ui/openwebui/src/lgos_openwebui/settings.py)
is the source of truth for these `DEMO_OPENWEBUI_` variables.

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEMO_OPENWEBUI_URL` | `http://localhost:3003` | Open WebUI API used by the sync command |
| `DEMO_OPENWEBUI_ADMIN_EMAIL` | `lgos@example.com` | Open WebUI sync account |
| `DEMO_OPENWEBUI_ADMIN_PASSWORD` | `lgos` | Open WebUI sync password |
| `DEMO_OPENWEBUI_OPENAI_CATALOG_BASE_URL` | `http://localhost:3000/v1` | Bifrost model catalog used for discovery |
| `DEMO_OPENWEBUI_OPENAI_BASE_URL` | `http://localhost:3000/openai_passthrough/v1` | Bifrost pass-through used for model metadata |
| `DEMO_OPENWEBUI_API_KEY` | `DUMMY` | Gateway key used by both OpenAI clients |

See [Chainlit settings](chainlit.md#settings-reference),
[Open WebUI setup](open-webui.md#setup), and the [example graph catalog](graphs.md)
for component-specific details.
