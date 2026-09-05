# Demo Settings And Commands

This reference describes the independently locked projects and Compose stack
under `demo/`. These commands and `DEMO_*` settings are not part of the
`langgraph-openai-serve` package API.

## Projects

| Path | Purpose | Imports LGOS? |
| --- | --- | --- |
| `demo/api` | Example FastAPI and LangGraph application | Yes, from PyPI by default |
| `demo/files_api` | OpenAI-compatible Files service and S3 adapter | No |
| `demo/ui/chainlit_ui` | Persistent OpenAI-protocol client | No |
| `demo/ui/openwebui` | Open WebUI Function sources and sync command | No |
| `demo/docker` | Compose gateway configuration and service data directories | No |

Each Python project has its own `pyproject.toml`, virtual environment, and
`uv.lock`; `demo/` deliberately is not a uv workspace.

## Common Commands

Run these from `demo/` after copying `.env.example` to `.env`:

| Command | Purpose |
| --- | --- |
| `make run-api` / `make run-api-a` | Run the published `lgos-a` container on port 3004 |
| `make run-api-b` | Run the published `lgos-b` container on port 3005 |
| `make run-files` | Run the published Files API container on port 3006 |
| `make run-bifrost` | Run Bifrost and its graph and Files API dependencies on port 3000 |
| `make run-litellm` | Run the LiteLLM UI edge and compatibility gateway with its API and Files dependencies on port 3007 |
| `make run-chainlit` | Run the published Chainlit container and its dependencies on port 3002 |
| `make run-api-local` / `make run-api-a-local` | Set up checkpoints and run the editable local `lgos-a` process |
| `make run-api-b-local` | Set up checkpoints and run the editable local `lgos-b` process |
| `make run-files-local` | Run the independently locked local Files API process |
| `make run-chainlit-local` | Apply Chainlit migrations and run the local UI process |
| `make sync-openwebui` | Sync the Open WebUI Functions and generated LGOS Workspace Models |
| `make compose` | Run the stack with published project-owned images |
| `make compose-dev` | Build the local API, Files API, and Chainlit images; overlay LGOS only into the graph API image |
| `make compose-otel` | Run published images with the OTEL overlay |
| `make compose-otel-dev` | Build the checkout and run it with the OTEL overlay |
| `make sync` | Synchronize all four projects from their lockfiles |
| `make test` | Test all four projects from their lockfiles |
| `make test-postgres` | Run the interrupt and Store persistence tests against PostgreSQL on port 3001 |
| `make lint` | Check all four projects with Ruff |
| `make check` | Run tests, lint, formatting checks, and Compose validation |

From the repository root, `make test-litellm` and `make test-bifrost` run the
focused OpenAI SDK checks. `OPENAI_GATEWAY_TYPE=litellm|bifrost` selects the
gateway used by both maintained UIs. Responses and Files use its normal
managed/native routes. A separate catalog-detail client retains LGOS model
extensions through authenticated pass-through; UI inference never does.

## Stack Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEMO_IMAGE_TAG` | `latest` | Tag selected for all project-owned demo images |
| `PUID` | `1000` | Host user ID used by Compose services |
| `PGID` | `1000` | Host group ID used by Compose services |
| `OPENAI_GATEWAY_TYPE` | `litellm` | Gateway used by both demo UIs: `litellm` or `bifrost` |
| `DEMO_LITELLM_MASTER_KEY` | demo-only value | LiteLLM bearer key shared by the two UI clients and default local Admin UI password for username `admin`; replace it outside local demos |
| `DEMO_OPENWEBUI_SECRET_KEY` | demo-only value | Open WebUI application secret; replace it outside local demos |

## OpenTelemetry Settings

These settings apply when using `make compose-otel` or
`make compose-otel-dev`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OTEL_COLLECTOR_GATEWAY_ENDPOINT` | required | OTLP/HTTP base URL for the host or platform gateway |
| `OTEL_SERVICE_NAMESPACE` | `lgos` | Namespace default for application and Collector signals |
| `OTEL_DEPLOYMENT_ENVIRONMENT` | `production` | Environment default for application and Collector signals |
| `OTEL_HOST_NAME` | required | Stable host identity added by the local Collector |

The OTEL overlay uses the OpenTelemetry `always_on` sampler, so application
traces are exported without SDK sampling. The selected remote backend owns
retention.

The OTEL overlay requires both `OTEL_COLLECTOR_GATEWAY_ENDPOINT` and
`OTEL_HOST_NAME`; set them per machine in `.env`. The endpoint URL scheme
controls transport security: use `https://` for TLS and `http://` only when the
gateway intentionally accepts cleartext OTLP/HTTP.

## Demo API Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEMO_API_PORT` | `8000` | HTTP port used by `lgos-demo-api` |
| `DEMO_API_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Upstream OpenAI-compatible base URL |
| `DEMO_API_OPENAI_API_KEY` | `DUMMY` | Upstream key for provider-backed graphs |
| `DEMO_API_OPENAI_MODEL` | `gpt-5.4-mini` | Upstream generation model |
| `DEMO_API_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model used by `lgos-rag` |
| `DEMO_API_POSTGRES_URI` | `postgresql://lgos:lgos@localhost:3001/lgos` | Database for LangGraph checkpoints, Store data, and interrupt coordination |
| `DEMO_API_FILES_BASE_URL` | `http://localhost:3006/v1` | Central Files API read by the `file-input` graph. |

The API also reads the package-owned `LGOS_OPENAI_API_PREFIX`,
`LGOS_OPENAI_API_DOCS_ENABLED`, and `LGOS_ENABLE_LANGFUSE` settings documented
in the package [Reference](../reference.md#settings). Its settings model supports
a local `.env` file; the installed LGOS package itself reads only process
environment values or explicit constructor arguments.

## Files API Settings

These settings belong only to the independent `demo/files_api` project.

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEMO_API_FILES_PORT` | `8000` | HTTP port used by `lgos-files-api`. |
| `DEMO_API_FILES_BUCKET` | unset | Required S3-compatible bucket. |
| `DEMO_API_FILES_S3_ENDPOINT` | unset | Optional S3-compatible endpoint; required by the Compose demo. |
| `DEMO_API_FILES_AWS_ACCESS_KEY_ID` | unset | Required S3 access key passed explicitly to boto3. |
| `DEMO_API_FILES_AWS_SECRET_ACCESS_KEY` | unset | Required S3 secret key passed explicitly to boto3. |
| `DEMO_API_FILES_AWS_DEFAULT_REGION` | unset | Required S3 signing region passed explicitly to boto3. |

## Open WebUI Sync Settings

The typed `demo/ui/openwebui/src/lgos_openwebui/settings.py` model is the source
of truth for these `DEMO_OPENWEBUI_` variables.

| Setting | Default | Purpose |
| --- | --- | --- |
| `DEMO_OPENWEBUI_URL` | `http://localhost:3003` | Open WebUI API used by the sync command |
| `DEMO_OPENWEBUI_ADMIN_EMAIL` | `lgos@example.com` | Open WebUI sync account |
| `DEMO_OPENWEBUI_ADMIN_PASSWORD` | `lgos` | Open WebUI sync password |
| `OPENAI_GATEWAY_TYPE` | `litellm` | Exact global selector shared with Chainlit: `litellm` or `bifrost` |
| `DEMO_OPENWEBUI_OPENAI_GATEWAY_BASE_URL` | selected local gateway | Optional root override; defaults to port 3007 for LiteLLM or 3000 for Bifrost |
| `DEMO_OPENWEBUI_API_KEY` | `sk-lgos-litellm-demo` | Gateway key used by the OpenAI clients |

See [Chainlit settings](chainlit.md#settings-reference),
[Open WebUI setup](open-webui.md#setup), and the
[example graph catalog](graphs/index.md)
for component-specific details.
