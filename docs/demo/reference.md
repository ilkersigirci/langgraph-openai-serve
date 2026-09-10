# Demo Settings And Commands

This reference describes the independently locked projects and Compose stack
under `demo/`. These commands and `DEMO_*` settings are not part of the
`langgraph-openai-serve` package API.

[`demo/.env.example`](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/demo/.env.example)
is the source of truth for demo environment values. Copy it to `.env` and
customize it before running the demo. This reference explains settings without
duplicating their defaults.

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
| `make run-postgres` | Start the demo PostgreSQL service on port 3001 |
| `make run-api` / `make run-api-a` | Run the published `lgos-a` container on port 3004 |
| `make run-api-b` | Run the published `lgos-b` container on port 3005 |
| `make deploy-api API_SERVICE=lgos-demo-api-a` | Deploy one API, wait for health, and register its metadata in the running LiteLLM gateway |
| `make run-files` | Run the published Files API container on port 3006 |
| `make run-bifrost` | Run Bifrost and its graph and Files API dependencies on port 3000 |
| `make run-litellm` | Run the LiteLLM UI edge and compatibility gateway with its API and Files dependencies on port 3000 |
| `make run-chainlit` | Run Chainlit on port 3002 and PostgreSQL; start the gateway separately or use `make compose` for the full stack |
| `make run-api-local` / `make run-api-a-local` | Set up checkpoints and run the editable local `lgos-a` process |
| `make run-api-b-local` | Set up checkpoints and run the editable local `lgos-b` process |
| `make run-files-local` | Run the independently locked local Files API process |
| `make run-chainlit-local` | Apply Chainlit migrations and run the local UI process |
| `make sync-openwebui` | Sync the Open WebUI Functions and generated LGOS Workspace Models |
| `make sync-litellm SYNC_ARGS='...'` | Register one LGOS catalog in LiteLLM's native model metadata; see [LiteLLM model sync](litellm-sync.md) |
| `make compose` | Run the stack with published project-owned images |
| `make compose-dev` | Build the local API, Files API, and Chainlit images; overlay LGOS only into the graph API image |
| `make compose-otel` | Run published images with the OTEL overlay |
| `make compose-otel-dev` | Build the checkout and run it with the OTEL overlay |
| `make sync` | Synchronize all four projects from their lockfiles |
| `make test` | Test all four projects from their lockfiles |
| `make test-postgres` | Run API interrupt/Store persistence and Chainlit OAuth token/refresh tests against PostgreSQL on port 3001 |
| `make lint` | Check all four projects with Ruff |
| `make check` | Run tests, lint, formatting, type checks, and Compose validation |

Host-side UI commands require a host-reachable `OPENAI_GATEWAY_BASE_URL`; see
the [Chainlit](chainlit.md#run-the-ui) and [Open WebUI](open-webui.md#setup)
client guides for the host-side commands.

From the repository root, `make test-litellm` and `make test-bifrost` run the
focused OpenAI SDK checks. `OPENAI_GATEWAY_TYPE=litellm|bifrost` selects the
gateway used by both maintained UIs. Responses and Files use its normal
managed/native routes. LiteLLM metadata comes from native `/model/info` after
[model sync](litellm-sync.md); only Bifrost uses catalog-detail pass-through.

## Stack Settings

| Setting | Purpose |
| --- | --- |
| `DEMO_IMAGE_TAG` | Tag selected for all project-owned demo images |
| `PUID` | Host user ID used by Compose services |
| `PGID` | Host group ID used by Compose services |
| `OPENAI_GATEWAY_TYPE` | Gateway used by both demo UIs: `litellm` or `bifrost` |
| `COMPOSE_PROFILES` | Native Compose profiles; `.env.example` selects the bundled gateway via `${OPENAI_GATEWAY_TYPE}`. Leave empty to use an existing gateway |
| `OPENAI_GATEWAY_BASE_URL` | Required gateway root without `/v1`; the example uses the selected service's Compose DNS name |
| `OPENAI_GATEWAY_API_KEY` | Gateway credential for Open WebUI and Chainlit mock login. Chainlit OAuth ignores it and uses the user's access token; see [Chainlit login](chainlit.md#persistence-and-login) |
| `LITELLM_SYNC_BASE_URL` | Native LiteLLM administrator-key root reachable from the deployment sync container; may differ from the UI's SSO endpoint |
| `LITELLM_MASTER_KEY` | Credential for model synchronization only. Export external admin keys from CI or the operator environment, not the shared UI `.env` |
| `DEMO_LITELLM_IMAGE` | Required image reference; change it in `.env` to select another compatible image. See [Docker Compose](docker.md#demo-services) |
| `RESTART_POLICY` | Restart policy for services configured by the OTEL overlay |
| `DEMO_OPENWEBUI_SECRET_KEY` | Open WebUI application secret; replace it outside local demos |

## OpenTelemetry Settings

These settings apply when using `make compose-otel` or
`make compose-otel-dev`:

| Setting | Purpose |
| --- | --- |
| `OTEL_COLLECTOR_GATEWAY_ENDPOINT` | Required. OTLP/HTTP base URL for the host or platform gateway |
| `OTEL_SERVICE_NAMESPACE` | Namespace default for application and Collector signals |
| `OTEL_DEPLOYMENT_ENVIRONMENT` | Environment default for application and Collector signals |
| `OTEL_HOST_NAME` | Required. Stable host identity added by the local Collector |

The OTEL overlay uses the OpenTelemetry `always_on` sampler, so application
traces are exported without SDK sampling. The selected remote backend owns
retention.

The OTEL overlay requires both `OTEL_COLLECTOR_GATEWAY_ENDPOINT` and
`OTEL_HOST_NAME`; set them per machine in `.env`. The endpoint URL scheme
controls transport security: use `https://` for TLS and `http://` only when the
gateway intentionally accepts cleartext OTLP/HTTP.

## Demo API Settings

| Setting | Purpose |
| --- | --- |
| `DEMO_API_PORT` | HTTP port used by `lgos-demo-api` |
| `DEMO_API_OPENAI_BASE_URL` | Upstream OpenAI-compatible base URL |
| `DEMO_API_OPENAI_API_KEY` | Upstream key for provider-backed graphs |
| `DEMO_API_OPENAI_MODEL` | Upstream generation model |
| `DEMO_API_OPENAI_EMBEDDING_MODEL` | Embedding model used by `lgos-rag` |
| `DEMO_API_POSTGRES_URI` | Database for LangGraph checkpoints, Store data, and interrupt coordination |
| `DEMO_API_FILES_BASE_URL` | Central Files API read by the `file-input` graph. |

The API also reads the package-owned `LGOS_OPENAI_API_PREFIX`,
`LGOS_OPENAI_API_DOCS_ENABLED`, and `LGOS_ENABLE_LANGFUSE` settings documented
in the package [Reference](../reference.md#settings). Its settings model supports
a local `.env` file; the installed LGOS package itself reads only process
environment values or explicit constructor arguments.

## Files API Settings

These settings belong only to the independent `demo/files_api` project.

| Setting | Purpose |
| --- | --- |
| `DEMO_API_FILES_PORT` | HTTP port used by `lgos-files-api`. |
| `DEMO_API_FILES_BUCKET` | Required S3-compatible bucket. |
| `DEMO_API_FILES_S3_ENDPOINT` | Optional S3-compatible endpoint; required by the Compose demo. |
| `DEMO_API_FILES_AWS_ACCESS_KEY_ID` | Required S3 access key passed explicitly to boto3. |
| `DEMO_API_FILES_AWS_SECRET_ACCESS_KEY` | Required S3 secret key passed explicitly to boto3. |
| `DEMO_API_FILES_AWS_DEFAULT_REGION` | Required S3 signing region passed explicitly to boto3. |

## Open WebUI Sync Settings

These settings configure the host-side Open WebUI synchronization command,
alongside the shared gateway values under [Stack Settings](#stack-settings).

| Setting | Purpose |
| --- | --- |
| `DEMO_OPENWEBUI_URL` | Open WebUI API used by the sync command |
| `DEMO_OPENWEBUI_ADMIN_EMAIL` | Open WebUI sync account |
| `DEMO_OPENWEBUI_ADMIN_PASSWORD` | Open WebUI sync password |

See [Chainlit settings](chainlit.md#settings-reference),
[Open WebUI setup](open-webui.md#setup), and the
[example graph catalog](graphs/index.md)
for component-specific details.
