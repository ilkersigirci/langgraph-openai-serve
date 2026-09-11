# Demo Settings And Commands

This reference describes the independently locked projects and Compose stack
under `demo/`. These commands and `DEMO_*` settings are not part of the
`langgraph-openai-serve` package API.

[`demo/.env.example`](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/demo/.env.example)
is the source of truth for demo environment values. Copy it to `demo/.env` and
customize it before starting services or running live integration tests.
Just loads it when present; exported environment variables take precedence.
Local tests, lint, formatting, type checks, and dependency synchronization can
run without it. This reference explains settings without duplicating their defaults.

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

Run these from the repository root. Configure `demo/.env` for service and
integration commands:

| Command | Purpose |
| --- | --- |
| `just demo/up <service>` | Start one published Compose service and its dependencies; add `--dev` for checkout images or `--wait` to detach |
| `just demo/api [--editable] [--port <port>]` | Set up checkpoints and run one local graph API process |
| `just demo/files [--port <port>]` | Run the independently locked local Files API process |
| `just demo/chainlit [--port <port>]` | Apply Chainlit migrations and run the local UI process |
| `just demo/marimo [--editable]` | Open the API notebook workspace |
| `just demo/sync-openwebui` | Sync the Open WebUI Functions and generated LGOS Workspace Models |
| `just demo/sync-litellm [--dev] -- <arguments>` | Run the one-shot container to register one LGOS catalog in LiteLLM; see [model sync](litellm-sync.md) |
| `just demo/compose` | Start the published stack in dependency order, run its gateway-specific syncs, and leave it healthy in the background |
| `just demo/compose --dev` | Build this checkout and run the same ordered startup and sync |
| `just demo/compose --otel` | Run the ordered published stack with the OTEL overlay |
| `just demo/compose --dev --otel` | Build the checkout and run the ordered stack with the OTEL overlay |
| `just demo/down` | Stop and remove every stack variant |
| `just demo/sync` | Synchronize all four projects from their lockfiles |
| `just demo/test [--editable]` | Test all four projects, optionally overlaying the parent LGOS checkout |
| `just demo/test-postgres [--editable]` | Run API interrupt/Store persistence and Chainlit delegated-token tests against PostgreSQL on port 3001 |
| `just demo/lint` | Check all four projects with Ruff |
| `just demo/format` | Format the Justfile and fix Python style in all four projects; accepts Ruff flags such as `--unsafe-fixes` |
| `just demo/type-check [--editable]` | Type-check all four projects |
| `just demo/check [--editable]` | Run tests, lint, type checks, and Compose validation |

Common service names are `lgos-db`, `lgos-demo-api-a`, `lgos-demo-api-b`,
`lgos-files-api`, `lgos-bifrost`, `lgos-litellm`, `lgos-chainlit`, and
`lgos-openwebui`. Put arguments for the underlying command after `--` when a
recipe has its own options, for example
`just demo/test --editable -- -q`.

Use `just demo/` to list recipes in the `local`, `integration`, `checks`,
`docker prod`, and `docker dev` groups. Docker recipes use published images by
default; `--dev` selects builds from the current checkout. Shared recipes appear
in both Docker groups.

`just --usage demo/api` shows its options and defaults. The `--port` option
overrides the dotenv value; exported variables work too, for example
`LGOS_A_PORT=3104 just demo/api`. Add Just's `--dry-run` before the recipe to
inspect commands. To validate Compose using the template without creating an
environment file, run:

```bash
just --dotenv-path demo/.env.example demo/compose-config
```

Host-side UI commands use the host-reachable `DEMO_GATEWAY_HOST_URL`; see
the [Chainlit](chainlit.md#run-the-ui) and [Open WebUI](open-webui.md#setup)
client guides for the host-side commands.

`just demo/test-litellm --editable` and
`just demo/test-bifrost --editable` run the focused OpenAI SDK
checks.
`OPENAI_GATEWAY_TYPE=litellm|bifrost` selects the
gateway used by both maintained UIs. Responses and Files use its normal
managed/native routes. LiteLLM metadata comes from native `/model/info` after
[model sync](litellm-sync.md); only Bifrost uses catalog-detail pass-through.

## Stack Settings

| Setting | Purpose |
| --- | --- |
| `DEMO_IMAGE_TAG` | Tag selected for all project-owned demo images |
| `PUID` | Host user ID used by Compose services |
| `PGID` | Host group ID used by Compose services |
| `LGOS_*_PORT` | Host ports for the gateway, database, UIs, demo APIs, and Files API |
| `DEMO_GATEWAY_HOST_URL` | Gateway root used by host-side synchronization and integration tests |
| `OPENAI_GATEWAY_TYPE` | Gateway used by both demo UIs: `litellm` or `bifrost` |
| `COMPOSE_PROFILES` | Native Compose profiles; `.env.example` selects the bundled gateway via `${OPENAI_GATEWAY_TYPE}`. Leave empty to use an existing gateway |
| `OPENAI_GATEWAY_BASE_URL` | Required gateway root without `/v1`; the example uses the selected service's Compose DNS name |
| `OPENAI_GATEWAY_API_KEY` | Static gateway credential used by Open WebUI. The bundled LiteLLM configuration also uses it as its demo master key |
| `DEMO_CHAINLIT_GATEWAY_API_KEY` | Static Chainlit gateway credential used with mock or OAuth login. Leave empty only when OAuth token forwarding is enabled |
| `DEMO_CHAINLIT_ENABLE_OAUTH_TOKEN_FORWARDING` | Forward the signed-in user's OAuth access token to the gateway instead of using the static Chainlit key; see [Chainlit login](chainlit.md#persistence-and-login) |
| `LITELLM_SYNC_BASE_URL` | Native LiteLLM administrator-key root reachable from the deployment sync container; may differ from the UI's SSO endpoint |
| `LITELLM_MASTER_KEY` | Credential for model synchronization only. Export external admin keys from CI or the operator environment, not the shared UI `demo/.env` |
| `DEMO_LITELLM_IMAGE` | Required image reference; change it in `demo/.env` to select another compatible image. See [Docker Compose](docker.md#demo-services) |
| `RESTART_POLICY` | Restart policy for services configured by the OTEL overlay |
| `DEMO_OPENWEBUI_SECRET_KEY` | Open WebUI application secret; replace it outside local demos |

## Integration Test Settings

Live test recipes read their endpoints from the `DEMO_TEST_*` values in
`demo/.env.example`. Their native options can override those defaults:

```bash
just demo/test-postgres --uri postgresql://lgos:lgos@localhost:5432/lgos --editable
just demo/test-direct --base-urls http://localhost:3104/v1 --files-url http://localhost:3106/v1
just demo/test-litellm --base-url https://litellm.example.com/v1 -- --verbose
```

Use `just --usage demo/test-bifrost` for the normalized, catalog, and
pass-through endpoint options. `--editable` overlays the parent LGOS checkout;
arguments after `--` go to pytest. CI can export `DEMO_API_TEST_POSTGRES_URI`
and run `just demo/test-postgres --editable` without a dotenv file.

## OpenTelemetry Settings

These settings apply when using `just demo/compose --otel`,
optionally together with `--dev`:

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
`OTEL_HOST_NAME`; set them per machine in `demo/.env`. The endpoint URL scheme
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
