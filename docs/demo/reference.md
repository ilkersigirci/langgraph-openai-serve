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
| `demo/api-coding-agent` | Coding-agent showcase and persistent workspace (currently Codex) | Yes, from PyPI by default |
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
| `just demo/up <service>` | Start one Compose service and its dependencies; add `--dev` for checkout images or `--wait` to detach |
| `just demo/api [--editable] [--port <port>]` | Set up checkpoints and run one local graph API process |
| `just demo/background-worker [--editable]` | Run the independently deployed Hatchet worker for polling-only background Responses |
| `just demo/files [--port <port>]` | Run the independently locked local Files API process |
| `just demo/test-coding-agent [--editable]` | Check coding-agent shell execution, persisted edits, streaming, usage, and history through the running gateway |
| `just demo/chainlit [--port <port>]` | Apply Chainlit migrations and run the local UI process |
| `just demo/marimo [--editable]` | Open the API notebook workspace |
| `just demo/sync-openwebui` | Sync the Open WebUI Functions, their gateway valves, and generated LGOS Workspace Models |
| `just demo/sync-litellm [--dev] -- <arguments>` | Run the one-shot container to register one LGOS catalog in LiteLLM; see [model sync](litellm-sync.md) |
| `just demo/sync-bifrost [--dev]` | Regenerate pricing and synchronize all LGOS graph catalogs into Bifrost; then refresh Open WebUI with `sync-openwebui` |
| `just demo/compose` | Start the default stack in dependency order, run its gateway-specific syncs, and leave it healthy in the background |
| `just demo/compose --dev` | Build this checkout and run the same ordered startup and sync |
| `just demo/compose --otel` | Run the ordered default stack with the OTEL overlay |
| `just demo/compose --dev --otel` | Build the checkout and run the ordered stack with the OTEL overlay |
| `just demo/compose --dev --chainlit-utils` | Build the checkout with Chainlit using an editable sibling [`chainlit-utils` checkout](docker.md#compose-modes) |
| `just demo/down` | Stop and remove every stack variant |
| `just demo/sync` | Synchronize every project from its lockfile |
| `just demo/test [--editable]` | Test every project, optionally overlaying the parent LGOS checkout |
| `just demo/test-postgres [--editable]` | Run API interrupt/Store persistence tests against PostgreSQL on port 3001 |
| `just demo/test-background-gateway [--editable]` | Exercise create, new-client polling, cancellation, idempotent replay, and polling-only validation through the selected gateway |
| `just demo/lint` | Check every project with Ruff |
| `just demo/format` | Format the Justfile and fix Python style in every project; accepts Ruff flags such as `--unsafe-fixes` |
| `just demo/type-check [--editable]` | Type-check every project |
| `just demo/check [--editable]` | Run tests, lint, type checks, and Compose validation |

Common service names are `lgos-db`, `lgos-demo-api-a`, `lgos-demo-api-b`,
`lgos-api-coding-agent`, `lgos-background-worker`,
`lgos-files-api`, `lgos-postgres-mcp`, `lgos-bifrost`, `lgos-litellm`,
`lgos-chainlit`, and `lgos-openwebui`. Put arguments for the underlying command
after `--` when a recipe has its own options, for example
`just demo/test --editable -- -q`.

Use `just demo/` to list recipes in the `local`, `integration`, `checks`,
`docker prod`, and `docker dev` groups. Docker recipes use published images by
default, except the coding-agent service, which always builds locally. `--dev`
selects checkout builds and editable LGOS for both graph API projects. Shared recipes appear
in both Docker groups.

`just --usage demo/api` shows its options and defaults. The `--port` option
overrides the dotenv value; exported variables work too, for example
`LGOS_A_PORT=3104 just demo/api`. Add Just's `--dry-run` before the recipe to
inspect commands. To validate Compose using the template without creating an
environment file, run:

```bash
just --dotenv-path demo/.env.example demo/compose-config
```

The local Chainlit and Open WebUI sync commands use the host-reachable
`DEMO_GATEWAY_HOST_URL`. Open WebUI itself remains the unchanged pinned
upstream image. See the
[Chainlit](chainlit.md#run-the-ui) and [Open WebUI](open-webui.md#setup) guides.

`just demo/test-litellm --editable` and
`just demo/test-bifrost --editable` run the focused OpenAI SDK
checks.
`just demo/test-background-gateway --editable` selects the route and model from
`OPENAI_GATEWAY_TYPE`; use `--base-url` and `--model` to override them.
`OPENAI_GATEWAY_TYPE=litellm|bifrost` selects the
gateway used by both maintained UIs. Responses and Files use its normal
managed/native routes. LiteLLM metadata comes from native `/model/info` after
[model sync](litellm-sync.md); Bifrost metadata comes from native `/v1/models`
after [catalog sync](bifrost.md#declarative-model-metadata).

## Stack Settings

| Setting | Purpose |
| --- | --- |
| `DEMO_IMAGE_TAG` | Tag selected for the published demo images; the coding-agent image is built locally |
| `PUID` | Host user ID used by Compose services |
| `PGID` | Host group ID used by Compose services |
| `LGOS_*_PORT` | Host ports for the gateway, database, UIs, demo APIs, and Files API |
| `DEMO_GATEWAY_HOST_URL` | Gateway root used by the local Chainlit, API, worker, and notebook processes and integration tests |
| `OPENAI_GATEWAY_TYPE` | Gateway used by the demo UIs and model clients: `litellm` or `bifrost` |
| `COMPOSE_PROFILES` | Native Compose profiles; `.env.example` selects the bundled gateway via `${OPENAI_GATEWAY_TYPE}`. Leave empty to use an existing gateway |
| `OPENAI_GATEWAY_BASE_URL` | Required gateway root without `/v1`; the example uses the selected service's Compose DNS name |
| `OPENAI_GATEWAY_API_KEY` | Shared static credential used by both UIs for model discovery, Responses, Files, speech, and MCP, and by the graph and coding-agent APIs for their model and vector-store calls; the bundled LiteLLM configuration uses it as its demo master key and Bifrost loads it as its scoped demo virtual key |
| `DEMO_AUDIO_STT_MODEL` | Gateway model ID both UIs use to transcribe microphone input through `/v1/audio/transcriptions`; Chainlit hides its microphone when empty |
| `DEMO_AUDIO_TTS_MODEL` | Gateway model ID both UIs use to speak answers through `/v1/audio/speech`; Chainlit hides its read-aloud button when empty |
| `DEMO_AUDIO_TTS_VOICE` | OpenAI voice for spoken answers; Chainlit defaults to `alloy` |
| `OPENAI_UPSTREAM_BASE_URL` | Root, without `/v1`, of the OpenAI-compatible upstream behind the bundled LiteLLM's `openai/*` models and OpenAI passthrough; defaults to the OpenAI API. Bifrost reads its upstream from `DEMO_BIFROST_CONFIG` |
| `OPENAI_UPSTREAM_API_KEY` | Upstream API key the bundled gateways use for their `openai/*` graph, embedding, and speech models and for vector-store passthrough. It must keep files and vector stores in one upstream account |
| `DEMO_BIFROST_CONFIG` | Bifrost configuration file, relative to `demo/docker/apps/`. Point it at a gitignored `config.local.json` copy to change the upstream URL, which Bifrost cannot read from the environment |
| `LGOS_MCP_DB_PASSWORD` | Password for the dedicated read-only `lgos_mcp` PostgreSQL login; DBHub receives it through an interpolated individual connection field, so URL encoding is not required |
| `LGOS_MCP_AUTH_TOKEN` | Internal bearer token used by LiteLLM or Bifrost when it connects to DBHub |
| `DEMO_CHAINLIT_ENABLE_OAUTH_TOKEN_FORWARDING` | Forward the signed-in user's OAuth access token to the gateway instead of using the static Chainlit key; see [Chainlit login](chainlit.md#persistence-and-login) |
| `LITELLM_SYNC_BASE_URL` | Native LiteLLM administrator-key root reachable from the deployment sync container; may differ from the UI's SSO endpoint |
| `LITELLM_MASTER_KEY` | Credential for model synchronization only. Export external admin keys from CI or the operator environment, not the shared UI `demo/.env` |
| `DEMO_LITELLM_IMAGE` | Required image reference; change it in `demo/.env` to select another compatible image. See [Docker Compose](docker.md#demo-services) |
| `RESTART_POLICY` | Restart policy for services configured by the OTEL overlay |
| `DEMO_OPENWEBUI_SECRET_KEY` | Open WebUI application secret that also encrypts stored valves; replace it outside local demos. Changing it resets every stored valve, so re-run the Open WebUI sync and set other valves again |

## Integration Test Settings

Live test recipes read their endpoints from the `DEMO_TEST_*` values in
`demo/.env.example`. Their native options can override those defaults:

```bash
just demo/test-postgres --uri postgresql://lgos:lgos@localhost:5432/lgos --editable
just demo/test-direct --base-urls http://localhost:3104/v1 --files-url http://localhost:3106/v1
just demo/test-litellm --base-url https://litellm.example.com/v1 -- --verbose
```

Use `just --usage demo/test-bifrost` for the normalized and catalog endpoint
options. `--editable` overlays the parent LGOS checkout;
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

The demo API and its worker run `lgos serve` and `lgos worker`. Their server
settings, including `LGOS_POSTGRES_URI`, `LGOS_INTERRUPT_TTL_MINUTES`,
`LGOS_BACKGROUND`, `LGOS_HATCHET_WORKER_SLOTS`, and `LGOS_CORS_ORIGINS`, are
described in [Run The LGOS Server](../how-to-guides/server.md#settings). The
graphs share `OPENAI_GATEWAY_BASE_URL` and `OPENAI_GATEWAY_API_KEY` with the UIs.
Model calls use `/v1`; the advanced graph's knowledge Files and vector stores
use `/openai_passthrough/v1` with that same credential. The local API, worker,
and notebook recipes set the gateway root to `DEMO_GATEWAY_HOST_URL`.

Graph-specific settings use the `DEMO_API_` prefix:

| Setting | Purpose |
| --- | --- |
| `DEMO_API_OPENAI_CHAT_COMPLETIONS_MODEL` | Gateway model ID for the Chat Completions graphs and the default of `simple-graph`'s `model` setting; it must call tools without reasoning-specific parameters |
| `DEMO_API_OPENAI_RESPONSES_MODEL` | Gateway model ID for the Responses graphs: `advanced-graph`, `file-input`, and `server-tool`. `simple-graph` also offers it through Chat Completions |
| `DEMO_API_VECTOR_STORE_ID` | Shared knowledge-base ID searched by `advanced-graph`; required for document search and saved notes |
| `DEMO_API_OPENAI_EMBEDDING_MODEL` | Gateway embedding model ID used by `lgos-rag` |
| `DEMO_API_WEB_SEARCH_BACKEND` | `http` for self-hosted search or `openai` for the upstream Responses tool |
| `DEMO_API_WEB_SEARCH_URL` | SearXNG or Degoog JSON search endpoint used by the `http` backend |
| `DEMO_API_FILES_BASE_URL` | Central Files API read by the `file-input` and `advanced-graph` graphs. |
| `HATCHET_CLIENT_TOKEN` | Hatchet's native client credential shared by the API replicas and worker; leave it out of committed files outside this local template. |
| `HATCHET_CLIENT_NAMESPACE` | Native Hatchet resource prefix shared by the API replicas and worker. |
| `HATCHET_CLIENT_OPENTELEMETRY_EXCLUDED_ATTRIBUTES` | Native SDK JSON list of span attributes to omit. The demo defaults to `["payload","additional_metadata"]`; trace propagation is preserved. |

The background demo uses the `create_hatchet_task` defaults: 30-minute
schedule and one-hour execution timeouts with no retries. Hatchet's data
retention decides how long Responses stay retrievable. Self-hosted deployments
should inject
any additional native `HATCHET_CLIENT_*` connection settings into both the API
and worker processes.

The API serves the OpenAI routes at the package's default `/v1` prefix, which
the gateway wiring assumes. It also reads the package-owned
`LGOS_OPENAI_API_DOCS_ENABLED` and `LGOS_ENABLE_LANGFUSE` settings documented
in the package [Reference](../reference.md#settings). Its settings model supports
a local `.env` file; the installed LGOS package itself reads only process
environment values or explicit constructor arguments.

## Coding Agent Settings

These settings belong to `demo/api-coding-agent`. Compose derives the base URL
and key from the gateway settings; the other defaults live in `demo/.env.example`.

| Setting | Purpose |
| --- | --- |
| `DEMO_CODING_AGENT_BASE_URL` | Responses base URL for Codex. Compose sets `${OPENAI_GATEWAY_BASE_URL}/v1`. |
| `DEMO_CODING_AGENT_API_KEY` | Model credential for Codex. Compose sets `OPENAI_GATEWAY_API_KEY`; available inside the coding-agent container. |
| `DEMO_CODING_AGENT_MODEL` | Codex-compatible gateway model ID, defaulting to `DEMO_API_OPENAI_RESPONSES_MODEL`; separate from the public graph ID. |
| `DEMO_CODING_AGENT_TIMEOUT_SECONDS` | Active request time limit, excluding time waiting for another workspace request. Runtime cleanup may exceed the limit. |

See the [coding-agent graph guide](graphs/coding-agent.md) for its shared
workspace, container execution boundary, and invocation examples.

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

These settings configure the host-side Open WebUI synchronization command
alongside the shared gateway values under [Stack Settings](#stack-settings).
The command stores those gateway values in the Generic Function's valves.

| Setting | Purpose |
| --- | --- |
| `DEMO_OPENWEBUI_URL` | Open WebUI API used by the sync command |
| `DEMO_OPENWEBUI_ADMIN_EMAIL` | Open WebUI sync account |
| `DEMO_OPENWEBUI_ADMIN_PASSWORD` | Open WebUI sync password |

See [Chainlit settings](chainlit.md#settings-reference),
[Open WebUI setup](open-webui.md#setup), and the
[example graph catalog](graphs/index.md)
for component-specific details.
