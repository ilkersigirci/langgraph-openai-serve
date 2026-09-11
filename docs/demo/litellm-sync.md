# LiteLLM Model Sync

Sync LGOS model metadata into LiteLLM's native `model_info.lgos` field.
The command belongs to LGOS; both UIs read the gateway's `/model/info` endpoint.

## Usage

The full-stack `make compose` variants sync both demo catalogs after LiteLLM
and the APIs are healthy, before starting the UIs.

To sync any other healthy LGOS API, run from `demo/`:

```bash
make sync-litellm \
  SYNC_ARGS='--source-url http://lgos-api:8000/v1 --prefix team-a'
```

The shared `lgos-model-sync` job starts no dependencies. The deployment system
owns rollout and health checks; this command only copies the catalog. Use a
unique prefix for each API. After a manual sync, reconnect Chainlit and run
`make sync-openwebui` with a
[host-reachable gateway URL](open-webui.md#setup).

## Gateway Credentials

- `LITELLM_SYNC_BASE_URL`: the native administrator-key root reachable from
  the sync container, not the delegated SSO endpoint.
- `LITELLM_MASTER_KEY`: export an external gateway's admin key from CI or the
  operator environment. Do not put it in the shared UI `.env` file.

The bundled defaults in `demo/.env.example` use `OPENAI_GATEWAY_BASE_URL` and
`OPENAI_GATEWAY_API_KEY`. Keep an existing `.env` in sync with those settings.
External gateways are not started or reconfigured. Use HTTPS outside trusted
local networks.

## Options

Preview changes with:

```bash
make sync-litellm SYNC_ARGS='--source-url http://lgos-demo-api-a:8000/v1 --prefix lgos-a --dry-run'
```

Remove `--dry-run` to apply. Use `make sync-litellm SYNC_ARGS='--help'` for all
options. The job uses the already pulled or built demo API image.

`--source-url` must be reachable from the sync container. LiteLLM uses the same
URL for new deployments unless you pass `--api-base` with a different
gateway-reachable LGOS `/v1` URL. For authenticated upstreams, use `--api-key-env`
and provide that variable to the job through Compose's `environment` or
`docker compose run -e`; otherwise the demo uses `DUMMY`.

??? note "Running without Docker"

    From `demo/`, run the CLI directly with host-reachable URLs:

    ```bash
    uv run --directory api --locked --with-editable ../.. --env-file ../.env \
      lgos-demo-api-sync-litellm --gateway-url http://localhost:3000 \
      --source-url http://localhost:3004/v1 --prefix lgos-a \
      --api-base http://lgos-demo-api-a:8000/v1
    ```

## Sync Behavior

- Validates the source catalog before writing. Ambiguous duplicate names and
  config-owned deployments are rejected.
- Creates missing `<prefix>/<model>` deployments with the full LGOS metadata
  and native streaming enabled. New deployments also allow Chat `user` forwarding
  via [`allowed_openai_params: [user]`](https://docs.litellm.ai/docs/completion/drop_params#set-allowed_openai_params-on-configyaml).
  This is caller-supplied context, not authentication. Models without
  `model_info.lgos` are not shown by the UI integrations.
- Updates only changed LGOS and streaming metadata. Existing routing,
  credentials, pricing, and rate limits are preserved.
- Does not delete removed upstream models. Manage retirement, routing, pricing,
  and credential rotation through [LiteLLM's Admin UI](https://docs.litellm.ai/docs/proxy/model_management).
- Stops on a failed write; earlier successful writes remain. Rerunning is safe.
