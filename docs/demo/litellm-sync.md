# LiteLLM Model Sync

Sync LGOS model metadata into LiteLLM's native `model_info.lgos` field.
The command belongs to LGOS; both UIs read the gateway's `/model/info` endpoint.

## Deployment Automation

From `demo/`, start LiteLLM with database model storage enabled
(`make run-litellm` for the bundled gateway). Then deploy each API:

```bash
make deploy-api API_SERVICE=lgos-demo-api-a
make deploy-api API_SERVICE=lgos-demo-api-b
```

Each command waits for API health, then runs a fresh, one-shot sync container
using the same API image. Use it as the API's CI/CD deployment step; ordinary
`compose up`, restarts, and development watch do not sync metadata.

A failed health check skips sync. A sync failure fails the command without
stopping or rolling back the API. The temporary sync container is removed.

After syncing, reload Chainlit's profiles and run `make sync-openwebui` with a
[host-reachable gateway URL](open-webui.md#setup) to refresh its Workspace Models.

For another API, add its matching `<service>-sync` job alongside its Compose
service, using the same image and a unique namespace. Follow
[`docker/apps/demo-api.yml`](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/demo/docker/apps/demo-api.yml).

??? note "Deploying development images"

    ```bash
    make deploy-api API_SERVICE=lgos-demo-api-a \
      COMPOSE='docker compose --env-file .env -f docker/compose/demo.yml -f docker/compose/development.yml'
    ```

    Also include `-f docker/compose/otel.yml` if that overlay is already in use.

## Gateway Credentials

- `LITELLM_SYNC_BASE_URL`: the native administrator-key root reachable from
  the sync container, not the delegated SSO endpoint.
- `LITELLM_MASTER_KEY`: export an external gateway's admin key from CI or the
  operator environment. Do not put it in the shared UI `.env` file.

The bundled defaults in `demo/.env.example` use `OPENAI_GATEWAY_BASE_URL` and
`OPENAI_GATEWAY_API_KEY`. Keep an existing `.env` in sync with those settings.
External gateways are not started or reconfigured. Use HTTPS outside trusted
local networks.

## Manual Sync

From `demo/`, with `LITELLM_MASTER_KEY` configured:

```bash
make sync-litellm SYNC_ARGS='--gateway-url https://litellm.example.com --source-url http://localhost:3004/v1 --prefix lgos-a --api-base http://graph-host:3004/v1 --dry-run'
```

Remove `--dry-run` to apply. Use `make sync-litellm SYNC_ARGS='--help'` for all
options. This host-side command overlays the current LGOS checkout.

`--source-url` must be reachable from the sync process; `--api-base` must be
reachable from LiteLLM. Both point to LGOS `/v1` and may differ. For authenticated
upstreams, pass `--api-key-env` with the key's environment variable name;
otherwise the command uses `DUMMY` for the unauthenticated demo API.

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
