# LiteLLM Model Sync

Reconcile an LGOS model catalog into LiteLLM and copy its metadata into the
native `model_info.lgos` field. The command belongs to LGOS; both UIs read the
gateway's `/model/info` endpoint.

## Usage

`just demo/compose` syncs both demo catalogs after LiteLLM and the
APIs are healthy, before starting the UIs. The `--dev` and `--otel` flags keep
the same sequence.

To sync any other healthy LGOS API:

```bash
just demo/sync-litellm -- \
  --source-url http://lgos-api:8000/v1 \
  --prefix team-a
```

The shared `lgos-model-sync` job starts no dependencies. The deployment system
owns rollout and health checks; this command only copies the catalog. Use a
unique prefix for each API. After a manual sync, reconnect Chainlit and run
`just demo/sync-openwebui` with a
[host-reachable gateway URL](open-webui.md#setup).

## Gateway Credentials

- `LITELLM_SYNC_BASE_URL`: the native administrator-key root reachable from
  the sync container, not the delegated SSO endpoint.
- `LITELLM_MASTER_KEY`: export an external gateway's admin key from CI or the
  operator environment. Do not put it in the shared UI `demo/.env` file.

The bundled defaults in `demo/.env.example` use `OPENAI_GATEWAY_BASE_URL` and
`OPENAI_GATEWAY_API_KEY`. Keep an existing `demo/.env` in sync with those
settings. External gateways are not started or reconfigured. Use HTTPS outside
trusted local networks.

## Options

Preview changes with:

```bash
just demo/sync-litellm -- \
  --source-url http://lgos-demo-api-a:8000/v1 \
  --prefix lgos-a \
  --dry-run
```

Remove `--dry-run` to apply. Use
`just demo/sync-litellm -- --help` for all options. The job uses the
already pulled or built demo API image.

`--source-url` must be reachable from the sync container. LiteLLM uses the same
URL for new deployments unless you pass `--api-base` with a different
gateway-reachable LGOS `/v1` URL. For authenticated upstreams, use `--api-key-env`
and provide that variable to the job through Compose's `environment` or
`docker compose run -e`; otherwise the demo uses `DUMMY`.

??? note "Running without Docker"

    Run the CLI directly with host-reachable URLs:

    ```bash
    uv run --directory demo/api --locked --with-editable ../.. --env-file ../.env \
      lgos-demo-api-sync-litellm --gateway-url http://localhost:3000 \
      --source-url http://localhost:3004/v1 --prefix lgos-a \
      --api-base http://lgos-demo-api-a:8000/v1
    ```

## Sync Behavior

- Validates the source catalog before writing. Ambiguous duplicate names and
  conflicting deployments not owned by this sync are rejected.
- Creates missing `<prefix>/<model>` deployments with the full LGOS metadata
  and native streaming enabled. New deployments also allow Chat `user` forwarding
  via [`allowed_openai_params: [user]`](https://docs.litellm.ai/docs/completion/drop_params#set-allowed_openai_params-on-configyaml).
  This is caller-supplied context, not authentication. Models without
  `model_info.lgos` are not shown by the UI integrations.
- Updates only changed LGOS and streaming metadata. Existing routing,
  credentials, pricing, and rate limits are preserved.
- Deletes database-backed models under the requested prefix when they are no
  longer in the LGOS source catalog. The prefix and explicit
  `model_info.lgos_sync` marker define ownership; unmarked models, config models,
  and models under other prefixes are not changed or removed.
- Leaves routing, pricing, credentials, and independently managed model
  retirement under [LiteLLM's Admin UI](https://docs.litellm.ai/docs/proxy/model_management).
- Stops on a failed write; earlier successful writes remain. Rerunning is safe.
