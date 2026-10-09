# LiteLLM Model Sync

Reconcile LGOS model catalogs into LiteLLM and copy their metadata into the
native `model_info.lgos` field. The command belongs to LGOS; both UIs read the
gateway's `/model/info` endpoint.

## Usage

`just demo/compose` syncs the demo API and coding-agent catalogs after LiteLLM and its
dependencies are healthy, then starts and syncs Open WebUI. Open the UIs after
the command completes. The `--dev` and `--otel` flags keep the same sequence.

To reconcile the bundled APIs together:

```bash
just demo/sync-litellm -- \
  --source-url http://lgos-demo-api:8000/v1 \
  --source-url http://lgos-api-coding-agent:8000/v1
```

The shared `lgos-model-sync` job starts no dependencies. The deployment system
owns rollout and health checks; this command only copies the catalog. Use a
single invocation containing every API in the `lgos` namespace. Graph IDs must
be unique across these APIs. For replicas of the same API, use one stable service
URL that balances traffic across replicas; their public model names stay unchanged.
After a manual sync, reconnect Chainlit and run
`just demo/sync-openwebui`; the host-side locked sync project uses the
[host-reachable gateway configuration](open-webui.md#setup).

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
  --source-url http://lgos-demo-api:8000/v1 \
  --source-url http://lgos-api-coding-agent:8000/v1 \
  --dry-run
```

Remove `--dry-run` to apply. Use
`just demo/sync-litellm -- --help` for all options. The job uses the
already pulled or built demo API image.

Repeat `--source-url` for each API. URLs must be reachable from the sync container
and LiteLLM. If these differ, repeat `--api-base` once per source in the same order
with the gateway-reachable URLs. `--prefix` defaults to `lgos`. For authenticated
upstreams, use `--api-key-env`
and provide that variable to the job through Compose's `environment` or
`docker compose run -e`; otherwise the demo uses `DUMMY`.

??? note "Running without Docker"

    For a separate namespace containing one host-reachable API:

    ```bash
    uv run --directory demo/api --locked --with-editable ../.. --env-file ../.env \
      lgos-demo-api-sync-litellm --gateway-url http://localhost:3000 \
      --source-url http://localhost:3004/v1 \
      --api-base http://lgos-demo-api:8000/v1 --prefix local
    ```

## Sync Behavior

- Validates every source catalog before writing. Duplicate graph IDs across APIs and
  conflicting deployments not owned by this sync are rejected.
- Creates missing `<prefix>/<model>` deployments with the full LGOS metadata
  and native streaming enabled. New deployments also allow Chat `user` forwarding
  via [`allowed_openai_params: [user]`](https://docs.litellm.ai/docs/completion/drop_params#set-allowed_openai_params-on-configyaml).
  This is caller-supplied context, not authentication. Models without
  `model_info.lgos` are not shown by the UI integrations.
- Updates only changed LGOS and streaming metadata. Existing routing,
  credentials, pricing, and rate limits are preserved.
- Deletes database-backed models under the requested prefix when they are no
  longer in any of the supplied LGOS catalogs. The prefix and explicit
  `model_info.lgos_sync` marker define ownership; unmarked models, config models,
  and models under other prefixes are not changed or removed.
- Leaves routing, pricing, credentials, and independently managed model
  retirement under [LiteLLM's Admin UI](https://docs.litellm.ai/docs/proxy/model_management).
- Stops on a failed write; earlier successful writes remain. Rerunning is safe.

When retiring or renaming a provider namespace, remove its sync-owned deployments
through LiteLLM's model management API or Admin UI. Syncing a new namespace does
not delete entries in another namespace. Then rerun Open WebUI sync to remove
its obsolete generated Workspace Models and hidden bases.
