# LGOS Open WebUI integration

The Generic Pipe serves the LGOS graphs of an OpenAI-compatible gateway in Open
WebUI. `lgos-openwebui-sync` installs it into a running Open WebUI through its
API, from this checkout, a deployment machine, or CI. The demo uses the same
command, so Open WebUI runs its image unchanged: no source mount, package
installation, or restart.

## Sync a deployment

Open WebUI must run a compatible
[`homeserver-open-webui`](https://github.com/ilkersigirci/homeserver-docker/tree/main/Dockerfiles/OpenWebUI)
image. Its patches give the Pipe per-model info, the user's OAuth token, and raw
uploads, and let a connection leave out the Pipe's models. The command needs
only an Open WebUI admin; keep its API key in a
private environment file, for example `/absolute/path/openwebui-sync.env`:

```dotenv
OPENWEBUI_API_KEY=YOUR_OPENWEBUI_ADMIN_API_KEY
```

Run it with uvx, which needs no checkout; `COMMIT` selects the Function source
to install:

```bash
uvx --env-file /absolute/path/openwebui-sync.env \
  --from "git+https://github.com/ilkersigirci/langgraph-openai-serve@COMMIT#subdirectory=demo/ui/openwebui" \
  lgos-openwebui-sync \
  --url https://chat.example.com \
  --gateway-type litellm \
  --gateway-url https://gateway.example.com \
  --model-prefix acme
```

From a checkout, `uv run --directory demo/ui/openwebui --locked lgos-openwebui-sync`
runs the same command with the locked dependencies.

`OPENWEBUI_API_KEY` is an admin's API key, which needs `ENABLE_API_KEYS=true`.
Where password sign-in is on, `OPENWEBUI_ADMIN_EMAIL` and
`OPENWEBUI_ADMIN_PASSWORD` work instead. `--gateway-url` is the gateway root
without `/v1`, as Open WebUI's own OpenAI connection reaches it. Run
`lgos-openwebui-sync --help` for every option.

The command creates the Generic Function and enables it, or updates its source;
an existing Function keeps its enabled state, metadata, and other valves. It
sets the Pipe's valves to the gateway, the `acme` model namespace, and the MCP
connection ID (`--mcp-connection-id`, default `lgos-gateway`). An unchanged run
writes nothing. Rerun the command after changing the Function source or these
settings; it never deletes, so after a `--function-id` change, delete the old
Function, which still lists the graphs.

The Pipe sends what Open WebUI's OpenAI connection to `<gateway-url>/v1` sends:
its API key with `bearer`, or each user's token with `system_oauth`. It reads
that connection on every request, so its valves hold no credential; without the
connection, it lists no graphs.

The gateway must already publish the graphs as `acme/<graph>`, for example
through the [LiteLLM sync](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/litellm-sync.md)
with `--prefix acme`. The Pipe lists them as `generic.acme/<graph>`, with the
descriptions, settings, and tools it reads from the gateway on every model-list
load; chats refer to that ID, so keep the prefix and `--function-id` stable.

The command does not configure Open WebUI itself. Set up the deployment as the
demo's Compose does: the OpenAI connection to the gateway, with
`"exclude_model_ids": ["acme/*"]` in its `OPENAI_API_CONFIGS` entry so that it
does not list the graphs again as plain `acme/<graph>` models without the Pipe's
features; an MCP connection to its `/mcp` route with the `--mcp-connection-id`
ID; `BYPASS_MODEL_ACCESS_CONTROL=true`; `ENABLE_BASE_MODELS_CACHE=false`; and
`ENABLE_FILE_UPLOAD_PROCESSING=false`. The
[integration guide](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/open-webui.md)
explains each setting and the image's patches.

## Demo

`just demo/compose` runs `just demo/sync-openwebui` once Open WebUI is up; rerun
it after editing a Function. The demo also installs the **UserValves Simple /
simple-graph** example with `--include-demo-functions`. The SSO overlay turns
password sign-in off, so set `DEMO_OPENWEBUI_SYNC_API_KEY` to an SSO admin's API
key.

## Development

The Pipe runs on the image's Python packages, so the dependency pins equal the
versions the image ships, and the tests run them. From this directory:

```bash
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ty check src
```

After changing the demo's Open WebUI image, run `just demo/test-openwebui`
against the running stack. The sync calls Open WebUI's undocumented Functions
endpoints, and the Pipe reads its connection settings through an Open WebUI
internal; the live checks catch an upgrade that changes either.
