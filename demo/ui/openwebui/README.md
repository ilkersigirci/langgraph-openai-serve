# LGOS Open WebUI integration

Standalone Open WebUI Function sources and a local synchronization command.
Open WebUI itself runs from its official image; this project is not built or
published as a container.

Start with **UserValves Simple / simple-graph** after synchronization. The small
[`uservalves_simple.py`](src/lgos_openwebui/functions/uservalves_simple.py)
Filter declares `use_history` and `audience` as native Open WebUI UserValves.
Open **Controls → Valves → Functions → UserValves Simple** to change your
preferences across chats. Its dedicated
Workspace Model has no Chat Variables form and reuses the Generic Pipe for
Responses transport. Keep the Filter attached only to this example.

The Function uses Responses exclusively and never connects directly to LGOS.
`OPENAI_GATEWAY_TYPE=litellm|bifrost` selects the gateway for both inference and
Files. LiteLLM uses managed Responses routing; Bifrost uses its native
Responses route. A separate catalog client uses only the gateway's
model-detail pass-through when necessary to preserve LGOS descriptions,
features, and client-setting schemas.

```bash
cp .env.example .env
uv run --locked --env-file .env lgos-openwebui-sync
```

The command creates or updates the bundled Functions, discovers LGOS models
through the selected gateway, and generates one Workspace Model per provider
and graph. Detailed metadata comes from catalog-only pass-through routes; each
model keeps the routing identity required by its gateway. Each generated model
exposes the current LGOS runtime settings as native per-chat Chat Variables.
When `lgos-a/simple-graph` has valid metadata, sync also adds the separate
`lgos.uservalves_simple` example with its UserValves Filter. Each raw `Generic / ...`
Function model is kept active and public but hidden from the chat model
selector. Each sync
removes obsolete generated wrappers and base visibility records without
deleting Functions or unrelated Workspace Models managed by users.

The Generic Function uploads chat attachments through the selected gateway's
normal OpenAI Files API and adds the returned `file_id` to the current user
message. LiteLLM assigns the operation to `litellm_proxy`; Bifrost assigns it
to `lgos-files`. Compose mounts a small ASGI wrapper
that forces Open WebUI's native upload endpoint to use `process=false`, so Open
WebUI stores the original bytes without extracting or embedding their content.
This is a temporary workaround until Open WebUI provides native per-model
control; the detailed removal condition and upstream links are in the
[integration guide](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/open-webui.md#file-input).

The typed [settings model](src/lgos_openwebui/settings.py) is the source of
truth for the sync command's environment names, defaults, and descriptions.
