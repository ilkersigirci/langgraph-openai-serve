# LGOS Open WebUI integration

Standalone Open WebUI Function sources and a local synchronization command.
Open WebUI itself runs from its official image; this project is not built or
published as a container.

```bash
cp .env.example .env
uv run --locked --env-file .env lgos-openwebui-sync
```

The command creates or updates the bundled Functions, discovers
provider-qualified LGOS models from Bifrost's catalog, and generates one
Workspace Model per LGOS model. Detailed metadata comes through Bifrost
pass-through so each generated model exposes the current LGOS runtime settings
as native per-chat Chat Variables. Each raw `Generic / ...` Function model is
kept active and public but hidden from the chat model selector. Each sync
removes obsolete generated wrappers and base visibility records without
deleting Functions or unrelated Workspace Models managed by users.

The Generic Function uploads chat attachments through the OpenAI Files API and
adds the returned `file_id` to the current user message. By default,
`OPENAI_FILES_BASE_URL` targets Bifrost and `OPENAI_FILES_PROVIDER` selects the
`lgos-files` provider. Point the base URL directly at `lgos-files-api` and leave
the provider empty to bypass the gateway. Compose mounts a small ASGI wrapper
that forces Open WebUI's native upload endpoint to use `process=false`, so Open
WebUI stores the original bytes without extracting or embedding their content.
This is a temporary workaround until Open WebUI provides native per-model
control; the detailed removal condition and upstream links are in the
[integration guide](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/open-webui.md#file-input).

The typed [settings model](src/lgos_openwebui/settings.py) is the source of
truth for the sync command's environment names, defaults, and descriptions.
