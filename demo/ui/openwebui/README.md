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

The typed [settings model](src/lgos_openwebui/settings.py) is the source of
truth for the sync command's environment names, defaults, and descriptions.
