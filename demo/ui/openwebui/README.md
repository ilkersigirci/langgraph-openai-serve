# LGOS Open WebUI integration

Standalone Open WebUI Function sources and a local synchronization command.
Open WebUI itself runs from its official image; this project is not built or
published as a container.

```bash
cp .env.example .env
uv run --env-file .env lgos-openwebui-sync
```

The command creates or updates the bundled Functions without deleting Functions
managed by users.

Application settings use the `DEMO_OPENWEBUI_` prefix.
