# LGOS Open WebUI integration

Standalone Open WebUI Function sources and a local synchronization command.
Open WebUI itself runs from its official image; this project is not built or
published as a container.

```bash
cp .env.example .env
uv run --env-file .env lgos-openwebui-sync
```

The command creates or updates the bundled Functions and generates one Workspace
Model per LGOS model. Each generated model exposes the current LGOS runtime
settings as native per-chat Chat Variables. Its public manifold base remains
hidden so only the generated model appears in the selector. The command does
not delete Functions or Workspace Models managed by users.

Application settings use the `DEMO_OPENWEBUI_` prefix.
