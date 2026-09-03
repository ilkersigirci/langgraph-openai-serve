# Coding Agents

Coding agents can use any registered LGOS graph as an OpenAI-compatible
`model`. The agent owns the tools: it sends tool definitions, receives tool
calls, executes them locally, and sends the matching `tool` messages on the
next turn.

The demo [`simple-graph-external-tools`](../demo/graphs/core-patterns.md#simple-graph-external-tools)
is the smallest tool-enabled graph. It forwards client-provided tools to the
upstream chat model and does not execute them.

## Endpoint And Model

Use the exact model ID returned by `GET /v1/models`:

| Connection | Base URL | Model ID |
| --- | --- | --- |
| Direct LGOS | `https://lgos.example.com/v1` | `simple-graph-external-tools` |
| Bifrost catalog | `https://bifrost.example.com/v1` | `lgos-a/simple-graph-external-tools` |

Bifrost adds the provider prefix to its catalog IDs. Replace `lgos-a` with the
provider that owns the selected LGOS deployment. See the
[Bifrost demo](../demo/bifrost.md) for the catalog and pass-through boundary.

=== "Codex"

    Codex uses the Responses API. LGOS uses Chat Completions, so put a
    Responses-to-Chat gateway such as Bifrost in front of it. Use the
    gateway's normal `/v1` route; a raw pass-through would forward the
    Responses request unchanged to LGOS.

    Based on Codex's [custom provider configuration](https://learn.chatgpt.com/docs/config-file/config-advanced):

    ```toml
    # ~/.codex/config.toml
    model = "lgos-a/simple-graph-external-tools"
    model_provider = "lgos_bifrost"

    [model_providers.lgos_bifrost]
    name = "LGOS via Bifrost"
    base_url = "https://bifrost.example.com/v1"
    wire_api = "responses"
    requires_openai_auth = false
    ```

    ```bash
    codex --model lgos-a/simple-graph-external-tools
    ```

    For an authenticated gateway, add `env_key = "BIFROST_API_KEY"` and
    export that variable. For an unauthenticated gateway, omit `env_key`; no
    environment variable is needed.

=== "OpenCode"

    OpenCode can call the Chat Completions route directly. Add a custom
    provider to `opencode.json` using its
    [OpenAI-compatible provider](https://opencode.ai/docs/providers):

    ```json
    {
      "$schema": "https://opencode.ai/config.json",
      "provider": {
        "lgos": {
          "npm": "@ai-sdk/openai-compatible",
          "name": "LGOS via Bifrost",
          "options": {
            "baseURL": "https://bifrost.example.com/v1"
          },
          "models": {
            "lgos-a/simple-graph-external-tools": {
              "name": "LGOS external tools"
            }
          }
        }
      }
    }
    ```

    Start OpenCode, run `/models`, and select `LGOS external tools`. For a
    direct LGOS URL, use the bare model ID `simple-graph-external-tools`.

=== "pi"

    pi uses `~/.pi/agent/models.json`. Its
    [OpenAI Completions provider](https://pi.dev/docs/latest/models) is the
    Chat Completions integration:

    ```json
    {
      "providers": {
        "lgos": {
          "baseUrl": "https://bifrost.example.com/v1",
          "api": "openai-completions",
          "apiKey": "DUMMY",
          "models": [
            {
              "id": "lgos-a/simple-graph-external-tools",
              "name": "LGOS external tools"
            }
          ]
        }
      }
    }
    ```

    ```bash
    pi --provider lgos --model lgos-a/simple-graph-external-tools
    ```

    `DUMMY` only makes the keyless model appear in pi's model selector; the
    unauthenticated gateway does not need an API key. Use `$BIFROST_API_KEY`
    instead when the gateway requires authentication.

## Other Agents

For any agent that supports OpenAI Chat Completions, configure its base URL as
`https://lgos.example.com/v1` (or the Bifrost `/v1` route) and use the model ID
from the model catalog. No LGOS-specific SDK or agent plugin is required.
