# Coding Agents

A coding agent can use a registered LGOS graph when the agent's OpenAI wire
requests stay inside LGOS's supported Responses or Chat Completions subset. The
agent owns client tools: it sends function definitions, receives calls, executes
them locally, and returns matching outputs.

The demo
[`simple-graph-external-tools`](../demo/graphs/core-patterns.md#simple-graph-external-tools)
is the smallest tool-enabled graph. It forwards client-provided tools to the
upstream model and does not execute them.

## Endpoint And Model

Connect directly to LGOS and use the exact model ID from `GET /v1/models`:

| Base URL | Model ID |
| --- | --- |
| `https://lgos.example.com/v1` | `simple-graph-external-tools` |

An optional gateway may use a provider-qualified routing ID, but it must pass
the native contract tests in the [proxy guide](../how-to-guides/openai-proxies.md).
The pinned Bifrost demo's native Responses route preserves the tested data-plane
contract, including `phase`; its normalized model-detail and error metadata
remain lossy. The raw OpenAI pass-through route passes the complete tested
subset.

## Client Compatibility

LGOS implements a deliberately bounded Responses surface. A client that always
sends hosted tools, reasoning configuration, `include`, prompt-cache options,
`previous_response_id`, or another unsupported field will receive an explicit
OpenAI `invalid_request_error`. Do not place a Responses-to-Chat translator in
front of LGOS to hide that mismatch.

Codex custom providers use the Responses wire API. Direct Codex compatibility
has not been verified against LGOS's bounded subset, and Codex can configure
Responses controls that LGOS does not accept. The configuration shape is
documented by
[Codex custom model providers](https://learn.chatgpt.com/docs/config-file/config-advanced#custom-model-providers);
add a runnable example here only after a direct integration test passes without
request rewriting.

The examples below intentionally exercise the direct Chat compatibility route.
They are demonstrations for clients whose provider adapters use Chat
Completions; the maintained Chainlit and Open WebUI demos remain
Responses-only.

=== "OpenCode"

    OpenCode's official provider guide assigns
    `@ai-sdk/openai-compatible` to `/v1/chat/completions` providers. Add a
    direct LGOS provider to `opencode.json`:

    ```json
    {
      "$schema": "https://opencode.ai/config.json",
      "provider": {
        "lgos": {
          "npm": "@ai-sdk/openai-compatible",
          "name": "Direct LGOS",
          "options": {
            "baseURL": "https://lgos.example.com/v1",
            "apiKey": "DUMMY"
          },
          "models": {
            "simple-graph-external-tools": {
              "name": "LGOS external tools"
            }
          }
        }
      }
    }
    ```

    Start OpenCode, run `/models`, and select `LGOS external tools`. See
    OpenCode's [custom provider documentation](https://opencode.ai/docs/providers/#custom-provider).

=== "pi"

    pi supports both OpenAI APIs. This compatibility example selects its
    `openai-completions` adapter in `~/.pi/agent/models.json`:

    ```json
    {
      "providers": {
        "lgos": {
          "baseUrl": "https://lgos.example.com/v1",
          "api": "openai-completions",
          "apiKey": "DUMMY",
          "compat": {
            "supportsReasoningEffort": false
          },
          "models": [
            {
              "id": "simple-graph-external-tools",
              "name": "LGOS external tools"
            }
          ]
        }
      }
    }
    ```

    ```bash
    pi --provider lgos --model simple-graph-external-tools
    ```

    `DUMMY` makes a keyless development model appear in pi's selector. Use a
    real deployment credential when the LGOS host enforces authentication. See
    pi's [custom model documentation](https://pi.dev/docs/latest/models).

## Other Agents

For a Responses client, compare its emitted request with the exact
[supported subset](../explanation/openai-compatibility.md#supported-responses-subset)
and verify streaming plus function continuation before adopting it. For a Chat
Completions client, use the direct base URL and registered graph name. No
LGOS-specific SDK or agent plugin is required.
