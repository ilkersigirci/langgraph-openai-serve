# LGOS Open WebUI integration

Standalone Open WebUI Function sources and the app factory that runs them.
Compose runs the pinned `homeserver-open-webui` image and starts its Uvicorn
server with `lgos_openwebui.app:create_app` from a read-only mount of
`src/lgos_openwebui`. When Open WebUI starts, the factory installs the bundled
Functions; restart the container after changing one. The
[integration guide](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/open-webui.md#setup)
describes the install and the image's patches.

The Functions and the app factory run on the image's Python packages, so this
project pins its dependencies to the versions the image ships and its tests run
the same versions; change them with the image pin. The image's
OpenAI SDK omits custom-tool output from one generated response union; the
Function applies a narrow compatibility shim for that type until Open WebUI
updates its SDK. Both `web_search` and the demo's `package_version` custom tool remain
available without replacing packages in the official image.

Start with **UserValves Simple / simple-graph**. The small
[`uservalves_simple.py`](src/lgos_openwebui/functions/uservalves_simple.py)
Filter declares `use_history` and `audience` as native Open WebUI UserValves.
Open **Controls → Valves → Functions → UserValves Simple** to change your
preferences across chats. The example has no Chat Variables form and reuses the
Generic Pipe for Responses transport, which attaches the Filter only to this
example.

The Function uses Responses exclusively and never connects directly to LGOS.
It reads `OPENAI_GATEWAY_TYPE=litellm|bifrost`, `OPENAI_GATEWAY_BASE_URL`,
and `OPENAI_GATEWAY_API_KEY` from Open WebUI's environment. Its model list,
inference, and Files send what Open WebUI's own OpenAI connection to
`${OPENAI_GATEWAY_BASE_URL}/v1` sends: the key with `bearer`, or the signed-in
user's SSO access token with `system_oauth`. LiteLLM uses managed Responses
routing; Bifrost uses its
native Responses route. LiteLLM metadata comes from native `/model/info`;
Bifrost uses native `/v1/models` with complete metadata encoded in
`additional_attributes.lgos`. Before using independently started LiteLLM
components, [sync the LGOS metadata](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/litellm-sync.md).
The full-stack `just demo/compose [--dev] [--otel]` variants do this.

Each time Open WebUI lists models, the Generic Pipe lists every LGOS model with
its description, file-upload capability, gateway MCP tools, and current LGOS
runtime settings as native per-chat Chat Variables. The image keeps that info
from the Pipe's list, so graph changes appear on Open WebUI's next model-list
load. Open WebUI's own OpenAI connection uses the same gateway and credential
for every other model, and the app factory hides its `lgos/` models. See the
[integration guide](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/open-webui.md#setup)
for the pinned-release contract and its access-control setting.

Compose declares one native `lgos-gateway` MCP connection through Open WebUI's
`TOOL_SERVER_CONNECTIONS`, and models that advertise the `mcp_tools` feature
select it by default. The connection uses `OPENAI_GATEWAY_BASE_URL` and
`OPENAI_GATEWAY_API_KEY`; the gateway decides which tools that credential may
discover.

The Generic Function uploads chat attachments through the selected gateway's
normal OpenAI Files API and adds the returned `file_id` to the current user
message. LiteLLM assigns the operation to `litellm_proxy`; Bifrost assigns it
to `lgos-files`. Compose sets the image's `ENABLE_FILE_UPLOAD_PROCESSING=false`,
so Open WebUI stores the original bytes without extracting or embedding their
content. This is a temporary workaround until Open WebUI can keep chat uploads
raw natively; the detailed removal condition and upstream links are in the
[integration guide](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/open-webui.md#file-input).

The demo's [`.env.example`](../../.env.example) file is the source of truth for
configured values.
