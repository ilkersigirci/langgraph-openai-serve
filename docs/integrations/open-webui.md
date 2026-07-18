# Open WebUI

The demo includes two optional Open WebUI Functions over the LGOS
OpenAI-compatible API:

- `functions/generic.py` is a
  [manifold Pipe](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/#creating-multiple-models-with-pipes)
  for all registered graphs. It handles streaming, citations, and interrupt
  approval without graph-specific settings.
- `functions/uservalves_simple.py` is a small single-model Pipe for demonstrating
  `simple-graph` runtime settings through `UserValves`.

## Setup

Start Open WebUI as described in [Docker](../how-to-guides/docker.md#demo-compose),
then synchronize both bundled Functions:

```bash
make sync-demo-openwebui-functions
```

The command signs in through `/api/v1/auths/signin` with the Compose demo admin
credentials and creates or updates only the two repository-managed Functions.
New Functions are enabled automatically. Updates preserve their existing active
state, valves, and user valves. Run the command again after changing a file in
`demo/ui/openwebui/functions/`; unchanged Functions are skipped.

The command discovers every top-level `.py` file in that directory except files
whose names start with `_`. The filename stem is the Function ID, and the
required Open WebUI frontmatter `title` is its display name. Function
filenames must be lowercase Python identifiers.

The defaults match `docker-compose.yml`. Override them for another local Open
WebUI instance with `OPEN_WEBUI_URL`, `WEBUI_ADMIN_EMAIL`, and
`WEBUI_ADMIN_PASSWORD`. Set the password in the environment rather than passing
it on the command line.

If both Functions are enabled, `simple-graph` appears through both; only the
dedicated Function has runtime-setting controls. Configure the LGOS base URL
and API key independently in each Function's admin valves.

Open WebUI labels the two entries as
`Generic / simple-graph` and
`UserValves-Simple / simple-graph`. These prefixes are display names only. The
manifold Pipe removes Open WebUI's Function-ID qualification, and both adapters
send `simple-graph` as the LGOS model.

The manifold Pipe fetches `/v1/models` and adds every registered graph to the
model selector. The simple Pipe appears as one model and always calls
`simple-graph`. Open WebUI stores Function code in its database, so a bind mount
of these Python files does not update the Functions.

## Runtime Settings

The single-model `uservalves_simple.py` Function declares Open WebUI
[`UserValves`](https://docs.openwebui.com/features/extensibility/plugin/development/valves/)
for `simple-graph`'s `use_history` and `audience` settings. Open WebUI renders
the boolean as a switch and the audience choices as a selector in the chat UI.

The Pipe removes values equal to its local defaults and sends the remaining
values as `metadata.langgraph_runtime_settings`. Its Pydantic `UserValves`
model deliberately matches the `SimpleContext` model in the demo API.

Selecting `simple-graph` through the general manifold Pipe does not expose or
send runtime settings; use the dedicated simple Pipe for that demo.

!!! note

    Open WebUI stores `UserValves` per user and Function, not per conversation.
    Add another dedicated single-model Pipe when another graph needs a different
    settings UI.

## Streaming And Citations

The general manifold Pipe streams assistant content unchanged, so Open WebUI
renders Markdown links and images normally. For streaming requests it also
forwards final OpenAI citation annotations without translating them.
Non-streaming generator results remain plain text. The simple Pipe streams only
assistant text. Neither bundled Pipe opts into LGOS client stream events;
support requires an explicit mapping from LGOS `status`, `progress`, and
`artifact` data to Open WebUI's UI-specific event shapes.

## Interrupt Approval

Select `interruptible-approval` from the manifold Pipe to try confirmation. The
Pipe sends `metadata.langgraph_thread_id`, presents the interrupt, and returns
the matching tool result when the user approves or rejects it.

See the core [citation contract](../explanation/openai-compatibility.md#citation-ownership)
and [interrupt protocol](../explanation/openai-compatibility.md#tool-calls-and-interrupts)
for the API behavior beneath the adapter.
