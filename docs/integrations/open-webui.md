# Open WebUI

The demo includes two optional Open WebUI Functions over the LGOS
OpenAI-compatible API:

- `openwebui_pipe.py` is a
  [manifold Pipe](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/#creating-multiple-models-with-pipes)
  for all registered graphs. It handles streaming, citations, and interrupt
  approval without graph-specific settings.
- `simple_graph_pipe.py` is a small single-model Pipe for demonstrating
  `simple-graph` runtime settings through `UserValves`.

## Setup

Start Open WebUI as described in [Docker](../how-to-guides/docker.md#demo-compose).
Import either file from `demo/ui/openwebui/` in `Admin Panel -> Functions`, then
enable it. If both are enabled, `simple-graph` appears through both Functions;
only the dedicated Function has runtime-setting controls. Configure the LGOS
base URL and API key independently in each Function's admin valves.

The manifold Pipe fetches `/v1/models` and adds every registered graph to the
model selector. The simple Pipe appears as one model and always calls
`simple-graph`. Open WebUI stores its own copy of imported Function code, so
update or re-import a Function after changing its local file.

## Runtime Settings

The single-model `simple_graph_pipe.py` Function declares Open WebUI
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
