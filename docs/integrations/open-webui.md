# Open WebUI

The included Open WebUI
[manifold Pipe](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/#creating-multiple-models-with-pipes)
is an optional adapter over the LGOS OpenAI-compatible API.

## Setup

Start Open WebUI as described in [Docker](../how-to-guides/docker.md#demo-compose).
Import `demo/ui/openwebui/openwebui_pipe.py` in
`Admin Panel -> Functions`, then enable it. Configure the LGOS base URL and API
key with the Function valves.

The Pipe fetches `/v1/models` and adds every registered graph to the model
selector. Open WebUI stores its own copy of imported Function code, so update or
re-import the Pipe after changing the local file.

## Runtime Settings

The Pipe does not retrieve detailed model descriptors and does not send
`metadata.langgraph_runtime_settings`. Graphs therefore use their registered
server defaults. The shipped Pipe has no remote per-model schema discovery
mechanism equivalent to the Chainlit adapter.

## Streaming And Citations

The Pipe streams assistant content unchanged, so Open WebUI renders Markdown
links and images normally. For streaming requests it also forwards final OpenAI
citation annotations without translating them. Non-streaming generator results
remain plain text.

See Open WebUI's
[Pipe streaming format](https://docs.openwebui.com/features/extensibility/pipelines/pipes/#streaming-response-format).

## Interrupt Approval

Select `interruptible-approval` to try confirmation. The Pipe sends
`metadata.langgraph_thread_id`, presents the interrupt, and returns the matching
tool result when the user approves or rejects it.

See the core [citation contract](../explanation/openai-compatibility.md#citation-ownership)
and [interrupt protocol](../explanation/openai-compatibility.md#tool-calls-and-interrupts)
for the API behavior beneath the adapter.
