# 08 — Open WebUI Client

- Status: **Waiting for core work**
- Priority: **P2**
- Dependencies: **01, 05, 06**

## Objective

Give the Open WebUI Function one typed path from its untrusted host arguments to
an OpenAI Responses request, while making the request, continuation, and
rendering stages of `Pipe._run()` visible without adding a framework.

## Assessment

The Generic Function is intentionally deployed as one flattened Python
namespace, but its source is already divided into responsibility modules.
`Pipe._run()` remains the main concentration: it validates loose Open WebUI
objects, resolves model metadata and gateway settings, prepares files and MCP
tools, chooses a fresh or resumed request, consumes streaming or non-streaming
Responses output, and emits host events.

`interrupts.py` is also large, but much of it translates between two real
protocols. Open WebUI's `ask_user` result persists a call ID and answers, while
LGOS requires a complete standard Responses function-call batch and
`previous_response_id`. The encoded cursor carries that missing continuation
state. Chainlit persists the same concept through message metadata, so these
clients should not share a storage abstraction.

The deployed image is digest-pinned to Open WebUI 0.11.3. Its source and the
official Function documentation confirm the `Pipe` entry point and the
`__event_emitter__`/`__event_call__` host boundary. Recheck the pinned source,
not only current docs, before changing injected arguments or event payloads.

## Files In Scope

- `demo/ui/openwebui/src/lgos_openwebui/functions/generic/contracts.py`
- `demo/ui/openwebui/src/lgos_openwebui/functions/generic/pipe.py`
- `demo/ui/openwebui/src/lgos_openwebui/functions/generic/responses.py`
- `demo/ui/openwebui/src/lgos_openwebui/functions/generic/interrupts.py`
- nearby gateway, file, metadata, and API helpers only when responsibility
  clearly belongs there
- `demo/ui/openwebui/src/lgos_openwebui/workspace_models.py`
- `demo/ui/openwebui/src/lgos_openwebui/sync_functions.py`
- `demo/ui/openwebui/src/lgos_openwebui/tool_servers.py`
- `bundle.py`, bundle tests, UI tests, and README text affected by the change

The bundle constraints in `demo/AGENTS.md` are part of the deployed runtime:
imports must remain acyclic in `GENERIC_BUNDLE` order and top-level names must
remain unique across the flattened modules.

## 08A — Function Runtime

1. Complete units 01 and 05 first and migrate directly to their final request
   and output boundaries.
2. Define the smallest strict models or typed values for the external host
   shapes actually consumed by `Pipe.pipe()`: body fields, user identity,
   request metadata, attached files, tool servers, and the persisted interrupt
   cursor. Validate once near entry, then pass precise values through `_run()`.
   Do not model unused Open WebUI fields.
3. Preserve the public `Pipe` methods and arguments expected by Open WebUI.
   Move pure normalization behind that boundary instead of spreading `Any`
   checks across the orchestration path.
4. Reshape `_run()` around four plain stages already represented by sibling
   modules: prepare the upstream request, execute one Responses turn, classify
   terminal text/tool/interrupt output, and emit Open WebUI events. Use early
   returns for host errors and completed outcomes.
5. Keep streaming and non-streaming transformations in `responses.py`. Reuse
   official SDK response and stream-event types; do not parse SSE or rebuild
   LGOS response events in the client.
6. Make `InterruptCursor` strict and keep cursor encode/decode as pure
   functions. The encoded value must include the prior response ID and complete
   function-call batch because the host does not persist those fields
   separately. Preserve mixed-batch rejection and exact call-ID matching.
7. Keep Open WebUI-specific `ask_user` questions, review details, and resume
   answer conversion in `interrupts.py`. Split pure payload/question formatting
   from event-emitter calls only where tests become simpler.
8. Keep the existing supported-version rule for MCP: the Function sends Open
   WebUI MCP server declarations only on the compatible streaming path. Do not
   invent a non-streaming representation or silently drop a requested server.
9. Preserve request transcript replay for display while sending
   `previous_response_id` only for interrupt continuation. A general chat turn
   must remain stateless at LGOS's Responses boundary.
10. Update `GENERIC_BUNDLE` only if a new responsibility module is demonstrably
    needed. Run the bundle compilation and duplicate-name tests after every
    source move.

The attachment path in `files.py` is a boundary worth cleaning with the Pipe.
Preserve its three native host sources in order: an available server path,
image bytes already embedded in message content, and an authenticated Open
WebUI Files download. Upload only the current turn's files to the OpenAI Files
provider, and retain the authenticated in-process ASGI path when a host request
exposes its app. Strict small models for consumed file metadata may replace
repeated dictionary checks; do not model the complete Open WebUI file object.

## 08B — Synchronization Control Plane

Treat Function, MCP server, and Workspace Model synchronization as a second
reviewable change. These modules call Open WebUI administrator endpoints and do
not run in the chat request path.

1. Validate the few fields read from Function exports, model exports, base
   models, and sign-in responses through small strict boundary models or one
   plainly named parser per payload. Do not pass raw `Any` beyond the response
   boundary.
2. Separate pure desired/existing reconciliation from HTTP writes where this
   makes create, update, unchanged, and stale-delete decisions directly
   testable. Preserve the current order: validate and import desired objects
   before deleting stale generated objects.
3. Preserve unrelated Functions and Models, existing access grants, hidden
   manifold bases, deterministic generated IDs, and the special demo
   UserValves model. These are ownership rules, not incidental dictionary
   transformations.
4. Keep Function bundling as an AST-based build step. Open WebUI stores one
   source string, so flattening the modular source and compiling it before
   upload is required. Do not check in a second generated `generic.py`.
5. Compare every administrator endpoint and payload with the pinned 0.11.3
   source before changing it; these endpoints do not have an official Python
   SDK.

## Required Behavior

- Invalid Open WebUI host data produces the current safe Function error shape
  without an upstream request.
- Streaming and non-streaming requests preserve text, usage/status reporting,
  citations, files, client function tools, server tools, refusals, incomplete
  results, and upstream error messages.
- MCP tool servers are forwarded only in their documented supported mode.
- A parallel interrupt batch produces one `ask_user` flow and resumes with one
  output for every call.
- A mixed interrupt and ordinary client-tool batch remains rejected because the
  host cannot safely execute both continuation modes together.
- Replayed UI transcript does not become duplicate model input.
- The bundled Function compiles and has the same behavior as the modular source.

## Validation

```bash
cd demo/ui/openwebui
uv run --locked pytest tests/test_openwebui_responses.py tests/test_openwebui_tool_servers.py tests/test_openwebui_upload_policy.py
uv run --locked pytest tests/test_openwebui_sync_functions.py tests/test_openwebui_workspace_models.py
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
uv run --locked ty check src
cd ../../..
cd demo && just test --editable
```

Use the browser-check skill from `demo/AGENTS.md` for a final live streaming,
file, MCP, and interrupt-resume pass when the stack is available.

Primary host references are the [Pipe Function guide](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/),
[Function event guide](https://docs.openwebui.com/features/extensibility/plugin/development/events/),
and pinned 0.11.3 [chat](https://github.com/open-webui/open-webui/blob/v0.11.3/backend/open_webui/utils/chat.py)
and [socket-event](https://github.com/open-webui/open-webui/blob/v0.11.3/backend/open_webui/socket/main.py)
sources.

## Stop Conditions

- If a local host model mostly reproduces Open WebUI objects that the Function
  never reads, reduce it to the fields needed by the current behavior.
- If a proposed helper would be called only once and merely renames a few
  statements, leave that stage inline.
- Do not share continuation storage or rendering code with Chainlit. Share only
  existing protocol constants or wire declarations whose behavior is truly
  identical.

## Outcome

Not started.
