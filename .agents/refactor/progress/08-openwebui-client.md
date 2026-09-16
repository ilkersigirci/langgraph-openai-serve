# 08 — Open WebUI Function Runtime

- Status: **Complete**
- Priority: **P2**
- Dependencies: **05, 06**

## Objective

Give the Open WebUI Function one validated path from its untrusted host
arguments to an OpenAI Responses request, while making the request,
continuation, and rendering stages of `Pipe._run()` visible without adding a
framework. Administrator synchronization is a separate control plane and is
not part of this runtime unit.

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
- `bundle.py`, bundle tests, UI tests, and README text affected by the change

The bundle constraints in `demo/AGENTS.md` are part of the deployed runtime:
imports must remain acyclic in `GENERIC_BUNDLE` order and top-level names must
remain unique across the flattened modules.

## Implementation Steps

1. Complete units 05 and 06 first and verify generated requests against their
   final documented wire behavior. Continue using official OpenAI SDK types;
   the independently deployed Function must not import LGOS server internals.
2. Define the smallest models or typed values for the external host shapes
   actually consumed by `Pipe.pipe()`: body fields, user identity, request
   metadata, attached files, tool servers, and the persisted interrupt cursor.
   Validate once near entry, then pass precise values through `_run()`. Open
   WebUI owns the host dictionaries and may add unrelated fields, so either
   project the consumed keys before strict validation or allow unknown host
   fields. Use `extra="forbid"` for LGOS-owned persisted cursor data. Do not
   model unused Open WebUI fields.
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

The attachment path in `files.py` may be cleaned only where Pipe entry
validation removes repeated checks. Preserve its three native host sources in
order: an available server path, image bytes already embedded in message
content, and an authenticated Open WebUI Files download. Upload only the current
turn's files to the OpenAI Files provider, and retain the authenticated
in-process ASGI path when a host request exposes its app. Small models for
consumed file metadata must tolerate additive host fields; do not model the
complete Open WebUI file object.

## Deferred Control Plane

`sync_functions.py`, `workspace_models.py`, and `tool_servers.py` call Open
WebUI administrator endpoints and do not participate in a chat request. They
already have focused reconciliation tests and several typed boundaries. Do not
change them in this unit. Open a separate work unit only for a concrete defect
or measured maintenance problem, preserving validation-before-deletion,
unrelated objects and access grants, deterministic IDs, hidden manifold bases,
and the special demo UserValves model. Host API response models must tolerate
additive fields even when LGOS-owned desired-state models are strict.

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
uv run --locked pytest tests/test_openwebui_responses.py tests/test_openwebui_settings.py tests/test_openwebui_upload_policy.py
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

Completed on 2026-09-16.

- `Pipe.pipe()` now validates one deliberately small projection of the Open
  WebUI body, user, metadata, files, and MCP tools that tolerates additive host
  fields. The runtime passes those typed values through request preparation
  instead of repeating loose dictionary checks. Invalid consumed host data
  returns the existing safe Function error shape in both response modes before
  an OpenAI client is called.
- `_run()` now exposes request preparation, one Responses turn owned by the SDK,
  typed terminal classification, and Open WebUI rendering as distinct stages.
  The request state retained across client-tool turns is explicit, while text,
  refusal, citation, completion status, and SDK output transformations remain
  in `responses.py`.
- The interrupt cursor is a strict LGOS-owned value containing the previous
  response ID and the complete standard `ResponseFunctionToolCall` batch. Its
  pure codec compresses that JSON before URL-safe encoding, rejects unknown
  cursor or call fields, duplicate call IDs, and compressed payload
  amplification. It restores exactly one output for every call in a parallel
  answer batch. Mixed interrupt and ordinary client-tool batches still fail
  closed.
- Open WebUI 0.11.3's stream-only native MCP loop remains enforced. Interrupt
  continuation sends only answer outputs with `previous_response_id`, while
  ordinary turns and any following display-file continuation use the UI
  transcript without replaying the paused transcript twice.
- Attachment handling retains the three host sources in order: an available
  server path, current-message image bytes, and an authenticated Open WebUI
  file download. Embedded image positions stay aligned when an earlier image
  uses its server path. Only the current turn's files are uploaded. The live
  native upload returned unique file content; focused behavior tests cover the
  authenticated download fallback.
- The Generic bundle order and module set did not change. A focused bundle test
  now rejects duplicate top-level definitions across flattened modules in
  addition to compiling and executing the deployed source. Administrator sync,
  Workspace Model, and tool-server control-plane code remained unchanged.
  Existing product documentation already describes the retained public
  behavior, so no product-documentation change was required.

Locked-version verification used the Open WebUI project's `uv.lock`,
`uv tree --locked`, the digest-pinned Open WebUI image declaration, and pinned
upstream source. The relevant resolved versions were Open WebUI 0.11.3, OpenAI
Python 2.46.0, Pydantic 2.13.4, HTTPX 0.28.1, and AnyIO 4.14.2. The
implementation was checked against these primary references:

- [Open WebUI Pipe Functions](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/)
- [Open WebUI Function events](https://docs.openwebui.com/features/extensibility/plugin/development/events/)
- [Open WebUI 0.11.3 Function host source](https://github.com/open-webui/open-webui/blob/v0.11.3/backend/open_webui/functions.py)
- [Open WebUI 0.11.3 chat integration source](https://github.com/open-webui/open-webui/blob/v0.11.3/backend/open_webui/utils/chat.py)
- [Open WebUI 0.11.3 chat middleware source](https://github.com/open-webui/open-webui/blob/v0.11.3/backend/open_webui/utils/middleware.py)
- [Open WebUI 0.11.3 socket-event source](https://github.com/open-webui/open-webui/blob/v0.11.3/backend/open_webui/socket/main.py)
- [OpenAI Python 2.46.0 Responses resource](https://github.com/openai/openai-python/blob/v2.46.0/src/openai/resources/responses/responses.py)
- [OpenAI Python 2.46.0 function-call type](https://github.com/openai/openai-python/blob/v2.46.0/src/openai/types/responses/response_function_tool_call.py)

| Validation | Result |
| --- | --- |
| Required focused Open WebUI pytest command | 97 passed |
| Bundle compilation and duplicate-definition tests | 13 passed |
| Open WebUI Ruff check and format check | Pass |
| `uv run --locked ty check src` | Pass |
| `cd demo && just test --editable` | API 111 passed, Files 12 passed, Chainlit 126 passed, Open WebUI 139 passed |
| Live Open WebUI browser check | Streaming Plotly output rendered in a 450 px interactive frame; a native file upload returned its unique content; `lgos-gateway` executed the read-only MCP report; and an approval interrupt resumed to terminal refund and notification output |

The live check used an isolated browser session against the digest-pinned demo
stack. Temporary authentication state and the uploaded text fixture were
removed after the session; the chart assertion screenshot is
`/tmp/lgos-openwebui-unit08-plot.png`.
