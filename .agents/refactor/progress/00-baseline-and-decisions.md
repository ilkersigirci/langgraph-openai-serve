# 00 — Baseline And Decisions

Status: **Analysis complete**

## Scope Reviewed

The review covered the package under `src/langgraph_openai_serve`, package tests,
all four demo projects, public documentation, current code-style instructions,
and relevant locked dependency APIs. Repository history was inspected where a
complex mechanism had clearly evolved in response to a regression.

The review started from clean `main` commit
`e5dad7d17b547c1dc8de34cbbb00acab26f1bdd3`. Only this progress directory was
added. No implementation was attempted.

## Baseline Validation

| Command | Result |
| --- | --- |
| `just check` | Ruff formatting and lint pass; `ty check src` passes |
| `just test` | 359 passed |
| `cd demo && just test --editable` | API 111 passed, Files 12 passed, Chainlit 116 passed, Open WebUI 120 passed; integration-marked tests deselected |
| `cd demo && just lint` | All four demo projects pass |
| `cd demo && just type-check --editable` | All four demo projects pass |
| `just docs` | Strict Zensical build passes |

The Docker Compose check, PostgreSQL integration tests, and live provider or
gateway tests were not run. They require external services and are listed in the
work units that can affect them.

## Locked Runtime Versions

These versions, rather than current `main` documentation alone, define the APIs
available to the implementation work:

| Package | Version |
| --- | --- |
| `langgraph` | 1.2.9 |
| `langgraph-checkpoint` | 4.1.1 |
| `langchain-core` | 1.4.9 |
| `openai` | 2.45.0 |
| `fastapi` | 0.139.2 |
| `starlette` | 1.3.1 |
| `pydantic` | 2.13.4 |
| `anyio` | 4.14.2 |
| `psycopg` | 3.3.4 |
| `psycopg-pool` | 3.3.1 |
| `chainlit` (demo lock) | 2.12.0 |
| Open WebUI (demo image) | 0.11.3, digest-pinned |

Do not update these packages or regenerate lockfiles as part of a refactor unit
unless that unit first demonstrates that the required simplification is
unavailable in the locked version.

## Size And Concentration

Raw Python line counts are inventory only:

| Area | Production | Tests |
| --- | ---: | ---: |
| Package | 5,786 | 8,922 |
| Demo API source | 4,262 | 5,495 |
| Demo API notebooks | 612 | — |
| Demo Files service | 776 | 386 |
| Chainlit UI | 2,622 | 3,458 |
| Open WebUI UI | 2,630 | 2,594 |

The largest package modules are `api/responses/streaming.py` (710 lines),
`graph/runner.py` (357), `api/responses/service.py` (280),
`graph/graph_registry.py` (269), and `graph/interrupt/state.py` (266). The largest
demo modules include Chainlit `hitl.py`, Open WebUI `interrupts.py` and `pipe.py`,
and the advanced demo graph. These numbers identify review targets; they do not
prove that a split is useful.

An AST triage of production functions found the highest branch concentration in
Open WebUI `Pipe._run()`, attachment normalization, and Workspace Model sync;
demo LiteLLM reconciliation; and Chainlit's ordinary response loop. Those paths
are assigned to units 07–09. In the package, the corresponding concentrations
are prepared-run ownership, streaming runner dispatch, Responses tool-item
assembly, and interrupt token construction, assigned to units 02–05.

The longest low-branch functions are graph builders: the advanced graph,
advanced notebook subgraph, research subgraph, and server-tool graph. Their
length largely exposes workflow topology and dependency-closing node functions.
Unit 09 requires a clearer responsibility before extracting them; moving every
node to another file would lower line counts while making the workflow harder to
read.

The same scan found `Any` concentrated at user-defined graph schemas and
untyped Open WebUI host objects. Units 02 and 06 narrow the former where native
generics are available; unit 08 validates only the host fields the Function
uses. Unit 10 reviews the remaining occurrences individually instead of setting
an arbitrary count target.

## Evidence Behind The Priorities

### Nested request fields violate the documented contract

`ResponseCreateRequest` inherits `extra="forbid"`, but its union imports
`ResponseCustomToolCall`, `ResponseCustomToolCallOutput`,
`ResponseFunctionWebSearch`, and annotation models from the OpenAI SDK's output
types. Those generated models allow extra fields. Direct validation accepted an
unknown `extra_x` field in all four nested shapes. Chat's top-level request also
forbids extras, while its local nested message, tool, function, choice, and
stream-options models use Pydantic's default of ignoring them.

This contradicts `docs/reference.md` and
`docs/explanation/openai-compatibility.md`, which say unknown request fields are
not silently ignored. Unit 01 fixes it first because silently accepted input is
observable API behavior.

### LangGraph already returns typed interrupts

With `version="v2"`, locked LangGraph exposes:

- `GraphOutput.value` and `GraphOutput.interrupts` from `ainvoke`;
- a discriminated `StreamPart` union from `astream`;
- `ValuesStreamPart.data` and `ValuesStreamPart.interrupts` for root and nested
  values events.

The current runner casts those results to `GraphOutput[Any]` and
`dict[str, Any]`, then calls `aget_state()` after execution. Local probes found
that root values parts expose direct, nested, and parallel interrupt data. A
parallel nested pause can arrive as more than one root values part, so the
streaming path must accumulate and de-duplicate interrupts by ID rather than
assuming the final part contains one complete batch. Unit 02 records the exact
required proof.

LangGraph now recommends its v3 event-streaming API for application code, and
that API exposes message, output, and interrupt projections. In locked 1.2.9 it
is explicitly decorated and documented as experimental. The core refactor will
use stable v2 types first. A v3 migration is allowed only as the bounded spike in
unit 03, after it passes the project's custom-event, nested update, usage,
durability, and cancellation behavior matrix with less local code.

### Exact interrupt generation cannot use IDs alone

A probe against locked LangGraph exercised two sequential `interrupt()` calls
in one task and the same graph invoked indirectly inside a parent node. Between
the first and second pause:

- the raw interrupt ID was reused;
- the relevant checkpoint ID was reused;
- the pending `__resume__` write count changed.

The existing HTTP regression in `tests/api/interrupt/test_state.py` proves the
public consequence: an old Response must fail after the graph advances to the
second pause. Removing the checkpoint-history fingerprint would make those two
generations indistinguishable in a stateless, restart-safe server. Unit 04
contains this dependency rather than pretending it is optional.

### Cancellation ownership is a contract

`_StreamOwner` is not present merely to wrap an iterator. It was added after
native response consumption failed to reliably cancel in-flight graph/provider
work and finish async finalizers. `tests/api/test_chat_cancellation.py` exercises
a real TCP disconnect and immediate close. Starlette's response iterator owns
ASGI consumption, while LangGraph and providers use asyncio-native teardown.

This still deserves simplification because run ownership is spread across
`GraphRun`, `prepare_run`, `finalize_run`, and `_StreamOwner`. Unit 03 must prove
any replacement against the same real transport behavior. Passing an in-process
ASGI transport test is insufficient.

### Responses assembly is protocol state, not a serializer call

`ResponsesStreamBuilder` owns standard sequence numbers, output indexes, item
and content-part lifecycles, final reconciliation, refusals, tool calls,
interrupts, usage, and terminal statuses. The OpenAI SDK provides generated
event and output models and a client stream decoder; it does not publish a
server-side event lifecycle builder. The project is already using the native
models correctly.

Streaming and non-streaming intentionally share this accumulator so server-tool
updates and final output cannot drift. Unit 05 may clarify ownership and remove
coarse runner inputs, but it must not duplicate the protocol in two paths.

### Native message conversion does not cover current error semantics

Responses replay validation remains protocol-specific. LangChain's general
message conversion targets role/content chat messages and does not enforce
LGOS's item IDs, output ordering, or call/output causality.

Chat conversion is small enough to retain unless a native helper reaches
behavior parity. In locked LangChain Core, `convert_to_messages()` raises a raw
JSON decode error for malformed assistant tool arguments and a `KeyError` for a
tool message without `tool_call_id`; LGOS currently turns those cases into
typed invalid calls or controlled OpenAI request errors.

## Primary Sources

- [LangGraph v2 streaming format](https://docs.langchain.com/oss/python/langgraph/streaming)
- [LangGraph event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming)
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph 1.2.9 result and stream types](https://github.com/langchain-ai/langgraph/blob/1.2.9/libs/langgraph/langgraph/types.py)
- [LangGraph 1.2.9 execution API](https://github.com/langchain-ai/langgraph/blob/1.2.9/libs/langgraph/langgraph/pregel/main.py)
- [LangGraph 1.2.9 v3 run-stream source](https://github.com/langchain-ai/langgraph/blob/1.2.9/libs/langgraph/langgraph/stream/run_stream.py)
- [LangChain message conversion](https://docs.langchain.com/oss/python/langchain/messages)
- [LangChain Core 1.4.9 message conversion source](https://github.com/langchain-ai/langchain/blob/langchain-core%3D%3D1.4.9/libs/core/langchain_core/messages/utils.py)
- [OpenAI Python 2.45.0 Responses request parameters](https://github.com/openai/openai-python/blob/v2.45.0/src/openai/types/responses/response_create_params.py)
- [OpenAI Python 2.45.0 Responses stream-event union](https://github.com/openai/openai-python/blob/v2.45.0/src/openai/types/responses/response_stream_event.py)
- [OpenAI Python 2.45.0 Responses resource](https://github.com/openai/openai-python/blob/v2.45.0/src/openai/resources/responses/responses.py)
- [LangGraph 1.2.9 release](https://github.com/langchain-ai/langgraph/releases/tag/1.2.9)
- [Pydantic extra-data behavior](https://docs.pydantic.dev/latest/concepts/models/#extra-data)
- [Pydantic discriminated unions](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions)
- [Pydantic `TypeAdapter`](https://docs.pydantic.dev/latest/concepts/type_adapter/)
- [Starlette response streaming](https://www.starlette.io/responses/)
- [Starlette 1.3.1 `StreamingResponse` source](https://github.com/Kludex/starlette/blob/1.3.1/starlette/responses.py)
- [Psycopg pool lifecycle](https://www.psycopg.org/psycopg3/docs/advanced/pool.html)
- [PostgreSQL advisory locks](https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS)
- [Chainlit 2.12.0 release](https://github.com/Chainlit/chainlit/releases/tag/2.12.0)
- [Chainlit `on_chat_resume`](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-chat-resume)
- [Chainlit persisted message metadata](https://docs.chainlit.io/api-reference/message)
- [Open WebUI 0.11.3 release](https://github.com/open-webui/open-webui/releases/tag/v0.11.3)
- [Open WebUI Pipe Functions](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe/)
- [Open WebUI Function events](https://docs.openwebui.com/features/extensibility/plugin/development/events/)
- [Open WebUI 0.11.3 chat integration source](https://github.com/open-webui/open-webui/blob/v0.11.3/backend/open_webui/utils/chat.py)
- [Open WebUI 0.11.3 socket event source](https://github.com/open-webui/open-webui/blob/v0.11.3/backend/open_webui/socket/main.py)

## Decisions That Later Agents Should Not Reopen Casually

1. Keep exact stale-resume rejection and restart-safe stateless continuation.
2. Keep one response accumulator for streaming and non-streaming output.
3. Keep official SDK types at the wire boundary, while owning strict local
   request models for the supported subset.
4. Keep cross-process run coordination for durable checkpointers.
5. Keep client-specific Chainlit and Open WebUI persistence and rendering.
6. Do not introduce a general internal framework for two protocol adapters or
   one lifecycle use case.

Reopen one of these decisions only with a smaller working implementation and a
behavior test that explains what changed.
