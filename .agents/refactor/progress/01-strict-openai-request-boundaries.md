# 01 — Strict OpenAI Request Boundaries

- Status: **Complete**
- Priority: **P0**
- Dependencies: **None**

## Objective

Make every model in the supported Responses and Chat Completions request trees
reject unknown fields, while preserving all documented valid SDK replay shapes
and the standard OpenAI error envelope.

## Problem

The top-level models are strict, but strictness stops at several nested values:

- `api/responses/schemas.py` imports SDK response models for custom calls,
  custom outputs, web-search calls, and output-text annotations. These generated
  output models use `extra="allow"`.
- `api/chat/schemas.py` gives `extra="forbid"` only to
  `ChatCompletionRequest`. Its nested messages, function calls, tool
  definitions, named choices, and stream options silently ignore extra fields.
- `ChatCompletionContentPartParam` is a broad SDK parameter union. LGOS needs a
  strict model of the subset it actually accepts, including the documented
  native Chat file part.

A direct probe accepted `extra_x` in four Responses nested item families and
five Chat nested families. This contradicts the public statement that unknown
request fields are not silently ignored.

## Files In Scope

- `src/langgraph_openai_serve/api/responses/schemas.py`
- `src/langgraph_openai_serve/api/responses/messages.py`
- `src/langgraph_openai_serve/api/chat/schemas.py`
- `src/langgraph_openai_serve/api/chat/messages.py`
- request-validation tests under `tests/api/`
- `docs/reference.md` and
  `docs/explanation/openai-compatibility.md` only if the supported field list
  changes after inspecting the locked SDK

Do not change response output models, event models, graph execution, or tool
semantics in this unit.

## Implementation Steps

1. Inventory every currently documented and tested nested request shape. Compare
   it with the locked OpenAI SDK parameter definitions. Treat generated response
   classes as output references, not request validators.
2. Define local Pydantic models for the supported nested Responses input items:
   custom call, custom output, web-search call and action, and URL-citation
   annotation. Inherit one strict request base and use discriminated unions where
   `type` already provides a discriminator.
3. Preserve SDK fields that are deliberately accepted for lossless replay. In
   particular, retain the documented nullable `parsed`, `caller`, and
   `namespace` fields and the accepted completed/incomplete statuses. Do not add
   fields merely because the latest upstream API has them.
4. Apply the same strict base to every local Chat nested request model. Replace
   the broad content-part alias with explicit models for the current supported
   content parts. Preserve text and native file-ID input behavior.
5. Keep protocol-specific semantic validation in the message adapters. Schema
   strictness should reject shape errors; replay ordering, duplicate IDs,
   call/output matching, role requirements, and malformed function arguments
   remain adapter concerns.
6. Add HTTP-level regression cases for an unknown field at each distinct nested
   boundary. Parametrize equivalent model families. Assert the status and
   OpenAI error `type`, `param`, and `code` fields through the official client or
   raw HTTP response, without asserting Pydantic's full prose.
7. Add positive replay cases built from locked SDK output objects. This protects
   the stated ability to append complete `response.output` items unchanged.

Use Pydantic's documented `ConfigDict(extra="forbid")` inheritance and tagged
unions. Do not write a recursive unknown-key walker; Pydantic already owns that
work.

## Required Behavior

- Unknown top-level and nested request fields fail before graph execution.
- Valid custom-call, custom-output, web-search, URL-citation, function-call, and
  assistant-message replay still reaches the graph unchanged.
- Chat tool calls, tool outputs, stream options, and native file parts keep
  their current valid behavior.
- Request failures use the existing OpenAI-compatible error handler.
- The OpenAPI schema describes the same strict subset accepted at runtime.

## Validation

```bash
just check
just test tests/api/test_errors.py tests/api/test_chat_messages.py tests/api/test_chat_completions.py tests/api/responses
just test
```

## Stop Conditions

- If an SDK-produced replay item contains a non-null field outside the documented
  subset, decide explicitly whether LGOS supports that meaning. Do not accept it
  only to make serialization convenient.
- If a Chat content part is currently accepted but undocumented, preserve the
  OpenAI contract only after confirming it has end-to-end graph semantics;
  otherwise reject it and update the docs in this same unit.

## Outcome

Completed on 2026-09-15.

### Final Design

- Responses and Chat Completions now own strict local Pydantic request models
  with inherited `ConfigDict(extra="forbid")`. Native tagged unions select
  content parts, tools, replay content, and Responses input items. The callable
  Responses discriminator exists only because easy input messages and replayed
  output messages both use `type="message"`.
- SDK output classes are no longer used as Responses request validators.
  Locally owned custom-call, custom-output, query-based web-search, and URL
  citation models expose only the documented LGOS subset.
- Locked OpenAI 2.45.0 replay artifacts remain valid: nullable `parsed`,
  `caller`, `namespace`, `created_by`, search `queries`/`sources`, and null or
  empty `logprobs` serialize successfully; supported item statuses remain
  accepted. Non-null parsed output, callers, namespaces, multiple search
  queries/sources, and non-empty logprobs remain outside the subset and fail
  validation.
- Chat content is explicitly limited to strings, strict text parts, and strict
  native `file.file_id` parts. The adapter serializes those local models back to
  the existing LangChain content shape. Locked SDK assistant-message null fields
  still replay unchanged; undocumented image, audio, inline-file, and
  prompt-cache shapes are rejected.
- Replay ordering, duplicate IDs, call/output matching, role requirements, and
  malformed function arguments remain in the existing protocol message
  adapters. No recursive unknown-field validator or compatibility shim was
  added.
- OpenAPI now exposes `additionalProperties: false` for the affected nested
  request objects and advertises only Chat text and file content-part tags.

The implementation was checked against `uv.lock`, the installed locked sources
for OpenAI 2.45.0 and Pydantic 2.13.4, the OpenAI 2.45.0 tagged request/output
models, and Pydantic's official extra-field and discriminated-union guidance.

### Changed Files

- `src/langgraph_openai_serve/api/responses/schemas.py`
- `src/langgraph_openai_serve/api/chat/schemas.py`
- `src/langgraph_openai_serve/api/chat/messages.py`
- `tests/api/test_errors.py`
- `tests/api/test_chat_completions.py`
- `tests/api/responses/test_non_streaming.py`
- `tests/api/test_openai_server.py`
- `docs/reference.md`
- `docs/explanation/openai-compatibility.md`
- `.agents/refactor/progress/README.md`
- `.agents/refactor/progress/01-strict-openai-request-boundaries.md`

### Validation

| Command | Result |
| --- | --- |
| `just check` | Pass; Ruff format/lint and `ty check src` |
| `just test tests/api/test_errors.py tests/api/test_chat_messages.py tests/api/test_chat_completions.py tests/api/responses` | 169 passed |
| `just test` | 384 passed |
| `just docs` | Strict Zensical build passed with no issues |

No required checks were skipped. Demo, Docker Compose, live gateway/provider,
and external PostgreSQL service suites were not run because this unit changes
only package request validation and its documented wire subset; no dependency or
lockfile was changed.
