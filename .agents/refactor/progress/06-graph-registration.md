# 06 — Graph Registration

- Status: **Complete**
- Priority: **P1**
- Dependencies: **02, 03, 04**

## Objective

Make graph registration a small immutable configuration boundary, perform static
checks once, and leave per-request graph factories dynamic.

## Problem

`graph/graph_registry.py` currently combines several concerns:

- Pydantic field and assignment validation for code-owned configuration;
- graph factory resolution;
- static feature relationship checks;
- resolved graph context/checkpointer capability checks;
- request input and context adapters;
- output rendering;
- registry key validation, mapping freezing, and serialization.

`GraphConfig.resolve_graph()` repeats static checks on every request. The
`GraphRegistry.registry` field uses an annotated `Mapping`, an after-validator,
a `MappingProxyType`, a plain serializer, and assignment validation to implement
a small read-only map with one explicit `register()` mutation method. Production
code does not mutate `GraphConfig`; several tests mutate it to arrange cases.

## Files In Scope

- `src/langgraph_openai_serve/graph/graph_registry.py`
- public exports and type aliases affected by the final shape
- model discovery and server construction callers
- package and demo graph registrations
- registration, model, runner, and client-settings tests
- public reference, getting-started, and custom-graph documentation

Do not change OpenAI model IDs, advertised metadata, graph adapter semantics, or
factory lifetime.

## Implementation Steps

1. Keep `GraphConfig` as a declarative value. Prefer a frozen Pydantic model with
   immutable tuple/frozenset fields for collections LGOS owns if that remains
   the smallest design; do not pretend freezing the outer model makes a caller-
   supplied callback manager immutable. Use a slotted dataclass only if explicit
   validation is shorter and equally clear. Do not maintain both
   representations.
2. Validate static relationships at construction or registration:
   `run_coordinator` is required for and permitted only on interrupt-enabled
   graphs, client settings preserve their contract, descriptions are non-empty,
   and names/tool sets are valid.
3. Keep graph-dependent checks after each factory resolution: compiled graph
   type, context schema relationship, checkpointer capabilities, and any runtime
   resource requirement. Extract one plainly named validation function rather
   than a `ResolvedGraph` wrapper used once.
4. Do not cache factory results. Sync and async zero-argument factories may
   intentionally create request-scoped graphs or bind current resources. A
   directly registered compiled graph can avoid repeating checks only if that
   optimization stays obvious.
5. Replace the registry's Pydantic serialization machinery with a plain mapping
   owner only if it reduces the total design. A straightforward option is one
   private dictionary, one read-only mapping view, explicit model-ID validation
   through a Pydantic `TypeAdapter`, `register()`, and lookup methods. Keeping
   the current model is a valid outcome if the replacement merely moves the
   same validation into handwritten code.
6. Preserve the rule that callers cannot mutate the registry through its public
   mapping. `register()` should validate before changing state and should not
   rebuild the entire mapping merely to trigger assignment validation.
   Preserve the documented `GraphRegistry(registry=...)` construction and
   read-only `.registry` view unless a public API change has a concrete benefit;
   an internal simplification alone does not require demo-wide call-site churn.
7. Update tests that mutate `GraphConfig` after construction. Construct the
   intended config directly or replace it through `register()`; this better
   reflects production use and allows immutable configuration.
8. Keep input, context, and output adapter callables on `GraphConfig`. Splitting
   each callback into a strategy class would add a framework without reducing
   the public concepts.

## ClientSettings Decision

`graph/client_settings.py` is reviewed and retained. Its cached JSON copies,
strict/frozen inherited configuration checks, validation of defaults and schema,
single execution of default factories, and rejection of non-finite nested
values each have focused behavior tests. Do not fold its models into
`GraphConfig` or relax those guarantees as part of registration cleanup.

## Checkpointer Capability Decision

Re-evaluate the five required asynchronous saver methods after units 02–04.
Keep every method used by LangGraph execution or LGOS continuation/deletion.
Replace the current “overrides base stub” introspection only if a simpler
capability check still fails at registration with an actionable error. Do not
defer a known invalid saver until the first user request executes halfway.

## Required Behavior

- The registry contains at least one graph and rejects empty, slash-containing,
  `.` and `..` model IDs.
- Public registry entries cannot be mutated except through `register()`.
- Model listing and lookup preserve insertion order and errors.
- Static invalid configurations fail once, close to construction.
- Factory-specific invalid graphs fail during resolution before execution.
- Sync factories, async factories, and direct compiled graphs retain their
  current lifetimes.
- All callers use the final clean API directly; no compatibility aliases or
  deprecation layer remains. If the public constructor stays unchanged, do not
  manufacture a demo migration for this requirement.

## Validation

```bash
just check
just test tests/graph tests/api/test_models.py tests/api/test_chat_completions.py tests/api/test_chat_messages.py tests/api/responses
just test
cd demo
just test --editable
just lint
just type-check --editable
cd ..
just docs
```

## Outcome

Completed on 2026-09-15.

- `GraphConfig` remains the single declarative Pydantic representation and is
  now frozen. LGOS-owned node names, features, and server-tool names are stored
  as tuples or frozen sets, while caller-owned callback objects retain their
  documented shallow mutability boundary.
- Non-empty descriptions, client settings, tool names, and the exact
  interrupts/run-coordinator relationship validate when a config is built.
  Resolved compiled-graph type, direct client-settings context schema, and
  interrupt checkpointer capabilities now live in one plainly named validation
  function that runs before each resolved graph executes.
- Direct compiled graphs remain reusable, while sync and async zero-argument
  factories are invoked and validated on every resolution. No factory result
  cache or alternate resolved-graph wrapper was introduced.
- `GraphRegistry` is now a small slotted mapping owner rather than a Pydantic
  serialization model. It copies the initial mapping, validates model IDs with
  one reusable `TypeAdapter`, exposes one insertion-ordered read-only live view,
  and validates `register()` inputs before changing its private dictionary.
  Replacements preserve their existing order without rebuilding the mapping.
- Package and demo tests that formerly mutated `GraphConfig` now construct a
  complete validated replacement and install it through `register()`. Responses
  request helpers accept the resulting abstract read-only server-tool set.
- `ClientSettings` and the five-method asynchronous saver capability check were
  retained. The latter still requires `aget_tuple()`, `alist()`, `aput()`,
  `aput_writes()`, and `adelete_thread()`; units 02–04 and the locked interface
  confirm that execution, continuation identity, and terminal cleanup need the
  full surface.
- Model IDs, model metadata, graph adapters, OpenAI wire behavior, and public
  `GraphRegistry(registry=...)` construction remain unchanged. Reference,
  getting-started, and custom-graph documentation now describe immutability,
  registration replacement, and graph-factory lifetime explicitly.

Locked-version verification used `uv.lock`, `uv tree --locked`, installed
package introspection, and official tagged primary sources. The resolved
versions were LangGraph 1.2.9, langgraph-checkpoint 4.1.1, and Pydantic 2.13.4:

- [Pydantic 2.13.4 faux immutability](https://github.com/pydantic/pydantic/blob/v2.13.4/docs/concepts/models.md#faux-immutability)
- [Pydantic 2.13.4 TypeAdapter](https://github.com/pydantic/pydantic/blob/v2.13.4/docs/concepts/type_adapter.md)
- [LangGraph 1.2.9 checkpoint interface](https://github.com/langchain-ai/langgraph/blob/1.2.9/libs/checkpoint/README.md#interface)
- [LangGraph 1.2.9 checkpoint base](https://github.com/langchain-ai/langgraph/blob/1.2.9/libs/checkpoint/langgraph/checkpoint/base/__init__.py)

| Validation | Result |
| --- | --- |
| `just check` | Pass |
| `just test tests/graph tests/api/test_models.py tests/api/test_chat_completions.py tests/api/test_chat_messages.py tests/api/responses` | 231 passed |
| `just test` | 405 passed |
| `cd demo && just test --editable` | API 111 passed, Files 12 passed, Chainlit 116 passed, Open WebUI 120 passed |
| `cd demo && just lint` | Pass |
| `cd demo && just type-check --editable` | Pass |
| `just docs` | Strict build passed |
