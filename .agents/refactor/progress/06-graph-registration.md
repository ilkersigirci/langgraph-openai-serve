# 06 — Graph Registration

- Status: **Waiting for core execution work**
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

Not started.
