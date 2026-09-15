# 09 — Demo Registration Migration And Audit

- Status: **Waiting for graph registration**
- Priority: **P2**
- Dependencies: **06**

## Objective

Migrate demo registrations only if unit 06 changes the public construction
shape, then validate the demos as consumers of the final package. Do not turn a
registration migration into an open-ended rewrite of graphs or standalone
services.

## Assessment

The demo code is broad because it is the executable feature catalog. Most of
that breadth is intentional:

- the advanced graph demonstrates routing, web research, file-backed knowledge,
  a durable notebook flow, interrupts, client tools, and provider outcomes;
- the persistent plot agent demonstrates native `create_agent`, LangGraph
  Store, OpenAI Files, client settings, and a client-owned display tool;
- smaller graphs isolate custom events, citations, MCP, subgraphs, file input,
  server tools, and custom adapters;
- the Files service is already a small FastAPI boundary over a typed repository
  protocol and an S3 implementation;
- the PostgreSQL runtime already owns saver, Store, pool, and session advisory
  lock lifecycles with the package-native APIs.

Long files here do not justify merging examples or inventing shared base
classes. Unit 06 should preserve `GraphConfig` and `GraphRegistry` construction
when possible; if it does, this unit is an integration audit with a legitimate
no-change outcome.

## Registration Follow-Through

Inventory all `GraphConfig` and `GraphRegistry` construction in the demo API and
notebooks after unit 06. If the public form is unchanged, make no mechanical
edits and record that the demos already use the final API. If a justified public
change remains, migrate all callers in one mechanical change. Keep graph IDs,
descriptions, feature declarations, adapters, client settings, and runtime
factory lifetimes unchanged. Update matching docs and examples without adding
aliases for the old form.

Validate the migration or no-change decision before considering any separate
demo cleanup so registry regressions remain attributable.

## Reviewed And Not Scheduled

The review found no concrete graph or service defect that belongs in this
refactor. Retain the following designs unless the registration migration reveals
a specific problem; open a new bounded work unit rather than appending the work
here:

- `advanced_graph/graph.py` keeps its native `StateGraph` topology and node
  closures over the model, knowledge service, Files client, checkpointer, and
  Store. Moving them would require a larger dependency object and hide the
  workflow.
- The durable notebook receipt keeps upload/index stages idempotent across
  retries. Preserve its failure and restart behavior.
- `persistent_plot_agent.py` keeps native `create_agent`, typed tools, Store,
  OpenAI Files, settings, and output adaptation together. Do not wrap it in a
  local agent framework.
- Smaller graph modules remain separate because they demonstrate different
  public features. Similar syntax alone is not a shared contract.
- MCP filtering and the external-tool loop remain explicit because their trust
  and tool-choice rules differ from ordinary model calls.
- `sync_litellm.py` already validates the relevant catalog and management
  payloads. Preserve conflict-before-write, ownership, deterministic IDs, and
  credential-safe errors.
- The Files service keeps its repository protocol, S3 paginator and streaming
  body, cursor behavior, metadata encoding, and guaranteed body close.
- `postgres_runtime()` and `PostgresRunCoordinator` keep saver, Store, pool, and
  session-lock ownership. Session advisory locks must remain on one checked-out
  connection, and an indeterminate session must not return to the pool.

When parsing provider, gateway, or host-owned JSON in a future unit, validate
the fields LGOS consumes but tolerate additive external fields unless that API
declares a closed schema. Strict unknown-field rejection is reserved for values
LGOS owns.

## Files In Scope

- `demo/api/src/lgos_demo_api/graphs/` and `app.py` only where registration is
  constructed
- `demo/api/notebooks/` only where registration is constructed
- matching demo tests and pages under `docs/demo/`

Any graph-internal, synchronization, Files, or PostgreSQL change requires a new
work unit with its own finding and acceptance criteria.

## Required Behavior

- Every model keeps the same public ID, advertised capabilities, description,
  settings schema, and `/v1` behavior unless a separately documented contract
  fix requires a change.
- Lifespan-managed graphs still resolve only after their dependencies exist.
- Demo tests confirm that advanced graph behavior, persistent chart state,
  LiteLLM synchronization, Files behavior, and PostgreSQL wiring were not
  accidentally affected by the package migration.

## Validation

```bash
cd demo
just test --editable
just lint
just type-check --editable
just compose-config
cd ..
just docs
```

Registration migration should not change durable graph behavior. If it does,
stop and create a bounded unit before additionally running the PostgreSQL suite:

```bash
cd demo
just test-postgres --editable --uri "$DEMO_API_TEST_POSTGRES_URI"
```

Run live direct or gateway suites only for an affected integration and use the
dedicated `just test-direct`, `just test-litellm`, or `just test-bifrost`
recipe. Follow the demo graph documentation skill before substantially revising
a graph page.

## Stop Conditions

- If a proposed demo abstraction needs feature flags to account for differences
  between graphs, keep the graphs separate.
- If a split makes the `StateGraph` topology harder to see, keep the nodes and
  edges together.
- If unit 06 preserved the public constructor, record a no-change outcome rather
  than rewriting equivalent registrations.
- Do not update a dependency or lockfile to obtain a stylistic simplification.

## Outcome

Not started.
