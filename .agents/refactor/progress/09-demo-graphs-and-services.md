# 09 — Demo Graphs And Services

- Status: **Waiting for graph registration**
- Priority: **P2**
- Dependencies: **06**

## Objective

Migrate every demo to the final package API, then simplify only the demo modules
where a mixed responsibility remains after that migration. Preserve each
example's purpose and use the native LangChain, LangGraph, OpenAI, Plotly,
FastAPI, boto3, and psycopg capabilities it demonstrates.

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
classes. The migration caused by unit 06 is the first useful test: code that
remains hard to explain afterward is a candidate for a separate cleanup.

## 09A — Registration Migration

Migrate all `GraphConfig` and `GraphRegistry` construction in the demo API and
notebooks in one mechanical change after unit 06. Keep graph IDs, descriptions,
feature declarations, adapters, client settings, and runtime factory lifetimes
unchanged. Update the matching graph docs and examples to the final API; do not
add aliases for the old registration form.

Validate this substep before any graph-internal cleanup so registry regressions
are distinguishable from demo behavior changes.

## 09B — Advanced And Persistent Graphs

Review `advanced_graph/graph.py` after the migration. Its small top-level
routing helpers are cohesive. The nested nodes and edge selectors close over
the model, knowledge service, Files client, checkpointer, and Store used to
compile one graph. Keep those closures when moving them would require a large
dependency object or strategy hierarchy.

Extract only a pure policy that is independently understandable and reused by
multiple nodes, or a node group that has one clear dependency boundary. Retain
the native `StateGraph` declaration together so readers can see the workflow.
Do not hide edges behind a graph-builder framework.

The durable notebook receipt in the advanced graph records upload/index stages
so retries do not duplicate side effects. Keep it unless a locked LangGraph or
OpenAI SDK primitive provides the same durable, idempotent workflow. Preserve
the failure and restart tests before changing its order.

`persistent_plot_agent.py` uses LangChain's native `create_agent`, typed tools,
Store, and OpenAI Files. Its graph setup, persistence scope, Plotly rendering,
and output adapter are distinct but closely related to one example. A split is
optional; prefer removing coarse `Any` at the compiled-graph and output boundary
if the locked generic types permit it. Do not wrap `create_agent` in a local
agent abstraction.

## 09C — Smaller Graphs And Standalone Services

Review smaller graph modules independently. Consolidate a helper only when two
graphs have the same contract, not merely similar syntax. In particular:

- keep MCP filtering and the external-tool loop explicit because their trust
  and tool-choice rules differ from ordinary model calls;
- keep citation and client-event helpers at the existing common boundary;
- keep graph fixtures separate when each demonstrates a different public
  feature;
- validate untyped provider or gateway JSON once at the edge, then use precise
  types internally.

`sync_litellm.py` already validates both the LGOS catalog and the relevant
LiteLLM management response before reconciliation. If cleanup is still useful,
separate pure desired/current reconciliation from HTTP writes so dry-run and
mutation share one decision path. Preserve ownership checks, deterministic IDs,
credential-safe errors, and the rule that all conflicts are detected before the
first write. Do not widen this into a general gateway synchronizer.

Retain the Files service repository protocol and S3 native paginator/streaming
body. Its cursor behavior, metadata encoding, and guaranteed body close have
focused contracts. Change it only for a concrete finding, and use botocore or
boto3 primitives instead of a second storage layer.

Retain `postgres_runtime()` and `PostgresRunCoordinator` ownership. PostgreSQL
session advisory locks must stay on one checked-out connection and survive
transaction rollback; a connection with indeterminate cancellation state must
not return to the pool. Unit 03 may simplify how the server owns the runtime,
but this demo should not reproduce that lifecycle.

## Files In Scope

- `demo/api/src/lgos_demo_api/graphs/`
- `demo/api/src/lgos_demo_api/app.py`
- `demo/api/src/lgos_demo_api/sync_litellm.py`
- `demo/api/notebooks/`
- `demo/files_api/` only for a concrete issue found during final migration
- matching demo tests and pages under `docs/demo/`

Keep 09A, 09B, and 09C as separate reviewable implementation changes. A later
agent may mark one substep complete without claiming the whole unit is done.

## Required Behavior

- Every model keeps the same public ID, advertised capabilities, description,
  settings schema, and `/v1` behavior unless a separately documented contract
  fix requires a change.
- Lifespan-managed graphs still resolve only after their dependencies exist.
- Advanced graph routing, research, note approval/revision, durable side
  effects, client tools, refusals, and incomplete outcomes remain covered.
- Persistent chart state remains scoped by user and conversation and survives
  runtime restart through LangGraph Store.
- LiteLLM sync remains deterministic, detects conflicts before mutation, and
  leaves unowned deployments untouched.
- Files list, upload, retrieval, deletion, byte streaming, and cursor behavior
  remain OpenAI-compatible.

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

For changes to durable graph behavior, additionally run the PostgreSQL suite:

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
- Do not update a dependency or lockfile to obtain a stylistic simplification.

## Outcome

Not started.
