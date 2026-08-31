---
name: demo-graph-doc
description: Create or revise concise docs for graphs under docs/demo/graphs, for both newcomers learning the request flow and maintainers preserving design knowledge.
---

# Demo Graph Documentation

Create a graph page that lets a newcomer quickly understand what the graph
does and lets its maintainer recover the important runtime and design decisions
later. Prefer a short, layered explanation over either a code walkthrough or a
minimal catalog entry.

Use `docs/demo/graphs/persistent-plot-agent.md` as a quality reference, not a
fixed template. Include only sections that explain the graph at hand.

## Establish the Facts

Before writing, trace the behavior through the relevant sources:

- the graph implementation, `GraphConfig`, registration, and lifespan wiring;
- request/input/context adapters and the LGOS runner behavior it depends on;
- Chainlit or Open WebUI adapters when they change the visible contract;
- focused tests that establish persistence, streaming, errors, or concurrency;
- existing architecture and compatibility docs, so this page links instead of
  redefining shared contracts.

Verify dependency behavior and links against current official documentation.
Do not infer behavior from names, prompts, stale prose, or an adjacent graph.
When code and docs disagree, describe the implemented behavior and fix stale
documentation within the requested scope.

## Explain the Graph

Start with a brief answer to:

- What user problem does this graph demonstrate?
- Is it a real model-backed agent or a deterministic transport example?
- What important state or external dependency, if any, does it use?

Then explain one request from client input to visible output. A compact numbered
flow is enough for a simple graph. Use a small Mermaid sequence or flow diagram
when ownership, persistence, branching, or three or more interacting components
would otherwise be hard to follow.

Every documented graph must also show the topology produced by LangGraph
itself. Construct the compiled graph with the same dependency-free or in-memory
setup used by `demo/api/notebooks/graph_visualization.py`, then generate concise
Mermaid source with:

```python
diagram = graph.get_graph(xray=True).draw_mermaid(with_styles=False)
```

Place the output in a `mermaid` fence under `LangGraph Topology`. If generated
node IDs do not render, replace only those IDs with Mermaid-safe aliases and
preserve their qualified names as labels. Use one content tab per graph when a
page documents several graphs. Refresh the diagram whenever nodes, edges,
routing, or nested subgraphs change. The native topology does not replace the
request flow: it shows compiled graph structure, while the request flow explains
adapters, persistence, events, UIs, and cleanup that are not necessarily graph
nodes.

Cover the meaningful stages, which may include:

1. client request fields and required correlation or settings;
2. LGOS validation, input conversion, context construction, and coordination;
3. model, node, or tool selection;
4. reads, writes, checkpoints, events, or external calls;
5. assistant text and rich-client rendering;
6. cleanup, retry, concurrency, or failure behavior.

Explain read and mutation paths separately when they differ. Use one concrete,
supported request for each important path. State limitations such as absolute
versus relative edits instead of implying unsupported behavior.

## Preserve the Important Boundaries

Name the source of truth and its lifetime. Clearly distinguish, when relevant:

- UI-owned conversation history and rendered elements;
- graph state persisted through a LangGraph checkpointer;
- application documents persisted through a LangGraph Store;
- request-scoped runtime settings and context;
- request-only coordination leases and process-local objects.

Describe namespace or thread scoping, defaults, write conditions, concurrency,
and restart behavior only to the depth needed to understand correctness. Never
present caller-provided IDs or hashing as authorization.

For rich artifacts, document the semantic graph event separately from each
UI's native rendering. Keep canonical application data behind the graph/API
boundary; explain why clients do not read persistence tables directly. For
large artifacts, mention IDs or authorized expiring URLs only when that design
guidance is relevant.

## Layer the Detail

Keep one canonical owner for each explanation:

- graph pages own graph behavior, request flow, state, and dependencies;
- the API guide owns invocation examples;
- UI guides own adapter-specific rendering, persistence, and recovery;
- the graph index owns only short catalog summaries and links.

Link across those boundaries instead of repeating their details. Group small,
related graphs on one page when separate pages would repeat the same concepts.

Keep the primary path easy to scan. Useful sections often include:

- a short introduction;
- `Request Flow`;
- persistence or external-dependency scope;
- output or UI rendering;
- ownership, security, or meaningful limitations;
- `Try It` with a minimal reproducible conversation.

Do not require every heading. Avoid repeating general LGOS architecture already
documented elsewhere. Put secondary deployment topology or component-level
ownership in a native `???` collapsible block instead of deleting knowledge or
interrupting the main flow.

When simplifying an existing page, compare before and after. Preserve current
operational rationale, ownership boundaries, persistence behavior, and failure
semantics. Remove statements that became false; do not retain them merely for
historical completeness.

## Writing Rules

- Lead with behavior and purpose, not filenames or implementation history.
- Explain why a boundary exists; do not narrate obvious functions line by line.
- Prefer short prose, small tables, and one useful diagram over exhaustive
  payload dumps. The required native topology and an authored request-flow
  diagram may both be useful when they explain different boundaries.
- Use exact public names for request fields, graph features, tools, events,
  status codes, and error codes.
- Say explicitly when a graph has no persistence or when a rich event requires
  streaming and client opt-in.
- Use official package capabilities and terminology; do not invent a parallel
  abstraction in the documentation.
- Follow the repository's Zensical Markdown conventions in `AGENTS.md`.
- Preserve established heading anchors. Search for inbound `#anchor` links
  before renaming or removing a heading.

## Validate

After editing:

1. Re-read the implementation and tests against every behavioral claim.
2. Search for inbound links to renamed headings and for stale terminology.
3. Run `make doc-build` and resolve every strict-build issue.
4. Run `git diff --check`.
5. Preview every changed Mermaid diagram in the browser and confirm it renders
   correctly, because the strict build does not parse Mermaid syntax.
