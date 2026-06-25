# LangGraph OpenAI Serve Docs

`langgraph-openai-serve` maps registered LangGraph graphs to OpenAI `model`
names and serves them through OpenAI-compatible chat, model, and health
endpoints.

## Start Here

- [Getting Started](tutorials/getting-started.md): run the demo API and inspect
  the included graph examples.
- [OpenAI Clients](tutorials/openai-clients.md): Python, JavaScript, streaming,
  and diagnostic `curl` calls.
- [Custom Graphs](tutorials/custom-graphs.md): default graph shape, adapters,
  async factories, streaming, and interrupts.
- [Reference](reference.md): endpoints, settings, public classes, commands, and
  demo graph names.

## Concepts And Operations

- [OpenAI Compatibility](explanation/openai-compatibility.md): public contract
  and compatibility boundaries.
- [Architecture](explanation/architecture.md): request flow and major modules.
- [LangGraph Integration](explanation/langgraph-integration.md): how requests
  become graph input and responses.
- [Authentication](how-to-guides/authentication.md): bearer-token auth that works
  with OpenAI-compatible clients.
- [Docker](how-to-guides/docker.md): compose demo and custom deployment notes.
