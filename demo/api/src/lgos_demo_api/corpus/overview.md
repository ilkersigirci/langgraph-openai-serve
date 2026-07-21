# LangGraph OpenAI Serve overview

LangGraph OpenAI Serve (LGOS) exposes registered LangGraph graphs through an
OpenAI-compatible `/v1` API. Applications choose their graph by sending its
registered name as the OpenAI `model`; LGOS adapts OpenAI messages to graph
input and converts graph output back to OpenAI responses.

The Python package provides the server integration, not a built-in application
graph. An application creates a `GraphRegistry`, registers its own graphs, and
binds `LanggraphOpenaiServe` to a FastAPI application. The
[demo API guide](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/api.md)
shows the complete demo, while the
[custom-graphs guide](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/tutorials/custom-graphs.md)
explains graph registration and adapters.

LGOS keeps OpenAI clients as its ingestion contract. Core behavior remains
reachable through `/v1`; project-specific chat envelopes, routes, headers, and
stream event formats are not required. Optional graph capabilities use standard
OpenAI request fields or namespaced, versioned response extensions. The
[compatibility explanation](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/explanation/openai-compatibility.md)
documents those boundaries.
