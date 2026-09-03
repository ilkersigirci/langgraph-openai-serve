# Bundled demo models

The demo API registers these graph names as OpenAI models:

- `simple-graph` demonstrates conversation-history and audience runtime
  settings.
- `simple-graph-external-tools` forwards client-provided tool definitions to the
  chat model and returns tool calls for the client to execute.
- `citation-events` emits structured OpenAI URL citations alongside portable
  Markdown links.
- `lgos-rag` retrieves from this packaged Markdown corpus, grades relevance,
  performs at most one query rewrite, and grounds its answer in retrieved text.
- `custom-input-output-context` demonstrates graph input, output, and context
  adapters.
- `advanced-mcp-tools` demonstrates an asynchronous graph factory and mock
  MCP-style tools.
- `complex-subgraphs` demonstrates routing across nested specialist graphs.
- `multi-node-streaming` combines streamed contributions from two sequential
  nodes into one final assistant message.
- `status-events` streams portable status updates for native client UI.
- `custom-event-showcase` streams explicitly public progress and artifact
  events among ordinary assistant text.
- `persistent-plot-agent` uses a tool-calling agent to edit chart data scoped
  to the current user and chat, with request-scoped presentation settings.
- `interruptible-approval` accepts a preset choice or custom reviewer feedback
  before protected actions, then resumes from a standard OpenAI tool result.

`GET /v1/models` lists the registered names. Detailed retrieval of one model
can additionally advertise versioned LGOS features and safe client settings.
The [demo graph catalog](https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/docs/demo/graphs/index.md)
summarizes the models and their runtime requirements.
