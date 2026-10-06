# Coding-agent API

A showcase for serving coding agents through LGOS at `/v1`. The current
implementation uses the official `openai-codex` Python SDK. It can inspect and
edit files, run Bash commands and tests, and stream progress and answers to
Chainlit or Open WebUI through the gateway.

The service and graph use agent-independent names. Codex-specific event and
runtime handling live in `codex_model.py` and `codex_runtime.py`; the graph accepts
a LangChain `BaseChatModel`. Codex is the only implementation currently included.

Every request goes directly to Codex. It can answer questions from the supplied
conversation or use its workspace tools as needed.

Configure `demo/.env` using the coding-agent settings in `.env.example`, then run:

```bash
just demo/compose --dev
```

Select `lgos-api-coding-agent/coding-agent` in either UI. Try:
“Create a Python calculator with a unittest suite, run it, and report the result.”
Follow up with “Add division and test division by zero.”

The workspace is the host directory `demo/docker/volumes/lgos-coding-agent`.
Copy or clone a project into it to edit existing code. All conversations
share that workspace; requests run one at a time, including runtime cleanup.
Codex remembers each UI conversation: requests carrying `user` and
`metadata.conversation_id` resume that conversation's Codex thread, stored in
`demo/docker/volumes/lgos-codex`. Other requests run on a thread that is
discarded when they end.

`DEMO_CODING_AGENT_BASE_URL`, `DEMO_CODING_AGENT_API_KEY`, and `DEMO_CODING_AGENT_MODEL` select the
upstream Responses model. Their defaults use the demo API's upstream. They can
point to another gateway deployment as long as it supports Codex's Responses
requests. The UI-facing gateway routes directly to `http://lgos-api-coding-agent:8000/v1`
using the demo's existing model catalog sync.

Docker supplies the execution boundary: a non-root process, read-only root,
writable workspace and temporary directories, and no Docker socket. Codex has
network access, including to the other demo services, and can read its own
model credential. Use this shared service
with trusted users and repositories. Running the coding agent directly on the host
would give its commands the host process's permissions.

See the [graph guide](../../docs/demo/graphs/coding-agent.md) for the
request flow and examples. This project has its own dependencies and lockfile.

```bash
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ty check src
```
