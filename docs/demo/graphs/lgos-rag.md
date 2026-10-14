# LGOS RAG

`lgos-rag` is a model-backed documentation assistant that answers questions
about langgraph-openai-serve from the docs of the installed LGOS release. It
searches every docs page and cites the exact sections it used, so answers
describe the release the demo runs.

The graph needs the gateway's chat and embedding models and outbound HTTPS
access to GitHub, which serves the release's docs. It has no checkpointer or
LangGraph Store.

## LangGraph Topology

Every request runs the same three nodes; only the planning call decides what to
search.

```mermaid
graph TD;
  __start__ --> plan;
  plan --> retrieve;
  retrieve --> answer;
  answer --> __end__;
```

## Request Flow

```mermaid
sequenceDiagram
  actor User
  participant UI as Chainlit / Open WebUI
  box LGOS API process
    participant Graph as lgos-rag
    participant Index as Docs index
  end
  participant GitHub
  participant Gateway as Gateway models

  User->>UI: Ask about LGOS
  UI->>Graph: /v1/responses, stream=true
  Graph-->>UI: Status "Understanding the question"
  Graph->>Gateway: Private plan call
  Gateway-->>Graph: Up to three search queries
  Graph-->>UI: Status "Searching the docs: …" per query
  opt First search in this process
    Index->>GitHub: GET the v{version} source archive
    Index->>Gateway: Embed every chunk
  end
  Graph->>Index: Search every query in parallel
  Index-->>Graph: Top ten sections per query
  Graph->>Gateway: Streamed answer call
  Graph-->>UI: Answer with URL citations
```

1. `plan` makes one private call with structured output. It first decides
   whether the documentation answers the message, then writes one standalone
   query per distinct part, at most three, resolving references to earlier
   turns. Greetings, thanks, questions about the conversation, unrelated
   requests, and messages too vague to search get no queries. LangGraph's `nostream` tag keeps every planning
   token out of the answer stream.
2. `retrieve` is code only. It publishes a status per query, which streaming
   Responses shows as commentary, searches the queries in parallel, and merges
   the top ten sections of each. With no queries it does nothing, so a
   greeting never loads the index.
3. `answer` is the only streamed call. With sections, it answers from them
   alone and cites them, or asks one clarifying question when the question
   could mean clearly different things. Without sections, it replies briefly,
   asks what a vague message means, or declines.
   Markdown links and bare URLs that exactly match a retrieved section become
   OpenAI `url_citation` annotations, so a citation does not depend on the
   model's link formatting. Other links stay ordinary Markdown.

## Why A Workflow

A tool-calling agent could search too, but for one library's docs the
model's only useful freedom is writing and splitting queries, and one
structured call does that. The fixed graph has a hard limit of two model calls,
a search-or-skip decision that tests can check, a status for each phase, and no
text written beside a tool call to reconcile with the streamed answer.

Three runs of the evaluation below, when it had 53 cases, failed at most one
case each, with a median latency of about 3.4 seconds. Questions whose second
search depends on the first search's results are the case a workflow cannot
handle; add a bounded, opt-in agent mode if evaluation shows them failing.

The graph asks its clarifying questions as ordinary replies, which the next
message answers through the conversation. An LGOS interrupt would pause and
checkpoint a read-only answer for nothing to approve, and declaring
`interrupts` would make the model unavailable through Chat Completions.

## Documentation Index

The graph indexes the Markdown under `docs/` in the GitHub source archive of
the installed release, tag `v<version>`, and links every section to the
published docs of that version,
`https://ilkersigirci.github.io/langgraph-openai-serve/<version>/`. The Markdown
keeps tables, code blocks, and links as the docs authors wrote them; the API
reference that the site generates from docstrings is not part of it.

Each page splits at its headings, outside code blocks, and long sections split
into chunks of at most 1,500 characters. A section links to its page and the
anchor Python-Markdown's table of contents gives its heading, which is how
Zensical builds the published anchors, including the numbered anchors of
repeated headings. Relative links become docs-site URLs, as they do on the
site, so a link the model copies still works. Each chunk starts with its
heading path, such as `Run The LGOS Server > Settings`, so embeddings, keyword
search, and the model all see where it belongs.

The graph answers only from a tagged release's docs. An unreleased LGOS
checkout, such as the one the `--dev` Compose stack installs, reports the
placeholder version `0.0.1`, because releases stamp their version from the tag.
For it, the graph uses the release that the docs site's `versions.json` marks as
`latest`, so its answers can trail docs changed in the checkout. Any other
version without release docs fails the request with a server error, and the API
log names the release archive it could not use.

## Client Settings

Clients that support runtime settings show two options; other clients send them
in `metadata.lgos_settings`.

`scope`, shown as **Docs scope**, is `all`, `library`, or `demo` and defaults to
`all`. `library` searches every page outside the Demo Stack section, and `demo`
searches only that section, so a question about an application of your own is
not answered from the demo stack's setup. For example, with `library`, "How do I
put LGOS behind LiteLLM?" finds the proxy guide instead of the demo's LiteLLM
sync page. The scope filters the search itself, and a restricted scope shows in
the status, such as "Searching the library docs: …". When a restricted search
does not answer the question, the answer says so and suggests setting
**Docs scope** to All.

`bm25`, shown as **BM25 keyword search**, defaults to `true`. Embedding search
finds sections by meaning and is weakest on exact names such as
`LGOS_INTERRUPT_TTL_MINUTES`, `RunBusyError`, or `GraphFeature.BACKGROUND`, so
the graph can also rank sections with BM25 keyword search. Each ranking then
contributes 20 candidates, and reciprocal rank fusion orders them before the
top ten are returned.

On the 0.30.0 docs, both modes put an answering page in the top ten for all 34
single-part evaluation questions. BM25 moved exact names up, such as
`LGOS_INTERRUPT_TTL_MINUTES` from second to first, and some plainly worded
questions down a place or two.

## Lifetime And Failure

The index lives in the API process. Its first search downloads the release
archive, about 2 MB, and embeds about 540 chunks, a few seconds of work and well
under a cent with `text-embedding-3-small`. Later searches reuse it until the
process restarts. If the download or embedding fails, the request fails instead
of being answered from the model's prior knowledge, and the next search tries
again.

The graph keeps no state between requests. Conversation history comes from the
messages the client sends.

## Evaluate The Graph

The evaluation runs as a Langfuse experiment, so every run keeps its traces and
scores for comparison. `demo/api/experiments/lgos_rag.py` holds 57 cases:
questions in a user's words, exact names, follow-ups that need the
conversation, multi-part questions, scoped searches, greetings, unrelated
requests, and the reply's shape after a restricted miss or a vague message.

```bash
just demo/experiment-rag
```

The script follows the Langfuse experiment runner convention: it exposes
`experiment(context)` and raises `RegressionError` when the run fails, so the
[Langfuse experiment action](https://github.com/langfuse/experiment-action) can
run the same file in CI.

Each run makes the hosted `lgos-rag` dataset's active items exactly these cases,
runs the whole graph on every item with the gateway's models and the published
docs, and records the graph's calls in the item's trace. The dataset's JSON
schemas come from the script's item models, so Langfuse rejects a malformed
item. Code then records each check that applies to a case as a `BOOLEAN` pass
or fail score:

| Score | Passes when |
| --- | --- |
| `route` | The graph searches exactly when the message needs the docs |
| `retrieval` | Each part of the question has an answering page among the retrieved sections |
| `scope` | A scoped case retrieves only sections in its scope |
| `links` | Every Markdown link in the answer comes from the retrieved sections |
| `citation` | The answer cites at least one retrieved section |
| `reply` | A reply-shape case shows the scope hint, or asks a clarifying question without declining |
| `passed` | The case has checks and passes all of them; a composite of the scores above |

The run's `pass_rate` is the share of cases scored `passed`, and its metadata
records the commit, LGOS version, models, and evaluator version. In
Langfuse, compare runs of the dataset to see which cases a prompt, model, or
retrieval change fixed or broke.

Model sampling flips a case or two between runs, so a run fails only when more
than three cases fail, and the error lists them. Add a case in the script
whenever a real question goes wrong, not in the Langfuse dataset, which each
run overwrites.

The script records to Langfuse when `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
and `LANGFUSE_BASE_URL` are set, as in `demo/.env`. Without them, as in CI, it
runs the same cases and checks on local data with Langfuse disabled, records
nothing, and still fails the run on too many failing cases. To run that way
locally, also turn off the API's own tracing:

```bash
LGOS_ENABLE_LANGFUSE=false LANGFUSE_PUBLIC_KEY= just demo/experiment-rag
```

## Try It

| Path | Prompt |
| --- | --- |
| No search | `Who are you?` |
| One search | `How do I enable Langfuse tracing?` |
| Exact name | `What does LGOS_INTERRUPT_TTL_MINUTES control?` |
| Several queries | `How do I resume an interrupted run in the background, and what does LGOS_INTERRUPT_TTL_MINUTES do?` |
| Follow-up | `How do I enable Langfuse tracing?`, then `How do I point it at a self-hosted instance?` |
| Library scope | Set **Docs scope** to `library`, then ask `How do I put LGOS behind LiteLLM?` |
| Scope hint | Set **Docs scope** to `library`, then ask `What does DEMO_GRAPH_GATEWAY_API_KEY set?` |
| Vague message | `Why is it failing?` |

### Python SDK

With the [demo API client](../api.md#call-a-graph), search only the library
docs, without BM25, for one request:

```python
response = client.responses.create(
    model="lgos-rag",
    input="How do I put LGOS behind LiteLLM?",
    metadata={"lgos_settings": '{"scope": "library", "bm25": false}'},
    store=False,
)
print(response.output_text)
```

The implementation is in `demo/api/src/lgos_demo_api/graphs/lgos_rag.py`.
