# File Input

`file-input` is a small model-backed graph for trying native Chat Completions
file parts end to end. It reads each central `file_id`, downloads the original
bytes, and sends them to the configured OpenAI Responses API. It has no graph
persistence.

## Request Flow

1. Chainlit uploads the current attachments to the central demo Files service
   and places the returned IDs in the user message.
2. LGOS preserves those native `file` content parts in the LangChain
   `HumanMessage`.
3. The graph retrieves the filename and bytes from `DEMO_API_FILES_BASE_URL`.
4. Images become inline `input_image` data URLs. Other files become inline
   `input_file` data URLs with their original filename.
5. The graph calls `responses.create` and returns `response.output_text` as the
   assistant message.

The Responses API accepts Base64 data in `input_file` items. Supported parsing
depends on the file type; see the official OpenAI
[file input guide](https://developers.openai.com/api/docs/guides/file-inputs).

```mermaid
sequenceDiagram
    participant C as Chainlit
    participant F as Central Files API
    participant G as file-input graph
    participant O as OpenAI Responses
    C->>F: POST /v1/files
    F-->>C: file_id
    C->>G: Chat completion with file_id
    G->>F: GET metadata and content
    F-->>G: filename and bytes
    G->>O: Inline input_file or input_image
    O-->>G: output_text
    G-->>C: Assistant text
```

## LangGraph Topology

```mermaid
graph TD;
    __start__ --> process_files;
    process_files --> __end__;
```

## Try It

Run the Compose Chainlit stack, select `lgos-a/file-input`, attach a supported
document or image, and send a request such as:

```text
Summarize this file in three bullets.
```

The demo downloads and forwards the entire attachment for each request. It is
therefore intended for small files, not retrieval over a large corpus. The
central `file_id` is only a reference, not authorization; production services
must enforce file access and retention at their own boundary.
