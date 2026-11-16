# Run The Files API

The independent `demo/files_api` project exposes an OpenAI-compatible Files API
over S3-compatible storage and extracts the text of uploaded documents. It has
its own package, lockfile, Dockerfile, image, settings, and tests. It does not
depend on `langgraph-openai-serve`, LangGraph, or the demo graph API.

## Run The Service

Configure the `DEMO_API_FILES_*` values in `demo/.env`, then choose one mode:

=== "Local process"

    ```bash
    just demo/files
    ```

=== "Published container"

    ```bash
    just demo/up lgos-files-api
    ```

The OpenAI base URL is `http://localhost:3006/v1`; health is available at
`http://localhost:3006/health`. The bucket must already exist.

The service supports upload, list, retrieve, content, and delete operations for
`purpose="user_data"`. The maintained UIs reach it through Bifrost's
`lgos-files` provider or LiteLLM's `litellm_proxy` Files provider; they do not
connect directly. Direct protocol checks may call it on port 3006. The
`file-input` graph uses the same service to resolve an incoming `file_id`.

## Extract Document Content

After an upload, the service extracts each document's content in the
background and stores it in S3 next to the original:

| Files | Extractor | Content |
| --- | --- | --- |
| `.pdf`, `.docx`, `.doc`, `.odt`, `.rtf`, `.pptx`, `.ppt`, `.html`, `.htm` | [xberg](https://github.com/xberg-io/xberg), OCR included | `{"type": "text", "version": 1, "text": "<Markdown>"}` |
| `.xlsx`, `.xls` | xberg | `{"type": "spreadsheet", "version": 1, "sheets": [...]}` |
| `.md`, `.markdown`, `.txt`, `.log`, `.csv`, `.tsv`, `.xml`, `.yaml`, `.yml`, `.toml`, `.sql`, and code: `.py`, `.js`, `.java`, `.c`, `.cpp`, `.h`, `.go`, `.rs`, `.rb`, `.php`, `.sh`, `.css` | None: the file's own text | `{"type": "text", "version": 1, "text": "..."}` |

Each sheet lists its `name`, `headers`, data `rows`, and `chunks`: groups of
rows rendered as `Row N | Header: value` lines for retrieval. Cells hold the
values xberg reads, such as dates as ISO dates. A CSV file keeps its own text,
so every value stays exactly as written.

Graphs forward the originals to OpenAI, so these are types OpenAI's file
inputs accept. The service stores text files as `text/plain`, whatever their
upload's label: clients label code and data files inconsistently, and OpenAI
rejects some of those labels. It labels other uploads that arrive as
`application/octet-stream`, as Bifrost forwards every upload, from their
filename. JSON stays out, since generated charts are JSON
files that clients display from their original bytes, and so does TypeScript,
which browsers label as video.

```mermaid
sequenceDiagram
  participant Client
  participant Files as Files API
  participant S3
  Client->>Files: POST /v1/files
  Files->>S3: Original upload
  Files-->>Client: file, status "uploaded"
  Files->>Files: Extract in the background
  Files->>S3: Extracted content
  Client->>Files: GET /v1/files/{id}
  Files-->>Client: status "processed"
  Client->>Files: GET /v1/files/{id}/content
  Files-->>Client: Extracted content JSON
```

OpenAI reports file processing through its deprecated `status` field, so SDK
clients can wait with `files.wait_for_processing`:

| `status` | Content endpoint |
| --- | --- |
| `uploaded` | A `409` error, which the OpenAI SDK retries, until the extraction ends |
| `processed` | The extracted content as JSON |
| `error`, with `status_details` | The extraction's error |

`GET /v1/files/{file_id}/content?raw=true` returns the original upload of any
file. Files without an extractor, such as the images and Plotly charts that
graphs generate, always return their original bytes, which the UIs display.

xberg runs in the service process, at most
`DEMO_API_FILES_MAX_CONCURRENT_EXTRACTIONS` documents at once with
`DEMO_API_FILES_EXTRACTION_THREADS` threads, each within
`DEMO_API_FILES_EXTRACTION_TIMEOUT_SECONDS` of getting a slot. Size the threads
to the container's CPU limit: xberg otherwise sizes its pools, and with them its
memory, by the host's cores. Its bundled Tesseract reads pages without a text
layer; the image ships English OCR data only, so no extraction downloads
anything at runtime.

| Failure | `status_details` | Extracted again |
| --- | --- | --- |
| A file the extractors cannot read | `Failed to parse the file.` | No |
| An extraction over its time limit | `Timed out while parsing the file.` | No |
| A storage or other service error | None; the file stays `uploaded` | By its next retrieve |
| Two extractions that never finished, such as by running the service out of memory | `The extraction did not finish.` | No |

The extraction state lives in S3 next to the files, so a restart loses nothing:
a document without content or an error is extracted again by its next
retrieve. Only extractions that get a slot count toward the two, so a restart
never fails a document that was waiting for one. Run one service process per
bucket, because processes do not share which extractions they are running.

The service refuses an upload over `DEMO_API_FILES_MAX_UPLOAD_BYTES` once
Starlette has written it to `/tmp`, so keep `/tmp` larger than the limit and
cap request bodies at the gateway or proxy in front of the service.

## Ownership Boundary

Only this service receives the `DEMO_API_FILES_AWS_*` credentials and bucket
configuration. LGOS APIs receive only the Files base URL needed by a
graph that resolves IDs. Chainlit's native element-storage credentials remain
separate, and Open WebUI retains its own copy of each upload, with the text
this service extracted, in its data volume.

The demo is a deliberately small reference implementation. Before exposing it
to untrusted clients, add authentication, tenant isolation, retention, malware
policy, and the availability controls required by the
deployment. See [Accept And Display Files](../how-to-guides/file-inputs.md) for the
client and graph flow and [Settings and Commands](reference.md#files-api-settings)
for configuration.
