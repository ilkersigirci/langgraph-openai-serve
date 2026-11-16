# LGOS Files API

This independent demo project exposes the OpenAI-compatible Files API over an
S3-compatible object store and extracts the text of uploaded documents. It does
not depend on `langgraph-openai-serve`, LangGraph, or the demo graph API.

Configure the `DEMO_API_FILES_*` values in `demo/.env`, then run:

```bash
just demo/files
```

The OpenAI base URL is `http://localhost:3006/v1`. The service supports upload,
list, retrieve, content, and delete for files with `purpose="user_data"`.

## Content Extraction

After an upload, the service extracts the content of these documents in the
background:

| Files | Extractor | Content |
| --- | --- | --- |
| `.pdf`, `.docx`, `.doc`, `.odt`, `.rtf`, `.pptx`, `.ppt`, `.html`, `.htm` | [xberg](https://github.com/xberg-io/xberg), OCR included | `{"type": "text", "version": 1, "text": "<Markdown>"}` |
| `.xlsx`, `.xls` | xberg | `{"type": "spreadsheet", "version": 1, "sheets": [{"name", "headers", "rows", "chunks"}]}` |
| `.md`, `.markdown`, `.txt`, `.log`, `.csv`, `.tsv`, `.xml`, `.yaml`, `.yml`, `.toml`, `.sql`, and code: `.py`, `.js`, `.java`, `.c`, `.cpp`, `.h`, `.go`, `.rs`, `.rb`, `.php`, `.sh`, `.css` | None: the file's own text | `{"type": "text", "version": 1, "text": "..."}` |

Text files are stored as `text/plain`, since clients label code and data
files inconsistently and OpenAI's file inputs reject some of those labels.
Other uploads labelled `application/octet-stream`, as Bifrost forwards every
upload, are labelled from their filename.
JSON, the format of generated charts, keeps its original bytes as content.

A document's `status` is `uploaded` until its content is stored, then
`processed`, or `error` with `status_details`, so OpenAI SDK clients can call
`files.wait_for_processing`. `GET /v1/files/{file_id}/content` returns the
content as JSON once the document is `processed`, a `409` error before, and the
original upload with `?raw=true`. Other files, such as the images and charts
that graphs generate, return their original bytes.

xberg runs in the service process, at most
`DEMO_API_FILES_MAX_CONCURRENT_EXTRACTIONS` documents at once with
`DEMO_API_FILES_EXTRACTION_THREADS` threads; size them to the CPUs the process
may use, since xberg otherwise sizes its pools, and its memory, by the host's
cores. Its bundled Tesseract OCRs pages without a text layer; the image ships
English language data only, which a local run downloads on its first OCR.

The extraction state lives in S3 next to the files. A document without content
or an error is extracted again by its next retrieve, such as after a restart.
Only extractions that get a slot count as attempts, and a document whose two
attempts never finished, for example by running the service out of memory,
fails. A document the extractors cannot read or that times out fails for good;
other failures, such as of storage, are retried by the next retrieve. Run one
service process per bucket.

This is a small-scale reference implementation. Add authentication, tenant
isolation, retention, and malware policy before exposing it to untrusted
clients.
