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
| `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt` | [xberg](https://github.com/xberg-io/xberg), OCR included | `{"type": "text", "version": 1, "text": "<Markdown>"}` |
| `.md`, `.txt` | The service | `{"type": "text", "version": 1, "text": "..."}` |
| `.xlsx`, `.xls` | The service | `{"type": "spreadsheet", "version": 1, "sheets": [{"name", "headers", "rows", "chunks"}]}` |

A document's `status` is `uploaded` until its content is stored, then
`processed`, or `error` with `status_details`, so OpenAI SDK clients can call
`files.wait_for_processing`. `GET /v1/files/{file_id}/content` returns the
content as JSON, `{}` while it is being extracted, and the original upload with
`?raw=true`. Other files, such as the images and charts that graphs generate,
return their original bytes.

xberg runs in the service process, at most
`DEMO_API_FILES_MAX_CONCURRENT_EXTRACTIONS` documents at once with
`DEMO_API_FILES_EXTRACTION_THREADS` threads; size them to the CPUs the process
may use, since xberg otherwise sizes its pools, and its memory, by the host's
cores. Its bundled Tesseract OCRs pages without a text layer; the image ships
English language data only, which a local run downloads on its first OCR.

The extraction state lives in S3 next to the files. Extractions that a shutdown
cuts short resume at the next start, once; a document that stops the service
twice fails. A file the extractors cannot read stays failed, and other failures
are extracted again by a retrieve five minutes later. Run one service process
per bucket. Files stored before content extraction, under `openai-files/`, are
not read.

This is a small-scale reference implementation. Add authentication, tenant
isolation, retention, and malware policy before exposing it to untrusted
clients.
