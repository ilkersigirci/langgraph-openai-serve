# Complex Subgraphs

`complex-subgraphs` demonstrates a coordinator graph with specialist subgraphs
and one nested grandchild graph. The OpenAI request adapter supplies the final
user message as `question`; the output adapter returns the selected specialist's
`answer` as assistant text.

```mermaid
flowchart TD
  request([OpenAI user message]) --> adapter["request_to_input"]
  adapter --> route["Normalize question and select route"]

  subgraph api["API contract subgraph"]
    checks["Collect contract checks"] --> api_summary["Stream contract summary"]
  end

  subgraph docs["Docs specialist subgraph"]
    subgraph keywords["Keyword grandchild subgraph"]
      extract["Stream keyword extraction"]
    end
    extract --> docs_summary["Stream docs summary"]
  end

  route -->|"API, OpenAI, adapter, stream, or serve"| checks
  route -->|"other questions"| extract
  api_summary --> output["output_to_message"]
  docs_summary --> output
  output --> response([OpenAI assistant text])
```

LGOS exposes only the explicitly configured nested streaming nodes:
`extract_keywords`, `summarize_contract`, and `summarize_docs`. The client does
not need to understand the internal graph hierarchy.
