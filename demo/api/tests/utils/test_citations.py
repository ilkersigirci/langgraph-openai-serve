from langchain_core.messages import AIMessage
from langgraph_openai_serve.graph.citations import citations_from_message

from lgos_demo_api.utils.citations import cite_sources

SOURCES = {
    "https://docs.example.com/install/": "Install",
    "https://docs.example.com/diagram.png": "Diagram",
}


def test_links_and_bare_urls_matching_a_source_are_cited() -> None:
    text = (
        "See [the install guide](https://docs.example.com/install/), "
        "https://docs.example.com/install/. "
        "Not [another site](https://example.com/), "
        "![a diagram](https://docs.example.com/diagram.png)."
    )

    cited = cite_sources(AIMessage(content=text), SOURCES)

    assert cited.text == text
    assert [
        (citation["url"], text[citation["start_index"] : citation["end_index"] + 1])
        for citation in citations_from_message(cited)
    ] == [
        ("https://docs.example.com/install/", "the install guide"),
        ("https://docs.example.com/install/", "https://docs.example.com/install/"),
    ]
