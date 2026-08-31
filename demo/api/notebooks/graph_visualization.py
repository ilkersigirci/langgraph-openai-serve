import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demo graph visualization

    Explore the LangGraph workflows registered by the demo API. The diagrams
    use LangGraph's native Mermaid representation with `xray=True`, recursively
    expanding all discoverable subgraphs in one diagram.

    **Single-node subgraphs:** When recursively expanded, a nested subgraph with
    only one real node has no internal edge. The Mermaid generator may then show
    its escaped qualified ID instead of a friendly node label.
    """)
    return


@app.cell
def _():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from lgos_demo_api.graphs.advanced_mcp import advanced_mcp_graph
    from lgos_demo_api.graphs.citations import citation_graph
    from lgos_demo_api.graphs.custom_events import custom_event_showcase_graph
    from lgos_demo_api.graphs.custom_io import custom_io_graph
    from lgos_demo_api.graphs.interruptible import create_interruptible_graph
    from lgos_demo_api.graphs.lgos_rag import lgos_rag
    from lgos_demo_api.graphs.multi_node_streaming import multi_node_streaming_graph
    from lgos_demo_api.graphs.persistent_plot_agent import (
        create_persistent_plot_agent,
    )
    from lgos_demo_api.graphs.simple import simple_graph
    from lgos_demo_api.graphs.status_events import status_event_graph
    from lgos_demo_api.graphs.subgraphs.specialist_team import (
        create_specialist_team_graph,
    )

    return (
        InMemorySaver,
        InMemoryStore,
        advanced_mcp_graph,
        citation_graph,
        create_interruptible_graph,
        create_persistent_plot_agent,
        create_specialist_team_graph,
        custom_event_showcase_graph,
        custom_io_graph,
        lgos_rag,
        multi_node_streaming_graph,
        simple_graph,
        status_event_graph,
    )


@app.cell
async def _(
    InMemorySaver,
    InMemoryStore,
    advanced_mcp_graph,
    citation_graph,
    create_interruptible_graph,
    create_persistent_plot_agent,
    create_specialist_team_graph,
    custom_event_showcase_graph,
    custom_io_graph,
    lgos_rag,
    multi_node_streaming_graph,
    simple_graph,
    status_event_graph,
):
    graphs = {
        "custom-input-output-context": custom_io_graph,
        "citation-events": citation_graph,
        "advanced-mcp-tools": await advanced_mcp_graph(),
        "complex-subgraphs": create_specialist_team_graph(),
        "status-events": status_event_graph,
        "custom-event-showcase": custom_event_showcase_graph,
        "multi-node-streaming": multi_node_streaming_graph,
        "persistent-plot-agent": create_persistent_plot_agent(InMemoryStore()),
        "interruptible-approval": create_interruptible_graph(InMemorySaver()),
        "simple-graph": simple_graph,
        "lgos-rag": lgos_rag,
    }
    return (graphs,)


@app.cell
def _(graphs, mo):
    graph_rows = []
    for name, graph in graphs.items():
        _summary_graph_view = graph.get_graph()
        _summary_subgraphs = list(graph.get_subgraphs(recurse=True))
        graph_rows.append(
            {
                "graph": name,
                "nodes": len(_summary_graph_view.nodes),
                "edges": len(_summary_graph_view.edges),
                "subgraphs": len(_summary_subgraphs),
            }
        )

    mo.vstack(
        [
            mo.md("## Available demo graphs"),
            mo.ui.table(graph_rows, pagination=False, selection=None),
        ]
    )
    return


@app.cell
def _(graphs, mo):
    graph_selector = mo.ui.dropdown(
        options=list(graphs),
        value="custom-input-output-context",
        label="Graph",
        searchable=True,
        full_width=True,
    )
    mo.vstack([graph_selector])
    return (graph_selector,)


@app.cell(hide_code=True)
def _(graph_selector, graphs, mo):
    selected_name = graph_selector.value
    selected_graph = graphs[selected_name]
    _selected_graph_view = selected_graph.get_graph(xray=True)
    _selected_diagram = _selected_graph_view.draw_mermaid(with_styles=False)

    mo.vstack(
        [
            mo.md(
                f"## `{selected_name}`\n\n"
                f"Graph: {len(_selected_graph_view.nodes)} nodes, "
                f"{len(_selected_graph_view.edges)} edges."
            ),
            mo.mermaid(_selected_diagram),
            mo.accordion(
                {"Mermaid source": mo.md(f"```text\n{_selected_diagram}\n```")}
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
