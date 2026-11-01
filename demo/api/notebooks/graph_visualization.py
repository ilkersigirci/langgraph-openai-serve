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
async def _():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
    from langgraph_openai_serve.server import ServerResources

    from lgos_demo_api.registry import open_registry

    # The served registry, over in-memory persistence; drawing sends no request.
    resources = ServerResources(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        run_coordinator=InMemoryRunCoordinator(),
    )
    async with open_registry(resources) as registry:
        graphs = {
            name: await config.resolve_graph()
            for name, config in registry.graphs.items()
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
