import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", sql_output="polars")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PostgreSQL tables

    Explore the PostgreSQL data shared by the LangGraph checkpointer, store,
    and Chainlit. The connection is read-only.

    Use the **Data Sources** panel for Marimo's native PostgreSQL browser, or
    choose a table below to inspect its columns and a small sample of rows.
    """)
    return


@app.cell
def _():
    import sqlalchemy
    from sqlalchemy.engine import make_url

    from lgos_demo_api.settings import settings

    _url = make_url(settings.POSTGRES_URI).set(drivername="postgresql+psycopg")
    postgres = sqlalchemy.create_engine(
        _url,
        connect_args={"options": "-c default_transaction_read_only=on"},
        pool_pre_ping=True,
    )
    return postgres, sqlalchemy


@app.cell
def _(mo, postgres):
    table_inventory = mo.sql(
        f"""
        WITH inventory AS (
            SELECT
                CASE
                    WHEN relname IN (
                        'checkpoint_blobs',
                        'checkpoint_migrations',
                        'checkpoint_writes',
                        'checkpoints'
                    ) THEN 'Checkpointer'
                    WHEN relname IN ('store', 'store_migrations', 'store_vectors')
                        THEN 'Store'
                    WHEN relname IN ('Element', 'Feedback', 'Step', 'Thread', 'User')
                        OR relname LIKE '%chainlit%schema%migrations%'
                        THEN 'Chainlit'
                    ELSE 'Other'
                END AS component,
                schemaname AS table_schema,
                relname AS table_name,
                n_live_tup AS estimated_rows,
                pg_size_pretty(pg_total_relation_size(relid)) AS total_size
            FROM pg_stat_user_tables
        )
        SELECT
            CASE
                WHEN component IN ('Checkpointer', 'Store') THEN 'LangGraph'
                ELSE component
            END AS owner,
            component,
            table_schema,
            table_name,
            estimated_rows,
            total_size
        FROM inventory
        ORDER BY owner, component, table_schema, table_name
        """,
        engine=postgres,
    )
    return (table_inventory,)


@app.cell
def _(mo, table_inventory):
    mo.stop(
        table_inventory.is_empty(),
        mo.md("No user tables found. Run `lgos-demo-api-setup` first."),
    )
    _tables = table_inventory.select("table_schema", "table_name").iter_rows()
    table_options = {f"{schema}.{table}": (schema, table) for schema, table in _tables}
    _default = next(
        (label for label in table_options if label.endswith(".checkpoints")),
        next(iter(table_options)),
    )
    table_selector = mo.ui.dropdown(
        options=table_options,
        value=_default,
        label="Table",
        searchable=True,
        full_width=True,
    )
    mo.vstack([table_selector])
    return (table_selector,)


@app.cell
def _(mo, postgres, sqlalchemy, table_selector):
    _schema, _table = table_selector.value
    _inspector = sqlalchemy.inspect(postgres)
    _primary_key = set(
        _inspector.get_pk_constraint(_table, schema=_schema)["constrained_columns"]
    )
    column_rows = [
        {
            "column": column["name"],
            "type": str(column["type"]),
            "nullable": column["nullable"],
            "primary_key": column["name"] in _primary_key,
            "default": column["default"],
        }
        for column in _inspector.get_columns(_table, schema=_schema)
    ]
    mo.vstack(
        [
            mo.md(f"## `{_schema}.{_table}` columns"),
            mo.ui.table(column_rows, pagination=False, selection=None),
        ]
    )
    return


@app.cell
def _(mo, postgres, table_selector):
    _schema, _table = table_selector.value
    _quote = postgres.dialect.identifier_preparer.quote
    _qualified_table = f"{_quote(_schema)}.{_quote(_table)}"
    _table_sample = mo.sql(
        f"""
        SELECT *
        FROM {_qualified_table}
        LIMIT 50
        """,
        engine=postgres,
    )
    return


if __name__ == "__main__":
    app.run()
