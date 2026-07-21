"""Versioned PostgreSQL migrations for Chainlit conversation persistence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, LiteralString, cast

from psycopg import AsyncConnection
from psycopg.rows import dict_row

MIGRATIONS_DIR = Path(__file__).with_name("migrations")
MIGRATIONS_TABLE = "_lgos_chainlit_schema_migrations"
MIGRATION_LOCK_ID = 0x4C474F53434C4442


class ChainlitMigrationError(RuntimeError):
    """Raised when an applied Chainlit migration no longer matches its source."""


@dataclass(frozen=True)
class Migration:
    """A checksum-protected SQL migration."""

    version: str
    checksum: str
    sql: str


def load_migrations(directory: Path = MIGRATIONS_DIR) -> tuple[Migration, ...]:
    """Load the bundled SQL migrations in version order."""
    migrations = []
    for path in sorted(directory.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                version=path.stem,
                checksum=sha256(sql.encode()).hexdigest(),
                sql=sql,
            )
        )
    if not migrations:
        raise ChainlitMigrationError(f"No Chainlit migrations found in {directory}.")
    return tuple(migrations)


async def apply_migrations(
    connection: AsyncConnection[dict[str, Any]],
    migrations: tuple[Migration, ...] | None = None,
) -> None:
    """Apply each pending migration atomically and reject migration drift."""
    await connection.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
    try:
        await connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
                version TEXT PRIMARY KEY,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor = await connection.execute(
            f"SELECT version, checksum FROM {MIGRATIONS_TABLE}"
        )
        applied = {row["version"]: row["checksum"] for row in await cursor.fetchall()}
        selected_migrations = load_migrations() if migrations is None else migrations
        known_versions = {migration.version for migration in selected_migrations}
        unknown_versions = sorted(set(applied) - known_versions)
        if unknown_versions:
            joined_versions = ", ".join(unknown_versions)
            raise ChainlitMigrationError(
                "Database contains Chainlit migrations unknown to this release: "
                f"{joined_versions}."
            )

        for migration in selected_migrations:
            applied_checksum = applied.get(migration.version)
            if applied_checksum == migration.checksum:
                continue
            if applied_checksum is not None:
                raise ChainlitMigrationError(
                    f"Applied Chainlit migration {migration.version!r} has changed."
                )

            async with connection.transaction():
                trusted_sql = cast(LiteralString, migration.sql)
                await connection.execute(trusted_sql, prepare=False)
                await connection.execute(
                    f"""
                    INSERT INTO {MIGRATIONS_TABLE} (version, checksum)
                    VALUES (%s, %s)
                    """,
                    (migration.version, migration.checksum),
                )
    finally:
        await connection.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_ID,))


async def setup_chainlit_schema(database_url: str) -> None:
    """Initialize or migrate the Chainlit schema as a deployment task."""
    connection = cast(
        "AsyncConnection[dict[str, Any]]",
        await AsyncConnection.connect(
            database_url,
            autocommit=True,
            row_factory=cast(Any, dict_row),
        ),
    )
    async with connection:
        await apply_migrations(connection)
