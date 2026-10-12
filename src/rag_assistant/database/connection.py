import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import Connection

load_dotenv()


def get_database_url() -> str:
    """Return the configured PostgreSQL connection URL."""

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not configured."
        )

    return database_url


@contextmanager
def get_connection() -> Iterator[Connection]:
    """Open a PostgreSQL connection with pgvector support."""

    with psycopg.connect(
        get_database_url()
    ) as connection:
        register_vector(connection)

        yield connection