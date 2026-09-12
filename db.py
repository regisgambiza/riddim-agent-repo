"""
Read-only SQLite access layer for the riddims database.

Deliberately exposes no write path of any kind: only SELECT statements
appear in this module, and the connection is opened in a mode that
does not create the file if missing (so a typo'd path fails loudly
instead of silently creating an empty DB).
"""

import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import config


@contextmanager
def _connect():
    """Read-only connection. Raises if DB_PATH does not exist."""
    db_path = Path(config.DB_PATH)
    if not db_path.exists():
        raise FileNotFoundError(f"riddims database not found at {db_path}")
    # uri=True + mode=ro enforces read-only at the SQLite driver level,
    # independent of any application-level discipline.
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _table_columns(conn, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row["name"] for row in cur.fetchall()]


def _get_normalized_name_from_db(conn) -> Any:
    """Create a computed column for normalized riddim names."""
    return conn.execute(
        """
        SELECT 
            id, riddim_name, year, label,
            regexp_replace(lower(replace(riddim_name, '_', ' ')), '[^a-z0-9]+', ' ') AS riddim_name_normalized
        FROM riddims
        """
    )


# Fields in the riddims table that are relevant to matching and safe to
# expose to the agent. Any column not in this allow-list (e.g. future
# internal/bookkeeping columns) is excluded from get_full_record.
_MATCHING_RELEVANT_FIELDS = {
    "id",
    "riddim_name",
    "year",
    "label",
    "producer",
    "genre",
    "spotify_id",
    "spotify_url",
    "source",
    "source_url",
    "match_status",
    "notes",
    # The schema currently has no artist / aliases / tracks columns.
    # If a future DB revision adds them, listing the column names here
    # (matching PRAGMA table_info output) is sufficient --
    # get_full_record() below intersects this set with the columns that
    # actually exist, so no code changes are required elsewhere.
    "artist",
    "aliases",
    "tracks",
}


def normalize(name: str) -> str:
    """
    Case/punctuation/whitespace/underscore-insensitive normalization.
    Used as the basis for the 'normalized exact match' strategy and as
    a shared building block for the fuzzy/token strategies.
    """
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = s.lower()
    s = s.replace("_", " ")
    # Drop bracket/paren-wrapped year-ish qualifiers only at the token
    # level is handled elsewhere (matching.py); here we just strip all
    # non-alphanumeric characters down to single spaces.
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def list_all_riddims() -> list[dict[str, Any]]:
    """All rows, minimal fields, for use by matching strategies."""
    with _connect() as conn:
        cols = _table_columns(conn, "riddims")
        cur = conn.execute(f"SELECT {', '.join(cols)} FROM riddims")
        return [dict(row) for row in cur.fetchall()]


def get_record(candidate_id: int) -> dict[str, Any] | None:
    """
    Fetch a single riddim record by id, restricted to matching-relevant
    fields. Returns None if no such id exists.
    """
    with _connect() as conn:
        existing_cols = set(_table_columns(conn, "riddims"))
        select_cols = sorted(existing_cols & _MATCHING_RELEVANT_FIELDS)
        if not select_cols:
            select_cols = ["id"]
        cur = conn.execute(
            f"SELECT {', '.join(select_cols)} FROM riddims WHERE id = ?",
            (candidate_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
