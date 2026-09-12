"""
Write path for enriching riddim records with Spotify metadata.

Unlike db.py (which is strictly read-only), this module performs UPDATE and
INSERT statements on the enriched riddims table (riddims_multi_year_new.db).
All writes are confined to specific columns: spotify_id, spotify_url,
producer, label, genre, source, source_url, match_status, notes.

The primary key 'id' and identifying fields (riddim_name, year, label,
riddim_name_normalized) are never modified -- enrichment only fills in
the metadata gaps.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import config
from db import normalize


def _connect() -> sqlite3.Connection:
    """Writable connection to the enriched riddims database."""
    db_path = Path(config.DB_PATH)
    if not db_path.exists():
        raise FileNotFoundError(f"riddims database not found at {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

_ENRICHABLE_COLUMNS = {
    "spotify_id", "spotify_url", "producer", "label", "genre",
    "source", "source_url", "match_status", "notes",
}


_VALID_MATCH_STATUS = {"UNVERIFIED", "CONFIRMED", "REVIEW", "NOT FOUND"}


def enrich_riddim(
    riddim_id: int,
    *,
    spotify_id: str | None = None,
    spotify_url: str | None = None,
    producer: str | None = None,
    label: str | None = None,
    genre: str | None = None,
    source: str | None = None,
    source_url: str | None = None,
    match_status: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Update enrichment fields for one riddim by id.

    Only non-None keyword arguments are written; existing values are
    preserved when a parameter is omitted (None). Returns the updated row.
    """
    if match_status is not None and match_status not in _VALID_MATCH_STATUS:
        raise ValueError(f"match_status must be one of {_VALID_MATCH_STATUS}")

    updates: dict[str, Any] = {}
    for col, val in {
        "spotify_id": spotify_id, "spotify_url": spotify_url,
        "producer": producer, "label": label, "genre": genre,
        "source": source, "source_url": source_url,
        "match_status": match_status, "notes": notes,
    }.items():
        if val is not None:
            updates[col] = val

    if not updates:
        # Nothing to write -- just return the current row.
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM riddims WHERE id = ?", (riddim_id,)
            ).fetchone()
            return dict(row) if row else {"id": riddim_id, "not_found": True}

    set_clause = ", ".join(f"{col} = ?" for col in updates)
    params = list(updates.values()) + [riddim_id]

    with _connect() as conn:
        conn.execute(
            f"UPDATE riddims SET {set_clause} WHERE id = ?",
            params,
        )
        row = conn.execute(
            "SELECT * FROM riddims WHERE id = ?", (riddim_id,)
        ).fetchone()
        return dict(row) if row else {"id": riddim_id, "not_found": True}


def upsert_tracks(riddim_id: int, tracks: list[dict[str, Any]]) -> int:
    """Replace all tracks for a riddim, then insert the new set.

    Each dict in `tracks` should have keys: track_name, artist, version,
    year, spotify_id, spotify_url.

    Returns the number of tracks inserted.
    """
    if not isinstance(tracks, list):
        raise TypeError("tracks must be a list of dicts")

    normalized_name = None
    with _connect() as conn:
        row = conn.execute(
            "SELECT riddim_name FROM riddims WHERE id = ?", (riddim_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"riddim_id={riddim_id} not found")
        normalized_name = normalize(row["riddim_name"]) if row else None

    with _connect() as conn:
        conn.execute("DELETE FROM tracks WHERE riddim_id = ?", (riddim_id,))
        for t in tracks:
            conn.execute(
                """
                INSERT INTO tracks (
                    riddim_id, track_name, artist, version, year,
                    spotify_id, spotify_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    riddim_id,
                    t.get("track_name", ""),
                    t.get("artist"),
                    t.get("version"),
                    t.get("year"),
                    t.get("spotify_id"),
                    t.get("spotify_url"),
                ),
            )
        return len(tracks)


def add_alt_name(riddim_id: int, alt_name: str) -> dict[str, Any]:
    """Insert an alternative name for a riddim if it doesn't already exist."""
    if not alt_name or not alt_name.strip():
        raise ValueError("alt_name must not be empty")

    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM alt_names WHERE riddim_id = ? AND alt_name = ?",
            (riddim_id, alt_name.strip()),
        ).fetchone()
        if existing:
            return {"id": existing["id"], "inserted": False}

        cur = conn.execute(
            "INSERT INTO alt_names (riddim_id, alt_name) VALUES (?, ?)",
            (riddim_id, alt_name.strip()),
        )
        return {
            "id": cur.lastrowid,
            "inserted": True,
            "riddim_id": riddim_id,
            "alt_name": alt_name.strip(),
        }


def get_unverified_batch(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    """Fetch a batch of UNVERIFIED riddims for enrichment processing."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, riddim_name, year, producer, label, genre,
                   spotify_id, spotify_url, source, source_url,
                   match_status, notes
            FROM riddims
            WHERE match_status = 'UNVERIFIED'
            ORDER BY id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def count_unverified() -> int:
    """Return the total number of UNVERIFIED riddims."""
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM riddims WHERE match_status = 'UNVERIFIED'"
        ).fetchone()[0]


def get_by_id(riddim_id: int) -> dict[str, Any] | None:
    """Fetch a full riddim record by id (writable connection context)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM riddims WHERE id = ?", (riddim_id,)
        ).fetchone()
        return dict(row) if row else None


def search_by_name(name: str) -> dict[str, Any] | None:
    """Find a riddim by exact normalized name match (writable context)."""
    norm = normalize(name)
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM riddims WHERE riddim_name_normalized = ?", (norm,)
        ).fetchone()
        return dict(row) if row else None