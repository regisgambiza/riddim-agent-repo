"""
Unit tests for the Spotify batch enrichment script.
These tests use mock data and a temporary SQLite database to verify
the enrichment logic without requiring Spotify credentials.
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from db_write import (
    get_unverified_batch,
    count_unverified,
    enrich_riddim,
    upsert_tracks,
    add_alt_name,
)
from db import get_record, normalize
from enrich_spotify import _process_one, _choose_variant, _error_result
from spotify_lookup import SpotifyAuthError, SpotifyQuotaExhausted


@pytest.fixture(autouse=True)
def _temp_db(monkeypatch):
    """Create a temporary SQLite DB with the required schema and point config at it."""
    # Create a temporary file
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name

    # Set config.DB_PATH before any imports that reference it.
    # We monkeypatch after import to ensure the temporary file is used.
    monkeypatch.setattr(config, "DB_PATH", db_path)

    # Create the tables that db_write.py expects.
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS riddims (
            id INTEGER PRIMARY KEY,
            riddim_name TEXT NOT NULL,
            year INTEGER,
            producer TEXT,
            label TEXT,
            genre TEXT,
            spotify_id TEXT,
            spotify_url TEXT,
            source TEXT,
            source_url TEXT,
            match_status TEXT DEFAULT 'UNVERIFIED',
            notes TEXT,
            riddim_name_normalized TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            riddim_id INTEGER,
            track_name TEXT,
            artist TEXT,
            version TEXT,
            year INTEGER,
            spotify_id TEXT,
            spotify_url TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alt_names (
            id INTEGER PRIMARY KEY,
            riddim_id INTEGER,
            alt_name TEXT
        )
    """
    )
    conn.commit()
    conn.close()

    yield

    # Cleanup
    try:
        os.unlink(db_path)
    except Exception:
        pass


def _insert_test_riddim(conn, riddim_id, name, year=None, producer=None, label=None):
    """Helper to insert a test riddim row."""
    norm = normalize(name)
    conn.execute(
        """
        INSERT INTO riddims (id, riddim_name, year, producer, label, genre,
                            spotify_id, spotify_url, source, source_url,
                            match_status, notes, riddim_name_normalized)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            riddim_id,
            name,
            year,
            producer,
            label,
            None,  # genre
            None,  # spotify_id
            None,  # spotify_url
            None,  # source
            None,  # source_url
            "UNVERIFIED",
            None,
            norm,
        ),
    )
    conn.commit()


def test_choose_variant():
    assert _choose_variant({"producer": "some producer"}) == "producer_name"
    assert _choose_variant({"label": "some label"}) == "label_name"
    assert _choose_variant({}) == "album_name"
    assert _choose_variant({"producer": "", "label": ""}) == "album_name"


def test_process_one_no_candidates(_temp_db):
    # Insert a test riddim
    conn = sqlite3.connect(config.DB_PATH)
    _insert_test_riddim(conn, 1, "Test Riddim", year=2020)
    conn.close()

    # Mock lookup_spotify to return empty list
    with patch("enrich_spotify.lookup_spotify", return_value=[]):
        result = _process_one({"id": 1, "riddim_name": "Test Riddim", "year": 2020, "producer": None, "label": None})

    assert result["status"] == "not_found"
    assert result["id"] == 1

    # Verify DB updated to NOT FOUND
    row = get_record(1)
    assert row["match_status"] == "NOT FOUND"
    assert row["source"] == "spotify"
    assert row["notes"] == "No Spotify candidates found"


def test_process_one_with_confident_match(_temp_db):
    conn = sqlite3.connect(config.DB_PATH)
    _insert_test_riddim(conn, 2, "Real rock Riddim", year=1990)
    conn.close()

    # Mock a high-scoring candidate
    mock_candidate = {
        "kind": "album",
        "spotify_id": "spotify:album:1",
        "url": "https://open.spotify.com/album/1",
        "name": "Real rock Riddim Album",
        "album_name": "Real rock Riddim",
        "year": 1990,
        "label": "Test Label",
        "artists": ["Test Artist"],
        "tracklist": ["Track 1", "Track 2"],
        "score": 0.95,
    }
    with patch("enrich_spotify.lookup_spotify", return_value=[mock_candidate]):
        result = _process_one({"id": 2, "riddim_name": "Real rock Riddim", "year": 1990, "producer": None, "label": None})

    assert result["status"] == "enriched"
    assert result["score"] == 0.95
    assert result["best_candidate"]["spotify_id"] == "spotify:album:1"

    # Verify DB updates
    row = get_record(2)
    assert row["match_status"] == "CONFIRMED"
    assert row["source"] == "spotify"
    assert row["spotify_id"] == "spotify:album:1"
    assert row["spotify_url"] == "https://open.spotify.com/album/1"
    # Producer and label should be updated because they were None
    assert row["producer"] == "Test Artist"
    assert row["label"] == "Test Label"
    # Notes should contain score and variant
    assert "score=0.95" in row["notes"]
    assert "query_variant=album_name" in row["notes"]

    # Verify tracks table
    conn = sqlite3.connect(config.DB_PATH)
    track_rows = conn.execute("SELECT track_name, artist FROM tracks WHERE riddim_id = 2 ORDER BY track_name").fetchall()
    conn.close()
    assert len(track_rows) == 2
    assert track_rows[0][0] == "Track 1"
    assert track_rows[0][1] == "Test Artist"
    assert track_rows[1][0] == "Track 2"
    assert track_rows[1][1] == "Test Artist"

    # Alt name added if different
    conn = sqlite3.connect(config.DB_PATH)
    alt_rows = conn.execute("SELECT alt_name FROM alt_names WHERE riddim_id = 2").fetchall()
    conn.close()
    assert len(alt_rows) == 1
    assert alt_rows[0][0] == "Real rock Riddim Album"


def test_process_one_low_score_review(_temp_db):
    conn = sqlite3.connect(config.DB_PATH)
    _insert_test_riddim(conn, 3, "Obscure Riddim", year=2021)
    conn.close()

    # Mock a low-scoring candidate
    mock_candidate = {
        "kind": "album",
        "spotify_id": "spotify:album:2",
        "url": "https://open.spotify.com/album/2",
        "name": "Somewhat Similar",
        "album_name": "Somewhat Similar Album",
        "year": 2020,
        "label": "Other Label",
        "artists": ["Other Artist"],
        "tracklist": [],
        "score": 0.5,  # below threshold
    }
    with patch("enrich_spotify.lookup_spotify", return_value=[mock_candidate]):
        result = _process_one({"id": 3, "riddim_name": "Obscure Riddim", "year": 2021, "producer": None, "label": None})

    assert result["status"] == "review"
    assert result["score"] == 0.5

    row = get_record(3)
    assert row["match_status"] == "REVIEW"
    assert row["source"] == "spotify"
    assert "low score (0.500)" in row["notes"]


def test_process_one_auth_error_retry(_temp_db):
    conn = sqlite3.connect(config.DB_PATH)
    _insert_test_riddim(conn, 4, "Auth Error Riddim", year=2022)
    conn.close()

    call_count = 0

    def mock_lookup(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise SpotifyAuthError("test auth error")
        # Return empty on second call
        return []

    with patch("enrich_spotify.lookup_spotify", side_effect=mock_lookup):
        result = _process_one({"id": 4, "riddim_name": "Auth Error Riddim", "year": 2022, "producer": None, "label": None})

    assert call_count == 2
    assert result["status"] == "not_found"  # after auth error retry, got empty candidates -> NOT FOUND
    row = get_record(4)
    assert row["match_status"] == "NOT FOUND"


def test_quota_exhausted_stops():
    # This test verifies that _process_one re-raises SpotifyQuotaExhausted
    # so the caller (run_enrichment) can wait and retry instead of stopping.
    with patch("enrich_spotify.lookup_spotify", side_effect=SpotifyQuotaExhausted("quota exhausted")):
        with pytest.raises(SpotifyQuotaExhausted):
            _process_one({"id": 99, "riddim_name": "Test", "year": None, "producer": None, "label": None})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])