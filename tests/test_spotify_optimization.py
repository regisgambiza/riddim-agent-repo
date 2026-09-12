"""
Unit tests for Spotify Web API optimizations:
  - Persistent SQLite caching
  - Official batch album lookup (GET /v1/albums?ids=...)
  - Multi-type search (albums and tracks in 1 request)
  - Request reduction and session pooling
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import spotify_lookup
from spotify_lookup import (
    SpotifyCache,
    fetch_albums_batch,
    lookup_spotify,
    batch_lookup_spotify,
    _candidates_from_search,
    CACHE_STATS,
)


@pytest.fixture
def temp_cache():
    """Create a temporary SQLite cache for tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        cache_path = tmp.name
    cache = SpotifyCache(cache_path)
    old_cache = spotify_lookup._CACHE
    spotify_lookup._CACHE = cache
    yield cache
    spotify_lookup._CACHE = old_cache
    try:
        os.unlink(cache_path)
    except Exception:
        pass


def test_cache_search_set_and_get(temp_cache):
    query = 'album:"1 Vibe Riddim"'
    search_type = "album,track"
    limit = 5
    mock_data = {"albums": {"items": [{"id": "a1", "name": "1 Vibe"}]}}

    assert temp_cache.get_search(query, search_type, limit) is None
    temp_cache.set_search(query, search_type, limit, mock_data)
    hit = temp_cache.get_search(query, search_type, limit)
    assert hit is not None
    assert hit["albums"]["items"][0]["id"] == "a1"

    hit_case = temp_cache.get_search('  ALBUM:"1 Vibe Riddim"  ', search_type, limit)
    assert hit_case is not None


def test_cache_album_batch_set_and_get(temp_cache):
    albums = {
        "alb1": {"id": "alb1", "name": "Album One", "label": "Label 1"},
        "alb2": {"id": "alb2", "name": "Album Two", "label": "Label 2"},
    }
    temp_cache.set_albums(albums)

    res = temp_cache.get_albums(["alb1", "alb2", "alb3"])
    assert len(res) == 2
    assert res["alb1"]["name"] == "Album One"
    assert res["alb2"]["name"] == "Album Two"
    assert "alb3" not in res


def test_candidates_from_search_multi_type():
    results = {
        "albums": {
            "items": [
                {
                    "id": "alb_1",
                    "name": "Riddim Album",
                    "release_date": "2020-05-01",
                    "label": "Greensleeves",
                    "artists": [{"name": "Various Artists"}],
                    "external_urls": {"spotify": "https://spotify/album/1"},
                }
            ]
        },
        "tracks": {
            "items": [
                {
                    "id": "trk_1",
                    "name": "Riddim Track",
                    "album": {
                        "id": "alb_2",
                        "name": "Single Album",
                        "release_date": "2021-01-01",
                    },
                    "artists": [{"name": "Lead Artist"}],
                    "external_urls": {"spotify": "https://spotify/track/1"},
                }
            ]
        },
    }

    cands = _candidates_from_search(results, kind="album,track", query="test")
    assert len(cands) == 2
    assert cands[0].kind == "album"
    assert cands[0].spotify_id == "alb_1"
    assert cands[1].kind == "track"
    assert cands[1].spotify_id == "trk_1"
    assert cands[1].album_id == "alb_2"


def test_fetch_albums_batch_makes_single_request_for_multiple_ids(temp_cache):
    album_ids = ["id1", "id2", "id3"]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "albums": [
            {"id": "id1", "name": "Alb 1", "label": "L1", "tracks": {"items": [{"name": "T1"}]}},
            {"id": "id2", "name": "Alb 2", "label": "L2", "tracks": {"items": [{"name": "T2"}]}},
            {"id": "id3", "name": "Alb 3", "label": "L3", "tracks": {"items": [{"name": "T3"}]}},
        ]
    }

    with patch.object(spotify_lookup._get_session(), "get", return_value=mock_resp) as mock_get:
        albums = fetch_albums_batch("test_token", album_ids, use_cache=True)
        assert len(albums) == 3
        assert mock_get.call_count == 1
        call_args = mock_get.call_args[1]
        assert call_args["params"]["ids"] == "id1,id2,id3"

        cached_albums = fetch_albums_batch("test_token", album_ids, use_cache=True)
        assert len(cached_albums) == 3
        assert mock_get.call_count == 1


def test_fetch_albums_batch_chunks_over_20_ids(temp_cache):
    album_ids = [f"id_{i}" for i in range(25)]
    mock_resp1 = MagicMock()
    mock_resp1.status_code = 200
    mock_resp1.json.return_value = {
        "albums": [{"id": f"id_{i}", "name": f"Alb {i}"} for i in range(20)]
    }

    mock_resp2 = MagicMock()
    mock_resp2.status_code = 200
    mock_resp2.json.return_value = {
        "albums": [{"id": f"id_{i}", "name": f"Alb {i}"} for i in range(20, 25)]
    }

    with patch.object(spotify_lookup._get_session(), "get", side_effect=[mock_resp1, mock_resp2]) as mock_get:
        albums = fetch_albums_batch("test_token", album_ids, use_cache=False)
        assert len(albums) == 25
        assert mock_get.call_count == 2
        first_call = mock_get.call_args_list[0][1]["params"]["ids"]
        second_call = mock_get.call_args_list[1][1]["params"]["ids"]
        assert len(first_call.split(",")) == 20
        assert len(second_call.split(",")) == 5


def test_lookup_spotify_caches_and_saves_requests(temp_cache):
    mock_search_resp = MagicMock()
    mock_search_resp.status_code = 200
    mock_search_resp.json.return_value = {
        "albums": {
            "items": [
                {
                    "id": "alb_test",
                    "name": "1 Vibe Riddim",
                    "release_date": "2022-01-01",
                    "label": "Test Label",
                    "artists": [{"name": "Producer X"}],
                    "external_urls": {"spotify": "https://spotify/album/test"},
                }
            ]
        },
        "tracks": {"items": []},
    }

    with patch("spotify_lookup._get_token", return_value="fake_token"):
        with patch.object(spotify_lookup._get_session(), "get", return_value=mock_search_resp) as mock_get:
            res1 = lookup_spotify("1 Vibe Riddim", variant="album_name", use_cache=True)
            assert len(res1) == 1
            assert mock_get.call_count == 1

            res2 = lookup_spotify("1 Vibe Riddim", variant="album_name", use_cache=True)
            assert len(res2) == 1
            assert res1[0]["spotify_id"] == res2[0]["spotify_id"]
            assert mock_get.call_count == 1


def test_lookup_spotify_batches_track_album_enrichment(temp_cache):
    mock_search_resp = MagicMock()
    mock_search_resp.status_code = 200
    mock_search_resp.json.return_value = {
        "albums": {"items": []},
        "tracks": {
            "items": [
                {
                    "id": "trk_1",
                    "name": "Track One",
                    "album": {"id": "alb_shared", "name": "Shared Album", "release_date": "2020"},
                    "artists": [{"name": "Artist 1"}],
                    "external_urls": {"spotify": "https://spotify/track/1"},
                },
                {
                    "id": "trk_2",
                    "name": "Track Two",
                    "album": {"id": "alb_shared", "name": "Shared Album", "release_date": "2020"},
                    "artists": [{"name": "Artist 2"}],
                    "external_urls": {"spotify": "https://spotify/track/2"},
                },
            ]
        },
    }

    mock_album_resp = MagicMock()
    mock_album_resp.status_code = 200
    mock_album_resp.json.return_value = {
        "albums": [
            {
                "id": "alb_shared",
                "name": "Shared Album",
                "label": "Riddim Records",
                "tracks": {"items": [{"name": "Track One"}, {"name": "Track Two"}]},
            }
        ]
    }

    with patch("spotify_lookup._get_token", return_value="fake_token"):
        with patch.object(spotify_lookup._get_session(), "get", side_effect=[mock_search_resp, mock_album_resp]) as mock_get:
            cands = lookup_spotify("Riddim", variant="track_name", enrich_top_tracks=True, use_cache=False)
            assert len(cands) == 2
            assert mock_get.call_count == 2
            assert cands[0]["label"] == "Riddim Records"
            assert "Track One" in cands[0]["tracklist"]
