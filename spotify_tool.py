"""
Agent-facing wrapper around spotify_lookup.lookup_spotify.

Contract:
  * One call = one Spotify query.
  * Never raises on quota or auth failure -- returns a structured error dict
    so the agent can decide what to do next.
  * Returns candidates as plain dicts; makes no decision and writes nothing.
"""

from __future__ import annotations

from typing import Any

import requests

from spotify_lookup import (
    lookup_spotify,
    SpotifyAuthError,
    SpotifyQuotaExhausted,
)


def search_spotify(
    riddim_name: str,
    query_variant: str = "album_name",
    year: int | None = None,
    producer: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """
    Search Spotify for a release matching `riddim_name`, using ONE query variant.

    Returns on success:
        {
          "ok": True,
          "query_variant": str,
          "candidates": [ {...}, ... ],   # up to 5, best-first
          "count": int,
        }

    Returns on quota exhaustion (do NOT retry):
        {
          "ok": False,
          "error": "quota_exhausted",
          "message": "...",
          "guidance": "Stop calling search_spotify for this folder.",
        }

    Returns on auth failure (one retry is reasonable):
        {"ok": False, "error": "auth_expired", "message": "..."}

    Returns on other HTTP failure:
        {"ok": False, "error": "http_error", "message": "..."}
    """
    try:
        candidates = lookup_spotify(
            riddim_name,
            year=year,
            producer=producer,
            label=label,
            variant=query_variant,
            limit=5,
        )
    except SpotifyQuotaExhausted as e:
        return {
            "ok": False,
            "error": "quota_exhausted",
            "message": str(e),
            "guidance": (
                "Spotify's quota for this app is exhausted. Do not call "
                "search_spotify again during this session. Continue your "
                "investigation using only database tools and the evidence "
                "you already have."
            ),
        }
    except SpotifyAuthError as e:
        return {
            "ok": False,
            "error": "auth_expired",
            "message": str(e),
            "guidance": "One retry is reasonable; the token will refresh.",
        }
    except requests.RequestException as e:
        return {
            "ok": False,
            "error": "http_error",
            "message": f"{type(e).__name__}: {e}",
        }
    except ValueError as e:
        # Unknown query_variant, etc.
        return {
            "ok": False,
            "error": "bad_arguments",
            "message": str(e),
        }

    return {
        "ok": True,
        "query_variant": query_variant,
        "count": len(candidates),
        "candidates": candidates,
    }


# --------------------------------------------------------------------------
# Tool schema -- paste this into tools.TOOL_SCHEMAS
# --------------------------------------------------------------------------
SEARCH_SPOTIFY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_spotify",
        "description": (
            "Search Spotify for a release by name. Call this ONE query at a "
            "time: run a variant, inspect the returned candidates, then decide "
            "whether a different variant is worth trying or whether you already "
            "have enough evidence to move on. This is an external evidence "
            "source, not a match oracle -- its results corroborate or suggest "
            "the existence of a release, but a 'matched' decision must still "
            "name a database record.\n\n"
            "Query variants and when to use them:\n"
            "  * 'album_name'    -- default; the release exists as an album/EP.\n"
            "  * 'track_name'    -- the riddim only exists as loose singles.\n"
            "  * 'label_name'    -- a label is known; narrows false positives.\n"
            "  * 'producer_name' -- a producer is known; they appear as artist.\n"
            "  * 'bare'          -- last resort when the above return nothing.\n\n"
            "Each result includes name, album_name, year, label, artists, "
            "tracklist, url, and a rule-based similarity score.\n\n"
            "If the response contains {'ok': false, 'error': 'quota_exhausted'}, "
            "stop calling search_spotify for the rest of the session and "
            "proceed with the evidence you have. If error is 'auth_expired', "
            "one retry is reasonable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "riddim_name": {
                    "type": "string",
                    "description": "The riddim name to search for.",
                },
                "query_variant": {
                    "type": "string",
                    "enum": ["album_name", "track_name", "label_name",
                             "producer_name", "bare", "multi"],
                    "description": "Which search strategy to use for this call.",
                },
                "year": {
                    "type": ["integer", "null"],
                    "description": "Expected year, for scoring only.",
                },
                "producer": {
                    "type": ["string", "null"],
                    "description": "Expected producer, for scoring and the producer_name variant.",
                },
                "label": {
                    "type": ["string", "null"],
                    "description": "Expected label, for scoring and the label_name variant.",
                },
            },
            "required": ["riddim_name", "query_variant"],
        },
    },
}