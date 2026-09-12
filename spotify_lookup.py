"""
Shared Spotify lookup primitive.

One call = one query. Returns ranked candidates as plain dicts.
No database access, no file I/O, no decisions, no side effects.
Callers decide what the results mean and what to do with them.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import config

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
SPOTIFY_ALBUM_URL = "https://api.spotify.com/v1/albums/{id}"
SPOTIFY_ALBUMS_URL = "https://api.spotify.com/v1/albums"

TOKEN_OVERLAP_WEIGHT = 0.5
LABEL_MATCH_BONUS = 0.25
STOPWORDS = {"riddim", "rhythm", "the", "a", "an", "of", "and", "&", "vol", "volume"}

# Cache the token in-module so repeated calls don't re-auth.
_TOKEN_CACHE: dict[str, Any] = {"value": None, "expires_at": 0.0}

# HTTP Session singleton for connection reuse and Keep-Alive
_SESSION: requests.Session | None = None

# Metrics for monitoring request savings
CACHE_STATS: dict[str, int] = {
    "search_hits": 0,
    "search_misses": 0,
    "album_hits": 0,
    "album_misses": 0,
    "saved_requests": 0,
}


class SpotifyQuotaExhausted(RuntimeError):
    """Raised when Spotify's Retry-After is too long to be a normal rate limit."""


class SpotifyAuthError(RuntimeError):
    pass


def _verbose(msg: str) -> None:
    # Respect an env var so the agent can stay quiet in production.
    if os.environ.get("SPOTIFY_LOOKUP_VERBOSE", "0") == "1":
        print(f"[spotify_lookup] {msg}", file=sys.stderr, flush=True)


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
            respect_retry_after_header=False,
        )
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retries,
        )
        _SESSION.mount("https://", adapter)
        _SESSION.mount("http://", adapter)
    return _SESSION


# --------------------------------------------------------------------------
# Persistent SQLite Cache
# --------------------------------------------------------------------------
class SpotifyCache:
    """
    Thread-safe, process-safe SQLite cache for Spotify search and album lookups.
    Saves official API requests by persisting static catalog responses.
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = getattr(config, "SPOTIFY_CACHE_PATH", Path(__file__).resolve().parent / ".spotify_cache.db")
        self.db_path = Path(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_cache (
                        cache_key TEXT PRIMARY KEY,
                        response_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS album_cache (
                        album_id TEXT PRIMARY KEY,
                        album_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_search_created ON search_cache(created_at)")
        except Exception as e:
            _verbose(f"Cache init failed: {e}")

    @staticmethod
    def make_search_key(query: str, search_type: str, limit: int) -> str:
        return f"{search_type.strip().lower()}:{limit}:{query.strip().lower()}"

    def get_search(self, query: str, search_type: str, limit: int, max_age_seconds: float = 7 * 86400) -> dict | None:
        try:
            key = self.make_search_key(query, search_type, limit)
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT response_json, created_at FROM search_cache WHERE cache_key = ?",
                    (key,)
                ).fetchone()
                if row:
                    resp_json, created_at = row
                    if time.time() - created_at <= max_age_seconds:
                        return json.loads(resp_json)
        except Exception as e:
            _verbose(f"Cache get_search error: {e}")
        return None

    def set_search(self, query: str, search_type: str, limit: int, data: dict) -> None:
        try:
            key = self.make_search_key(query, search_type, limit)
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO search_cache (cache_key, response_json, created_at) VALUES (?, ?, ?)",
                    (key, json.dumps(data), time.time())
                )
        except Exception as e:
            _verbose(f"Cache set_search error: {e}")

    def get_album(self, album_id: str) -> dict | None:
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT album_json FROM album_cache WHERE album_id = ?",
                    (album_id,)
                ).fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            _verbose(f"Cache get_album error: {e}")
        return None

    def get_albums(self, album_ids: list[str]) -> dict[str, dict]:
        found: dict[str, dict] = {}
        if not album_ids:
            return found
        try:
            with self._get_conn() as conn:
                placeholders = ",".join("?" for _ in album_ids)
                rows = conn.execute(
                    f"SELECT album_id, album_json FROM album_cache WHERE album_id IN ({placeholders})",
                    album_ids
                ).fetchall()
                for aid, aj in rows:
                    found[aid] = json.loads(aj)
        except Exception as e:
            _verbose(f"Cache get_albums error: {e}")
        return found

    def set_albums(self, albums: dict[str, dict]) -> None:
        if not albums:
            return
        try:
            now = time.time()
            with self._get_conn() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO album_cache (album_id, album_json, created_at) VALUES (?, ?, ?)",
                    [(aid, json.dumps(adata), now) for aid, adata in albums.items()]
                )
        except Exception as e:
            _verbose(f"Cache set_albums error: {e}")

    def clear(self) -> None:
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM search_cache")
                conn.execute("DELETE FROM album_cache")
        except Exception as e:
            _verbose(f"Cache clear error: {e}")


_CACHE = SpotifyCache()


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def _get_token() -> str:
    now = time.time()
    if _TOKEN_CACHE["value"] and now < _TOKEN_CACHE["expires_at"] - 30:
        return _TOKEN_CACHE["value"]

    cid = config.SPOTIFY_CLIENT_ID
    csec = config.SPOTIFY_CLIENT_SECRET
    if not cid or not csec:
        raise SpotifyAuthError(
            "Missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET env vars."
        )

    creds = base64.b64encode(f"{cid}:{csec}".encode()).decode()
    session = _get_session()
    r = session.post(
        SPOTIFY_TOKEN_URL,
        headers={"Authorization": f"Basic {creds}"},
        data={"grant_type": "client_credentials"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    _TOKEN_CACHE["value"] = data["access_token"]
    _TOKEN_CACHE["expires_at"] = now + int(data.get("expires_in", 3600))
    _verbose("token refreshed")
    return _TOKEN_CACHE["value"]


# --------------------------------------------------------------------------
# Text helpers (kept identical in spirit to spotify_enrich.py)
# --------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t and t not in STOPWORDS]


def _normalize(text: str) -> str:
    return " ".join(_tokenize(text))


def _name_similarity(a: str, b: str) -> float:
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return seq
    inter = len(ta & tb)
    token_score = inter / max(len(ta), len(tb))
    return (1 - TOKEN_OVERLAP_WEIGHT) * seq + TOKEN_OVERLAP_WEIGHT * token_score


def _normalize_label(label: str | None) -> str:
    if not label:
        return ""
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _labels_match(db_label: str | None, spotify_label: str | None) -> bool:
    a, b = _normalize_label(db_label), _normalize_label(spotify_label)
    if not a or not b:
        return False
    if a == b:
        return True
    short_a = re.sub(r"(records|music|entertainment|group|inc|llc|ltd)$", "", a)
    short_b = re.sub(r"(records|music|entertainment|group|inc|llc|ltd)$", "", b)
    return bool(short_a) and (short_a == short_b or short_a in short_b or short_b in short_a)


# --------------------------------------------------------------------------
# Candidate record
# --------------------------------------------------------------------------
@dataclass
class _Cand:
    kind: str
    spotify_id: str
    url: str
    name: str
    album_name: str | None
    album_id: str | None
    year: int | None
    label: str | None
    artists: list[str] = field(default_factory=list)
    tracklist: list[str] = field(default_factory=list)
    name_similarity: float = 0.0
    year_diff: int = 99
    label_match: bool = False
    score: float = 0.0
    query: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "spotify_id": self.spotify_id,
            "url": self.url,
            "name": self.name,
            "album_name": self.album_name,
            "year": self.year,
            "label": self.label,
            "artists": self.artists,
            "tracklist": self.tracklist[:10],
            "name_similarity": round(self.name_similarity, 3),
            "year_diff": self.year_diff,
            "label_match": self.label_match,
            "score": round(self.score, 3),
            "query": self.query,
        }


# --------------------------------------------------------------------------
# Query building -- one variant per call, chosen by the caller
# --------------------------------------------------------------------------
VARIANTS = {
    "album_name":     lambda n, p, l: (f'album:"{n}"', "album,track"),
    "album_only":     lambda n, p, l: (f'album:"{n}"', "album"),
    "track_name":     lambda n, p, l: (f'track:"{n}"', "track"),
    "label_name":     lambda n, p, l: (f'label:"{l}" {n}', "album,track") if l else (f'album:"{n}"', "album,track"),
    "producer_name":  lambda n, p, l: (f'artist:"{p}" {n}', "album,track") if p else (f'album:"{n}"', "album,track"),
    "bare":           lambda n, p, l: (n, "album,track"),
    "album_and_track":lambda n, p, l: (n, "album,track"),
    "multi":          lambda n, p, l: (n, "album,track"),
}


def _build_query(riddim_name: str, producer: str | None, label: str | None,
                 variant: str) -> tuple[str, str]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown query_variant: {variant!r}. "
                         f"Valid: {sorted(VARIANTS)}")
    return VARIANTS[variant](riddim_name, producer, label)


# --------------------------------------------------------------------------
# Spotify HTTP & Batch Operations
# --------------------------------------------------------------------------
def _search(
    token: str,
    query: str,
    search_type: str,
    limit: int,
    *,
    use_cache: bool = True,
) -> dict:
    if use_cache:
        cached = _CACHE.get_search(query, search_type, limit)
        if cached is not None:
            _verbose(f"search cache hit: type={search_type} q={query!r}")
            CACHE_STATS["search_hits"] += 1
            CACHE_STATS["saved_requests"] += 1
            return cached

    CACHE_STATS["search_misses"] += 1
    _verbose(f"search type={search_type} q={query!r}")
    session = _get_session()
    resp = session.get(
        SPOTIFY_SEARCH_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={"q": query, "type": search_type, "limit": limit},
        timeout=15,
    )
    if resp.status_code == 401:
        # Force token refresh on next call.
        _TOKEN_CACHE["value"] = None
        raise SpotifyAuthError("Spotify token expired (401). Retry the call.")
    if resp.status_code == 429:
        try:
            reason = resp.json().get("error", {}).get("reason", "")
        except Exception:
            reason = ""
        wait = int(resp.headers.get("Retry-After", "30"))
        if wait > 300:
            raise SpotifyQuotaExhausted(
                f"Spotify says retry in {wait}s ({reason or 'rate limit'}). "
                f"Quota exhausted; do not retry in this session."
            )
        _verbose(f"429 rate-limit ({reason or 'rate limit'}); sleeping {wait}s")
        time.sleep(wait + 1)
        return _search(token, query, search_type, limit, use_cache=use_cache)
    resp.raise_for_status()
    data = resp.json()

    if use_cache:
        _CACHE.set_search(query, search_type, limit, data)

    return data


def fetch_albums_batch(
    token: str,
    album_ids: list[str],
    *,
    use_cache: bool = True,
) -> dict[str, dict]:
    """
    Fetch full album metadata for multiple albums in batch using Spotify's official
    batch endpoint: GET /v1/albums?ids={ids} (up to 20 per request).

    Checks the persistent cache first to save requests.
    Returns:
        dict mapping album_id -> album_object
    """
    if not album_ids:
        return {}

    # Normalize and deduplicate IDs
    unique_ids = []
    seen = set()
    for aid in album_ids:
        if aid and aid not in seen:
            seen.add(aid)
            unique_ids.append(aid)

    out: dict[str, dict] = {}
    uncached: list[str] = []

    if use_cache:
        cached = _CACHE.get_albums(unique_ids)
        out.update(cached)
        CACHE_STATS["album_hits"] += len(cached)
        CACHE_STATS["saved_requests"] += len(cached)
        uncached = [aid for aid in unique_ids if aid not in cached]
    else:
        uncached = unique_ids

    if not uncached:
        return out

    CACHE_STATS["album_misses"] += len(uncached)
    session = _get_session()

    # Spotify limits to 20 IDs per batch request
    for i in range(0, len(uncached), 20):
        chunk = uncached[i:i + 20]
        ids_param = ",".join(chunk)
        _verbose(f"batch fetching {len(chunk)} albums: {ids_param}")

        resp = session.get(
            SPOTIFY_ALBUMS_URL,
            headers={"Authorization": f"Bearer {token}"},
            params={"ids": ids_param},
            timeout=15,
        )

        if resp.status_code == 401:
            _TOKEN_CACHE["value"] = None
            raise SpotifyAuthError("Spotify token expired (401). Retry the call.")
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "30"))
            if wait > 300:
                raise SpotifyQuotaExhausted(
                    f"Spotify says retry in {wait}s. Quota exhausted; do not retry."
                )
            _verbose(f"429 rate-limit on batch albums; sleeping {wait}s")
            time.sleep(wait + 1)
            # Retry this chunk
            sub = fetch_albums_batch(token, chunk, use_cache=False)
            out.update(sub)
            continue

        resp.raise_for_status()
        data = resp.json()
        new_albums: dict[str, dict] = {}
        for album in data.get("albums", []) or []:
            if album and isinstance(album, dict) and album.get("id"):
                aid = album["id"]
                new_albums[aid] = album
                out[aid] = album

        if use_cache and new_albums:
            _CACHE.set_albums(new_albums)

    return out


def _fetch_album(token: str, album_id: str, *, use_cache: bool = True) -> dict:
    """Fetch single album by delegating to fetch_albums_batch."""
    albums = fetch_albums_batch(token, [album_id], use_cache=use_cache)
    return albums.get(album_id, {})


def _candidates_from_search(results: dict, kind: str, query: str) -> list[_Cand]:
    out: list[_Cand] = []
    include_albums = "album" in kind or ("albums" in results and "track" not in kind)
    include_tracks = "track" in kind or ("tracks" in results and "album" not in kind)

    if include_albums:
        for item in (results.get("albums", {}) or {}).get("items", []) or []:
            if not item:
                continue
            year = None
            rd = item.get("release_date") or ""
            if rd:
                try:
                    year = int(rd[:4])
                except ValueError:
                    pass
            out.append(_Cand(
                kind="album",
                spotify_id=item.get("id"),
                url=item.get("external_urls", {}).get("spotify"),
                name=item.get("name", ""),
                album_name=item.get("name", ""),
                album_id=item.get("id"),
                year=year,
                label=item.get("label"),
                artists=[a["name"] for a in item.get("artists", []) if a and "name" in a],
                query=query,
            ))

    if include_tracks:
        for item in (results.get("tracks", {}) or {}).get("items", []) or []:
            if not item:
                continue
            year = None
            album_obj = item.get("album") or {}
            rd = album_obj.get("release_date") or ""
            if rd:
                try:
                    year = int(rd[:4])
                except ValueError:
                    pass
            out.append(_Cand(
                kind="track",
                spotify_id=item.get("id"),
                url=item.get("external_urls", {}).get("spotify"),
                name=item.get("name", ""),
                album_name=album_obj.get("name"),
                album_id=album_obj.get("id"),
                year=year,
                label=None,
                artists=[a["name"] for a in item.get("artists", []) if a and "name" in a],
                query=query,
            ))

    return out


def _score(c: _Cand, target_name: str, target_year: int | None,
           db_label: str | None, producer: str | None) -> _Cand:
    c.name_similarity = _name_similarity(target_name, c.name)
    if c.kind == "track" and c.album_name:
        c.name_similarity = max(
            c.name_similarity, _name_similarity(target_name, c.album_name)
        )
    if c.year and target_year:
        c.year_diff = abs(c.year - target_year)
    else:
        c.year_diff = 99
    c.label_match = _labels_match(db_label, c.label)

    producer_hit = False
    if producer:
        p_norm = _normalize(producer)
        for a in c.artists:
            if p_norm and p_norm in _normalize(a):
                producer_hit = True
                break

    s = c.name_similarity
    s -= 0.03 * min(c.year_diff, 10)
    if c.label_match:
        s += LABEL_MATCH_BONUS
    if producer_hit:
        s += 0.10
    if c.kind == "track":
        s -= 0.02
    c.score = s
    return c


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def lookup_spotify(
    riddim_name: str,
    *,
    year: int | None = None,
    producer: str | None = None,
    label: str | None = None,
    variant: str = "album_name",
    limit: int = 5,
    enrich_top_tracks: bool = True,
    use_cache: bool = True,
) -> list[dict]:
    """
    Run ONE Spotify search and return ranked candidates as dicts.

    Optimized using official Spotify Web API capabilities:
      * Persistent SQLite response caching (avoids duplicate requests)
      * Multi-type search (album,track in 1 query)
      * Official batch album fetching (GET /v1/albums?ids=...) in 1 call instead of N

    Args:
        riddim_name:  the name to search for.
        year:         expected year (used for scoring only).
        producer:     expected producer (used for scoring + producer_name variant).
        label:        expected label (used for scoring + label_name variant).
        variant:      one of VARIANTS: album_name, track_name, label_name,
                      producer_name, bare, album_only, multi.
        limit:        max candidates to return.
        enrich_top_tracks: fetch album metadata for top track candidates so
                      label and tracklist are populated.
        use_cache:    whether to check/update the persistent SQLite cache.

    Returns:
        list of candidate dicts, sorted best-first. Empty list if nothing found.

    Raises:
        SpotifyQuotaExhausted: Spotify's daily quota is gone. Do not retry.
        SpotifyAuthError:      credentials missing or token just expired;
                               one retry after a short pause is reasonable.
        requests.RequestException: network/HTTP failure unrelated to the above.
    """
    query, search_type = _build_query(riddim_name, producer, label, variant)

    token = _get_token()
    results = _search(token, query, search_type, limit=max(limit, 5), use_cache=use_cache)
    cands = _candidates_from_search(results, search_type, query)

    # De-dup by spotify_id (search sometimes returns dupes across result pages).
    seen, unique = set(), []
    for c in cands:
        if c.spotify_id and c.spotify_id not in seen:
            seen.add(c.spotify_id)
            unique.append(c)

    scored = [
        _score(c, riddim_name, year, label, producer)
        for c in unique
    ]
    scored.sort(key=lambda c: c.score, reverse=True)

    if enrich_top_tracks:
        needed_album_ids = [
            c.album_id for c in scored[:3]
            if c.kind == "track" and c.label is None and c.album_id
        ]
        if needed_album_ids:
            try:
                albums_by_id = fetch_albums_batch(token, needed_album_ids, use_cache=use_cache)
                for c in scored[:3]:
                    if c.kind == "track" and c.album_id and c.album_id in albums_by_id:
                        album = albums_by_id[c.album_id]
                        c.label = album.get("label") or c.label
                        c.tracklist = [
                            t.get("name", "")
                            for t in (album.get("tracks") or {}).get("items", [])
                            if t and t.get("name")
                        ]
            except requests.RequestException as e:
                _verbose(f"batch album enrich failed: {e}")

    return [c.to_dict() for c in scored[:limit]]


def batch_lookup_spotify(
    records: list[dict[str, Any]],
    *,
    variant: str | None = None,
    limit: int = 5,
    enrich_top_tracks: bool = True,
    use_cache: bool = True,
) -> list[list[dict]]:
    """
    Perform batch search and album enrichment for multiple riddim records.
    Collects all track candidates needing album enrichment and fetches them
    in batches of 20 via GET /v1/albums?ids=... to minimize total requests.

    Args:
        records: list of dicts, each having 'riddim_name', optional 'year', 'producer', 'label', 'variant'.
        variant: default query variant to use if not specified per record.
        limit: max candidates per record.
        enrich_top_tracks: whether to enrich track candidates with album metadata.
        use_cache: whether to use the persistent SQLite cache.

    Returns:
        list of candidate lists, one per input record.
    """
    if not records:
        return []

    token = _get_token()
    all_scored: list[list[_Cand]] = []
    all_needed_album_ids: list[str] = []

    # Step 1: Run searches (leveraging cache)
    for rec in records:
        name = rec["riddim_name"]
        year = rec.get("year")
        producer = rec.get("producer")
        label = rec.get("label")
        var = rec.get("variant") or variant or "album_name"

        query, search_type = _build_query(name, producer, label, var)
        results = _search(token, query, search_type, limit=max(limit, 5), use_cache=use_cache)
        cands = _candidates_from_search(results, search_type, query)

        seen = set()
        unique = []
        for c in cands:
            if c.spotify_id and c.spotify_id not in seen:
                seen.add(c.spotify_id)
                unique.append(c)

        scored = [_score(c, name, year, label, producer) for c in unique]
        scored.sort(key=lambda c: c.score, reverse=True)
        all_scored.append(scored)

        if enrich_top_tracks:
            for c in scored[:3]:
                if c.kind == "track" and c.label is None and c.album_id:
                    all_needed_album_ids.append(c.album_id)

    # Step 2: Batch fetch all needed albums in chunks of 20
    if enrich_top_tracks and all_needed_album_ids:
        try:
            albums_by_id = fetch_albums_batch(token, all_needed_album_ids, use_cache=use_cache)
            for scored in all_scored:
                for c in scored[:3]:
                    if c.kind == "track" and c.album_id and c.album_id in albums_by_id:
                        album = albums_by_id[c.album_id]
                        c.label = album.get("label") or c.label
                        c.tracklist = [
                            t.get("name", "")
                            for t in (album.get("tracks") or {}).get("items", [])
                            if t and t.get("name")
                        ]
        except requests.RequestException as e:
            _verbose(f"cross-record batch album enrich failed: {e}")

    return [[c.to_dict() for c in scored[:limit]] for scored in all_scored]