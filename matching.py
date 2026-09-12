"""
Deterministic matching strategies used by get_candidates(), compare_names()
and compare_track_evidence(). Nothing in this file calls an LLM or makes a
judgment call about which candidate is "right" -- it only computes scores
and structured differences for the agent to interpret.
"""

import difflib
import re
from typing import Any

from db import normalize, _connect, _table_columns


# Year-like token, e.g. 2022, (2022), - 2022
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")

# Year-like token, e.g. 2022, (2022), - 2022
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def _tokens(name: str) -> set[str]:
    return set(normalize(name).split())


def fuzzy_ratio(a: str, b: str) -> float:
    """Character-level similarity in [0, 1] via difflib's SequenceMatcher."""
    return round(difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio(), 4)


def token_overlap(a: str, b: str) -> float:
    """
    Overlap coefficient over normalized word sets: intersection size
    divided by the smaller set's size. Robust to word-order changes and
    extra qualifier words (e.g. "(2022)", "V2").
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    smaller = min(len(ta), len(tb))
    return round(len(inter) / smaller, 4)


def normalized_exact(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)


def alias_match(folder_name: str, record: dict[str, Any]) -> bool:
    """
    Checks folder_name against any alias field the record might carry.
    The current DB schema has no aliases column, so this deterministically
    returns False until/unless such a column exists -- it never guesses.
    """
    aliases = record.get("aliases")
    if not aliases:
        return False
    if isinstance(aliases, str):
        alias_list = [x.strip() for x in re.split(r"[;,|]", aliases) if x.strip()]
    else:
        alias_list = list(aliases)
    return any(normalized_exact(folder_name, alias) for alias in alias_list)


def score_candidate(folder_name: str, record: dict[str, Any]) -> dict[str, Any]:
    """Per-strategy scores for one (folder_name, db record) pair."""
    riddim_name = record.get("riddim_name", "")
    return {
        "normalized_exact": normalized_exact(folder_name, riddim_name),
        "fuzzy": fuzzy_ratio(folder_name, riddim_name),
        "token_overlap": token_overlap(folder_name, riddim_name),
        "alias_match": alias_match(folder_name, record),
    }


def rank_candidates(folder_name: str, records: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    """
    Scores every record against folder_name using all strategies and
    returns the top_n by a simple combined ranking key. The combined key
    is ONLY used to decide which candidates are worth returning to the
    agent -- it is explicitly not exposed as a single collapsed
    "confidence" number, and the agent must not treat rank order alone
    as a decision.
    """
    scored = []
    for rec in records:
        scores = score_candidate(folder_name, rec)
        rank_key = (
            (1.0 if scores["normalized_exact"] else 0.0)
            + (1.0 if scores["alias_match"] else 0.0)
            + scores["fuzzy"]
            + scores["token_overlap"]
        )
        scored.append(
            {
                "candidate_id": rec["id"],
                "name": rec.get("riddim_name"),
                "year": rec.get("year"),
                "scores": scores,
                "_rank_key": rank_key,
            }
        )
    scored.sort(key=lambda x: x["_rank_key"], reverse=True)
    top = scored[:top_n]
    for item in top:
        item.pop("_rank_key", None)
    return top


def find_candidates_by_normalized_name(normalized_name: str, top_n: int = 8) -> list[dict[str, Any]]:
    """
    Find candidate riddims using normalized name matching.

    Uses SQL LIKE on the precomputed riddim_name_normalized column to
    pre-filter candidates, then computes full scoring only on the filtered
    subset. This is much more efficient than loading all records and
    computing fuzzy matching for each, especially with large databases.

    Key fix: computes a weighted relevance score in SQL and orders by it,
    so that riddims matching the most specific tokens (e.g. "cry baby",
    "095", "driven") rank higher than generic "riddim" matches.
    """
    tokens = normalized_name.split()
    if not tokens:
        return []

    with _connect() as conn:
        cols = _table_columns(conn, "riddims")
        
        # Common tokens that appear in almost every riddim name - exclude from filtering
        # but still compute relevance for them
        common_tokens = {'riddim', 'the', 'and', 'of', 'a', 'in', 'to', 'is', 'for', 'on', 'by', 'with'}
        
        # Separate unique tokens from common ones, and compute weights
        unique_tokens = [t for t in tokens if t.lower() not in common_tokens]
        
        if not unique_tokens:
            # Fallback: only common tokens detected, use token lengths as weights
            unique_tokens = tokens
        
        # Build relevance score computation
        # Each unique token match contributes 1.0 to relevance
        score_parts = []
        for token in unique_tokens:
            weight = 1.0 + len(token) * 0.1  # Longer tokens slightly weighted more
            score_parts.append(f"CASE WHEN riddim_name_normalized LIKE ? THEN {weight} ELSE 0.0 END")
        
        # Build WHERE clause using only unique tokens
        like_clauses = []
        params = []
        for token in unique_tokens:
            like_clauses.append("riddim_name_normalized LIKE ?")
            params.append(f"%{token}%")
        
        where_clause = " OR ".join(like_clauses) if like_clauses else "1=1"
        
        # Build the full SQL query with relevance scoring
        if score_parts:
            relevance_expr = " + ".join(score_parts)
        else:
            relevance_expr = "0.0"
        
        # Full query: compute relevance score and order by it
        relevance_sql = f"""
        SELECT id, riddim_name, year, label, riddim_name_normalized,
          ({relevance_expr}) AS relevance_score
        FROM riddims
        WHERE {where_clause}
        ORDER BY relevance_score DESC, riddim_name_normalized
        LIMIT ?
        """
        
        # Execute with all parameters: the score_parts and where_clause share the same LIKE placeholders
        all_params = params + params + [top_n * 3]
        
        cur = conn.execute(relevance_sql, all_params)
        rows = cur.fetchall()
        records = [dict(row) for row in rows]

    # Now compute scores on the filtered subset
    scored = []
    for rec in records:
        scores = score_candidate(normalized_name, rec)
        rank_key = (
            (1.0 if scores["normalized_exact"] else 0.0)
            + (1.0 if scores["alias_match"] else 0.0)
            + scores["fuzzy"]
            + scores["token_overlap"]
        )
        scored.append(
            {
                "candidate_id": rec["id"],
                "name": rec.get("riddim_name"),
                "year": rec.get("year"),
                "scores": scores,
                "_rank_key": rank_key,
            }
        )
    scored.sort(key=lambda x: x["_rank_key"], reverse=True)
    top = scored[:top_n]
    for item in top:
        item.pop("_rank_key", None)
    return top


def compare_names_deterministic(name_a: str, name_b: str) -> dict[str, Any]:
    """Deterministic structured diff between two names. No judgment."""
    norm_a, norm_b = normalize(name_a), normalize(name_b)
    tokens_a, tokens_b = norm_a.split(), norm_b.split()
    set_a, set_b = set(tokens_a), set(tokens_b)

    differences = []
    if norm_a == norm_b:
        differences.append("normalized forms are identical")
    else:
        if sorted(tokens_a) == sorted(tokens_b) and tokens_a != tokens_b:
            differences.append("word order swapped")
        removed = set_a - set_b
        added = set_b - set_a
        if removed:
            differences.append(f"tokens only in name_a: {sorted(removed)}")
        if added:
            differences.append(f"tokens only in name_b: {sorted(added)}")
        if re.sub(r"[-_.]", "", name_a.lower()) == re.sub(r"[-_.]", "", name_b.lower()):
            differences.append("only punctuation (hyphen/underscore/period) differs")
        years_a, years_b = set(_YEAR_RE.findall(name_a)), set(_YEAR_RE.findall(name_b))
        if years_a != years_b:
            differences.append(f"year-like tokens differ: name_a={sorted(years_a)} name_b={sorted(years_b)}")

    overlap = len(set_a & set_b) / min(len(set_a), len(set_b)) if set_a and set_b else 0.0

    return {
        "normalized_a": norm_a,
        "normalized_b": norm_b,
        "normalized_equal": norm_a == norm_b,
        "differences": differences or ["no differences detected beyond normalization"],
        "token_overlap_pct": round(overlap * 100, 2),
        "edit_distance": _levenshtein(norm_a, norm_b),
    }


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def compare_track_evidence_deterministic(
    audio_files: list[str], candidate_record: dict[str, Any]
) -> dict[str, Any]:
    """
    Deterministic comparison between folder audio filenames and whatever
    track/artist/year/label evidence the candidate record actually carries.

    NOTE on this DB's schema: the riddims table currently has no
    tracks/artist column, so there is no stored tracklist to diff against.
    This function is schema-tolerant: if `tracks` (or `artist`) fields are
    present on a future record they are used directly and given the most
    weight; otherwise it falls back to the next-best deterministic signal
    actually available -- looking for the candidate's year and label tokens
    inside the audio filenames -- and it labels its own output
    accordingly so the agent never mistakes a weak fallback for confirmed
    track-level evidence.
    """
    stored_tracks = candidate_record.get("tracks")
    stored_artist = candidate_record.get("artist")

    if stored_tracks or stored_artist:
        track_list = []
        if isinstance(stored_tracks, str):
            track_list = [t.strip() for t in re.split(r"[;,|]", stored_tracks) if t.strip()]
        elif isinstance(stored_tracks, list):
            track_list = list(stored_tracks)

        matches, mismatches = [], []
        for f in audio_files:
            f_norm = normalize(f)
            hit = any(normalize(t) in f_norm or f_norm in normalize(t) for t in track_list)
            if stored_artist and normalize(stored_artist) in f_norm:
                hit = True
            (matches if hit else mismatches).append(f)

        return {
            "evidence_tier": "track_or_artist_metadata",
            "stored_tracks": track_list,
            "stored_artist": stored_artist,
            "matched_files": matches,
            "unmatched_files": mismatches[:10] if len(mismatches) > 10 else mismatches,
            "total_unmatched_count": len(mismatches),
            "overlap_ratio": round(len(matches) / len(audio_files), 4) if audio_files else 0.0,
        }

    # Fallback tier: no stored tracklist/artist in this DB schema. Check
    # for the candidate's year and label appearing in filenames instead.
    year = candidate_record.get("year")
    label = candidate_record.get("label") or ""
    label_norm = normalize(label)

    year_hits, label_hits, no_signal = [], [], []
    for f in audio_files:
        f_norm = normalize(f)
        found_year = bool(year) and str(year) in f
        found_label = bool(label_norm) and label_norm in f_norm
        if found_year:
            year_hits.append(f)
        if found_label:
            label_hits.append(f)
        if not found_year and not found_label:
            no_signal.append(f)

    year_matches_in_files = sorted({y for f in audio_files for y in _YEAR_RE.findall(f)})
    year_contradicted = bool(year) and any(y != str(year) for y in year_matches_in_files)

    return {
        "evidence_tier": "fallback_year_label_in_filename",
        "note": (
            "This DB schema has no tracks/artist column for this candidate, "
            "so this is weaker fallback evidence (year/label mentions in "
            "filenames), not confirmed tracklist evidence. Treat accordingly."
        ),
        "candidate_year": year,
        "candidate_label": label,
        "files_mentioning_candidate_year": year_hits,
        "files_mentioning_candidate_label": label_hits,
        "files_with_no_year_or_label_signal": no_signal[:10] if len(no_signal) > 10 else no_signal,
        "total_files_with_no_signal_count": len(no_signal),
        "year_like_tokens_found_in_filenames": year_matches_in_files,
        "year_contradicted_by_filenames": year_contradicted,
    }
