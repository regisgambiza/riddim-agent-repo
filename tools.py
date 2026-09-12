"""
Tool implementations exposed to the investigation agent.

IMPORTANT: this module intentionally contains NO tool that writes to the
filesystem or database beyond reading folder/file names. There is no
copy/move/rename/delete/write-DB tool anywhere in this file or in this
project. log_decision() only appends an in-memory record that main.py
later serializes to match_proposals.json -- it never touches source files
or the riddims table.
"""

import config
import db
import matching
import spotify_tool
from pathlib import Path
from typing import Any
from spotify_tool import SEARCH_SPOTIFY_SCHEMA
from db import normalize


class ToolError(Exception):
    pass


# --------------------------------------------------------------------------
# 1. list_source_folders
# --------------------------------------------------------------------------

def list_source_folders() -> list[str]:
    root = Path(config.SOURCE_ROOT)
    if not root.exists():
        raise ToolError(f"source root not found: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


# --------------------------------------------------------------------------
# 2. get_folder_contents
# --------------------------------------------------------------------------

def get_folder_contents(folder_name: str) -> dict[str, list[str]]:
    folder = Path(config.SOURCE_ROOT) / folder_name
    if not folder.exists() or not folder.is_dir():
        raise ToolError(f"folder not found under source root: {folder_name}")

    audio_files, other_files = [], []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in config.AUDIO_EXTENSIONS:
            audio_files.append(p.name)
        else:
            other_files.append(p.name)

    return {"audio_files": audio_files, "other_files": other_files}


# --------------------------------------------------------------------------
# 3. get_candidates
# --------------------------------------------------------------------------

def get_candidates(folder_name: str, top_n: int = config.DEFAULT_TOP_N) -> list[dict[str, Any]]:
    normalized = normalize(folder_name)
    return matching.find_candidates_by_normalized_name(normalized, top_n)


# --------------------------------------------------------------------------
# 4. get_full_record
# --------------------------------------------------------------------------

def get_full_record(candidate_id: int) -> dict[str, Any]:
    record = db.get_record(candidate_id)
    if record is None:
        raise ToolError(f"no riddim record with id={candidate_id}")
    return record


# --------------------------------------------------------------------------
# 5. compare_names
# --------------------------------------------------------------------------

def compare_names(name_a: str, name_b: str) -> dict[str, Any]:
    return matching.compare_names_deterministic(name_a, name_b)


# --------------------------------------------------------------------------
# 6. compare_track_evidence
# --------------------------------------------------------------------------

def compare_track_evidence(folder_name: str, candidate_id: int) -> dict[str, Any]:
    contents = get_folder_contents(folder_name)
    record = get_full_record(candidate_id)
    return matching.compare_track_evidence_deterministic(contents["audio_files"], record)


# --------------------------------------------------------------------------
# 7. search_spotify
# --------------------------------------------------------------------------

def search_spotify(
    riddim_name: str,
    query_variant: str = "album_name",
    year: int | None = None,
    producer: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Search Spotify for a release matching `riddim_name`, using ONE query variant.

    Returns on success:
        {"ok": True, "query_variant": str, "candidates": [...], "count": int}

    Returns on quota exhaustion (do NOT retry):
        {"ok": False, "error": "quota_exhausted", "message": "...", "guidance": "..."}

    Returns on auth failure (one retry is reasonable):
        {"ok": False, "error": "auth_expired", "message": "..."}

    Returns on other HTTP failure:
        {"ok": False, "error": "http_error", "message": "..."}
    """
    return spotify_tool.search_spotify(
        riddim_name,
        query_variant=query_variant,
        year=year,
        producer=producer,
        label=label,
    )

_VALID_STATUS = {"matched", "needs_review", "no_match"}
_VALID_CONFIDENCE = {"high", "medium", "low", None}


def validate_decision(decision: dict[str, Any]) -> list[str]:
    """Returns a list of schema-violation strings; empty list = valid."""
    problems = []

    required_top = {
        "folder_name", "source_path", "status", "proposed_match",
        "confidence", "candidates_considered", "sanity_check", "decision_summary",
    }
    missing = required_top - decision.keys()
    if missing:
        problems.append(f"missing required fields: {sorted(missing)}")

    status = decision.get("status")
    if status not in _VALID_STATUS:
        problems.append(f"invalid status: {status!r}")

    confidence = decision.get("confidence")
    if confidence not in _VALID_CONFIDENCE:
        problems.append(f"invalid confidence: {confidence!r}")

    if status == "matched":
        if decision.get("proposed_match") is None:
            problems.append("status=matched requires a non-null proposed_match")
        if confidence not in {"high", "medium", "low"}:
            problems.append("status=matched requires confidence to be high/medium/low")
    else:
        if decision.get("proposed_match") is not None:
            problems.append(f"status={status} must have proposed_match=null")
        if confidence is not None:
            problems.append(f"status={status} must have confidence=null")

    proposed = decision.get("proposed_match")
    if proposed is not None:
        # The year must come from the record itself, never be invented.
        # We re-fetch the record and cross-check the year matches exactly.
        cid = proposed.get("candidate_id")
        record = db.get_record(cid) if cid is not None else None
        if record is None:
            problems.append(f"proposed_match.candidate_id={cid!r} does not exist in the database")
        else:
            db_year = record.get("year")
            claimed_year = proposed.get("year")
            if claimed_year != db_year:
                problems.append(
                    "proposed_match.year does not match the database record's "
                    f"year (claimed={claimed_year!r}, db={db_year!r}); "
                    "the agent must never state a year that did not come "
                    "directly from the matched database record"
                )

    sanity = decision.get("sanity_check") or {}
    for key in (
        "supporting_evidence", "strongest_alternatives",
        "why_alternatives_weaker", "contradicting_evidence",
        "considered_needs_review",
    ):
        if key not in sanity:
            problems.append(f"sanity_check missing key: {key}")

    if status == "high" and confidence == "high":
        pass  # placeholder to keep symmetry; real high-confidence rule below

    if confidence == "high":
        alternatives = decision.get("candidates_considered") or []
        investigated_competitors = [
            c for c in alternatives
            if c.get("candidate_id") != (proposed or {}).get("candidate_id") and c.get("investigated")
        ]
        # high confidence requires that competing candidates were at least
        # investigated (even if the DB only ever surfaced one candidate).
        if len(alternatives) > 1 and not investigated_competitors:
            problems.append(
                "confidence=high but competing candidates were not marked "
                "investigated=true; high confidence requires explicit "
                "elimination of plausible competitors"
            )

    return problems


_NULL_LIKE_STRINGS = {"null", "none", "nil", ""}


def _sanitize_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """
    Local models occasionally emit the literal string "null" (or "None",
    "") instead of a real JSON null when a field is meant to be absent --
    e.g. {"confidence": "null"} instead of {"confidence": null}. This
    normalizes those specific, well-understood quirks to real None values
    BEFORE schema validation, so a formatting slip doesn't get treated the
    same as an actual invented value. It does not touch anything else.
    """
    clean = dict(decision)
    for key in ("confidence", "proposed_match"):
        val = clean.get(key)
        if isinstance(val, str) and val.strip().lower() in _NULL_LIKE_STRINGS:
            clean[key] = None
    proposed = clean.get("proposed_match")
    if isinstance(proposed, dict):
        year = proposed.get("year")
        if isinstance(year, str) and year.strip().lower() in _NULL_LIKE_STRINGS:
            proposed["year"] = None
    return clean


def make_log_decision(sink: list):
    """
    Returns a log_decision(**decision) callable bound to `sink`
    (a plain list that main.py owns and later serializes). This keeps
    tools.py side-effect-free at import time and makes the "exactly once
    per folder" contract easy to enforce by the caller (agent.py).
    """

    def log_decision(**decision: Any) -> dict[str, Any]:
        decision = _sanitize_decision(decision)
        problems = validate_decision(decision)
        if problems:
            raise ToolError(
                "log_decision schema/consistency check failed: " + "; ".join(problems)
            )
        sink.append(decision)
        return {"logged": True, "folder_name": decision.get("folder_name")}

    return log_decision


# --------------------------------------------------------------------------
# OpenAI-style tool schemas (for llm_client / agent.py)
# --------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_source_folders",
            "description": "List all source folders under Folder1/ that need to be investigated.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_folder_contents",
            "description": (
                "Get the structured contents of a source folder, split into "
                "audio_files (likely track evidence, by extension) and "
                "other_files (images, text, etc. -- not track evidence)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"folder_name": {"type": "string"}},
                "required": ["folder_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_candidates",
            "description": (
                "Get ranked candidate riddims from the database using multiple "
                "deterministic strategies (normalized exact match, fuzzy string "
                "similarity, token-set overlap, alias lookup). Returns per-strategy "
                "scores, not one collapsed number. A higher score is evidence, "
                "never a decision by itself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_name": {"type": "string"},
                    "top_n": {"type": "integer", "description": "How many candidates to return."},
                },
                "required": ["folder_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_record",
            "description": "Get all matching-relevant fields for one candidate by id.",
            "parameters": {
                "type": "object",
                "properties": {"candidate_id": {"type": "integer"}},
                "required": ["candidate_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_names",
            "description": (
                "Deterministically compare two names: normalized forms, specific "
                "differences (e.g. hyphen removed, word order swapped, year-like "
                "tokens differ), token overlap percentage, edit distance. This "
                "tool makes no judgment -- you interpret the output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name_a": {"type": "string"},
                    "name_b": {"type": "string"},
                },
                "required": ["name_a", "name_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_track_evidence",
            "description": (
                "Deterministically compare a folder's audio filenames against a "
                "candidate's stored track/artist/year/label metadata. When "
                "reliable track/artist overlap exists, treat this as substantially "
                "stronger evidence than folder-name similarity alone. If the "
                "database has no tracklist for this candidate, the result is "
                "labeled as weaker fallback evidence -- read evidence_tier to "
                "tell which case you're in."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_name": {"type": "string"},
                    "candidate_id": {"type": "integer"},
                },
                "required": ["folder_name", "candidate_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_decision",
            "description": (
                "Log your FINAL decision for this folder. Call this EXACTLY ONCE "
                "per folder, only after completing your sanity check. This is your "
                "only output -- there is no file/copy/move tool. status must be "
                "'matched' (a specific candidate proposed, with confidence "
                "high/medium/low), 'needs_review' (multiple credible candidates, "
                "insufficient evidence to choose), or 'no_match' (no credible "
                "candidate at all). confidence must be null unless status is "
                "'matched'. The year in proposed_match must come directly from "
                "the matched database record -- never state a year yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_name": {"type": "string"},
                    "source_path": {"type": "string"},
                    "status": {"type": "string", "enum": ["matched", "needs_review", "no_match"]},
                    "proposed_match": {
                        "type": ["object", "null"],
                        "properties": {
                            "candidate_id": {"type": "integer"},
                            "name": {"type": "string"},
                            "year": {"type": ["integer", "null"]},
                        },
                    },
                    "confidence": {"type": ["string", "null"], "enum": ["high", "medium", "low", None]},
                    "candidates_considered": {
                        "type": "array",
                        "description": "List of candidates investigated. Keep lightweight: only candidate_id, name, year, and investigated=true. Do NOT dump raw tool output or evidence objects here.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "year": {"type": ["integer", "null"]},
                                "investigated": {"type": "boolean"},
                            },
                            "required": ["candidate_id", "investigated"],
                        },
                    },
                    "sanity_check": {
                        "type": "object",
                        "properties": {
                            "supporting_evidence": {"type": "string"},
                            "strongest_alternatives": {"type": "string"},
                            "why_alternatives_weaker": {"type": "string"},
                            "contradicting_evidence": {"type": "string"},
                            "considered_needs_review": {"type": "boolean"},
                        },
                    },
                    "decision_summary": {"type": "string"},
                },
                "required": [
                    "folder_name", "source_path", "status", "proposed_match",
                    "confidence", "candidates_considered", "sanity_check", "decision_summary",
                ],
            },
        },
    },
    SEARCH_SPOTIFY_SCHEMA,
]

# The original riddims database and source folders remain read-only.
PRIMARY_DB_READ_ONLY = True
SECONDARY_REGISTRY_WRITE_TOOL = None
