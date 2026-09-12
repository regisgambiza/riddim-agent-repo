import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import config
import tools


@pytest.fixture(autouse=True)
def _ensure_test_source_root(monkeypatch):
    monkeypatch.setattr(config, "SOURCE_ROOT", config.APP_ROOT / "Folder1")


def test_list_source_folders_includes_examples():
    folders = tools.list_source_folders()
    assert "1 Vibe Riddim (Official)" in folders
    assert "Totally Unknown Riddim XYZ" in folders


def test_get_folder_contents_splits_audio_and_other():
    contents = tools.get_folder_contents("1 Vibe Riddim (Official)")
    assert any(f.endswith(".mp3") for f in contents["audio_files"])
    assert any(f.endswith(".png") for f in contents["other_files"])
    assert not any(f.endswith(".png") for f in contents["audio_files"])


def test_get_folder_contents_unknown_folder_raises():
    with pytest.raises(tools.ToolError):
        tools.get_folder_contents("Does Not Exist Folder")


def test_get_candidates_returns_scored_list():
    candidates = tools.get_candidates("1 Vibe Riddim (Official)", top_n=5)
    assert len(candidates) <= 5
    assert all("scores" in c for c in candidates)
    top = candidates[0]
    assert top["name"] == "1 Vibe Riddim"


def test_get_full_record_unknown_id_raises():
    with pytest.raises(tools.ToolError):
        tools.get_full_record(999999)


def test_compare_track_evidence_end_to_end():
    result = tools.compare_track_evidence("BombayRiddim", candidate_id=9)  # Bombay Riddim 2022
    assert "evidence_tier" in result


# --------------------------------------------------------------------------
# log_decision schema / consistency enforcement
# --------------------------------------------------------------------------

def _base_decision(**overrides):
    d = {
        "folder_name": "1 Vibe Riddim (Official)",
        "source_path": "/tmp/Folder1/1 Vibe Riddim (Official)",
        "status": "matched",
        "proposed_match": {"candidate_id": 1, "name": "1 Vibe Riddim", "year": 2026},
        "confidence": "high",
        "candidates_considered": [
            {
                "candidate_id": 1, "name": "1 Vibe Riddim", "year": 2026,
                "scores": {"normalized_exact": True, "fuzzy": 0.95, "token_overlap": 1.0, "alias_match": False},
                "investigated": True,
                "evidence": {"name_comparison": {}, "track_evidence": {}, "metadata_notes": "exact normalized match"},
            }
        ],
        "sanity_check": {
            "supporting_evidence": "exact normalized name match",
            "strongest_alternatives": "none",
            "why_alternatives_weaker": "n/a",
            "contradicting_evidence": "none",
            "considered_needs_review": False,
        },
        "decision_summary": "Exact match, no competitors.",
    }
    d.update(overrides)
    return d


def test_valid_matched_decision_passes():
    problems = tools.validate_decision(_base_decision())
    assert problems == []


def test_log_decision_rejects_invented_year():
    """The agent must never claim a year that didn't come from the DB record."""
    bad = _base_decision(proposed_match={"candidate_id": 1, "name": "1 Vibe Riddim", "year": 1999})
    problems = tools.validate_decision(bad)
    assert any("year" in p for p in problems)


def test_log_decision_rejects_matched_without_confidence():
    bad = _base_decision(confidence=None)
    problems = tools.validate_decision(bad)
    assert any("confidence" in p for p in problems)


def test_log_decision_rejects_needs_review_with_proposed_match():
    bad = _base_decision(status="needs_review", confidence=None,
                          proposed_match={"candidate_id": 1, "name": "x", "year": 2026})
    problems = tools.validate_decision(bad)
    assert any("proposed_match=null" in p for p in problems)


def test_log_decision_needs_review_valid():
    valid = _base_decision(
        status="needs_review", confidence=None, proposed_match=None,
        candidates_considered=[
            {"candidate_id": 9, "name": "Bombay Riddim 2022", "year": 2022, "scores": {}, "investigated": True, "evidence": {}},
            {"candidate_id": 10, "name": "Bombay Riddim (2022)", "year": 2022, "scores": {}, "investigated": True, "evidence": {}},
        ],
        sanity_check={
            "supporting_evidence": "ambiguous",
            "strongest_alternatives": "9 and 10 are indistinguishable",
            "why_alternatives_weaker": "n/a - genuinely tied",
            "contradicting_evidence": "none decisive",
            "considered_needs_review": True,
        },
        decision_summary="Two near-identical DB entries, no evidence to disambiguate.",
    )
    assert tools.validate_decision(valid) == []


def test_log_decision_high_confidence_requires_investigated_competitors():
    bad = _base_decision(
        candidates_considered=[
            {"candidate_id": 1, "name": "1 Vibe Riddim", "year": 2026, "scores": {}, "investigated": True, "evidence": {}},
            {"candidate_id": 2, "name": "16 Bars Riddim ll (2026)", "year": 2026, "scores": {}, "investigated": False, "evidence": {}},
        ]
    )
    problems = tools.validate_decision(bad)
    assert any("investigated=true" in p for p in problems)


def test_log_decision_tolerates_string_null_confidence():
    """
    Local models sometimes emit the literal string "null" instead of a
    real JSON null. This specific, well-understood quirk should be
    normalized rather than rejected as an invented value.
    """
    sink = []
    log_decision = tools.make_log_decision(sink)
    args = _base_decision(status="no_match", confidence="null", proposed_match="null")
    args["candidates_considered"] = []
    result = log_decision(**args)
    assert result["logged"] is True
    assert sink[0]["confidence"] is None
    assert sink[0]["proposed_match"] is None


def test_make_log_decision_appends_to_sink_and_returns_confirmation():
    sink = []
    log_decision = tools.make_log_decision(sink)
    result = log_decision(**_base_decision())
    assert len(sink) == 1
    assert result["logged"] is True


def test_make_log_decision_raises_on_invalid_decision():
    sink = []
    log_decision = tools.make_log_decision(sink)
    with pytest.raises(tools.ToolError):
        log_decision(**_base_decision(confidence=None))
    assert sink == []  # nothing appended on failure


# --------------------------------------------------------------------------
# Primary-DB / source-folder safety guarantee
# --------------------------------------------------------------------------

def test_only_secondary_registry_write_tool_is_exposed():
    assert tools.PRIMARY_DB_READ_ONLY is True
    assert tools.SECONDARY_REGISTRY_WRITE_TOOL is None

    exposed_names = {t["function"]["name"] for t in tools.TOOL_SCHEMAS}
    forbidden_substrings = ("copy", "move", "delete", "rename", "write", "insert", "update", "drop")
    for name in exposed_names:
        low = name.lower()
        if name in {"log_decision"}:
            continue
        assert not any(sub in low for sub in forbidden_substrings), f"suspicious tool name: {name}"

    expected = {
        "list_source_folders", "get_folder_contents", "get_candidates",
        "get_full_record", "compare_names", "compare_track_evidence",
        "log_decision", "search_spotify",
    }
    assert exposed_names == expected


def test_log_decision_never_touches_filesystem_or_db(tmp_path, monkeypatch):
    """
    Calling log_decision must not create/modify any file on disk (besides
    whatever the test harness itself does) and must not touch the DB file's
    mtime.
    """
    import db as db_module
    import config as config_module

    db_mtime_before = Path(config_module.DB_PATH).stat().st_mtime
    source_root_snapshot_before = sorted(
        str(p) for p in Path(config_module.SOURCE_ROOT).rglob("*")
    )

    sink = []
    log_decision = tools.make_log_decision(sink)
    log_decision(**_base_decision())

    db_mtime_after = Path(config_module.DB_PATH).stat().st_mtime
    source_root_snapshot_after = sorted(
        str(p) for p in Path(config_module.SOURCE_ROOT).rglob("*")
    )

    assert db_mtime_before == db_mtime_after
    assert source_root_snapshot_before == source_root_snapshot_after
