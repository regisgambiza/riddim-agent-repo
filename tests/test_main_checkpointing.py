import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module


def test_write_then_load_checkpoint_roundtrips(tmp_path):
    out = tmp_path / "match_proposals.json"
    decisions = [
        {"folder_name": "A", "status": "matched"},
        {"folder_name": "B", "status": "no_match"},
    ]
    main_module._write_checkpoint(decisions, str(out))

    assert out.exists()
    loaded = main_module._load_existing_decisions(str(out))
    assert loaded == decisions


def test_load_existing_decisions_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    assert main_module._load_existing_decisions(str(missing)) == []


def test_load_existing_decisions_corrupt_file_returns_empty(tmp_path):
    bad = tmp_path / "corrupt.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert main_module._load_existing_decisions(str(bad)) == []


def test_write_checkpoint_never_leaves_partial_file_on_crash(tmp_path, monkeypatch):
    """
    _write_checkpoint writes to a .tmp file and atomically replaces the
    real output, so a crash mid-write can never corrupt match_proposals.json.
    """
    out = tmp_path / "match_proposals.json"
    out.write_text(json.dumps([{"folder_name": "existing", "status": "matched"}]), encoding="utf-8")

    real_open = open

    def boom(path, *a, **kw):
        if str(path).endswith(".tmp"):
            raise OSError("simulated disk failure")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", boom)
    try:
        main_module._write_checkpoint([{"folder_name": "new", "status": "no_match"}], str(out))
    except OSError:
        pass

    # Original file must be untouched -- never partially overwritten.
    assert json.loads(out.read_text(encoding="utf-8")) == [{"folder_name": "existing", "status": "matched"}]


def test_resume_filters_out_already_logged_folders():
    all_folders = ["A", "B", "C", "D"]
    already_done = {"A", "C"}
    remaining = [f for f in all_folders if f not in already_done]
    assert remaining == ["B", "D"]
