import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


def test_normalize_case_punct_whitespace_underscore():
    assert db.normalize("Bombay_Riddim - (2022)") == db.normalize("bombay riddim 2022")


def test_normalize_empty():
    assert db.normalize("") == ""
    assert db.normalize(None) == ""


def test_list_all_riddims_returns_rows():
    rows = db.list_all_riddims()
    assert len(rows) >= 11
    assert all("riddim_name" in r for r in rows)


def test_get_record_known_id():
    rec = db.get_record(1)
    assert rec is not None
    assert rec["riddim_name"] == "1 Vibe Riddim"
    assert rec["year"] == 2026


def test_get_record_unknown_id():
    assert db.get_record(999999) is None


def test_db_module_has_no_write_statements():
    """Static guard: db.py must never contain INSERT/UPDATE/DELETE/DROP SQL."""
    source = Path(db.__file__).read_text().upper()
    for forbidden in ("INSERT INTO", "UPDATE ", "DELETE FROM", "DROP TABLE", "ALTER TABLE"):
        assert forbidden not in source, f"found forbidden SQL keyword: {forbidden}"
