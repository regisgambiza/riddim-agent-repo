import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matching


def test_normalized_exact_true_for_punct_variants():
    assert matching.normalized_exact("Bombay Riddim (2022)", "bombay_riddim_2022")


def test_normalized_exact_false_for_real_difference():
    assert not matching.normalized_exact("Bombay Riddim V2", "Bombay Riddim 2022")


def test_fuzzy_ratio_high_for_typo():
    r = matching.fuzzy_ratio("186 Ridim", "186 Riddim")
    assert r > 0.85


def test_token_overlap_handles_word_order():
    assert matching.token_overlap("Riddim Bombay", "Bombay Riddim") == 1.0


def test_alias_match_false_when_no_aliases_field():
    assert matching.alias_match("Anything", {"riddim_name": "x"}) is False


def test_compare_names_flags_word_order_swap():
    result = matching.compare_names_deterministic("Riddim Bombay", "Bombay Riddim")
    assert result["normalized_equal"] is False
    assert any("word order" in d for d in result["differences"])


def test_compare_names_flags_year_difference():
    result = matching.compare_names_deterministic("Bombay Riddim 2021", "Bombay Riddim 2022")
    assert any("year-like tokens differ" in d for d in result["differences"])


def test_compare_names_identical_after_normalization():
    result = matching.compare_names_deterministic("Bombay Riddim (2022)", "Bombay Riddim - 2022")
    assert result["normalized_equal"] is True


def test_compare_track_evidence_fallback_tier_when_no_tracks_column():
    record = {"riddim_name": "Bombay Riddim 2022", "year": 2022, "label": "Riddim Empire"}
    result = matching.compare_track_evidence_deterministic(
        ["BombayRiddim_2022_Artist_Three.mp3", "unrelated_file.mp3"], record
    )
    assert result["evidence_tier"] == "fallback_year_label_in_filename"
    assert "BombayRiddim_2022_Artist_Three.mp3" in result["files_mentioning_candidate_year"]
    assert "unrelated_file.mp3" in result["files_with_no_year_or_label_signal"]


def test_compare_track_evidence_uses_tracks_when_present():
    record = {"riddim_name": "X Riddim", "year": 2020, "tracks": "Song A, Song B", "artist": "MC Test"}
    result = matching.compare_track_evidence_deterministic(["01 Song A (MC Test).mp3", "zz.mp3"], record)
    assert result["evidence_tier"] == "track_or_artist_metadata"
    assert "01 Song A (MC Test).mp3" in result["matched_files"]
    assert "zz.mp3" in result["unmatched_files"]


def test_rank_candidates_bombay_ambiguity():
    """
    The DB contains multiple records that normalize identically but differ
    in year (e.g. 'Real rock Riddim'). rank_candidates should surface
    several of these near the top -- it must NOT silently collapse
    to one -- confirming the agent actually has to disambiguate rather
    than the tool doing it for them.
    """
    import db as db_module

    records = db_module.list_all_riddims()

    # Find a name with multiple distinct records sharing normalized form
    from collections import Counter
    norm_counts = Counter(db_module.normalize(r["riddim_name"]) for r in records)
    multi_names = [n for n, c in norm_counts.items() if c >= 3]
    assert multi_names, "No name has 3+ distinct records with same normalized form"
    query_name = multi_names[0]
    expected_ids = {r["id"] for r in records if db_module.normalize(r["riddim_name"]) == query_name}

    ranked = matching.rank_candidates(query_name, records, top_n=6)
    top_ids = {c["candidate_id"] for c in ranked[:3]}
    # Several records with the same normalized name should be near the top
    assert len(top_ids) >= 2, f"Expected multiple distinct near-duplicates, got {top_ids}"
    # At least some top candidates should share the normalized name
    overlap = top_ids & expected_ids
    assert overlap, f"No top candidates share normalized name {query_name!r}; got {top_ids}"
