"""
These tests exercise agent.investigate_folder() end-to-end against a
scripted FakeLlmClient instead of a real llama-server, so they run fast
and require no GPU/model. They verify the loop-control contract:
exactly-once log_decision, tool dispatch, error feedback on invalid
log_decision calls, and the runaway-loop safety fallback.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import agent
import tools


class FakeLlmClient:
    """
    Mimics llm_client.LlmClient.chat(). `script` is a list of "message"
    dicts (OpenAI chat-completions message shape) returned in order, one
    per .chat() call. If the script is exhausted, keeps returning the
    last entry (useful for safety-cap tests).
    """

    def __init__(self, script: list[dict]):
        self.script = script
        self.calls = 0

    def chat(self, messages, tools=None, temperature=None):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[idx]


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def test_agent_resolves_exact_match_quickly():
    """Exact-match folder should resolve within a couple of tool calls."""
    script = [
        _tool_call("c1", "get_candidates", {"folder_name": "1 Vibe Riddim (Official)", "top_n": 5}),
        _tool_call(
            "c2",
            "log_decision",
            {
                "folder_name": "1 Vibe Riddim (Official)",
                "source_path": str(Path(config.SOURCE_ROOT) / "1 Vibe Riddim (Official)"),
                "status": "matched",
                "proposed_match": {"candidate_id": 1, "name": "1 Vibe Riddim", "year": 2026},
                "confidence": "high",
                "candidates_considered": [
                    {
                        "candidate_id": 1, "name": "1 Vibe Riddim", "year": 2026,
                        "scores": {"normalized_exact": False, "fuzzy": 0.9, "token_overlap": 1.0, "alias_match": False},
                        "investigated": True,
                        "evidence": {"name_comparison": {}, "track_evidence": {}, "metadata_notes": "near-exact"},
                    }
                ],
                "sanity_check": {
                    "supporting_evidence": "near-exact normalized match, single strong candidate",
                    "strongest_alternatives": "none scored close",
                    "why_alternatives_weaker": "n/a",
                    "contradicting_evidence": "none",
                    "considered_needs_review": False,
                },
                "decision_summary": "Clear single match.",
            },
        ),
    ]
    client = FakeLlmClient(script)
    decisions: list = []
    decision = agent.investigate_folder(client, "1 Vibe Riddim (Official)", decisions)

    assert decision["status"] == "matched"
    assert decision["proposed_match"]["candidate_id"] == 1
    assert decision["proposed_match"]["year"] == 2026  # came from DB, matches record
    assert len(decisions) == 1
    assert client.calls == 2  # resolved quickly, no wasted round-trips


def test_agent_never_logs_more_than_once_even_if_model_tries_twice():
    """
    If the model (buggily) tries to call log_decision twice, the second
    call must never be reached because the loop returns as soon as the
    sink grows -- exactly-once is enforced by loop control, not by trust.
    """
    good_decision_args = {
        "folder_name": "1 Vibe Riddim (Official)",
        "source_path": "irrelevant",
        "status": "no_match",
        "proposed_match": None,
        "confidence": None,
        "candidates_considered": [],
        "sanity_check": {
            "supporting_evidence": "none", "strongest_alternatives": "none",
            "why_alternatives_weaker": "n/a", "contradicting_evidence": "none",
            "considered_needs_review": True,
        },
        "decision_summary": "No credible candidate.",
    }
    script = [
        _tool_call("c1", "log_decision", good_decision_args),
        _tool_call("c2", "log_decision", good_decision_args),  # should never actually run
    ]
    client = FakeLlmClient(script)
    decisions: list = []
    agent.investigate_folder(client, "1 Vibe Riddim (Official)", decisions)

    assert len(decisions) == 1
    assert client.calls == 1


def test_agent_feeds_back_schema_error_without_logging_bad_decision():
    """An invalid log_decision call (e.g. invented year) must not land in the sink."""
    bad_year_call = _tool_call(
        "c1", "log_decision",
        {
            "folder_name": "1 Vibe Riddim (Official)",
            "source_path": "irrelevant",
            "status": "matched",
            "proposed_match": {"candidate_id": 1, "name": "1 Vibe Riddim", "year": 1999},  # WRONG, DB says 2026
            "confidence": "high",
            "candidates_considered": [],
            "sanity_check": {
                "supporting_evidence": "x", "strongest_alternatives": "x",
                "why_alternatives_weaker": "x", "contradicting_evidence": "x",
                "considered_needs_review": False,
            },
            "decision_summary": "x",
        },
    )
    corrected_call = _tool_call(
        "c2", "log_decision",
        {
            "folder_name": "1 Vibe Riddim (Official)",
            "source_path": "irrelevant",
            "status": "matched",
            "proposed_match": {"candidate_id": 1, "name": "1 Vibe Riddim", "year": 2026},  # corrected
            "confidence": "high",
            "candidates_considered": [
                {"candidate_id": 1, "name": "1 Vibe Riddim", "year": 2026, "scores": {}, "investigated": True, "evidence": {}},
            ],
            "sanity_check": {
                "supporting_evidence": "x", "strongest_alternatives": "x",
                "why_alternatives_weaker": "x", "contradicting_evidence": "x",
                "considered_needs_review": False,
            },
            "decision_summary": "x",
        },
    )
    client = FakeLlmClient([bad_year_call, corrected_call])
    decisions: list = []
    decision = agent.investigate_folder(client, "1 Vibe Riddim (Official)", decisions)

    assert len(decisions) == 1
    assert decision["proposed_match"]["year"] == 2026
    assert client.calls == 2


def test_agent_recovers_from_transient_llm_error():
    """
    A transient LlmClientError (e.g. llama-server's own JSON parser
    choking on a truncated generation, as seen in production) must not
    propagate out of investigate_folder -- the loop should feed back a
    retry nudge and keep going.
    """
    from llm_client import LlmClientError

    class FlakyThenGoodClient:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, temperature=None):
            self.calls += 1
            if self.calls == 1:
                raise LlmClientError("llama-server returned 500: json parse error")
            return _tool_call(
                "c1", "log_decision",
                {
                    "folder_name": "1 Vibe Riddim (Official)",
                    "source_path": "irrelevant",
                    "status": "no_match",
                    "proposed_match": None,
                    "confidence": None,
                    "candidates_considered": [],
                    "sanity_check": {
                        "supporting_evidence": "none", "strongest_alternatives": "none",
                        "why_alternatives_weaker": "n/a", "contradicting_evidence": "none",
                        "considered_needs_review": True,
                    },
                    "decision_summary": "No credible candidate.",
                },
            )

    client = FlakyThenGoodClient()
    decisions: list = []
    decision = agent.investigate_folder(client, "1 Vibe Riddim (Official)", decisions)

    assert decision["status"] == "no_match"
    assert len(decisions) == 1
    assert client.calls == 2  # one failed attempt, one successful retry


def test_agent_safety_cap_forces_needs_review_not_a_guess(monkeypatch):
    """
    If the model never calls log_decision at all, the loop must not run
    forever and must never fabricate a 'matched' result -- it force-logs
    needs_review once the round-trip cap is hit.
    """
    monkeypatch.setattr(config, "MAX_TOOL_ROUNDTRIPS_PER_FOLDER", 2)

    # Model just keeps calling get_candidates forever, never log_decision.
    stuck_call = _tool_call("c", "get_candidates", {"folder_name": "1 Vibe Riddim (Official)", "top_n": 3})
    client = FakeLlmClient([stuck_call])
    decisions: list = []
    decision = agent.investigate_folder(client, "1 Vibe Riddim (Official)", decisions)

    assert decision["status"] == "needs_review"
    assert decision["proposed_match"] is None
    assert decision["confidence"] is None
    assert "safety cap" in decision["decision_summary"]
    assert len(decisions) == 1


def test_agent_nudges_model_when_it_returns_plain_text_with_no_tool_call():
    plain_text_msg = {"role": "assistant", "content": "Let me think about this.", "tool_calls": []}
    final_call = _tool_call(
        "c1", "log_decision",
        {
            "folder_name": "1 Vibe Riddim (Official)",
            "source_path": "irrelevant",
            "status": "no_match",
            "proposed_match": None,
            "confidence": None,
            "candidates_considered": [],
            "sanity_check": {
                "supporting_evidence": "none", "strongest_alternatives": "none",
                "why_alternatives_weaker": "n/a", "contradicting_evidence": "none",
                "considered_needs_review": True,
            },
            "decision_summary": "No credible candidate.",
        },
    )
    client = FakeLlmClient([plain_text_msg, final_call])
    decisions: list = []
    decision = agent.investigate_folder(client, "1 Vibe Riddim (Official)", decisions)
    assert decision["status"] == "no_match"
    assert client.calls == 2


def test_agent_aborts_on_consecutive_llm_errors():
    from llm_client import LlmClientError

    class AlwaysFailingClient:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, temperature=None):
            self.calls += 1
            raise LlmClientError("llama-server returned 500: syntax error unexpected end of input")

    client = AlwaysFailingClient()
    decisions: list = []
    decision = agent.investigate_folder(client, "1 Vibe Riddim (Official)", decisions, verbose=False)

    assert decision["status"] == "needs_review"
    assert decision["proposed_match"] is None
    assert client.calls == config.MAX_CONSECUTIVE_LLM_ERRORS
    assert "consecutive rounds" in decision["decision_summary"]
    assert len(decisions) == 1
