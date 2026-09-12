"""
The investigation agent's reasoning loop.

This is an agentic loop, not a fixed pipeline: the LLM decides which
tools to call, in what order, how many times, and when it has gathered
enough evidence to conclude -- per folder. The only thing this module
enforces top-down is (a) exactly one log_decision call per folder, and
(b) a generous runaway-loop safety cap that -- if ever hit -- force-logs
needs_review rather than letting a bug produce a silent guessed match.
"""

import json
import sys
from pathlib import Path
from typing import Any

import config
import tools
from llm_client import LlmClient, LlmClientError

SYSTEM_PROMPT = """You are an autonomous archival investigation agent. Your job is to
determine, for each source folder, which entry (if any) in the riddims
database it matches. You have deterministic tools to search candidates
via multiple strategies, inspect full records, inspect the audio files
inside the folder, and compare names/tracks objectively. Use only
evidence your tools actually return -- never assume or infer facts
(such as a riddim's year or artists) from your own general knowledge.
Treat every similarity score as evidence, never as a decision, and
treat track/artist overlap as strong evidence when available. Investigate
comparatively: establish that your chosen candidate beats every
plausible alternative. Match your investigation depth to the actual
difficulty of the case -- go deep only when candidates are genuinely
close or ambiguous. Before finalizing, run through your sanity check:
what supports this match, what were the alternatives, why are they
weaker, is there contradicting evidence, and would review be safer?
When evidence is ambiguous, mark the folder needs_review rather than
guessing; when no credible candidate exists, mark no_match. A wrong
proposed match is worse than an honest 'needs review.' You do not move
or copy files and you never state a year yourself -- the year always
comes from the matched database record. Your only output is one logged
decision per folder, with full structured evidence, for a human to
review afterward.

Available tools: list_source_folders, get_folder_contents, get_candidates,
get_full_record, compare_names, compare_track_evidence,
search_spotify, log_decision.
You do not have and must not attempt to call any tool that writes,
copies, moves, renames, or deletes files, or that writes to the
database -- no such tool exists. You must call log_decision exactly
once for the current folder, after your sanity check, and then stop.

If the database investigation finds no credible candidate, log no_match
directly. The external research tool has been removed; do not attempt to
call search_internet_for_riddim or record_internet_confirmed_riddim.

`search_spotify` returns external release data for a name you already have. Use it sparingly -- only when database candidates are weak or absent and you need corroboration that a release under this name exists, or when a label/producer cross-check would materially change your confidence. Its results are evidence of a release, not evidence of a match: a 'matched' decision must still name a database record. Call it one query variant at a time; inspect the results before choosing another variant. If it returns an `error` of `quota_exhausted`, stop calling it and log your decision on the evidence you already have.

Keep every text field in log_decision (decision_summary and each
sanity_check field) to 1-3 concise sentences. In candidates_considered,
include only candidate_id, name, year, and investigated=true for candidates
you actually checked. Never dump raw tool outputs, score dictionaries, or
evidence dictionaries into candidates_considered -- keep it strictly lightweight.
Being concise is not a shortcut on rigor: investigate as thoroughly as the
case needs, but write up your conclusion efficiently. All confidence/proposed_match
fields that should be absent must be the JSON literal null, never the text "null"."""


class AgentRunError(RuntimeError):
    pass


_TOOL_IMPL = {
    "list_source_folders": tools.list_source_folders,
    "get_folder_contents": tools.get_folder_contents,
    "get_candidates": tools.get_candidates,
    "get_full_record": tools.get_full_record,
    "compare_names": tools.compare_names,
    "compare_track_evidence": tools.compare_track_evidence,
    "search_spotify": tools.search_spotify,
    # log_decision is bound per-run, see investigate_folder()
}


def _dispatch(name: str, args: dict[str, Any], log_decision_fn) -> Any:
    if name == "log_decision":
        return log_decision_fn(**args)
    impl = _TOOL_IMPL.get(name)
    if impl is None:
        raise tools.ToolError(f"unknown tool: {name}")
    return impl(**args)


def investigate_folder(
    client: LlmClient,
    folder_name: str,
    decisions_sink: list,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Runs the agent loop for exactly one folder until it calls
    log_decision (or the safety cap is hit). Returns the logged decision.
    """
    source_path = str((Path(config.SOURCE_ROOT) / folder_name).resolve())
    log_decision_fn = tools.make_log_decision(decisions_sink)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Investigate source folder: {folder_name!r}\n"
                f"Its absolute source_path is: {source_path!r}\n"
                "Use your tools to gather evidence, then call log_decision "
                "exactly once with your final decision for this folder, "
                "using this exact source_path value."
            ),
        },
    ]

    decisions_before = len(decisions_sink)
    consecutive_llm_errors = 0

    for round_idx in range(config.MAX_TOOL_ROUNDTRIPS_PER_FOLDER):
        try:
            message = client.chat(messages, tools=tools.TOOL_SCHEMAS)
            consecutive_llm_errors = 0
        except LlmClientError as e:
            # Server-side failure (e.g. JSON parser choked on truncated output).
            # If errors repeat consecutively, abort this folder to avoid burning
            # dozens of round-trips and tens of minutes in a retry loop.
            consecutive_llm_errors += 1
            if verbose:
                print(f"  [round {round_idx}] llm error ({consecutive_llm_errors}/{config.MAX_CONSECUTIVE_LLM_ERRORS}): {e}", file=sys.stderr)

            if consecutive_llm_errors >= config.MAX_CONSECUTIVE_LLM_ERRORS:
                if verbose:
                    print(f"  [agent] aborting folder after {consecutive_llm_errors} consecutive LLM errors.", file=sys.stderr)
                forced = {
                    "folder_name": folder_name,
                    "source_path": source_path,
                    "status": "needs_review",
                    "proposed_match": None,
                    "confidence": None,
                    "candidates_considered": [],
                    "sanity_check": {
                        "supporting_evidence": "n/a",
                        "strongest_alternatives": "n/a",
                        "why_alternatives_weaker": "n/a",
                        "contradicting_evidence": "n/a",
                        "considered_needs_review": True,
                    },
                    "decision_summary": (
                        f"Forced needs_review: LLM failed {consecutive_llm_errors} consecutive rounds "
                        f"(e.g. server 500 / token limit exceeded / JSON parse error): {e}"
                    ),
                }
                decisions_sink.append(forced)
                return forced

            messages.append({
                "role": "user",
                "content": (
                    "Your last tool call could not be parsed by the server (token limit exceeded or malformed JSON). "
                    "Call log_decision IMMEDIATELY with minimal fields: "
                    "status ('no_match' or 'needs_review' or 'matched'), "
                    "candidates_considered (only candidate_id, name, year, investigated; DO NOT include raw evidence/scores), "
                    "and brief 1-2 sentence sanity_check and decision_summary fields."
                ),
            })
            continue

        messages.append(message)

        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            # Model produced plain text with no tool call. Nudge it back
            # toward using log_decision rather than silently ending the
            # loop with nothing logged.
            if verbose:
                print(f"  [round {round_idx}] (no tool call) {message.get('content', '')[:200]}",
                      file=sys.stderr)
            messages.append({
                "role": "user",
                "content": (
                    "You must call a tool. If you have finished investigating, "
                    "call log_decision now with your final structured decision."
                ),
            })
            continue

        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name", "")
            try:
                fn_args = LlmClient.parse_tool_call_args(tc)
            except LlmClientError as e:
                result: Any = {"error": str(e)}
            else:
                if verbose:
                    print(f"  [round {round_idx}] tool call: {fn_name}({fn_args})", file=sys.stderr)
                try:
                    result = _dispatch(fn_name, fn_args, log_decision_fn)
                except tools.ToolError as e:
                    result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", fn_name),
                "content": json.dumps(result, default=str),
            })

        if len(decisions_sink) > decisions_before:
            # log_decision succeeded this round -- folder is done.
            return decisions_sink[-1]

    # Safety cap hit without a successful log_decision. Force a
    # needs_review entry rather than ever fabricating a match. This is a
    # code-level failsafe, not a model decision.
    forced = {
        "folder_name": folder_name,
        "source_path": source_path,
        "status": "needs_review",
        "proposed_match": None,
        "confidence": None,
        "candidates_considered": [],
        "sanity_check": {
            "supporting_evidence": "n/a",
            "strongest_alternatives": "n/a",
            "why_alternatives_weaker": "n/a",
            "contradicting_evidence": "n/a",
            "considered_needs_review": True,
        },
        "decision_summary": (
            f"Forced needs_review: agent did not successfully call "
            f"log_decision within {config.MAX_TOOL_ROUNDTRIPS_PER_FOLDER} "
            "tool round-trips (runaway-loop safety cap)."
        ),
    }
    decisions_sink.append(forced)
    return forced
