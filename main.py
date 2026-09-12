"""
Entry point for the riddim archive investigation agent.

Usage:
    python main.py                  # investigate every folder under Folder1/
    python main.py --folder "Name"  # investigate just one folder
    python main.py --dry-run        # print planned config, don't start llama-server

This script:
  1. starts a local llama-server pointed at LLAMA_MODEL_PATH (config.py),
  2. runs the investigation agent over each source folder, one at a time,
  3. stops llama-server (even on error / Ctrl-C),
  4. writes every logged decision to match_proposals.json.

It never moves, copies, renames, or deletes any source file, and never
writes to the riddims database.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import config
import tools
from agent import investigate_folder
from llama_server_manager import LlamaServerManager, LlamaServerError
from llm_client import LlmClient


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", help="Investigate only this one folder name under Folder1/.")
    p.add_argument("--dry-run", action="store_true", help="Print config and exit without starting llama-server.")
    p.add_argument("--output", default=str(config.OUTPUT_PATH), help="Path to write match_proposals.json")
    p.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing output file and re-investigate every folder from scratch.",
    )
    return p.parse_args()


def _load_existing_decisions(output_path: str) -> list:
    path = Path(output_path)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            return existing
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _write_checkpoint(decisions: list, output_path: str) -> None:
    # Write to a temp file then replace, so a crash mid-write never leaves
    # match_proposals.json truncated/corrupted.
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, default=str)
    Path(tmp_path).replace(output_path)


def main() -> int:
    args = parse_args()

    if args.folder:
        folder_names = [args.folder]
    else:
        try:
            folder_names = tools.list_source_folders()
        except tools.ToolError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    decisions: list = [] if args.no_resume else _load_existing_decisions(args.output)
    already_done = {d.get("folder_name") for d in decisions}
    remaining = [f for f in folder_names if f not in already_done]

    print(f"Source root : {config.SOURCE_ROOT}")
    print(f"Database    : {config.DB_PATH}")
    print(f"Model       : {config.LLAMA_MODEL_PATH}")
    print(f"Folders     : {len(folder_names)} total, {len(already_done)} already logged, {len(remaining)} remaining")
    print(f"Output      : {args.output}")

    if args.dry_run:
        print("\n[dry-run] not starting llama-server. Exiting.")
        return 0

    if not remaining:
        print("Nothing left to investigate.")
        return 0

    server = LlamaServerManager()
    try:
        server.start()
        client = LlmClient(base_url=server.base_url)

        for i, folder_name in enumerate(remaining, 1):
            print(f"\n=== [{i}/{len(remaining)}] Investigating: {folder_name!r} ===", file=sys.stderr)
            try:
                decision = investigate_folder(client, folder_name, decisions)
                print(
                    f"    -> status={decision['status']} "
                    f"confidence={decision.get('confidence')} "
                    f"match={(decision.get('proposed_match') or {}).get('name')}",
                    file=sys.stderr,
                )
            except Exception as e:
                # A single folder must never take down a 1000+ folder batch.
                # Log the failure as needs_review with the error attached and
                # keep going -- a human reviewing the output will see exactly
                # which folders need a manual look or a re-run.
                print(f"    !! error investigating {folder_name!r}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                decisions.append({
                    "folder_name": folder_name,
                    "source_path": str((Path(config.SOURCE_ROOT) / folder_name).resolve()),
                    "status": "needs_review",
                    "proposed_match": None,
                    "confidence": None,
                    "candidates_considered": [],
                    "sanity_check": {
                        "supporting_evidence": "n/a", "strongest_alternatives": "n/a",
                        "why_alternatives_weaker": "n/a", "contradicting_evidence": "n/a",
                        "considered_needs_review": True,
                    },
                    "decision_summary": f"Investigation crashed with an unhandled error: {e}",
                })

            if config.CHECKPOINT_EVERY_FOLDER:
                _write_checkpoint(decisions, args.output)

    except LlamaServerError as e:
        print(f"\nllama-server error: {e}", file=sys.stderr)
        return 2
    finally:
        server.stop()
        _write_checkpoint(decisions, args.output)
        print(f"\nWrote {len(decisions)} decision(s) to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
