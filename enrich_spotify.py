#!/usr/bin/env python3
"""
Batch Spotify enrichment for the riddims database.

This script processes all UNVERIFIED riddims in riddims_multi_year_new.db,
queries Spotify for each one, and enriches the records with metadata
such as producer, genre, source, match status, notes, tracks, and alt_names.

It includes quota handling, batching, checkpointing, and progress reporting.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, List, Dict

import config
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from db_write import (
    get_unverified_batch,
    count_unverified,
    enrich_riddim,
    upsert_tracks,
    add_alt_name,
)

from db import get_record, normalize

from spotify_lookup import (
    lookup_spotify,
    batch_lookup_spotify,
    SpotifyQuotaExhausted,
    SpotifyAuthError,
    CACHE_STATS,
)


# -------------------------- Configuration --------------------------

LOG_VERBOSE = os.environ.get("SPOTIFY_ENRICH_VERBOSE", "1") != "0"
BATCH_SIZE = int(os.environ.get("SPOTIFY_ENRICH_BATCH_SIZE", "25"))
MAX_RETRIES = int(os.environ.get("SPOTIFY_ENRICH_MAX_RETRIES", "3"))
CHECKPOINT_PATH = Path(os.environ.get("SPOTIFY_ENRICH_CHECKPOINT", "spotify_enrichment_checkpoint.json"))

# Scoring threshold for considering a candidate confident enough to confirm.
# Adjust based on experimentation.
CONFIDENCE_THRESHOLD = float(os.environ.get("SPOTIFY_ENRICH_CONFIDENCE_THRESHOLD", "0.7"))


# Agentic logging symbols
SYMBOLS = {
    'start': '🔍',
    'variant': '├─',
    'candidate': '│  ├─',
    'selected': '│  └─',
    'enriching': '├─',
    'tracks': '│  ├─',
    'alt_name': '│  ├─',
    'done': '└─',
    'success': '✅',
    'review': '⚠️',
    'not_found': '❌',
    'error': '💥',
    'quota': '⏸️',
    'auth': '🔐',
    'db_write': '💾',
    'time': '⏱️'
}


def _log(msg: str) -> None:
    """Standard logging function."""
    print(f"[spotify_enrich] {msg}", flush=True)


def _format_variant_reason(variant: str, producer: Optional[str], label: Optional[str]) -> str:
    """Return a human-readable explanation of why a variant was chosen."""
    if variant == "producer_name" and producer:
        return " (producer known in DB)"
    if variant == "label_name" and label:
        return " (label known in DB)"
    if variant == "album_name":
        return " (default - no producer/label in DB)"
    if variant == "bare":
        return " (last resort - bare name)"
    if variant == "track_name":
        return " (track-only search)"
    return ""


def _variant_query(variant: str, name: str, producer: Optional[str], label: Optional[str]) -> str:
    """Build the query string for display purposes."""
    if variant == "producer_name" and producer:
        return f'artist:"{producer}" {name}'
    if variant == "label_name" and label:
        return f'label:"{label}" {name}'
    if variant == "bare":
        return name
    if variant == "track_name":
        return f'artist:"{name}"'
    return f'album:"{name}"'


def _agentic_log(record_id: int, name: str, year: Optional[int], 
                  producer: Optional[str], label: Optional[str],
                  variant: str, query: str, candidates: List[Dict],
                  best_candidate: Optional[Dict], best_score: float,
                  confident: bool, tracks_enriched: bool,
                  alt_name_added: bool, db_updates: Dict[str, Any],
                  processing_time: float, error: Optional[str] = None) -> None:
    """Agentic logging that shows the thought process for each record.
    
    This mimics how coding agents like OpenCode/Codex show their thinking.
    """
    if not LOG_VERBOSE:
        return
        
    # Record header
    year_str = f" (year={year})" if year else ""
    producer_str = f", producer={producer}" if producer else ""
    label_str = f", label={label}" if label else ""
    print(f"{SYMBOLS['start']} Record #{record_id}: \"{name}\"{year_str}{producer_str}{label_str}", flush=True)
    
    # Variant selection
    variant_reason = ""
    if variant == "producer_name" and producer:
        variant_reason = " (producer known)"
    elif variant == "label_name" and label:
        variant_reason = " (label known)"
    elif variant == "album_name":
        variant_reason = " (default - no producer/label)"
    elif variant == "bare":
        variant_reason = " (last resort)"
    elif variant == "track_name":
        variant_reason = " (track-only search)"
        
    print(f"  {SYMBOLS['variant']} Variant: {variant}{variant_reason}", flush=True)
    print(f"  {SYMBOLS['variant']} Query: {query}", flush=True)
    
    # Candidates
    if candidates:
        print(f"  {SYMBOLS['variant']} Candidates: {len(candidates)} returned", flush=True)
        for i, cand in enumerate(candidates[:5]):  # Show top 5
            is_best = (best_candidate and cand.get('spotify_id') == best_candidate.get('spotify_id'))
            marker = SYMBOLS['selected'] if is_best else SYMBOLS['candidate']
            c_name = cand.get('name', 'Unknown')
            score = cand.get('score', 0.0)
            cand_year = cand.get('year')
            cand_label = cand.get('label')
            artists = cand.get('artists', [])
            artist_str = f", artist={artists[0]}" if artists else ""
            
            year_part = f", year={cand_year}" if cand_year else ""
            label_part = f", label='{cand_label}'" if cand_label else ""
            
            line = f"{marker} \"{c_name}\" (score={score:.3f}{year_part}{label_part}{artist_str})"
            if is_best and confident:
                line += f" {SYMBOLS['success']} SELECTED (score ≥ {CONFIDENCE_THRESHOLD})"
            elif is_best and not confident:
                line += f" {SYMBOLS['review']} SELECTED (score < {CONFIDENCE_THRESHOLD})"
            print(line, flush=True)
            
        if len(candidates) > 5:
            print(f"  {SYMBOLS['candidate']} ... and {len(candidates) - 5} more", flush=True)
    else:
        print(f"  {SYMBOLS['variant']} Candidates: 0 returned {SYMBOLS['not_found']}", flush=True)
    
    # Results and actions
    if error:
        if "quota" in error.lower():
            print(f"  {SYMBOLS['quota']} Quota exhausted: {error}", flush=True)
        elif "auth" in error.lower():
            print(f"  {SYMBOLS['auth']} Auth error: {error}", flush=True)
        elif "db" in error.lower():
            print(f"  {SYMBOLS['db_write']} DB write error: {error}", flush=True)
        else:
            print(f"  {SYMBOLS['error']} Error: {error}", flush=True)
    elif not candidates:
        print(f"  {SYMBOLS['done']} No candidates found → NOT FOUND {SYMBOLS['not_found']}", flush=True)
    else:
        if confident:
            print(f"  {SYMBOLS['selected']} Score {best_score:.3f} ≥ {CONFIDENCE_THRESHOLD} → CONFIRMED {SYMBOLS['success']}", flush=True)
        else:
            print(f"  {SYMBOLS['selected']} Score {best_score:.3f} < {CONFIDENCE_THRESHOLD} → REVIEW {SYMBOLS['review']}", flush=True)
        
        # Show what was updated
        if db_updates:
            updates = []
            if 'spotify_id' in db_updates:
                updates.append(f"spotify_id={db_updates['spotify_id']}")
            if 'spotify_url' in db_updates:
                updates.append(f"spotify_url={db_updates['spotify_url']}")
            if 'producer' in db_updates and db_updates['producer']:
                updates.append(f"producer={db_updates['producer']}")
            if 'label' in db_updates and db_updates['label']:
                updates.append(f"label={db_updates['label']}")
            if 'source' in db_updates:
                updates.append(f"source={db_updates['source']}")
            if 'match_status' in db_updates:
                updates.append(f"match_status={db_updates['match_status']}")
            if 'notes' in db_updates:
                # Truncate long notes
                notes = db_updates['notes']
                if len(notes) > 50:
                    notes = notes[:47] + "..."
                updates.append(f"notes='{notes}'")
            
            if updates:
                print(f"  {SYMBOLS['db_write']} Updated: {', '.join(updates)}", flush=True)
        
        if tracks_enriched:
            print(f"  {SYMBOLS['tracks']} Tracks enriched: tracklist upserted", flush=True)
                  
        if alt_name_added:
            print(f"  {SYMBOLS['alt_name']} Alt name added: '{best_candidate.get('name')}'", flush=True)
    
    print(f"  {SYMBOLS['time']} Completed in {processing_time:.2f}s", flush=True)
    print(flush=True)  # Blank line for readability


def _load_checkpoint() -> Dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"offset": 0, "processed": 0, "errors": 0}


def _save_checkpoint(cp: Dict[str, Any]) -> None:
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cp, f, indent=2)
    tmp.replace(CHECKPOINT_PATH)


def _choose_variant(record: Dict[str, Any]) -> str:
    """Choose Spotify search variant based on available DB fields."""
    if record.get("producer"):
        return "producer_name"
    if record.get("label"):
        return "label_name"
    return "album_name"


def _process_one(record: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a single riddim record using Spotify.

    Returns a dict with enrichment status and details.
    """
    start_time = time.time()
    riddim_id = record["id"]
    name = record["riddim_name"]
    year = record.get("year")
    producer = record.get("producer")
    label = record.get("label")

    best = None
    best_score = -1.0
    all_candidates: List[Dict] = []
    error = None

    # Determine which variant(s) to try. We'll try the primary variant based on DB fields,
    # and if that yields low score, we may try alternatives? For simplicity and to conserve
    # quota, we try only the primary variant. The agent's approach uses one call.
    variant = _choose_variant(record)
    query = _variant_query(variant, name, producer, label)

    for attempt in range(MAX_RETRIES):
        try:
            candidates = lookup_spotify(
                name,
                year=year,
                producer=producer,
                label=label,
                variant=variant,
                limit=5,
                enrich_top_tracks=True,  # we want tracklist and label from album
            )
            break
        except SpotifyQuotaExhausted:
            # Re-raise quota exhausted so the caller can handle it with waiting
            raise
        except SpotifyAuthError:
            if attempt < MAX_RETRIES - 1:
                backoff = 2 ** (attempt + 1)
                _log(f"Auth error for riddim {riddim_id}, retry {attempt + 1}/{MAX_RETRIES} in {backoff}s")
                time.sleep(backoff)
                continue
            error = f"Spotify auth failed: {str(sys.exc_info()[1])}"
            return _error_result(riddim_id, "spotify_auth_failed", error)
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            error = f"Spotify lookup failed: {str(sys.exc_info()[1])}"
            return _error_result(riddim_id, "spotify_lookup_failed", error)

    if not candidates:
        # No candidates found at all.
        enrich_riddim(
            riddim_id,
            match_status="NOT FOUND",
            source="spotify",
            notes="No Spotify candidates found",
        )
        processing_time = time.time() - start_time
        _agentic_log(
            record_id=riddim_id, name=name, year=year, producer=producer, label=label,
            variant=variant, query=query, candidates=[], best_candidate=None, best_score=-1.0,
            confident=False, tracks_enriched=False, alt_name_added=False,
            db_updates={"match_status": "NOT FOUND", "source": "spotify", 
                       "notes": "No Spotify candidates found"},
            processing_time=processing_time, error="No candidates found"
        )
        return {"id": riddim_id, "status": "not_found", "candidates": []}

    # Score candidates (lookup_spotify already returns scored candidates sorted by score).
    # We'll take the top candidate.
    top = candidates[0]
    score = top.get("score", 0.0)
    if score > best_score:
        best = top
        best_score = score

    # Determine if we have a confident match.
    confident = best_score >= CONFIDENCE_THRESHOLD

    # Prepare enrichment fields (only fill if we have data).
    spotify_id = best.get("spotify_id")
    spotify_url = best.get("url")
    # Artist list: may be empty.
    artists: List[str] = best.get("artists", [])
    # Use first artist as producer if we don't already have producer and artists list non-empty.
    spotify_producer = artists[0] if artists else None
    # Label from album data (may be None).
    spotify_label = best.get("label")
    # Genre not available from Spotify search; leave as None.
    spotify_genre = None
    # Source and source_url.
    spotify_source = "spotify"
    spotify_source_url = spotify_url

    # Build kwargs for enrich_riddim, only include fields we want to update.
    # We'll always update spotify_id, spotify_url, source, source_url.
    # For producer, label, we only update if currently NULL/empty (to avoid overwriting existing data).
    # However the spec says "independently enrich all database entries through Spotify",
    # which suggests we want to fill missing fields, not necessarily overwrite.
    # We'll implement conservative: only update if DB field is None or empty string.
    updates: Dict[str, Any] = {
        "spotify_id": spotify_id,
        "spotify_url": spotify_url,
        "source": spotify_source,
        "source_url": spotify_source_url,
    }
    # Producer: update only if DB producer is missing.
    if not producer:
        updates["producer"] = spotify_producer
    # Label: update only if DB label is missing.
    if not label:
        updates["label"] = spotify_label
    # Genre: Spotify doesn't provide genre; skip.
    # Note: We do NOT update match_status or notes here; we'll set them below.
    try:
        enrich_riddim(riddim_id, **updates)
    except Exception as e:
        processing_time = time.time() - start_time
        _agentic_log(
            record_id=riddim_id, name=name, year=year, producer=producer, label=label,
            variant=variant, query=query, candidates=candidates, best_candidate=best, best_score=best_score,
            confident=confident, tracks_enriched=False, alt_name_added=False,
            db_updates={}, processing_time=processing_time, error=str(e)
        )
        return _error_result(riddim_id, "db_write_failed", str(e))

    # Enrich tracks if we have tracklist from the candidate.
    tracks_enriched = False
    if best.get("tracklist"):
        try:
            # Build track dicts: track_name, artist (first artist), version NULL, year NULL,
            # spotify_id and spotify_url we don't have per-track; we could fetch but skip for now.
            # We'll set track-level spotify fields to NULL; they can be filled later if needed.
            tracks = [
                {
                    "track_name": t,
                    "artist": artists[0] if artists else None,
                    "version": None,
                    "year": None,
                    "spotify_id": None,
                    "spotify_url": None,
                }
                for t in best["tracklist"]
            ]
            upsert_tracks(riddim_id, tracks)
            tracks_enriched = True
        except Exception as e:
            _log(f"Failed to upsert tracks for riddim {riddim_id}")

    # Add best candidate name as alt_name if different from original.
    best_name = best.get("name")
    alt_name_added = False
    if best_name and normalize(best_name) != normalize(name):
        try:
            add_alt_name(riddim_id, best_name)
            alt_name_added = True
        except Exception:
            pass

    # Determine final match status.
    if confident:
        match_status = "CONFIRMED"
        notes = f"score={best_score:.3f} query_variant={variant}"
    else:
        # Low score but we have candidates -> needs review.
        match_status = "REVIEW"
        notes = f"low score ({best_score:.3f}) query_variant={variant} candidates={len(candidates)}"

    # Update match status and notes.
    try:
        enrich_riddim(
            riddim_id,
            match_status=match_status,
            notes=notes,
        )
        db_updates = {
            "match_status": match_status,
            "notes": notes
        }
        # Add any updates from the first enrich_riddim call if they weren't already included
        if "spotify_id" not in db_updates:
            db_updates["spotify_id"] = spotify_id
        if "spotify_url" not in db_updates:
            db_updates["spotify_url"] = spotify_url
        if "producer" not in db_updates:
            db_updates["producer"] = spotify_producer
        if "label" not in db_updates:
            db_updates["label"] = spotify_label
        if "source" not in db_updates:
            db_updates["source"] = spotify_source
        if "source_url" not in db_updates:
            db_updates["source_url"] = spotify_source_url
    except Exception as e:
        processing_time = time.time() - start_time
        _agentic_log(
            record_id=riddim_id, name=name, year=year, producer=producer, label=label,
            variant=variant, query=query, candidates=candidates, best_candidate=best, best_score=best_score,
            confident=confident, tracks_enriched=tracks_enriched, alt_name_added=alt_name_added,
            db_updates={}, processing_time=processing_time, error=str(e)
        )
        return _error_result(riddim_id, "db_write_status_failed", str(e))

    processing_time = time.time() - start_time
    _agentic_log(
        record_id=riddim_id, name=name, year=year, producer=producer, label=label,
        variant=variant, query=query, candidates=candidates, best_candidate=best, best_score=best_score,
        confident=confident, tracks_enriched=tracks_enriched, alt_name_added=alt_name_added,
        db_updates=db_updates, processing_time=processing_time
    )

    return {
        "id": riddim_id,
        "status": "enriched" if confident else "review",
        "best_candidate": best,
        "score": best_score,
        "tracks_enriched": tracks_enriched,
        "candidates": candidates,
    }


def _error_result(riddim_id: int, status: str, error: str) -> Dict[str, Any]:
    """Record an error and set the riddim to REVIEW for manual inspection."""
    try:
        enrich_riddim(
            riddim_id,
            match_status="REVIEW",
            source="spotify",
            notes=f"Error: {status}: {error}",
        )
    except Exception:
        pass
    return {"id": riddim_id, "status": status, "error": error}


def enrich_batch(limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
    """Enrich a batch of UNVERIFIED records and return summary."""
    records = get_unverified_batch(limit=BATCH_SIZE if limit is None else limit, offset=offset)
    if not records:
        return {"processed": 0, "status": "complete"}

    results = []
    for record in records:
        result = _process_one(record)
        results.append(result)

    return {
        "batch_offset": offset,
        "batch_size": len(records),
        "results": results,
    }


def run_enrichment(
    limit: Optional[int] = None,
    checkpoint: bool = True,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run full batch enrichment across all UNVERIFIED records.

    Args:
        limit: Maximum number of records to process (None = all).
        checkpoint: Save progress to checkpoint file.
        resume: Resume from existing checkpoint if available.

    Returns:
        Summary dict with processed count, errors, and status.
    """
    cp = _load_checkpoint() if resume else {"offset": 0, "processed": 0, "errors": 0}
    offset = cp["offset"]
    processed = cp["processed"]
    errors = cp["errors"]

    total = count_unverified()
    _log(f"Starting enrichment: {total} UNVERIFIED records, offset={offset}")

    # Track timing for progress estimation
    start_time = time.time()
    last_report_time = start_time
    records_since_last_report = 0

    # Track result counts for final summary
    confirmed_count = 0
    review_count = 0
    not_found_count = 0
    quota_stopped = False

    while not quota_stopped:
        if limit is not None and processed >= limit:
            break

        records = get_unverified_batch(limit=BATCH_SIZE, offset=offset)
        if not records:
            break

        for record in records:
            try:
                result = _process_one(record)
                if result["status"] == "enriched":
                    confirmed_count += 1
                elif result["status"] == "review":
                    review_count += 1
                elif result["status"] == "not_found":
                    not_found_count += 1
                else:
                    errors += 1
                processed += 1
            except SpotifyQuotaExhausted as e:
                wait_match = re.search(r'retry in (\d+)s', str(e))
                wait_seconds = int(wait_match.group(1)) if wait_match else 60
                hours = wait_seconds // 3600
                minutes = (wait_seconds % 3600) // 60
                seconds = wait_seconds % 60
                
                reset_time = datetime.now() + timedelta(seconds=wait_seconds + 1)
                reset_str = reset_time.strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"\n{SYMBOLS['quota']} Spotify API rate limited (429)...")
                print(f"   Spotify Retry-After: {wait_seconds}s (~{hours}h {minutes}m {seconds}s).")
                print(f"   Agent will resume at {reset_str}")
                print(f"   Agent holding off until reset. Checkpoint at offset {offset} ({processed} processed so far).")
                if checkpoint:
                    cp = {"offset": offset, "processed": processed, "errors": errors}
                    _save_checkpoint(cp)
                
                last_minute_printed = -1
                for remaining in range(wait_seconds + 1, 0, -1):
                    current_minute = (remaining - 1) // 60
                    
                    if current_minute != last_minute_printed or remaining <= 60:
                        mins, secs = divmod(remaining - 1, 60)
                        hrs, mins = divmod(mins, 60)
                        time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
                        
                        bar_len = 20
                        filled = int(bar_len * (wait_seconds + 1 - remaining) / (wait_seconds + 1))
                        bar = "█" * filled + "░" * (bar_len - filled)
                        
                        print(
                            f"\r[spotify_enrich] {SYMBOLS['quota']} "
                            f"Resuming at {time_str} |{bar}| {remaining}s remaining" + " " * 20,
                            end="",
                            flush=True,
                        )
                        last_minute_printed = current_minute
                    
                    time.sleep(1)
                
                print()
                _log(f"{SYMBOLS['success']} Rate limit reset. Resuming...")
                continue
            except Exception:
                errors += 1
                processed += 1

            offset += 1
            records_since_last_report += 1

            # Report progress every 5 records or every 5 seconds (whichever comes first)
            current_time = time.time()
            if records_since_last_report >= 5 or (current_time - last_report_time) >= 5.0:
                elapsed = current_time - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                
                # Create a simple progress bar
                progress = min(100, int((processed / total) * 100)) if total > 0 else 0
                bar_length = 20
                filled_length = int(bar_length * processed // total) if total > 0 else 0
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                _log(f"Progress: |{bar}| {progress}% | {processed}/{total} enriched | "
                     f"{errors} errors | {rate:.1f} rec/s | ETA: {eta:.0f}s")
                
                last_report_time = current_time
                records_since_last_report = 0

            if checkpoint:
                cp = {"offset": offset, "processed": processed, "errors": errors}
                _save_checkpoint(cp)

    if not quota_stopped and checkpoint and CHECKPOINT_PATH.exists():
        try:
            CHECKPOINT_PATH.unlink()
        except Exception:
            pass

    # Final timing and summary
    total_time = time.time() - start_time
    final_rate = processed / total_time if total_time > 0 else 0
    
    summary = {
        "processed": processed,
        "errors": errors,
        "confirmed": confirmed_count,
        "review": review_count,
        "not_found": not_found_count,
        "total_unverified": total,
        "final_offset": offset,
        "status": "complete" if offset >= total else ("quota_exhausted" if quota_stopped else "partial"),
        "time_seconds": round(total_time, 2),
        "rate_per_second": round(final_rate, 2),
        "cache_stats": dict(CACHE_STATS),
    }
    
    # Enhanced final summary
    _log("═" * 60)
    _log("  Enrichment Complete")
    _log("═" * 60)
    _log(f"  Processed:       {processed:,}")
    _log(f"  Confirmed:       {confirmed_count:,} ({confirmed_count / processed * 100:.1f}%)" if processed else "  Confirmed:       0")
    _log(f"  Review:          {review_count:,} ({review_count / processed * 100:.1f}%)" if processed else "  Review:          0")
    _log(f"  Not Found:       {not_found_count:,} ({not_found_count / processed * 100:.1f}%)" if processed else "  Not Found:       0")
    _log(f"  Errors:          {errors:,}")
    _log(f"  Requests Saved:  {CACHE_STATS['saved_requests']:,} (Search hits: {CACHE_STATS['search_hits']}, Album hits: {CACHE_STATS['album_hits']})")
    _log(f"  Time:            {total_time:.2f}s")
    _log(f"  Throughput:      {final_rate:.2f} records/second")
    _log("═" * 60)
    
    _log(f"Enrichment complete: {summary}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Spotify enrichment for riddims database.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process (default: all).",
    )
    parser.add_argument(
        "--no-checkpoint",
        dest="checkpoint",
        action="store_false",
        help="Do not save checkpoint progress.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume from existing checkpoint.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed candidate logging (default: enabled).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode: suppress detailed candidate logging.",
    )
    args = parser.parse_args()

    if args.quiet:
        os.environ["SPOTIFY_ENRICH_VERBOSE"] = "0"
        LOG_VERBOSE = False
    else:
        os.environ["SPOTIFY_ENRICH_VERBOSE"] = "1"
        LOG_VERBOSE = True

    # Immediate startup banner
    total_unverified = count_unverified()
    cid_masked = config.SPOTIFY_CLIENT_ID[:6] + "..." + config.SPOTIFY_CLIENT_ID[-4:] if config.SPOTIFY_CLIENT_ID else "None"
    print("╔════════════════════════════════════════════════════════════════╗", flush=True)
    print("║          Riddim Archive Spotify Enrichment Agent               ║", flush=True)
    print("╚════════════════════════════════════════════════════════════════╝", flush=True)
    print(f"  • Database:        {config.DB_PATH}", flush=True)
    print(f"  • Unverified count: {total_unverified:,} records", flush=True)
    print(f"  • Batch size:      {BATCH_SIZE}", flush=True)
    print(f"  • Record limit:    {args.limit if args.limit is not None else 'All'}", flush=True)
    print(f"  • Cache path:      {config.SPOTIFY_CACHE_PATH}", flush=True)
    print(f"  • Spotify Client:  {cid_masked}", flush=True)
    print("─" * 66, flush=True)

    summary = run_enrichment(
        limit=args.limit,
        checkpoint=args.checkpoint,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["status"] == "complete" else 1)