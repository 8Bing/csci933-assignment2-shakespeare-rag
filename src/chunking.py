"""
Chunking utilities for the Shakespeare RAG system.

Two strategies are implemented:

``scene``
    One chunk per scene. The chunk text concatenates speaker turns inside
    the scene and is prefixed with a one-line modern-English
    ``scene_summary`` (when present in the source data) so that the
    embedding model captures both the original Early-Modern English wording
    and a beginner-friendly paraphrase. This is the default strategy
    because Shakespeare scenes are self-contained narrative units that
    retain enough surrounding context to explain individual quotes.

``utterance``
    One chunk per speaker turn. Retained for diagnostic comparison.
    Utterance chunks are short and precise but frequently lose the context
    required to answer "why?"-style questions.

Every chunk preserves play / act / scene / speaker metadata so retrieved
evidence can be traced back to the source text.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from config import CHUNK_STRATEGY, PLAY_DISPLAY_NAME

import re

Record = Dict[str, Any]
Chunk = Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_text(record):
    """Return the cleanest text field present in a record."""
    for key in ("text", "utterance", "excerpt", "content", "passage"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalise_play_name(record):
    play_field = record.get("play")
    if isinstance(play_field, str) and play_field.strip():
        return play_field.strip()
    return PLAY_DISPLAY_NAME.get(record.get("play_key", ""), "Unknown play")


def _scene_key(record):
    """Group records into scenes using the dataset-provided scene id."""
    sid = record.get("scene_id")
    if isinstance(sid, str) and sid:
        return sid
    play = record.get("play_key") or _normalise_play_name(record).lower()
    act = record.get("act")
    scene = record.get("scene")
    if act is None or scene is None:
        return None
    return f"{play}_{act}_{scene}"
# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


REMOVE_SPEAKERS = ["FLOURISH"]


def preprocess_records(records):
    """
    Clean and normalize Shakespeare records before chunking.
    """

    cleaned_records = []

    for rec in records:

        speaker = (rec.get("speaker") or "").strip()

        # Skip unwanted records
        if speaker in REMOVE_SPEAKERS:
            continue

        # Clean text spacing
        text = _record_text(rec)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        cleaned = {
            "play": _normalise_play_name(rec),
            "act": rec.get("act"),
            "scene": rec.get("scene"),
            "scene_id": _scene_key(rec),
            "speaker": speaker,
            "scene_summary": rec.get("scene_summary"),
            "keywords": rec.get("keywords", []),
            "text": text,
        }

        cleaned_records.append(cleaned)

    return cleaned_records

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _utterance_chunks(records):
    chunks = []
    for i, rec in enumerate(records):
        text = _record_text(rec)
        if not text or rec.get("speaker") == "STAGE_DIRECTION":
            # Stage directions are not useful for evidence-grounded answers.
            continue
        chunks.append(
            {
                # "chunk_id": rec.get("source_id") or rec.get("utterance_id") or f"u_{i:06d}",
                "chunk_id": f"u_{i:06d}",
                "play": _normalise_play_name(rec),
                "act": rec.get("act"),
                "scene": rec.get("scene"),
                "speaker": rec.get("speaker"),
                "scene_summary": rec.get("scene_summary"),
                "keywords": rec.get("keywords", []),
                "text": text,
                "chunk_type": "utterance",
            }
        )
    return chunks


def _scene_chunks(records):
    """Group records into one chunk per scene."""
    grouped = {}

    for rec in records:
        key = _scene_key(rec)
        if key is None:
            continue
        text = _record_text(rec)
        speaker = rec.get("speaker")
        if speaker == "STAGE_DIRECTION":
            piece = text
        else:
            piece = f"{speaker}: {text}" if speaker and text else text

        if not piece:
            continue

        bucket = grouped.setdefault(
            key,
            {
                "chunk_id": key,
                "play": _normalise_play_name(rec),
                "act": rec.get("act"),
                "scene": rec.get("scene"),
                "scene_summary": rec.get("scene_summary"),
                "keywords": list(rec.get("keywords", []) or []),
                "speakers": [],
                "pieces": [],
                "chunk_type": "scene",
            },
        )
        if speaker and speaker != "STAGE_DIRECTION" and speaker not in bucket["speakers"]:
            bucket["speakers"].append(speaker)
        bucket["pieces"].append(piece)

    chunks = []
    for key, bucket in grouped.items():
        scene_body = "\n".join(bucket.pop("pieces"))
        summary = bucket.get("scene_summary") or ""
        keywords = bucket.get("keywords") or []
        header_parts = []
        if summary:
            header_parts.append(f"Scene summary: {summary}")
        if keywords:
            header_parts.append("Keywords: " + ", ".join(keywords))
        header_parts.append(
            f"This is {bucket.get('play')}, Act {bucket.get('act')}, "
            f"Scene {bucket.get('scene')}."
        )
        indexed_text = "\n".join(header_parts) + "\n\n" + scene_body
        chunks.append(
            {
                **bucket,
                "speaker": ", ".join(bucket["speakers"][:6]),
                "text": indexed_text,
                "scene_body": scene_body,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_chunks(records, strategy=CHUNK_STRATEGY):
    """Convert structured records into retrieval chunks using ``strategy``."""
    if strategy == "scene":
        return _scene_chunks(records)
    if strategy == "utterance":
        return _utterance_chunks(records)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def format_chunk_for_display(chunk, max_chars=600):
    """Format a retrieved chunk for human-readable display."""
    play = chunk.get("play", "Unknown play")
    act = chunk.get("act", "?")
    scene = chunk.get("scene", "?")
    speaker = chunk.get("speaker") or ""

    header = f"{play}, Act {act}, Scene {scene}"
    if speaker:
        header += f" | Speakers: {speaker}"

    body = chunk.get("scene_body") or chunk.get("text", "")
    if len(body) > max_chars:
        body = body[: max_chars - 3].rstrip() + "..."
    summary = chunk.get("scene_summary")
    if summary:
        return f"[{header}]\nSummary: {summary}\n{body}"
    return f"[{header}]\n{body}"


if __name__ == "__main__":
    from data_loader import load_all_plays
    import json

    # Load raw records
    raw_records = load_all_plays()

    # Preprocess records
    cleaned_records = preprocess_records(raw_records)

    # Show BEFORE vs AFTER
    print("=" * 80)
    print("RAW RECORD")
    print("=" * 80)
    print(json.dumps(raw_records[0], indent=2))

    print("\n")
    print("=" * 80)
    print("CLEANED RECORD")
    print("=" * 80)
    print(json.dumps(cleaned_records[0], indent=2))

    # Create chunks
    scene_chunks = create_chunks(cleaned_records, "scene")
    utterance_chunks = create_chunks(cleaned_records, "utterance")

    print("\n")
    print("=" * 80)
    print("CHUNK STATISTICS")
    print("=" * 80)

    print(f"Raw records: {len(raw_records)}")
    print(f"Cleaned records: {len(cleaned_records)}")
    print(f"Scene chunks: {len(scene_chunks)}")
    print(f"Utterance chunks: {len(utterance_chunks)}")

    print("\n")
    print("=" * 80)
    print("SAMPLE SCENE CHUNK")
    print("=" * 80)

    print(format_chunk_for_display(scene_chunks[0], max_chars=300))