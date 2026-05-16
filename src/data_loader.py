"""
Data loading utilities.

The processed Shakespeare dataset stores each play as a JSON document of
the form ``{"metadata": {...}, "scenes": [...]}`` where every scene contains
a list of structured ``utterances``. This loader flattens that hierarchy
into a list of utterance-level records that downstream chunking and
retrieval code can consume directly.

A record returned by :func:`load_all_plays` has the schema::

    {
        "play":           "Macbeth",
        "play_key":       "macbeth",
        "act":            1,
        "scene":          3,
        "scene_id":       "macbeth_1_3",
        "scene_summary":  "Witches greet Macbeth with prophecies ...",
        "keywords":       ["prophecy", "witches"],
        "location":       "",
        "speaker":        "MACBETH",
        "text":           "So foul and fair a day I have not seen.",
        "source_id":      "macbeth_1_3_macbeth_001"
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from config import PLAY_FILES


Record = Dict[str, Any]


def _flatten_scene(scene: Dict[str, Any]) -> List[Record]:
    """Return every utterance inside ``scene`` enriched with scene metadata."""
    scene_meta = {
        "play": scene.get("play"),
        "act": scene.get("act"),
        "scene": scene.get("scene"),
        "scene_id": scene.get("scene_id"),
        "scene_summary": scene.get("scene_summary"),
        "keywords": scene.get("keywords", []),
        "location": scene.get("location", ""),
    }
    flat: List[Record] = []
    for utt in scene.get("utterances", []):
        record = dict(scene_meta)
        record.update(utt)
        flat.append(record)
    return flat


def _extract_records(obj: Any) -> List[Record]:
    """Extract a list of utterance-level records from a JSON object."""
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # Preferred path: scene-structured datasets such as the instructor
        # files (hamlet.json, macbeth.json, romeo_and_juliet.json).
        if "scenes" in obj and isinstance(obj["scenes"], list):
            flat: List[Record] = []
            for scene in obj["scenes"]:
                flat.extend(_flatten_scene(scene))
            if flat:
                return flat

        # Fallback for already-flat datasets.
        for key in ("records", "utterances", "chunks", "data"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]

    raise ValueError(
        "Could not extract records. Expected a list or a dictionary "
        "containing one of: scenes, records, utterances, chunks, data."
    )


def load_json_records(path: Path) -> List[Record]:
    """Load one processed Shakespeare JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset file: {path}\n"
            "Place the provided dataset files in data/processed/."
        )

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    return _extract_records(obj)


def load_all_plays() -> List[Record]:
    """Load records from all three compulsory plays."""
    all_records: List[Record] = []

    for play_key, path in PLAY_FILES.items():
        records = load_json_records(path)
        for r in records:
            r.setdefault("play_key", play_key)
        all_records.extend(records)

    return all_records


if __name__ == "__main__":
    records = load_all_plays()
    print(f"Loaded {len(records)} records.")
    print("First record:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))
