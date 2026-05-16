"""
Configuration for the Shakespeare-aware RAG system.

Centralises paths, default hyperparameters, and model identifiers used
across the pipeline. Values are deliberately conservative so that the
system runs on a standard laptop without GPU acceleration.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROMPT_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

PLAY_FILES = {
    "hamlet": DATA_DIR / "hamlet.json",
    "macbeth": DATA_DIR / "macbeth.json",
    "romeo_and_juliet": DATA_DIR / "romeo_and_juliet.json",
}

PLAY_DISPLAY_NAME = {
    "hamlet": "Hamlet",
    "macbeth": "Macbeth",
    "romeo_and_juliet": "Romeo and Juliet",
}

# ---------------------------------------------------------------------------
# Retrieval and generation defaults
# ---------------------------------------------------------------------------

# Number of chunks returned by retrieval per query.
DEFAULT_TOP_K = 3

# Suggested lightweight embedding model. The retrieval module falls back
# to a TF-IDF vector space when sentence-transformers is not installed.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking strategy. "scene" groups utterances by scene and is the default
# because Shakespeare scenes form coherent narrative units that retain
# enough context for beginner-friendly explanation while remaining small
# enough to fit into a short language-model prompt. "utterance" is also
# supported for diagnostic comparison.
CHUNK_STRATEGY = "scene"

# Maximum number of words from each retrieved passage that may be quoted
# verbatim in a generated answer. This guard prevents the extractive
# composer from echoing arbitrarily long passages.
MAX_QUOTE_WORDS = 60

# Hard upper bound on stylised Shakespearean output, in words, as required
# by the assignment specification.
STYLISED_MAX_WORDS = 150
