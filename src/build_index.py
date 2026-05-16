"""
Build and exercise the retrieval index.

This script is a sanity check that:
1. the dataset can be loaded and flattened to utterance-level records;
2. scene-level chunks can be created with metadata preserved;
3. the embedding (or TF-IDF fallback) index can be constructed;
4. retrieval returns plausible passages for a sample query.

Run as::

    cd src && python build_index.py
"""

from __future__ import annotations

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


SAMPLE_QUERIES = [
    "Why does Macbeth kill Duncan?",
    "Who is Hamlet?",
    "What is the conflict between the Montagues and the Capulets?",
]


def main() -> None:
    records = load_all_plays()
    scene_chunks = create_chunks(records, "scene")
    utterance_chunks = create_chunks(records, "utterance")

    print(f"Loaded {len(records)} flat utterance records.")
    print(f"Created {len(scene_chunks)} scene chunks "
          f"and {len(utterance_chunks)} utterance chunks for diagnostic comparison.")

    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(scene_chunks)
    print(f"Retrieval backend: {retriever.backend_name}")

    for query in SAMPLE_QUERIES:
        print()
        print("=" * 78)
        print(f"Query: {query}")
        results = retriever.retrieve(query, top_k=DEFAULT_TOP_K)
        for rank, (chunk, score) in enumerate(results, start=1):
            print("-" * 78)
            print(f"Rank {rank} | similarity={score:.4f}")
            print(format_chunk_for_display(chunk, max_chars=350))


if __name__ == "__main__":
    main()
