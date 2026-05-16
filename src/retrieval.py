"""
Embedding and retrieval utilities for the Shakespeare RAG system.

The system supports two backends with the same public API
(``EmbeddingRetriever.build_index`` and ``EmbeddingRetriever.retrieve``):

1. ``sentence-transformers`` (preferred when available) -- dense sentence
   embeddings via ``sentence-transformers/all-MiniLM-L6-v2``. This is a
   384-dimensional model with 22M parameters, runs comfortably on CPU,
   and is one of the embedding models explicitly listed in the
   assignment specification.

2. TF-IDF + cosine similarity (default fallback) -- scikit-learn's
   ``TfidfVectorizer`` with unigrams and bigrams, English stop-word
   removal, and L2-normalised vectors. The specification also lists
   "a simple NumPy-based cosine similarity implementation" as
   acceptable.

Picking the backend is automatic and transparent. The active backend is
exposed via ``EmbeddingRetriever.backend_name`` so that the report can
truthfully describe which retriever produced the evaluated results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


Chunk = Dict[str, Any]


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class _TfidfBackend:
    """TF-IDF + cosine similarity backend (no external model weights)."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,
            norm="l2",
        )
        self._fitted = False

    def fit(self, texts):
        self._vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts, show_progress_bar=False):
        if not self._fitted:
            self.fit(texts)
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray()


class EmbeddingRetriever:
    """Embedding-based retriever with automatic backend selection."""

    def __init__(self, embedding_model_name: str):
        self.embedding_model_name = embedding_model_name
        self.chunks: List[Chunk] = []
        self.embeddings = None

        if _sentence_transformers_available():
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model_name)
            self.backend_name = f"sentence-transformers ({embedding_model_name})"
        else:
            self.model = _TfidfBackend()
            self.backend_name = "tfidf-fallback (scikit-learn 1-2 grams, English stop words)"

    def build_index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks supplied to build_index().")
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.embeddings = np.asarray(
            self.model.encode(texts, show_progress_bar=False)
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if self.embeddings is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")
        query_embedding = np.asarray(self.model.encode([query]))
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
