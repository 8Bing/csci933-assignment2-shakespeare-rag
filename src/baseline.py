"""
Baseline system.

The baseline is intentionally minimal so that the contribution of the RAG
pipeline (retrieval + grounding + scene summaries) can be measured. Two
baselines are provided:

``PromptOnlyBaseline``
    Returns a generic, template answer that does *not* see any retrieved
    Shakespeare text. Its job is to expose how much an answer benefits
    from the retrieved evidence in the RAG system. This is the default
    baseline used by ``evaluate.py``.

``KeywordRetrievalBaseline``
    A simple keyword-overlap retriever with no generation step. It
    returns the highest-overlap scene_summary verbatim. It demonstrates
    that retrieval alone, without prompt design or grounded composition,
    is insufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import DEFAULT_TOP_K
from generation import _tokens, classify_question


Chunk = Dict[str, Any]
RetrievedItem = Tuple[Chunk, float]


# ---------------------------------------------------------------------------
# Prompt-only baseline
# ---------------------------------------------------------------------------


_GENERIC_HINTS = {
    "macbeth": (
        "Macbeth is a Scottish tragedy in which a noble warrior, prompted by "
        "supernatural prophecy and persuaded by his wife, murders the king to "
        "seize the throne and is then destroyed by guilt and consequence."
    ),
    "hamlet": (
        "Hamlet is a Danish tragedy about a prince who, after his father's "
        "death, is asked by the ghost of the old king to avenge the murder "
        "and spends much of the play hesitating, brooding, and testing the "
        "guilt of his uncle Claudius."
    ),
    "romeo": (
        "Romeo and Juliet is a tragedy about two young lovers from the "
        "feuding Montague and Capulet families in Verona whose secret "
        "marriage ends in their deaths."
    ),
    "juliet": (
        "Juliet Capulet is a young noblewoman in Verona who falls in love "
        "with Romeo despite the feud between their families."
    ),
}


@dataclass
class PromptOnlyBaseline:
    """A baseline that answers from a fixed generic prior, with no retrieval."""

    def generate(self, query: str) -> str:
        q = query.lower()
        question_type = classify_question(query)

        # Try to surface a relevant generic hint.
        for keyword, hint in _GENERIC_HINTS.items():
            if keyword in q:
                head = hint
                break
        else:
            head = (
                "Without consulting the source text, only general knowledge "
                "of Shakespeare's plays is available."
            )

        if question_type == "stylised_generation":
            return (
                "[STYLISED CREATIVE OUTPUT -- not textual evidence.]\n"
                "Hark, gentle one, my words must shape the tale anew; "
                "yet have I no quotation from the page, only such report "
                "as common knowledge yields."
            )

        return (
            head
            + " (Baseline note: this answer was produced without any "
            "retrieval, so specific scenes, speakers, or quotations are "
            "not cited.)"
        )


# ---------------------------------------------------------------------------
# Keyword retrieval baseline
# ---------------------------------------------------------------------------


@dataclass
class KeywordRetrievalBaseline:
    """Keyword overlap over scene summaries; returns the best summary verbatim."""

    top_k: int = DEFAULT_TOP_K

    def fit(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self._index = [(set(_tokens((c.get("scene_summary") or "") + " " + c.get("text", ""))), c)
                       for c in chunks]

    def retrieve(self, query: str) -> List[RetrievedItem]:
        qset = set(_tokens(query))
        scored: List[Tuple[float, Chunk]] = []
        for toks, chunk in self._index:
            if not toks:
                continue
            inter = len(qset & toks)
            if inter == 0:
                continue
            # Jaccard similarity
            union = len(qset | toks)
            score = inter / union if union else 0.0
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(c, s) for s, c in scored[: self.top_k]]

    def generate(self, query: str) -> Tuple[str, List[RetrievedItem]]:
        retrieved = self.retrieve(query)
        if not retrieved:
            return ("[No keyword match in the corpus.]", retrieved)
        top, score = retrieved[0]
        summary = (top.get("scene_summary") or "").strip()
        if not summary:
            return (top.get("text", "")[:400], retrieved)
        return (
            f"{summary} (Source: {top.get('play')}, "
            f"Act {top.get('act')}, Scene {top.get('scene')}; "
            f"jaccard={score:.3f}.)",
            retrieved,
        )


if __name__ == "__main__":
    bot = PromptOnlyBaseline()
    for q in [
        "Who is Hamlet?",
        "Why does Macbeth kill Duncan?",
        "Write a stylised Shakespearean response from Juliet.",
    ]:
        print(">", q)
        print(bot.generate(q))
        print()
