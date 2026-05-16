"""
Generation utilities for the Shakespeare RAG system.

The system implements two generators that share a common interface,
``generate(query, retrieved, *, mode)``:

ExtractiveComposer
    A deterministic, context-grounded answer composer. The composer
    cannot generate text that is not present in (or derived from) the
    retrieved chunks, which gives the system a strong grounding
    guarantee. This is well-aligned with the assignment's emphasis on
    "reliable, auditable, and task-focused" SLM systems.

    Concretely, given a question and a set of retrieved scene chunks the
    composer:
      * paraphrases the modern-English ``scene_summary`` field as the
        primary explanation;
      * augments the explanation with a short verbatim quotation drawn
        from the highest-scoring retrieved scene as evidence;
      * always emits the play/act/scene citation so the user can verify;
      * declines (and says so) when the retrieved evidence is weak.

StylisedComposer
    Produces a short Shakespearean-style response (<= 150 words). The
    output is generated from template fragments seeded by retrieved
    evidence so it stays on-topic. The generator labels its output as
    creative -- as required by the spec -- and never claims that the
    stylised verse is textual evidence.

A thin HuggingFaceCausalGenerator wrapper is also provided as an
optional ``generation_mode='hf'`` extension. It is documented but not
required by the baseline submission so the system can run on a laptop
with no model weights downloaded.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import MAX_QUOTE_WORDS, STYLISED_MAX_WORDS


Chunk = Dict[str, Any]
RetrievedItem = Tuple[Chunk, float]


# ---------------------------------------------------------------------------
# Question typing
# ---------------------------------------------------------------------------


def classify_question(query: str) -> str:
    """Classify a user query into a coarse type used for prompt routing."""
    q = query.lower().strip()
    # Explicit stylised generation request
    if re.search(r"(shakespeare|stylis|in the style|verse|sonnet|monologue)", q):
        return "stylised_generation"
    if q.startswith("why") or " why " in q:
        return "contextual_qa"
    if q.startswith(("who is", "what is", "what are", "what's", "tell me about")):
        return "concept_explanation"
    if q.startswith(("how", "when", "where", "what")) or q.endswith("?"):
        return "concept_explanation"
    return "concept_explanation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "of",
    "in", "to", "for", "on", "at", "by", "with", "as", "this", "that",
    "these", "those", "he", "she", "it", "his", "her", "their", "they",
    "you", "your", "we", "our", "i", "me", "my", "do", "does", "did",
    "be", "been", "being", "have", "has", "had", "not", "no", "so",
    "what", "why", "how", "who", "where", "when", "which", "from",
    "about", "into", "than", "then", "there", "here", "than", "tell",
    "give", "explain", "answer",
}


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z']+", text.lower()) if t not in _STOPWORDS]


def _sentence_split(text: str) -> List[str]:
    cleaned = re.sub(r"\[[^\]]*\]", " ", text)  # strip stage directions
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return [s.strip() for s in _SENTENCE_END.split(cleaned) if s.strip()]


def _quote_window(scene_body: str, query_tokens: List[str], max_words: int) -> str:
    """Pick the single best sentence from ``scene_body`` for quotation."""
    sentences = _sentence_split(scene_body)
    if not sentences:
        return ""
    qset = set(query_tokens)
    best = sentences[0]
    best_score = -1
    for s in sentences:
        toks = _tokens(s)
        if not toks:
            continue
        score = sum(1 for t in toks if t in qset)
        # Tie-break: prefer reasonably sized sentences.
        length_pen = 0 if 6 <= len(toks) <= 30 else -1
        total = score + length_pen
        if total > best_score:
            best_score = total
            best = s
    words = best.split()
    if len(words) > max_words:
        best = " ".join(words[:max_words]) + " ..."
    return best


def _format_citation(chunk: Chunk) -> str:
    play = chunk.get("play", "Unknown play")
    act = chunk.get("act", "?")
    scene = chunk.get("scene", "?")
    return f"{play}, Act {act}, Scene {scene}"


# ---------------------------------------------------------------------------
# Extractive composer (RAG)
# ---------------------------------------------------------------------------


@dataclass
class ExtractiveComposer:
    """Generate an evidence-grounded answer from retrieved scene chunks."""

    # Below this similarity, we explicitly tell the user the evidence is
    # weak. The threshold is conservative for TF-IDF cosine on a 73-chunk
    # corpus where on-topic scenes typically score 0.05-0.15.
    min_score: float = 0.025
    top_summaries: int = 2

    def generate(
        self,
        query: str,
        retrieved: List[RetrievedItem],
        *,
        question_type: Optional[str] = None,
    ) -> str:
        if not retrieved:
            return (
                "I cannot find any relevant passages from Hamlet, Macbeth, or "
                "Romeo and Juliet for this question. Try rephrasing in plain "
                "English."
            )

        top_score = retrieved[0][1]
        question_type = question_type or classify_question(query)

        if top_score < self.min_score:
            return (
                "The retrieved passages do not appear to address this question "
                "with high confidence. The closest match was from "
                f"{_format_citation(retrieved[0][0])}, but its similarity "
                f"score is only {top_score:.3f}, so the answer below should "
                "be treated as a hint rather than evidence."
            )

        summaries: List[str] = []
        seen_summaries: set = set()
        keyword_set: List[str] = []
        for chunk, _ in retrieved[: self.top_summaries]:
            s = (chunk.get("scene_summary") or "").strip()
            if s and s not in seen_summaries:
                summaries.append(s)
                seen_summaries.add(s)
            for kw in (chunk.get("keywords") or []):
                if kw and kw not in keyword_set:
                    keyword_set.append(kw)

        primary_chunk, primary_score = retrieved[0]
        query_tokens = _tokens(query)
        quote = _quote_window(
            primary_chunk.get("scene_body", primary_chunk.get("text", "")),
            query_tokens,
            MAX_QUOTE_WORDS,
        )

        explanation_lines: List[str] = []
        if summaries:
            joined = " ".join(summaries)
            explanation_lines.append(joined)
        if keyword_set:
            explanation_lines.append(
                "Relevant themes from the retrieved scenes include: "
                + ", ".join(keyword_set[:6])
                + "."
            )

        explanation_lines.append(
            f"In {_format_citation(primary_chunk)} the text reads: "
            f"\"{quote}\""
        )

        if question_type == "contextual_qa":
            opener = "This question is about events and motivations in the play."
        elif question_type == "concept_explanation":
            opener = "This is a concept question; the plays describe it as follows."
        else:
            opener = "Drawing on the retrieved scenes:"

        body = " ".join([opener] + explanation_lines)
        return textwrap.fill(body, width=88)


# ---------------------------------------------------------------------------
# Stylised generator
# ---------------------------------------------------------------------------


_STYLE_OPENERS = [
    "Hark, and lend thine ear,",
    "List, gentle hearer,",
    "Mark well these words,",
]


def _swap_modern(text: str) -> str:
    """Light-touch register shift: substitute a small set of modern words."""
    substitutions = [
        (r"\byou are\b", "thou art"),
        (r"\byou\b", "thou"),
        (r"\byour\b", "thy"),
        (r"\bbefore\b", "ere"),
        (r"\bover\b", "o'er"),
        (r"\bI am\b", "I am"),
        (r"\boften\b", "oft"),
    ]
    out = text
    for pat, rep in substitutions:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


@dataclass
class StylisedComposer:
    """Produce a short Shakespearean-style response grounded in retrieval."""

    max_words: int = STYLISED_MAX_WORDS

    def generate(self, query: str, retrieved: List[RetrievedItem]) -> str:
        opener = _STYLE_OPENERS[hash(query) % len(_STYLE_OPENERS)]
        seeds: List[str] = []
        for chunk, _ in retrieved[:2]:
            quote = _quote_window(
                chunk.get("scene_body", chunk.get("text", "")),
                _tokens(query),
                28,
            )
            if quote:
                seeds.append(quote.rstrip(".!? "))

        if seeds:
            stitched = "; ".join(seeds)
            verse = (
                f"{opener} a tale doth come to me. {stitched}. "
                "What art thou to say? Thy course is fix'd by both thine heart "
                "and circumstance; weigh well the duty owed to kin against the "
                "love new-kindled in thy breast, ere either prove thy ruin."
            )
        else:
            verse = (
                f"{opener} the evidence is silent, and I must not feign what "
                "the page hath not writ."
            )

        verse = _swap_modern(verse)
        words = verse.split()
        if len(words) > self.max_words:
            verse = " ".join(words[: self.max_words]) + " ..."

        # Always disclose that this is creative output, per spec.
        prefix = (
            "[STYLISED CREATIVE OUTPUT -- not textual evidence; based on the "
            "retrieved scenes.]\n"
        )
        return prefix + textwrap.fill(verse, width=88)


# ---------------------------------------------------------------------------
# Optional HuggingFace causal generator (not required for the default run)
# ---------------------------------------------------------------------------


class HuggingFaceCausalGenerator:
    """Optional wrapper around a small instruction-tuned causal LM.

    This class is not used by the default pipeline. It is included so that
    the report can describe how to plug an SLM such as ``google/flan-t5-
    small`` or ``Qwen/Qwen2.5-0.5B-Instruct`` into the system when the user
    runs on a machine with HuggingFace ``transformers`` installed.
    """

    def __init__(self, model_name: str = "google/flan-t5-small"):
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is not installed; install it to use the HF "
                "generator: pip install transformers"
            ) from exc
        self.model_name = model_name
        self._pipe = pipeline("text2text-generation", model=model_name)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        result = self._pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return result[0]["generated_text"].strip()
