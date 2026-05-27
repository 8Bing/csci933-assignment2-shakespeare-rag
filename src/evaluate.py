"""
Evaluation harness.

Runs both the baseline (prompt-only) and the RAG-based system on every
instructor-provided and group-designed evaluation question, then scores
each output against the five evaluation criteria using simple, deterministic
heuristics. Scoring is intentionally rubric-style rather than free-form so
that the comparison between the baseline and the RAG system is reproducible.

Outputs (written to ``results/``):

* ``evaluation_results.csv`` -- the full table required by the assignment;
* ``evaluation_results.jsonl`` -- the same rows in JSONL form with the
  retrieved chunks attached for the evaluation appendix;
* ``evaluation_summary.json`` -- mean scores per criterion and per system.

Scoring rubric (each criterion is 1, 3, or 5):

* Correctness -- 5 if the answer mentions the expected_focus tokens, 3 if
  it shows partial overlap, 1 otherwise. Stylised questions are scored as
  N/A for correctness and instead rely on style quality.
* Grounding -- 5 if at least one explicit citation (play + act + scene)
  is present in the answer, 3 if a play name is named, 1 if nothing is
  cited.
* Retrieval relevance -- 5 if the top retrieved chunk is from the
  expected play (inferred from the question), 3 if any retrieved chunk
  is, 1 otherwise. Stylised questions are graded by retrieval coverage
  only.
* Usefulness -- 5 if the answer contains a brief beginner-friendly
  explanation (>= 25 informative words) and >= one cited piece of
  evidence; 3 if either is missing; 1 if the answer is essentially a
  refusal.
* Style quality -- 5 if a stylised request produces an answer that
  includes archaic register markers ("thou", "thy", "ere", "o'er"),
  3 if some of them are present, 1 otherwise. Only applied to
  stylised questions.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import RESULTS_DIR
from baseline import PromptOnlyBaseline
from rag_chatbot import RagChatbot
from chunking import format_chunk_for_display


Chunk = Dict[str, Any]


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


INSTRUCTOR_QUESTIONS_PATH = RESULTS_DIR / "instructor_questions.json"
GROUP_QUESTIONS_PATH = RESULTS_DIR / "group_questions.json"
OUTPUT_CSV = RESULTS_DIR / "evaluation_results.csv"
OUTPUT_JSONL = RESULTS_DIR / "evaluation_results.jsonl"
OUTPUT_SUMMARY = RESULTS_DIR / "evaluation_summary.json"


def load_questions() -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    for path, source in [
        (INSTRUCTOR_QUESTIONS_PATH, "instructor"),
        (GROUP_QUESTIONS_PATH, "group"),
    ]:
        if not path.exists():
            print(f"WARNING: missing question file {path}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as f:
            for q in json.load(f):
                q.setdefault("source", source)
                questions.append(q)
    return questions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


_PLAY_KEYWORDS = {
    "hamlet": "Hamlet",
    "macbeth": "Macbeth",
    "duncan": "Macbeth",
    "lady macbeth": "Macbeth",
    "romeo": "Romeo and Juliet",
    "juliet": "Romeo and Juliet",
    "montague": "Romeo and Juliet",
    "capulet": "Romeo and Juliet",
    "ghost": "Hamlet",
    "claudius": "Hamlet",
}


def _expected_play(question: Dict[str, Any]) -> Optional[str]:
    text = (question.get("question", "") + " " + question.get("expected_focus", "")).lower()
    for keyword, play in _PLAY_KEYWORDS.items():
        if keyword in text:
            return play
    return None


def _normalise_quotes(text: str) -> str:
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )


def _focus_tokens(focus: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z']{4,}", _normalise_quotes(focus).lower())
            if t not in {"shakespearean", "creative", "style", "stylised", "output", "long", "from"}]


def score_correctness(answer: str, expected_focus: str) -> int:
    if not expected_focus:
        return 3
    tokens = _focus_tokens(expected_focus)
    if not tokens:
        return 3
    ans = _normalise_quotes(answer).lower()
    hits = sum(1 for t in tokens if t in ans)
    ratio = hits / len(tokens)
    if ratio >= 0.5:
        return 5
    if ratio > 0:
        return 3
    return 1


def score_grounding(answer: str) -> int:
    if re.search(r"act\s*\d+,\s*scene\s*\d+", answer, flags=re.IGNORECASE):
        return 5
    if re.search(r"(hamlet|macbeth|romeo|juliet|montague|capulet)", answer, flags=re.IGNORECASE):
        return 3
    return 1


def score_retrieval(retrieved: List[Tuple[Chunk, float]], expected_play: Optional[str]) -> int:
    if not retrieved:
        return 1
    if expected_play is None:
        # We cannot pin to a play -- award 3 (passed but unverified).
        return 3
    top_play = retrieved[0][0].get("play", "")
    if top_play == expected_play:
        return 5
    if any(c.get("play") == expected_play for c, _ in retrieved):
        return 3
    return 1


_REFUSAL_PHRASES = (
    "cannot find", "do not appear", "does not appear", "doesn't appear",
    "no keyword match", "placeholder", "not in the corpus",
    "outside the scope", "outside my knowledge", "out of scope",
    "not relevant", "no relevant", "not about", "unrelated",
    "cannot answer", "unable to find", "unable to answer",
    "beyond the scope", "no information", "no mention",
    "not covered", "cannot provide", "lack information",
)


def score_usefulness(answer: str, has_citation: bool) -> int:
    words = re.findall(r"[A-Za-z']+", answer)
    if any(p in answer.lower() for p in _REFUSAL_PHRASES):
        return 1
    if len(words) >= 25 and has_citation:
        return 5
    if len(words) >= 15:
        return 3
    return 1


def score_style(answer: str) -> Optional[int]:
    markers = ("thou", "thy", "ere", "o'er", "hark", "thine", "doth", "didst")
    hits = sum(1 for m in markers if m in answer.lower())
    if hits >= 3:
        return 5
    if hits >= 1:
        return 3
    return 1


def _is_refusal(answer: str) -> bool:
    # Require both a negative phrase AND a short answer (<40 words) to avoid
    # false positives where the model hedges but then provides real content.
    lower = answer.lower()
    has_negative = any(p in lower for p in _REFUSAL_PHRASES)
    is_short = len(answer.split()) < 40
    return has_negative and is_short


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _retrieved_passages_summary(retrieved: List[Tuple[Chunk, float]], limit: int = 2) -> str:
    blocks = []
    for chunk, score in retrieved[:limit]:
        blocks.append(
            f"[{chunk.get('play')}, Act {chunk.get('act')}, Scene {chunk.get('scene')} "
            f"| sim={score:.3f}] {chunk.get('scene_summary') or ''}"
        )
    return " || ".join(blocks)


def run_evaluation() -> Dict[str, Any]:
    questions = load_questions()
    if not questions:
        raise RuntimeError("No evaluation questions found.")

    print(f"Loaded {len(questions)} evaluation questions.")

    print("Building RAG system...")
    bot = RagChatbot().build()
    baseline_bot = PromptOnlyBaseline()

    print(f"Indexed {len(bot.chunks)} scene chunks using {bot.backend_name}.")

    rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []

    for q in questions:
        qid = q.get("question_id")
        query = q["question"]
        expected_focus = q.get("expected_focus", "")
        qtype = q.get("type", "concept_explanation")
        expected_play = _expected_play(q)

        retrieved = bot.retrieve(query)
        rag_result = bot.answer(query)
        rag_answer = rag_result["response"]

        baseline_answer = baseline_bot.generate(query)

        rag_generator = rag_result["generator"]

        for system_name, answer, retrieved_for_system in [
            ("baseline", baseline_answer, []),
            ("rag", rag_answer, retrieved),
        ]:
            style = score_style(answer) if qtype == "stylised_generation" else None

            if qtype == "out_of_domain":
                refused = _is_refusal(answer)
                corr = 5 if refused else 1
                useful = 5 if refused else 1
                ground = None
                ret = None
            else:
                ground = score_grounding(answer)
                ret = score_retrieval(retrieved_for_system, expected_play) if system_name == "rag" else None
                if qtype == "stylised_generation":
                    corr = None
                    useful = score_usefulness(answer, has_citation=ground == 5)
                else:
                    corr = score_correctness(answer, expected_focus)
                    useful = score_usefulness(answer, has_citation=ground == 5)

            # RAG cited a play/scene (ground >= 3) but retrieved chunks from the
            # wrong play (ret == 1) — response is not supported by what was retrieved.
            potential_hallucination = (
                system_name == "rag"
                and ground is not None
                and ret is not None
                and ground >= 3
                and ret == 1
            )

            generator_path = rag_generator if system_name == "rag" else "prompt_only_baseline"

            row = {
                "question_id": qid,
                "question": query,
                "question_type": qtype,
                "source": q.get("source", "?"),
                "expected_focus": expected_focus,
                "system": system_name,
                "generator_path": generator_path,
                "retrieved_passages": _retrieved_passages_summary(retrieved_for_system),
                "generated_response": answer.replace("\n", " | "),
                "correctness_score": corr,
                "grounding_score": ground,
                "retrieval_relevance_score": ret,
                "usefulness_score": useful,
                "style_quality_score": style,
                "potential_hallucination": potential_hallucination,
                "comments": "",
            }
            rows.append(row)
            detailed.append({
                **row,
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "score": float(score),
                        "play": chunk.get("play"),
                        "act": chunk.get("act"),
                        "scene": chunk.get("scene"),
                        "speaker": chunk.get("speaker"),
                        "scene_summary": chunk.get("scene_summary"),
                        "chunk_id": chunk.get("chunk_id"),
                    }
                    for rank, (chunk, score) in enumerate(retrieved_for_system, start=1)
                ],
            })

    # --------- Write CSV ---------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id", "question", "question_type", "source", "expected_focus",
        "system", "generator_path", "retrieved_passages", "generated_response",
        "correctness_score", "grounding_score", "retrieval_relevance_score",
        "usefulness_score", "style_quality_score", "potential_hallucination", "comments",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --------- Write JSONL ---------
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in detailed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --------- Summary ---------
    summary = _summarise(rows)
    summary["retrieval_backend"] = bot.backend_name
    summary["n_chunks_indexed"] = len(bot.chunks)
    summary["n_questions"] = len(questions)
    summary["generators_used"] = sorted({r["generator_path"] for r in rows})
    summary["potential_hallucinations"] = [
        {"question_id": r["question_id"], "system": r["system"], "question": r["question"]}
        for r in rows if r.get("potential_hallucination")
    ]
    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("Evaluation summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    flagged = summary["potential_hallucinations"]
    print()
    if flagged:
        print(f"Potential hallucinations ({len(flagged)}):")
        for h in flagged:
            print(f"  [{h['system']:8s}] {h['question_id']}: {h['question']}")
    else:
        print("No potential hallucinations detected.")

    print()
    print(f"Wrote {OUTPUT_CSV.relative_to(RESULTS_DIR.parent)}")
    print(f"Wrote {OUTPUT_JSONL.relative_to(RESULTS_DIR.parent)}")
    print(f"Wrote {OUTPUT_SUMMARY.relative_to(RESULTS_DIR.parent)}")

    return summary


_SCORE_COLS = (
    "correctness_score",
    "grounding_score",
    "retrieval_relevance_score",
    "usefulness_score",
    "style_quality_score",
)

# Preferred weights for the composite score. When a criterion is N/A for a
# subset (e.g. style is only scored for stylised questions), its weight is
# dropped and the remaining weights are renormalised so they still sum to 1.
_SCORE_WEIGHTS: Dict[str, float] = {
    "correctness_score": 0.30,
    "grounding_score": 0.25,
    "retrieval_relevance_score": 0.20,
    "usefulness_score": 0.20,
    "style_quality_score": 0.05,
}


def _bucket_stats(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
    b: Dict[str, Any] = {"n": len(subset)}
    for col in _SCORE_COLS:
        values = [r[col] for r in subset if r[col] is not None]
        b[f"mean_{col}"] = round(statistics.mean(values), 3) if values else None
        b[f"n_{col}"] = len(values)

    # Weighted composite over whichever criteria are applicable in this subset.
    applicable = {col: w for col, w in _SCORE_WEIGHTS.items() if b[f"mean_{col}"] is not None}
    if applicable:
        total_weight = sum(applicable.values())
        b["composite_score"] = round(
            sum(b[f"mean_{col}"] * w / total_weight for col, w in applicable.items()),
            3,
        )
    else:
        b["composite_score"] = None

    return b


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for system in ("baseline", "rag"):
        out[system] = _bucket_stats([r for r in rows if r["system"] == system])

    qtypes = sorted({r["question_type"] for r in rows})
    by_type: Dict[str, Any] = {}
    for qtype in qtypes:
        by_type[qtype] = {
            system: _bucket_stats(
                [r for r in rows if r["question_type"] == qtype and r["system"] == system]
            )
            for system in ("baseline", "rag")
        }
    out["by_type"] = by_type

    return out


if __name__ == "__main__":
    run_evaluation()
