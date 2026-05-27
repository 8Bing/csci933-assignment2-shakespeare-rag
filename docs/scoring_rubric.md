## Shakespeare RAG System — Manual Evaluation

---

## Scoring Rubric

Each criterion is scored 1–5. Descriptions below define what each score means. Apply these consistently across all questions.

---

### Correctness
*Does the response accurately convey the facts of the play?*

| Score | Meaning |
|-------|---------|
| **1** | Invents information and is completely off topic |
| **2** | Contains factual errors or invents information |
| **3** | Contains factual errors but invents nothing |
| **4** | Mostly accurate, only minor incorrect or incomplete details |
| **5** | Completely accurate |

> **Note:** Do not score Correctness for `stylised_generation` questions — mark as **N/A**.

---

### Grounding
*Is the response supported by retrieved passages, or does it generate unsupported claims?*

| Score | Meaning |
|-------|---------|
| **1** | Invents passages that do not exist |
| **2** | Uses incorrect passages |
| **3** | Uses some correct passages but also ignores or mixes in non-retrieved information |
| **4** | Mostly grounded in retrieved passages, with minor details not directly from them |
| **5** | Completely grounded in retrieved passages |

> **Note:** For the Baseline (no retrieval), score Grounding based on whether the response invents specific scenes/quotes or stays at a general level.

---

### Retrieval Relevance
*Were the passages retrieved actually the right ones for the question?*

| Score | Meaning |
|-------|---------|
| **1** | Retrieved passages are completely irrelevant to the question |
| **2** | Passages are from the correct play but wrong topic or character |
| **3** | Passages are partially relevant but missing key context |
| **4** | Passages are relevant but not the most precise or complete for the question |
| **5** | Retrieved passages are exactly the most relevant for the question |

> **Note:** Mark as **N/A** for all Baseline rows (no retrieval takes place).

> **Auto-scoring note:** The automated `score_retrieval()` produces only 1 / 3 / 5:
> - **5** = exact act + scene match (for `evidence_retrieval`) or top chunk from correct play (all other types)
> - **3** = correct play retrieved but wrong scene (for `evidence_retrieval`), or any chunk from correct play (other types)
> - **1** = wrong play or nothing retrieved
>
> Scores of **2** and **4** require manual review — e.g. "right play but completely off-topic" → 2; "right scene but partially incomplete retrieval" → 4. Manual adjustments may be recorded in the `comments` column; final summary means should reflect the adjusted values.

---

### Usefulness
*How useful is this response to someone with no prior knowledge of Shakespeare?*

| Score | Meaning |
|-------|---------|
| **1** | Incomprehensible or completely unhelpful for a beginner |
| **2** | Very hard to understand; uses technical language or assumes prior knowledge |
| **3** | Partially useful — explains some things but leaves parts unexplained or confusing |
| **4** | Mostly clear and useful, with one term or concept that could be better explained |
| **5** | Completely clear, accessible, and useful for someone with no prior knowledge |

---

### Style Quality
*For stylised generation questions only — does the response successfully adopt Shakespearean style?*

| Score | Meaning |
|-------|---------|
| **1** | No attempt at Shakespearean style, or completely incomprehensible |
| **2** | Attempts the style but fails — sounds forced or artificial and is hard to understand |
| **3** | Has some Shakespearean elements but is inconsistent or partially confusing |
| **4** | Good Shakespearean style and mostly comprehensible, with minor confusing moments |
| **5** | Reflects Shakespearean tone well and is completely comprehensible |

> **Note:** Only score this criterion for questions of type `stylised_generation` (Q5, G7, G10). Mark as **N/A** for all other rows.

---

## Quick Reference Card

| Criterion | Applies to | Baseline | RAG |
|-----------|-----------|----------|-----|
| Correctness | All except `stylised_generation` | Score 1–5 | Score 1–5 |
| Grounding | All | Score 1–5 | Score 1–5 |
| Retrieval Relevance | All | **N/A** | Score 1–5 |
| Usefulness | All | Score 1–5 | Score 1–5 |
| Style Quality | `stylised_generation` only | Score 1–5 | Score 1–5 |

---

