# Model Development Lead — Checklist

Your responsibility (from the assignment spec):
**Baseline model, RAG pipeline (generation step), prompt design,
optional lightweight fine-tuning.**

Everything in this file is something **you** sign off on before submission.

---

## Stack agreed with the group

| Component       | Choice                                  | Status |
|-----------------|-----------------------------------------|--------|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2  | Wired up in `src/retrieval.py`. Just needs `pip install sentence-transformers` on your laptop. |
| Language model  | `phi3:3.8b` via Ollama                  | Wired up in `src/ollama_client.py` and `src/rag_chatbot.py`. Needs `ollama pull phi3:3.8b`. |
| Baseline        | Prompt-only (no retrieval)              | `src/baseline.py` (`PromptOnlyBaseline`). |
| Top-k retrieval | 3                                       | `DEFAULT_TOP_K` in `src/config.py`. |
| Stylised max    | 150 words                               | `STYLISED_MAX_WORDS` in `src/config.py`. |

---

## One-time setup on your laptop

1. Install Ollama from https://ollama.com (Windows installer).
2. `ollama pull phi3:3.8b`            ← downloads ~2 GB of model weights.
3. Activate the project venv, then `pip install sentence-transformers`.
4. Smoke test:
   ```
   ollama run phi3:3.8b "hello"
   cd src
   python build_index.py
   python rag_chatbot.py
   ```
5. Confirm: in the chatbot, ask "Who is Hamlet?" The answer text should
   look more natural than before — that means phi3 is doing the
   generation.

---

## Iteration loop (run this every time you change a prompt)

1. Edit `prompts/system_prompt.txt`.
2. From `src/`, run:
   ```
   python evaluate.py
   ```
3. Look at the headline numbers in `results/evaluation_summary.json`.
4. Spot-check 2 questions in `results/evaluation_results.csv`
   (especially Q1 and Q5).
5. Commit:
   ```
   git add prompts/system_prompt.txt results/
   git commit -m "Prompt iteration N: <one-line summary>"
   git push
   ```

Keep a log of every prompt variation in
`docs/PROMPT_HISTORY.md` (see template below). The report Appendix B
will reference this file.

---

## Prompt design ideas (try in order)

1. **Strict grounding** — "If the retrieved passages do not mention X,
   you must say 'I cannot find this in the retrieved scenes.'"
2. **Beginner-friendly framing** — "Explain like the user has never
   read Shakespeare. Avoid quoting Early-Modern English unless you
   explain what it means."
3. **Forced citation** — "End every claim with (Play, Act X, Scene Y)."
4. **Word budget** — "Keep answers under 120 words for concept
   questions, under 180 words for 'why' questions."
5. **Stylised gating** — "Only switch to Shakespearean register if
   the user explicitly asks for a 'stylised', 'in the style of', or
   'Shakespearean' response."

Each variation should be a single small change so the effect on
evaluation scores is interpretable.

---

## Required report content (your section)

In `report/assignment2_report.tex`:

| Section               | What to write                                                                 |
|-----------------------|-------------------------------------------------------------------------------|
| 3.1 Baseline System   | Why a prompt-only baseline. What it has access to. Why it is fair.            |
| 3.2 RAG-Based System  | Embedding model, retriever, top-k, prompt structure, generation model.        |
| 3.3 Model Choice      | Why phi3:3.8b. SLM principles: <8B params, local, free, deterministic.        |
| Appendix B            | Final system_prompt.txt + 1-paragraph history of evolution.                   |

A reasonable target word count for sections 3.1–3.3 is 600–900 words.

---

## "Done" definition for your role

- [ ] Ollama + phi3:3.8b runs on your laptop.
- [ ] `python rag_chatbot.py` answers questions using phi3.
- [ ] `python evaluate.py` produces fresh CSV + summary using phi3.
- [ ] Prompt iterations logged in `docs/PROMPT_HISTORY.md`.
- [ ] Sections 3.1, 3.2, 3.3 of the report finalised.
- [ ] Appendix B of the report carries the final prompt.
- [ ] One paragraph added to the GenAI usage log describing your AI
      usage for prompt drafting.
- [ ] All changes pushed to GitHub.
