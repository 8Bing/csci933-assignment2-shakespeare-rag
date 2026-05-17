# Shakespeare-Aware RAG Chatbot

CSCI933 Assignment 2 — A retrieval-augmented question-answering system over three Shakespeare plays (*Hamlet*, *Macbeth*, *Romeo and Juliet*).

The system answers plain-English questions using a retrieval-augmented generation (RAG) pipeline that runs entirely on a standard laptop — no GPU, no external API.

---

## What This System Does

Given a question such as *"Why does Macbeth kill Duncan?"*, the system will:

1. Embed the question into a vector representation (sentence-transformers, with a TF–IDF fallback).
2. Retrieve the top-*k* most relevant scene-level passages from the indexed corpus (73 scenes total).
3. Construct a prompt that combines the retrieved evidence with a system instruction.
4. Generate a beginner-friendly answer using a small local language model.
5. Display the retrieved evidence (play / act / scene) below the answer so users can verify the response.

Two systems are implemented for comparison:

- **Baseline** — prompt-only generation, no retrieval (`PromptOnlyBaseline`).
- **RAG** — retrieval + generation (`ExtractiveComposer` + `StylisedComposer`, with optional Ollama / HuggingFace backends).

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.10 and 3.11 |
| git | any recent | For cloning the repo |
| Ollama (optional) | latest | https://ollama.com — runs the local `phi3:3.8b` model |
| Disk space | ~3 GB | Model + embeddings + dependencies |
| RAM | 8 GB+ | 16 GB recommended for smoother inference |

---

## Setup

### 1. Clone the repository

```bash
cd ~/Documents
git clone git@github.com:8Bing/csci933-assignment2-shakespeare-rag.git
cd csci933-assignment2-shakespeare-rag
```

> Use SSH (`git@github.com:...`), not HTTPS.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install requests
```

`requirements.txt` lists `numpy`, `pandas`, `scikit-learn`, `sentence-transformers`, and `tqdm`. The retrieval module automatically falls back to a scikit-learn TF–IDF backend if `sentence-transformers` is not installed.

### 4. Place the dataset files

The Shakespeare dataset is provided separately. Place the JSON and JSONL play files here:

```
data/processed/hamlet.json
data/processed/macbeth.json
data/processed/romeo_and_juliet.json
data/processed/*_scene_chunks.jsonl    (optional, used by some chunkers)
data/processed/*_utterances.jsonl      (optional)
```

The dataset is excluded from git (see `.gitignore`).

### 5. (Optional) Install Ollama for local generation

```bash
# Install Ollama from https://ollama.com, then:
ollama pull phi3:3.8b

# Sanity check
ollama run phi3:3.8b "Say hello in one sentence."
```

Verify the API is running:

```bash
curl http://localhost:11434/api/tags
```

---

## How to Run

```bash
cd src

# 1. Smoke test — load corpus, build index, retrieve a few samples.
python build_index.py

# 2. Reproduce the full evaluation reported in the technical report.
#    Writes:
#      ../results/evaluation_results.csv
#      ../results/evaluation_results.jsonl
#      ../results/evaluation_summary.json
python evaluate.py

# 3. Interactive chatbot. Type 'quit' to exit.
python rag_chatbot.py
```

---

## Headline Results

Across 12 evaluation questions (5 instructor-provided + 7 group-designed), with `phi3:3.8b` as the generator and MiniLM as the embedder. 1–5 rubric, higher is better.

| Criterion | Baseline | RAG | Δ |
|---|---|---|---|
| Correctness | 2.60 | 4.20 | +1.60 |
| Grounding | 2.33 | 3.00 | +0.67 |
| Retrieval relevance | 1.00 | 4.83 | +3.83 |
| Usefulness | 3.00 | 3.17 | +0.17 |
| Style quality | 3.00 | 3.00 | 0.00 |

Full discussion and failure analysis are in `report/assignment2_report.pdf`.

---

## Project Structure

```
csci933-assignment2-shakespeare-rag/
│
├── data/processed/      JSON + JSONL processed plays (input corpus)
│
├── prompts/
│   ├── system_prompt.txt        Default system instruction for RAG generation
│   └── stylised_prompt.txt      Prompt for Shakespearean-style responses
│
├── results/
│   ├── instructor_questions.json    5 instructor-provided questions
│   ├── group_questions.json         7 group-designed questions
│   ├── evaluation_results.csv       Per-question scores per system
│   ├── evaluation_results.jsonl     Full detail including retrieved chunks
│   └── evaluation_summary.json      Aggregate mean scores per system
│
├── src/
│   ├── config.py            Paths, default top-k, model names
│   ├── data_loader.py       JSON → flat utterance records
│   ├── chunking.py          Scene-level (default) and utterance chunkers
│   ├── retrieval.py         sentence-transformers (preferred) or TF–IDF fallback
│   ├── generation.py        ExtractiveComposer + StylisedComposer (+ optional HF / Ollama)
│   ├── baseline.py          PromptOnlyBaseline (prompt-only, no retrieval)
│   ├── ollama_client.py     HTTP client for the local Ollama API
│   ├── rag_chatbot.py       End-to-end RAG chatbot + interactive CLI
│   ├── build_index.py       Smoke test: load → chunk → retrieve
│   └── evaluate.py          Evaluation harness writing to results/
│
├── report/
│   ├── assignment2_report.tex                 Working LaTeX source (IEEE conference format)
│   └── assignment2_report.pdf                 Compiled report (committed at milestones only)
│
├── docs/
│   ├── MODEL_LEAD_CHECKLIST.md  Internal: Model Dev working notes
│   └── PROMPT_HISTORY.md        Internal: prompt iterations and rationale
│
├── requirements.txt
├── .gitignore
└── README.md            This file
```

---

## Design Choices (Quick Reference)

| Decision | Choice | Justification (full text in report) |
|---|---|---|
| Chunking | Scene-level (73 chunks) | Better context for *why/how* questions; dataset already pre-chunked at scene level |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, TF–IDF fallback | Lightweight (~90 MB), works without GPU; TF–IDF makes the system robust if dependencies fail |
| Retrieval | Cosine similarity | Simple, fast, transparent; no external index needed for 73 chunks |
| Generation | Extractive / Stylised composer, optional Ollama `phi3:3.8b` | Composers ensure deterministic output; Ollama for fully generative answers |
| Baseline | Prompt-only, no retrieval | Isolates the contribution of retrieval |
| Evaluation | 12 questions (5 instructor + 7 group), 1–5 rubric, 5 criteria | Meets spec; covers concept QA, contextual QA, evidence retrieval, stylised generation |

---

## Reproducibility Notes

- The TF–IDF index is rebuilt at the start of every run; on a 73-scene corpus this takes well under a second.
- The evaluation rubric is deterministic and implemented in `src/evaluate.py`. Re-running `python evaluate.py` produces byte-identical CSV / JSONL outputs.
- The CLI displays retrieved evidence alongside every generated answer, satisfying the "grounded retrieval" requirement.

---

## Development Workflow

### Branching

We work on a single `main` branch with small, frequent commits. Pull before pushing:

```bash
git pull --rebase
# make changes
git add <specific files>
git commit -m "Short imperative message: what changed"
git push
```

### What to commit

✅ Source code, prompts, configuration, the LaTeX `.tex` file, this README.

❌ Do **not** commit: virtual environments, embedding caches, dataset JSON files, LaTeX build artefacts (`*.aux`, `*.log`, etc.), or `.DS_Store`. The `.gitignore` already excludes these.

`report/assignment2_report.pdf` is also ignored by default. Force-add it only at milestones:

```bash
git add -f report/assignment2_report.pdf
git commit -m "Milestone PDF: <what milestone>"
```

### Compiling the report locally (optional)

Only required for members who want to preview the report locally. The Integration Lead handles compilation and pushes the resulting PDF at milestones.

```bash
cd report
latexmk -pdf assignment2_report.tex
```

Requires BasicTeX (`brew install --cask basictex`) plus extras:

```bash
sudo tlmgr install ieeetran enumitem microtype booktabs tabularx \
  collection-fontsrecommended chktex latexmk
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: code` | Open VS Code → `Cmd+Shift+P` → "Shell Command: Install 'code' command in PATH" |
| `Permission denied (publickey)` when pushing | Set up an SSH key: `ssh-keygen -t ed25519 -C "your@email"`, then add `~/.ssh/id_ed25519.pub` to GitHub → Settings → SSH keys |
| `Authentication failed for https://...` | Your remote is HTTPS; switch to SSH: `git remote set-url origin git@github.com:8Bing/csci933-assignment2-shakespeare-rag.git` |
| `File 'XXX.sty' not found` (LaTeX) | `sudo tlmgr install XXX` |
| Ollama connection refused | Make sure the Ollama app is running. On macOS, run `ollama serve` or open the Ollama menu-bar app. |
| `divergent branches` on `git pull` | Run `git config --global pull.rebase false`, then `git pull` again. Resolve conflicts in VS Code. |

---

## Submission Checklist

Final submission package (due **Friday, 29 May 2026, 23:59**):

- [ ] Source code repository (this repo, latest commit on `main`)
- [ ] Technical report PDF (`report/assignment2_report.pdf`)
- [ ] Evaluation appendix (in report)
- [ ] GenAI usage log (in Appendix D of report)
- [ ] Demo video (5–10 minutes)

---

## License and Acknowledgements

The Shakespeare source texts originate from Project Gutenberg:

- Hamlet: https://www.gutenberg.org/cache/epub/1787/pg1787.txt
- Macbeth: https://www.gutenberg.org/cache/epub/1795/pg1795.txt
- Romeo and Juliet: https://www.gutenberg.org/cache/epub/1777/pg1777.txt

The structured dataset was prepared by the course instructor for CSCI933 educational use.

This project is submitted as coursework and is not licensed for redistribution.
