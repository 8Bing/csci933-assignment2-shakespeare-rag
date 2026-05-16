<<<<<<< HEAD
# Shakespeare-Aware RAG Chatbot

CSCI433/933 Assignment 2 — A retrieval-augmented question-answering system over three Shakespeare plays (*Hamlet*, *Macbeth*, *Romeo and Juliet*).

The system retrieves relevant passages from a structured corpus and uses a locally-run small language model (Ollama) to generate beginner-friendly answers, while displaying the source evidence alongside each response.

---

## What This System Does

Given a question such as *"Why does Macbeth kill Duncan?"*, the system will:

1. Embed the question into a vector representation.
2. Retrieve the top-*k* most relevant scene-level passages from the indexed corpus.
3. Construct a prompt that combines the retrieved evidence with a system instruction.
4. Generate a beginner-friendly answer using a small local language model.
5. Display the retrieved evidence (play / act / scene) below the answer so users can verify the response.

Two systems are implemented for comparison:

- **Baseline** — prompt-only generation, no retrieval.
- **RAG** — retrieval + generation with scene-level context.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.10 and 3.11 |
| git | any recent | For cloning the repo |
| Ollama | latest | https://ollama.com — runs the local model |
| Disk space | ~3 GB | Model + embeddings + dependencies |
| RAM | 8 GB+ | 16 GB recommended for smoother inference |

> ⚠️ **Important**: keep this project outside iCloud Drive. iCloud paths contain characters (`~`, spaces, parentheses) that break LaTeX and confuse some Python tools. We recommend `~/Documents/` or `~/Projects/`.

---

## Setup

### 1. Clone the repository

```bash
cd ~/Documents
git clone git@github.com:8Bing/csci933-assignment2-shakespeare-rag.git
cd csci933-assignment2-shakespeare-rag
```

> Use SSH (`git@github.com:...`), not HTTPS. GitHub disabled password authentication for HTTPS in 2021.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

The prompt should now start with `(.venv)`.

### 3. Install Python dependencies
=======
# A Lightweight Retrieval-Augmented Shakespeare Assistant

This repository contains the working system, evaluation harness, and
LaTeX report for the CSCI433/933 Assignment 2 group project.

The system answers plain-English questions about *Hamlet*, *Macbeth*,
and *Romeo and Juliet* using a retrieval-augmented generation (RAG)
pipeline that runs entirely on a standard laptop, with no GPU and no
external API.

## Repository layout

```
assessment 2/
├── data/processed/      JSON + JSONL processed plays (input corpus)
├── prompts/             system_prompt.txt, stylised_prompt.txt
├── results/
│   ├── instructor_questions.json   5 instructor-provided questions
│   ├── group_questions.json        7 group-designed questions
│   ├── evaluation_results.csv      12 questions × 2 systems = 24 rows
│   ├── evaluation_results.jsonl    full detail incl. retrieved chunks
│   └── evaluation_summary.json     mean scores per system
├── src/
│   ├── config.py        paths, defaults
│   ├── data_loader.py   JSON → flat utterance records
│   ├── chunking.py      scene-level (default) and utterance chunkers
│   ├── retrieval.py     sentence-transformers (preferred) or TF–IDF
│   ├── generation.py    ExtractiveComposer + StylisedComposer (+ optional HF)
│   ├── baseline.py      PromptOnlyBaseline (used for the report)
│   ├── rag_chatbot.py   end-to-end RAG chatbot + CLI entry point
│   ├── build_index.py   smoke test: load → chunk → retrieve
│   └── evaluate.py      evaluation harness that writes results/*
├── report/
│   ├── assignment2_report.tex   technical report (LaTeX)
│   ├── assignment2_report.pdf   compiled report
│   └── …                        instructor template, specification PDF
├── requirements.txt
└── README.md            this file
```

## Setup
>>>>>>> 4cbfe9363745258be01b72a785a4b8d910a6bab5

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install requests
```

<<<<<<< HEAD
The first install downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) on first use.

### 4. Place the dataset files

The Shakespeare dataset is provided separately to all group members. Place the three play files here:

```
data/processed/hamlet.json
data/processed/macbeth.json
data/processed/romeo_and_juliet.json
```

And the instructor questions here:

```
results/instructor_questions.json
```

The dataset is excluded from git (see `.gitignore`) because it is provided to every student.

### 5. Install and start Ollama

Download Ollama from https://ollama.com and install. Then pull the model:

```bash
ollama pull phi3:3.8b
```

Verify the model works (optional sanity check):

```bash
ollama run phi3:3.8b "Say hello in one sentence."
# Type /bye or Ctrl+D to exit the interactive prompt.
```

Verify Ollama's API is responding:

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON response listing your installed models.

---

## Usage

### Build the retrieval index (one-time)

```bash
cd src
python build_index.py
```

This loads the three plays, creates scene-level chunks, and runs a test query (*"Why does Macbeth kill Duncan?"*) to verify retrieval is returning sensible passages.

<!-- TODO(Model Dev): once embedding cache is implemented, document where the cache is saved (e.g., results/embeddings.npy). -->

### Run the RAG chatbot

```bash
cd src
python rag_chatbot.py
```

You will be prompted for questions interactively. Type `quit` to exit.

<!-- TODO(Model Dev): add 1–2 example session transcripts here once generate_answer() is wired to Ollama. -->

### Run the baseline system

```bash
cd src
python baseline.py
```

<!-- TODO(Model Dev): document the baseline once it is implemented. -->

### Run the evaluation

```bash
cd src
python evaluate.py
```

This produces a CSV template in `results/`. Full evaluation pipeline runs both systems on all 10 evaluation questions.

<!-- TODO(Eval Lead): document the full evaluation flow once run_evaluation.py exists, including how scoring is applied. -->

---

## Project Structure

```
csci933-assignment2-shakespeare-rag/
│
├── data/
│   ├── raw/                 # Original raw text (unused; pre-processed dataset is provided)
│   └── processed/           # Place the three play JSON files here
│
├── prompts/
│   └── system_prompt.txt    # System instruction prepended to the LLM context
│
├── results/
│   ├── instructor_questions.json   # 5 instructor-provided evaluation questions
│   └── (CSV outputs written here at evaluation time)
│
├── src/
│   ├── config.py            # Paths, model names, default top-k
│   ├── data_loader.py       # Loads scene records from the play JSON files
│   ├── chunking.py          # Converts records into retrieval chunks (scene-level)
│   ├── retrieval.py         # MiniLM embeddings + sklearn cosine similarity
│   ├── build_index.py       # Sanity-check script: builds index and runs a test query
│   ├── baseline.py          # Baseline system (prompt-only, no retrieval)
│   ├── rag_chatbot.py       # Interactive RAG chatbot
│   └── evaluate.py          # Produces evaluation CSV template
│
├── report/
│   ├── main.tex             # IEEE-style LaTeX report
│   └── main.pdf             # Compiled report (committed for convenience)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Design Choices (Quick Reference)

| Decision | Choice | Justification (full text in report) |
|---|---|---|
| Chunking | Scene-level | Better context for *why/how* questions; dataset already pre-chunked at scene level |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight (~90 MB), strong retrieval performance for short queries |
| Retrieval | Cosine similarity (sklearn) | Simple, fast, transparent; no external index needed for ~73 chunks |
| Generation | Ollama + `phi3:3.8b` | Local, free, CPU-friendly; aligns with the SLM principle |
| Baseline | Prompt-only, no retrieval | Isolates the contribution of retrieval |

<!-- TODO(Data Lead): expand chunking justification once chunking.py is finalised. -->
<!-- TODO(Model Dev): add Ollama model justification details after testing. -->

---

## Development Workflow

### Branching

We work on a single `main` branch with small, frequent commits. Pull before pushing:

```bash
git pull --rebase
# make changes
git add .
git commit -m "Short imperative message: what changed"
git push
```

### What to commit

✅ Source code, prompts, configuration, the LaTeX `.tex` file, this README.

❌ Do **not** commit: virtual environments, embedding caches, dataset JSON files, LaTeX build artefacts, or `.DS_Store`. The `.gitignore` already excludes these.

### Compiling the report locally (optional)

Only required for members who want to preview the report PDF locally. Otherwise, the Integration Lead handles compilation and pushes the resulting `report/main.pdf` for everyone to view.

```bash
cd report
latexmk -pdf main.tex
```

Requires BasicTeX (`brew install --cask basictex`) plus a few extra packages:

```bash
sudo tlmgr install ieeetran enumitem microtype booktabs tabularx \
  collection-fontsrecommended chktex latexmk
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: code` | Open VS Code → Cmd+Shift+P → "Shell Command: Install 'code' command in PATH" |
| `Permission denied (publickey)` when pushing | Set up an SSH key: `ssh-keygen -t ed25519 -C "your@email"`, then add `~/.ssh/id_ed25519.pub` to GitHub → Settings → SSH keys |
| `Authentication failed for https://...` | Your remote is HTTPS; switch to SSH: `git remote set-url origin git@github.com:8Bing/csci933-assignment2-shakespeare-rag.git` |
| `File 'XXX.sty' not found` (LaTeX) | `sudo tlmgr install XXX` |
| `Filename contains character not allowed` (LaTeX) | The project is in iCloud or another path with `~`. Move it to `~/Documents/`. |
| Ollama connection refused | Make sure the Ollama app is running. On macOS, run `ollama serve` or open the Ollama menu-bar app. |

---

## Submission Checklist

Final submission package (due **Friday, 29 May 2026, 23:59**):

- [ ] Source code repository (this repo, latest commit on `main`)
- [ ] Technical report PDF (`report/main.pdf`)
- [ ] Evaluation appendix (in report)
- [ ] GenAI usage log (in Appendix D of report)
- [ ] Demo video (5–10 minutes)

---

## License and Acknowledgements

The Shakespeare source texts originate from Project Gutenberg:

- Hamlet: https://www.gutenberg.org/cache/epub/1787/pg1787.txt
- Macbeth: https://www.gutenberg.org/cache/epub/1795/pg1795.txt
- Romeo and Juliet: https://www.gutenberg.org/cache/epub/1777/pg1777.txt

The structured dataset was prepared by the course instructor for CSCI433/933 educational use.

This project is submitted as coursework and is not licensed for redistribution.
=======
`requirements.txt` lists `numpy`, `pandas`, `scikit-learn`,
`sentence-transformers`, and `tqdm`. The retrieval module automatically
falls back to a scikit-learn TF–IDF backend if
`sentence-transformers` is not installed; both backends share the same
public API.

## How to run

```bash
cd src

# 1. Smoke test: load the data, build the index, retrieve a few samples.
python build_index.py

# 2. Reproduce the evaluation reported in the technical report.
#    Writes:
#      ../results/evaluation_results.csv
#      ../results/evaluation_results.jsonl
#      ../results/evaluation_summary.json
python evaluate.py

# 3. Interactive Shakespeare chatbot. Type 'quit' to exit.
python rag_chatbot.py
```

## Headline results

Across twelve evaluation questions (1–5 rubric, higher is better):

| Criterion           | Baseline | RAG  | Δ     |
|---------------------|----------|------|-------|
| Correctness         | 2.60     | 3.00 | +0.40 |
| Grounding           | 2.33     | 4.67 | +2.33 |
| Retrieval relevance | 1.00     | 4.83 | +3.83 |
| Usefulness          | 3.00     | 4.33 | +1.33 |
| Style quality       | 3.00     | 5.00 | +2.00 |

Full discussion and failure analysis are in
`report/assignment2_report.pdf`.

## Reproducibility notes

* The TF–IDF index is rebuilt at the start of every run; on a 73-scene
  corpus this takes well under a second.
* The evaluation rubric is deterministic and implemented in
  `src/evaluate.py`. Re-running `python evaluate.py` will produce
  byte-identical CSV / JSONL outputs.
* The CLI displays retrieved evidence alongside every generated answer,
  satisfying the "grounded retrieval" requirement.
>>>>>>> 4cbfe9363745258be01b72a785a4b8d910a6bab5
