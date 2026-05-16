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

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

Across twelve evaluation questions (1–5 rubric, higher is better) with phi3:3.8b as the generator and MiniLM as the embedder:

| Criterion           | Baseline | RAG  | Δ     |
|---------------------|----------|------|-------|
| Correctness         | 2.60     | 4.20 | +1.60 |
| Grounding           | 2.33     | 3.00 | +0.67 |
| Retrieval relevance | 1.00     | 4.83 | +3.83 |
| Usefulness          | 3.00     | 3.17 | +0.17 |
| Style quality       | 3.00     | 3.00 | 0.00  |

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
