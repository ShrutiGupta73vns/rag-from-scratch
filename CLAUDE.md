# RAG Practice

Learning project: build a RAG pipeline by hand to understand the internals, using open-source tools. The user knows RAG from Salesforce Data Cloud, where it all comes out of the box, and is new to the code side. **Goal: be able to ace RAG/AI interviews**, so cover the concepts behind each step, not just working code.

## Guidelines
- Keep answers short (to save tokens) and explain *why* each step exists.
- Build each step by hand before introducing frameworks.
- At the end of each turn, list what was done and why (with commands), so the user can learn it.
- The user is also learning Python: explain the syntax of new code line by line (imports, comprehensions, f-strings, etc.), not only what it does.
- Point out likely interview questions and trade-offs (e.g. brute-force vs ANN/HNSW/IVF/PQ, recall vs speed).

## Environment
- Windows, Python 3.11.4, VS Code
- `.venv` set up with requirements.txt; Ollama 0.40.6 with `llama3.2` pulled (2026-10-17)

## Roadmap
1. **From scratch:** pypdf → chunking → sentence-transformers embeddings → numpy cosine search → Ollama
2. **Tools:** Chroma/Qdrant, then LlamaIndex/LangChain
3. **Better retrieval:** hybrid (BM25 + vector), reranking, chunking strategies, query rewriting
4. **Evaluation:** RAGAS/DeepEval with a Q&A test set
5. **Advanced:** agentic RAG, GraphRAG, FastAPI + Streamlit

Corpus idea: public Salesforce/Data Cloud docs.

## Structure
- `data/raw/`: source docs
- `notebooks/`: experiments
- `src/`: config, load, chunk, embed, store (NumpyStore: add/search), retrieve, generate, pipeline. Run: `.venv\Scripts\python.exe -m src.pipeline "question"`
- `eval/`: questions.json + run_eval.py (`python -m eval.run_eval <run_name>` → eval/results/<run_name>.json). The llama3.2 judge gives false negatives, so check failures by hand.
- Baseline (2026-10-25, numpy, 150/30 chunks, k=5): retrieval 22/22, answers 21/24 reported, 23/24 after hand-check (q10, q16 judge errors; q21 partial)
- `notes/interview.md`: concepts in the user's own words, for interview revision. It has an "Interview Questions" section (numbered Q1, Q2, ... with short key points, grouped by topic); whenever a new interview-worthy concept comes up, add a question there without being asked. This file is git-ignored.
