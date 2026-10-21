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
- `src/`: load, chunk, embed, store, retrieve, generate, pipeline
- `eval/`: test questions
- `notes/interview.md`: concepts in the user's own words, for interview revision; add each new concept here as it's covered
