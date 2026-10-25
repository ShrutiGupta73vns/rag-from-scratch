"""Ties the steps together.

Run from the project root:
    python -m src.pipeline "What is the default TTL for Halcyon's shared Redis cache?"
"""
import sys

from src import config
from src.chunk import chunk_pages
from src.embed import embed
from src.generate import generate
from src.load import load_pdfs
from src.retrieve import retrieve
from src.store import NumpyStore


def build_index() -> NumpyStore:
    """Indexing (offline): load -> chunk -> embed -> store."""
    chunks = chunk_pages(load_pdfs(config.DATA_DIR))
    store = NumpyStore()
    store.add(chunks, embed([c["text"] for c in chunks]))
    return store


def answer(question: str, store: NumpyStore) -> dict:
    """Querying (online): retrieve -> generate. Returns the answer and the chunks used."""
    results = retrieve(question, store)
    return {"answer": generate(question, results), "results": results}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ")
    out = answer(question, build_index())
    print(out["answer"])
    print("\nSources:")
    for r in out["results"]:
        print(f"  {r['score']:.3f}  {r['id']}")
