"""Step 4a: the "vector DB" -- vectors + their chunks, with brute-force search.

Phase 2 swaps this for Chroma behind the same two methods (add, search),
so nothing else in the pipeline has to change.
"""
import numpy as np


class NumpyStore:
    def __init__(self):
        self.chunks: list[dict] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)

    def add(self, chunks: list[dict], vectors: np.ndarray) -> None:
        """Store chunks and their vectors; row i of vectors belongs to chunks[i]."""
        self.chunks = list(chunks)
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        """Return the k most similar chunks, each with a "score"."""
        scores = self.vectors @ query_vector  # cosine similarity (vectors are normalized)
        top_idx = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top_idx]
