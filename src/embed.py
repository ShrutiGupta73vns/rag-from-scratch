"""Step 3: text -> normalized vectors."""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once; later calls reuse it."""
    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Return one normalized vector per text, shape (len(texts), 384)."""
    return get_model().encode(texts, normalize_embeddings=True)
