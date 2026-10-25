"""Step 4b: question -> top-k chunks."""
from src import config
from src.embed import embed
from src.store import NumpyStore


def retrieve(question: str, store: NumpyStore, k: int = config.TOP_K) -> list[dict]:
    query_vector = embed([question])[0]  # same model as the chunks, or the vectors can't be compared
    return store.search(query_vector, k)
