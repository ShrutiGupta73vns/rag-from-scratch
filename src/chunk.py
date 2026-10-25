"""Step 2: pages -> fixed-size word chunks with overlap."""
from src import config


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of `chunk_size` words; neighbours share `overlap` words."""
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):  # this chunk reached the end
            break
    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Chunk every page; each chunk keeps its page's metadata plus a unique id."""
    chunks = []
    for p in pages:
        for i, text in enumerate(chunk_text(p["text"])):
            chunks.append({
                "id": f"{p['source']}-p{p['page']}-c{i}",
                "text": text,
                "source": p["source"],
                "page": p["page"],
            })
    return chunks
