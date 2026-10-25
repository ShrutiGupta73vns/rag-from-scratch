"""All settings in one place, so experiments change one line instead of hunting through code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root, wherever the code is run from
DATA_DIR = ROOT / "data" / "raw"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3.2"

CHUNK_SIZE = 150  # words per chunk
CHUNK_OVERLAP = 30  # words shared by neighbouring chunks
TOP_K = 5  # chunks retrieved per question
TEMPERATURE = 0
