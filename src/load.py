"""Step 1: PDFs -> one record per page, with metadata."""
from pathlib import Path

from pypdf import PdfReader


def load_pdfs(folder: Path) -> list[dict]:
    """Return one record per PDF page: {text, source, page}."""
    pages = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""  # scanned/image pages return None
            pages.append({"text": text, "source": pdf_path.name, "page": page_num})
    return pages
