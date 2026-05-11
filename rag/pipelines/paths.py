from __future__ import annotations

from pathlib import Path

RAG_DIR = Path(__file__).resolve().parents[1]
PDFS_DIR = RAG_DIR / "pdfs"
PROCESSED_DIR = RAG_DIR / "processed"
QDRANT_DIR = RAG_DIR / "qdrant"


def ensure_rag_dirs() -> None:
    for path in (PDFS_DIR, PROCESSED_DIR, QDRANT_DIR):
        path.mkdir(parents=True, exist_ok=True)
