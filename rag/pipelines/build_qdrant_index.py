from __future__ import annotations

import argparse
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

from rag.pipelines.paths import PROCESSED_DIR, QDRANT_DIR, ensure_rag_dirs


@dataclass(frozen=True)
class ChunkRecord:
    text: str
    metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk processed markdown files and build a local Qdrant vector database."
    )
    parser.add_argument(
        "--markdown-dir",
        default=str(PROCESSED_DIR),
        help="Directory containing processed markdown files. Defaults to rag/processed.",
    )
    parser.add_argument(
        "--qdrant-dir",
        default=str(QDRANT_DIR),
        help="Directory for the local Qdrant data. Defaults to rag/qdrant.",
    )
    parser.add_argument(
        "--collection-name",
        default="markdown_rag",
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL for the OpenAI-compatible embeddings endpoint.",
    )
    parser.add_argument(
        "--embedding-model",
        required=True,
        help="Embedding model name served by the endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "no-key-needed"),
        help="API key for the endpoint. Defaults to OPENAI_API_KEY or 'no-key-needed'.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the collection if it already exists.",
    )
    return parser


def collect_markdown_files(markdown_dir: Path) -> list[Path]:
    if not markdown_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {markdown_dir}")

    markdown_files = sorted(markdown_dir.glob("*.md"))
    if not markdown_files:
        raise RuntimeError(f"No markdown files found in {markdown_dir}")
    return markdown_files


def extract_page_numbers(value: Any) -> list[int]:
    page_numbers: set[int] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key == "page_no" and isinstance(item, int):
                    page_numbers.add(item)
                else:
                    visit(item)
            return
        if isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    return sorted(page_numbers)


def chunk_to_metadata(markdown_path: Path, chunk: Any, chunk_index: int) -> dict[str, Any]:
    dump: Any = {}
    if hasattr(chunk, "model_dump"):
        try:
            dump = chunk.model_dump(mode="json")
        except TypeError:
            dump = chunk.model_dump()

    headings = list(getattr(chunk, "headings", []) or [])
    captions = list(getattr(chunk, "captions", []) or [])

    return {
        "source_path": str(markdown_path),
        "source_file": markdown_path.name,
        "document_id": markdown_path.stem,
        "chunk_index": chunk_index,
        "chunk_type": type(chunk).__name__,
        "headings": headings,
        "captions": captions,
        "page_numbers": extract_page_numbers(dump),
    }


def chunk_markdown_file(markdown_path: Path) -> list[ChunkRecord]:
    try:
        from docling.chunking import HybridChunker
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError(
            "Docling is required for chunking markdown. Install docling in the active environment."
        ) from exc

    converter = DocumentConverter()
    doc = converter.convert(source=str(markdown_path)).document
    chunker = HybridChunker()

    records: list[ChunkRecord] = []
    for chunk_index, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        text = chunker.contextualize(chunk).strip()
        if not text:
            continue
        records.append(
            ChunkRecord(
                text=text,
                metadata=chunk_to_metadata(markdown_path, chunk, chunk_index),
            )
        )

    return records


def load_chunk_records(markdown_files: list[Path]) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for markdown_path in markdown_files:
        records.extend(chunk_markdown_file(markdown_path))
    if not records:
        raise RuntimeError("Chunking completed, but no non-empty chunks were produced.")
    return records


def build_embeddings(base_url: str, api_key: str, embedding_model: str) -> Any:
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai is required for embeddings. Install langchain-openai in the active environment."
        ) from exc

    return OpenAIEmbeddings(
        model=embedding_model,
        base_url=base_url,
        api_key=api_key,
    )


def upsert_chunks(
    *,
    records: list[ChunkRecord],
    qdrant_dir: Path,
    collection_name: str,
    embeddings: Any,
    recreate: bool,
) -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required for local vector storage. Install qdrant-client in the active environment."
        ) from exc

    texts = [record.text for record in records]
    vectors = embeddings.embed_documents(texts)
    if not vectors:
        raise RuntimeError("Embedding step returned no vectors.")

    qdrant_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_dir))

    collection_exists = False
    try:
        client.get_collection(collection_name)
        collection_exists = True
    except Exception:
        collection_exists = False

    vector_size = len(vectors[0])
    if recreate and collection_exists:
        client.delete_collection(collection_name=collection_name)
        collection_exists = False

    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    points = []
    for record, vector in zip(records, vectors, strict=True):
        chunk_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{record.metadata['source_path']}#{record.metadata['chunk_index']}",
            )
        )
        payload = {
            "text": record.text,
            **record.metadata,
        }
        points.append(
            models.PointStruct(
                id=chunk_id,
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
    )


def main() -> None:
    ensure_rag_dirs()
    args = build_parser().parse_args()
    markdown_dir = Path(args.markdown_dir).expanduser().resolve()
    qdrant_dir = Path(args.qdrant_dir).expanduser().resolve()

    markdown_files = collect_markdown_files(markdown_dir)
    records = load_chunk_records(markdown_files)
    embeddings = build_embeddings(
        base_url=args.base_url,
        api_key=args.api_key,
        embedding_model=args.embedding_model,
    )
    upsert_chunks(
        records=records,
        qdrant_dir=qdrant_dir,
        collection_name=args.collection_name,
        embeddings=embeddings,
        recreate=args.recreate,
    )
    print(
        f"Indexed {len(records)} chunks from {len(markdown_files)} markdown files into "
        f"{qdrant_dir} (collection: {args.collection_name})."
    )


if __name__ == "__main__":
    main()
