from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

from rag.pipelines.convert_pdf_to_markdown import ConversionConfig, process_pdf
from rag.pipelines.paths import PDFS_DIR, PROCESSED_DIR, ensure_rag_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-convert PDFs into markdown files for the local RAG pipeline."
    )
    parser.add_argument(
        "--input-dir",
        default=str(PDFS_DIR),
        help="Directory containing source PDFs. Defaults to rag/pdfs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_DIR),
        help="Directory for generated markdown files. Defaults to rag/processed.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL for the OpenAI-compatible VLM endpoint.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model alias/name served by the VLM endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "no-key-needed"),
        help="API key for the endpoint. Defaults to OPENAI_API_KEY or 'no-key-needed'.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep extracted figures, rendered pages, and JSON manifests for each PDF.",
    )
    return parser


def main() -> None:
    ensure_rag_dirs()
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input PDF directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {input_dir}")

    for pdf_path in pdf_paths:
        output_path = process_pdf(
            ConversionConfig(
                base_url=args.base_url,
                model=args.model,
                pdf_path=pdf_path,
                output_dir=output_dir,
                api_key=args.api_key,
                debug=args.debug,
            )
        )
        print(output_path)


if __name__ == "__main__":
    main()
