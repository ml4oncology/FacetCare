from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

MIN_FIGURE_SIZE_PX = 150
MIN_VECTOR_DRAWINGS_PER_PAGE = 12
MIN_VECTOR_DRAWINGS_PER_CLUSTER = 3
MIN_VECTOR_CLUSTER_AREA_RATIO = 0.015
VECTOR_CLUSTER_MARGIN_PX = 12
VECTOR_RENDER_SCALE = 2.5
PAGE_RENDER_SCALE = 2.0
MIN_PAGE_DRAWINGS_FOR_VLM = 8
MIN_SHORT_BLOCKS_FOR_VLM = 8
MAX_AVG_WORDS_PER_BLOCK_FOR_LAYOUT_PAGE = 18


@dataclass(frozen=True)
class ConversionConfig:
    base_url: str
    model: str
    pdf_path: Path
    output_dir: Path
    api_key: str = "no-key-needed"
    debug: bool = False


@dataclass(frozen=True)
class FigureRecord:
    index: int
    image_path: Path
    description: str
    sent_to_vlm: bool
    source: str
    page_number: int | None = None


@dataclass(frozen=True)
class PageRecord:
    page_number: int
    text_markdown: str
    visual_summary: str | None
    visual_summary_generated: bool
    figures: list[FigureRecord]
    drawings_count: int
    text_block_count: int
    short_block_count: int


@dataclass(frozen=True)
class WorkingDirs:
    root_dir: Path
    figures_dir: Path
    pages_dir: Path
    debug_dir: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a PDF into page-anchored markdown with optional debug artifacts."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL for the OpenAI-compatible VLM endpoint, for example http://localhost:881/v1",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model alias/name served by the VLM endpoint.",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the markdown file will be saved.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "no-key-needed"),
        help="API key for the endpoint. Defaults to OPENAI_API_KEY or 'no-key-needed'.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep debug artifacts such as extracted figures, rendered pages, and JSON manifests.",
    )
    return parser


def parse_args() -> ConversionConfig:
    args = build_parser().parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF input is supported right now: {pdf_path}")

    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    return ConversionConfig(
        base_url=args.base_url,
        model=args.model,
        pdf_path=pdf_path,
        output_dir=output_dir,
        api_key=args.api_key,
        debug=args.debug,
    )


def create_client(config: ConversionConfig) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai is required for PDF conversion. Install openai in the active environment."
        ) from exc

    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
    )


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def markdown_output_path(pdf_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{pdf_path.stem}.md"


def debug_output_dir(pdf_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{pdf_path.stem}_debug"


def figures_manifest_output_path(pdf_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{pdf_path.stem}.figures.json"


def figures_gallery_output_path(pdf_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{pdf_path.stem}.figures.md"


def pages_manifest_output_path(pdf_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{pdf_path.stem}.pages.json"


@contextmanager
def working_dirs(config: ConversionConfig) -> Iterator[WorkingDirs]:
    if config.debug:
        root_dir = debug_output_dir(config.pdf_path, config.output_dir)
        figures_dir = root_dir / "figures"
        pages_dir = root_dir / "pages"
        figures_dir.mkdir(parents=True, exist_ok=True)
        pages_dir.mkdir(parents=True, exist_ok=True)
        yield WorkingDirs(
            root_dir=root_dir,
            figures_dir=figures_dir,
            pages_dir=pages_dir,
            debug_dir=root_dir,
        )
        return

    with tempfile.TemporaryDirectory(prefix=f"{config.pdf_path.stem}_", dir=str(config.output_dir)) as tmp_dir:
        root_dir = Path(tmp_dir)
        figures_dir = root_dir / "figures"
        pages_dir = root_dir / "pages"
        figures_dir.mkdir(parents=True, exist_ok=True)
        pages_dir.mkdir(parents=True, exist_ok=True)
        yield WorkingDirs(
            root_dir=root_dir,
            figures_dir=figures_dir,
            pages_dir=pages_dir,
            debug_dir=None,
        )


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def is_meaningful_figure(image_path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for PDF conversion. Install pillow in the active environment."
        ) from exc

    with Image.open(image_path) as img:
        width, height = img.size
    return width >= MIN_FIGURE_SIZE_PX and height >= MIN_FIGURE_SIZE_PX


def prepare_figure_for_vlm(image: Any) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for PDF conversion. Install pillow in the active environment."
        ) from exc

    prepared = image.convert("RGB")
    width, height = prepared.size
    longest_side = max(width, height)

    if longest_side < 1400:
        scale = min(2.0, 1400 / max(longest_side, 1))
        prepared = prepared.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    return prepared


def rect_area(rect: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = rect
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def expand_rect(
    rect: tuple[float, float, float, float],
    margin: float,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    return (
        max(0.0, x0 - margin),
        max(0.0, y0 - margin),
        min(page_width, x1 + margin),
        min(page_height, y1 + margin),
    )


def rects_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


def merge_rects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def cluster_rects(
    rects: list[tuple[float, float, float, float]],
    *,
    page_width: float,
    page_height: float,
    margin: float,
) -> list[tuple[tuple[float, float, float, float], int]]:
    clusters: list[dict[str, Any]] = []

    for rect in rects:
        expanded = expand_rect(rect, margin, page_width, page_height)
        merged = False

        for cluster in clusters:
            if rects_intersect(expanded, cluster["expanded_rect"]):
                cluster["rect"] = merge_rects(cluster["rect"], rect)
                cluster["expanded_rect"] = expand_rect(
                    cluster["rect"],
                    margin,
                    page_width,
                    page_height,
                )
                cluster["count"] += 1
                merged = True
                break

        if not merged:
            clusters.append(
                {
                    "rect": rect,
                    "expanded_rect": expanded,
                    "count": 1,
                }
            )

    return [(cluster["rect"], cluster["count"]) for cluster in clusters]


def describe_figure(
    *,
    client: Any,
    model: str,
    image_path: Path,
    figure_index: int,
) -> str:
    image_b64 = encode_image_base64(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are describing figures for a retrieval pipeline. "
                    "Be precise and exhaustive. "
                    "If the figure is a flowchart, process diagram, decision tree, timeline, "
                    "or architecture diagram, explicitly describe how the elements are connected, "
                    "the direction of arrows, branch conditions, loops, start/end points, "
                    "and the order of steps. "
                    "List all legible labels and text found inside the figure. "
                    "If any text is hard to read, say that it is partially unreadable rather than omitting it. "
                    "Do not hallucinate content that is not visible."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Describe figure {figure_index} in markdown for downstream retrieval.\n"
                            "Include:\n"
                            "1. A one-paragraph overview.\n"
                            "2. A bullet list of the major visual elements.\n"
                            "3. A bullet list of relationships or flow between elements.\n"
                            "4. A bullet list of all visible labels/text."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            },
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def describe_page_visuals(
    *,
    client: Any,
    model: str,
    image_path: Path,
    page_number: int,
) -> str:
    image_b64 = encode_image_base64(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are analyzing a PDF page for semantic retrieval. "
                    "Focus on visual meaning, not just OCR. "
                    "Describe flowcharts, tables, diagrams, relationships, arrows, branches, "
                    "hierarchy, visual layout, and the role those visuals play on the page. "
                    "Ignore decorative elements. "
                    "If the page has no meaningful non-text visual content, say so plainly. "
                    "Do not hallucinate invisible content."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Analyze page {page_number} of a PDF and produce a markdown summary.\n"
                            "Include:\n"
                            "1. A short paragraph describing the important visual content on the page.\n"
                            "2. A bullet list of any diagrams, flowcharts, tables, or structured layouts.\n"
                            "3. For any flowchart or process diagram, explain the flow, branches, and key labels.\n"
                            "4. If there is no meaningful visual content beyond regular text, say that clearly."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            },
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def infer_page_number_from_docling_item(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    if prov:
        first = prov[0]
        for attr_name in ("page_no", "page", "page_number"):
            page_value = getattr(first, attr_name, None)
            if isinstance(page_value, int):
                return page_value
    return None


def collect_docling_figures(
    *,
    pdf_path: Path,
    client: Any,
    model: str,
    figures_dir: Path,
) -> list[FigureRecord]:
    try:
        from docling.document_converter import DocumentConverter
        from docling_core.types.doc import PictureItem
    except ImportError as exc:
        raise RuntimeError(
            "Docling is required for PDF conversion. Install docling/docling-core in the active environment."
        ) from exc

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    figures: list[FigureRecord] = []
    figure_index = 0

    for item, _level in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue

        image = item.get_image(doc)
        if image is None:
            continue

        figure_index += 1
        image_path = figures_dir / f"{figure_index:03d}.png"
        prepared_image = prepare_figure_for_vlm(image)
        prepared_image.save(image_path)

        sent_to_vlm = is_meaningful_figure(image_path)
        if sent_to_vlm:
            description = describe_figure(
                client=client,
                model=model,
                image_path=image_path,
                figure_index=figure_index,
            )
        else:
            description = "Figure too small to describe reliably."

        figures.append(
            FigureRecord(
                index=figure_index,
                image_path=image_path,
                description=description,
                sent_to_vlm=sent_to_vlm,
                source="docling_picture",
                page_number=infer_page_number_from_docling_item(item),
            )
        )

    return figures


def normalize_page_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs).strip()


def render_page_markdown(text: str) -> str:
    cleaned = normalize_page_text(text)
    return cleaned if cleaned else "_No text extracted from this page._"


def render_page_image(page: Any, output_path: Path) -> Path:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for page rendering. Install pymupdf in the active environment."
        ) from exc

    pix = page.get_pixmap(
        matrix=fitz.Matrix(PAGE_RENDER_SCALE, PAGE_RENDER_SCALE),
        alpha=False,
    )
    pix.save(str(output_path))
    return output_path


def collect_vector_figures_for_page(
    *,
    page: Any,
    page_number: int,
    client: Any,
    model: str,
    figures_dir: Path,
    start_index: int,
) -> list[FigureRecord]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for vector-figure extraction. Install pymupdf in the active environment."
        ) from exc

    drawings = page.get_drawings()
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    page_area = max(1.0, page_width * page_height)

    drawing_rects: list[tuple[float, float, float, float]] = []
    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None:
            continue
        rect_tuple = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        if rect_area(rect_tuple) <= 0:
            continue
        drawing_rects.append(rect_tuple)

    clusters = cluster_rects(
        drawing_rects,
        page_width=page_width,
        page_height=page_height,
        margin=VECTOR_CLUSTER_MARGIN_PX,
    )
    figures: list[FigureRecord] = []
    next_index = start_index

    for cluster_idx, (cluster_rect, cluster_count) in enumerate(clusters, start=1):
        cluster_area_ratio = rect_area(cluster_rect) / page_area
        if cluster_count < MIN_VECTOR_DRAWINGS_PER_CLUSTER:
            continue
        if cluster_area_ratio < MIN_VECTOR_CLUSTER_AREA_RATIO:
            continue

        next_index += 1
        clip = fitz.Rect(
            *expand_rect(
                cluster_rect,
                VECTOR_CLUSTER_MARGIN_PX,
                page_width,
                page_height,
            )
        )
        pix = page.get_pixmap(
            matrix=fitz.Matrix(VECTOR_RENDER_SCALE, VECTOR_RENDER_SCALE),
            clip=clip,
            alpha=False,
        )
        image_path = figures_dir / f"{next_index:03d}_vector_p{page_number}_{cluster_idx}.png"
        pix.save(str(image_path))

        sent_to_vlm = is_meaningful_figure(image_path)
        if sent_to_vlm:
            description = describe_figure(
                client=client,
                model=model,
                image_path=image_path,
                figure_index=next_index,
            )
        else:
            description = "Vector figure crop too small to describe reliably."

        figures.append(
            FigureRecord(
                index=next_index,
                image_path=image_path,
                description=description,
                sent_to_vlm=sent_to_vlm,
                source="pymupdf_vector",
                page_number=page_number,
            )
        )

    return figures


def collect_raster_figures_for_page(
    *,
    page: Any,
    page_number: int,
    client: Any,
    model: str,
    figures_dir: Path,
    start_index: int,
    pdf_document: Any,
    global_seen_xrefs: set[int],
) -> list[FigureRecord]:
    figures: list[FigureRecord] = []
    next_index = start_index

    for image_info in page.get_images(full=True):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for PDF conversion. Install pillow in the active environment."
            ) from exc

        xref = int(image_info[0])
        width = int(image_info[2])
        height = int(image_info[3])

        if xref in global_seen_xrefs:
            continue
        global_seen_xrefs.add(xref)

        if width < MIN_FIGURE_SIZE_PX or height < MIN_FIGURE_SIZE_PX:
            continue

        try:
            extracted = pdf_document.extract_image(xref)
            image_bytes = extracted["image"]
        except Exception:
            continue

        next_index += 1
        image_path = figures_dir / f"{next_index:03d}_raster_p{page_number}_xref{xref}.png"

        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_image:
                prepared = prepare_figure_for_vlm(pil_image)
                prepared.save(image_path)
        except Exception:
            continue

        sent_to_vlm = is_meaningful_figure(image_path)
        if sent_to_vlm:
            description = describe_figure(
                client=client,
                model=model,
                image_path=image_path,
                figure_index=next_index,
            )
        else:
            description = "Raster figure too small to describe reliably."

        figures.append(
            FigureRecord(
                index=next_index,
                image_path=image_path,
                description=description,
                sent_to_vlm=sent_to_vlm,
                source="pymupdf_raster",
                page_number=page_number,
            )
        )

    return figures


def word_count(text: str) -> int:
    return len(text.split())


def count_short_text_blocks(blocks: list[tuple[Any, ...]]) -> int:
    short_blocks = 0
    for block in blocks:
        if len(block) < 5:
            continue
        text = str(block[4] or "").strip()
        if not text:
            continue
        words = word_count(text)
        if 0 < words <= MAX_AVG_WORDS_PER_BLOCK_FOR_LAYOUT_PAGE:
            short_blocks += 1
    return short_blocks


def should_generate_page_summary(
    *,
    page_number: int,
    drawings_count: int,
    text_blocks: list[tuple[Any, ...]],
    page_figures: list[FigureRecord],
    page_text: str,
) -> bool:
    short_block_count = count_short_text_blocks(text_blocks)
    sparse_text = word_count(page_text) < 180
    has_visual_regions = any(figure.sent_to_vlm for figure in page_figures)

    if has_visual_regions:
        return True
    if drawings_count >= MIN_PAGE_DRAWINGS_FOR_VLM and short_block_count >= MIN_SHORT_BLOCKS_FOR_VLM:
        return True
    if page_number == 1 and drawings_count >= MIN_PAGE_DRAWINGS_FOR_VLM:
        return True
    if sparse_text and short_block_count >= MIN_SHORT_BLOCKS_FOR_VLM:
        return True
    return False


def render_figure_block(
    figure: FigureRecord,
    output_dir: Path,
    *,
    include_debug_details: bool,
) -> str:
    page_text = figure.page_number if figure.page_number is not None else "unknown"
    parts = [
        f"#### Figure {figure.index}\n\n",
        f"- Page: {page_text}\n",
        f"- Description: {figure.description or 'No description generated.'}\n",
    ]

    if include_debug_details:
        rel_path = os.path.relpath(figure.image_path, output_dir)
        status = "yes" if figure.sent_to_vlm else "no"
        parts.insert(1, f"- Source: {figure.source}\n")
        parts.insert(3, f"- Sent to VLM: {status}\n")
        parts.insert(4, f"- Image: `{rel_path}`\n")
        parts.append(f"\n![Figure {figure.index}]({rel_path})\n")

    return "".join(parts).rstrip()


def render_page_section(
    page_record: PageRecord,
    output_dir: Path,
    *,
    include_debug_details: bool,
) -> str:
    parts = [f"## Page {page_record.page_number}\n"]
    parts.append("\n### Extracted Text\n\n")
    parts.append(page_record.text_markdown)
    parts.append("\n\n### Visual Summary\n\n")

    if page_record.visual_summary_generated:
        parts.append(page_record.visual_summary or "No visual summary generated.")
    else:
        parts.append("No important non-text visual content detected.")

    if page_record.figures:
        parts.append("\n\n### Detected Visual Regions\n\n")
        for figure in page_record.figures:
            parts.append(render_figure_block(figure, output_dir, include_debug_details=include_debug_details))
            parts.append("\n\n")

    return "".join(parts).rstrip()


def render_document_markdown(
    *,
    title: str,
    pages: list[PageRecord],
    output_dir: Path,
    include_debug_details: bool,
) -> str:
    parts = [f"# {title}\n"]
    for page in pages:
        parts.append("\n\n---\n\n")
        parts.append(
            render_page_section(
                page,
                output_dir,
                include_debug_details=include_debug_details,
            )
        )
    parts.append("\n")
    return "".join(parts)


def write_debug_outputs(
    *,
    pdf_path: Path,
    output_dir: Path,
    pages: list[PageRecord],
) -> None:
    figures_manifest_path = figures_manifest_output_path(pdf_path, output_dir)
    figures_gallery_path = figures_gallery_output_path(pdf_path, output_dir)
    pages_manifest_path = pages_manifest_output_path(pdf_path, output_dir)

    all_figures = [figure for page in pages for figure in page.figures]
    figure_manifest = []
    figure_gallery_parts = ["# Extracted Figure Gallery\n"]

    for figure in all_figures:
        rel_path = os.path.relpath(figure.image_path, output_dir)
        figure_manifest.append(
            {
                "figure_index": figure.index,
                "image_path": rel_path,
                "sent_to_vlm": figure.sent_to_vlm,
                "source": figure.source,
                "page_number": figure.page_number,
                "description": figure.description,
            }
        )
        figure_gallery_parts.append(
            render_figure_block(figure, output_dir, include_debug_details=True)
        )
        figure_gallery_parts.append("\n\n")

    page_manifest = []
    for page in pages:
        page_manifest.append(
            {
                "page_number": page.page_number,
                "visual_summary_generated": page.visual_summary_generated,
                "visual_summary": page.visual_summary,
                "drawings_count": page.drawings_count,
                "text_block_count": page.text_block_count,
                "short_block_count": page.short_block_count,
                "figure_indices": [figure.index for figure in page.figures],
            }
        )

    figures_manifest_path.write_text(
        json.dumps(figure_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    figures_gallery_path.write_text("".join(figure_gallery_parts), encoding="utf-8")
    pages_manifest_path.write_text(
        json.dumps(page_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def process_pdf(config: ConversionConfig) -> Path:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF conversion. Install pymupdf in the active environment."
        ) from exc

    ensure_output_dir(config.output_dir)
    client = create_client(config)

    with working_dirs(config) as dirs:
        docling_figures = collect_docling_figures(
            pdf_path=config.pdf_path,
            client=client,
            model=config.model,
            figures_dir=dirs.figures_dir,
        )

        figures_by_page: dict[int, list[FigureRecord]] = {}
        for figure in docling_figures:
            if figure.page_number is not None:
                figures_by_page.setdefault(figure.page_number, []).append(figure)

        pages: list[PageRecord] = []
        next_figure_index = len(docling_figures)

        with fitz.open(config.pdf_path) as pdf:
            seen_raster_xrefs: set[int] = set()
            for page_idx, page in enumerate(pdf):
                page_number = page_idx + 1
                page_text = page.get_text("text", sort=True)
                text_markdown = render_page_markdown(page_text)
                blocks = page.get_text("blocks", sort=True)
                drawings_count = len(page.get_drawings())
                page_figures = list(figures_by_page.get(page_number, []))

                vector_figures = collect_vector_figures_for_page(
                    page=page,
                    page_number=page_number,
                    client=client,
                    model=config.model,
                    figures_dir=dirs.figures_dir,
                    start_index=next_figure_index,
                )
                if vector_figures:
                    next_figure_index = max(figure.index for figure in vector_figures)
                    page_figures.extend(vector_figures)

                raster_figures = collect_raster_figures_for_page(
                    page=page,
                    page_number=page_number,
                    client=client,
                    model=config.model,
                    figures_dir=dirs.figures_dir,
                    start_index=next_figure_index,
                    pdf_document=pdf,
                    global_seen_xrefs=seen_raster_xrefs,
                )
                if raster_figures:
                    next_figure_index = max(figure.index for figure in raster_figures)
                    page_figures.extend(raster_figures)

                short_block_count = count_short_text_blocks(blocks)
                visual_summary_generated = should_generate_page_summary(
                    page_number=page_number,
                    drawings_count=drawings_count,
                    text_blocks=blocks,
                    page_figures=page_figures,
                    page_text=page_text,
                )

                visual_summary: str | None = None
                if visual_summary_generated:
                    page_image_path = dirs.pages_dir / f"page_{page_number:03d}.png"
                    render_page_image(page, page_image_path)
                    visual_summary = describe_page_visuals(
                        client=client,
                        model=config.model,
                        image_path=page_image_path,
                        page_number=page_number,
                    )

                pages.append(
                    PageRecord(
                        page_number=page_number,
                        text_markdown=text_markdown,
                        visual_summary=visual_summary,
                        visual_summary_generated=visual_summary_generated,
                        figures=page_figures,
                        drawings_count=drawings_count,
                        text_block_count=len(blocks),
                        short_block_count=short_block_count,
                    )
                )

        final_markdown = render_document_markdown(
            title=config.pdf_path.stem.replace("_", " ").replace("-", " "),
            pages=pages,
            output_dir=dirs.debug_dir or config.output_dir,
            include_debug_details=config.debug,
        )

        if config.debug and dirs.debug_dir is not None:
            write_debug_outputs(
                pdf_path=config.pdf_path,
                output_dir=dirs.debug_dir,
                pages=pages,
            )

    output_path = markdown_output_path(config.pdf_path, config.output_dir)
    output_path.write_text(final_markdown, encoding="utf-8")
    return output_path


def main() -> None:
    config = parse_args()
    output_path = process_pdf(config)
    print(output_path)


if __name__ == "__main__":
    main()
