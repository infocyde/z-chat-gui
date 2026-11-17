"""Turn uploaded files into attachments: text extraction for documents, storage for images."""

from __future__ import annotations

import base64
import io
import mimetypes
import secrets
from dataclasses import dataclass
from pathlib import Path

from . import config, zai

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".xml", ".html",
             ".htm", ".py", ".js", ".ts", ".css", ".sql", ".log", ".ini", ".toml", ".rst", ".srt"}
# Below this many extracted characters per page we assume a scanned PDF and fall back to OCR.
MIN_CHARS_PER_PAGE = 40


class UnsupportedFile(Exception):
    pass


@dataclass
class Processed:
    kind: str  # document | image
    filename: str
    path: str  # relative to project root
    mime: str
    size: int
    text: str = ""
    note: str = ""


def safe_name(filename: str) -> str:
    name = Path(filename or "file").name
    return "".join(c for c in name if c.isalnum() or c in "._- ")[:150] or "file"


def store_upload(filename: str, data: bytes) -> Path:
    """Save under a generated name (never trust the client filename for paths)."""
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.UPLOAD_DIR / f"{secrets.token_hex(8)}{Path(filename).suffix.lower()[:10]}"
    dest.write_bytes(data)
    return dest


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def abs_path(stored: str) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else config.ROOT / p


def data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _truncate(text: str) -> tuple[str, str]:
    if len(text) > config.MAX_DOC_CHARS:
        return text[: config.MAX_DOC_CHARS], f"truncated to {config.MAX_DOC_CHARS:,} chars"
    return text, ""


def extract_pdf(data: bytes) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    text = "\n\n".join(f"--- page {i} ---\n{t}" for i, t in enumerate(pages, 1) if t)
    return text, len(reader.pages)


def extract_docx(data: bytes) -> str:
    import docx

    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


async def process_upload(filename: str, data: bytes) -> Processed:
    name = safe_name(filename)
    ext = Path(name).suffix.lower()
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    if ext in IMAGE_EXTS:
        dest = store_upload(name, data)
        return Processed("image", name, rel(dest), mime, len(data))

    note = ""
    if ext == ".pdf":
        text, pages = extract_pdf(data)
        if len(text) < MIN_CHARS_PER_PAGE * max(pages, 1):
            try:
                text = await zai.ocr(f"data:application/pdf;base64,{base64.b64encode(data).decode()}")
                note = "OCR"
            except zai.ZaiError as e:
                note = f"little text found; OCR unavailable: {e}"[:290]
    elif ext == ".docx":
        text = extract_docx(data)
    elif ext in TEXT_EXTS or mime.startswith("text/"):
        text = decode_text(data)
    else:
        raise UnsupportedFile(f"{name}: unsupported file type. Try PDF, DOCX, text/markdown, or an image.")

    text, trunc = _truncate(text.strip())
    note = ", ".join(x for x in (note, trunc) if x)
    dest = store_upload(name, data)
    return Processed("document", name, rel(dest), mime, len(data), text=text, note=note)
