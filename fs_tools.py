"""Core file-system tools for the LLM File Assistant.

Every tool returns plain Python data (dicts / lists) that is JSON-serialisable,
so results can be handed straight back to an LLM. Tools never raise for
expected problems (missing file, bad format, permissions); they return
``{"success": False, "error": "..."}`` instead.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SUPPORTED_READ_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _error(message: str, **extra: Any) -> dict:
    return {"success": False, "error": message, **extra}


def _extract_txt(path: Path) -> tuple[str, dict]:
    try:
        return path.read_text(encoding="utf-8"), {}
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1"), {"encoding_fallback": "latin-1"}


def _extract_pdf(path: Path) -> tuple[str, dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages).strip()
    extra: dict = {"pages": len(reader.pages)}
    if not text:
        extra["warning"] = "No extractable text (the PDF may be scanned images)."
    return text, extra


def _extract_docx(path: Path) -> tuple[str, dict]:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]
    # Resumes often put skills / experience in tables.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts).strip(), {"paragraphs": len(document.paragraphs)}


_EXTRACTORS = {
    ".txt": _extract_txt,
    ".md": _extract_txt,
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
}


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

def read_file(filepath: str) -> dict:
    """Read a resume file (PDF, TXT, DOCX, MD) and extract its text.

    Returns:
        {"success": True, "content": str, "metadata": {...}} on success, or
        {"success": False, "error": str, "filepath": str} on failure.
    """
    path = Path(filepath).expanduser()
    if not path.exists():
        return _error(f"File not found: {filepath}", filepath=filepath)
    if not path.is_file():
        return _error(f"Not a file: {filepath}", filepath=filepath)

    ext = path.suffix.lower()
    if ext not in SUPPORTED_READ_EXTENSIONS:
        return _error(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_READ_EXTENSIONS)}",
            filepath=filepath,
        )

    try:
        content, extra = _EXTRACTORS[ext](path)
    except PermissionError:
        return _error(f"Permission denied: {filepath}", filepath=filepath)
    except Exception as exc:  # corrupt / malformed documents
        logger.exception("Failed to read %s", filepath)
        return _error(f"Could not read {ext} file: {exc}", filepath=filepath)

    metadata = {
        "name": path.name,
        "path": str(path.resolve()),
        "extension": ext,
        "size_bytes": path.stat().st_size,
        "modified": _iso_mtime(path),
        "char_count": len(content),
        "word_count": len(content.split()),
        **extra,
    }
    return {"success": True, "content": content, "metadata": metadata}


def list_files(directory: str, extension: Optional[str] = None) -> list:
    """List files (not sub-directories) in ``directory``.

    Args:
        directory: folder to list.
        extension: optional filter such as ".pdf" or "pdf" (case-insensitive).

    Returns:
        A list of dicts: name, path, extension, size_bytes, modified.
        Returns an empty list if the directory does not exist or is unreadable.
    """
    folder = Path(directory).expanduser()
    if not folder.is_dir():
        logger.warning("list_files: not a directory: %s", directory)
        return []

    wanted = None
    if extension:
        wanted = extension.lower().strip()
        if not wanted.startswith("."):
            wanted = "." + wanted

    results = []
    try:
        entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        logger.warning("list_files: permission denied: %s", directory)
        return []

    for entry in entries:
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if wanted and ext != wanted:
            continue
        stat = entry.stat()
        results.append(
            {
                "name": entry.name,
                "path": str(entry.resolve()),
                "extension": ext,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return results


def write_file(filepath: str, content: str) -> dict:
    """Write ``content`` (UTF-8) to ``filepath``, creating parent directories.

    Returns:
        {"success": True, "path": str, "bytes_written": int} or
        {"success": False, "error": str, "filepath": str}.
    """
    path = Path(filepath).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        path.write_bytes(data)
    except Exception as exc:
        return _error(f"Could not write file: {exc}", filepath=filepath)
    return {"success": True, "path": str(path.resolve()), "bytes_written": len(data)}


def search_in_file(filepath: str, keyword: str, context_chars: int = 80) -> dict:
    """Case-insensitive keyword search inside any readable file.

    Each match includes the line number, the exact matched text and
    ``context_chars`` characters of surrounding text on either side.
    """
    if not keyword or not keyword.strip():
        return _error("Keyword must be a non-empty string.", filepath=filepath)

    read = read_file(filepath)
    if not read["success"]:
        return read

    text = read["content"]
    pattern = re.compile(re.escape(keyword.strip()), re.IGNORECASE)
    matches = []
    for m in pattern.finditer(text):
        start, end = m.start(), m.end()
        ctx_start = max(0, start - context_chars)
        ctx_end = min(len(text), end + context_chars)
        context = text[ctx_start:ctx_end].replace("\n", " ").strip()
        if ctx_start > 0:
            context = "..." + context
        if ctx_end < len(text):
            context = context + "..."
        matches.append(
            {
                "position": start,
                "line": text.count("\n", 0, start) + 1,
                "match": m.group(0),
                "context": context,
            }
        )

    return {
        "success": True,
        "filepath": read["metadata"]["path"],
        "file_name": read["metadata"]["name"],
        "keyword": keyword,
        "match_count": len(matches),
        "matches": matches,
    }


__all__ = ["read_file", "list_files", "write_file", "search_in_file"]
