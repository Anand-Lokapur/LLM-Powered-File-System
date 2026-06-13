from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _get_file_metadata(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def read_file(filepath: str) -> Dict[str, Any]:
    """
    Read resume files (PDF, TXT, DOCX), extract text content,
    and return structured response with content and metadata.
    """
    path = Path(filepath)

    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {filepath}",
            "filepath": str(path),
        }

    if not path.is_file():
        return {
            "success": False,
            "error": f"Path is not a file: {filepath}",
            "filepath": str(path),
        }

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return {
            "success": False,
            "error": f"Unsupported file type '{extension}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            "filepath": str(path),
        }

    try:
        if extension == ".txt":
            content = _read_txt(path)
        elif extension == ".pdf":
            content = _read_pdf(path)
        else:
            content = _read_docx(path)

        return {
            "success": True,
            "filepath": str(path.resolve()),
            "metadata": _get_file_metadata(path),
            "content": content,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to read file: {exc}",
            "filepath": str(path),
        }


def list_files(directory: str, extension: str | None = None) -> List[Dict[str, Any]]:
    """
    List all files in a directory and optionally filter by extension.
    Returns file metadata (name, size, modified date, etc).
    """
    dir_path = Path(directory)

    if not dir_path.exists() or not dir_path.is_dir():
        return []

    normalized_ext = None
    if extension:
        normalized_ext = extension.lower()
        if not normalized_ext.startswith("."):
            normalized_ext = f".{normalized_ext}"

    results: List[Dict[str, Any]] = []
    for path in sorted(dir_path.rglob("*")):
        if not path.is_file():
            continue

        if normalized_ext and path.suffix.lower() != normalized_ext:
            continue

        results.append(_get_file_metadata(path))

    return results


def write_file(filepath: str, content: str) -> Dict[str, Any]:
    """
    Write content to file, create directories if needed,
    and return structured success/failure status.
    """
    path = Path(filepath)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "filepath": str(path.resolve()),
            "bytes_written": len(content.encode("utf-8")),
            "metadata": _get_file_metadata(path),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to write file: {exc}",
            "filepath": str(path),
        }


def search_in_file(filepath: str, keyword: str) -> Dict[str, Any]:
    """
    Search for keyword in file content (case-insensitive).
    Return matches with surrounding context.
    """
    if not keyword:
        return {
            "success": False,
            "error": "Keyword cannot be empty.",
            "filepath": filepath,
            "keyword": keyword,
        }

    read_result = read_file(filepath)
    if not read_result.get("success"):
        return {
            "success": False,
            "error": read_result.get("error", "Unable to read file."),
            "filepath": filepath,
            "keyword": keyword,
            "matches": [],
            "matches_count": 0,
        }

    content = read_result.get("content", "")
    matches = []
    for match in re.finditer(re.escape(keyword), content, flags=re.IGNORECASE):
        start, end = match.start(), match.end()
        context_start = max(0, start - 60)
        context_end = min(len(content), end + 60)
        line_number = content.count("\n", 0, start) + 1

        matches.append(
            {
                "match": content[start:end],
                "start_index": start,
                "end_index": end,
                "line_number": line_number,
                "context": content[context_start:context_end].strip(),
            }
        )

    return {
        "success": True,
        "filepath": str(Path(filepath).resolve()),
        "keyword": keyword,
        "matches_count": len(matches),
        "matches": matches,
    }
