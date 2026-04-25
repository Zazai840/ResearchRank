#!/usr/bin/env python3
"""ingest.py — PDF ingestion for ResearchRank."""
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
import typer
from refextract import extract_references_from_file

# refextract logs INFO/WARNING verbosely; silence it
for _log in ("refextract", "refextract.references", "refextract.core"):
    logging.getLogger(_log).setLevel(logging.ERROR)


def _patch_refextract_for_macos() -> None:
    # macOS lacks mremap(), so mmap.resize() raises SystemError inside
    # refextract's clean_pdf_file.  Replace it with a pure-Python equivalent
    # that uses regular file I/O.  The patch is idempotent and only needed on
    # platforms where mmap resize is unavailable.
    import mmap as _mmap
    import refextract.references.engine as _eng

    if getattr(_eng, "_macos_patched", False):
        return
    try:
        m = _mmap.mmap(-1, 4096)
        m.resize(8192)
        m.close()
    except SystemError:
        def _clean_pdf_portable(filename: str) -> None:
            with open(filename, "rb") as f:
                data = f.read()
            start = data.find(b"%PDF-")
            if start == -1:
                return
            end = data.rfind(b"%%EOF")
            if end == -1:
                return
            end_off = end + len(b"%%EOF")
            if start == 0 and end_off == len(data):
                return
            with open(filename, "wb") as f:
                f.write(data[start:end_off])

        _eng.clean_pdf_file = _clean_pdf_portable
    _eng._macos_patched = True


_patch_refextract_for_macos()

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

app = typer.Typer(add_completion=False)

# Ordered list of headings used to terminate a section's text
_SENTINEL_RE = re.compile(
    r"^(abstract|introduction|background|related\s+work|method(?:s|ology)?|"
    r"experiment(?:s|al)?|results?|discussion|conclusions?|acknowledgem?ents?|"
    r"references|bibliography)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def extract_section(text: str, name: str) -> str:
    """Return text between the named heading and the next known heading."""
    m = re.search(
        rf"^{re.escape(name)}s?\s*$", text, re.IGNORECASE | re.MULTILINE
    )
    if not m:
        return ""
    start = m.end()
    nxt = _SENTINEL_RE.search(text, start)
    return text[start : nxt.start() if nxt else len(text)].strip()


def _first(lst) -> str:
    """Return first element of a list or the value itself, as a string."""
    if isinstance(lst, list):
        return lst[0] if lst else ""
    return lst or ""


def normalise_ref(ref: dict) -> dict:
    # refextract uses raw_ref (list); output schema wants raw (str)
    return {
        "raw": _first(ref.get("raw_ref")),
        "title": _first(ref.get("title")),
        "author": _first(ref.get("author")),
        "year": _first(ref.get("year")),
    }


def extract_metadata(
    doc: fitz.Document, first_page: str
) -> tuple:  # (str, list[str], Optional[int])
    meta = doc.metadata or {}

    title = (meta.get("title") or "").strip()
    author_raw = (meta.get("author") or "").strip()
    authors = (
        [a.strip() for a in re.split(r",|;|\band\b", author_raw) if a.strip()]
        if author_raw
        else []
    )

    year: Optional[int] = None
    for date_key in ("creationDate", "modDate"):
        m = _YEAR_RE.search(meta.get(date_key) or "")
        if m:
            year = int(m.group())
            break

    lines = [l.strip() for l in first_page.splitlines() if l.strip()]

    if not title and lines:
        title = lines[0]

    if not authors and len(lines) > 1:
        parts = [p.strip() for p in re.split(r",|\band\b", lines[1]) if p.strip()]
        if parts and all(len(p) < 60 for p in parts):
            authors = parts

    if not year:
        m = _YEAR_RE.search(first_page)
        if m:
            year = int(m.group())

    return title, authors, year


def process_pdf(path: Path) -> dict:
    doc = fitz.open(str(path))
    full_text = "\n".join(page.get_text() for page in doc)
    first_page = doc[0].get_text() if doc.page_count > 0 else ""
    title, authors, year = extract_metadata(doc, first_page)
    doc.close()

    try:
        raw_refs = extract_references_from_file(str(path)) or []
        references = [normalise_ref(r) for r in raw_refs]
    except Exception as exc:
        logging.warning("%s: reference extraction failed — %s", path.name, exc)
        references = []

    return {
        "id": path.stem,
        "path": str(path.resolve()),
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": extract_section(full_text, "abstract"),
        "introduction": extract_section(full_text, "introduction"),
        "conclusion": extract_section(full_text, "conclusion"),
        "references": references,
    }


@app.command()
def ingest(
    folder: Path = typer.Argument(..., help="Folder containing PDF files"),
    output: Path = typer.Option(None, "--output", "-o", help="Output JSON path"),
):
    """Ingest PDFs and write papers.json."""
    if not folder.is_dir():
        typer.echo(f"Error: {folder} is not a directory", err=True)
        raise typer.Exit(1)

    out_path = output or folder / "papers.json"
    cache_path = folder / "papers.cache.json"

    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            pass

    existing: dict = {}
    if out_path.exists():
        try:
            existing = {p["id"]: p for p in json.loads(out_path.read_text()).get("papers", [])}
        except Exception:
            pass

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        typer.echo("No PDF files found.")
        raise typer.Exit(0)

    papers = []
    for pdf in pdfs:
        pid = pdf.stem
        mtime = pdf.stat().st_mtime
        if pid in cache and cache[pid] == mtime and pid in existing and existing[pid].get("references"):
            typer.echo(f"[skip]  {pdf.name}")
            papers.append(existing[pid])
            continue

        typer.echo(f"[parse] {pdf.name}")
        try:
            paper = process_pdf(pdf)
            cache[pid] = mtime
            papers.append(paper)
        except Exception as exc:
            logging.warning("%s: skipped — %s", pdf.name, exc)

    out_path.write_text(json.dumps({"papers": papers}, indent=2, ensure_ascii=False))
    cache_path.write_text(json.dumps(cache, indent=2))
    typer.echo(f"\nWrote {len(papers)} paper(s) → {out_path}")


if __name__ == "__main__":
    app()
