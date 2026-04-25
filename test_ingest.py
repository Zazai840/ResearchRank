"""Tests for ingest.py — reference extraction and cache invalidation."""
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

import ingest

SAMPLE_DIR = Path(__file__).parent / "sample_pdf"

NATURE_PDF = SAMPLE_DIR / "nature14539.pdf"
SCIENCE_PDF = SAMPLE_DIR / "science.aaa8415.pdf"
MTS_PDF = SAMPLE_DIR / "79b5af1ab7fc4c16a780abebacb5d0e43d12.pdf"


# ---------------------------------------------------------------------------
# normalise_ref
# ---------------------------------------------------------------------------

class TestNormaliseRef:
    def test_picks_first_element_from_lists(self):
        raw = {
            "raw_ref": ["full citation text"],
            "title": ["Some Title"],
            "author": ["Smith, J."],
            "year": ["2020"],
        }
        result = ingest.normalise_ref(raw)
        assert result == {
            "raw": "full citation text",
            "title": "Some Title",
            "author": "Smith, J.",
            "year": "2020",
        }

    def test_missing_fields_become_empty_string(self):
        result = ingest.normalise_ref({})
        assert result == {"raw": "", "title": "", "author": "", "year": ""}

    def test_empty_lists_become_empty_string(self):
        result = ingest.normalise_ref({"raw_ref": [], "title": [], "author": [], "year": []})
        assert result == {"raw": "", "title": "", "author": "", "year": ""}

    def test_scalar_values_pass_through(self):
        raw = {"raw_ref": "plain string", "title": "T", "author": "A", "year": "2021"}
        result = ingest.normalise_ref(raw)
        assert result["raw"] == "plain string"


# ---------------------------------------------------------------------------
# process_pdf — reference extraction against real PDFs
# ---------------------------------------------------------------------------

class TestProcessPdfReferences:
    def test_nature_refs_non_empty(self):
        paper = ingest.process_pdf(NATURE_PDF)
        assert len(paper["references"]) > 0, "nature14539.pdf should yield references"

    def test_nature_first_ref_contains_krizhevsky(self):
        paper = ingest.process_pdf(NATURE_PDF)
        first = paper["references"][0]
        assert "Krizhevsky" in first["raw"], "first ref should cite Krizhevsky et al."
        assert first["year"] == "2012"
        assert "Krizhevsky" in first["author"]

    def test_science_refs_non_empty(self):
        paper = ingest.process_pdf(SCIENCE_PDF)
        assert len(paper["references"]) > 0, "science.aaa8415.pdf should yield references"

    def test_science_first_ref_contains_hastie(self):
        paper = ingest.process_pdf(SCIENCE_PDF)
        first = paper["references"][0]
        assert "Hastie" in first["raw"], "first ref should cite Hastie et al."

    def test_mts_refs_non_empty(self):
        paper = ingest.process_pdf(MTS_PDF)
        assert len(paper["references"]) > 0, "MTS paper should yield references"

    def test_mts_first_ref_contains_expected_token(self):
        paper = ingest.process_pdf(MTS_PDF)
        first = paper["references"][0]
        assert first["raw"], "first ref raw text should not be empty"
        assert first["year"], "first ref should have a year"

    def test_ref_schema_keys(self):
        paper = ingest.process_pdf(NATURE_PDF)
        for ref in paper["references"]:
            assert set(ref.keys()) == {"raw", "title", "author", "year"}

    def test_refextract_exception_returns_empty_list(self):
        with patch("ingest.extract_references_from_file", side_effect=RuntimeError("boom")):
            paper = ingest.process_pdf(NATURE_PDF)
        assert paper["references"] == []


# ---------------------------------------------------------------------------
# Cache invalidation — papers with empty references must be re-processed
# ---------------------------------------------------------------------------

class TestCacheInvalidation:
    def _make_folder(self, tmp_path: Path, mtime: float, references: list) -> Path:
        """Copy sample PDFs into a temp folder with a controlled mtime and given references."""
        folder = tmp_path / "pdfs"
        folder.mkdir()
        dest = shutil.copy(NATURE_PDF, folder / NATURE_PDF.name)
        pid = NATURE_PDF.stem
        cache = {pid: mtime}
        existing = {
            "papers": [{
                "id": pid,
                "path": str(dest),
                "title": "Cached title",
                "authors": [],
                "year": 2015,
                "abstract": "",
                "introduction": "",
                "conclusion": "",
                "references": references,
            }]
        }
        (folder / "papers.cache.json").write_text(json.dumps(cache))
        (folder / "papers.json").write_text(json.dumps(existing))
        os.utime(dest, (mtime, mtime))
        return folder

    def test_stale_empty_references_triggers_reprocess(self, tmp_path):
        """A cached paper with references=[] must be re-processed even if mtime matches."""
        mtime = NATURE_PDF.stat().st_mtime
        folder = self._make_folder(tmp_path, mtime, references=[])

        reprocessed = []
        original_process_pdf = ingest.process_pdf

        def spy_process_pdf(path):
            reprocessed.append(path)
            return original_process_pdf(path)

        with patch("ingest.process_pdf", side_effect=spy_process_pdf):
            CliRunner().invoke(ingest.app, [str(folder)])

        assert reprocessed, "process_pdf should have been called for the stale entry"
        output = json.loads((folder / "papers.json").read_text())
        refs = output["papers"][0]["references"]
        assert len(refs) > 0, "after re-processing, references should be populated"

    def test_valid_cached_paper_is_skipped(self, tmp_path):
        """A cached paper with non-empty references must NOT be re-processed."""
        mtime = NATURE_PDF.stat().st_mtime
        existing_refs = [{"raw": "some ref", "title": "", "author": "", "year": "2012"}]
        folder = self._make_folder(tmp_path, mtime, references=existing_refs)

        reprocessed = []

        def spy_process_pdf(path):
            reprocessed.append(path)
            return ingest.process_pdf(path)

        with patch("ingest.process_pdf", side_effect=spy_process_pdf):
            CliRunner().invoke(ingest.app, [str(folder)])

        assert reprocessed == [], "process_pdf should NOT be called for a valid cached entry"
