from pathlib import Path

import fs_tools


# ---------- read_file ----------

def test_read_txt(resume_dir):
    r = fs_tools.read_file(str(resume_dir / "jane_smith.txt"))
    assert r["success"] is True
    assert "Jane Smith" in r["content"]
    meta = r["metadata"]
    assert meta["name"] == "jane_smith.txt"
    assert meta["extension"] == ".txt"
    assert meta["size_bytes"] > 0
    assert meta["word_count"] > 0
    assert "modified" in meta


def test_read_docx(resume_dir):
    r = fs_tools.read_file(str(resume_dir / "raj_patel.docx"))
    assert r["success"] is True
    assert "Frontend Developer" in r["content"]


def test_read_pdf(resume_dir):
    r = fs_tools.read_file(str(resume_dir / "john_doe.pdf"))
    assert r["success"] is True
    assert "Machine Learning Engineer" in r["content"]
    assert r["metadata"]["pages"] == 1


def test_read_missing_file(tmp_path):
    r = fs_tools.read_file(str(tmp_path / "nope.txt"))
    assert r["success"] is False
    assert "not found" in r["error"].lower()


def test_read_unsupported_extension(tmp_path):
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG")
    r = fs_tools.read_file(str(p))
    assert r["success"] is False
    assert "unsupported" in r["error"].lower()


def test_read_directory_is_error(tmp_path):
    r = fs_tools.read_file(str(tmp_path))
    assert r["success"] is False


def test_read_corrupt_pdf(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_text("not really a pdf")
    r = fs_tools.read_file(str(p))
    assert r["success"] is False
    assert r["error"]


# ---------- list_files ----------

def test_list_all_files_excludes_dirs(resume_dir):
    files = fs_tools.list_files(str(resume_dir))
    assert isinstance(files, list)
    names = [f["name"] for f in files]
    assert names == sorted(["jane_smith.txt", "raj_patel.docx", "john_doe.pdf"])
    assert "subfolder" not in names


def test_list_metadata_fields(resume_dir):
    f = fs_tools.list_files(str(resume_dir))[0]
    for key in ("name", "path", "size_bytes", "modified", "extension"):
        assert key in f


def test_list_filter_extension_variants(resume_dir):
    for ext in (".pdf", "pdf", ".PDF"):
        files = fs_tools.list_files(str(resume_dir), ext)
        assert [f["name"] for f in files] == ["john_doe.pdf"]


def test_list_missing_directory(tmp_path):
    assert fs_tools.list_files(str(tmp_path / "missing")) == []


# ---------- write_file ----------

def test_write_creates_nested_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "summary.txt"
    r = fs_tools.write_file(str(target), "hello ✓")
    assert r["success"] is True
    assert target.read_text(encoding="utf-8") == "hello ✓"
    assert r["bytes_written"] == len("hello ✓".encode("utf-8"))


def test_write_invalid_path(tmp_path):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    r = fs_tools.write_file(str(blocker / "child.txt"), "data")
    assert r["success"] is False
    assert r["error"]


# ---------- search_in_file ----------

def test_search_case_insensitive_with_context(resume_dir):
    r = fs_tools.search_in_file(str(resume_dir / "jane_smith.txt"), "python")
    assert r["success"] is True
    assert r["match_count"] == 2
    first = r["matches"][0]
    assert first["match"] == "PYTHON"
    assert "SQL" in first["context"]
    assert first["line"] == 3


def test_search_in_pdf(resume_dir):
    r = fs_tools.search_in_file(str(resume_dir / "john_doe.pdf"), "PyTorch")
    assert r["success"] is True
    assert r["match_count"] == 1


def test_search_no_matches(resume_dir):
    r = fs_tools.search_in_file(str(resume_dir / "raj_patel.docx"), "python")
    assert r["success"] is True
    assert r["match_count"] == 0
    assert r["matches"] == []


def test_search_empty_keyword(resume_dir):
    r = fs_tools.search_in_file(str(resume_dir / "jane_smith.txt"), "  ")
    assert r["success"] is False


def test_search_missing_file(tmp_path):
    r = fs_tools.search_in_file(str(tmp_path / "x.txt"), "python")
    assert r["success"] is False
