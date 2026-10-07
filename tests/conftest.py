import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_pdf(path: Path, lines: list[str]) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 18
    c.save()
    return path


def make_docx(path: Path, paragraphs: list[str]) -> Path:
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(str(path))
    return path


@pytest.fixture
def resume_dir(tmp_path: Path) -> Path:
    d = tmp_path / "resumes"
    d.mkdir()
    (d / "jane_smith.txt").write_text(
        "Jane Smith\nSenior Data Engineer\nSkills: PYTHON, SQL, Airflow\n"
        "Built python pipelines processing 2TB/day.\n",
        encoding="utf-8",
    )
    make_docx(d / "raj_patel.docx", ["Raj Patel", "Frontend Developer", "Skills: React, TypeScript"])
    make_pdf(d / "john_doe.pdf", ["John Doe", "Machine Learning Engineer", "Skills: Python, PyTorch"])
    (d / "subfolder").mkdir()
    return d
