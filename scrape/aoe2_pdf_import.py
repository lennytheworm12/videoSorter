"""
Import an Age of Empires II written PDF guide into knowledge.db.

Usage:
    uv run python -m scrape.aoe2_pdf_import "/path/to/hera-guide.pdf" --title "Hera Strategy Guide 2025"
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
from datetime import datetime, timezone

from core.db_paths import activate_knowledge_db

activate_knowledge_db()

from core.database import get_connection, init_db, insert_video, set_transcription

PDF_SOURCE = "aoe2_pdf"


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_pdf_text(path: pathlib.Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pypdf or run `uv sync` first.") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = _clean_pdf_text(page_text)
        if page_text:
            pages.append(f"## Page {index}\n{page_text}")
    text = "\n\n".join(pages)
    if not text.strip():
        raise ValueError(f"No extractable text found in PDF: {path}")
    return text


def _stable_pdf_id(path: pathlib.Path, title: str | None = None) -> str:
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode("utf-8", errors="ignore"))
    if title:
        h.update(title.encode("utf-8", errors="ignore"))
    return f"{PDF_SOURCE}_{h.hexdigest()[:12]}"


def import_pdf(path: pathlib.Path, title: str | None = None) -> str:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    title = title or path.stem
    video_id = _stable_pdf_id(path, title)
    text = extract_pdf_text(path)

    insert_video(
        video_id=video_id,
        video_url=str(path),
        video_title=title,
        description=f"Imported PDF guide: {path}",
        game="aoe2",
        role="general",
        subject=None,
        champion=None,
        rank=None,
        website_rating=None,
        message_timestamp=datetime.now(timezone.utc).isoformat(),
        source=PDF_SOURCE,
    )
    set_transcription(video_id, text)
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE videos
            SET video_url = ?,
                video_title = ?,
                description = ?,
                game = 'aoe2',
                role = 'general',
                subject = NULL,
                champion = NULL,
                source = ?
            WHERE video_id = ?
            """,
            (str(path), title, f"Imported PDF guide: {path}", PDF_SOURCE, video_id),
        )
        conn.commit()
    return video_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Import an AoE2 PDF guide into knowledge.db")
    parser.add_argument("pdf_path", help="Path to a text-based PDF guide")
    parser.add_argument("--title", help="Title to store for the PDF source")
    args = parser.parse_args()

    init_db()
    video_id = import_pdf(pathlib.Path(args.pdf_path), title=args.title)
    print(f"Imported PDF guide as {video_id}")


if __name__ == "__main__":
    main()
