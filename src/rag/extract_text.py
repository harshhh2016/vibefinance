"""Extract text page by page from every document in data/documents/manifest.json.

Writes data/documents/pages/<doc_id>.jsonl (one JSON object per page) and
prints a summary. Run from the project root:  python -m src.rag.extract_text
"""
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "data" / "documents"
RAW = DOCS / "raw"
PAGES = DOCS / "pages"
MANIFEST_JSON = DOCS / "manifest.json"

MIN_CHARS = 30  # a page with fewer characters counts as "nearly empty"


def clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_doc(doc: dict) -> dict:
    path = RAW / doc["filename"]
    reader = PdfReader(str(path))
    PAGES.mkdir(parents=True, exist_ok=True)
    out_path = PAGES / f"{doc['doc_id']}.jsonl"
    n_pages = len(reader.pages)
    total_chars, short_pages = 0, []
    with open(out_path, "w", encoding="utf-8") as out:
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = clean(page.extract_text() or "")
            except Exception as e:  # one bad page must not stop the run
                print(f"  warning: {doc['doc_id']} page {i} failed: {e}")
                text = ""
            if len(text) < MIN_CHARS:
                short_pages.append(i)
            total_chars += len(text)
            record = {
                "doc_id": doc["doc_id"],
                "symbol": doc["symbol"],
                "page": i,
                "publication_date": doc["publication_date"],
                "data_status": doc["data_status"],
                "text": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"pages": n_pages, "chars": total_chars, "short_pages": short_pages}


def main() -> int:
    if not MANIFEST_JSON.is_file():
        print("manifest.json not found. Run: python -m src.rag.check_manifest")
        return 1
    docs = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    for doc in docs:
        print(f"Extracting {doc['doc_id']} ...")
        info = extract_doc(doc)
        short = info["short_pages"]
        shown = ", ".join(map(str, short[:15])) + (" ..." if len(short) > 15 else "")
        print(f"  pages: {info['pages']}  chars: {info['chars']}  "
              f"nearly empty pages: {len(short)}" + (f" ({shown})" if short else ""))
    print(f"Done. Page files are in {PAGES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())