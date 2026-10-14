"""Validate data/documents/manifest.csv and write data/documents/manifest.json.

Run from the project root:  python -m src.rag.check_manifest
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "data" / "documents"
RAW = DOCS / "raw"
MANIFEST_CSV = DOCS / "manifest.csv"
MANIFEST_JSON = DOCS / "manifest.json"

SYMBOLS = {"RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "SBIN"}
DOC_TYPES = {"annual_report", "quarterly_results", "transcript", "announcement", "presentation"}
DATA_STATUSES = {"real", "synthetic"}
COLUMNS = ["doc_id", "symbol", "title", "doc_type", "publication_date",
           "source_url", "filename", "data_status", "notes"]
ID_RE = re.compile(r"^[A-Za-z0-9-]+$")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_row(row: dict, seen_ids: set) -> list:
    problems = []
    doc_id = (row.get("doc_id") or "").strip()
    if not doc_id or not ID_RE.match(doc_id):
        problems.append("doc_id must be letters, digits and dashes only")
    elif doc_id in seen_ids:
        problems.append("duplicate doc_id")
    if (row.get("symbol") or "").strip() not in SYMBOLS:
        problems.append("symbol not in the 7 stocks")
    if (row.get("doc_type") or "").strip() not in DOC_TYPES:
        problems.append("doc_type must be one of " + ", ".join(sorted(DOC_TYPES)))
    status = (row.get("data_status") or "").strip()
    if status not in DATA_STATUSES:
        problems.append("data_status must be real or synthetic")
    pub = (row.get("publication_date") or "").strip()
    try:
        d = datetime.strptime(pub, "%Y-%m-%d").date()
        if d > date.today():
            problems.append("publication_date is in the future")
    except ValueError:
        problems.append("publication_date must be YYYY-MM-DD")
    if status == "real" and not (row.get("source_url") or "").strip().startswith("http"):
        problems.append("real documents need a source_url starting with http")
    fname = (row.get("filename") or "").strip()
    if not fname:
        problems.append("filename is empty")
    elif not (RAW / fname).is_file():
        problems.append(f"file not found in data\\documents\\raw: {fname}")
    return problems


def main() -> int:
    if not MANIFEST_CSV.is_file():
        print(f"Cannot find {MANIFEST_CSV}")
        return 1
    with open(MANIFEST_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            print("Header problem. Expected columns:")
            print(",".join(COLUMNS))
            print("Found:")
            print(",".join(reader.fieldnames or []))
            return 1
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]

    seen, valid_docs, n_problems = set(), [], 0
    for i, row in enumerate(rows, start=2):  # line 1 is the header
        if None in row:
            print(f"line {i}: too many columns (a comma inside a field? wrap it in double quotes)")
            n_problems += 1
            continue
        problems = check_row(row, seen)
        doc_id = (row.get("doc_id") or "").strip()
        if doc_id:
            seen.add(doc_id)
        if problems:
            for p in problems:
                print(f"line {i} ({doc_id or 'no id'}): {p}")
            n_problems += len(problems)
            continue
        entry = {k: (row[k] or "").strip() for k in COLUMNS}
        path = RAW / entry["filename"]
        entry["sha256"] = sha256_of(path)
        entry["size_bytes"] = path.stat().st_size
        valid_docs.append(entry)

    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(valid_docs, f, indent=2, ensure_ascii=False)

    print(f"{len(valid_docs)} valid documents, {n_problems} problems")
    for sym, n in sorted(Counter(d["symbol"] for d in valid_docs).items()):
        print(f"  {sym}: {n}")
    return 0 if n_problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())