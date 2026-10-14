"""Search the extracted page files and print the text around each match.

Usage (from the project root):
    python -m src.rag.find_text TCS-PR-Q2FY27 "10.3"
"""
import json
import re
import sys
from pathlib import Path

PAGES = Path(__file__).resolve().parents[2] / "data" / "documents" / "pages"


def main() -> int:
    if len(sys.argv) != 3:
        print('Usage: python -m src.rag.find_text <doc_id> "<text to find>"')
        return 1
    doc_id, needle = sys.argv[1], sys.argv[2]
    path = PAGES / f"{doc_id}.jsonl"
    if not path.is_file():
        print(f"No page file for {doc_id}. Run: python -m src.rag.extract_text")
        return 1
    hits = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            for m in re.finditer(re.escape(needle), text):
                hits += 1
                if hits > 8:
                    break
                a, b = max(0, m.start() - 150), min(len(text), m.end() + 150)
                snippet = text[a:b].replace("\n", " | ")
                print(f"--- page {rec['page']} ---")
                print(snippet)
    print(f"\n{hits} match(es) for {needle!r} in {doc_id}" + (" (showing first 8)" if hits > 8 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())