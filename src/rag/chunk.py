"""Split extracted pages into chunks with evidence IDs.

Reads  data/documents/manifest.json and data/documents/pages/<doc_id>.jsonl
Writes data/documents/chunks.jsonl

Rules:
- A chunk never crosses a page boundary, so every chunk has an exact page number.
- Target size TARGET_WORDS words; consecutive chunks on one page share about
  OVERLAP_WORDS words.
- Evidence ID format: doc:<doc_id>:p<page, 3 digits>:c<chunk, 2 digits>
  Example: doc:TCS-PR-Q2FY27:p02:c01  (page 2 of the press release)

Run from the project root:  python -m src.rag.chunk
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "data" / "documents"
PAGES = DOCS / "pages"
MANIFEST_JSON = DOCS / "manifest.json"
CHUNKS_JSONL = DOCS / "chunks.jsonl"

TARGET_WORDS = 280   # about 400-500 tokens for normal text
OVERLAP_WORDS = 40
MIN_WORDS = 15       # drop chunks smaller than this (page headers, cover pages)


def evidence_id(doc_id: str, page: int, chunk: int) -> str:
    return f"doc:{doc_id}:p{page:03d}:c{chunk:02d}"


def _split_long_line(line: str, limit: int) -> list:
    words = line.split()
    return [" ".join(words[i:i + limit]) for i in range(0, len(words), limit)]


def chunk_page(text: str, target: int = TARGET_WORDS, overlap: int = OVERLAP_WORDS) -> list:
    """Return a list of chunk texts for one page (line-aware, with overlap)."""
    lines = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        if len(raw.split()) > target:
            lines.extend(_split_long_line(raw, target))
        else:
            lines.append(raw)
    if not lines:
        return []

    chunks, current, count = [], [], 0
    for line in lines:
        n = len(line.split())
        if current and count + n > target:
            chunks.append("\n".join(current))
            # carry the last few lines forward as overlap
            tail, tail_words = [], 0
            for prev in reversed(current):
                w = len(prev.split())
                if tail_words + w > overlap:
                    break
                tail.insert(0, prev)
                tail_words += w
            current, count = tail, tail_words
        current.append(line)
        count += n
    if current:
        last = "\n".join(current)
        # avoid a final chunk that is only overlap text from the previous one
        if not chunks or count > len(current[0].split()) and last != chunks[-1]:
            chunks.append(last)
    return chunks


def main() -> int:
    if not MANIFEST_JSON.is_file():
        print("manifest.json not found. Run: python -m src.rag.check_manifest")
        return 1
    docs = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    total, dropped, seen = 0, 0, set()
    with open(CHUNKS_JSONL, "w", encoding="utf-8") as out:
        for doc in docs:
            page_file = PAGES / f"{doc['doc_id']}.jsonl"
            if not page_file.is_file():
                print(f"Missing {page_file.name}. Run: python -m src.rag.extract_text")
                return 1
            n_doc, words_doc = 0, []
            with open(page_file, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    pieces = chunk_page(rec["text"])
                    k = 0
                    for piece in pieces:
                        n_words = len(piece.split())
                        if n_words < MIN_WORDS:
                            dropped += 1
                            continue
                        k += 1
                        eid = evidence_id(doc["doc_id"], rec["page"], k)
                        assert eid not in seen, f"duplicate evidence id {eid}"
                        seen.add(eid)
                        out.write(json.dumps({
                            "evidence_id": eid,
                            "doc_id": doc["doc_id"],
                            "symbol": doc["symbol"],
                            "doc_type": doc["doc_type"],
                            "page": rec["page"],
                            "chunk": k,
                            "publication_date": doc["publication_date"],
                            "data_status": doc["data_status"],
                            "source_url": doc.get("source_url", ""),
                            "n_words": n_words,
                            "text": piece,
                        }, ensure_ascii=False) + "\n")
                        n_doc += 1
                        words_doc.append(n_words)
            total += n_doc
            avg = sum(words_doc) // max(1, len(words_doc))
            print(f"{doc['doc_id']}: {n_doc} chunks, words per chunk avg {avg}, "
                  f"max {max(words_doc, default=0)}")
    print(f"Total {total} chunks ({dropped} tiny pieces dropped). Wrote {CHUNKS_JSONL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())