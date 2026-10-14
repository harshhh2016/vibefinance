"""Score retrieval methods with recall@k on the labelled question set.

Usage (from the project root, venv active):
    $env:PYTHONUTF8 = "1"
    python -m src.rag.eval_retrieval --method baseline --split dev
    python -m src.rag.eval_retrieval --method baseline --split heldout --final

Rules: tune on --split dev only. Use --split heldout (needs --final) once, at the very end.
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

KS = (1, 3, 5, 10, 15)
CHUNK_ID = re.compile(r"^(?P<doc>.+?):p(?P<page>\d+)")

# ---------------------------------------------------------------- methods
# Each method takes (query, k, as_of) and returns a ranked list of hits.
def baseline(query, k, as_of, company="TCS"):
    from src.rag.retrieve import retrieve
    return retrieve(query, company, as_of, k=k)

METHODS = {"baseline": baseline}
# Later: METHODS["hybrid"] = ..., METHODS["capped"] = ...

# ---------------------------------------------------------------- hit parsing
def hit_key(hit):
    """Return (doc_id, page) from a retrieve() result (dict, object or chunk-id string)."""
    def get(name):
        if isinstance(hit, dict):
            return hit.get(name)
        return getattr(hit, name, None)
    doc, page = get("doc_id") or get("document_id"), get("page")
    if doc is not None and page is not None:
        return str(doc), int(page)
    cid = hit if isinstance(hit, str) else (get("chunk_id") or get("id"))
    m = CHUNK_ID.match(str(cid))
    if not m:
        raise ValueError(f"Cannot read doc/page from hit: {hit!r}")
    return m["doc"], int(m["page"])

def first_hit_rank(hits, expected):
    ok = {(e["doc"], e["page"]) for e in expected}
    for rank, h in enumerate(hits, 1):
        if hit_key(h) in ok:
            return rank
    return None

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="baseline", choices=sorted(METHODS))
    ap.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    ap.add_argument("--final", action="store_true", help="required to touch the held-out split")
    ap.add_argument("--questions", default="data/eval/questions.json")
    ap.add_argument("--as-of", default="2026-10-15")
    ap.add_argument("--out", default=None, help="optional path to save per-question results as JSON")
    a = ap.parse_args()

    if a.split in ("heldout", "all") and not a.final:
        sys.exit("Held-out split is locked. Tune on --split dev; add --final only for the last run.")

    rows = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    rows = [r for r in rows if a.split == "all" or r["split"] == a.split]
    scored = [r for r in rows if r["answerable"] and r["expected"]]
    skipped = [r["id"] for r in rows if r["answerable"] and not r["expected"]]
    n_unans = sum(1 for r in rows if not r["answerable"])

    fn, kmax = METHODS[a.method], max(KS)
    results, by_type = [], defaultdict(list)
    for r in scored:
        rank = first_hit_rank(fn(r["question"], kmax, a.as_of), r["expected"])
        results.append({"id": r["id"], "type": r["type"], "rank": rank, "question": r["question"]})
        by_type[r["type"]].append(rank)

    def recall(ranks, k):
        return sum(1 for x in ranks if x is not None and x <= k) / len(ranks) if ranks else float("nan")
    def mrr(ranks):
        return sum(1 / x for x in ranks if x) / len(ranks) if ranks else float("nan")

    allranks = [x["rank"] for x in results]
    print(f"\nmethod={a.method}  split={a.split}  scored={len(scored)}  "
          f"skipped(todo)={len(skipped)}  unanswerable(not scored)={n_unans}\n")
    print("MISSES (not in top %d):" % kmax)
    for x in results:
        if x["rank"] is None:
            print(f"  {x['id']} [{x['type']}] {x['question']}")
    print("\nrank of first correct page:")
    for x in results:
        print(f"  {x['id']:<5} {str(x['rank'] or '-'):>3}  [{x['type']}]")
    header = "".join(f"  R@{k:<2}" for k in KS)
    print(f"\n{'group':<14}{'n':>3}{header}    MRR")
    for name, ranks in [("ALL", allranks)] + sorted(by_type.items()):
        print(f"{name:<14}{len(ranks):>3}" + "".join(f"  {recall(ranks,k):.2f}" for k in KS) + f"   {mrr(ranks):.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
