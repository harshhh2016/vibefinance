import csv
import json

from src.eval import audit


def rec(sid, system, claims):
    return {"scenario": sid, "system": system, "efce": {"supported": claims}}


def cl(cid, text, eids):
    return {"claim_id": cid, "text": text, "evidence_ids": eids}


def test_pool_dedups_skips_synthetic_and_separates_b0(tmp_path):
    (tmp_path / "s01__P.json").write_text(json.dumps(rec("s01", "P", [cl("a", "t1", ["doc:X"]), cl("b", "t2", ["doc:X"])])))
    (tmp_path / "s02__P.json").write_text(json.dumps(rec("s02", "P", [cl("a", "t1", ["doc:X"]), cl("x1", "bad", ["doc:NOPE"])])))
    (tmp_path / "s08__P.json").write_text(json.dumps(rec("s08", "P", [cl("c1", "syn", ["doc:S"])])))
    (tmp_path / "s01__B0.json").write_text(json.dumps(rec("s01", "B0", [cl("b1", "t1", ["doc:X"]), cl("b2", "u", [])])))
    p = audit.pool(tmp_path)
    assert [c["text"] for _, _, c in p["agents"]] == ["t1", "t2"]
    assert [c["claim_id"] for _, _, c in p["B0"]] == ["b1", "b2"]


def test_make_prefills_no_cite_and_tally_requires_all_verdicts(tmp_path, capsys):
    (tmp_path / "s01__B0.json").write_text(json.dumps(rec("s01", "B0", [cl("b1", "t", []), cl("b2", "u", ["px:Z:2026-01-01"])])))
    out = tmp_path / "audit.csv"
    audit.make(tmp_path, out)
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert [r["verdict"] for r in rows] == ["no_cite", ""]
    audit.tally(out)
    assert "NOT DONE" in capsys.readouterr().out
    rows[1]["verdict"] = "ok"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    audit.tally(out)
    assert "B0" in capsys.readouterr().out
