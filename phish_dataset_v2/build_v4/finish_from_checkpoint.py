"""
Write the builder's workbook + id map from a finished checkpoint, with a fast privacy check.

Usage:
    python build_v4/finish_from_checkpoint.py private/v4/checkpoint_v4.jsonl private/v4/build_v4.xlsx private/v4/id_map_v4.csv

Why: build_dataset_fast.privacy_findings() tests every needle (subjects, names, addresses,
body snippets of every email) against every text cell one by one, which grows with the
square of the dataset size and takes hours at ~10k emails. This script runs the SAME check
(same needles, same case-insensitive substring rule, same e-mail-address regex) with an
Aho-Corasick automaton, then writes the workbook with the builder's own write_workbook()
and build_inventory(). Feature values come unchanged from the checkpoint.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import ahocorasick

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import build_dataset_fast as bdf  # noqa: E402


def privacy_findings_fast(tables, privs):
    needles = set()
    for p in privs:
        if p["subject"] and len(p["subject"]) >= 6:
            needles.add(p["subject"].lower())
        needles |= {n.lower() for n in p["names"]}
        needles |= set(p["addresses"])
        needles |= {b.lower() for b in p["body"]}
    needles.discard("")
    # A needle longer than the longest text cell can never be a substring of any cell, so
    # dropping those leaves the result identical (and keeps the automaton small: body
    # snippets can be >200k characters).
    max_cell = max((len(v) for t in tables.values() for r in t for v in r.values() if isinstance(v, str)), default=0)
    needles = {n for n in needles if len(n) <= max_cell}
    auto = ahocorasick.Automaton()
    for n in needles:
        auto.add_word(n, n)
    auto.make_automaton()
    bad = set()
    for name, table in tables.items():
        for r in table:
            for k, v in r.items():
                if not isinstance(v, str):
                    continue
                if next(auto.iter(v.lower()), None) is not None:
                    bad.add(f"{name}.{k}: private string")
                if bdf.EMAIL_RE.search(v):
                    bad.add(f"{name}.{k}: e-mail address")
    return sorted(bad), len(needles)


def main():
    ckpt, out, id_map = sys.argv[1:4]
    t0 = time.time()
    rows, link_rows, errors, privs, idmap = [], [], [], [], []
    for line in open(ckpt, encoding="utf-8"):
        rec = json.loads(line)
        rows.append(rec["row"])
        link_rows.extend(rec.get("links", []))
        errors.extend(rec.get("errors", []))
        if rec.get("priv"):
            privs.append(rec["priv"])
        idmap.append((rec["email_id"], rec["path"], rec["sha256"]))
    print(f"loaded {len(rows)} emails, {len(link_rows)} links ({time.time() - t0:.0f}s)", flush=True)

    bad, n_needles = privacy_findings_fast({"Dataset": rows, "Link_Details": link_rows}, privs)
    print(f"privacy check: {n_needles} needles, {len(bad)} findings ({time.time() - t0:.0f}s)", flush=True)
    for b in bad:
        errors.append({"email_id": "*", "error_type": "PrivacyCheck", "error_message": b,
                       "failed_module": "privacy_filter", "failed_feature": "-"})

    inventory = bdf.build_inventory(rows, link_rows)
    per = Counter(i["module"] for i in inventory if i["kind"] == "feature" and i["sheet"] == "Dataset")
    stats = {
        "total emails (unique, in Dataset)": len(rows),
        "legitimate emails (label 0)": sum(r["label"] == 0 for r in rows),
        "phishing emails (label 1)": sum(r["label"] == 1 for r in rows),
        "successfully processed (all 6 modules ran)": sum(r.get("processing_status") == "ok" for r in rows),
        "partially processed (some module failed)": sum(str(r.get("processing_status", "")).startswith("partial") for r in rows),
        "failed emails (no module ran)": sum(r.get("processing_status") == "failed" for r in rows),
        "total links (Link_Details rows)": len(link_rows),
        "total feature columns (Dataset, module features)": sum(per.values()),
        "privacy check": "PASS" if not bad else f"FAIL ({len(bad)} findings, see Processing_Errors)",
        "privacy check method": "Aho-Corasick (same needles and rule as build_dataset_fast.privacy_findings)",
        "workbook written from checkpoint": ckpt,
        "generated at (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    bdf.write_workbook(out, rows, link_rows, errors, stats, inventory)
    with open(id_map, "w", encoding="utf-8") as fh:  # PRIVATE - never commit
        fh.write("email_id,source_file,sha256\n")
        fh.writelines(f"{e},{p},{h}\n" for e, p, h in idmap)
    print(json.dumps(stats, indent=1))
    print(f"wrote {out} and {id_map} ({time.time() - t0:.0f}s)")
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
