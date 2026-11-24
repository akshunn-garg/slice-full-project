"""
Turn a builder checkpoint into feature rows for train_slice_final.py --extra-train.

Usage:
    python build_v4/assemble_extra.py <checkpoint.jsonl> <prov.csv> <source_name> <out.csv>

Keeps processing_status == ok and >= 1 link, and adds the same provenance columns as
slice_dataset_v4 (source, arrival time, sender org, template), so grouped CV works unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from build_v4.preselect import sender_org  # noqa: E402


def main():
    ckpt, prov_csv, source, out = sys.argv[1:5]
    recs = [json.loads(l) for l in open(ckpt, encoding="utf-8")]
    rows = pd.DataFrame([r["row"] for r in recs])
    rows["file"] = [Path(str(r["path"]).replace("\\", "/")).name for r in recs]
    links = pd.DataFrame([x for r in recs for x in r.get("links", [])])
    prov = pd.read_csv(prov_csv, dtype=str)[["file", "arrival_time"]]
    d = rows.merge(prov, on="file", how="left")
    d = d[(d["processing_status"] == "ok") & (pd.to_numeric(d["BAS_meta__number_of_links"], errors="coerce") > 0)].copy()
    dom = links.groupby("email_id")["destination"].apply(lambda s: "|".join(sorted({sender_org(str(x)) for x in s.dropna()})))
    d["email_id"] = source + "_" + d["email_id"].astype(str)
    dom.index = source + "_" + dom.index.astype(str)
    d["prov__source"] = source
    d["prov__arrival_time"] = pd.to_datetime(d["arrival_time"], errors="coerce")
    d["prov__sender_org"] = d["STS_meta__from_domain"].map(sender_org)
    d["prov__template"] = d["prov__sender_org"].fillna("") + "#" + d["email_id"].map(dom).fillna("")
    d = d.drop(columns=["file", "arrival_time"])
    d.to_csv(out, index=False)
    print(f"{source}: {len(d)} rows -> {out} | labels {d['label'].value_counts().to_dict()} | "
          f"arrival {d['prov__arrival_time'].min()} .. {d['prov__arrival_time'].max()}")


if __name__ == "__main__":
    main()
