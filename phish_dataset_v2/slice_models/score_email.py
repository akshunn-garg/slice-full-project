"""
Score raw emails (.eml) end to end with the final SLiCE models, timing each stage.

    python -m slice_models.score_email path/to/mail.eml [more.eml ...]
    python -m slice_models.score_email --dir some/folder --out scores.csv

Pipeline per email (identical to how the training data was built):
  1. normalise  - build_v4/normalize.py: receiver hops stripped, one canonical
                  Authentication-Results header, recipient addresses replaced
  2. features   - the six unmodified feature modules, through build_dataset_fast's own
                  worker functions (live DNS / RDAP / TLS / redirect lookups, shared cache)
  3. models     - slice_final_predict.predict_final (models/final3: 4 link scores + INFRA ->
                  link model, + rule-based STS with fixed weight 3)
The feature modules contact every link host (HTTP GET, DNS, TLS). Only score emails you are
willing to have those lookups made for; unsubscribe-looking links are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import build_dataset_fast as bdf  # noqa: E402
from build_v4.normalize import normalise  # noqa: E402
from slice_models.slice_final_predict import predict_final  # noqa: E402

SHOW = ["file", "FINAL_prob", "FINAL_flag", "FINAL_link_prob", "STS_rule", "BAS_prob", "DMS_prob", "LOS_prob",
        "RCS_prob", "top_reasons", "STS_rule_reasons", "n_links", "t_normalise_s", "t_features_s", "t_models_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emails", nargs="*", help=".eml files")
    ap.add_argument("--dir", help="score every .eml in this folder")
    ap.add_argument("--out", help="write all columns to this CSV")
    ap.add_argument("--model-dir", default=None, help="default: models/final3")
    ap.add_argument("--cache", default=str(ROOT / "private" / "lookup_cache.sqlite"))
    a = ap.parse_args()
    files = [Path(p) for p in a.emails] + (sorted(Path(a.dir).glob("*.eml")) if a.dir else [])
    if not files:
        ap.error("give .eml files or --dir")

    bdf._worker_init(True, a.cache)                       # same modules, cache and observer as the builder
    tmp = Path(tempfile.mkdtemp(prefix="slice_score_"))
    rows, timing = [], []
    try:
        for i, f in enumerate(files, 1):
            raw = f.read_bytes()
            t0 = time.perf_counter()
            norm, _ = normalise(raw, ["phishing@pot"])
            p = tmp / f"email_{i:05d}.eml"
            p.write_bytes(norm)
            t1 = time.perf_counter()
            res = bdf._worker_run((str(p), f"email_{i:05d}", -1, hashlib.sha256(norm).hexdigest(), False, True))
            t2 = time.perf_counter()
            row = res["row"]
            row["file"] = f.name
            rows.append(row)
            timing.append({"t_normalise_s": t1 - t0, "t_features_s": t2 - t1})
            print(f"[{i}/{len(files)}] {f.name}: features in {t2 - t1:.1f}s ({row.get('processing_status')})", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    df = pd.DataFrame(rows).reset_index(drop=True)
    t3 = time.perf_counter()
    out = predict_final(df, model_dir=a.model_dir, explain=True)
    per_email_model_s = (time.perf_counter() - t3) / len(df)
    result = pd.concat([df[["file"]], out, pd.DataFrame(timing)], axis=1)
    result["n_links"] = pd.to_numeric(df.get("BAS_meta__number_of_links"), errors="coerce")
    result["t_models_s"] = per_email_model_s
    result["verdict"] = result["FINAL_flag"].map({1: "PHISHING", 0: "legit"})
    no_links = result["n_links"].fillna(0) == 0
    result.loc[no_links, "verdict"] += " (no links: outside the model's scope)"
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(result[["file", "verdict"] + [c for c in SHOW[1:] if c in result]].round(3).to_string(index=False))
    if a.out:
        result.to_csv(a.out, index=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
