"""
Non-phishing spam test: Gmail Spam emails (hand-checked by the mailbox owner: spam, none phishing).

    python -m slice_models.eval_spam [--checkpoint private/v4/spam_checkpoint.jsonl]

Same receiving server (Gmail) as the legit training mail, so no receiver differences. Measures how
often unwanted-but-not-phishing mail is called phishing, by SLiCE (final threshold and the 3-tier
block/warn/pass policy from VAL) and by the conventional all-features model (B2).
Emails without links are outside the models' scope and are reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slice_models.benchmark_baselines import fit_lgb, load
from slice_models.slice_final_predict import predict_final
from slice_models.slice_predict import SENTINELS, family_columns, preprocess
from slice_models.train_slice_models import select_threshold

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "private" / "v4" / "spam_checkpoint.jsonl"))
    a = ap.parse_args()
    rows = pd.DataFrame([json.loads(l)["row"] for l in open(a.checkpoint, encoding="utf-8")])
    n_all = len(rows)
    ok = rows[rows.processing_status == "ok"].reset_index(drop=True)
    has_links = pd.to_numeric(ok.BAS_meta__number_of_links, errors="coerce").fillna(0) > 0
    s = ok[has_links].reset_index(drop=True)

    art = joblib.load(ROOT / "models" / "final3" / "slice_final_model.joblib")
    p = predict_final(s).FINAL_prob.to_numpy()
    tiers = pd.read_csv(ROOT / "reports" / "benchmark" / "policy_three_tier.csv")  # existence check only
    d, _ = load()
    y, sp = d.label.to_numpy(), d.prov__time_split.to_numpy()
    ps_val = predict_final(d[sp == "val"]).FINAL_prob.to_numpy()
    t_block, _ = select_threshold(y[sp == "val"], ps_val, 0.001)
    t_warn = art["threshold"]

    cols = [c for f in ("STS", "BAS", "DMS", "LOS", "RCS", "SRC", "INFRA") for c in family_columns(d.columns, f)]
    sent = {k: v for k, v in SENTINELS.items() if k in cols}
    b2 = fit_lgb(preprocess(d, cols, sent), y, sp == "train", sp == "val")
    t2, _ = select_threshold(y[sp == "val"], b2(preprocess(d[sp == "val"], cols, sent)), 0.01)
    p2 = b2(preprocess(s, cols, sent))

    out = pd.DataFrame([
        {"model": "SLiCE final3", "spam emails (with links)": len(s),
         "called phishing %": round(100 * (p >= t_warn).mean(), 1),
         "blocked %": round(100 * (p >= t_block).mean(), 1),
         "warned %": round(100 * ((p >= t_warn) & (p < t_block)).mean(), 1),
         "passed %": round(100 * (p < t_warn).mean(), 1)},
        {"model": "B2 all-features flat", "spam emails (with links)": len(s),
         "called phishing %": round(100 * (p2 >= t2).mean(), 1), "blocked %": np.nan, "warned %": np.nan,
         "passed %": round(100 * (p2 < t2).mean(), 1)},
    ])
    auth = s[["STS_meta__spf_result", "STS_meta__dkim_result", "STS_meta__dmarc_result"]].apply(
        lambda c: (c == "pass").mean()).round(3).to_dict()
    out.to_csv(ROOT / "reports" / "benchmark" / "spam_not_phishing.csv", index=False)
    print(f"Gmail spam: {n_all} built, {len(ok)} ok, {int(has_links.sum())} with links (scored), "
          f"{int((~has_links).sum())} without links (out of scope)")
    print("authentication pass rates among scored spam:", auth)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
