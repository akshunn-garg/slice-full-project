"""
Robustness of SLiCE vs the conventional all-features model (B2) - writes reports/benchmark/robustness_*.csv

    python -m slice_models.benchmark_robustness

1. Legit authentication-failure stress test: the legit TEST emails are given realistic
   authentication failures (their links and content unchanged) and the share wrongly flagged
   is measured at each model's own VAL threshold:
     forwarded       SPF fails (DKIM and DMARC still pass)
     mailing_list    SPF, DKIM and DMARC fail; the DKIM signer is the list, not the sender
     no_auth_sender  small sender with no SPF/DKIM/DMARC set up (no DKIM domain)
   Every feature that depends on these facts is changed consistently (STS verdicts and binary
   features, DKIM alignment, and the DKIM-domain comparisons in BAS/DMS).
2. Matched operating points: Nazario recall for both models at thresholds giving the same VAL
   false-alarm rate (0.5%, 1%, 2%).
"""
from __future__ import annotations

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

SCENARIOS = {
    "as observed": None,
    "forwarded (SPF fails)": dict(spf="fail", dkim="pass", dmarc="pass", spf_pass=0, dkim_pass=1, dmarc_pass=1,
                                  aligned=1, from_eq_dkim=1, keep_dkim_dest=True),
    "mailing list (SPF/DKIM/DMARC fail, list signs)": dict(spf="fail", dkim="fail", dmarc="fail", spf_pass=0,
                                                           dkim_pass=0, dmarc_pass=0, aligned=0, from_eq_dkim=0,
                                                           keep_dkim_dest=False),
    "small sender, no authentication": dict(spf="none", dkim="none", dmarc="none", spf_pass=0, dkim_pass=0,
                                            dmarc_pass=0, aligned=0, from_eq_dkim=0, keep_dkim_dest=False),
}


def perturb(x: pd.DataFrame, s: dict) -> pd.DataFrame:
    x = x.copy()
    x["STS_meta__spf_result"], x["STS_meta__dkim_result"], x["STS_meta__dmarc_result"] = s["spf"], s["dkim"], s["dmarc"]
    x["STS__spf_pass"], x["STS__dkim_pass"], x["STS__dmarc_pass"] = s["spf_pass"], s["dkim_pass"], s["dmarc_pass"]
    x["STS__dkim_aligned"] = s["aligned"]
    x["BAS__from_domain_equals_dkim_domain"] = s["from_eq_dkim"]
    if not s["keep_dkim_dest"]:
        x["BAS__authenticated_domain_equals_destination"] = 0
        x["DMS__dkim_domain_equals_destination"] = 0
    return x


def main():
    d, nz = load()
    y = d.label.to_numpy()
    sp = d.prov__time_split.to_numpy()
    tr, va, te = sp == "train", sp == "val", sp == "test"
    cols = [c for f in ("STS", "BAS", "DMS", "LOS", "RCS", "SRC", "INFRA") for c in family_columns(d.columns, f)]
    sent = {k: v for k, v in SENTINELS.items() if k in cols}
    b2 = fit_lgb(preprocess(d, cols, sent), y, tr, va)
    p2 = b2(preprocess(d, cols, sent))
    t2, _ = select_threshold(y[va], p2[va], 0.01)
    art = joblib.load(ROOT / "models" / "final3" / "slice_final_model.joblib")
    ps = predict_final(d).FINAL_prob.to_numpy()
    ts = art["threshold"]

    legit_te = d[te & (y == 0)].reset_index(drop=True)
    rows = []
    for name, s in SCENARIOS.items():
        x = legit_te if s is None else perturb(legit_te, s)
        f2 = (b2(preprocess(x, cols, sent)) >= t2).mean()
        fs = (predict_final(x).FINAL_prob.to_numpy() >= ts).mean()
        rows.append({"legit test scenario": name, "B2 all-features: legit flagged %": round(100 * f2, 2),
                     "SLiCE final: legit flagged %": round(100 * fs, 2)})
    stress = pd.DataFrame(rows)

    pn2 = b2(preprocess(nz, cols, sent))
    pns = predict_final(nz).FINAL_prob.to_numpy()
    match = []
    for target in (0.005, 0.01, 0.02):
        a2, _ = select_threshold(y[va], p2[va], target)
        as_, _ = select_threshold(y[va], ps[va], target)
        match.append({"VAL false-alarm target": f"{target:.1%}",
                      "B2 test FPR %": round(100 * (p2[te & (y == 0)] >= a2).mean(), 2),
                      "SLiCE test FPR %": round(100 * (ps[te & (y == 0)] >= as_).mean(), 2),
                      "B2 test recall %": round(100 * (p2[te & (y == 1)] >= a2).mean(), 2),
                      "SLiCE test recall %": round(100 * (ps[te & (y == 1)] >= as_).mean(), 2),
                      "B2 Nazario recall %": round(100 * (pn2 >= a2).mean(), 2),
                      "SLiCE Nazario recall %": round(100 * (pns >= as_).mean(), 2)})
    match = pd.DataFrame(match)
    rep = ROOT / "reports" / "benchmark"
    stress.to_csv(rep / "robustness_legit_auth_failures.csv", index=False)
    match.to_csv(rep / "robustness_matched_operating_points.csv", index=False)
    pd.set_option("display.width", 250)
    print("Legit authentication-failure stress test (test legit, n=%d):" % len(legit_te))
    print(stress.to_string(index=False))
    print("\nMatched operating points:")
    print(match.to_string(index=False))


if __name__ == "__main__":
    main()
