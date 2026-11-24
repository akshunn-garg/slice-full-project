"""
Rule-based Source Trust Score (STS), following the patent's evidence table (section 6.3, step 4).

    from slice_models.sts_rules import sts_rule_score
    out = sts_rule_score(df)          # columns: STS_rule (1 = trustworthy, 0 = suspicious), STS_rule_risk, STS_rule_reasons

The weights below were fixed BEFORE looking at any dataset and are not fitted, so the score
cannot learn source or receiver artefacts from the training data. Inputs are the STS module's
own outputs (STS__* features and STS_meta__* verdict strings). STS__received_hops is
deliberately not used: the hop count depends on the receiving provider.

Risk per evidence item is in [0, 1]; STS_rule = 1 - weighted mean risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {
    "spf": 1.0,
    "dkim": 1.0,
    "dmarc": 1.5,
    "reply_to_mismatch": 1.0,
    "return_path_mismatch": 0.5,   # bulk senders legitimately use a different bounce domain
    "sender_domain_age": 1.0,
    "sender_domain_entropy": 0.5,
    "unusual_received_chain": 0.5,
}

SPF_RISK = {"pass": 0.0, "fail": 1.0}                       # everything else (softfail, neutral, none, errors) -> 0.5
DMARC_RISK = {"pass": 0.0, "fail": 1.0, "permerror": 1.0}   # none / temperror / unknown -> 0.5
AGE_RISK = {0: 1.0, 1: 1.0, 2: 0.6, 3: 0.3, 4: 0.0, -1: 0.3}  # new_sender_domain label; -1 = age unknown (neutral)


def _dkim_risk(result: str, aligned: float) -> float:
    if result == "pass":
        return 0.0 if aligned == 1 else 0.5
    if result in ("none", "unknown", "", "nan"):
        return 0.6
    return 1.0                                               # fail, neutral, policy, temperror, permerror


def _entropy_risk(h: float) -> float:
    if pd.isna(h):
        return 0.0
    return 1.0 if h > 3.8 else (0.5 if h > 3.5 else 0.0)


def sts_rule_score(df: pd.DataFrame) -> pd.DataFrame:
    spf = df["STS_meta__spf_result"].astype(str).str.lower()
    dkim = df["STS_meta__dkim_result"].astype(str).str.lower()
    dmarc = df["STS_meta__dmarc_result"].astype(str).str.lower()
    risks = pd.DataFrame(index=df.index)
    risks["spf"] = spf.map(SPF_RISK).fillna(0.5)
    risks["dkim"] = [_dkim_risk(r, a) for r, a in zip(dkim, df["STS__dkim_aligned"])]
    risks["dmarc"] = dmarc.map(DMARC_RISK).fillna(0.5)
    risks["reply_to_mismatch"] = df["STS__from_reply_to_mismatch"].fillna(0).astype(float)
    risks["return_path_mismatch"] = df["STS__from_return_path_mismatch"].fillna(0).astype(float)
    risks["sender_domain_age"] = df["STS__new_sender_domain"].fillna(-1).astype(int).map(AGE_RISK).fillna(0.3)
    risks["sender_domain_entropy"] = df["STS__sender_domain_entropy"].map(_entropy_risk)
    risks["unusual_received_chain"] = df["STS__unusual_received_chain"].fillna(0).astype(float)

    w = pd.Series(WEIGHTS)
    risk = (risks[w.index] * w).sum(axis=1) / w.sum()
    contrib = risks[w.index] * w
    reasons = [
        "; ".join(f"{k} ({row[k]:.1f})" for k in row.sort_values(ascending=False).index if row[k] > 0)
        for _, row in contrib.iterrows()
    ]
    return pd.DataFrame({"STS_rule": (1 - risk).round(4), "STS_rule_risk": risk.round(4),
                         "STS_rule_reasons": reasons}, index=df.index)
