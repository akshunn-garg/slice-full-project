"""
Assemble slice_dataset_v4.xlsx from the builder's output on the normalised emails.

Usage (from the project root):
    python build_v4/assemble_v4.py --build private/v4/build_v4.xlsx --id-map private/v4/id_map_v4.csv \
        --prov private/v4/legit_prov.csv private/v4/phish_prov_all.csv \
        --legit-feeds private/v4/legit_feed_labels.csv --per-class 5000

The candidate emails were already chosen by build_v4/preselect.py (>= 1 link, feed check,
template and sender-org caps, time matching). Here, after the feature build, with identical
rules for both classes:
  1. processing_status == ok and >= 1 link as counted by the modules
  2. legit emails whose links hit a phishing feed are dropped (guard)
  3. <= --per-template emails per (class, sender org, link-organisation set)
  4. time matching again (the builder can drop emails): same count per arrival month in both
     classes, at most --per-class per class
  5. prov__diverse_subset flags a stricter, time-matched subset (<= --diverse-per-template)
Then: rule-based STS (slice_models.sts_rules), lookup-lag column, time-ordered split,
shortcut audit (single-feature ROC, missingness by class), and a datasheet sheet.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from feature_modules.sts_features import organizational_domain  # noqa: E402
from slice_models.sts_rules import WEIGHTS, sts_rule_score  # noqa: E402

FAMILIES = ("STS", "BAS", "DMS", "LOS", "RCS", "SRC", "INFRA")


from build_v4.preselect import sender_org  # noqa: E402  (same org rule as the pre-selection)


def capped_sample(df, key, cap, rng):
    keep = []
    for _, g in df.groupby(key, sort=False):
        idx = g.index.to_numpy()
        keep.extend(rng.choice(idx, cap, replace=False) if len(idx) > cap else idx)
    return df.loc[sorted(keep)]


def time_matched(df, per_class, rng):
    """Same number of emails per arrival month in both classes (min of the two, scaled so
    each class has at most per_class rows)."""
    counts = df.groupby(["prov__arrival_month", "label"]).size().unstack(fill_value=0)
    if 0 not in counts or 1 not in counts:
        return df.iloc[0:0]
    matched = counts.min(axis=1)
    scale = min(1.0, per_class / matched.sum()) if matched.sum() else 0
    take = np.floor(matched * scale).astype(int)
    keep = []
    for (month, lab), g in df.groupby(["prov__arrival_month", "label"]):
        n = int(take.get(month, 0))
        if n:
            keep.extend(rng.choice(g.index.to_numpy(), n, replace=False))
    return df.loc[sorted(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="+", required=True)
    ap.add_argument("--id-map", nargs="+", required=True)
    ap.add_argument("--prov", nargs="+", required=True)
    ap.add_argument("--legit-feeds")
    ap.add_argument("--out", default=str(ROOT / "slice_dataset_v4.xlsx"))
    ap.add_argument("--per-class", type=int, default=5000)
    ap.add_argument("--per-template", type=int, default=100, help="same value as the pre-selection")
    ap.add_argument("--diverse-per-template", type=int, default=20)
    ap.add_argument("--build-date", default=datetime.now().strftime("%Y-%m-%d"),
                    help="date the external lookups were made (for the lag column)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    log = []

    d = pd.concat([pd.read_excel(b, sheet_name="Dataset") for b in a.build], ignore_index=True)
    links = pd.concat([pd.read_excel(b, sheet_name="Link_Details") for b in a.build], ignore_index=True)
    if not d["email_id"].is_unique:
        raise ValueError("email_id collision across builds")

    def read_idmap(p):
        first = open(p, encoding="utf-8").readline().split(",")[0]
        return pd.read_csv(p) if first == "email_id" else pd.read_csv(p, header=None, names=["email_id", "path", "sha256"])
    idm = pd.concat([read_idmap(x) for x in a.id_map], ignore_index=True)
    idm = idm.rename(columns={c: "path" for c in idm.columns if c in ("source_file", "file")})
    idm["file"] = idm["path"].astype(str).str.replace("\\", "/").str.split("/").str[-1]
    prov = pd.concat([pd.read_csv(p, dtype=str) for p in a.prov], ignore_index=True)
    prov = prov.drop_duplicates("file")
    d = d.merge(idm[["email_id", "file"]], on="email_id", how="left").merge(prov, on="file", how="left")
    log.append(("rows from builder", len(d)))

    d["prov__source"] = d["source"]
    d["prov__receiver"] = d["receiver"]
    d["prov__arrival_time"] = pd.to_datetime(d["arrival_time"], errors="coerce")
    d["prov__arrival_month"] = d["prov__arrival_time"].dt.strftime("%Y-%m")
    d["prov__sender_org"] = d["STS_meta__from_domain"].map(sender_org)
    d["prov__lookup_lag_days"] = (pd.Timestamp(a.build_date) - d["prov__arrival_time"]).dt.days
    d["prov__n_links"] = d["BAS_meta__number_of_links"].fillna(0).astype(int)

    # 1. ok + >=1 link
    d = d[(d["processing_status"] == "ok") & (d["prov__n_links"] > 0)]
    log.append(("ok and >=1 link", d.groupby("label").size().to_dict()))
    # 2. legit with feed-listed links (pre-selection already applied this; kept as a guard)
    if a.legit_feeds:
        lf = pd.read_csv(a.legit_feeds)
        bad = set(lf.loc[lf.feed_phishing == 1, "file"])
        d = d[~((d.label == 0) & d["file"].isin(bad))]
        log.append((f"legit not on phishing feeds ({len(bad)} matched)", d.groupby("label").size().to_dict()))
    # 3. template id (sender org + set of link organisations) for grouped CV / diversity checks
    dom_sets = links.groupby("email_id")["destination"].apply(
        lambda s: "|".join(sorted({sender_org(str(x)) for x in s.dropna()})))
    d["prov__template"] = d["prov__sender_org"] + "#" + d["email_id"].map(dom_sets).fillna("")
    d = capped_sample(d, ["label", "prov__template"], a.per_template, rng)
    log.append((f"<= {a.per_template} per template", d.groupby("label").size().to_dict()))
    # 4. time matching: the same number of emails per arrival month in both classes
    d = time_matched(d, a.per_class, rng)
    log.append((f"time-matched, <= {a.per_class} per class", d.groupby("label").size().to_dict()))
    # 5. diverse subset flag: <= --diverse-per-template per template, then time-matched again
    div = time_matched(capped_sample(d, ["label", "prov__template"], a.diverse_per_template, rng), a.per_class, rng)
    d["prov__diverse_subset"] = d.index.isin(div.index).astype(int)
    log.append((f"diverse subset (<= {a.diverse_per_template} per template, time-matched)",
                div.groupby("label").size().to_dict()))
    d = d.reset_index(drop=True)

    # rule-based STS
    d = pd.concat([d, sts_rule_score(d)], axis=1)

    # time-ordered split (cut points from the arrival times of both classes together)
    t1, t2 = d["prov__arrival_time"].quantile([0.70, 0.85])
    d["prov__time_split"] = np.where(d["prov__arrival_time"] <= t1, "train",
                                     np.where(d["prov__arrival_time"] <= t2, "val", "test"))
    log.append(("split counts", d.groupby(["prov__time_split", "label"]).size().to_dict()))

    # shortcut audit
    feats = [c for c in d.columns if c.split("__")[0] in FAMILIES and "__" in c and "_meta__" not in c]
    audit = []
    for c in feats + ["STS_rule_risk"]:
        x = pd.to_numeric(d[c], errors="coerce")
        miss = x.isna().groupby(d["label"]).mean()
        auc = np.nan
        if x.notna().sum() > 10 and x.nunique() > 1:
            auc = roc_auc_score(d.loc[x.notna(), "label"], x[x.notna()])
            auc = max(auc, 1 - auc)
        audit.append({"feature": c, "single_feature_roc": round(auc, 4) if pd.notna(auc) else np.nan,
                      "missing_legit": round(miss.get(0, 0), 3), "missing_phish": round(miss.get(1, 0), 3),
                      "flag": "CHECK: near-perfect alone" if pd.notna(auc) and auc > 0.95 else
                              ("CHECK: missingness differs" if abs(miss.get(0, 0) - miss.get(1, 0)) > 0.3 else "")})
    audit = pd.DataFrame(audit).sort_values("single_feature_roc", ascending=False)

    prov_cols = [c for c in d.columns if c.startswith("prov__")]
    lead = ["email_id", "label"] + prov_cols + ["STS_rule", "STS_rule_risk", "STS_rule_reasons"]
    auth_cols = [c for c in d.columns if c.startswith("auth_")]
    feat_cols = [c for c in d.columns if c.split("__")[0] in FAMILIES or c.split("__")[0].endswith("_meta")]
    out = d[lead + auth_cols + [c for c in feat_cols if c not in lead]]
    out = out.rename(columns={c: "prov__" + c for c in auth_cols})
    link_out = links[links["email_id"].isin(out["email_id"])].merge(
        out[["email_id", "label", "prov__time_split"]], on="email_id")

    datasheet = pd.DataFrame([
        ("Sources", "legit: one Gmail inbox (received by Gmail); phishing: rf-peixoto/phishing_pot (received by one "
                    "Microsoft 365 organisation). Gmail Spam is excluded until hand-labelled (private/v4/spam_to_label.xlsx)."),
        ("Normalisation", "build_v4/normalize.py, identical for every email: receiver's own Received hops removed; "
                          "receiver SPF/DKIM/DMARC verdicts (computed at arrival) mapped to one vocabulary and written as one "
                          "canonical Authentication-Results header; receiver-only headers removed; recipient addresses "
                          "(mailbox owner, phishing@pot) replaced by recipient@example.invalid incl. URL-encoded forms."),
        ("Features", "The six unmodified feature modules, run by build_dataset_fast.py on the normalised copies."),
        ("STS", f"STS_rule = rule-based score, weights fixed a priori (slice_models/sts_rules.py): {WEIGHTS}. "
                "STS__received_hops is not used by the rule."),
        ("Selection", " -> ".join(f"{k}: {v}" for k, v in log)),
        ("Lookups", f"Domain age, DNS, TLS, redirects were looked up around {a.build_date}; prov__lookup_lag_days = days "
                    "between arrival and lookup. Restrict RCS/INFRA analyses to small lags for a fresh-lookup check."),
        ("Splits", "prov__time_split: time-ordered 70/15/15 on arrival time (same cut dates for both classes). "
                   "prov__sender_org and prov__template support grouped CV."),
        ("Privacy", "Builder privacy check (same rule, run by build_v4/finish_from_checkpoint.py) reports 39 columns: all "
                    "false positives from short sender display names / brand words ('a', 'com', 'netflix', 'naukri') and "
                    "generic one-word subjects ('reminder', 'donation', 'kinder') occurring inside domains/URLs. A targeted "
                    "check found no e-mail addresses, no multi-word names and no body text in any output cell."),
        ("Known limits","One legit mailbox and one phishing receiver organisation; Gmail and Microsoft verdicts are "
                         "standards-based but computed by different receivers; labels of phishing_pot not hand-verified; "
                         "see the Audit sheet for features that still separate the classes on their own."),
    ], columns=["item", "description"])

    with pd.ExcelWriter(a.out, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="Dataset", index=False)
        link_out.to_excel(xw, sheet_name="Link_Details", index=False)
        audit.to_excel(xw, sheet_name="Audit", index=False)
        datasheet.to_excel(xw, sheet_name="Datasheet", index=False)
    out.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    for k, v in log:
        print(f"{k}: {v}")
    print(audit.head(12).to_string(index=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
