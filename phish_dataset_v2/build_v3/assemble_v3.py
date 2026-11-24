"""Assemble slice_dataset_v3.xlsx from the two existing workbooks + provenance.csv.

Usage: python assemble_v3.py <project_dir> <provenance_csv>

Nothing in the existing workbooks is modified; a new workbook is written.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT, PROV = Path(sys.argv[1]), sys.argv[2]
INPUTS = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT  # where the two v2 workbooks are read from
sys.path.insert(0, str(PROJECT))
from feature_modules.sts_features import organizational_domain  # noqa: E402

SEED = 42
DOMAIN_CAP_FRAC = 0.05  # max share of one sender org within a class of the core subset


def main():
    legit = pd.read_excel(INPUTS / "phishing_email_dataset.xlsx", sheet_name="Dataset")
    phish = pd.read_excel(INPUTS / "phishing_pot_dataset.xlsx", sheet_name="Dataset")
    d = pd.concat([legit, phish], ignore_index=True)

    prov = pd.read_csv(PROV, dtype={"email_id": str, "source_file": str, "ms_dest_folder": str})
    idmap = prov[prov.email_id.str.startswith("phish_", na=False)][["email_id", "source_file", "sha256"]]
    prov = prov[prov.email_id.str.match(r"^(email|phish)_\d+$", na=False)]
    d = d.merge(prov, on="email_id", how="left", validate="one_to_one")

    # ---- validate the reconstructed mapping against values the builder stored ----
    def rate(a, b):
        ok = a.notna() & b.notna()
        return float((a[ok].astype(str) == b[ok].astype(str)).mean()), int(ok.sum())
    checks = {}
    for src, g in d.groupby("source"):
        checks[src] = {
            "received_hops": rate(g["STS__received_hops"].astype(float), g["raw_received_hops"].astype(float)),
            "unusual_received_chain": rate(g["STS__unusual_received_chain"].astype(float), g["raw_unusual_received_chain"].astype(float)),
            "from_domain": rate(g["STS_meta__from_domain"].fillna("").str.lower(), g["from_domain_check"].fillna("").str.lower()),
            "number_of_links": rate(g["BAS_meta__number_of_links"].astype(float), g["n_links_check"].astype(float)),
        }
    print("mapping validation (match rate, n):")
    for s, c in checks.items():
        print(" ", s, {k: (round(v[0], 4), v[1]) for k, v in c.items()})

    # ---- receiver normalisation: replace the two received-chain features ----
    d["STS_raw__received_hops"] = d["STS__received_hops"]
    d["STS_raw__unusual_received_chain"] = d["STS__unusual_received_chain"]
    d["STS__received_hops"] = d["rn_received_hops"]
    d["STS__unusual_received_chain"] = d["rn_unusual_received_chain"]

    # ---- provenance columns ----
    d["prov__source"] = d["source"]
    d["prov__receiver"] = d["receiver"]
    d["prov__receiver_hops_removed"] = d["receiver_hops_removed"]
    d["prov__arrival_time"] = pd.to_datetime(d["arrival_time"], errors="coerce")
    d["prov__arrival_month"] = d["prov__arrival_time"].dt.strftime("%Y-%m")
    d["prov__ms_delivery_folder"] = d["ms_dest_folder"].map({"I": "inbox", "J": "junk", "C": "other"}).fillna("")
    # Each <tenant>.onmicrosoft.com is a separate (often attacker-created) Microsoft 365
    # organisation, so it is its own sender org rather than one "onmicrosoft.com".
    def sender_org(x):
        if not x:
            return ""
        org = organizational_domain(x)
        return x.lower() if org == "onmicrosoft.com" else org
    d["prov__sender_org"] = d["STS_meta__from_domain"].fillna("").map(sender_org)
    d["prov__has_links"] = (d["BAS_meta__number_of_links"].fillna(0) > 0).astype(int)

    # ---- core subset ----
    lo = d.loc[d.label == 0, "prov__arrival_time"].min()
    hi = d.loc[d.label == 0, "prov__arrival_time"].max()
    d["prov__in_time_window"] = d["prov__arrival_time"].between(lo, hi).astype(int)
    cand = (d.prov__in_time_window == 1) & (d.prov__has_links == 1) & (d.processing_status == "ok")
    rng = np.random.default_rng(SEED)
    keep = pd.Series(False, index=d.index)
    for lab in (0, 1):
        idx = d.index[cand & (d.label == lab)]
        cap = max(1, int(DOMAIN_CAP_FRAC * len(idx)))
        for _, g in d.loc[idx].groupby("prov__sender_org"):
            take = g.index.to_numpy()
            if len(take) > cap:
                take = rng.choice(take, cap, replace=False)
            keep.loc[take] = True
    d["prov__core"] = keep.astype(int)
    d["prov__excluded_by_domain_cap"] = (cand & ~keep).astype(int)

    # ---- time-ordered split on the core (cut points = legit 70% / 85% arrival quantiles) ----
    core_legit_t = d.loc[(d.prov__core == 1) & (d.label == 0), "prov__arrival_time"]
    t1, t2 = core_legit_t.quantile(0.70), core_legit_t.quantile(0.85)
    d["prov__time_split"] = ""
    c = d.prov__core == 1
    d.loc[c & (d.prov__arrival_time <= t1), "prov__time_split"] = "train"
    d.loc[c & (d.prov__arrival_time > t1) & (d.prov__arrival_time <= t2), "prov__time_split"] = "val"
    d.loc[c & (d.prov__arrival_time > t2), "prov__time_split"] = "test"

    helper = ["source", "receiver", "receiver_hops_removed", "arrival_time", "ms_dest_folder", "sha256", "mid_hash",
              "raw_received_hops", "raw_unusual_received_chain", "rn_received_hops", "rn_unusual_received_chain",
              "from_domain_check", "n_links_check", "ok", "source_file", "analysis_error", "error"]
    d = d.drop(columns=[x for x in helper if x in d.columns])
    lead = ["email_id", "label", "processing_status"]
    provc = [x for x in d.columns if x.startswith("prov__")]
    d = d[lead + provc + [x for x in d.columns if x not in lead + provc]]

    # ---- stats ----
    rows = []
    def add(m, v): rows.append({"metric": m, "value": v})
    add("all rows", len(d))
    for lab in (0, 1):
        s = d[d.label == lab]
        add(f"label {lab}: rows", len(s))
        add(f"label {lab}: receivers", s.prov__receiver.value_counts().to_dict())
        add(f"label {lab}: arrival range", f"{s.prov__arrival_time.min()} .. {s.prov__arrival_time.max()}")
        add(f"label {lab}: in legit time window", int(s.prov__in_time_window.sum()))
        add(f"label {lab}: with >=1 link", int(s.prov__has_links.sum()))
        add(f"label {lab}: core rows", int(s.prov__core.sum()))
        add(f"label {lab}: core distinct sender orgs", int(s[s.prov__core == 1].prov__sender_org.nunique()))
        add(f"label {lab}: core split counts", s[s.prov__core == 1].prov__time_split.value_counts().to_dict())
    add("time window (legit arrival min..max)", f"{lo} .. {hi}")
    add("time split cut points", f"train <= {t1} < val <= {t2} < test")
    add("domain cap", f"{DOMAIN_CAP_FRAC:.0%} of a class's core candidates per sender org, seed {SEED}")
    for s, cdict in checks.items():
        for k, v in cdict.items():
            add(f"mapping check {s}: {k}", f"{v[0]:.4f} over {v[1]} rows")
    stats = pd.DataFrame(rows)
    stats["value"] = stats["value"].astype(str)

    guide = pd.DataFrame([
        ("STS__received_hops", "Receiver-normalised: Received headers counted after removing the receiving provider's own leading hops (Gmail usually 2, Microsoft 365 usually 4). Same module function (sts_features.analyze_received_chain)."),
        ("STS__unusual_received_chain", "Receiver-normalised in the same way."),
        ("STS_raw__received_hops / STS_raw__unusual_received_chain", "Original values from the v2 workbooks (not prefixed STS__, so the training code ignores them)."),
        ("all other <FAMILY>__ columns", "Unchanged from the v2 workbooks."),
        ("prov__source", "gmail_inbox (label 0) or phishing_pot (label 1)."),
        ("prov__receiver", "Provider of the top Received 'by' host: google / microsoft / proton / other."),
        ("prov__receiver_hops_removed", "How many leading Received headers were stripped before the normalised features."),
        ("prov__arrival_time / prov__arrival_month", "Time stamped by the receiving server in the top Received header (falls back to Date)."),
        ("prov__ms_delivery_folder", "phishing_pot only: where Microsoft 365 delivered it (inbox = got past Microsoft filtering)."),
        ("prov__sender_org", "Organizational domain of the From address (use for grouped splits)."),
        ("prov__has_links", "1 if the email has at least one http(s) <a href> link (the patent's claim scope)."),
        ("prov__in_time_window", "Arrival within the legit set's window."),
        ("prov__core", "Recommended training/evaluation subset: in time window, >=1 link, processing ok, sender-org cap applied."),
        ("prov__excluded_by_domain_cap", "Would be core but dropped by the per-sender-org cap."),
        ("prov__time_split", "Time-ordered train/val/test for core rows (same cut dates for both classes)."),
    ], columns=["column", "meaning"])

    link_l = pd.read_excel(INPUTS / "phishing_email_dataset.xlsx", sheet_name="Link_Details")
    link_p = pd.read_excel(INPUTS / "phishing_pot_dataset.xlsx", sheet_name="Link_Details")
    links = pd.concat([link_l, link_p], ignore_index=True)
    links = links.merge(d[["email_id", "label", "prov__core", "prov__time_split"]], on="email_id", how="left")

    out = PROJECT / "slice_dataset_v3.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        d.to_excel(xw, sheet_name="Dataset", index=False)
        links.to_excel(xw, sheet_name="Link_Details", index=False)
        stats.to_excel(xw, sheet_name="Dataset_Stats", index=False)
        guide.to_excel(xw, sheet_name="Column_Guide", index=False)
    d[d.prov__core == 1].to_csv(PROJECT / "slice_dataset_v3_core.csv", index=False)
    idmap.to_csv(PROJECT / "private" / "phishing_pot_id_map_reconstructed.csv", index=False)
    print(stats.to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
