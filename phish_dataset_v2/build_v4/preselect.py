"""
Pick which normalised emails go through the (slow, lookup-heavy) feature build.

Usage:
    python build_v4/preselect.py --legit-dir private/v4/legit_all --legit-prov private/v4/legit_prov.csv \
        --phish-dir private/v4/phish_all --phish-prov private/v4/phish_prov_all.csv \
        --legit-feeds private/v4/legit_feed_labels.csv --out-dir private/v4 --per-class 6000

Identical rules for both classes, using only cheap, local information:
  1. >= 1 http(s) link (link extraction exactly as bas_features does it)
  2. legit emails whose links hit a phishing feed are dropped
  3. <= --per-template emails per (sender org, set of link hosts' registered domains)
  4. <= --org-cap of the class per sender org (each *.onmicrosoft.com tenant is its own org)
  5. time matching: in every arrival month both classes get the SAME number of emails
     (min of the two, scaled so each class totals about --per-class)
Writes legit_selected.txt / phish_selected.txt and preselect_summary.csv into --out-dir.
A buffer above the final per-class size is intended: the builder can still drop emails.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from feature_modules.bas_features import extract_links, get_html  # noqa: E402
from feature_modules.sts_features import organizational_domain  # noqa: E402


def link_info(path: str):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(Path(path).read_bytes())
        urls = [l["url"] for l in extract_links(get_html(msg))]
    except Exception:
        urls = []
    def host(u):
        try:
            return (urlsplit(u).hostname or "").lower()
        except ValueError:  # malformed URL (e.g. bad bracketed host)
            return ""
    doms = sorted({organizational_domain(h) for h in map(host, urls) if h})
    return Path(path).name, len(urls), "|".join(doms)


TWO_PART_SUFFIXES = {"co.in", "ac.in", "gov.in", "org.in", "net.in", "nic.in", "res.in", "edu.in", "co.uk",
                     "org.uk", "ac.uk", "gov.uk", "com.br", "net.br", "org.br", "gov.br", "com.au", "net.au",
                     "co.jp", "co.za", "com.mx", "com.ar", "com.co", "co.id", "web.id", "com.tr", "com.sg",
                     "com.my", "co.nz", "com.pk", "com.ng", "com.cn", "com.hk", "com.tw", "co.kr"}


def sender_org(x):
    """Organisation of a sender domain. Each *.onmicrosoft.com tenant is its own org, and
    two-part country suffixes (co.in, com.br, ...) keep one more label."""
    if not isinstance(x, str) or not x:
        return ""
    x = x.lower().strip(".")
    o = organizational_domain(x)
    if o == "onmicrosoft.com":
        return x
    if o in TWO_PART_SUFFIXES:
        parts = x.split(".")
        return ".".join(parts[-3:]) if len(parts) >= 3 else x
    return o


def cap(df, key, n, rng):
    keep = []
    for _, g in df.groupby(key, sort=False):
        idx = g.index.to_numpy()
        keep.extend(rng.choice(idx, n, replace=False) if len(idx) > n else idx)
    return df.loc[sorted(keep)]


def candidates(eml_dir, prov_csv, workers):
    prov = pd.read_csv(prov_csv, dtype=str)
    prov = prov[prov["ok"] == "1"]
    files = [str(Path(eml_dir) / f) for f in prov["file"]]
    with ProcessPoolExecutor(workers) as ex:
        info = pd.DataFrame(list(ex.map(link_info, files, chunksize=64)), columns=["file", "n_links", "link_orgs"])
    d = prov.merge(info, on="file")
    d["arrival"] = pd.to_datetime(d["arrival_time"], errors="coerce")
    d["month"] = d["arrival"].dt.strftime("%Y-%m")
    d["org"] = d["from_domain"].map(sender_org)
    return d


def main():
    ap = argparse.ArgumentParser()
    for k in ("legit-dir", "legit-prov", "phish-dir", "phish-prov", "out-dir"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--legit-feeds")
    ap.add_argument("--per-class", type=int, default=6000)
    ap.add_argument("--per-template", type=int, default=5)
    ap.add_argument("--org-cap", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []

    pools = {}
    for name, ddir, prov in (("legit", a.legit_dir, a.legit_prov), ("phish", a.phish_dir, a.phish_prov)):
        d = candidates(ddir, prov, a.workers)
        rows.append((name, "normalised ok", len(d)))
        d = d[(d["n_links"] > 0) & d["arrival"].notna()]
        rows.append((name, ">=1 link", len(d)))
        if name == "legit" and a.legit_feeds:
            lf = pd.read_csv(a.legit_feeds)
            d = d[~d["file"].isin(set(lf.loc[lf.feed_phishing == 1, "file"]))]
            rows.append((name, "not on phishing feeds", len(d)))
        d["template"] = d["org"] + "#" + d["link_orgs"]
        d = cap(d.reset_index(drop=True), "template", a.per_template, rng)
        rows.append((name, f"<= {a.per_template} per template", len(d)))
        d = cap(d, "org", max(1, int(a.org_cap * len(d))), rng)
        rows.append((name, f"sender-org cap {a.org_cap:.0%}", len(d)))
        pools[name] = d

    # time matching: same count per month for both classes
    common = pd.concat([pools["legit"].groupby("month").size().rename("legit"),
                        pools["phish"].groupby("month").size().rename("phish")], axis=1).fillna(0)
    common["matched"] = common.min(axis=1).astype(int)
    total = common["matched"].sum()
    scale = min(1.0, a.per_class / total) if total else 0
    common["take"] = np.floor(common["matched"] * scale).astype(int)
    picked = {}
    for name, d in pools.items():
        sel = []
        for m, n in common["take"].items():
            g = d[d["month"] == m]
            if n > 0:
                sel.extend(rng.choice(g["file"].to_numpy(), n, replace=False))
        picked[name] = sorted(sel)
        rows.append((name, "time-matched selection", len(sel)))
        (Path(a.out_dir) / f"{name}_selected.txt").write_text("\n".join(sel) + "\n", encoding="utf-8")
    common.to_csv(Path(a.out_dir) / "preselect_months.csv")
    pd.DataFrame(rows, columns=["class", "step", "emails"]).to_csv(Path(a.out_dir) / "preselect_summary.csv", index=False)
    for r in rows:
        print(*r, sep=" | ")
    print("months with matched data:", int((common["take"] > 0).sum()),
          f"({common.index[common['take'] > 0].min()} .. {common.index[common['take'] > 0].max()})")


if __name__ == "__main__":
    main()
