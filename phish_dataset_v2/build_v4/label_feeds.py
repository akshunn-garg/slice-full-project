"""
Label emails by matching their links against public phishing / malware URL feeds.

Usage:
    python build_v4/label_feeds.py <eml_dir> <out_csv> [--feeds-dir private/feeds]

An email is 'feed_phishing' if any http(s) link's full URL appears in OpenPhish or URLhaus
(recent), or its exact hostname appears in Phishing.Database (ACTIVE domains). Host matches
on shared platforms (SHARED_SUFFIXES: Google, GitHub, S3, email-service tracking hosts, ...)
are ignored, because those feeds list the platform itself after a single abuse. Emails
without a match are left unlabelled, not called legitimate. Feeds are downloaded once and kept in --feeds-dir with their fetch date.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from datetime import date
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_modules import bas_features  # noqa: E402  (link extraction exactly as the modules do it)

FEEDS = {
    "openphish": "https://openphish.com/feed.txt",
    "urlhaus": "https://urlhaus.abuse.ch/downloads/csv_recent/",
    "phishing_database": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt",
}


def fetch(feeds_dir: Path) -> dict[str, Path]:
    feeds_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in FEEDS.items():
        p = feeds_dir / f"{name}_{date.today().isoformat()}.txt"
        if not p.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "slice-research/1.0"})
            p.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        out[name] = p
    return out


# Shared platforms: a feed listing the platform host itself (because someone once abused it)
# says nothing about other users' links there, so host-level matches on these are ignored.
SHARED_SUFFIXES = (
    "google.com", "googleusercontent.com", "github.com", "githubusercontent.com", "amazonaws.com",
    "cloudfront.net", "azureedge.net", "windows.net", "sharepoint.com", "office.com", "live.com",
    "microsoft.com", "dropbox.com", "onelink.me", "bit.ly", "instagram.com", "facebook.com",
    "linkedin.com", "twitter.com", "x.com", "youtube.com", "rs6.net", "sparkpostmail.com",
    "sparkpostmail1.com", "sendgrid.net", "list-manage.com", "mailchimp.com", "hubspotlinks.com",
)


def _shared(host: str) -> bool:
    return any(host == s or host.endswith("." + s) for s in SHARED_SUFFIXES)


def load(paths: dict[str, Path]):
    """URL feeds (openphish, urlhaus) match full URLs only; the domain feed matches hosts,
    except shared platforms."""
    urls, hosts = {}, {}
    for name, p in paths.items():
        text = p.read_text(encoding="utf-8", errors="ignore")
        if name == "urlhaus":
            rows = csv.reader(io.StringIO("\n".join(l for l in text.splitlines() if not l.startswith("#"))))
            items = [r[2] for r in rows if len(r) > 2]
        else:
            items = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
        for it in items:
            if "://" in it:
                urls.setdefault(it.rstrip("/").lower(), set()).add(name)
            else:
                h = it.lower().strip(".")
                if h and not _shared(h):
                    hosts.setdefault(h, set()).add(name)
    return urls, hosts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eml_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--feeds-dir", default="private/feeds")
    a = ap.parse_args()
    paths = fetch(Path(a.feeds_dir))
    urls, hosts = load(paths)
    print("feed sizes:", {k: sum(1 for _ in open(v, encoding="utf-8", errors="ignore")) for k, v in paths.items()},
          "| urls", len(urls), "hosts", len(hosts))
    rows = []
    for f in sorted(Path(a.eml_dir).glob("*.eml")):
        try:
            msg = BytesParser(policy=policy.default).parsebytes(f.read_bytes())
            links = [l["url"] for l in bas_features.extract_links(bas_features.get_html(msg))]
        except Exception:
            links = []
        hits = set()
        for u in links:
            hits |= urls.get(u.rstrip("/").lower(), set())
            try:
                hits |= hosts.get((urlsplit(u).hostname or "").lower(), set())
            except ValueError:  # malformed URL
                pass
        rows.append({"file": f.name, "n_links": len(links), "feed_phishing": int(bool(hits)),
                     "feeds": "|".join(sorted(hits))})
    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["file", "n_links", "feed_phishing", "feeds"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} emails | with links {sum(r['n_links'] > 0 for r in rows)} | "
          f"feed-matched {sum(r['feed_phishing'] for r in rows)} -> {a.out_csv}")


if __name__ == "__main__":
    main()
