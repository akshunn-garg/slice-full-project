"""Per-email provenance + receiver-normalised received-chain features.

Usage: python provenance.py <project_dir> <pot_raw_dir> <tree.json> <out_csv>

Raw .eml bytes are only parsed (never executed). The received-chain values are
computed with the project's own sts_features.analyze_received_chain, applied to
(a) the unmodified message (to validate the email_id mapping) and (b) the same
message with the receiving provider's own leading Received headers removed.
"""
import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path

PROJECT, POT_RAW, TREE, OUT = sys.argv[1:5]
sys.path.insert(0, PROJECT)

MS = ("outlook.com", "office365.com", "microsoft.com")


def provider(by_host):
    b = by_host or ""
    if b.endswith(MS) or "protection.outlook" in b:
        return "microsoft"
    if b.endswith("google.com") or re.match(r"^[0-9a-f]{1,4}:", b):
        return "google"
    if "proton" in b:
        return "proton"
    if "yahoo" in b:
        return "yahoo"
    return ("other:" + ".".join(b.split(".")[-2:])) if b else "none"


def by_host(h):
    m = re.search(r"\bby\s+([\w.:-]+)", str(h), re.I)
    return m.group(1).lower() if m else ""


def received_time(h):
    try:
        dt = parsedate_to_datetime(str(h).rsplit(";", 1)[-1].strip())
        return dt.astimezone(tz=None).replace(tzinfo=None) if dt.tzinfo else dt
    except Exception:
        return None


def analyse(raw: bytes):
    from feature_modules import sts_features, bas_features

    msg = BytesParser(policy=policy.default).parsebytes(raw)
    rec = msg.get_all("Received") or []
    provs = [provider(by_host(h)) for h in rec]
    top = provs[0] if provs else "none"
    k = 0
    while k < len(rec) and provs[k] == top and top not in ("none",) and not top.startswith("other:"):
        k += 1

    raw_chain = sts_features.analyze_received_chain(msg)

    # Remove exactly the first k Received headers (the receiver's own hops).
    stripped = BytesParser(policy=policy.default).parsebytes(raw)
    kept = [str(h) for h in (stripped.get_all("Received") or [])][k:]
    del stripped["Received"]
    for h in kept:
        stripped["Received"] = h
    norm_chain = sts_features.analyze_received_chain(stripped)

    arrival = received_time(rec[0]) if rec else None
    if arrival is None:
        try:
            arrival = parsedate_to_datetime(str(msg.get("Date"))).replace(tzinfo=None)
        except Exception:
            arrival = None

    try:
        n_links = len(bas_features.extract_links(bas_features.get_html(msg)))
    except Exception:
        n_links = -1

    deliv = re.search(r"dest:([A-Z])", str(msg.get("X-Microsoft-Antispam-Mailbox-Delivery") or ""))
    return {
        "receiver": top,
        "receiver_hops_removed": k,
        "raw_received_hops": raw_chain["received_hops"],
        "raw_unusual_received_chain": raw_chain["unusual_received_chain"],
        "rn_received_hops": norm_chain["received_hops"],
        "rn_unusual_received_chain": norm_chain["unusual_received_chain"],
        "arrival_time": arrival.isoformat(sep=" ") if arrival else "",
        "from_domain_check": sts_features.extract_email_domain(str(msg.get("From", ""))) or "",
        "n_links_check": n_links,
        "ms_dest_folder": deliv.group(1) if deliv else "",
    }


def job(item):
    key, path = item
    try:
        raw = Path(path).read_bytes()
    except Exception as e:
        # Local copy unreadable (e.g. quarantined by antivirus): read the same file
        # from the repository into memory instead.
        if key[0] != "pot":
            return {"key": key, "ok": 0, "error": f"read: {type(e).__name__}"}
        import urllib.request
        try:
            raw = urllib.request.urlopen(
                "https://raw.githubusercontent.com/rf-peixoto/phishing_pot/main/" + key[1], timeout=60).read()
        except Exception as e2:
            return {"key": key, "ok": 0, "error": f"read: {type(e).__name__}; fetch: {type(e2).__name__}"}
    # Identity fields exactly as the builder computes them (needed for the ID replay
    # even if the analysis below fails on a malformed message).
    ident = {"sha256": hashlib.sha256(raw).hexdigest(), "mid_hash": ""}
    try:
        mid = (BytesParser(policy=policy.default).parsebytes(raw).get("Message-ID") or "").strip().strip("<>").lower()
        ident["mid_hash"] = hashlib.sha256(mid.encode()).hexdigest() if mid else ""
    except Exception:
        pass
    try:
        r = analyse(raw)
        r["ok"] = 1
    except Exception as e:  # keep going; record the failure
        r = {"ok": 1, "analysis_error": f"{type(e).__name__}: {e}"[:200]}
    r.update(ident)
    r["key"] = key
    return r


def main():
    project = Path(PROJECT)
    # Legit: the builder's own private id map (email_id -> local file).
    legit = []
    with open(project / "private" / "id_map.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            legit.append((("legit", row["email_id"]), str(project / row["source_file"].replace("\\", "/"))))
    # Phishing: every file, in the builder's order (sorted raw URLs == sorted paths).
    paths = sorted(x["path"] for x in json.load(open(TREE))["tree"]
                   if x["type"] == "blob" and x["path"].startswith("email/") and x["path"].lower().endswith(".eml"))
    phish = [(("pot", p), str(Path(POT_RAW) / p.split("/", 1)[1])) for p in paths]

    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(job, legit + phish, chunksize=32))

    # Replay the stream builder's ID assignment: sorted order, skip a file whose raw
    # SHA-256 or Message-ID hash was already seen, number the rest phish_000001...
    seen_raw, seen_mid, n = set(), set(), 0
    out = []
    for r in res:
        src, ident = r.pop("key")
        r["source"] = "gmail_inbox" if src == "legit" else "phishing_pot"
        if src == "legit":
            r["email_id"], r["source_file"] = ident, ""
        else:
            r["source_file"] = ident
            if not r.get("ok"):
                r["email_id"] = ""
            elif r["sha256"] in seen_raw or (r["mid_hash"] and r["mid_hash"] in seen_mid):
                r["email_id"] = "DUPLICATE"
            else:
                seen_raw.add(r["sha256"])
                if r["mid_hash"]:
                    seen_mid.add(r["mid_hash"])
                n += 1
                r["email_id"] = f"phish_{n:06d}"
        out.append(r)
    cols = sorted({k for r in out for k in r})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(out)
    print("phishing ids assigned:", n, "duplicates:", sum(r["email_id"] == "DUPLICATE" for r in out),
          "failures:", sum(not r.get("ok") for r in out), flush=True)


if __name__ == "__main__":
    main()
