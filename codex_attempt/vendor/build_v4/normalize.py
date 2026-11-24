"""
Receiver-independent normalisation of raw .eml files (same procedure for every class).

Usage:
    python build_v4/normalize.py <source_name> <in_dir> <out_dir> <provenance_csv> [--recipient-token TOKEN ...]

For each email (the original file is never modified):
  1. Receiver hops: the receiving provider's own leading Received headers are removed
     (Gmail typically 2, Microsoft 365 typically 4).
  2. Authentication verdicts: the receiving server's SPF / DKIM / DMARC results (computed at
     arrival, by the same standards at every provider) are read from its top-most
     Authentication-Results header, mapped to one vocabulary (e.g. Microsoft's
     dmarc=bestguesspass -> none) and written back as ONE canonical header:
        Authentication-Results: slice-normalised; spf=..; dkim=.. header.d=..; dmarc=..
     All other Authentication-Results / ARC-* / Received-SPF headers are removed, so the
     feature modules read identical wording for every source.
  3. Receiver-only headers (X-MS-*, X-Microsoft-*, X-Forefront-*, X-Gm-*, X-Received,
     Delivered-To, X-Original-To) are removed.
  4. Recipient anonymisation: every recipient token (the mailbox owner's addresses,
     phishing_pot's 'phishing@pot') is replaced by recipient@example.invalid, including
     URL-encoded forms, in headers and in decoded text parts.
Feature modules are not changed; they simply run on the normalised copies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from email import encoders, policy
from email.generator import BytesGenerator
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path

PLACEHOLDER = "recipient@example.invalid"
MS = ("outlook.com", "office365.com", "microsoft.com")
DROP_PREFIXES = ("x-ms-", "x-microsoft-", "x-forefront-", "x-gm-", "arc-")
DROP_EXACT = {"authentication-results", "received-spf", "x-received", "delivered-to", "x-original-to",
              "authentication-results-original"}
SPF_OK = {"pass", "fail", "softfail", "neutral", "none", "temperror", "permerror"}
DKIM_OK = {"pass", "fail", "neutral", "none", "temperror", "permerror", "policy"}
DMARC_MAP = {"pass": "pass", "fail": "fail", "none": "none", "bestguesspass": "none",
             "temperror": "temperror", "permerror": "permerror"}


def provider(by_host: str) -> str:
    b = by_host or ""
    if b.endswith(MS) or "protection.outlook" in b:
        return "microsoft"
    if b.endswith("google.com") or re.match(r"^[0-9a-f]{1,4}:", b):
        return "google"
    if "proton" in b:
        return "proton"
    return ("other:" + ".".join(b.split(".")[-2:])) if b else "none"


def by_host(h) -> str:
    m = re.search(r"\bby\s+([\w.:-]+)", str(h), re.I)
    return m.group(1).lower() if m else ""


def org(domain: str) -> str:
    parts = (domain or "").lower().strip(".").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else (domain or "").lower()


def from_domain(msg) -> str:
    m = re.search(r"[\w.+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", str(msg.get("From", "")))
    return m.group(1).lower() if m else ""


def canonical_auth(ar_header: str | None, fdom: str) -> dict:
    """Map one receiver Authentication-Results header to canonical verdicts."""
    out = {"spf": "none", "spf_domain": "", "dkim": "none", "dkim_domain": "", "dmarc": "none", "ar_found": 0}
    if not ar_header:
        return out
    out["ar_found"] = 1
    h = re.sub(r"\s+", " ", ar_header)
    m = re.search(r"\bspf\s*=\s*([a-z]+)", h, re.I)
    if m and m.group(1).lower() in SPF_OK:
        out["spf"] = m.group(1).lower()
    m = re.search(r"smtp\.mailfrom\s*=\s*\"?([^\s;\"]+)", h, re.I)
    if m:
        out["spf_domain"] = m.group(1).split("@")[-1].lower()
    # every dkim=<result> ... header.d=<domain> clause; prefer an aligned pass, then any pass
    clauses = re.findall(r"\bdkim\s*=\s*([a-z]+)([^;]*)", h, re.I)
    best = None
    for res, rest in clauses:
        res = res.lower()
        d = re.search(r"header\.(?:d|i)\s*=\s*@?([A-Za-z0-9.-]+)", rest, re.I)
        d = d.group(1).lower().split("@")[-1] if d else ""
        rank = (res == "pass", res == "pass" and org(d) == org(fdom), bool(d))
        if res in DKIM_OK and (best is None or rank > best[0]):
            best = (rank, res, d)
    if best:
        out["dkim"], out["dkim_domain"] = best[1], best[2]
    m = re.search(r"\bdmarc\s*=\s*([a-z]+)", h, re.I)
    if m:
        out["dmarc"] = DMARC_MAP.get(m.group(1).lower(), "none")
    return out


def token_variants(tokens):
    out = set()
    for t in tokens:
        t = t.strip().lower()
        if not t or "@" not in t:
            continue
        out |= {t, t.replace("@", "%40"), t.replace("@", "%2540")}
    return sorted(out, key=len, reverse=True)


def replace_tokens(text: str, variants) -> tuple[str, int]:
    n = 0
    for v in variants:
        rep = PLACEHOLDER if "@" in v else PLACEHOLDER.replace("@", "%40" if "%40" in v else "%2540")
        text, k = re.subn(re.escape(v), rep, text, flags=re.I)
        n += k
    return text, n


def normalise(raw: bytes, extra_tokens) -> tuple[bytes, dict]:
    msg = BytesParser(policy=policy.compat32).parsebytes(raw)
    headers = list(msg._headers)                       # [(name, value)] in original order
    rec_idx = [i for i, (k, _) in enumerate(headers) if k.lower() == "received"]
    provs = [provider(by_host(headers[i][1])) for i in rec_idx]
    top = provs[0] if provs else "none"
    k = 0
    if top in ("google", "microsoft", "proton"):
        while k < len(provs) and provs[k] == top:
            k += 1
    strip_idx = set(rec_idx[:k])

    # receiver's verdicts: top-most Authentication-Results header
    ar = next((str(v) for kk, v in headers if kk.lower() == "authentication-results"), None)
    fdom = from_domain(msg)
    auth = canonical_auth(ar, fdom)

    arrival = ""
    if rec_idx:
        try:
            dt = parsedate_to_datetime(str(headers[rec_idx[0]][1]).rsplit(";", 1)[-1].strip())
            arrival = (dt.astimezone(tz=None).replace(tzinfo=None) if dt.tzinfo else dt).isoformat(sep=" ")
        except Exception:
            arrival = ""

    # recipient tokens: mailbox owner (Delivered-To / X-Original-To) + any supplied tokens
    tokens = set(extra_tokens)
    for kk, v in headers:
        if kk.lower() in ("delivered-to", "x-original-to"):
            tokens |= set(re.findall(r"[\w.+-]+@[\w.-]+", str(v)))
    variants = token_variants(tokens)

    kept, n_tok = [], 0
    for i, (kk, v) in enumerate(headers):
        low = kk.lower()
        if i in strip_idx or low in DROP_EXACT or low.startswith(DROP_PREFIXES):
            continue
        sv = str(v)
        sv2, n = replace_tokens(sv, variants)
        n_tok += n
        kept.append((kk, sv2 if n else v))
    canon = (f"slice-normalised; spf={auth['spf']}"
             + (f" smtp.mailfrom={auth['spf_domain']}" if auth["spf_domain"] else "")
             + f"; dkim={auth['dkim']}" + (f" header.d={auth['dkim_domain']}" if auth["dkim_domain"] else "")
             + f"; dmarc={auth['dmarc']}" + (f" header.from={fdom}" if fdom else ""))
    msg._headers = [("Authentication-Results", canon)] + kept

    body_rewritten = 0
    for part in msg.walk():
        if part.is_multipart() or part.get_content_maintype() != "text":
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        cs = part.get_content_charset() or "utf-8"
        try:
            text = payload.decode(cs, errors="replace")
        except LookupError:
            cs, text = "utf-8", payload.decode("utf-8", errors="replace")
        new, n = replace_tokens(text, variants)
        if not n:
            continue
        n_tok += n
        body_rewritten = 1
        cte = str(part.get("Content-Transfer-Encoding", "")).lower()
        del part["Content-Transfer-Encoding"]
        part.set_payload(new.encode(cs, errors="replace"))
        if cte == "base64":
            encoders.encode_base64(part)
        elif cte == "quoted-printable":
            encoders.encode_quopri(part)
        else:
            part["Content-Transfer-Encoding"] = "8bit"

    buf = BytesIO()
    BytesGenerator(buf, mangle_from_=False, maxheaderlen=0).flatten(msg)
    info = {"receiver": top, "receiver_hops_removed": k, "arrival_time": arrival,
            "from_domain": fdom, "recipient_tokens_replaced": n_tok, "body_rewritten": body_rewritten,
            **{f"auth_{x}": auth[x] for x in ("spf", "dkim", "dkim_domain", "dmarc", "ar_found")}}
    return buf.getvalue(), info


def job(args):
    src, out_dir, tokens = args
    raw = None
    try:
        raw = Path(src).read_bytes()
    except OSError as e:
        return {"file": Path(src).name, "ok": 0, "error": f"read: {type(e).__name__}"}
    rec = {"file": Path(src).name, "sha256_original": hashlib.sha256(raw).hexdigest()}
    try:
        data, info = normalise(raw, tokens)
        (Path(out_dir) / Path(src).name).write_bytes(data)
        rec.update(info, ok=1)
    except Exception as e:  # keep going; record the failure
        rec.update(ok=0, error=f"{type(e).__name__}: {e}"[:200])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("provenance_csv")
    ap.add_argument("--recipient-token", action="append", default=[])
    ap.add_argument("--files", help="optional text file listing the file names to process")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = None
    if a.files:
        names = {l.strip() for l in open(a.files, encoding="utf-8") if l.strip()}
    files = sorted(p for p in Path(a.in_dir).glob("*.eml") if names is None or p.name in names)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, [(str(f), str(out), a.recipient_token) for f in files], chunksize=32))
    for r in rows:
        r["source"] = a.source
    cols = sorted({k for r in rows for k in r})
    with open(a.provenance_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{a.source}: {sum(r.get('ok', 0) for r in rows)} normalised, "
          f"{sum(not r.get('ok') for r in rows)} failed -> {out}")


if __name__ == "__main__":
    main()
