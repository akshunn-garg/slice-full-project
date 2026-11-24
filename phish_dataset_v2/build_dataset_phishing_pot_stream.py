#!/usr/bin/env python3
"""
.eml folders  ->  phishing_email_dataset.xlsx  using the SIX exact feature modules.

The modules in feature_modules/ are byte-identical copies of the uploaded code.
This file only orchestrates: it calls

    STS   sts_features.extract_sts_features(eml_path)            -> (features, metadata)
    BAS   bas_features.extract_bas_features(eml_path)            -> (features, metadata)
    DMS   dms_features.extract_dms_features(eml_path)            -> (features, metadata)
    LOS   los_features.extract_los_features(eml_path)            -> (features, metadata)
    RCS   rcs_features.extract_rcs_features(eml_path)            -> (features, metadata)
    SRC/INFRA source_link_infra_features.extract_source_link_vector(eml_path) -> dict

and writes what they return. No feature is computed, re-aggregated or altered
here. Email-level columns are the modules' own email-level outputs (including
their own link -> email aggregation); link-level outputs go to Link_Details.

External lookups made inside the modules are OBSERVED (wrappers record the
outcome and then return / raise exactly what the original did) so that
failures can be reported without changing any feature value.

Usage
  python build_dataset.py --inbox-dir private/inbox_eml --limit 5 --out test.xlsx   # test batch
  python build_dataset.py --inbox-dir private/inbox_eml --phishing-dir private/phish_eml
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import re
import socket
import ssl
import sys
import threading
import time
import traceback
from collections import Counter
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

import sqlite3
import pickle
import socket as _socket_module

# ---------------------------------------------------------------------------
# Persistent external-lookup cache
#
# This cache is orchestration-only. It does not change any feature formula or
# module code. It memoizes the exact return values/errors of external lookups
# so repeated URLs/domains across emails and worker processes do not trigger
# the same network request again.
# ---------------------------------------------------------------------------
class PersistentLookupCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.local = threading.local()
        self._init_db()

    def _conn(self):
        c = getattr(self.local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.path), timeout=30)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=30000")
            self.local.conn = c
        return c

    def _init_db(self):
        c = sqlite3.connect(str(self.path), timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA busy_timeout=30000")
        c.execute("""
            CREATE TABLE IF NOT EXISTS lookup_cache (
                cache_key TEXT PRIMARY KEY,
                ok INTEGER NOT NULL,
                payload BLOB,
                error_type TEXT,
                error_message TEXT
            )
        """)
        c.commit()
        c.close()

    def get(self, key):
        row = self._conn().execute(
            "SELECT ok, payload, error_type, error_message "
            "FROM lookup_cache WHERE cache_key=?",
            (key,)
        ).fetchone()
        return row

    def put_ok(self, key, value):
        c = self._conn()
        try:
            c.execute(
                "INSERT OR REPLACE INTO lookup_cache "
                "(cache_key,ok,payload,error_type,error_message) VALUES (?,?,?,?,?)",
                (key, 1, sqlite3.Binary(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)), None, None)
            )
            c.commit()
        except sqlite3.OperationalError:
            c.rollback()

    def put_error(self, key, exc):
        c = self._conn()
        try:
            c.execute(
                "INSERT OR REPLACE INTO lookup_cache "
                "(cache_key,ok,payload,error_type,error_message) VALUES (?,?,?,?,?)",
                (key, 0, None, type(exc).__name__, str(exc)[:1000])
            )
            c.commit()
        except sqlite3.OperationalError:
            c.rollback()

    def close(self):
        c = getattr(self.local, "conn", None)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
            self.local.conn = None


def _cache_key(kind, *parts):
    raw = kind + "\0" + "\0".join(repr(x) for x in parts)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


def _recreate_cached_exception(name, message):
    cls = getattr(requests.exceptions, name, requests.exceptions.RequestException)
    try:
        return cls(message)
    except Exception:
        return requests.exceptions.RequestException(message)


def _response_pack(r):
    history = []
    for h in getattr(r, "history", []) or []:
        history.append({
            "status_code": int(getattr(h, "status_code", 0)),
            "url": getattr(h, "url", ""),
            "headers": dict(getattr(h, "headers", {}) or {}),
            "encoding": getattr(h, "encoding", None),
            "content": bytes(getattr(h, "content", b"") or b""),
        })
    return {
        "status_code": int(getattr(r, "status_code", 0)),
        "url": getattr(r, "url", ""),
        "headers": dict(getattr(r, "headers", {}) or {}),
        "encoding": getattr(r, "encoding", None),
        "content": bytes(getattr(r, "content", b"") or b""),
        "history": history,
    }


def _response_unpack(data):
    def one(d):
        r = requests.Response()
        r.status_code = d["status_code"]
        r.url = d["url"]
        r.headers.update(d.get("headers", {}))
        r.encoding = d.get("encoding")
        r._content = d.get("content", b"")
        return r
    r = one(data)
    r.history = [one(x) for x in data.get("history", [])]
    return r


def _install_persistent_cache(cache):
    """Install process-local wrappers backed by one shared SQLite cache."""
    orig_get = requests.get
    orig_ghbn = socket.gethostbyname

    def cached_get(url, *args, **kwargs):
        # Only cache the request forms used by the supplied feature modules.
        headers = kwargs.get("headers") or {}
        allow_redirects = kwargs.get("allow_redirects", True)
        key = _cache_key("requests.get", str(url), bool(allow_redirects),
                         tuple(sorted((str(k), str(v)) for k, v in headers.items())))
        row = cache.get(key)
        if row is not None:
            ok, payload, err_name, err_msg = row
            if ok:
                return _response_unpack(pickle.loads(payload))
            raise _recreate_cached_exception(err_name, err_msg)

        try:
            r = orig_get(url, *args, **kwargs)
        except Exception as exc:
            cache.put_error(key, exc)
            raise
        cache.put_ok(key, _response_pack(r))
        return r

    def cached_ghbn(hostname):
        key = _cache_key("socket.gethostbyname", str(hostname).lower())
        row = cache.get(key)
        if row is not None:
            ok, payload, err_name, err_msg = row
            if ok:
                return pickle.loads(payload)
            raise OSError(err_msg)

        try:
            value = orig_ghbn(hostname)
        except Exception as exc:
            cache.put_error(key, exc)
            raise
        cache.put_ok(key, value)
        return value

    requests.get = cached_get
    socket.gethostbyname = cached_ghbn

    # Cache the pure helper results used by the supplied modules. These wrappers
    # preserve the helpers' exact returned values; they only avoid repeat I/O.
    modules = [importlib.import_module(m) for _, m, _ in MODULES]

    def wrap_func(mod, name, cache_name):
        original = getattr(mod, name, None)
        if original is None or getattr(original, "_phish_cache_wrapped", False):
            return
        def wrapped(*args, **kwargs):
            key = _cache_key(cache_name, *args, *sorted(kwargs.items()))
            row = cache.get(key)
            if row is not None:
                ok, payload, err_name, err_msg = row
                if ok:
                    return pickle.loads(payload)
                raise _recreate_cached_exception(err_name, err_msg)
            try:
                value = original(*args, **kwargs)
            except Exception as exc:
                cache.put_error(key, exc)
                raise
            cache.put_ok(key, value)
            return value
        wrapped._phish_cache_wrapped = True
        setattr(mod, name, wrapped)

    # STS/RCS domain-age lookups share one cache namespace.
    for mod in modules:
        if hasattr(mod, "get_domain_creation_date"):
            wrap_func(mod, "get_domain_creation_date", "rdap.domain_creation")
    rcs = importlib.import_module("feature_modules.rcs_features")
    src = importlib.import_module("feature_modules.source_link_infra_features")

    for name in ("resolve_domain_to_ip", "get_ip_metadata"):
        if hasattr(rcs, name):
            wrap_func(rcs, name, "rcs." + name)

    for name in ("get_ip_metadata", "get_dns_ttl", "get_dns_records",
                 "get_nameserver_info", "get_ip_from_domain", "get_certificate_info"):
        if hasattr(src, name):
            wrap_func(src, name, "src." + name)

    return cache


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
log = logging.getLogger("build_dataset")

# ---------------------------------------------------------------------------
# Modules (pipeline order). SRC module's vector is split into the two groups the
# module itself computes in separate blocks ("SOURCE-LINK VECTOR" and
# "INFRASTRUCTURE"). The infrastructure key set below is copied verbatim from
# the module's own `else:` default dict in extract_source_link_vector().
# ---------------------------------------------------------------------------
MODULES = [
    ("STS", "feature_modules.sts_features", "extract_sts_features"),
    ("BAS", "feature_modules.bas_features", "extract_bas_features"),
    ("DMS", "feature_modules.dms_features", "extract_dms_features"),
    ("LOS", "feature_modules.los_features", "extract_los_features"),
    ("RCS", "feature_modules.rcs_features", "extract_rcs_features"),
    ("SRCINFRA", "feature_modules.source_link_infra_features", "extract_source_link_vector"),
]
INFRA_VECTOR_KEYS = {
    "subdomain_depth_avg", "subdomain_depth_max", "DNS_TTL_avg", "DNS_TTL_min",
    "nameserver_count_avg", "nameserver_count_max", "unique_ASN_count",
    "unique_hosting_country_count", "certificate_existence_any", "certificate_age_avg",
    "certificate_age_min", "certificate_SAN_count_avg", "certificate_SAN_count_max",
}
FEATURE_GROUPS = ["STS", "BAS", "DMS", "LOS", "RCS", "SRC", "INFRA"]

# Metadata values that are personal or body text and therefore never exported
PII_METADATA = {"sender_name", "claimed_brand"}
# Per-link keys never exported: anchor text = body text; tag = HTML object;
# redirect_chain = full URLs with tokens (its domains are kept as redirect_domains)
LINK_DROP = {"anchor", "anchor_text", "tag", "redirect_chain", "link_number"}
LINK_URL_KEYS = {"url", "final_url", "visible_url"}

UNSUB_URL = re.compile(r"unsub|opt[-_]?out|email[-_]?preferences|manage[-_]?preferences", re.I)
UNSUB_TEXT = re.compile(r"unsubscribe|opt[\s-]?out|manage (?:your |email )?preferences|stop receiving", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


# ===========================================================================
# Lookup observer
# ===========================================================================
class LookupObserver:
    """Wraps network primitives and selected module helper functions.

    Every wrapper returns/raises exactly what the original returns/raises; it
    only appends an event {module, kind, target(domain), status}.
    """

    def __init__(self, block_unsubscribe: bool):
        self.block_unsubscribe = block_unsubscribe
        self.blocked_urls: set[str] = set()
        self.events: list[dict] = []
        self.module = None
        self.local = threading.local()
        self.dns_available = True

    def _ctx(self):
        return getattr(self.local, "ctx", None)

    def add(self, kind, target, status, **kw):
        self.events.append({"module": self.module, "kind": kind, "target": target, "status": status, **kw})

    @staticmethod
    def host(url):
        try:
            return (urlsplit(str(url)).hostname or "").lower()
        except ValueError:
            return ""

    def install(self, srcinfra_module):
        obs = self
        orig_get, orig_ghbn = requests.get, socket.gethostbyname
        orig_cc, orig_wrap = socket.create_connection, ssl.SSLContext.wrap_socket

        def get(url, *a, **kw):
            h = obs.host(url)
            kind = "rdap" if h == "rdap.org" else "ipinfo" if h == "ipinfo.io" else "redirect"
            target = str(url).rsplit("/", 1)[-1] if kind == "rdap" else (
                "ip" if kind == "ipinfo" else h)
            if kind == "redirect" and obs.block_unsubscribe and url in obs.blocked_urls:
                obs.add(kind, target, "skipped_unsubscribe_guard", url=url)
                raise requests.exceptions.ConnectionError("blocked by dataset-builder unsubscribe guard")
            try:
                r = orig_get(url, *a, **kw)
            except Exception as exc:
                obs.add(kind, target, f"failed:{type(exc).__name__}", url=url)
                raise
            obs.add(kind, target, "success" if r.status_code == 200 else f"http_{r.status_code}", url=url)
            return r

        def ghbn(hostname):
            try:
                ip = orig_ghbn(hostname)
            except Exception as exc:
                obs.add("dns_a", hostname, f"failed:{type(exc).__name__}")
                raise
            obs.add("dns_a", hostname, "success")
            return ip

        def cc(address, *a, **kw):
            try:
                return orig_cc(address, *a, **kw)
            except Exception as exc:
                if obs._ctx() == "tls":
                    obs.local.tls_reason = f"connect:{type(exc).__name__}"
                raise

        def wrap(self_ctx, sock, *a, **kw):
            try:
                return orig_wrap(self_ctx, sock, *a, **kw)
            except Exception as exc:
                if obs._ctx() == "tls":
                    obs.local.tls_reason = f"handshake:{type(exc).__name__}"
                raise

        requests.get = get
        socket.gethostbyname = ghbn
        socket.create_connection = cc
        ssl.SSLContext.wrap_socket = wrap

        # --- result observation of the 6th module's lookup helpers ----------
        m = srcinfra_module
        o_ttl, o_ns, o_cert = m.get_dns_ttl, m.get_nameserver_info, m.get_certificate_info

        def ttl(domain):
            v = o_ttl(domain)
            obs.add("dns_ttl", domain, "success" if v is not None else
                    ("failed:dnspython_not_installed" if not obs.dns_available else "failed:no_A_answer"))
            return v

        def ns(domain):
            v = o_ns(domain)
            obs.add("dns_ns", domain, "success" if v.get("nameserver_count") else
                    ("failed:dnspython_not_installed" if not obs.dns_available else "failed:no_NS_answer"))
            return v

        def cert(domain):
            obs.local.ctx, obs.local.tls_reason = "tls", None
            try:
                v = o_cert(domain)
            finally:
                obs.local.ctx = None
            obs.add("tls", domain, "success" if v.get("certificate_existence") else
                    f"failed:{obs.local.tls_reason or 'no_certificate'}")
            return v

        m.get_dns_ttl, m.get_nameserver_info, m.get_certificate_info = ttl, ns, cert
        try:
            import dns.resolver  # noqa: F401
        except ImportError:
            self.dns_available = False

    def reset(self, blocked=()):
        self.events, self.blocked_urls = [], set(blocked)


# ===========================================================================
# helpers
# ===========================================================================
def parse(raw: bytes):
    return BytesParser(policy=policy.default).parsebytes(raw)


def message_id_hash(msg):
    mid = (msg.get("Message-ID") or "").strip().strip("<>").lower()
    return hashlib.sha256(mid.encode()).hexdigest() if mid else None


def private_strings(msg) -> dict:
    out = {"addresses": set(), "names": set(), "subject": None, "body": []}
    for hdr in ("From", "To", "Cc", "Bcc", "Delivered-To", "Reply-To", "Return-Path", "Sender"):
        try:
            for name, addr in getaddresses([str(v) for v in (msg.get_all(hdr) or [])]):
                if addr:
                    out["addresses"].add(addr.lower())
                if name and len(name.strip()) >= 4:
                    out["names"].add(name.strip())
        except Exception:
            pass
    try:
        out["subject"] = str(msg.get("Subject") or "").strip() or None
        b = msg.get_body(preferencelist=("plain",))
        if b is not None:
            out["body"] = [ln.strip() for ln in b.get_content().splitlines() if len(ln.strip()) >= 30][:5]
    except Exception:
        pass
    return out


def recipient_tokens(msg) -> set[str]:
    toks = set()
    for hdr in ("To", "Cc", "Delivered-To"):
        try:
            for _, addr in getaddresses([str(v) for v in (msg.get_all(hdr) or [])]):
                if "@" in addr and len(addr.split("@")[0]) >= 4:
                    toks.add(addr.split("@")[0].lower())
        except Exception:
            pass
    return toks


def redact_url(url, rtoks, full):
    if not url or not isinstance(url, str):
        return url
    if not full:
        try:
            p = urlsplit(url)
            url = urlunsplit((p.scheme, p.netloc, p.path, "", ""))
        except ValueError:
            pass
    url = EMAIL_RE.sub("[email]", url)
    for t in rtoks:
        url = re.sub(re.escape(t), "[recipient]", url, flags=re.I)
    return url


def unsubscribe_urls(msg, bas_mod) -> set[str]:
    urls = set()
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(bas_mod.get_html(msg), "html.parser")
        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if not href.lower().startswith(("http://", "https://")):
                continue
            prev = a.find_previous(string=True)
            ctx = str(prev)[-160:] if prev else ""
            if UNSUB_URL.search(href) or UNSUB_TEXT.search(a.get_text(" ", strip=True)) or UNSUB_TEXT.search(ctx):
                urls.add(href)
    except Exception:
        pass
    return urls


def cell(v):
    if isinstance(v, (list, tuple, set)):
        return "|".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, default=str)
    return v


# ===========================================================================
# per email
# ===========================================================================
def run_modules(path: Path, modules: dict, observer: LookupObserver):
    """Return {tag: (output, error)} with each module's untouched return value."""
    out = {}
    for tag, (mod, fn) in modules.items():
        observer.module = tag
        try:
            out[tag] = (getattr(mod, fn)(str(path)), None)
        except Exception as exc:
            out[tag] = (None, exc)
    return out


def process_one(path, email_id, label, modules, observer, args):
    msg = parse(path.read_bytes())
    rtoks = recipient_tokens(msg)
    observer.reset(unsubscribe_urls(msg, modules["BAS"][0]) if args.block_unsubscribe else ())
    outputs = run_modules(path, modules, observer)

    row = {"email_id": email_id, "label": label}
    errors, failed_modules = [], []
    links: dict[int, dict] = {}
    link_urls: dict[int, dict] = {}

    def link_row(n):
        return links.setdefault(n, {"email_id": email_id, "link_number": n})

    def put_link(tag, n, key, val, url_of_link=None):
        if key in LINK_DROP:
            return
        if key in LINK_URL_KEYS:
            if key == "url":
                link_urls.setdefault(n, {})[tag] = val
            val = redact_url(val, rtoks, args.full_urls)
        link_row(n)[f"{tag}__{key}"] = cell(val)

    for tag, (res, exc) in outputs.items():
        if exc is not None:
            failed_modules.append(tag)
            errors.append({"email_id": email_id, "error_type": type(exc).__name__,
                           "error_message": str(exc)[:500], "failed_module": tag,
                           "failed_feature": "(entire module)"})
            continue
        if tag == "SRCINFRA":
            for k, v in res["vector"].items():
                grp = "INFRA" if k in INFRA_VECTOR_KEYS else "SRC"
                row[f"{grp}__{k}"] = cell(v)
            row["SRCINFRA_meta__number_of_links"] = res.get("number_of_links")
            # per-link: url features (one per link) and infrastructure (one per link WITH hostname,
            # in the same order - mirrors the module's own loop)
            infra = list(res.get("infrastructure", []))
            infra_i = 0
            for n, pl in enumerate(res.get("per_link", []), start=1):
                for k, v in pl.items():
                    put_link("SRC", n, k, v)
                try:
                    has_host = bool(urlsplit(pl.get("url", "")).hostname)
                except ValueError:
                    has_host = False
                if has_host and infra_i < len(infra):
                    for k, v in infra[infra_i].items():
                        put_link("INFRA", n, k, v)
                    infra_i += 1
            continue
        features, metadata = res
        for k, v in features.items():
            row[f"{tag}__{k}"] = cell(v)
        for k, v in metadata.items():
            if k == "link_results" or k in PII_METADATA:
                continue
            row[f"{tag}_meta__{k}"] = cell(v)
        for i, lr in enumerate(metadata.get("link_results", []) or [], start=1):
            n = lr.get("link_number", i)
            for k, v in lr.items():
                put_link(tag, n, k, v)

    # ---- Link_Details common columns + per-link lookup status ---------------
    ev = observer.events
    for n, r in links.items():
        urls = {u for u in link_urls.get(n, {}).values() if u}
        url = sorted(urls)[0] if urls else None
        host = LookupObserver.host(url) if url else ""
        r["URL"] = redact_url(url, rtoks, args.full_urls)
        r["destination"] = r.get("DMS__destination_domain", r.get("BAS__destination_domain"))
        r["final_redirect_domain"] = r.get("DMS__final_redirect_domain", r.get("BAS__final_redirect_domain"))
        r["url_consistent_across_modules"] = len(urls) <= 1
        red = [e for e in ev if e["kind"] == "redirect" and e.get("url") in urls]
        r["redirect_lookup_status"] = "|".join(sorted({f"{e['module']}:{e['status']}" for e in red})) or "not_attempted"
        for kind in ("dns_ttl", "dns_ns", "tls"):
            st = [e["status"] for e in ev if e["kind"] == kind and e["target"] == host]
            r[f"INFRA_status__{kind}"] = st[0] if st else "not_attempted"
        if not r["url_consistent_across_modules"]:
            errors.append({"email_id": email_id, "error_type": "LinkAlignmentWarning",
                           "error_message": f"modules extracted different URLs at link {n} "
                                            "(BAS reads all HTML parts, the others the first HTML part)",
                           "failed_module": "link_merge", "failed_feature": "link_number"})

    # ---- lookup failures -> status columns + Processing_Errors --------------
    fails = [e for e in ev if e["status"] != "success"]
    row["lookups_attempted"] = len(ev)
    row["lookups_failed"] = len(fails)
    for kind in ("rdap", "ipinfo", "redirect", "dns_a", "dns_ttl", "dns_ns", "tls"):
        row[f"lookups_failed_{kind}"] = sum(1 for e in fails if e["kind"] == kind)
    row["missing_external_lookup"] = bool(fails)
    grouped = Counter((e["module"], e["kind"], e["status"]) for e in fails)
    examples = {}
    for e in fails:
        examples.setdefault((e["module"], e["kind"], e["status"]), e["target"])
    affects = {"rdap": "domain age (STS new_sender_domain / RCS registration age)",
               "ipinfo": "ASN / country", "redirect": "final redirect domain / redirect chain",
               "dns_a": "final IP / ASN", "dns_ttl": "DNS_TTL", "dns_ns": "nameserver_count",
               "tls": "certificate_existence / age / SAN"}
    for (mod, kind, status), cnt in sorted(grouped.items()):
        errors.append({"email_id": email_id, "error_type": "ExternalLookupFailed",
                       "error_message": f"{kind} {status} x{cnt} (e.g. {examples[(mod, kind, status)]})",
                       "failed_module": mod, "failed_feature": affects[kind]})

    if not failed_modules:
        row["processing_status"] = "ok"
    elif len(failed_modules) == len(modules):
        row["processing_status"] = "failed"
    else:
        row["processing_status"] = "partial:" + ",".join(failed_modules)
    return row, list(links.values()), errors, private_strings(msg)


# ===========================================================================
# Feature inventory (built from what the modules actually returned)
# ===========================================================================
def dtype_of(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    types = {type(v).__name__ for v in vals}
    nullable = len(vals) < len(values)
    if not types:
        t = "unknown (always null in this run)"
    elif types <= {"bool"}:
        t = "boolean"
    elif types <= {"int"}:
        t = "integer"
    elif types <= {"int", "float"}:
        t = "float"
    elif types <= {"str"}:
        t = "string"
    else:
        t = "mixed:" + "/".join(sorted(types))
    return t + (" (nullable)" if nullable else "")


def build_inventory(rows, link_rows):
    inv = []
    skip_email = {"email_id", "label", "processing_status"}
    for sheet, table, level in (("Dataset", rows, "email"), ("Link_Details", link_rows, "link")):
        cols = []
        for r in table:
            for c in r:
                if c not in cols:
                    cols.append(c)
        for c in cols:
            if "__" not in c or c in skip_email:
                continue
            grp, name = c.split("__", 1)
            if grp.endswith("_status") or grp.startswith("lookups"):
                continue
            kind = "feature" if grp in FEATURE_GROUPS else "metadata (returned by module)"
            module = grp.replace("_meta", "")
            inv.append({"module": module, "feature_name": name, "column": c, "sheet": sheet,
                        "email_level_or_link_level": level, "kind": kind,
                        "data_type": dtype_of([r.get(c) for r in table if c in r])})
    return inv


# ===========================================================================
# privacy verification
# ===========================================================================
def privacy_findings(tables, privs):
    bad = set()
    needles = set()
    for p in privs:
        if p["subject"] and len(p["subject"]) >= 6:
            needles.add(p["subject"].lower())
        needles |= {n.lower() for n in p["names"]}
        needles |= set(p["addresses"])
        needles |= {b.lower() for b in p["body"]}
    for name, table in tables.items():
        for r in table:
            for k, v in r.items():
                if not isinstance(v, str):
                    continue
                lv = v.lower()
                if any(nd and nd in lv for nd in needles):
                    bad.add(f"{name}.{k}: private string")
                if EMAIL_RE.search(v):
                    bad.add(f"{name}.{k}: e-mail address")
    return sorted(bad)


# ===========================================================================
# workbook
# ===========================================================================
def write_workbook(out, rows, link_rows, errors, stats, inventory):
    import pandas as pd
    order = {g: i for i, g in enumerate(["STS", "BAS", "DMS", "LOS", "RCS", "SRC", "INFRA", "SRCINFRA"])}

    def key(c):
        grp = c.split("__")[0]
        base = grp.replace("_meta", "")
        return (order.get(base, 99), 1 if "_meta" in grp else 0, c)

    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    lead = ["email_id", "label", "processing_status"]
    status = [c for c in cols if c.startswith("lookups") or c == "missing_external_lookup"]
    feats = sorted([c for c in cols if c not in lead and c not in status], key=key)
    ds = pd.DataFrame(rows, columns=lead + feats + status)

    lcols = ["email_id", "link_number", "URL", "destination", "final_redirect_domain",
             "redirect_lookup_status", "INFRA_status__dns_ttl", "INFRA_status__dns_ns", "INFRA_status__tls",
             "url_consistent_across_modules"]
    rest = []
    for r in link_rows:
        for c in r:
            if c not in lcols and c not in rest:
                rest.append(c)
    lk = pd.DataFrame(link_rows, columns=lcols + sorted(rest, key=key))
    er = pd.DataFrame(errors, columns=["email_id", "error_type", "error_message", "failed_module", "failed_feature"])
    st = pd.DataFrame(list(stats.items()), columns=["metric", "value"])
    iv = pd.DataFrame(inventory, columns=["module", "feature_name", "data_type", "email_level_or_link_level",
                                          "kind", "sheet", "column"])
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        ds.to_excel(xw, sheet_name="Dataset", index=False)
        lk.to_excel(xw, sheet_name="Link_Details", index=False)
        st.to_excel(xw, sheet_name="Dataset_Stats", index=False)
        er.to_excel(xw, sheet_name="Processing_Errors", index=False)
        iv.to_excel(xw, sheet_name="Feature_Inventory", index=False)
        for name, df in (("Dataset", ds), ("Link_Details", lk), ("Feature_Inventory", iv),
                         ("Dataset_Stats", st), ("Processing_Errors", er)):
            ws = xw.sheets[name]
            ws.freeze_panes = "B2"
            for i, c in enumerate(df.columns, start=1):
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(12, len(str(c)) + 2), 60)
    return ds, lk, iv


# ===========================================================================
# main
# ===========================================================================
_W = {}


def _worker_init(block_unsubscribe, cache_path):
    """Each worker imports the six modules and installs shared lookup caching + observer."""
    modules = {tag: (importlib.import_module(m), fn) for tag, m, fn in MODULES}
    cache = PersistentLookupCache(cache_path)
    _install_persistent_cache(cache)
    observer = LookupObserver(block_unsubscribe)
    observer.install(modules["SRCINFRA"][0])
    _W.update(modules=modules, observer=observer, cache=cache)


def _worker_run(task):
    path, email_id, label, h, full_urls, block = task
    from types import SimpleNamespace
    args = SimpleNamespace(full_urls=full_urls, block_unsubscribe=block)
    try:
        row, lrows, errs, priv = process_one(Path(path), email_id, label, _W["modules"], _W["observer"], args)
        priv = {k: (sorted(v) if isinstance(v, set) else v) for k, v in priv.items()}
    except Exception as exc:  # never stop the batch
        row = {"email_id": email_id, "label": label, "processing_status": "failed"}
        lrows, priv = [], None
        errs = [{"email_id": email_id, "error_type": type(exc).__name__, "error_message": str(exc)[:500],
                 "failed_module": "pipeline", "failed_feature": traceback.format_exc().splitlines()[-1][:200]}]
    return {"email_id": email_id, "path": path, "sha256": h, "row": row, "links": lrows, "errors": errs,
            "priv": priv}


def collect(d, label, prefix):
    return [(p, label, prefix) for p in sorted(Path(d).rglob("*.eml")) if p.is_file()] if d else []


def _github_email_urls(repo="rf-peixoto/phishing_pot", branch="main"):
    """Return .eml blob URLs from the repository's email/ directory.

    Only the GitHub tree metadata is fetched up front. Individual .eml files
    are downloaded one at a time during processing.
    """
    import requests as _requests

    api = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    r = _requests.get(api, timeout=60)
    r.raise_for_status()
    data = r.json()

    if data.get("truncated"):
        raise RuntimeError(
            "GitHub returned a truncated repository tree. "
            "Use a repository snapshot/clone or a paginated source instead."
        )

    urls = []
    prefix = "email/"
    for item in data.get("tree", []):
        path = item.get("path", "")
        if item.get("type") == "blob" and path.startswith(prefix) and path.lower().endswith(".eml"):
            urls.append(
                f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
            )
    return sorted(urls)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stream rf-peixoto/phishing_pot .eml files one at a time "
                    "through the exact existing six-module feature pipeline."
    )
    ap.add_argument("--out", default="phishing_pot_dataset.xlsx")
    ap.add_argument("--limit", type=int, help="first N unique phishing emails (test batch)")
    ap.add_argument("--full-urls", action="store_true",
                    help="keep query strings in URL columns")
    ap.add_argument("--allow-unsubscribe-requests", dest="block_unsubscribe",
                    action="store_false")
    ap.add_argument("--id-map", default="private/phishing_pot_id_map.csv",
                    help="PRIVATE email_id -> GitHub URL map")
    ap.add_argument("--cache", default="private/lookup_cache.sqlite",
                    help="persistent shared cache for external lookups (delete to refresh)")
    ap.add_argument("--checkpoint", default="private/phishing_pot_checkpoint.jsonl",
                    help="resume file; '' to disable. Delete it to start over")
    ap.add_argument("--repo", default="rf-peixoto/phishing_pot")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--temp-dir", default="private/phishing_pot_tmp",
                    help="temporary directory; at most one .eml is kept at a time")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log.info("persistent lookup cache: %s", Path(args.cache).resolve())

    try:
        import dns.resolver  # noqa: F401
        dns_available = True
    except ImportError:
        dns_available = False

    import os
    import tempfile
    import requests as _requests

    # IMPORTANT: use the exact same modules and lookup observer as the uploaded
    # builder. Nothing in feature_modules/ is modified.
    modules = {tag: (importlib.import_module(m), fn) for tag, m, fn in MODULES}
    cache = PersistentLookupCache(args.cache)
    _install_persistent_cache(cache)
    observer = LookupObserver(args.block_unsubscribe)
    observer.install(modules["SRCINFRA"][0])

    sources = _github_email_urls(args.repo, args.branch)
    log.info("%d .eml files found in %s/%s", len(sources), args.repo, "email")

    pipeline_errors = []
    if not dns_available:
        pipeline_errors.append({
            "email_id": "*",
            "error_type": "DependencyMissing",
            "error_message": "dnspython not installed: the module's get_dns_ttl / "
                             "get_nameserver_info cannot succeed (pip install dnspython)",
            "failed_module": "SRCINFRA",
            "failed_feature": "DNS_TTL, nameserver_count"
        })

    if any(os.environ.get(v) for v in
           ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")):
        pipeline_errors.append({
            "email_id": "*",
            "error_type": "EnvironmentWarning",
            "error_message": "an HTTP(S) proxy is configured: the modules' requests "
                             "(redirects, RDAP, ipinfo) go through it and TLS may be "
                             "intercepted - network-based feature values may not describe "
                             "the real sites",
            "failed_module": "*",
            "failed_feature": "all lookup-based features"
        })

    # Resume by source URL. This is separate from the normal local .eml builder's
    # checkpoint because GitHub URLs, not local file paths, identify these emails.
    done = {}
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    if ckpt and ckpt.exists():
        with open(ckpt, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done[rec["source_url"]] = rec
                except (ValueError, KeyError):
                    continue
        log.info("checkpoint: %d emails already processed, resuming", len(done))

    errors = list(pipeline_errors)
    rows, link_rows, privs, idmap = [], [], [], []
    seen_raw, seen_mid = set(), set()
    duplicates = 0
    processed = 0
    skipped_checkpoint = 0
    cross = 0
    t_start = time.time()

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    ck_fh = open(ckpt, "a", encoding="utf-8") if ckpt else None

    try:
        for source_url in sources:
            if source_url in done:
                skipped_checkpoint += 1
                rec = done[source_url]
                # Rehydrate previously completed rows so the final workbook can
                # be regenerated after a resume.
                if rec.get("row") is not None:
                    rows.append(rec["row"])
                link_rows.extend(rec.get("links", []))
                errors.extend(rec.get("errors", []))
                if rec.get("priv"):
                    privs.append(rec["priv"])
                if rec.get("sha256"):
                    seen_raw.add(rec["sha256"])
                if rec.get("message_id_hash"):
                    seen_mid.add(rec["message_id_hash"])
                idmap.append((
                    rec.get("email_id"),
                    source_url,
                    rec.get("sha256", "")
                ))
                continue

            if args.limit is not None and processed >= args.limit:
                break

            email_id = f"phish_{processed + 1:06d}"
            tmp_path = None
            rec = None

            try:
                # Download exactly ONE email.
                rr = _requests.get(source_url, timeout=60)
                rr.raise_for_status()
                raw = bytes(rr.content)

                h = hashlib.sha256(raw).hexdigest()
                msg = parse(raw)
                mh = message_id_hash(msg)

                # Same duplicate logic as the local builder:
                # raw SHA-256 OR Message-ID.
                if h in seen_raw or (mh and mh in seen_mid):
                    duplicates += 1
                    rec = {
                        "source_url": source_url,
                        "email_id": email_id,
                        "sha256": h,
                        "message_id_hash": mh,
                        "duplicate": True,
                        "row": None,
                        "links": [],
                        "errors": [],
                        "priv": None,
                    }
                    if ck_fh:
                        ck_fh.write(json.dumps(rec, default=str) + "\n")
                        ck_fh.flush()
                    continue

                seen_raw.add(h)
                if mh:
                    seen_mid.add(mh)

                # The existing feature modules require a file path. Create only
                # one temporary .eml, run the exact modules, then delete it.
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".eml", prefix="phish_", dir=temp_dir,
                    delete=False
                ) as tf:
                    tf.write(raw)
                    tmp_path = Path(tf.name)

                args_obj = argparse.Namespace(
                    full_urls=args.full_urls,
                    block_unsubscribe=args.block_unsubscribe
                )

                row, lrows, errs, priv = process_one(
                    tmp_path, email_id, 1, modules, observer, args_obj
                )

                rows.append(row)
                link_rows.extend(lrows)
                errors.extend(errs)
                if priv:
                    privs.append(priv)
                idmap.append((email_id, source_url, h))

                rec = {
                    "source_url": source_url,
                    "email_id": email_id,
                    "sha256": h,
                    "message_id_hash": mh,
                    "duplicate": False,
                    "row": row,
                    "links": lrows,
                    "errors": errs,
                    "priv": priv,
                }

                processed += 1

                if processed % 25 == 0 or processed == args.limit:
                    elapsed = time.time() - t_start
                    rate = elapsed / processed if processed else 0
                    remaining = max(0, len(sources) - processed - skipped_checkpoint)
                    eta = rate * remaining
                    log.info(
                        "%d phishing emails processed (~%.0f min elapsed, ~%.0f min left)",
                        processed, elapsed / 60, eta / 60
                    )

            except Exception as exc:
                # Download/parse/pipeline errors do not stop the batch.
                errs = [{
                    "email_id": email_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                    "failed_module": "download/pipeline",
                    "failed_feature": "-"
                }]
                errors.extend(errs)

                rec = {
                    "source_url": source_url,
                    "email_id": email_id,
                    "sha256": None,
                    "message_id_hash": None,
                    "duplicate": False,
                    "row": {
                        "email_id": email_id,
                        "label": 1,
                        "processing_status": "failed"
                    },
                    "links": [],
                    "errors": errs,
                    "priv": None,
                }
                rows.append(rec["row"])
                idmap.append((email_id, source_url, ""))

            finally:
                # Never retain the email corpus locally.
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            if ck_fh and rec is not None:
                ck_fh.write(json.dumps(rec, default=str) + "\n")
                ck_fh.flush()

    finally:
        if ck_fh:
            ck_fh.close()
        cache.close()

    # Same final privacy check, inventory, stats and workbook format.
    bad = privacy_findings({"Dataset": rows, "Link_Details": link_rows}, privs)
    for b in bad:
        errors.append({
            "email_id": "*",
            "error_type": "PrivacyCheck",
            "error_message": b,
            "failed_module": "privacy_filter",
            "failed_feature": "-"
        })

    inventory = build_inventory(rows, link_rows)
    per = Counter(i["module"] for i in inventory
                  if i["kind"] == "feature" and i["sheet"] == "Dataset")
    per_link = Counter(i["module"] for i in inventory
                       if i["kind"] == "feature" and i["sheet"] == "Link_Details")
    n_feat_cols = sum(per.values())

    stats = {
        "source": f"https://github.com/{args.repo}/tree/{args.branch}/email",
        "total emails (unique, in Dataset)": len(rows),
        "legitimate emails (label 0)": 0,
        "phishing emails (label 1)": sum(r.get("label") == 1 for r in rows),
        "duplicate emails removed": duplicates,
        "of which cross-source duplicates (first copy kept)": cross,
        "checkpoint records reused": skipped_checkpoint,
        "successfully processed (all 6 modules ran)": sum(
            r.get("processing_status") == "ok" for r in rows
        ),
        "failed emails (no module ran)": sum(
            r.get("processing_status") == "failed" for r in rows
        ),
        "partially processed (some module failed)": sum(
            str(r.get("processing_status", "")).startswith("partial")
            for r in rows
        ),
        "emails with missing external lookups": sum(
            bool(r.get("missing_external_lookup")) for r in rows
        ),
        "total links (Link_Details rows)": len(link_rows),
        "total feature columns (Dataset, module features)": n_feat_cols,
        "total metadata columns (Dataset, module metadata)": sum(
            1 for i in inventory
            if i["sheet"] == "Dataset" and i["kind"] != "feature"
        ),
        "total columns (Dataset)": len({c for r in rows for c in r}),
    }

    for g in FEATURE_GROUPS:
        stats[f"{g} email-level features"] = per.get(g, 0)
    for g in ["BAS", "DMS", "LOS", "RCS", "SRC", "INFRA"]:
        stats[f"{g} link-level features"] = per_link.get(g, 0)

    stats.update({
        "privacy check": "PASS" if not bad else f"FAIL ({len(bad)} findings, see Processing_Errors)",
        "unsubscribe guard": "on" if args.block_unsubscribe else "off",
        "URL redaction": "full URLs" if args.full_urls
                         else "query string/fragment removed, recipients masked",
        "dnspython available": dns_available,
        "workers": 1,
        "download mode": "streaming; one temporary .eml at a time",
        "runtime (s)": round(time.time() - t_start, 1),
        "generated at (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_workbook(args.out, rows, link_rows, errors, stats, inventory)

    Path(args.id_map).parent.mkdir(parents=True, exist_ok=True)
    with open(args.id_map, "w", encoding="utf-8") as fh:
        fh.write("email_id,source_file,sha256\n")
        for e, p, h in idmap:
            fh.write(f"{e},{p},{h}\n")

    print(json.dumps(stats, indent=1, default=str))
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
