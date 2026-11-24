#!/usr/bin/env python3
"""
Download Gmail INBOX messages as raw RFC-822 .eml files (READ-ONLY scope).

Setup (once):
  1. https://console.cloud.google.com -> create project -> enable "Gmail API"
  2. OAuth consent screen: External, add yourself as test user
  3. Credentials -> Create OAuth client ID -> "Desktop app" -> download JSON
     and save it as private/credentials.json
  4. pip install google-api-python-client google-auth-oauthlib

Run:
  python gmail_fetch.py --out private/inbox_eml            # whole Inbox
  python gmail_fetch.py --out private/inbox_eml --limit 5  # test batch
  python gmail_fetch.py --out private/spam_eml --label SPAM # Spam folder (kept 30 days by Gmail)

Files are written as <gmail_message_id>.eml. Already-downloaded messages are
skipped, so the command can be re-run to resume. Keep private/ out of git.
"""
from __future__ import annotations

import argparse
import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def service(cred_path: Path, token_path: Path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES).run_console()
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def list_inbox_ids(svc, limit=None, label="INBOX"):
    ids, token = [], None
    while True:
        resp = svc.users().messages().list(userId="me", labelIds=[label], maxResults=500,
                                           pageToken=token).execute()
        ids += [m["id"] for m in resp.get("messages", [])]
        token = resp.get("nextPageToken")
        if not token or (limit and len(ids) >= limit):
            return ids[:limit] if limit else ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="private/inbox_eml")
    ap.add_argument("--credentials", default="private/credentials.json")
    ap.add_argument("--token", default="private/token.json")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--label", default="INBOX", help="Gmail label to fetch, e.g. INBOX or SPAM")
    ap.add_argument("--threads", type=int, default=1, help="parallel downloads (each thread has its own API client)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    svc = service(Path(a.credentials), Path(a.token))
    ids = list_inbox_ids(svc, a.limit, a.label)
    print(f"{len(ids)} {a.label} messages")
    todo = [mid for mid in ids if not (out / f"{mid}.eml").exists()]
    print(f"{len(ids) - len(todo)} already downloaded, {len(todo)} to fetch")

    local = threading.local()

    def fetch(mid):
        if not hasattr(local, "svc"):
            local.svc = svc if a.threads == 1 else service(Path(a.credentials), Path(a.token))
        dest = out / f"{mid}.eml"
        for attempt in range(5):
            try:
                msg = local.svc.users().messages().get(userId="me", id=mid, format="raw").execute()
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(base64.urlsafe_b64decode(msg["raw"].encode("ascii")))
                tmp.replace(dest)
                return True
            except Exception as exc:  # rate limit / transient: back off and retry
                if attempt == 4:
                    print(f"FAILED {mid}: {type(exc).__name__}: {exc}")
                    return False
                time.sleep(2 ** attempt)

    done = failed = 0
    with ThreadPoolExecutor(max(1, a.threads)) as ex:
        for i, ok in enumerate(ex.map(fetch, todo), 1):
            done += ok
            failed += not ok
            if i % 500 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print(f"downloaded {done}, failed {failed}, folder {out}")


if __name__ == "__main__":
    main()
