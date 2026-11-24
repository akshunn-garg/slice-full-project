"""Lookup-failure rates per email in a builder checkpoint (to catch network saturation).

Usage: python build_v4/lookup_health.py private/v4/checkpoint_v4.jsonl
A healthy legit baseline (16 workers, first 605 emails): ~0.5 redirect ConnectionErrors per email.
"""
import collections
import json
import re
import sys

st = {}
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    key = "legit" if str(r.get("email_id", "")).startswith("email") else "phish"
    n_c = st.setdefault(key, [0, collections.Counter()])
    n_c[0] += 1
    for e in r.get("errors") or []:
        if e.get("error_type") == "ExternalLookupFailed":
            m = re.match(r"(\w+) (\S+) x(\d+)", str(e.get("error_message")))
            if m:
                n_c[1][f"{m.group(1)} {m.group(2)}"] += int(m.group(3))
for k, (n, c) in sorted(st.items()):
    print(k, "emails", n, "| per email:",
          {x: round(c[x] / n, 2) for x in ("redirect failed:ConnectionError", "redirect failed:ConnectTimeout",
                                           "rdap failed:ConnectionError", "dns_a failed:gaierror")})
