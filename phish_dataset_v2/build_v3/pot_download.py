"""Download every phishing_pot .eml (raw bytes, unchanged) into one cache dir; resumable."""
import sys, json, time, pathlib, urllib.request, concurrent.futures as cf
tree, out = sys.argv[1], pathlib.Path(sys.argv[2])
paths = sorted(x["path"] for x in json.load(open(tree))["tree"] if x["type"] == "blob" and x["path"].startswith("email/") and x["path"].lower().endswith(".eml"))
def get(p):
    dst = out / p.split("/", 1)[1]
    if dst.exists() and dst.stat().st_size > 0:
        return 1
    for a in range(6):
        try:
            b = urllib.request.urlopen("https://raw.githubusercontent.com/rf-peixoto/phishing_pot/main/" + p, timeout=60).read()
            tmp = dst.with_suffix(".part"); tmp.write_bytes(b); tmp.replace(dst)
            return 1
        except Exception:
            time.sleep(2 ** a)
    print("FAILED", p, flush=True); return 0
ok = 0
with cf.ThreadPoolExecutor(24) as ex:
    for n, r in enumerate(ex.map(get, paths), 1):
        ok += r
        if n % 2000 == 0: print("progress", n, flush=True)
print("DONE", ok, "of", len(paths), flush=True)
