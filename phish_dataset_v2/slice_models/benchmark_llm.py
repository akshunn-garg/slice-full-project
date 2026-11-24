"""
LLM-classifier baseline (local Ollama models) on the same test / Nazario emails as SLiCE.

    python -m slice_models.benchmark_llm --model qwen2.5:1.5b --n-test 300 --n-nazario 150
    python -m slice_models.benchmark_llm --model qwen2.5:7b   --n-test 100 --n-nazario 50

Prompt in the style of ChatSpamDetector (Koide et al. 2024): key headers (From, Reply-To,
Return-Path, Subject, Date, the canonical Authentication-Results), up to 15 link URLs and the
body text (HTML converted to text, first 2,500 characters) of the same normalised .eml files
the SLiCE features were built from. The model answers JSON {"phishing_score": 0-10,
"is_phishing": true|false}; temperature 0. Wall-clock time per call is recorded.
Results are cached per email (reports/benchmark/llm_<model>.jsonl), so runs can resume.
The sample is stratified and seeded; SLiCE is scored on exactly the same emails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from email import policy
from email.parser import BytesParser
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slice_models.slice_final_predict import predict_final

ROOT = Path(__file__).resolve().parent.parent
PROMPT = """You are an email security analyst. Decide whether the following email is a phishing email
(an attempt to steal credentials, money or information, or to deliver malware) or a legitimate email.
Examine the headers (sender, reply path, authentication results), the links and the text.

Answer ONLY with JSON: {{"phishing_score": <integer 0-10>, "is_phishing": <true or false>}}

HEADERS
{headers}

LINKS
{links}

BODY
{body}
"""


def email_view(path: Path) -> tuple[str, str, str]:
    msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    headers = "\n".join(f"{h}: {str(msg.get(h, ''))[:300]}" for h in
                        ("From", "Reply-To", "Return-Path", "Subject", "Date", "Authentication-Results"))
    html, text = "", ""
    for part in msg.walk():
        if part.is_multipart():
            continue
        try:
            content = part.get_content()
        except Exception:
            continue
        if part.get_content_type() == "text/html" and isinstance(content, str):
            html += content
        elif part.get_content_type() == "text/plain" and isinstance(content, str):
            text += content
    links = []
    if html:
        soup = BeautifulSoup(html, "html.parser")
        links = [a.get("href") for a in soup.find_all("a") if a.get("href", "").startswith("http")]
        text = text or soup.get_text(" ", strip=True)
    body = re.sub(r"\s+", " ", text)[:2500]
    return headers, "\n".join(links[:15]) or "(none)", body or "(empty)"


def ask(model: str, prompt: str, think: bool | None = None) -> tuple[dict, float]:
    payload = {"model": model, "prompt": prompt, "format": "json", "stream": False,
                       "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 60}}
    if think is not None:
        payload["think"] = think          # Ollama: disable/enable reasoning for thinking models
    data = json.dumps(payload).encode()
    t = time.perf_counter()
    resp = urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate", data=data,
                                                         headers={"Content-Type": "application/json"}), timeout=300)
    out = json.loads(resp.read())
    elapsed = time.perf_counter() - t
    try:
        ans = json.loads(out.get("response", "{}"))
    except json.JSONDecodeError:
        ans = {}
    return ans, elapsed


def sample(n_test: int, n_naz: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.read_excel(ROOT / "slice_dataset_v4.xlsx", sheet_name="Dataset")
    idm = pd.read_csv(ROOT / "private" / "v4" / "id_map_v4.csv")
    idm["path"] = idm["source_file"].str.replace("\\", "/", regex=False)
    te = d[d.prov__time_split == "test"].merge(idm[["email_id", "path"]], on="email_id")
    parts = [te[te.label == lab].iloc[rng.choice(int((te.label == lab).sum()), n_test // 2, replace=False)]
             for lab in (0, 1)]
    s = pd.concat(parts).assign(set="test")
    nz = pd.DataFrame([{**json.loads(l)["row"], "path": json.loads(l)["path"].replace("\\", "/")}
                       for l in open(ROOT / "private" / "nazario" / "checkpoint_nazario.jsonl", encoding="utf-8")])
    nz = nz[(nz.processing_status == "ok") & (pd.to_numeric(nz.BAS_meta__number_of_links, errors="coerce") > 0)]
    nz = nz.iloc[rng.choice(len(nz), n_naz, replace=False)].assign(set="nazario", label=1)
    return pd.concat([s, nz], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-nazario", type=int, default=150)
    ap.add_argument("--no-think", action="store_true",
                    help="turn off the reasoning (thinking) mode of models that have one, e.g. qwen3.5")
    a = ap.parse_args()
    s = sample(a.n_test, a.n_nazario)
    rep = ROOT / "reports" / "benchmark"
    rep.mkdir(parents=True, exist_ok=True)
    cache_path = rep / f"llm_{a.model.replace(':', '_')}.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {r["email_id"]: r for r in map(json.loads, open(cache_path, encoding="utf-8"))}
    with open(cache_path, "a", encoding="utf-8") as fh:
        for i, r in s.iterrows():
            if r.email_id in cache:
                continue
            headers, links, body = email_view(ROOT / r.path)
            ans, sec = {}, np.nan
            for attempt in range(2):                     # a hung call is retried once, then recorded as no answer
                try:
                    ans, sec = ask(a.model, PROMPT.format(headers=headers, links=links, body=body), False if a.no_think else None)
                    break
                except Exception as exc:
                    print(f"{r.email_id}: attempt {attempt + 1} failed ({type(exc).__name__})", flush=True)
            rec = {"email_id": r.email_id, "set": r.set, "label": int(r.label), "seconds": sec,
                   "score": ans.get("phishing_score"), "is_phishing": ans.get("is_phishing")}
            cache[r.email_id] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if len(cache) % 25 == 0:
                print(f"{len(cache)}/{len(s)} done, last call {sec:.1f}s", flush=True)

    llm = pd.DataFrame([cache[e] for e in s.email_id])
    llm["score"] = pd.to_numeric(llm["score"], errors="coerce")
    llm["pred"] = llm["is_phishing"].map(lambda v: 1 if v in (True, "true", "True", 1) else 0)
    sl = predict_final(s)
    import joblib
    thr = joblib.load(ROOT / "models" / "final3" / "slice_final_model.joblib")["threshold"]
    s["slice_prob"], s["slice_pred"] = sl.FINAL_prob.to_numpy(), (sl.FINAL_prob.to_numpy() >= thr).astype(int)

    from sklearn.metrics import roc_auc_score
    out = []
    for name, pred, score in (("LLM " + a.model, llm.pred.to_numpy(), llm.score.fillna(5).to_numpy()),
                              ("SLiCE final3", s.slice_pred.to_numpy(), s.slice_prob.to_numpy())):
        t = (s.set == "test").to_numpy()
        y = s.label.to_numpy()
        tp, fp = ((pred == 1) & (y == 1) & t).sum(), ((pred == 1) & (y == 0) & t).sum()
        fn, tn = ((pred == 0) & (y == 1) & t).sum(), ((pred == 0) & (y == 0) & t).sum()
        out.append({"model": name, "test_n": int(t.sum()), "test_accuracy": (tp + tn) / t.sum(),
                    "test_precision": tp / max(tp + fp, 1), "test_recall": tp / max(tp + fn, 1),
                    "test_fpr": fp / max(fp + tn, 1), "test_roc_auc": roc_auc_score(y[t], score[t]),
                    "nazario_recall": pred[~t].mean()})
    res = pd.DataFrame(out)
    res["llm_seconds_per_email_median"] = [llm.seconds.median(), np.nan]
    res["unparsed_answers"] = [int(llm.score.isna().sum()), 0]
    res.to_csv(rep / f"llm_vs_slice_{a.model.replace(':', '_')}.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
