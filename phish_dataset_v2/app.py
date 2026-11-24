"""
SLiCE web app: upload a .eml file and see everything the pipeline computes.

    python app.py            # opens http://127.0.0.1:5000 in your browser

Pipeline per upload (identical to how the training data and final models were built):
  1. normalise  - build_v4/normalize.py (receiver hops stripped, one canonical
                  Authentication-Results header, recipient addresses replaced)
  2. features   - the six unmodified feature modules via build_dataset_fast's worker
                  (live DNS / RDAP / TLS / redirect lookups, shared lookup cache)
  3. models     - models/final3: BAS, DMS, LOS, RCS -> link model (+ INFRA), rule-based STS,
                  final = link log-odds + 3 * (STS risk - 0.5); block / warn / pass thresholds
                  chosen on the validation split.
Runs locally only (127.0.0.1). The feature modules contact every link host in the email.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import threading
import time
import webbrowser
from email import policy
from email.parser import BytesParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import build_dataset_fast as bdf  # noqa: E402
from build_v4.normalize import normalise  # noqa: E402
from slice_models.slice_final_predict import predict_final  # noqa: E402
from slice_models.sts_rules import WEIGHTS as STS_WEIGHTS, sts_rule_score  # noqa: E402
from slice_models.train_slice_models import select_threshold  # noqa: E402

MODEL_DIR = ROOT / "models" / "final3"
CACHE = ROOT / "private" / "lookup_cache.sqlite"
POLICY_FILE = MODEL_DIR / "policy_thresholds.json"
FAMILIES = ("STS", "BAS", "DMS", "LOS", "RCS", "SRC", "INFRA")

app = Flask(__name__)
LOCK = threading.Lock()          # the feature modules keep per-process state: one email at a time
STATE: dict = {}


# --------------------------------------------------------------------------- setup
def policy_thresholds() -> dict:
    """warn = the final model's validation threshold (<=1% legit flagged); block = <=0.1% on VAL."""
    art = joblib.load(MODEL_DIR / "slice_final_model.joblib")
    if POLICY_FILE.exists():
        return json.loads(POLICY_FILE.read_text())
    d = pd.read_excel(ROOT / "slice_dataset_v4.xlsx", sheet_name="Dataset")
    v = d[d.prov__time_split == "val"].reset_index(drop=True)
    p = predict_final(v, model_dir=MODEL_DIR).FINAL_prob.to_numpy()
    t_block, _ = select_threshold(v.label.to_numpy(), p, 0.001)
    out = {"warn": float(art["threshold"]), "block": float(t_block),
           "source": "validation split of slice_dataset_v4 (warn: <=1% legit flagged, block: <=0.1%)"}
    POLICY_FILE.write_text(json.dumps(out, indent=1))
    return out


def setup():
    print("Loading feature modules and models (first start may take ~1 minute)...", flush=True)
    bdf._worker_init(True, str(CACHE))
    STATE["final"] = joblib.load(MODEL_DIR / "slice_final_model.joblib")
    STATE["scores"] = {s: joblib.load(MODEL_DIR / f"slice_{s.lower()}_model.joblib")
                       for s in STATE["final"]["score_inputs"]}
    STATE["used"] = {c for a in STATE["scores"].values() for c in a["feature_columns"]} | \
                    set(STATE["final"]["extra_feature_columns"])
    STATE["policy"] = policy_thresholds()
    print(f"Ready. Thresholds: block >= {STATE['policy']['block']:.3f}, warn >= {STATE['policy']['warn']:.3f}", flush=True)


# --------------------------------------------------------------------------- helpers
def clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (v is None or math.isnan(v) or math.isinf(v)) else round(float(v), 6)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def at_prior(p: float, prior: float) -> float:
    odds = p / max(1 - p, 1e-12) * prior / (1 - prior)
    return odds / (1 + odds)


def header_view(raw: bytes) -> dict:
    m = BytesParser(policy=policy.default).parsebytes(raw)
    return {h: str(m.get(h, ""))[:300] for h in ("From", "To", "Reply-To", "Return-Path", "Subject", "Date")}


def analyse(raw: bytes) -> dict:
    t0 = time.perf_counter()
    norm, ninfo = normalise(raw, ["phishing@pot"])
    t1 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="slice_app_") as tmp:
        path = Path(tmp) / "email.eml"
        path.write_bytes(norm)
        res = bdf._worker_run((str(path), "email_000001", -1, hashlib.sha256(norm).hexdigest(), False, True))
    t2 = time.perf_counter()
    row = res["row"]
    df = pd.DataFrame([row])
    out = predict_final(df, model_dir=MODEL_DIR, explain=True, top_k=6).iloc[0]
    sts = sts_rule_score(df).iloc[0]
    t3 = time.perf_counter()

    pol = STATE["policy"]
    p = float(out["FINAL_prob"])
    n_links = int(pd.to_numeric(row.get("BAS_meta__number_of_links"), errors="coerce") or 0)
    tier = "BLOCK" if p >= pol["block"] else ("WARN" if p >= pol["warn"] else "PASS")

    scores = []
    for s, a in STATE["scores"].items():
        scores.append({"name": s, "prob": clean(out[f"{s}_prob"]), "threshold": clean(a["threshold"]),
                       "algorithm": a.get("algorithm"), "n_features": len(a["feature_columns"])})
    features = {}
    for fam in FAMILIES:
        features[fam] = [{"name": k.split("__", 1)[1], "value": clean(v), "used": k in STATE["used"]}
                         for k, v in row.items() if k.startswith(fam + "__")]
    meta = {k: clean(v) for k, v in row.items() if "_meta__" in k}
    links = [{k: clean(v) for k, v in lr.items() if k not in ("email_id",)} for lr in res.get("links", [])]
    errors = [{k: clean(v) for k, v in e.items() if k != "email_id"} for e in res.get("errors", [])]

    return {
        "verdict": tier, "final_prob": p, "link_prob": clean(out["FINAL_link_prob"]),
        "prob_at_1pct": at_prior(p, 0.01), "prob_at_01pct": at_prior(p, 0.001),
        "policy": pol, "in_scope": n_links > 0, "n_links": n_links,
        "processing_status": row.get("processing_status"),
        "sts": {"score": clean(sts["STS_rule"]), "risk": clean(sts["STS_rule_risk"]),
                "reasons": sts["STS_rule_reasons"], "weights": STS_WEIGHTS,
                "combination": STATE["final"]["sts_combination"]["formula"]},
        "scores": scores, "top_reasons": out.get("top_reasons", ""),
        "headers": header_view(raw), "normalisation": {k: clean(v) for k, v in ninfo.items()},
        "features": features, "meta": meta, "links": links, "errors": errors,
        "timing_s": {"normalise": t1 - t0, "features (lookups)": t2 - t1, "models": t3 - t2, "total": t3 - t0},
        "model": {"dir": str(MODEL_DIR.relative_to(ROOT)), "final_algorithm": STATE["final"].get("algorithm"),
                  "inputs": STATE["final"]["input_columns"]},
    }


# --------------------------------------------------------------------------- routes
@app.post("/analyze")
def analyze_route():
    f = request.files.get("eml")
    if not f or not f.filename.lower().endswith(".eml"):
        return jsonify({"error": "Please upload a .eml file."}), 400
    raw = f.read()
    if not raw.strip():
        return jsonify({"error": "The file is empty."}), 400
    try:
        with LOCK:
            result = analyse(raw)
        result["filename"] = f.filename
        return jsonify(result)
    except Exception as exc:  # show the error in the page instead of a blank failure
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.get("/")
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLiCE Email Analyzer</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1d2330;--muted:#5f6b7a;--line:#e3e7ed;--accent:#2f5bd3;
--block:#c62828;--block-bg:#fdecea;--warn:#b26a00;--warn-bg:#fff4e0;--pass:#2e7d32;--pass-bg:#e8f5e9}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--text:#e7eaf0;--muted:#9aa5b4;--line:#2c333f;
--accent:#7aa2ff;--block:#ff8a80;--block-bg:#3a1d1d;--warn:#ffcc80;--warn-bg:#3a2e18;--pass:#a5d6a7;--pass-bg:#1c3320}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}
.card h2{font-size:16px;margin:0 0 10px}.drop{border:2px dashed var(--line);border-radius:10px;padding:26px;text-align:center;cursor:pointer}
.drop.over{border-color:var(--accent)}button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:9px 16px;font-size:14px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}.note{color:var(--muted);font-size:13px}
.verdict{display:flex;gap:18px;align-items:center;flex-wrap:wrap;border-radius:10px;padding:16px 18px}
.verdict .tag{font-size:28px;font-weight:700;letter-spacing:.5px}.BLOCK{background:var(--block-bg);color:var(--block)}
.WARN{background:var(--warn-bg);color:var(--warn)}.PASS{background:var(--pass-bg);color:var(--pass)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.kv div{background:var(--bg);border-radius:8px;padding:8px 10px}
.kv b{display:block;font-size:12px;color:var(--muted);font-weight:500}.bar{height:8px;background:var(--line);border-radius:4px;position:relative;margin-top:4px}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:4px;background:var(--accent)}.bar s{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--text)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top;word-break:break-word}
th{color:var(--muted);font-weight:600}.unused{color:var(--muted)}details summary{cursor:pointer;font-weight:600;margin:6px 0}
ul{margin:4px 0 0 18px;padding:0}.err{color:var(--block)}.wrap{overflow-x:auto}.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
.spin{display:inline-block;width:16px;height:16px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:s 1s linear infinite;vertical-align:-3px;margin-right:6px}
@keyframes s{to{transform:rotate(360deg)}}
</style></head><body><main>
<h1>SLiCE Email Analyzer</h1>
<p class="sub">Upload a <b>.eml</b> file to run the full pipeline: normalisation, the six feature modules, the four link scores, the rule-based STS and the final model.</p>
<div class="card">
  <div id="drop" class="drop">Drop a .eml file here or click to choose<br><span class="note" id="fname">No file selected</span></div>
  <input id="file" type="file" accept=".eml" hidden>
  <p class="note">The feature modules make live lookups for every link (HTTP, DNS, TLS) — only analyse emails whose links you are willing to have contacted. Unsubscribe-looking links are skipped. Everything runs locally.</p>
  <button id="go" disabled>Analyse</button> <span id="status" class="note"></span>
</div>
<div id="out"></div>
</main>
<script>
const $=s=>document.querySelector(s);let file=null;
const drop=$('#drop'),inp=$('#file'),go=$('#go'),out=$('#out'),st=$('#status');
drop.onclick=()=>inp.click();inp.onchange=()=>pick(inp.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');pick(e.dataTransfer.files[0])};
function pick(f){if(!f)return;file=f;$('#fname').textContent=f.name;go.disabled=false}
const esc=v=>v===null||v===undefined?'<span class="note">—</span>':String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=v=>v===null||v===undefined?'—':(100*v).toFixed(1)+'%';
go.onclick=async()=>{if(!file)return;go.disabled=true;out.innerHTML='';st.innerHTML='<span class="spin"></span>Analysing (live lookups can take 10–30 s for new domains)…';
  const fd=new FormData();fd.append('eml',file);
  try{const r=await fetch('/analyze',{method:'POST',body:fd});const j=await r.json();
    if(!r.ok||j.error){st.innerHTML='<span class="err">'+esc(j.error||r.statusText)+'</span>';}else{st.textContent='Done in '+j.timing_s.total.toFixed(1)+' s';render(j)}}
  catch(e){st.innerHTML='<span class="err">'+esc(e)+'</span>'}go.disabled=false};
function bar(p,t){return `<div class="bar"><i style="width:${Math.min(100,100*(p||0))}%"></i>${t!=null?`<s style="left:${100*t}%" title="threshold"></s>`:''}</div>`}
function render(j){
  const v=j.verdict, P=j.policy;
  let h=`<div class="card"><div class="verdict ${v}"><span class="tag">${v}</span><div>
   <div><b>Final phishing probability:</b> ${pct(j.final_prob)} <span class="note">(under a 50/50 prior)</span></div>
   <div class="note">At a realistic 1% phishing rate: ${pct(j.prob_at_1pct)} · at 0.1%: ${pct(j.prob_at_01pct)}</div>
   <div class="note">Policy: block ≥ ${P.block.toFixed(3)}, warn ≥ ${P.warn.toFixed(3)}, else pass (thresholds from the validation split)</div></div></div>
   ${j.in_scope?'':'<p class="err">This email has no http(s) links — it is outside the model\'s scope (the patent covers emails with at least one link); the score is not meaningful.</p>'}
   ${j.processing_status!=='ok'?`<p class="err">Processing status: ${esc(j.processing_status)} — some modules failed; see Errors below.</p>`:''}</div>`;
  h+=`<div class="card"><h2>Scores</h2><div class="kv">
   <div><b>Link model (BAS, DMS, LOS, RCS + INFRA)</b>${pct(j.link_prob)}${bar(j.link_prob,null)}</div>
   <div><b>STS — source trust (rule-based)</b>${j.sts.score} <span class="note">(1 = trustworthy)</span>${bar(j.sts.score,null)}</div>
   ${j.scores.map(s=>`<div><b>${s.name} <span class="note">(${s.algorithm}, ${s.n_features} features)</span></b>${pct(s.prob)} <span class="note">threshold ${pct(s.threshold)}</span>${bar(s.prob,s.threshold)}</div>`).join('')}
   </div><p class="note">Final log-odds = ${esc(j.sts.combination.replace('final log-odds = ',''))}</p></div>`;
  h+=`<div class="card"><h2>Why</h2><div class="kv"><div><b>Top contributions to the link model (feature = value, +log-odds)</b>${esc(j.top_reasons)||'—'}</div>
   <div><b>STS rules that raised risk (weighted)</b>${esc(j.sts.reasons)||'none — source fully trustworthy'}</div></div></div>`;
  h+=`<div class="card"><h2>Email</h2><table>${Object.entries(j.headers).map(([k,x])=>`<tr><th style="width:130px">${k}</th><td>${esc(x)}</td></tr>`).join('')}</table>
   <h2 style="margin-top:14px">Normalisation & authentication (receiver verdicts, canonical form)</h2><table>${Object.entries(j.normalisation).map(([k,x])=>`<tr><th style="width:230px">${esc(k)}</th><td>${esc(x)}</td></tr>`).join('')}</table></div>`;
  h+=`<div class="card"><h2>All features</h2><p class="note">Grey = computed but not used by the final models (e.g. the DKIM-derived features removed from the link side, and SRC).</p>`;
  for(const [fam,rows] of Object.entries(j.features)){h+=`<details ${fam==='STS'?'open':''}><summary>${fam} (${rows.length})</summary><table><tr><th>feature</th><th>value</th></tr>
   ${rows.map(r=>`<tr class="${r.used?'':'unused'}"><td>${esc(r.name)}</td><td>${esc(r.value)}</td></tr>`).join('')}</table></details>`}
  h+=`<details><summary>Module metadata (${Object.keys(j.meta).length})</summary><table>${Object.entries(j.meta).map(([k,x])=>`<tr><td>${esc(k)}</td><td>${esc(x)}</td></tr>`).join('')}</table></details></div>`;
  if(j.links.length){const cols=Object.keys(j.links.reduce((a,r)=>Object.assign(a,r),{}));
   h+=`<div class="card"><h2>Links (${j.links.length})</h2><div class="wrap"><table class="mono"><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr>
   ${j.links.map(r=>`<tr>${cols.map(c=>`<td>${esc(r[c])}</td>`).join('')}</tr>`).join('')}</table></div></div>`}
  h+=`<div class="card"><h2>Timing</h2><div class="kv">${Object.entries(j.timing_s).map(([k,x])=>`<div><b>${k}</b>${x.toFixed(3)} s</div>`).join('')}</div>
   <p class="note">Model folder: ${esc(j.model.dir)} · final model: ${esc(j.model.final_algorithm)}</p></div>`;
  if(j.errors.length){h+=`<div class="card"><h2>Errors / failed lookups (${j.errors.length})</h2><table>${j.errors.map(e=>`<tr><td>${esc(e.error_type)}</td><td>${esc(e.error_message)}</td></tr>`).join('')}</table></div>`}
  out.innerHTML=h;}
</script></body></html>"""


if __name__ == "__main__":
    setup()
    url = "http://127.0.0.1:5000"
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"SLiCE app running at {url}  (Ctrl+C to stop)", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
