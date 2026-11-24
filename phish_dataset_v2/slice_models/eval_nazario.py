"""
Held-out test on the Nazario phishing corpus (2024-25) with the FROZEN final models.

    python -m slice_models.eval_nazario [--checkpoint private/nazario/checkpoint_nazario.jsonl]

Nazario is never used for training or tuning: the models, the STS rules and every threshold come
from models/final3 as saved. Nazario is phishing-only, so this measures recall (phishing caught),
compared with recall on the phishing_pot test split. False alarms can only be measured on legit mail.

Interpretation fixed before running (generalisation of the final decision):
  recall >= 90%   -> generalises well          75-90% -> partly source-specific
  recall <  75%   -> learned phishing_pot rather than phishing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slice_models.slice_final_predict import predict_final

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "final3"


def recall_table(out: pd.DataFrame, art: dict, model_dir: Path) -> dict:
    r = {}
    for s in art["score_inputs"]:
        thr = joblib.load(model_dir / f"slice_{s.lower()}_model.joblib")["threshold"]
        r[s] = float((out[f"{s}_prob"] >= thr).mean())
    chosen = pd.read_csv(Path(REPORT_DIR) / "chosen_methods.csv").set_index("model")
    r["STS (rule)"] = float(((1 - out["STS_rule"]) >= chosen.loc["STS (rule)", "threshold"]).mean())
    r["LINK MODEL"] = float((out["FINAL_link_prob"] >= art["link_threshold"]).mean())
    r["FINAL (link + STS)"] = float((out["FINAL_prob"] >= art["threshold"]).mean())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "private" / "nazario" / "checkpoint_nazario.jsonl"))
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "final3"))
    ap.add_argument("--report-dir", default=str(ROOT / "reports" / "final3"))
    a = ap.parse_args()
    global REPORT_DIR
    REPORT_DIR = a.report_dir
    model_dir = Path(a.model_dir)
    art = joblib.load(model_dir / "slice_final_model.joblib")

    rows = [json.loads(l)["row"] for l in open(a.checkpoint, encoding="utf-8")]
    nz = pd.DataFrame(rows)
    n_all = len(nz)
    nz = nz[(nz["processing_status"] == "ok") & (pd.to_numeric(nz["BAS_meta__number_of_links"], errors="coerce") > 0)]
    nz = nz.reset_index(drop=True)
    out_nz = predict_final(nz, model_dir=model_dir, explain=True)

    v4 = pd.read_excel(ROOT / "slice_dataset_v4.xlsx", sheet_name="Dataset")
    v4 = v4[(v4["prov__time_split"] == "test") & (v4["label"] == 1)].reset_index(drop=True)
    out_pp = predict_final(v4, model_dir=model_dir)

    rec_nz, rec_pp = recall_table(out_nz, art, model_dir), recall_table(out_pp, art, model_dir)
    tab = pd.DataFrame({"phishing_pot test recall": rec_pp, "Nazario recall": rec_nz})
    tab["difference (points)"] = 100 * (tab["Nazario recall"] - tab["phishing_pot test recall"])
    final = rec_nz["FINAL (link + STS)"]
    verdict = ("generalises well" if final >= 0.90 else
               "partly source-specific" if final >= 0.75 else "learned phishing_pot rather than phishing")

    rep = Path(a.report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    tab.to_csv(rep / "nazario_recall.csv")
    missed = out_nz[out_nz["FINAL_prob"] < art["threshold"]]
    missed.assign(email_id=nz.loc[missed.index, "email_id"]).to_csv(rep / "nazario_missed.csv", index=False)
    print(f"Nazario emails built: {n_all} | used (ok, >=1 link): {len(nz)} | phishing_pot test phishing: {len(v4)}")
    print((tab * [100, 100, 1]).round(1).to_string())
    print(f"FINAL recall on Nazario {100 * final:.1f}% -> {verdict}")
    print("mean FINAL_prob: Nazario", round(out_nz["FINAL_prob"].mean(), 3), "| phishing_pot test", round(out_pp["FINAL_prob"].mean(), 3))


if __name__ == "__main__":
    main()
