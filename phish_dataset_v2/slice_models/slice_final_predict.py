"""
Inference for the SLiCE v3 stack: five score models -> final calibrated probability.

    import pandas as pd
    from slice_models.slice_final_predict import predict_final

    df = pd.read_excel("slice_dataset_v3.xlsx", sheet_name="Dataset")   # any frame with the feature columns
    out = predict_final(df)                     # STS_prob ... RCS_prob, FINAL_prob, FINAL_flag
    out = predict_final(df, prior=0.01)         # + FINAL_prob_at_prior for a 1% phishing base rate
    out = predict_final(df, explain=True)       # + top_reasons (largest contributions toward phishing)

If the artifact was trained with --sts-fixed-weight, the learned model covers only the link
side (BAS, DMS, LOS, RCS + SRC/INFRA) and STS is added afterwards with the fixed weight:
FINAL_link_prob = learned link probability, FINAL_prob = link log-odds + w * (STS risk - 0.5).

FINAL_prob is calibrated under a 50/50 prior (class-balanced Platt on validation);
FINAL_flag uses the validation-selected threshold on that scale. `prior` re-weights the
odds to a deployment base rate: odds' = odds * prior / (1 - prior).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slice_models.slice_predict import SCORES, preprocess, predict_score
from slice_models.sts_rules import sts_rule_score

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models" / "final3"   # best run (fewer false alarms); others load via model_dir=
_FINAL: dict[str, dict] = {}


def load_final_model(model_dir: Path | str | None = None) -> dict:
    path = Path(model_dir or DEFAULT_DIR) / "slice_final_model.joblib"
    key = str(path)
    if key not in _FINAL:
        _FINAL[key] = joblib.load(path)
    return _FINAL[key]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def predict_final(df: pd.DataFrame, model_dir: Path | str | None = None, prior: float | None = None,
                  explain: bool = False, top_k: int = 3) -> pd.DataFrame:
    model_dir = Path(model_dir or DEFAULT_DIR)
    art = load_final_model(model_dir)
    out = pd.DataFrame(index=df.index)
    for s in art["score_inputs"]:
        if s == "STS" and art.get("sts_mode") == "rule":
            rule = sts_rule_score(df)
            out["STS_prob"] = rule["STS_rule_risk"].to_numpy()      # risk = 1 - STS, same input as in training
            out["STS_rule_reasons"] = rule["STS_rule_reasons"].to_numpy()
        else:
            out[f"{s}_prob"] = predict_score(df, s, model_dir)
    M = pd.concat([out[[f"{s}_prob" for s in art["score_inputs"]]],
                   preprocess(df, art["extra_feature_columns"], {})], axis=1)[art["input_columns"]]
    if art.get("logit_score_inputs"):          # linear final model was fit on score logits
        M = M.copy()
        n = len(art["score_inputs"])
        M.iloc[:, :n] = _logit(M.iloc[:, :n].to_numpy())
    raw = art["model"].predict_proba(M)[:, 1]
    p = art["calibrator"].predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
    comb = art.get("sts_combination", {})
    if comb.get("mode") == "fixed_weight_after_ml":
        # STS is not inside the learned model: add it with the fixed, pre-set weight
        rule = sts_rule_score(df)
        out["STS_rule"] = rule["STS_rule"].to_numpy()
        out["STS_rule_reasons"] = rule["STS_rule_reasons"].to_numpy()
        out["FINAL_link_prob"] = p
        z = _logit(p) + comb["weight"] * (rule["STS_rule_risk"].to_numpy(dtype=float) - comb["center"])
        p = 1 / (1 + np.exp(-z))
    out["FINAL_prob"] = p
    out["FINAL_flag"] = (p >= art["threshold"]).astype(int)
    if prior is not None:
        if not 0 < prior < 1:
            raise ValueError("prior must be in (0, 1)")
        odds = p / np.clip(1 - p, 1e-12, None) * prior / (1 - prior)
        out["FINAL_prob_at_prior"] = odds / (1 + odds)
    if explain:
        names = np.array(art["input_columns"])
        reasons = []
        if hasattr(art["model"], "booster_"):     # LightGBM: exact per-input log-odds contributions
            contrib = art["model"].predict(M, pred_contrib=True)[:, :-1]
            for row, vals in zip(contrib, M.to_numpy()):
                order = np.argsort(row)[::-1][:top_k]
                reasons.append("; ".join(f"{names[i]}={vals[i]:.3g} (+{row[i]:.2f})" for i in order if row[i] > 0))
        else:                                       # other models: score-level reasons (link scores above 0.5)
            score_cols = [f"{s}_prob" for s in art["score_inputs"]]
            for _, r in out[score_cols].iterrows():
                hi = r[r >= 0.5].sort_values(ascending=False)[:top_k]
                reasons.append("; ".join(f"{k.replace('_prob', '')}={v:.2f}" for k, v in hi.items()) or "no link score >= 0.5")
        out["top_reasons"] = reasons
    return out
