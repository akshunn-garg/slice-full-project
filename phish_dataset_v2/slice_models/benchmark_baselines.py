"""
Baselines + deployment metrics for the final SLiCE model (models/final3).

    python -m slice_models.benchmark_baselines      # writes reports/benchmark/*.csv

Same data as the final model: slice_dataset_v4 splits + the same training-only extra rows.
Every baseline is a LightGBM model with the same 5 candidate configurations, chosen on VAL
PR-AUC, Platt-calibrated on VAL, threshold = max recall with VAL FPR <= 1%. Test and Nazario
2024-25 (held out) are scored once.

  B1 url_only            LOS + SRC (URL text only; no headers, no network)
  B2 all_features_flat   all 73 module features in one model (the conventional design;
                         includes authentication, so it can learn the authentication shortcut)
  B3 slice_offline       SLiCE with no network lookups: link model on BAS/DMS (without
                         DKIM-derived and redirect features) + LOS + SRC, plus the STS rule
                         with sender-domain age set to unknown, fixed weight 3
  SLiCE (final3)         the final model

Also: bootstrap 95% CIs, precision at 1% and 0.1% phishing prevalence, and a 3-tier policy
(block / warn / pass) for SLiCE with thresholds from VAL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, matthews_corrcoef, roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slice_models.slice_final_predict import predict_final
from slice_models.slice_predict import SENTINELS, family_columns, preprocess
from slice_models.sts_rules import sts_rule_score
from slice_models.train_slice_final import AUTH_DERIVED
from slice_models.train_slice_models import select_threshold

ROOT = Path(__file__).resolve().parent.parent
EXTRA = [ROOT / "private" / "nazario_train" / "train_rows.csv", ROOT / "private" / "v4" / "extra_legit_rows.csv",
         ROOT / "private" / "v4" / "unused_train_rows.csv"]
NETWORK_FEATURES = {"STS__new_sender_domain", "BAS__claimed_or_authenticated_brand_equals_final_redirect",
                    "DMS__final_domain_equals_expected"}          # + all RCS__ and INFRA__ (live lookups)
CONFIGS = [dict(num_leaves=15, learning_rate=0.05, n_estimators=300, min_child_samples=20),
           dict(num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=20),
           dict(num_leaves=7, learning_rate=0.1, n_estimators=200, min_child_samples=50),
           dict(num_leaves=31, learning_rate=0.02, n_estimators=600, min_child_samples=50, reg_lambda=5.0),
           dict(num_leaves=63, learning_rate=0.05, n_estimators=300, min_child_samples=10, colsample_bytree=0.8)]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load():
    d = pd.read_excel(ROOT / "slice_dataset_v4.xlsx", sheet_name="Dataset")
    for f in EXTRA:
        e = pd.read_csv(f)
        e["prov__time_split"] = "train"
        d = pd.concat([d, e[[c for c in e.columns if c in d.columns]]], ignore_index=True)
    nz = pd.DataFrame([json.loads(l)["row"] for l in open(ROOT / "private" / "nazario" / "checkpoint_nazario.jsonl", encoding="utf-8")])
    nz = nz[(nz.processing_status == "ok") & (pd.to_numeric(nz.BAS_meta__number_of_links, errors="coerce") > 0)]
    return d.reset_index(drop=True), nz.reset_index(drop=True)


def fit_lgb(X, y, tr, va, seed=42):
    best = None
    for cfg in CONFIGS:
        m = lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=4, deterministic=True,
                               force_row_wise=True, verbose=-1, **cfg).fit(X[tr], y[tr])
        pv = m.predict_proba(X[va])[:, 1]
        ap, ll = average_precision_score(y[va], pv), log_loss(y[va], np.clip(pv, 1e-7, 1 - 1e-7))
        if best is None or (ap, -ll) > (best[0], -best[1]):
            best = (ap, ll, m)
    m = best[2]
    cal = LogisticRegression(class_weight="balanced", C=1e6, max_iter=1000).fit(
        logit(m.predict_proba(X[va])[:, 1]).reshape(-1, 1), y[va])
    return lambda Z: cal.predict_proba(logit(m.predict_proba(Z)[:, 1]).reshape(-1, 1))[:, 1]


def metrics(y, p, thr, rng, n_boot=1000):
    pred = (p >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    rec, fpr = tp / max(tp + fn, 1), fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    out = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "recall": rec, "fpr": fpr,
           "precision": prec, "f1": 2 * prec * rec / max(prec + rec, 1e-12), "mcc": matthews_corrcoef(y, pred),
           "fp": fp, "fn": fn}
    for prev in (0.01, 0.001):
        out[f"precision_at_{prev:g}_prevalence"] = rec * prev / max(rec * prev + fpr * (1 - prev), 1e-12)
    boots = {"roc_auc": [], "recall": [], "fpr": []}
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        yb, pb = y[b], p[b]
        if yb.min() == yb.max():
            continue
        boots["roc_auc"].append(roc_auc_score(yb, pb))
        boots["recall"].append(((pb >= thr) & (yb == 1)).sum() / max((yb == 1).sum(), 1))
        boots["fpr"].append(((pb >= thr) & (yb == 0)).sum() / max((yb == 0).sum(), 1))
    for k, v in boots.items():
        out[f"{k}_ci95"] = f"[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]"
    return out


def main():
    rng = np.random.default_rng(42)
    d, nz = load()
    y = d.label.to_numpy()
    sp = d.prov__time_split.to_numpy()
    tr, va, te = sp == "train", sp == "val", sp == "test"
    rep = ROOT / "reports" / "benchmark"
    rep.mkdir(parents=True, exist_ok=True)
    fam = lambda f: family_columns(d.columns, f)
    rows, nz_rows, preds = [], [], {}

    def run(name, cols, combine_sts=False, sts_offline=False):
        sent = {k: v for k, v in SENTINELS.items() if k in cols}
        X, Xn = preprocess(d, cols, sent), preprocess(nz, cols, sent)
        f = fit_lgb(X, y, tr, va)
        p, pn = f(X), f(Xn)
        if combine_sts:
            def sts_risk(frame):
                g = frame.copy()
                if sts_offline:
                    g["STS__new_sender_domain"] = -1          # age unknown without a network lookup
                return sts_rule_score(g)["STS_rule_risk"].to_numpy(dtype=float)
            p = 1 / (1 + np.exp(-(logit(p) + 3.0 * (sts_risk(d) - 0.5))))
            pn = 1 / (1 + np.exp(-(logit(pn) + 3.0 * (sts_risk(nz) - 0.5))))
        thr, _ = select_threshold(y[va], p[va], 0.01)
        rows.append({"model": name, "n_features": len(cols), **metrics(y[te], p[te], thr, rng)})
        nz_rows.append({"model": name, "nazario_recall": float((pn >= thr).mean())})
        preds[name] = (p, thr)

    run("B1 url_only (LOS + SRC)", fam("LOS") + fam("SRC"))
    run("B2 all_features_flat (73, incl. authentication)",
        [c for f in ("STS", "BAS", "DMS", "LOS", "RCS", "SRC", "INFRA") for c in fam(f)])
    offline_link = [c for c in fam("BAS") + fam("DMS") if c not in AUTH_DERIVED and c not in NETWORK_FEATURES] \
        + fam("LOS") + fam("SRC")
    run("B3 slice_offline (no network lookups) + STS rule", offline_link, combine_sts=True, sts_offline=True)

    # the final SLiCE model (thresholds from its own artifact, chosen on VAL)
    art = joblib.load(ROOT / "models" / "final3" / "slice_final_model.joblib")
    o, on = predict_final(d), predict_final(nz)
    p, pn = o.FINAL_prob.to_numpy(), on.FINAL_prob.to_numpy()
    rows.append({"model": "SLiCE final (models/final3)", "n_features": len(art["input_columns"]),
                 **metrics(y[te], p[te], art["threshold"], rng)})
    nz_rows.append({"model": "SLiCE final (models/final3)", "nazario_recall": float((pn >= art["threshold"]).mean())})

    res = pd.DataFrame(rows).merge(pd.DataFrame(nz_rows), on="model")
    res.to_csv(rep / "baselines_test_and_nazario.csv", index=False)

    # 3-tier policy for SLiCE: block if VAL FPR would be <= 0.1%, warn if <= 1%, else pass
    t_block, _ = select_threshold(y[va], p[va], 0.001)
    t_warn = art["threshold"]
    def tiers(prob):
        return np.where(prob >= t_block, "block", np.where(prob >= t_warn, "warn", "pass"))
    pol = []
    for name, prob, lab in (("test legit", p[te & (y == 0)], None), ("test phishing", p[te & (y == 1)], None),
                            ("Nazario phishing", pn, None)):
        c = pd.Series(tiers(prob)).value_counts(normalize=True)
        pol.append({"set": name, "n": len(prob), **{k: round(100 * c.get(k, 0), 2) for k in ("block", "warn", "pass")}})
    pol = pd.DataFrame(pol)
    pol.to_csv(rep / "policy_three_tier.csv", index=False)

    pd.set_option("display.width", 250)
    show = ["model", "n_features", "roc_auc", "roc_auc_ci95", "recall", "recall_ci95", "fpr", "fpr_ci95", "precision",
            "f1", "mcc", "precision_at_0.01_prevalence", "precision_at_0.001_prevalence", "nazario_recall"]
    print(res[show].round(4).to_string(index=False))
    print(f"\n3-tier policy (thresholds from VAL: block >= {t_block:.3f}, warn >= {t_warn:.3f}), % of emails:")
    print(pol.to_string(index=False))


if __name__ == "__main__":
    main()
