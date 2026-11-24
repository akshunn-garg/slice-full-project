"""
SLiCE v3: five score models (STS, BAS, DMS, LOS, RCS) + one final stacked model.

Run from the project root:
    python -m slice_models.train_slice_stack            # v4 defaults: rule-based STS added with fixed weight 3

Data: slice_dataset_v3.xlsx, rows with prov__core == 1, split = prov__time_split
(time-ordered: train on earlier mail, validate / test on later mail).

  1. Score models. Per family, hyper-parameters are chosen by grouped 5-fold CV on
     TRAIN (StratifiedGroupKFold, groups = sender organisation), then refit on all of
     TRAIN. Their out-of-fold TRAIN predictions are the final model's training inputs,
     so the final model never sees in-sample base scores.
  2. Final model. LightGBM on [5 score probabilities + SRC__ + INFRA__ features]; the
     five probabilities are monotone-increasing constraints (a higher score can never
     lower the final risk). Config chosen on VAL PR-AUC, then a class-balanced Platt
     calibration is fitted on VAL: the final probability assumes a 50/50 prior and can
     be re-weighted to a deployment prior in slice_final_predict.
  3. Thresholds (max recall s.t. FPR <= --max-fpr) on VAL. TEST is scored once, after
     every artifact is saved.

Outputs: models/v3/*.joblib, slice_training_dataset.xlsx, reports/slice_v3/*.
The v1 artifacts in models/ and the v1 training script are not touched.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in (None, ""):  # allow `python slice_models/train_slice_stack.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slice_models.slice_predict import SCORES, SENTINELS, family_columns, preprocess
from slice_models.sts_rules import WEIGHTS as RULE_WEIGHTS, sts_rule_score
from slice_models.train_slice_models import (candidate_configs, full_metrics, select_threshold,
                                             sha256_file)

ROOT = Path(__file__).resolve().parent.parent
MODEL_VERSION = "3.0.0"
EXTRA_PREFIXES = ("SRC__", "INFRA__")          # "other parameters" fed to the final model
PROB_COLS = [f"{s}_prob" for s in SCORES]
INFO_COLS = ["email_id", "label", "prov__time_split", "prov__source", "prov__receiver",
             "prov__arrival_time", "prov__sender_org", "prov__ms_delivery_folder"]

log = logging.getLogger("slice_v3")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def configs(n_random: int, seed: int) -> list[dict]:
    """The v1 search space plus a number of boosting rounds (no early stopping inside CV)."""
    cfgs = candidate_configs(n_random, seed)
    rng = np.random.default_rng(seed + 1)
    for i, c in enumerate(cfgs):
        c["n_estimators"] = 200 if i == 0 else int(rng.choice([100, 200, 400]))
    return cfgs


def make_model(cfg: dict, args, monotone: list[int] | None = None) -> lgb.LGBMClassifier:
    p = dict(cfg)
    n_est = p.pop("n_estimators")
    extra = {}
    if monotone is not None:
        extra = dict(monotone_constraints=monotone, monotone_constraints_method="advanced")
    return lgb.LGBMClassifier(objective="binary", n_estimators=n_est,
                              subsample_freq=1 if p["subsample"] < 1.0 else 0,
                              random_state=args.seed, n_jobs=args.threads, deterministic=True,
                              force_row_wise=True, verbose=-1, **p, **extra)


def oof_predict(cfg, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, args) -> np.ndarray:
    folds = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.full(len(y), np.nan)
    for tr, te in folds.split(X, y, groups):
        m = make_model(cfg, args).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    assert not np.isnan(oof).any()
    return oof


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def environment() -> dict:
    return {"python": platform.python_version(), "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
            "platform": platform.platform()}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="Dataset")
    if "prov__core" in d.columns:            # v3 marks its subset; v4 rows are all selected already
        d = d[d["prov__core"] == 1]
    d = d.reset_index(drop=True)
    if not d["email_id"].is_unique:
        raise ValueError("duplicate email_id in core subset")
    if set(d["prov__time_split"]) != {"train", "val", "test"}:
        raise ValueError(f"unexpected splits: {set(d['prov__time_split'])}")
    if "processing_status" in d.columns and (d["processing_status"] != "ok").any():  # v4 keeps only ok rows
        raise ValueError("core subset must contain only processing_status == ok")
    return d


# ---------------------------------------------------------------------------
# 1. score models
# ---------------------------------------------------------------------------
def train_score(score, d, masks, args, cfgs, model_dir, data_info):
    tr, va = masks["train"], masks["val"]
    y = d["label"].to_numpy()
    groups = d["prov__sender_org"].fillna("").astype(str).to_numpy()
    feats = family_columns(d.columns, score)
    sentinels = {k: v for k, v in SENTINELS.items() if k in feats}
    X = preprocess(d, feats, sentinels)

    rows, best = [], None
    for i, cfg in enumerate(cfgs):
        oof = oof_predict(cfg, X[tr], y[tr], groups[tr], args)
        ap, ll = average_precision_score(y[tr], oof), log_loss(y[tr], np.clip(oof, 1e-7, 1 - 1e-7))
        rows.append({"score": score, "config_id": i, "cv_pr_auc": ap, "cv_log_loss": ll,
                     **{k: v for k, v in cfg.items()}})
        if best is None or (ap, -ll) > (best["ap"], -best["ll"]):
            best = {"ap": ap, "ll": ll, "id": i, "oof": oof}
    cfg = cfgs[best["id"]]
    model = make_model(cfg, args).fit(X[tr], y[tr])
    p_full = model.predict_proba(X)[:, 1]
    thr, rule = select_threshold(y[va], p_full[va], args.max_fpr)
    val_metrics = full_metrics(y[va], p_full[va], thr)

    stack_input = p_full.copy()
    stack_input[tr] = best["oof"]           # out-of-fold for TRAIN rows, refit model for VAL/TEST

    art = {
        "model": model, "feature_columns": feats,
        "preprocessing": {"steps": ["select feature_columns in order", "to float64", "+/-inf -> NaN",
                                    "documented sentinels -> NaN", "LightGBM native missing handling"],
                          "sentinels": sentinels},
        "sampling_strategy": "none", "threshold": thr, "threshold_rule": rule,
        "model_name": f"slice_{score.lower()}_v3", "model_version": MODEL_VERSION,
        "validation_metrics": val_metrics,
        "training_config": {"hyperparameters": cfg, "config_id": best["id"],
                            "selection": f"grouped {args.folds}-fold CV PR-AUC on TRAIN (groups = sender org)",
                            "cv_pr_auc": best["ap"], "seed": args.seed, "data": data_info,
                            "n_train": int(tr.sum()), "n_val": int(va.sum())},
        "environment": environment(),
    }
    joblib.dump(art, model_dir / f"slice_{score.lower()}_model.joblib")
    log.info("[%s] %d features | best config %d | CV PR-AUC %.4f | val PR-AUC %.4f | thr %.4f",
             score, len(feats), best["id"], best["ap"], val_metrics["pr_auc"], thr)
    return stack_input, p_full, pd.DataFrame(rows), art


# ---------------------------------------------------------------------------
# 2. final model
# ---------------------------------------------------------------------------
def fit_selected(M, y, masks, cfgs, args, monotone):
    """Fit every config on TRAIN, keep the best VAL PR-AUC (tie -> lower VAL log loss)."""
    tr, va = masks["train"], masks["val"]
    best, rows = None, []
    for i, cfg in enumerate(cfgs):
        m = make_model(cfg, args, monotone).fit(M[tr], y[tr])
        pv = m.predict_proba(M[va])[:, 1]
        ap, ll = average_precision_score(y[va], pv), log_loss(y[va], np.clip(pv, 1e-7, 1 - 1e-7))
        rows.append({"config_id": i, "val_pr_auc": ap, "val_log_loss": ll, **cfg})
        if best is None or (ap, -ll) > (best[0], -best[1]):
            best = (ap, ll, i, m)
    return best[3], cfgs[best[2]], best[2], pd.DataFrame(rows)


def calibrate(raw_val, y_val):
    """Class-balanced Platt scaling on VAL -> probability under a 50/50 prior."""
    cal = LogisticRegression(class_weight="balanced", C=1e6, max_iter=1000)
    cal.fit(logit(raw_val).reshape(-1, 1), y_val)
    return cal


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "slice_dataset_v4.xlsx"))
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "v4"))
    ap.add_argument("--report-dir", default=str(ROOT / "reports" / "slice_v4"))
    ap.add_argument("--excel-out", default=str(ROOT / "slice_training_dataset_v4.xlsx"))
    ap.add_argument("--n-configs", type=int, default=12, help="random configs per model (+1 baseline)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sts-rule", action="store_true",
                    help="STS = rule-based score (slice_models.sts_rules, not trained); the other four are ML")
    ap.add_argument("--sts-fixed-weight", type=float, default=3.0,
                    help="exclude STS from the learned final model and add it afterwards with this FIXED weight: "
                         "final log-odds = link log-odds + w * (STS risk - 0.5). Implies --sts-rule.")
    args = ap.parse_args(argv)
    if args.sts_fixed_weight is not None:
        args.sts_rule = True
    fixed = args.sts_fixed_weight is not None

    model_dir, report_dir = Path(args.model_dir), Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(ROOT / "logs" / f"train_slice_stack_{stamp}.log", encoding="utf-8")])
    t0 = time.time()

    data_path = Path(args.data)
    d = load(data_path)
    data_info = {"file": data_path.name, "sha256": sha256_file(data_path), "rows_core": int(len(d))}
    masks = {s: (d["prov__time_split"] == s).to_numpy() for s in ("train", "val", "test")}
    y = d["label"].to_numpy()
    log.info("core rows %d | split x label %s", len(d),
             d.groupby(["prov__time_split", "label"]).size().to_dict())

    # ---- 1. five score models -------------------------------------------------------
    cfgs = configs(args.n_configs, args.seed)
    stack_in, full_pred, cv_tables, arts = {}, {}, [], {}
    for s in SCORES:
        if s == "STS" and args.sts_rule:
            # Rule-based STS: fixed a-priori weights, nothing learned, so the same value is used for
            # every split (no out-of-fold step needed). Input to the final model = risk (1 - STS).
            risk = sts_rule_score(d)["STS_rule_risk"].to_numpy(dtype=float)
            stack_in[s] = full_pred[s] = risk
            thr, rule = select_threshold(y[masks["val"]], risk[masks["val"]], args.max_fpr)
            arts[s] = {"threshold": thr, "threshold_rule": rule, "feature_columns": family_columns(d.columns, "STS"),
                       "training_config": {"hyperparameters": {"rule_weights": RULE_WEIGHTS}},
                       "validation_metrics": full_metrics(y[masks["val"]], risk[masks["val"]], thr)}
            log.info("[STS] rule-based (not trained) | val PR-AUC %.4f | thr %.4f",
                     arts[s]["validation_metrics"]["pr_auc"], thr)
            continue
        stack_in[s], full_pred[s], tab, arts[s] = train_score(s, d, masks, args, cfgs, model_dir, data_info)
        cv_tables.append(tab)

    # ---- 2. final model ---------------------------------------------------------------
    extra = [c for c in d.columns if c.startswith(EXTRA_PREFIXES)]
    M = pd.concat([pd.DataFrame({f"{s}_prob": stack_in[s] for s in SCORES}),
                   preprocess(d, extra, {})], axis=1)
    score_cols = [c for c in PROB_COLS if not (fixed and c == "STS_prob")]
    final_cols = score_cols + extra
    monotone = [1] * len(score_cols) + [0] * len(extra)
    meta_cfgs = configs(args.n_configs, args.seed + 7)
    final_model, final_cfg, final_id, meta_tab = fit_selected(M[final_cols], y, masks, meta_cfgs, args, monotone)
    raw = final_model.predict_proba(M[final_cols])[:, 1]
    cal = calibrate(raw[masks["val"]], y[masks["val"]])
    link_prob = cal.predict_proba(logit(raw).reshape(-1, 1))[:, 1]

    def combine(p_link, sts_risk, w):
        z = logit(p_link) + w * (np.asarray(sts_risk, dtype=float) - 0.5)
        return 1 / (1 + np.exp(-z))
    final_prob = combine(link_prob, stack_in["STS"], args.sts_fixed_weight) if fixed else link_prob
    final_thr, final_rule = select_threshold(y[masks["val"]], final_prob[masks["val"]], args.max_fpr)
    final_val = full_metrics(y[masks["val"]], final_prob[masks["val"]], final_thr)
    contrib_val = final_model.predict(M.loc[masks["val"], final_cols], pred_contrib=True)[:, :-1]
    importance = pd.DataFrame({"input": final_cols, "gain": final_model.booster_.feature_importance("gain"),
                               "mean_abs_shap_val": np.abs(contrib_val).mean(axis=0),
                               "monotone": monotone}).sort_values("mean_abs_shap_val", ascending=False)

    final_art = {
        "model": final_model, "calibrator": cal,
        "score_inputs": [c[:-5] for c in score_cols],
        "sts_mode": "rule" if args.sts_rule else "model",
        "sts_combination": ({"mode": "fixed_weight_after_ml", "weight": args.sts_fixed_weight, "center": 0.5,
                             "formula": "final log-odds = calibrated link log-odds + weight * (STS_rule_risk - 0.5)",
                             "note": "weight fixed before training, not tuned"} if fixed else {"mode": "inside_ml"}),
        "input_columns": final_cols, "extra_feature_columns": extra, "monotone_constraints": monotone,
        "preprocessing": {"score_inputs": "each score model's predict_proba (models/v3)",
                          "extra_features": "same preprocess() as the score models, no sentinels"},
        "calibration": "class-balanced Platt on VAL (probability under a 50/50 prior)",
        "threshold": final_thr, "threshold_rule": final_rule,
        "model_name": "slice_final_v3", "model_version": MODEL_VERSION,
        "validation_metrics": final_val,
        "training_config": {"hyperparameters": final_cfg, "config_id": final_id,
                            "selection": "VAL PR-AUC (tie -> VAL log loss)",
                            "train_inputs": "out-of-fold score probabilities on TRAIN", "seed": args.seed,
                            "data": data_info},
        "environment": environment(),
    }
    joblib.dump(final_art, model_dir / "slice_final_model.joblib")
    log.info("[FINAL] %d inputs | config %d | val PR-AUC %.4f | thr %.4f", len(final_cols), final_id,
             final_val["pr_auc"], final_thr)

    # ---- ablations (same protocol, for the report only) -------------------------------
    abl = {}
    m5, _, _, _ = fit_selected(M[PROB_COLS], y, masks, meta_cfgs, args, [1] * 5)
    abl["final_5_scores_only (STS weight learned)" if fixed else "final_5_scores_only"] = m5.predict_proba(M[PROB_COLS])[:, 1]
    if fixed:
        abl["FINAL_link_only (no STS)"] = link_prob
        m_in, _, _, _ = fit_selected(M[PROB_COLS + extra], y, masks, meta_cfgs, args, [1] * 5 + [0] * len(extra))
        abl["final_STS_inside_ML (learned weight)"] = m_in.predict_proba(M[PROB_COLS + extra])[:, 1]
        for w in (0.0, 1.0, 2.0, 3.0, 4.0):
            abl[f"sensitivity: fixed STS weight w={w:g}"] = combine(link_prob, stack_in["STS"], w)
    else:
        no_sts = [c for c in final_cols if c != "STS_prob"]
        m_ns, _, _, _ = fit_selected(M[no_sts], y, masks, meta_cfgs, args, [1] * 4 + [0] * len(extra))
        abl["final_without_STS"] = m_ns.predict_proba(M[no_sts])[:, 1]
    all_feats = [c for s in list(SCORES) + ["SRC", "INFRA"] for c in family_columns(d.columns, s)]
    X_all = preprocess(d, all_feats, {k: v for k, v in SENTINELS.items() if k in all_feats})
    m_all, _, _, _ = fit_selected(X_all, y, masks, cfgs, args, None)
    abl["single_model_all_features"] = m_all.predict_proba(X_all)[:, 1]
    log.info("all artifacts saved; scoring TEST once")

    # ---- 3. TEST (once) ----------------------------------------------------------------
    te = masks["test"]
    folder = d["prov__ms_delivery_folder"].fillna("").to_numpy() if "prov__ms_delivery_folder" in d else None
    hard = te & ((y == 0) | (folder == "inbox")) if folder is not None else None
    results = []
    def add(name, p, thr, kind):
        for slice_name, m in (("test", te), ("test_hard_positives", hard)):
            if m is None:
                continue
            r = full_metrics(y[m], p[m], thr) if thr is not None else \
                {**full_metrics(y[m], p[m], 0.5), "threshold": np.nan}
            results.append({"model": name, "kind": kind, "slice": slice_name, **r})
    for s in SCORES:
        add(s, full_pred[s], arts[s]["threshold"], "rule (not trained)" if s == "STS" and args.sts_rule else "score model")
    add("FINAL", final_prob, final_thr,
        f"final: ML(BAS,DMS,LOS,RCS + SRC + INFRA) + fixed STS weight {args.sts_fixed_weight:g}" if fixed
        else "final (5 scores + SRC + INFRA)")
    for k, p in abl.items():
        add(k, p, None, "ablation (threshold 0.5, ranking metrics only)")
    res = pd.DataFrame(results)
    keep = ["model", "kind", "slice", "n", "n_pos", "n_neg", "roc_auc", "pr_auc", "log_loss", "threshold",
            "precision", "recall", "f1", "fpr", "fnr", "tn", "fp", "fn", "tp"]
    res = res[keep]
    log.info("TEST results:\n%s", res[res.slice == "test"][["model", "roc_auc", "pr_auc", "recall", "fpr", "fp", "fn"]]
             .to_string(index=False))

    # ---- reports ----------------------------------------------------------------------
    res.to_csv(report_dir / f"{report_dir.name}_test_results.csv", index=False)
    pd.concat(cv_tables).to_csv(report_dir / f"{report_dir.name}_score_model_cv.csv", index=False)
    meta_tab.to_csv(report_dir / f"{report_dir.name}_final_model_search.csv", index=False)
    importance.to_csv(report_dir / f"{report_dir.name}_final_input_importance.csv", index=False)

    # ---- training dataset workbook ----------------------------------------------------
    info = d[[c for c in INFO_COLS if c in d.columns]].rename(columns={"prov__time_split": "split"})
    fam_cols = {s: family_columns(d.columns, s) for s in list(SCORES) + ["SRC", "INFRA"]}
    final_input = pd.concat([info[["email_id", "label", "split"]],
                             M[PROB_COLS].round(6).set_axis([f"{c}_input" for c in PROB_COLS], axis=1),
                             d[extra]], axis=1)
    final_input.insert(3, "score_input_source", np.where(masks["train"], "out-of-fold", "score model (trained on train)"))
    if fixed:
        final_input["FINAL_link_prob"] = np.where(masks["train"], np.nan, link_prob.round(6))
    final_input["FINAL_prob"] = np.where(masks["train"], np.nan, final_prob.round(6))
    final_input["FINAL_flag"] = np.where(masks["train"], np.nan, (final_prob >= final_thr).astype(int))

    readme = pd.DataFrame({"item": [
        "What", "Rows", "Split", "Score sheets", "Final_Model_Input", "Probabilities", "Models",
        "Features", "Limitations"], "description": [
        "Training data for the five SLiCE score models and the final model, built only from real emails "
        "(legit: Gmail inbox; phishing: phishing_pot honeypot) via the six unmodified feature modules.",
        f"{len(d)} emails from {data_path.name}" + (" (prov__core subset)" if "prov__core" in d.columns else "")
        + f", arrival {d['prov__arrival_time'].min()} .. {d['prov__arrival_time'].max()}.",
        "Time-ordered: train = earlier mail, val / test = later mail (same cut dates for both classes). "
        + json.dumps({f"{sp}/label{lab}": int(n) for (sp, lab), n
                      in d.groupby(["prov__time_split", "label"]).size().items()}),
        "STS, BAS, DMS, LOS, RCS: email_id, label, split + that family's features (exactly what each score model reads).",
        "The final model's inputs: the 5 score probabilities (out-of-fold for train rows) + SRC__ and INFRA__ "
        "features, and the final calibrated probability for val/test rows.",
        "Final probability = class-balanced Platt calibration on val, i.e. under a 50/50 prior; "
        "use slice_final_predict.predict_final(df, prior=...) for a deployment base rate.",
        f"{Path(args.model_dir).name}: slice_<score>_model.joblib for the ML scores + slice_final_model.joblib. "
        + ("STS is the rule-based score (slice_models/sts_rules.py, fixed weights, not trained); its column in "
           "Final_Model_Input is the rule's risk (1 - STS)." if args.sts_rule else "All five scores are LightGBM models."),
        "Features come from the six unmodified feature modules; see the dataset's own datasheet / column guide for "
        "normalisation (receiver hops, canonical authentication header, recipient anonymisation).",
        "Legit mail is one person's mailbox; phishing is one honeypot (phishing_pot). Lookups (domain age, DNS, TLS, "
        f"redirects) were made after arrival. Legit test emails: {int(((d.prov__time_split == 'test') & (d.label == 0)).sum())}."]})
    cfg_rows = [{"model": s, "threshold": arts[s]["threshold"], "threshold_rule": arts[s]["threshold_rule"],
                 "n_features": len(arts[s]["feature_columns"]),
                 "hyperparameters": json.dumps(arts[s]["training_config"]["hyperparameters"])} for s in SCORES]
    cfg_rows.append({"model": "FINAL", "threshold": final_thr, "threshold_rule": final_rule,
                     "n_features": len(final_cols), "hyperparameters": json.dumps(final_cfg)})

    with pd.ExcelWriter(args.excel_out, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        pd.concat([info.rename(columns=lambda c: c.replace("prov__", "")),
                   d[[c for cols in fam_cols.values() for c in cols]]], axis=1).to_excel(xw, sheet_name="Dataset", index=False)
        for s in SCORES:
            pd.concat([info[["email_id", "label", "split"]], d[fam_cols[s]]], axis=1).to_excel(xw, sheet_name=s, index=False)
        final_input.to_excel(xw, sheet_name="Final_Model_Input", index=False)
        res.to_excel(xw, sheet_name="Results_Test", index=False)
        pd.DataFrame(cfg_rows).to_excel(xw, sheet_name="Model_Config", index=False)
        importance.to_excel(xw, sheet_name="Final_Input_Importance", index=False)
        pd.DataFrame([{"family": f, "n_features": len(c), "columns": ", ".join(c)} for f, c in fam_cols.items()]
                     ).to_excel(xw, sheet_name="Feature_List", index=False)
    log.info("wrote %s | reports in %s | done in %.0fs", args.excel_out, report_dir, time.time() - t0)


if __name__ == "__main__":
    main()
