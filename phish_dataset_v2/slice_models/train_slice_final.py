"""
SLiCE final training: method selection -> 4 link scores -> final link model (+ rule-based STS).

    python -m slice_models.train_slice_final

Data: slice_dataset_v4.xlsx (unchanged), split = prov__time_split (time-ordered train / val / test).

  1. Method selection per score (BAS, DMS, LOS, RCS). Candidates: LightGBM, sklearn
     HistGradientBoosting, logistic regression (median impute + missing indicators + scaling),
     random forest; several configurations each. Chosen by grouped 5-fold CV on TRAIN only
     (StratifiedGroupKFold, groups = prov__template, so repeated newsletter templates never sit
     on both sides of a fold): highest CV PR-AUC, tie -> lower CV log loss.
  2. Each score is refit on all of TRAIN with its winner; its out-of-fold TRAIN predictions are
     the final model's training inputs.
  3. Final link model: inputs = the 4 score probabilities + INFRA__ features. Candidates:
     LightGBM and HistGradientBoosting with the 4 scores monotone-increasing, and logistic
     regression on the score logits + INFRA. Chosen on VAL PR-AUC (tie -> VAL log loss), then
     class-balanced Platt calibration on VAL (probability under a 50/50 prior).
  4. STS is the rule-based score (slice_models/sts_rules.py), not trained or tuned on this data.
     It joins the link model with the fixed weight 3:  log-odds = logit(link) + 3 * (STS_risk - 0.5).
     (In deployment the RL combiner, slice_models/rl_combiner.py, starts from this and learns.)
  5. Thresholds on VAL (max recall s.t. FPR <= --max-fpr). TEST is scored once, after every
     artifact is saved. "+SRC" (adding SRC__ features to the final model) is reported for
     information only; it does not change the saved model.
Outputs: models/final3/, reports/final3/, slice_training_dataset_final3.xlsx.
Defaults reproduce the final run (run 3): DKIM-derived features excluded; Nazario Dec 2022-2023,
502 diverse legit and 113 unused training-period emails added to TRAIN; final model LightGBM,
inputs (4 scores + INFRA vs 4 scores) chosen on VAL.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slice_models.slice_predict import SENTINELS, family_columns, preprocess
from slice_models.sts_rules import WEIGHTS as RULE_WEIGHTS, sts_rule_score
from slice_models.train_slice_models import full_metrics, select_threshold, sha256_file

ROOT = Path(__file__).resolve().parent.parent
LINK_SCORES = ("BAS", "DMS", "LOS", "RCS")
# Features computed from the DKIM signing domain: they encode source authentication, not link
# evidence, so --exclude-auth-features keeps them out of the link side (authentication enters
# only through the rule-based STS). Found by the held-out Nazario test.
AUTH_DERIVED = ("BAS__from_domain_equals_dkim_domain", "BAS__authenticated_domain_equals_destination",
                "DMS__dkim_domain_equals_destination")
STS_WEIGHT = 3.0
log = logging.getLogger("slice_final")


# ---------------------------------------------------------------------------
# candidate methods
# ---------------------------------------------------------------------------
def candidates(seed: int, threads: int, monotone: list[int] | None = None) -> list[tuple[str, dict, object]]:
    """(algorithm, config, unfitted estimator). `monotone` (+1/0 per column) is applied where supported."""
    out = []
    lgb_grid = [dict(num_leaves=15, learning_rate=0.05, n_estimators=300, min_child_samples=20),
                dict(num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=20),
                dict(num_leaves=7, learning_rate=0.1, n_estimators=200, min_child_samples=50),
                dict(num_leaves=31, learning_rate=0.02, n_estimators=600, min_child_samples=50, reg_lambda=5.0),
                dict(num_leaves=63, learning_rate=0.05, n_estimators=300, min_child_samples=10, colsample_bytree=0.8)]
    for c in lgb_grid:
        extra = dict(monotone_constraints=monotone, monotone_constraints_method="advanced") if monotone else {}
        out.append(("lightgbm", c, lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=threads,
                                                      deterministic=True, force_row_wise=True, verbose=-1,
                                                      **c, **extra)))
    for c in [dict(learning_rate=0.05, max_leaf_nodes=15, max_iter=300, l2_regularization=0.0),
              dict(learning_rate=0.1, max_leaf_nodes=31, max_iter=200, l2_regularization=1.0),
              dict(learning_rate=0.05, max_leaf_nodes=7, max_iter=400, l2_regularization=5.0)]:
        out.append(("hist_gradient_boosting", c, HistGradientBoostingClassifier(
            random_state=seed, early_stopping=False, monotonic_cst=monotone, **c)))
    for C in (0.01, 0.1, 1.0, 10.0):
        out.append(("logistic_regression", {"C": C}, make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(C=C, max_iter=2000))))
    if monotone is None:      # random forest has no monotone option; used for the score models only
        for leaf in (1, 5, 20):
            out.append(("random_forest", {"min_samples_leaf": leaf, "n_estimators": 400}, RandomForestClassifier(
                n_estimators=400, min_samples_leaf=leaf, random_state=seed, n_jobs=threads)))
    return out


def clone(est):
    return sklearn.base.clone(est)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def environment() -> dict:
    return {"python": platform.python_version(), "lightgbm": lgb.__version__, "sklearn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "platform": platform.platform()}


# ---------------------------------------------------------------------------
# 1-2. score models
# ---------------------------------------------------------------------------
def select_and_train_score(score, d, masks, groups, args, data_info, model_dir):
    tr, va = masks["train"], masks["val"]
    y = d["label"].to_numpy()
    feats = [c for c in family_columns(d.columns, score) if not (args.exclude_auth_features and c in AUTH_DERIVED)]
    sentinels = {k: v for k, v in SENTINELS.items() if k in feats}
    X = preprocess(d, feats, sentinels)
    Xtr, ytr, gtr = X[tr].reset_index(drop=True), y[tr], groups[tr]
    folds = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(Xtr, ytr, gtr))

    rows, best = [], None
    for algo, cfg, est in candidates(args.seed, args.threads):
        oof = np.zeros(len(ytr))
        for a, b in folds:
            oof[b] = clone(est).fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]
        ap = average_precision_score(ytr, oof)
        ll = log_loss(ytr, np.clip(oof, 1e-7, 1 - 1e-7))
        rows.append({"score": score, "algorithm": algo, "config": json.dumps(cfg), "cv_pr_auc": ap, "cv_log_loss": ll})
        if best is None or (ap, -ll) > (best["ap"], -best["ll"]):
            best = {"ap": ap, "ll": ll, "algo": algo, "cfg": cfg, "est": est, "oof": oof}

    model = clone(best["est"]).fit(Xtr, ytr)
    p_full = model.predict_proba(X)[:, 1]
    thr, rule = select_threshold(y[va], p_full[va], args.max_fpr)
    stack_input = p_full.copy()
    stack_input[tr] = best["oof"]
    art = {"model": model, "feature_columns": feats,
           "preprocessing": {"steps": ["select feature_columns in order", "to float64", "+/-inf -> NaN",
                                       "documented sentinels -> NaN"], "sentinels": sentinels},
           "algorithm": best["algo"], "config": best["cfg"], "sampling_strategy": "none",
           "threshold": thr, "threshold_rule": rule, "model_name": f"slice_{score.lower()}_final",
           "validation_metrics": full_metrics(y[va], p_full[va], thr),
           "training_config": {"hyperparameters": best["cfg"], "algorithm": best["algo"],
                               "selection": f"grouped {args.folds}-fold CV PR-AUC on TRAIN (groups = prov__template)",
                               "cv_pr_auc": best["ap"], "seed": args.seed, "data": data_info},
           "environment": environment()}
    joblib.dump(art, model_dir / f"slice_{score.lower()}_model.joblib")
    log.info("[%s] winner %s %s | CV PR-AUC %.4f | val PR-AUC %.4f", score, best["algo"], best["cfg"],
             best["ap"], art["validation_metrics"]["pr_auc"])
    return stack_input, p_full, pd.DataFrame(rows), art


# ---------------------------------------------------------------------------
# 3. final link model
# ---------------------------------------------------------------------------
def final_inputs(prob: dict, d: pd.DataFrame, extra_cols: list[str]) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({f"{s}_prob": prob[s] for s in LINK_SCORES}, index=d.index),
                      preprocess(d, extra_cols, {})], axis=1)


def select_final(M, y, masks, args, n_scores):
    tr, va = masks["train"], masks["val"]
    mono = [1] * n_scores + [0] * (M.shape[1] - n_scores)
    rows, best = [], None
    for algo, cfg, est in candidates(args.seed, args.threads, monotone=mono):
        if args.final_algos and algo not in args.final_algos:
            continue
        Xfit = M.copy()
        if algo == "logistic_regression":   # linear model works on the score logits
            Xfit.iloc[:, :n_scores] = logit(Xfit.iloc[:, :n_scores].to_numpy())
        m = clone(est).fit(Xfit[tr], y[tr])
        pv = m.predict_proba(Xfit[va])[:, 1]
        ap, ll = average_precision_score(y[va], pv), log_loss(y[va], np.clip(pv, 1e-7, 1 - 1e-7))
        rows.append({"algorithm": algo, "config": json.dumps(cfg), "val_pr_auc": ap, "val_log_loss": ll})
        if best is None or (ap, -ll) > (best["ap"], -best["ll"]):
            best = {"ap": ap, "ll": ll, "algo": algo, "cfg": cfg, "model": m,
                    "logit_inputs": algo == "logistic_regression"}
    return best, pd.DataFrame(rows)


def predict_link(best, M, n_scores):
    X = M.copy()
    if best["logit_inputs"]:
        X.iloc[:, :n_scores] = logit(X.iloc[:, :n_scores].to_numpy())
    return best["model"].predict_proba(X)[:, 1]


def combine(p_link, sts_risk, w):
    z = logit(p_link) + w * (np.asarray(sts_risk, dtype=float) - 0.5)
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "slice_dataset_v4.xlsx"))
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "final3"))
    ap.add_argument("--report-dir", default=str(ROOT / "reports" / "final3"))
    ap.add_argument("--excel-out", default=str(ROOT / "slice_training_dataset_final3.xlsx"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep-auth-features", dest="exclude_auth_features", action="store_false",
                    help=f"keep the DKIM-derived features {AUTH_DERIVED} in the link side (default: excluded)")
    ap.add_argument("--final-algos", nargs="*", default=["lightgbm"],
                    help="restrict the final model's candidate algorithms (e.g. lightgbm for exact per-feature reasons)")
    ap.add_argument("--final-variants", nargs="*", default=["infra", "none"], choices=["infra", "none"],
                    help="final-model input sets to compare on VAL: 'infra' = 4 scores + INFRA, 'none' = 4 scores only")
    ap.add_argument("--extra-train", nargs="*",
                    default=[str(ROOT / "private" / "nazario_train" / "train_rows.csv"),
                             str(ROOT / "private" / "v4" / "extra_legit_rows.csv"),
                             str(ROOT / "private" / "v4" / "unused_train_rows.csv")],
                    help="extra feature-row workbooks/CSVs added to the TRAIN split only (e.g. a second phishing source)")
    args = ap.parse_args(argv)

    model_dir, report_dir = Path(args.model_dir), Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
                            ROOT / "logs" / f"train_slice_final_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8")])
    t0 = time.time()

    data_path = Path(args.data)
    d = pd.read_excel(data_path, sheet_name="Dataset").reset_index(drop=True)
    data_info = {"file": data_path.name, "sha256": sha256_file(data_path), "rows": int(len(d)),
                 "exclude_auth_features": bool(args.exclude_auth_features), "extra_train": []}
    for extra_path in args.extra_train:
        e = pd.read_csv(extra_path) if str(extra_path).endswith(".csv") else pd.read_excel(extra_path)
        e["prov__time_split"] = "train"
        missing = [c for c in d.columns if c.startswith(("BAS__", "DMS__", "LOS__", "RCS__", "INFRA__")) and c not in e]
        if missing:
            raise ValueError(f"{extra_path}: missing feature columns {missing[:5]}")
        d = pd.concat([d, e[[c for c in e.columns if c in d.columns]]], ignore_index=True)
        data_info["extra_train"].append({"file": Path(extra_path).name, "rows": int(len(e)),
                                         "labels": e["label"].value_counts().to_dict()})
    masks = {s: (d["prov__time_split"] == s).to_numpy() for s in ("train", "val", "test")}
    y = d["label"].to_numpy()
    groups = d["prov__template"].fillna("").astype(str).to_numpy()
    log.info("rows %d | split x label %s", len(d), d.groupby(["prov__time_split", "label"]).size().to_dict())

    # 1-2. score models
    stack_in, full_pred, sel_tables, arts = {}, {}, [], {}
    for s in LINK_SCORES:
        stack_in[s], full_pred[s], tab, arts[s] = select_and_train_score(s, d, masks, groups, args, data_info, model_dir)
        sel_tables.append(tab)

    # 3. final link model: 4 scores + INFRA (as specified); +SRC reported for information only
    src = family_columns(d.columns, "SRC")
    variants = {"infra": family_columns(d.columns, "INFRA"), "none": []}
    best, final_tab, infra = None, [], None
    for v in args.final_variants:
        Mv = final_inputs(stack_in, d, variants[v])
        bv, tab = select_final(Mv, y, masks, args, len(LINK_SCORES))
        final_tab.append(tab.assign(inputs=v))
        log.info("[FINAL variant %s] best %s val PR-AUC %.4f", v, bv["algo"], bv["ap"])
        if best is None or (bv["ap"], -bv["ll"]) > (best["ap"], -best["ll"]):
            best, infra, M, chosen_variant = bv, variants[v], Mv, v
    final_tab = pd.concat(final_tab)
    raw = predict_link(best, M, len(LINK_SCORES))
    cal = LogisticRegression(class_weight="balanced", C=1e6, max_iter=1000).fit(
        logit(raw[masks["val"]]).reshape(-1, 1), y[masks["val"]])
    link_prob = cal.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
    link_thr, link_rule = select_threshold(y[masks["val"]], link_prob[masks["val"]], args.max_fpr)

    # 4. STS rule (not trained) + fixed combination
    sts = sts_rule_score(d)
    final_prob = combine(link_prob, sts["STS_rule_risk"].to_numpy(), STS_WEIGHT)
    final_thr, final_rule = select_threshold(y[masks["val"]], final_prob[masks["val"]], args.max_fpr)
    sts_thr, sts_rule = select_threshold(y[masks["val"]], sts["STS_rule_risk"].to_numpy()[masks["val"]], args.max_fpr)

    final_art = {
        "model": best["model"], "calibrator": cal, "logit_score_inputs": best["logit_inputs"],
        "algorithm": best["algo"], "config": best["cfg"],
        "score_inputs": list(LINK_SCORES), "sts_mode": "rule",
        "input_columns": list(M.columns), "extra_feature_columns": infra, "final_inputs_variant": chosen_variant,
        "monotone_constraints": [1] * len(LINK_SCORES) + [0] * len(infra),
        "sts_combination": {"mode": "fixed_weight_after_ml", "weight": STS_WEIGHT, "center": 0.5,
                            "formula": "final log-odds = calibrated link log-odds + weight * (STS_rule_risk - 0.5)",
                            "note": "weight fixed before training, not tuned; the RL combiner learns from deployment feedback"},
        "calibration": "class-balanced Platt on VAL (probability under a 50/50 prior)",
        "link_threshold": link_thr, "threshold": final_thr, "threshold_rule": final_rule,
        "model_name": "slice_final_link_model", "validation_metrics": full_metrics(
            y[masks["val"]], final_prob[masks["val"]], final_thr),
        "training_config": {"selection": "VAL PR-AUC (tie -> VAL log loss)",
                            "train_inputs": "out-of-fold score probabilities on TRAIN", "seed": args.seed,
                            "data": data_info},
        "environment": environment(),
    }
    joblib.dump(final_art, model_dir / "slice_final_model.joblib")
    log.info("[FINAL LINK] winner %s %s | val PR-AUC %.4f", best["algo"], best["cfg"], best["ap"])

    # information only: +SRC
    M_src = final_inputs(stack_in, d, infra + src)
    best_src, _ = select_final(M_src, y, masks, args, len(LINK_SCORES))
    raw_src = predict_link(best_src, M_src, len(LINK_SCORES))
    log.info("all artifacts saved; scoring TEST once")

    # 5. TEST once
    te = masks["test"]
    res = []
    def add(name, p, thr, kind):
        r = full_metrics(y[te], p[te], thr) if thr is not None else {**full_metrics(y[te], p[te], 0.5), "threshold": np.nan}
        res.append({"model": name, "kind": kind, **r})
    for s in LINK_SCORES:
        add(s, full_pred[s], arts[s]["threshold"], f"score model ({arts[s]['algorithm']})")
    add("STS (rule)", sts["STS_rule_risk"].to_numpy(), sts_thr, "rule, not trained")
    add(f"LINK MODEL (4 scores{' + INFRA' if infra else ''})", link_prob, link_thr, f"final link model ({best['algo']})")
    add("FINAL (link + fixed STS w=3)", final_prob, final_thr, "deployment day-one decision")
    add("info: link model + SRC", raw_src, None, f"not saved ({best_src['algo']})")
    res = pd.DataFrame(res)[["model", "kind", "n", "n_pos", "n_neg", "roc_auc", "pr_auc", "log_loss", "threshold",
                             "precision", "recall", "f1", "fpr", "fnr", "tn", "fp", "fn", "tp"]]
    log.info("TEST results:\n%s", res[["model", "roc_auc", "pr_auc", "recall", "fpr", "fp", "fn"]].to_string(index=False))

    # reports + workbook
    sel = pd.concat(sel_tables)
    sel.to_csv(report_dir / "score_method_selection_cv.csv", index=False)
    final_tab.to_csv(report_dir / "final_method_selection_val.csv", index=False)
    res.to_csv(report_dir / "test_results.csv", index=False)
    winners = pd.DataFrame([{"model": s, "algorithm": arts[s]["algorithm"], "config": json.dumps(arts[s]["config"]),
                             "cv_pr_auc": arts[s]["training_config"]["cv_pr_auc"],
                             "val_pr_auc": arts[s]["validation_metrics"]["pr_auc"], "threshold": arts[s]["threshold"]}
                            for s in LINK_SCORES] +
                           [{"model": "LINK MODEL", "algorithm": best["algo"], "config": json.dumps(best["cfg"]),
                             "cv_pr_auc": np.nan, "val_pr_auc": best["ap"], "threshold": link_thr},
                            {"model": "STS (rule)", "algorithm": "fixed rules", "config": json.dumps(RULE_WEIGHTS),
                             "cv_pr_auc": np.nan, "val_pr_auc": np.nan, "threshold": sts_thr}])
    winners.to_csv(report_dir / "chosen_methods.csv", index=False)
    info = d[["email_id", "label", "prov__time_split", "prov__template"]].rename(columns={"prov__time_split": "split"})
    final_input = pd.concat([info, M.round(6), sts[["STS_rule", "STS_rule_risk", "STS_rule_reasons"]]], axis=1)
    final_input.insert(4, "score_input_source", np.where(masks["train"], "out-of-fold", "score model (trained on train)"))
    final_input["LINK_prob"] = np.where(masks["train"], np.nan, link_prob.round(6))
    final_input["FINAL_prob"] = np.where(masks["train"], np.nan, final_prob.round(6))
    with pd.ExcelWriter(args.excel_out, engine="openpyxl") as xw:
        pd.DataFrame({"item": ["Data", "Score models", "Final link model", "STS", "Combination", "Test"],
                      "description": [f"{data_path.name}, {len(d)} emails, time-ordered split",
                                      "BAS, DMS, LOS, RCS: method chosen by grouped CV on train (groups = sender template)",
                                      "4 score probabilities (out-of-fold on train) + INFRA features; method chosen on val; Platt-calibrated",
                                      "rule-based (slice_models/sts_rules.py), weights fixed a priori, not trained",
                                      f"final log-odds = logit(link) + {STS_WEIGHT:g} * (STS risk - 0.5); RL combiner learns in deployment",
                                      "scored once after all artifacts were saved"]}).to_excel(xw, sheet_name="README", index=False)
        winners.to_excel(xw, sheet_name="Chosen_Methods", index=False)
        sel.to_excel(xw, sheet_name="Score_Method_Selection", index=False)
        final_tab.to_excel(xw, sheet_name="Final_Method_Selection", index=False)
        final_input.to_excel(xw, sheet_name="Final_Model_Input", index=False)
        res.to_excel(xw, sheet_name="Test_Results", index=False)
    log.info("done in %.0fs | models in %s", time.time() - t0, model_dir)


if __name__ == "__main__":
    main()
