"""
Train five independent LightGBM models, one per SLiCE score family
(STS, BAS, DMS, LOS, RCS), from the existing feature workbooks.

Run from the project root:
    python -m slice_models.train_slice_models

Workflow (all decisions are made on TRAIN + VALIDATION only):
  1. Load both `Dataset` sheets, check schema/labels, de-duplicate.
  2. One stratified 70/15/15 split (seed 42) shared by all five models.
  3. Per family: original / random-undersample 1:1 / random-oversample 1:1
     (training split only) x the same hyper-parameter candidates.
  4. Select config + sampling on validation PR-AUC; select threshold on
     validation (max recall s.t. FPR <= --max-fpr).
  5. Save artifacts. Only then score the untouched test split once.

Read-only on the original workbooks and feature modules.
"""
from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):  # allow `python slice_models/train_slice_models.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slice_models.slice_predict import SCORES, SENTINELS, family_columns, model_path, preprocess

ROOT = Path(__file__).resolve().parent.parent
MODEL_VERSION = "1.0.0"
SAMPLING_STRATEGIES = ("none", "undersample", "oversample")
SAMPLING_ORDER = {s: i for i, s in enumerate(SAMPLING_STRATEGIES)}  # tie-break: simplest first

# Columns that are never predictive inputs (identifiers, labels, bookkeeping).
NON_FEATURE_PREFIXES = ("STS_meta__", "BAS_meta__", "DMS_meta__", "LOS_meta__", "RCS_meta__",
                        "SRCINFRA_meta__", "lookups_")
NON_FEATURE_COLUMNS = {"email_id", "label", "processing_status", "missing_external_lookup"}

log = logging.getLogger("slice")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(legit_path: Path, phish_path: Path) -> tuple[pd.DataFrame, dict]:
    info = {}
    frames = []
    for path, expected_label, source in ((legit_path, 0, "gmail_inbox"), (phish_path, 1, "phishing_pot")):
        t = time.time()
        df = pd.read_excel(path, sheet_name="Dataset")
        log.info("loaded %s: %s rows x %s cols (%.1fs)", path.name, *df.shape, time.time() - t)
        labels = set(df["label"].unique())
        if labels != {expected_label}:
            raise ValueError(f"{path.name}: expected only label {expected_label}, found {labels}")
        df["_source"] = source
        frames.append(df)
        info[source] = {"file": path.name, "sha256": sha256_file(path), "rows": int(len(df)),
                        "processing_status": df["processing_status"].value_counts().to_dict()}
    a, b = frames
    if list(a.columns) != list(b.columns):
        raise ValueError(f"schema mismatch: {set(a.columns) ^ set(b.columns)}")
    df = pd.concat(frames, ignore_index=True)

    # failed emails carry no features
    failed = df["processing_status"].astype(str).str.startswith("failed")
    info["dropped_failed"] = int(failed.sum())
    df = df[~failed]

    # De-duplication. Both builders already removed duplicate *emails* by raw
    # SHA-256 or Message-ID (phishing_pot: 236 removed). Hashes for the
    # phishing rows are not available locally, so here we can only guarantee
    # email_id uniqueness. Identical feature vectors are NOT treated as
    # duplicate emails (families have few distinct values; different emails
    # legitimately share vectors).
    dup = df.duplicated("email_id", keep="first")
    info["dropped_duplicate_email_id"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    info["rows_after_cleaning"] = int(len(df))
    info["label_counts"] = df["label"].value_counts().sort_index().to_dict()
    return df, info


def split_data(df: pd.DataFrame, seed: int) -> pd.Series:
    """Stratified 70/15/15 split; returns a Series of 'train'/'val'/'test'."""
    idx = df.index.to_numpy()
    tr, rest = train_test_split(idx, test_size=0.30, stratify=df["label"], random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=df.loc[rest, "label"], random_state=seed)
    split = pd.Series("", index=df.index, dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    return split


def family_failed_mask(df: pd.DataFrame, score: str) -> pd.Series:
    """Rows where the builder reports this family's module failed (e.g. 'partial:RCS,SRCINFRA').
    Their features are all-NaN; keeping them would let the model learn
    'missing => phishing' from 5 phishing-only failures."""
    st = df["processing_status"].astype(str)
    return st.str.startswith("partial:") & st.str.split(":").str[1].fillna("").str.split(",").apply(
        lambda mods: score in mods)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def resample(X: pd.DataFrame, y: np.ndarray, strategy: str, seed: int):
    """Random under/over-sampling to 1:1 on the TRAINING split only."""
    if strategy == "none":
        return X, y
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    if strategy == "undersample":
        keep = np.concatenate([minority, rng.choice(majority, size=len(minority), replace=False)])
    elif strategy == "oversample":
        extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
        keep = np.concatenate([majority, minority, extra])
    else:
        raise ValueError(strategy)
    keep = rng.permutation(keep)
    return X.iloc[keep].reset_index(drop=True), y[keep]


# ---------------------------------------------------------------------------
# hyper-parameters
# ---------------------------------------------------------------------------
BASELINE = dict(num_leaves=31, max_depth=-1, learning_rate=0.05, min_child_samples=20,
                subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0, reg_alpha=0.0)
SEARCH_SPACE = dict(num_leaves=[7, 15, 31, 63], max_depth=[-1, 4, 6, 8],
                    learning_rate=[0.02, 0.05, 0.1], min_child_samples=[10, 20, 50, 100],
                    subsample=[0.7, 0.85, 1.0], colsample_bytree=[0.7, 0.85, 1.0],
                    reg_lambda=[0.0, 1.0, 5.0, 10.0], reg_alpha=[0.0, 0.1, 1.0])


def candidate_configs(n_random: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs, seen = [dict(BASELINE)], {json.dumps(BASELINE, sort_keys=True)}
    while len(configs) < n_random + 1:
        c = {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}
        c = {k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()}
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(c)
    return configs


def make_model(cfg: dict, args) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", metric="average_precision", n_estimators=args.max_estimators,
        subsample_freq=1 if cfg["subsample"] < 1.0 else 0, random_state=args.seed,
        n_jobs=args.threads, deterministic=True, force_row_wise=True, verbose=-1, **cfg)


# ---------------------------------------------------------------------------
# metrics & threshold
# ---------------------------------------------------------------------------
def select_threshold(y: np.ndarray, p: np.ndarray, max_fpr: float) -> tuple[float, str]:
    """Objective: maximise phishing recall subject to validation FPR <= max_fpr.
    Threshold is placed midway between the chosen score and the next lower
    distinct score (same validation outcome, less brittle)."""
    fpr, tpr, thr = roc_curve(y, p, drop_intermediate=False)
    ok = np.flatnonzero(fpr <= max_fpr)
    best = ok[np.argmax(tpr[ok])]
    if tpr[best] == 0:  # constraint unattainable with any positive recall -> Youden's J
        best = int(np.argmax(tpr - fpr))
        rule = f"fallback_youden_j (no threshold reached FPR<={max_fpr} with recall>0)"
    else:
        rule = f"max_recall_s.t._fpr<={max_fpr}"
    t = float(thr[best])
    lower = np.unique(p[p < t])
    if lower.size:
        t = float((t + lower[-1]) / 2)
    return min(t, 1.0), rule


def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                fpr=fp / (fp + tn) if fp + tn else float("nan"),
                fnr=fn / (fn + tp) if fn + tp else float("nan"))


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def full_metrics(y, p, t) -> dict:
    return {**ranking_metrics(y, p), **threshold_metrics(y, p, t), "threshold": t, "n": int(len(y)),
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}


# ---------------------------------------------------------------------------
# per-family training
# ---------------------------------------------------------------------------
def train_family(score, df, split, args, configs, out_dir: Path):
    feats = family_columns(df.columns, score)
    assert feats and all(c.startswith(f"{score}__") for c in feats)
    assert not set(feats) & NON_FEATURE_COLUMNS and not any(c.startswith(NON_FEATURE_PREFIXES) for c in feats)
    sentinels = {k: v for k, v in SENTINELS.items() if k in feats}

    excluded = family_failed_mask(df, score)
    sub = df[~excluded]
    parts = {s: sub[split[sub.index] == s] for s in ("train", "val", "test")}
    X = {s: preprocess(d, feats, sentinels) for s, d in parts.items()}
    y = {s: d["label"].to_numpy(dtype=int) for s, d in parts.items()}

    # "Fitted" preprocessing state: training-split statistics only. The
    # transform itself is stateless (LightGBM handles NaN natively, no
    # imputation/scaling needed for trees), so nothing learnt from val/test.
    xt = X["train"]
    train_stats = {c: dict(missing_rate=float(xt[c].isna().mean()), min=float(np.nanmin(xt[c])) if xt[c].notna().any() else None,
                           max=float(np.nanmax(xt[c])) if xt[c].notna().any() else None, n_unique=int(xt[c].nunique()),
                           constant_in_train=bool(xt[c].nunique(dropna=True) <= 1)) for c in feats}
    sentinel_counts = {c: {s: int(parts[s][c].isin(v).sum()) for s in parts} for c, v in sentinels.items()}
    log.info("[%s] %d features | excluded rows (module failed) %d | train %d val %d test %d",
             score, len(feats), int(excluded.sum()), len(y["train"]), len(y["val"]), len(y["test"]))
    if sentinel_counts:
        log.info("[%s] sentinel -> NaN counts: %s", score, sentinel_counts)

    # ---- search: sampling strategy x config, scored on validation only
    rows, best_by_strategy = [], {}
    for strategy in SAMPLING_STRATEGIES:
        Xs, ys = resample(X["train"], y["train"], strategy, args.seed)
        for ci, cfg in enumerate(configs):
            m = make_model(cfg, args)
            m.fit(Xs, ys, eval_set=[(X["val"], y["val"])],
                  callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)])
            pv = m.predict_proba(X["val"])[:, 1]
            r = ranking_metrics(y["val"], pv)
            rows.append(dict(score=score, sampling=strategy, config_id=ci, best_iteration=int(m.best_iteration_ or args.max_estimators),
                             val_roc_auc=r["roc_auc"], val_pr_auc=r["pr_auc"], val_log_loss=r["log_loss"], **cfg))
            key = (round(r["pr_auc"], 4), -r["log_loss"])  # PR-AUC, tie-break lower log loss
            if strategy not in best_by_strategy or key > best_by_strategy[strategy]["key"]:
                best_by_strategy[strategy] = dict(key=key, model=m, cfg=cfg, config_id=ci, pv=pv,
                                                  n_train=len(ys), n_train_pos=int(ys.sum()))
    search = pd.DataFrame(rows)

    # ---- compare strategies (each at its best config + own validation threshold)
    comp = []
    for strategy, b in best_by_strategy.items():
        t, rule = select_threshold(y["val"], b["pv"], args.max_fpr)
        mt = full_metrics(y["val"], b["pv"], t)
        b.update(threshold=t, threshold_rule=rule, val_metrics=mt)
        comp.append(dict(score=score, sampling=strategy, config_id=b["config_id"], n_train_after_sampling=b["n_train"],
                         n_train_pos_after_sampling=b["n_train_pos"], best_iteration=int(b["model"].best_iteration_ or 0),
                         **{f"val_{k}": v for k, v in mt.items()}))
    comp = pd.DataFrame(comp)

    # Selection rule: highest validation PR-AUC; within --tie-tol prefer higher
    # recall at the FPR-constrained threshold, then the simpler strategy
    # (none < undersample < oversample). Balanced training is NOT assumed better.
    top = comp["val_pr_auc"].max()
    cand = comp[comp["val_pr_auc"] >= top - args.tie_tol].copy()
    cand["_ord"] = cand["sampling"].map(SAMPLING_ORDER)
    chosen = cand.sort_values(["val_recall", "_ord"], ascending=[False, True]).iloc[0]["sampling"]
    comp["selected"] = comp["sampling"] == chosen
    b = best_by_strategy[chosen]
    model = b["model"]
    log.info("[%s] validation by strategy:\n%s", score, comp[["sampling", "config_id", "val_pr_auc", "val_roc_auc",
             "val_recall", "val_precision", "val_fpr", "val_fp", "val_fn", "val_threshold"]].to_string(index=False))
    log.info("[%s] selected sampling=%s config=%d threshold=%.4f", score, chosen, b["config_id"], b["threshold"])

    # ---- diagnostics on validation (never used for selection)
    diags = {}
    if score == "STS" and "STS__new_sender_domain" in feats:
        # ablation: how much does the sentinel-heavy feature contribute?
        f2 = [c for c in feats if c != "STS__new_sender_domain"]
        Xs, ys = resample(X["train"][f2], y["train"], chosen, args.seed)
        m2 = make_model(b["cfg"], args).fit(Xs, ys, eval_set=[(X["val"][f2], y["val"])],
                                             callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)])
        diags["val_without_new_sender_domain"] = ranking_metrics(y["val"], m2.predict_proba(X["val"][f2])[:, 1])
        diags["val_with_new_sender_domain"] = ranking_metrics(y["val"], b["pv"])
        log.info("[STS] ablation (validation): with new_sender_domain PR-AUC=%.4f, without=%.4f",
                 diags["val_with_new_sender_domain"]["pr_auc"], diags["val_without_new_sender_domain"]["pr_auc"])

    # ---- feature importance (train-fitted model; SHAP on validation)
    booster = model.booster_
    contrib = booster.predict(X["val"], num_iteration=model.best_iteration_, pred_contrib=True)
    fi = pd.DataFrame({"feature": feats,
                       "gain": booster.feature_importance("gain", iteration=model.best_iteration_),
                       "split": booster.feature_importance("split", iteration=model.best_iteration_),
                       "mean_abs_shap_val": np.abs(contrib[:, :-1]).mean(axis=0),
                       "train_missing_rate": [train_stats[c]["missing_rate"] for c in feats],
                       "constant_in_train": [train_stats[c]["constant_in_train"] for c in feats]})
    fi["gain_share"] = fi["gain"] / fi["gain"].sum() if fi["gain"].sum() else 0.0
    fi = fi.sort_values("gain", ascending=False).reset_index(drop=True)
    fi.insert(0, "rank", np.arange(1, len(fi) + 1))
    fi_path = out_dir / "feature_importance" / f"slice_{score.lower()}_feature_importance.csv"
    fi.to_csv(fi_path, index=False)

    # ---- artifact (saved BEFORE the test split is scored)
    artifact = dict(
        model=model, model_name=f"SLiCE-{score}-LightGBM", model_version=MODEL_VERSION, score_family=score,
        feature_columns=list(feats), n_features=len(feats),
        preprocessing=dict(steps=["select feature_columns in stored order", "coerce to float64 (non-numeric -> NaN)",
                                  "+/-inf -> NaN", "documented sentinels -> NaN", "NaN handled natively by LightGBM"],
                           sentinels=sentinels, fitted_on="train split only", train_feature_stats=train_stats,
                           function="slice_models.slice_predict.preprocess"),
        sampling_strategy=chosen, threshold=b["threshold"], threshold_rule=b["threshold_rule"],
        positive_class=1, output="predict_proba(X)[:, 1] = phishing model score (uncalibrated)",
        calibration=None, validation_metrics=b["val_metrics"],
        validation_sampling_comparison=comp.to_dict("records"), diagnostics=diags,
        training_config=dict(hyperparameters=b["cfg"], config_id=b["config_id"], best_iteration=int(model.best_iteration_ or 0),
                             lgbm_params=model.get_params(), seed=args.seed, max_fpr=args.max_fpr, tie_tol=args.tie_tol,
                             n_candidate_configs=len(configs), early_stopping_rounds=args.early_stopping,
                             early_stopping_metric="average_precision on validation", split="stratified 70/15/15",
                             n_train=int(len(y["train"])), n_train_after_sampling=b["n_train"],
                             n_train_pos_after_sampling=b["n_train_pos"], n_val=int(len(y["val"])), n_test=int(len(y["test"])),
                             rows_excluded_module_failed=int(excluded.sum())),
        environment=dict(python=platform.python_version(), lightgbm=lgb.__version__, sklearn=sklearn.__version__,
                         pandas=pd.__version__, numpy=np.__version__),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return artifact, search, comp, X, y, parts


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legit", default=str(ROOT / "phishing_email_dataset.xlsx"))
    ap.add_argument("--phish", default=str(ROOT / "phishing_pot_dataset.xlsx"))
    ap.add_argument("--model-dir", default=str(ROOT / "models"))
    ap.add_argument("--report-dir", default=str(ROOT / "reports" / "slice_five_models"))
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-configs", type=int, default=24, help="random configs in addition to the baseline")
    ap.add_argument("--max-estimators", type=int, default=3000)
    ap.add_argument("--early-stopping", type=int, default=100)
    ap.add_argument("--max-fpr", type=float, default=0.01, help="threshold objective: max validation FPR")
    ap.add_argument("--tie-tol", type=float, default=0.001, help="PR-AUC tolerance for sampling tie-break")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    model_dir, rep, log_dir = Path(args.model_dir), Path(args.report_dir), Path(args.log_dir)
    for d in (model_dir, rep, rep / "feature_importance", log_dir):
        d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(log_dir / f"train_slice_models_{stamp}.log", encoding="utf-8")])
    t0 = time.time()

    df, data_info = load_data(Path(args.legit), Path(args.phish))
    log.info("data: %s", json.dumps({k: v for k, v in data_info.items()}, default=str))
    split = split_data(df, args.seed)
    log.info("split counts:\n%s", pd.crosstab(split, df["label"]).to_string())

    # split assignment (email_id is a non-personal running id) for reproducibility/verification
    pd.DataFrame({"email_id": df["email_id"], "label": df["label"], "source": df["_source"], "split": split}).to_csv(
        rep / "slice_split_assignments.csv", index=False)

    configs = candidate_configs(args.n_configs, args.seed)
    artifacts, searches, comps, held = {}, [], [], {}
    for score in SCORES:
        art, search, comp, X, y, parts = train_family(score, df, split, args, configs, rep)
        art["training_config"]["data"] = data_info
        joblib.dump(art, model_path(score, model_dir), compress=3)
        log.info("[%s] saved %s", score, model_path(score, model_dir))
        artifacts[score], held[score] = art, (X, y, parts)
        searches.append(search)
        comps.append(comp)
    pd.concat(searches).to_csv(rep / "slice_hyperparameter_search_validation.csv", index=False)
    pd.concat(comps).to_csv(rep / "slice_sampling_comparison_validation.csv", index=False)

    # ------------------------------------------------------------------
    # All models, sampling choices and thresholds are now frozen and saved.
    # The test split is scored exactly once below.
    # ------------------------------------------------------------------
    log.info("all selections frozen; evaluating untouched test split")
    train_domains = set(df.loc[split == "train", "STS_meta__from_domain"].dropna().str.lower())
    results, diag_rows, test_preds = [], [], {}
    for score in SCORES:
        art = artifacts[score]
        X, y, parts = held[score]
        p_te = art["model"].predict_proba(X["test"])[:, 1]
        test_preds[score] = pd.Series(p_te, index=parts["test"].index)
        mt = full_metrics(y["test"], p_te, art["threshold"])
        mv, tc = art["validation_metrics"], art["training_config"]
        results.append(dict(
            score=score, model_file=model_path(score, model_dir).name, n_features=art["n_features"],
            sampling_strategy=art["sampling_strategy"], threshold=art["threshold"],
            val_roc_auc=mv["roc_auc"], val_pr_auc=mv["pr_auc"], val_recall=mv["recall"], val_fpr=mv["fpr"],
            test_roc_auc=mt["roc_auc"], test_pr_auc=mt["pr_auc"], test_accuracy=mt["accuracy"],
            test_precision=mt["precision"], test_recall=mt["recall"], test_f1=mt["f1"],
            test_tn=mt["tn"], test_fp=mt["fp"], test_fn=mt["fn"], test_tp=mt["tp"],
            test_fpr=mt["fpr"], test_fnr=mt["fnr"], n_train=tc["n_train"], n_train_after_sampling=tc["n_train_after_sampling"],
            n_val=tc["n_val"], n_test=tc["n_test"], test_pos_rate=mt["n_pos"] / mt["n"]))

        # leakage diagnostic: emails whose sender domain never appears in training
        for s in ("val", "test"):
            dom = parts[s]["STS_meta__from_domain"].str.lower()
            unseen = (dom.notna() & ~dom.isin(train_domains)).to_numpy()
            p = art["model"].predict_proba(X[s])[:, 1]
            yy = y[s][unseen]
            row = dict(score=score, split=s, subset="sender_domain_not_in_train", n=int(unseen.sum()),
                       n_pos=int(yy.sum()), n_neg=int((1 - yy).sum()))
            if len(np.unique(yy)) == 2:
                row.update({k: v for k, v in full_metrics(yy, p[unseen], art["threshold"]).items()
                            if k in ("roc_auc", "pr_auc", "recall", "fpr", "precision")})
            diag_rows.append(row)

    res = pd.DataFrame(results).sort_values("test_pr_auc", ascending=False).reset_index(drop=True)
    res.insert(0, "rank_by_test_pr_auc", np.arange(1, len(res) + 1))
    res.to_csv(rep / "slice_five_models_results.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(rep / "slice_leakage_diagnostics.csv", index=False)
    tp = pd.DataFrame(test_preds)
    tp.insert(0, "label", df.loc[tp.index, "label"])
    tp.insert(0, "email_id", df.loc[tp.index, "email_id"])
    tp.to_csv(rep / "slice_test_predictions.csv", index=False)

    pd.set_option("display.width", 250)
    log.info("FINAL (ranked by test PR-AUC):\n%s", res[["rank_by_test_pr_auc", "score", "n_features", "sampling_strategy",
             "threshold", "val_pr_auc", "test_roc_auc", "test_pr_auc", "test_recall", "test_fpr", "test_precision",
             "test_f1", "test_tn", "test_fp", "test_fn", "test_tp"]].to_string(index=False))
    log.info("done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
