"""
Reload the five saved SLiCE artifacts from disk and smoke-test them on real
held-out (test-split) rows.

    python -m slice_models.verify_slice_models [--n 200]

Checks per model: loads; required keys present; feature columns present in the
data and all from its own family; model was fitted on exactly those columns in
that order; predictions finite, in [0, 1], correct shape; reloaded predictions
match the ones recorded during training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slice_models.slice_predict import SCORES, load_slice_model, model_path, predict_all_scores, predict_score

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_KEYS = {"model", "feature_columns", "preprocessing", "sampling_strategy", "threshold", "model_name",
                 "model_version", "validation_metrics", "training_config"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="held-out test rows to score")
    ap.add_argument("--report-dir", default=str(ROOT / "reports" / "slice_five_models"))
    ap.add_argument("--model-dir", default=str(ROOT / "models"))
    args = ap.parse_args(argv)
    md = args.model_dir
    rep = Path(args.report_dir)

    split = pd.read_csv(rep / "slice_split_assignments.csv")
    test_ids = split.loc[split["split"] == "test"]
    sample_ids = pd.concat([test_ids[test_ids.label == 0].sample(args.n // 2, random_state=0),
                            test_ids[test_ids.label == 1].sample(args.n - args.n // 2, random_state=0)])["email_id"]
    data = pd.concat([pd.read_excel(ROOT / f, sheet_name="Dataset") for f in
                      ("phishing_email_dataset.xlsx", "phishing_pot_dataset.xlsx")], ignore_index=True)
    rows = data.set_index("email_id").loc[sample_ids].reset_index()
    recorded = pd.read_csv(rep / "slice_test_predictions.csv").set_index("email_id")

    all_ok = True
    for score in SCORES:
        checks = {}
        art = load_slice_model(score, md, use_cache=False)
        checks["file exists"] = model_path(score, md).exists()
        checks["required keys"] = REQUIRED_KEYS <= set(art)
        feats = art["feature_columns"]
        checks["features present in data"] = all(c in rows.columns for c in feats)
        checks["only own-family features"] = all(c.startswith(f"{score}__") for c in feats)
        checks["model fitted on exactly these columns/order"] = list(art["model"].feature_name_) == feats
        checks["n_features matches"] = art["model"].n_features_in_ == len(feats) == art["n_features"]
        # extra foreign columns in the input must be ignored, not used
        p, flag = predict_score(rows, score, md, return_flag=True)
        checks["shape"] = p.shape == (len(rows),) and flag.shape == (len(rows),)
        checks["finite"] = bool(np.isfinite(p).all())
        checks["in [0,1]"] = bool(((p >= 0) & (p <= 1)).all())
        rec = recorded.loc[rows["email_id"], score].to_numpy()
        m = ~np.isnan(rec)
        checks["matches training-time predictions"] = bool(np.allclose(p[m], rec[m], atol=1e-9))
        ok = all(checks.values())
        all_ok &= ok
        print(f"{score}: {'PASS' if ok else 'FAIL'}  ({art['model_name']} v{art['model_version']}, "
              f"{len(feats)} features, sampling={art['sampling_strategy']}, threshold={art['threshold']:.4f}, "
              f"mean score legit={p[rows.label.to_numpy() == 0].mean():.3f} phish={p[rows.label.to_numpy() == 1].mean():.3f})")
        for k, v in checks.items():
            if not v:
                print(f"   FAILED: {k}")

    out = predict_all_scores(rows, md)
    print(f"\npredict_all_scores -> {out.shape}, columns {list(out.columns)}")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
