"""Quick before/after diagnostics for slice_dataset_v3 (LightGBM defaults, not tuned)."""
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
d = pd.read_excel(sys.argv[1], sheet_name="Dataset")
fam = lambda f: [c for c in d.columns if c.startswith(f + "__")]
mdl = lambda: LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=4, random_state=0)


def cv(name, X, y):
    p = cross_val_predict(mdl(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print(f"  {name:48s} n={len(y):5d} legit={int((y == 0).sum()):5d}  ROC={roc_auc_score(y, p):.4f}  PR={average_precision_score(y, p):.4f}")


def temporal(name, cols, sub):
    tr, te = sub[sub.prov__time_split == "train"], sub[sub.prov__time_split == "test"]
    m = mdl().fit(tr[cols].astype(float), tr.label)
    p = m.predict_proba(te[cols].astype(float))[:, 1]
    # precision at 1% prevalence from test TPR/FPR at the threshold giving FPR<=1%
    neg = np.sort(p[te.label.values == 0])[::-1]
    thr = neg[max(0, int(0.01 * len(neg)) - 1)] if len(neg) else 0.5
    tpr = (p[te.label.values == 1] > thr).mean(); fpr = (p[te.label.values == 0] > thr).mean()
    prec1 = (0.01 * tpr) / (0.01 * tpr + 0.99 * fpr) if (tpr + fpr) > 0 else float("nan")
    print(f"  {name:48s} train={len(tr):5d} test={len(te):4d} (legit {int((te.label == 0).sum())})  "
          f"ROC={roc_auc_score(te.label, p):.4f}  PR={average_precision_score(te.label, p):.4f}  "
          f"recall@FPR1%={tpr:.3f}  precision@1%prev={prec1:.3f}")


y = d.label.values
print("A. STS shortcut check, all rows (5-fold CV)")
cv("received_hops RAW only", d[["STS_raw__received_hops"]], y)
cv("received_hops RECEIVER-NORMALISED only", d[["STS__received_hops"]], y)
raw_sts = [c for c in fam("STS") if c not in ("STS__received_hops", "STS__unusual_received_chain")] + ["STS_raw__received_hops", "STS_raw__unusual_received_chain"]
cv("STS with RAW received chain", d[raw_sts], y)
cv("STS receiver-normalised", d[fam("STS")], y)

core = d[d.prov__core == 1]
print("\nB. Core subset, time-ordered split (train on earlier mail, test on later mail)")
print("   core counts:", core.label.value_counts().to_dict(), "split:", core.groupby(["prov__time_split", "label"]).size().to_dict())
for f in ["STS", "BAS", "DMS", "LOS", "RCS", "SRC", "INFRA"]:
    temporal(f, fam(f), core)
temporal("ALL families", sum((fam(f) for f in ["STS", "BAS", "DMS", "LOS", "RCS", "SRC", "INFRA"]), []), core)
hard = core[(core.label == 0) | (core.prov__ms_delivery_folder == "inbox")]
print("\nC. Hard positives only (phishing Microsoft delivered to the inbox) vs legit, same split")
temporal("ALL families", sum((fam(f) for f in ["STS", "BAS", "DMS", "LOS", "RCS", "SRC", "INFRA"]), []), hard)
