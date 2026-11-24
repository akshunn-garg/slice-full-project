# SLiCE — five independent score models (STS, BAS, DMS, LOS, RCS)

One LightGBM binary classifier per SLiCE score family, trained on the existing
`Dataset` sheets. Feature modules, builders and the two workbooks are untouched
(read-only).

## Run

```bash
# from the project root (phish_dataset_v2/); deps: lightgbm, scikit-learn, pandas, openpyxl, joblib
python -m slice_models.train_slice_models      # ~17 min on 4 threads
python -m slice_models.verify_slice_models     # reload + smoke-test the 5 artifacts
```

Useful options: `--max-fpr 0.01` (threshold objective), `--n-configs 24`, `--seed 42`, `--threads 4`.

## Use

```python
import pandas as pd
from slice_models.slice_predict import predict_score, predict_all_scores, load_slice_model

df = pd.read_excel("phishing_pot_dataset.xlsx", sheet_name="Dataset")   # any frame with the family columns
p_sts = predict_score(df, "STS")                         # phishing score in [0,1] = predict_proba[:, 1]
p, flag = predict_score(df, "RCS", return_flag=True)     # flag uses the validation-selected threshold
all5 = predict_all_scores(df)                            # STS_prob, STS_flag, ..., RCS_prob, RCS_flag
art = load_slice_model("LOS")                            # dict: model, feature_columns, threshold, ...
```

Each model reads only its own `<FAMILY>__` columns (extra columns are ignored).
The scores are **uncalibrated model scores**, not real-world probabilities
(undersampling/oversampling also shifts them); no calibration was fitted.

## Layout

| Path | Content |
|---|---|
| `models/slice_{sts,bas,dms,los,rcs}_model.joblib` | artifact dict: `model`, `feature_columns` (ordered), `preprocessing` (steps, sentinels, train-split stats), `sampling_strategy`, `threshold` (+ rule), `model_name`, `model_version`, `validation_metrics`, `validation_sampling_comparison`, `training_config` (hyper-params, seed, split sizes, data-file SHA-256), `environment` |
| `reports/slice_five_models/slice_five_models_results.csv` | final table, ranked by test PR-AUC |
| `reports/slice_five_models/slice_sampling_comparison_validation.csv` | none / undersample / oversample per family (validation) |
| `reports/slice_five_models/slice_hyperparameter_search_validation.csv` | every config x sampling run (validation) |
| `reports/slice_five_models/feature_importance/slice_<fam>_feature_importance.csv` | gain, split count, mean abs SHAP (validation), train missing rate |
| `reports/slice_five_models/slice_leakage_diagnostics.csv` | metrics on val/test rows whose sender domain is not in train |
| `reports/slice_five_models/slice_split_assignments.csv`, `slice_test_predictions.csv` | split per `email_id`; test scores |
| `logs/train_slice_models_*.log` | full training log |

## Method

**Data.** 5,000 Gmail-inbox emails (label 0) + 12,035 phishing_pot emails (label 1). Schemas
identical (106 cols); each file holds only its expected label. `email_id` is unique (0 dropped).
The builders already removed duplicate emails by raw SHA-256 or Message-ID (236 in phishing_pot);
phishing hashes aren't stored locally, so no further cross-source hash check was possible.
Identical *feature vectors* were not removed: the families are low-cardinality (BAS has only 452
distinct vectors in 17k emails), so equal vectors don't mean equal emails.

**Features.** Only `STS__` (10), `BAS__` (6), `DMS__` (7), `LOS__` (10), `RCS__` (9). Excluded:
`email_id`, `label`, `processing_status`, `*_meta__*`, `lookups_*`, `missing_external_lookup`,
and `SRC__`/`INFRA__` (not part of the five scores). The two constant columns
(`LOS__nonstandard_port`, `RCS__final_ip_in_abuse_intelligence`, always 0) are kept for schema
fidelity and get zero importance.

**Preprocessing** (shared code path in `slice_predict.preprocess`): select the stored columns in their stored order →
float64 → ±inf→NaN → documented sentinels→NaN → LightGBM native missing handling. No
imputation or scaling, so nothing is learned from val/test; train-only statistics are stored for reference.

**`STS__new_sender_domain` sentinel.** In `sts_features.new_sender_domain_label`, `-1` means
*age unknown*: no From domain, an RDAP non-200 response, no `registration` event, or any exception. It is
not an age bucket, so it is mapped to NaN. Missingness is heavily label-dependent (−1 for 17% of
legit vs 87% of phishing): it reflects lookup failures, many on historical phishing domains that no
longer resolve in RDAP, and domain age is measured **at extraction time, not at send time**. The
model can still use "missing" as a signal. Ablation on validation: STS PR-AUC is 1.0000 with or
without this feature, so it isn't what drives STS.

**Other sentinel collisions (documented, not changed).** RCS maps an unknown registration age to
`0`, the same value as "≤7 days", and then averages it over links. Emails with no links get
all-zero BAS/DMS/LOS/RCS vectors. Neither can be separated from the stored values without
modifying the feature code.

**Partial rows.** 5 phishing rows have `partial:RCS,SRCINFRA` (all RCS = NaN). They are excluded
from the RCS model only; otherwise "all missing ⇒ phishing" would be learned from 5 rows.

**Split.** One stratified 70/15/15 split (seed 42), shared by all models: train 11,924
(3,500/8,424), val 2,555 (750/1,805), test 2,556 (750/1,806); RCS has 2 fewer train rows and 3 fewer val rows.
*Grouped split was investigated and rejected*: by sender domain, `hdfcsec.com` alone is 59.5% of
legit mail (161 legit domains), so a domain-grouped 70/15/15 split can't keep the class ratios.
Instead `slice_leakage_diagnostics.csv` scores the val/test rows whose sender domain never appears
in train. Limitation: that subset is ~555 phishing vs only 5–7 legit emails, so its ROC-AUC/FPR are
very noisy and it mainly tests unseen *phishing* senders. There is no campaign ID.

**Sampling** (train split only; val/test keep their original distribution): `none`; `undersample` (random
1:1, drops phishing rows; the minority class is legit); `oversample` (random 1:1 with
replacement of legit rows). Seed 42. No class weights are combined with sampling.

**Tuning.** Baseline (31 leaves, lr 0.05) + 24 random configs (leaves, depth, lr,
min_child_samples, subsample, colsample, L1/L2), the same configs for every strategy and family;
early stopping (100 rounds) on validation average precision. `deterministic=True`, 4 threads.
Two full runs gave byte-identical reports.

**Selection (validation only).** Within each strategy, the config with the highest val PR-AUC wins (tie → lower log
loss). Across strategies: highest val PR-AUC; within 0.001 → higher recall at the threshold below,
then the simpler strategy (none < undersample < oversample).

**Threshold objective.** Maximise phishing recall subject to validation FPR ≤ 1% (legit
flagged as phishing). The threshold is set midway to the next lower validation score. With 750 legit
validation emails, 1% = 7 FPs, so the test FPR can drift (it did: 1.2–1.5% for STS/BAS/LOS).

The test split was scored once, after all five artifacts were saved.

## Results (actual run, 2026-10-09)

Ranked by test PR-AUC (prevalence baseline = 0.707):

| # | Score | Feat. | Sampling | Thr. | Val ROC | Val PR | Test ROC | Test PR | Acc | Prec | Recall | F1 | TN | FP | FN | TP | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | STS | 10 | none | 0.050 | 1.0000 | 1.0000 | 0.9995 | 0.9997 | 0.996 | 0.995 | 0.999 | 0.997 | 741 | 9 | 1 | 1805 | 1.2% | 0.06% |
| 2 | BAS | 6 | undersample | 0.853 | 0.9793 | 0.9868 | 0.9756 | 0.9839 | 0.854 | 0.992 | 0.799 | 0.885 | 739 | 11 | 363 | 1443 | 1.5% | 20.1% |
| 3 | LOS | 10 | undersample | 0.836 | 0.9639 | 0.9802 | 0.9582 | 0.9774 | 0.881 | 0.994 | 0.836 | 0.908 | 741 | 9 | 296 | 1510 | 1.2% | 16.4% |
| 4 | RCS | 9 | undersample | 0.894 | 0.8722 | 0.9207 | 0.8753 | 0.9210 | 0.640 | 0.998 | 0.491 | 0.658 | 748 | 2 | 919 | 887 | 0.3% | 50.9% |
| 5 | DMS | 7 | oversample | 0.556 | 0.6640 | 0.7904 | 0.6847 | 0.8011 | 0.383 | 0.975 | 0.130 | 0.229 | 744 | 6 | 1572 | 234 | 0.8% | 87.0% |

Sampling made little difference: val PR-AUC across the three strategies differs by ≤0.0025 in
every family; the choice was mostly decided by the recall tie-break.

## Source–label confounding (read before using these numbers)

Every legit email comes from **one Gmail inbox**; every phishing email comes from **the
phishing_pot honeypot corpus**. So the label is perfectly confounded with the source, and a random
split **does not show generalisation to unseen sources or real mail streams**. Concretely:

* `STS__received_hops` alone (no model) gives ROC-AUC 0.988 on validation: 96% of legit emails have ≤3 hops
  (Gmail delivery) and 99% of phishing have ≥4. It carries 77% of the STS model's gain. The near-perfect STS
  scores mostly measure relay path and source, not sender trust.
* SPF/DKIM/DMARC pass rates for legit are ~100% because they are Gmail-delivered mail from a few
  large senders (`hdfcsec.com` = 59.5%).
* Domain-age lookups ran years after many phishing emails were sent, so missing/young ages track dataset age.

Treat these as in-distribution results. Before relying on them, evaluate on legit and phishing mail
captured through the **same** pipeline (same mailbox/gateway, same period).

---

# v3 — five score models + final stacked model (`train_slice_stack.py`)

Trained on the `prov__core` subset of `slice_dataset_v3.xlsx` (5,152 real emails, same arrival window
15 Apr – 8 Oct 2026, ≥1 link, sender-org cap 5%, receiver-normalised `STS__received_hops` /
`STS__unusual_received_chain`) with its time-ordered split (train 2,553 / val 1,461 / test 1,138).
The v1 artifacts above are not touched; v3 artifacts live in `models/v3/`.

```bash
python -m slice_models.train_slice_stack        # ~90 s; writes models/v3, reports/slice_v3, slice_training_dataset.xlsx
```

```python
from slice_models.slice_final_predict import predict_final
out = predict_final(df, prior=0.01, explain=True)   # STS_prob..RCS_prob, FINAL_prob, FINAL_flag, FINAL_prob_at_prior, top_reasons
```

* **Score models**: one LightGBM per family; config chosen by grouped 5-fold CV on train
  (groups = sender org); refit on train; threshold = max recall s.t. val FPR ≤ 1%.
* **Final model**: LightGBM on the 5 score probabilities (monotone-increasing) + `SRC__` and `INFRA__`
  features, trained on **out-of-fold** score probabilities, config chosen on val PR-AUC,
  class-balanced Platt calibration on val (probability under a 50/50 prior; `prior=` re-weights it).
* `slice_training_dataset.xlsx`: per-score training sheets, `Final_Model_Input` (the final model's inputs),
  `Results_Test`, `Model_Config`, `Final_Input_Importance`, `Feature_List`.
* Same source limitations as above apply (legit = one Gmail inbox that always passes SPF/DKIM/DMARC;
  phishing = one Microsoft 365 honeypot; lookups made Oct 2026; only 256 legit test emails).

---

# v4 deployment: two outputs + RL combiner

```
STS (rule, slice_models/sts_rules.py, never trained) ─────────────────────┐
BAS, DMS, LOS, RCS (ML) ─► link model (ML, + SRC/INFRA) ─► link_prob ─────┤
                                                                          ▼
                       RL combiner (slice_models/rl_combiner.py): flag / allow
                       reward from feedback: +1 right, -5 missed phishing, -1 false alarm
```

* The link model is the only part trained on `slice_dataset_v4.xlsx` (`python -m slice_models.train_slice_stack`).
* The combiner never learns from the dataset (its legit mail almost always passes SPF/DKIM/DMARC). It starts at
  the tested fixed combination `logit(link) + 3 * (STS_risk - 0.5)` (`models/final3/rl_combiner_state.json`) and
  learns from deployment feedback only: Bayesian online logistic regression + Thompson sampling, weights
  non-negative, STS weight capped at 4, threshold from the costs.
* `python -m slice_models.simulate_rl` replays the val+test emails (arrival order) under four scenarios.
  Result (5 seeds): the combiner lowers its STS weight 3 -> ~2.2 when 30% of legit fail authentication and
  ~1.9 when phishing also comes from authenticated senders, raises it to ~3.3 when authentication is clean,
  and learns a bias of ~+1 that corrects the fixed combination's calibration. Total reward stays within ~0.5%
  of the fixed combination, so its value is adaptation in deployment, not a gain on this replay.
