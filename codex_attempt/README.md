# SLiCE offline retraining attempt

This folder contains a new training run, not the original saved models.
The user chose to remove information unavailable at email arrival. The stack therefore has three learned families (BAS, DMS, LOS), static SRC features, and a fixed STS rule. RCS, live INFRA, redirect results, DKIM-derived link features, and sender domain age are excluded. No email link is fetched during training or inference.

## Run

From the parent project directory in PowerShell:

```powershell
.\codex_attempt\.venv\Scripts\python.exe -B -u codex_attempt\train.py
.\codex_attempt\.venv\Scripts\python.exe -B codex_attempt\predict.py path\to\email.eml
```

The environment is isolated in `.venv`. `requirements-lock.txt` records installed packages.
Original project files are read, not edited. Frozen copies of required feature helpers are in `vendor/`.

## Local email review app

From the parent project folder, run `powershell -ExecutionPolicy Bypass -File .\codex_attempt\run_app.ps1`, then open http://127.0.0.1:5001 in Chrome. The launcher starts a hidden background server and returns control to the terminal. Logs are in `private/app_stdout.log` and `private/app_stderr.log`. For foreground operation, add `-Foreground`.

Alternatively, run `.\codex_attempt\.venv\Scripts\python.exe -B codex_attempt\app.py --port 5001`. Install the app dependencies when setting up a fresh environment with `.\codex_attempt\.venv\Scripts\python.exe -m pip install -r codex_attempt\requirements-app.txt`.

Upload a raw `.eml` file to see the risk score, BAS/DMS/LOS scores, combined link score, STS risk and trust, authentication evidence, extracted hyperlinks, feature values and processing time. Scores are risk indicators, not verified phishing probabilities. Missing authentication is labelled unknown; the unchanged STS rule still assigns risk to missing evidence. Uploaded authentication headers are read, not cryptographically verified.

Emails with supported HTML HTTP(S) hyperlinks use the unchanged frozen combined model and threshold. Emails without supported HTML hyperlinks now receive an **STS-only assessment**, using `predict.rule_risk` and the existing standalone threshold in `models/individual/sts_rules.json` (0.25). The link classifier does not run for these emails. This added app policy has not been evaluated on an independent linkless-email dataset; the published combined-model metrics do not describe it. `in_scope` and `link_model_in_scope` still identify the original link model's scope; `assessment_mode` identifies the decision path used by the app.

The app is offline by default. Enable **Add optional live RCS assessment** to run a separate worker that uses the original RCS extractor and `phish_dataset_v2/models/final3/slice_rcs_model.joblib`. This legacy model is outside the codex_attempt leakage audit and evaluation. It follows current redirects and queries current RDAP/IP metadata, displays its own score and threshold, and **never changes the combined or STS-only verdict**. RCS is inapplicable without supported HTML links. Scans are limited to five HTML links, five redirects per request, thirty requests, and a 45-second worker timeout. Only public HTTP(S) endpoints on ports 80/443 are allowed; DNS-pinned transport validates every redirect. Unsubscribe/sign-out targets are blocked. Failed or incomplete measurements produce a partial/unavailable result without a model score. The legacy abuse-intelligence feature remains a placeholder. Raw emails are not retained by either path.

Existing false-positive limitations remain. No model was retrained or threshold retuned for these app changes. App and mocked live-network integration checks: `.\codex_attempt\.venv\Scripts\python.exe -B -m unittest discover -s codex_attempt -p 'test_*.py' -v`.

## Evaluation protocol

The protocol is recorded in `PROTOCOL.json` before fitting. Main-cohort arrival quantiles define TRAIN (first 55%), TUNE (next 15%), CALIBRATION (next 10%), THRESHOLD (next 5%), and TEST (last 15%). Training additions are accepted only before the TRAIN cutoff. These are time windows, not guaranteed retained proportions after purging.

Raw SHA256, normalized body fingerprints, Message-IDs, and sender/link-organization templates define connected groups. Exact raw duplicates are collapsed, conflicting raw labels excluded, and connected groups retained only in their latest partition. Earlier rows in those groups are purged, preserving temporal order. All known grouping keys are checked for zero overlap across every pair of retained partitions. Fingerprints and paths remain in `private/`.

Base method selection uses grouped three-fold CV within TRAIN. Nested outer grouped folds produce final-model training scores: each held-out prediction comes from a model whose fitting and method selection exclude that outer fold. Each full-TRAIN winner is then fitted for inference. Candidate methods are random forest, LightGBM and logistic regression, with fixed configurations.

The link model is trained on nested out-of-fold TRAIN scores plus static SRC. Its candidate configuration is selected on TUNE. Class-balanced Platt calibration uses CALIBRATION only. The fixed combination is `sigmoid(logit(calibrated_link) + 3*(STS_risk - 0.5))`, with sender-age evidence removed from the STS weights. The final threshold maximizes recall at empirical THRESHOLD FPR <=1%. It is not selected on TEST.

The artifact is saved before the final test metrics are computed. The training script refuses to run again after `reports/metrics.json` exists. This prevents accidental in-place iteration on its test results; it does not make historically exposed data unseen again.

## Outputs

- `models/model.joblib`: three newly fitted base models, learned link model, calibrator, threshold, feature schema and protocol.
- `reports/metrics.json`: accuracy, precision, recall, F1, FPR, specificity, ROC-AUC, PR-AUC, log loss, confusion counts, and cluster-bootstrap intervals.
- `reports/latency.json`: warm single-email inference, batch throughput, offline extraction, raw-email-to-verdict latency and cached model reload time.
- `reports/leakage_audit.json`: partition counts, dates, exclusions and group-overlap assertions.
- `reports/base_selection.csv`, `reports/link_selection.csv`: selection evidence.
- `reports/model_manifest.json`: actual model input columns, threshold and model digest.
- `private/split_manifest.csv`, `private/exclusions.csv`, `private/test_predictions.csv`: reproducibility details containing potentially identifying metadata; do not publish.
- `reports/RESULTS.md`: readable results after successful evaluation.

## Interpretation and limits

This is a stricter **retrospective protocol-validation rerun**, not proof of leakage-free deployment accuracy. The original cohort has already influenced earlier project choices; no new split can reverse that exposure. Its inclusion was also affected by earlier live-processing success and threat-feed screening. Source-label confounding remains: most legitimate emails come from one Gmail mailbox, most phishing from one Microsoft honeypot. Training additions diversify this somewhat but do not remove it.

Static code lists are frozen at the present version, not reconstructed historically. Known-template and fingerprint isolation does not prove every semantic campaign or near-duplicate has been found. Authentication verdicts are assumed to be genuine receiver-provided headers; arbitrary uploads are not authenticated by the tool. Labels are inherited rather than newly hand-verified.

The final combined output is a risk score. Platt calibration covers the link score before adding STS; the resulting final score must not be presented as a verified deployment probability. Precision depends strongly on phishing prevalence, so the metrics also include hypothetical 1% and 0.1% prevalence projections. Those projections assume recall and FPR transfer.

Emails without extracted links are outside the frozen combined model's evaluation scope. `predict.py` marks `in_scope=false` for these inputs. The app adds a separately labelled STS-only fallback, whose linkless-email performance has not been established. PASS/flag outcomes are advisory and do not modify any mailbox.

Latency includes no live network requests. Warm single-email time is different from batched amortized time; cached model reload time is not cold process startup. These measurements are machine-specific.

## Completed run

The model is fitted and evaluated. Start with [reports/RESULTS.md](reports/RESULTS.md). Independent checks passed; original-vs-normalized raw input scores matched exactly on 50 samples, with zero network attempts.

Additional commands:

```powershell
.\codex_attempt\.venv\Scripts\python.exe -B codex_attempt\verify.py
.\codex_attempt\.venv\Scripts\python.exe -B codex_attempt\benchmark_raw.py
.\codex_attempt\.venv\Scripts\python.exe -B codex_attempt\render_report.py
```

`reports/raw_latency.json` is the benchmark on original unnormalized inputs; prefer it for end-to-end latency. `latency.json` also contains prepared-input timings and batch-only model throughput. Startup measurements use a fresh interpreter but a warm OS cache, with only three samples.

The threshold selection set has only 55 legitimate emails. Zero observed false alarms there did not establish a population FPR <=1%; the frozen test FPR was 8.95%. This is a material limitation of the current fitted operating point. Do not adjust its threshold using the test labels and continue calling that test untouched.

The fixed candidate grid here is smaller/different from the legacy search. The same modular score/stack/rule approach is tested with stricter isolation, but this is not a controlled measurement of leakage inflation alone.

The additional KD CSV was inspected for format compatibility only: it contains no From/Received headers or HTML anchors. It was not used to train, select or evaluate this raw-email detector; manufacturing headers/anchors would change its inputs and would not supply a faithful external email evaluation.

## Individual score exports

See `reports/INDIVIDUAL_SCORES.md`, `individual_scores.csv` and `individual_scores.json` for the component results on the same historical test. Standalone frozen BAS/DMS/LOS artifacts are under `models/individual/`; STS is supplied as `sts_rules.json` plus the `predict.rule_risk` implementation. These exports preserve the trained estimators. Their supplementary thresholds were selected on the existing THRESHOLD partition, without test-label selection. The combined model and its threshold are unchanged.

Example:

```powershell
.\codex_attempt\.venv\Scripts\python.exe -B codex_attempt\individual_scores.py --family LOS --email path\to\mail.eml
```

The STS rule outperformed the combined model on this test. This is a descriptive finding, not evidence of real-world accuracy or a reason to choose a replacement model using the same test. BAS raised no flags, so its precision is undefined. Individual classifier thresholds are supplementary: the stack consumes continuous scores, not those flags. RCS is excluded in this offline variant.

## Independent public-source test

See [reports/EXTERNAL_DATASET_REVIEW.md](reports/EXTERNAL_DATASET_REVIEW.md). On previously unused SpamAssassin ham, the frozen detector falsely flagged 125/170 eligible link-bearing legitimate messages (73.53% FPR). The other 3,926 retained messages were linkless and out of scope. This historical, difficult, negative-only challenge cannot estimate overall phishing accuracy, but shows poor external false-positive generalization. No model or threshold was changed.
