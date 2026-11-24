# SLiCE phishing email detection

Email feature extraction, model training, evaluation, and local review apps. The repository contains the original project and a separate offline retraining attempt.

| Folder | Contents |
| --- | --- |
| `phish_dataset_v2/` | Original feature modules, dataset builders, training notebook, fitted models, evaluation reports, and app. |
| `codex_attempt/` | Offline feature extraction, frozen retrained models, leakage checks, evaluation reports, and an app exposing component scores. |

## Run the offline app

In PowerShell, from this repository's root:

```powershell
py -m venv codex_attempt\.venv
.\codex_attempt\.venv\Scripts\python.exe -m pip install -r codex_attempt\requirements-app.txt
powershell -ExecutionPolicy Bypass -File .\codex_attempt\run_app.ps1
```

Open http://127.0.0.1:5001 in Chrome and upload a raw `.eml` file. If the environment already exists, use the last command. The launcher runs the server in the background; add `-Foreground` to keep it attached to the terminal.

The app displays the final risk score, BAS/DMS/LOS scores, combined link score, rule-based STS risk and trust, authentication evidence, extracted links, feature values, and processing time. Emails with supported HTML links use the saved combined model and threshold. Without those links, the app uses a separately labelled STS-only fallback and its existing standalone threshold (0.25). No model is retrained and no threshold is retuned. Uploaded emails are not retained.

The app is offline by default. Its optional **live RCS** checkbox follows current links and queries current domain/IP metadata in a separate, bounded worker. The worker uses the original project's RCS model, which is outside the offline model's leakage audit. Its result never changes the main verdict. Failed measurements do not produce an RCS model score. See the offline project README for scan limits and scope.

To check the app:

```powershell
.\codex_attempt\.venv\Scripts\python.exe -B -m unittest discover -s codex_attempt -p 'test_*.py' -v
```

## Model scope and evaluation

The offline model uses BAS, DMS, LOS, static URL features, and fixed sender rules. It excludes live infrastructure lookups, redirect outcomes, RCS, and sender domain age. It aggregates link features at email level; it does not produce a learned score for each URL. The app's STS-only fallback for emails without supported HTML links has not been evaluated on an independent linkless-email dataset, and the optional legacy live RCS assessment is outside the offline model's evaluation.

Scores are risk indicators, not verified probabilities of phishing. The historical test reported 89.52% accuracy and 8.95% false-positive rate. An independent historical legitimate-only challenge produced a 73.53% false-positive rate on 170 eligible emails. These results show that deployment accuracy has not been established and legitimate emails can be flagged. The interface does not fix those limitations.

See [the offline project README](codex_attempt/README.md), [evaluation results](codex_attempt/reports/RESULTS.md), [individual model results](codex_attempt/reports/INDIVIDUAL_SCORES.md), and [external evaluation review](codex_attempt/reports/EXTERNAL_DATASET_REVIEW.md) for methodology and limits.

## Local data

Private mailbox data, OAuth credentials, virtual environments, downloaded archives, dataset workbooks, caches, logs, and per-email evaluation records are excluded from Git. Source code, fitted model artifacts, aggregate evaluation reports, and synthetic test fixtures are included. Dataset creation and training require separately supplied local data; see each project's documentation.
