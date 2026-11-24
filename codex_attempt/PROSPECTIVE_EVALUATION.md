# Fresh-email evaluation

Status: evaluator prepared and protocol checks passed. **No new real-world accuracy has been measured.** Fresh independently reviewed emails are still required.

The model, feature extractor and threshold stay frozen. Do not use the existing historical benchmark again and call it a fresh evaluation. Do not label Inbox as legitimate or Spam as phishing: both can contain either class.

For a prospective deployment sample, collect consecutive incoming emails after the model was frozen, including the same gateway's Inbox and Spam deliveries. Record the selection process and delivery interval. Avoid sampling only suspicious or model-flagged messages. A balanced collection assembled separately is a useful challenge set but does not measure deployment precision at natural prevalence.

## Prepare a blinded review

Place the original raw `.eml` inputs in a new directory. From the workspace root:

```powershell
.\codex_attempt\.venv\Scripts\python.exe -B codex_attempt\real_world_eval.py prepare --input-dir path\to\fresh_emails --packet fresh_batch_01 --sampling "Consecutive incoming mail from [gateway], [start/end], including Inbox and Spam"
```

Preparation produces a locked sample and review files under `codex_attempt/private/prospective/fresh_batch_01/`. It computes no model predictions. Exact known messages, repeated Message-IDs, duplicate messages, missing delivery timestamps, old deliveries and implausibly future deliveries are rejected with reasons. It does not exclude recurring sender organizations or repeated newsletter templates: those can be part of a real future mail stream.

The delivery-time candidate is the first `Received` header, never the sender's `Date`. Collection provenance must establish that this header was added by the trusted receiving gateway. Arbitrary uploaded headers are not independently authenticated by this tool. The cutoff is the frozen model artifact's modification time, recorded in UTC in the sample lock; preserve the original artifact.

Review each plaintext file under `review/`, independently of model scores. Fill every row in `labels.csv`:

- `label`: `0` for non-phishing, `1` for verified phishing, or `uncertain` when unresolved.
- `reviewer`: the person responsible for verification.
- `evidence`: the basis for the label. Folder membership or authentication results alone are insufficient.

Ordinary unsolicited marketing is not automatically phishing. Keep the phishing definition consistent across the sample. Blinding means not consulting model predictions while assigning labels. The plaintext review preserves hyperlink targets as text and does not retrieve them.

## Evaluate once after review

```powershell
.\codex_attempt\.venv\Scripts\python.exe -B codex_attempt\real_world_eval.py evaluate --packet fresh_batch_01
```

The evaluator requires the entire selected batch to have reviewer labels/evidence, locks the label-file digest before scoring, verifies the model digest/threshold and raw-message hashes, disables network requests, and records predictions and metrics inside the private packet. It refuses to overwrite a completed evaluation.

Outputs include accuracy, precision, recall, false-positive rate, confusion counts, ROC-AUC/PR-AUC when both classes exist, latency, sample coverage and Wilson intervals. Linkless and uncertain messages are counted separately and excluded from classifier metrics; they are never silently treated as correctly allowed mail. If the sample contains no verified phishing, recall and ROC-AUC are unavailable: that sample cannot establish phishing-detection performance. Zero false alarms on a small sample does not establish zero population FPR.

Any result describes the sampled gateway, period, prevalence and reviewed ground truth. It does not establish universal real-world accuracy. Campaign correlation can make per-message Wilson intervals optimistic. Keep this evaluation out of model and threshold selection; future improvements require a separate final test.
