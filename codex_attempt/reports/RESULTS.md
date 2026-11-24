# Measured results

This is a stricter offline retrospective rerun on previously exposed project data, not an untouched external evaluation.

| Metric | Test result | 95% group-bootstrap interval |
|---|---:|---:|
| accuracy | 89.52% | 85.98% to 92.53% |
| precision | 90.90% | 84.01% to 95.86% |
| recall | 88.02% | 84.85% to 91.07% |
| f1 | 89.43% | 85.56% to 92.47% |
| fpr | 8.95% | 4.14% to 15.67% |
| roc_auc | 0.9533 | 0.9177 to 0.9780 |
| pr_auc | 0.9370 | 0.8665 to 0.9811 |

Test n=1441; TN=651, FP=64, FN=87, TP=639. Threshold=0.215460.
Resampled template/duplicate components: 610. Confidence intervals resample these groups, not individual correlated emails.

| Partition | Legitimate | Phishing |
|---|---:|---:|
| train | 1020 | 2750 |
| tune | 132 | 528 |
| calibration | 72 | 422 |
| threshold | 55 | 156 |
| test | 715 | 726 |

Excluded rows: 4046. Every audited cross-partition group overlap is zero.

| Latency | Median | p95 |
|---|---:|---:|
| single_email_warm_model | 122.08 ms | 143.83 ms |
| batch_amortized_per_email | 0.12 ms | 0.16 ms |
| offline_feature_extraction | 178.95 ms | 293.53 ms |
| raw_email_to_verdict_warm | 308.97 ms | 444.02 ms |

Model reload with OS cache warm: 106.27 ms. Threads: 4.
Raw bytes read before timer; features include normalization, parsing and temporary-file I/O; no network, no lookup cache. Warm repeated measurements on this machine, not service SLA.

Hypothetical precision at 1% prevalence: 9.04%; at 0.1%: 0.97%.

## Limitations

- Historical cohort already used during earlier project development; this is a protocol-validation rerun, not a pristine external holdout.
- Source-label confounding remains: one main legitimate mailbox and one main phishing honeypot.
- Existing cohort eligibility was influenced by earlier live-processing success and feed screening; offline re-extraction cannot undo that selection.
- Static shortener/brand lists are frozen current code, not historically versioned threat intelligence.
- Content fingerprinting and template grouping reduce repetition but do not prove every near-duplicate or semantic campaign has been found.
- Authentication verdicts are trusted receiver-provided arrival headers; arbitrary uploaded header authenticity is not verified.

Do not claim a leakage-free real-world accuracy from these data. A newly collected, prospectively locked, independently labelled evaluation is still required.


## Original raw-email latency

| Stage | Median | p95 |
|---|---:|---:|
| original_raw_feature_extraction | 126.23 ms | 263.85 ms |
| original_raw_model_inference | 79.69 ms | 114.53 ms |
| original_raw_read_to_verdict | 269.20 ms | 413.58 ms |
| fresh_process_read_to_verdict | 7807.89 ms | 8797.68 ms |

Original inbox/phishing-pot .eml inputs, not pre-normalized copies; warm measurements include disk read, header normalization, static feature extraction and inference. Fresh-process measurements include interpreter startup/imports and model loading, with OS file cache warm; n=3.
Original-vs-normalized score difference: 0.0; network attempts: 0.

## Threshold and comparison cautions

The threshold partition contained 55 legitimate emails and 156 phishing emails. It had 0 false positives. The empirical <=1% selection target was not a statistical guarantee: the later test FPR was 8.95%.

This run changes the feature set, search grid, calibration protocol and training population relative to the legacy model. Its performance difference cannot be attributed entirely to data leakage. The main test window was already exposed during earlier project development; source bias and historical cohort selection remain.

Verification status: PASS. Final model and threshold were not changed after test evaluation.
