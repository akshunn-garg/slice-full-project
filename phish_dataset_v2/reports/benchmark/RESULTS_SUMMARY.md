# SLiCE — results summary (for the paper)

All numbers come from the scripts named in each section; raw CSVs are in this folder and in
`reports/final3/`. Final model: `models/final3/` (BAS, DMS, LOS, RCS score models → LightGBM link
model on the 4 scores + INFRA; STS rule-based, combined with fixed weight 3).

## Data
- `slice_dataset_v4.xlsx`: 4,801 legit (one Gmail inbox) + 4,801 phishing (phishing_pot, one
  Microsoft 365 honeypot), same count per arrival month (Dec 2022 – Oct 2026), receiver-normalised
  headers, one canonical Authentication-Results header, identical recipient anonymisation.
  Time-ordered split: train 6,721 / val 1,440 / test 1,441.
- Training-only additions: 405 Nazario phishing (Dec 2022–2023), 502 diverse legit, 113 unused
  training-period emails.
- Held-out phishing source: Nazario 2024–25, 778 emails (never trained or tuned on).

## Shortcut findings (method story)
| Finding | Evidence |
|---|---|
| Relay hop count identified the receiving server, not phishing | `received_hops` alone ROC 0.99 (v2) → 0.85 after receiver normalisation |
| DKIM-derived BAS/DMS features encoded authentication | Nazario recall 67.1% (run 1) → 90.9% after removing them + adding authenticated phishing (run 2) |
| Adding diverse legit senders reduced false alarms | Test FPR 3.1% (run 2) → 1.26% (run 3), recall 95.2% → 95.3%; Nazario 90.9% → 88.8% |

Disclosures: run 3's change was motivated by analysing test false alarms (test slightly optimistic;
validation FPR 0.9%). Nazario 2024–25 was evaluated once per run (3 runs), never used for choices.
Nazario 2023 (same collector) is in training, so the held-out test is later-in-time, partly same source.

## Baselines (same data, same protocol) — `benchmark_baselines.py`
| Model | Test ROC [95% CI] | Recall | FPR | Precision @1% / @0.1% prevalence | Nazario recall |
|---|---|---|---|---|---|
| B1 URL-only (LOS+SRC) | 0.975 [0.967, 0.980] | 81.0% | 3.50% | 19.0% / 2.3% | 72.4% |
| B2 all 73 features, one model | 0.997 [0.995, 0.999] | 98.5% | 1.96% | 33.7% / 4.8% | 96.8% |
| B3 SLiCE offline (no lookups) | 0.983 [0.977, 0.988] | 82.9% | 2.24% | 27.2% / 3.6% | 78.7% |
| **SLiCE final** | 0.996 [0.995, 0.998] | 95.3% | **1.26%** | **43.3% / 7.0%** | 88.8% |

## Robustness to legit authentication failures — `benchmark_robustness.py`
Legit test emails (n=715) with realistic authentication failures, links/content unchanged:
| Scenario | B2 all-features: legit flagged | **SLiCE: legit flagged** |
|---|---|---|
| as observed | 1.96% | 1.26% |
| forwarded (SPF fails) | 13.99% | 1.54% |
| mailing list (SPF/DKIM/DMARC fail) | **100%** | **1.96%** |
| small sender, no authentication | **100%** | **1.82%** |

Matched operating points (VAL FPR target 0.5% / 1% / 2%): B2 has higher in-distribution recall
(97.0/98.5/99.0% vs 92.8/95.3/97.3%) and Nazario recall (94.2/96.8/98.7% vs 83.9/88.8/93.6%).
→ B2 is more accurate when authentication behaves as in the training data; SLiCE is robust when it does not.

## 3-tier policy (SLiCE, thresholds from VAL) — `policy_three_tier.csv`
| Set | block | warn | pass |
|---|---|---|---|
| test legit (715) | 0.00% | 1.26% | 98.74% |
| test phishing (726) | 90.50% | 4.82% | 4.68% |
| Nazario phishing (778) | 76.99% | 11.83% | 11.18% |

## Non-phishing spam (same receiving server as legit) — `eval_spam.py`
Gmail Spam folder, 92 emails, hand-checked by the mailbox owner: spam, none phishing. 91 have links.
All 91 pass SPF, DKIM and DMARC (bulk-mail services).
| Model | Called phishing | Blocked | Warned | Passed |
|---|---|---|---|---|
| SLiCE final | 1.1% (1/91) | 0.0% | 1.1% | 98.9% |
| B2 all-features flat | 2.2% (2/91) | – | – | 97.8% |
Small sample; supports that SLiCE separates phishing from ordinary unwanted mail. Not a phishing test
(no phishing was present), so the independent-phishing-source gap remains.

## LLM classifiers (local, same emails) — `benchmark_llm.py`
Prompt in the style of ChatSpamDetector (headers, links, body text); temperature 0; CPU only.
| Model | Test n | Accuracy | Precision | Recall | FPR | ROC | Nazario recall (n) | s/email (median) | unusable answers |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5 1.5B | 200 | 75.0% | 68.4% | 93.0% | 43.0% | 0.890 | 90.0% (100) | 17.5 | 20/300 |
| SLiCE (same 300) | 200 | 98.5% | 100% | 97.0% | 0.0% | 0.998 | 94.0% (100) | ~0.06–0.1 offline | 0 |
| qwen2.5 7B | 60 | 96.7% | 100% | 93.3% | 0.0% | 0.997 | 77.5% (40) | 53.5 | 5/100 |
| SLiCE (same 100) | 60 | 100% | 100% | 100% | 0.0% | 1.000 | 87.5% (40) | ~0.06–0.1 offline | 0 |
Small samples (especially 7B): wide confidence intervals. Cloud LLMs (GPT-class) were not tested.

## Latency (measured on the same machine)
| Stage | Time per email |
|---|---|
| normalise | 9 ms median (p95 111 ms) |
| LOS features (local) | 29 ms median (p95 70 ms) |
| parse + link extraction | 24 ms median (p95 69 ms) |
| SLiCE offline pipeline (B3) | ~60–100 ms |
| SLiCE model inference | 0.3 ms (batched), ~190 ms single call, 5 s one-time load |
| full feature extraction, live lookups | ~15 s for new domains; ~0.1–0.2 s cached |
| qwen2.5 1.5B / 7B (CPU) | 17.5 s / 53.5 s |

## Limitations to state
One legit mailbox; one main phishing honeypot; held-out source partly shares a collector with
training; lookups made after arrival (dead phishing domains); email-level labels (patent claims
link level); RL combiner only simulated; LLM baselines local and small-sample; no fully independent
third phishing source (the hand-checked Gmail Spam contained no phishing).
