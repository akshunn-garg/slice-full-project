# New-source SpamAssassin ham test

Downloaded 4150 labelled legitimate emails; retained 4096 after overlap checks.
Link-bearing, in-scope: 170; linkless, out of scope: 3926.
False alarms: 125/170 (73.53%; Wilson 95% interval 66.43% to 79.59%).

This measures external false-positive behavior only. There are no labelled phishing emails in this test, so phishing recall, precision and overall two-class accuracy are unavailable.

Model and threshold were unchanged. No email links were fetched.

Source: https://spamassassin.apache.org/old/publiccorpus/readme.html

Legitimate-only data cannot estimate phishing recall, precision, ROC-AUC or overall two-class accuracy.
Historical 2002-2003 mail predates modern authentication and is not a representative current mailbox.
Publisher labels are inherited, not newly independently hand-verified.
Canonical-body exclusion can conservatively remove related messages, not just byte-identical duplicates.

## Component false-positive rates at frozen thresholds

| Component | False positives / 170 | FPR |
|---|---:|---:|
| BAS | 0 | 0.00% |
| DMS | 0 | 0.00% |
| LOS | 0 | 0.00% |
| STS | 170 | 100.00% |
| LINK | 89 | 52.35% |
| FINAL | 125 | 73.53% |

Most in-scope messages (157/170) are deliberately difficult hard_ham. No development-message overlap was found on the audited keys. These results describe a historical legitimate-only challenge, not a representative modern incoming-email sample.
