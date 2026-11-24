# Independent public-dataset search and evaluation

The trained model and all thresholds remain unchanged. No downloaded email links or attachments were executed or fetched.

## Completed new-source test: Apache SpamAssassin ham

Publisher source: https://spamassassin.apache.org/old/publiccorpus/readme.html

Downloaded the 20030228 easy_ham, easy_ham_2 and hard_ham archives. The public publisher labels these as non-spam messages. They were not part of the four training input files. Comparison with every record in the 10,622-email development extraction cache found no matching normalized SHA256, Message-ID or canonical body fingerprint. Removed 54 within-batch body-fingerprint repeats.

- Downloaded: 4,150 labelled legitimate emails.
- Retained after duplicate audit: 4,096.
- Eligible link-bearing inputs: 170 (4.15% coverage).
- Linkless / outside the model's scope: 3,926; not counted as successful predictions.
- Falsely flagged: 125/170 = 73.53% (Wilson 95% interval 66.43% to 79.59%).
- Correctly allowed in-scope legitimate emails: 45/170 = 26.47%.

Most eligible examples are hard_ham (157/170). The publisher deliberately selected that subset to resemble spam despite being legitimate. These are 2002?2003-era messages with different authentication and content distributions from the training inbox. The rate must not be extrapolated to a current mail stream. Nonetheless, it demonstrates that the frozen detector is not reliably generalizing to this new source.

There are no labelled phishing examples in this ham-only test. Do not call 26.47% overall phishing accuracy, do not infer phishing recall, and do not mix generic spam into a phishing class to manufacture a two-class score.

See `external_spamassassin/metrics.json`, `external_spamassassin/components.csv` and `external_spamassassin/RESULTS.md`. Archive hashes and download URLs are recorded in the protocol and the private download manifest.

## Other candidates checked

| Candidate | Finding | Decision |
|---|---|---|
| IWSPA-AP full-header corpus | Downloaded 4,585 records (4,082 legitimate, 503 phishing) from the public participant mirror at https://github.com/BarathiGanesh-HB/IWSPA-AP/tree/master/data. Archive SHA256 ce0e17d35439465724ea720178fc74539210cd61053ae1d9aebbe181bc4524ec. Original authors explicitly replace URLs with placeholders, normalize source domains and strip HTML/base64 text. The mirror's MIME parse produced empty text bodies in 3,886 legitimate and 167 phishing records. It also draws from Nazario, a source already represented in this project. | New archive, but unsuitable for faithful full URL/header model evaluation. No artificial restoration of links or headers. |
| MeAJOR | https://zenodo.org/records/18471483 contains preprocessed records, replaces URLs and identities with tokens, and aggregates TREC, Nazario and Nigerian Fraud. | URL evidence destroyed; partial source overlap. Not a compatible untouched raw-email benchmark. |
| cw-l/message-corpus | https://github.com/cw-l/message-corpus contains real emails, but routing headers are stripped, URLs defanged and spam/scam/phishing subclasses not supplied. | Useful research collection; not ready-made independent phishing ground truth for this model. |
| EPVME | https://github.com/sunknighteric/EPVME-Dataset constructs malicious examples by editing headers/body payloads, with existing corpora as source material. | Synthetic malicious-email robustness challenge, not an independent real-world phishing accuracy estimate. |
| Sting9 | Public dataset repository listing currently contains the licence only; no dataset dump located. | Landing-page claims are insufficient to establish downloadable evaluation data. |
| SpaPhish | https://github.com/lbustio/spa_phish documents 1,395 Spanish emails (731 phishing, 664 legitimate) with human annotations, but public distribution is structured content and derived features rather than the original full MIME artifacts. | Promising independent collection; cannot faithfully supply this model's complete original input from published CSV alone. |

IWSPA author preprocessing reference: https://www2.cs.uh.edu/~shahryar/files/IWSPA-AP.pdf (section 2.2).

The useful finding is a completed independent false-positive test with poor performance, not a validated universal real-world accuracy. Model changes motivated by these results would require a different final external holdout.
