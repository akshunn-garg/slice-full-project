# Individual frozen score models

Descriptive evaluation of the existing frozen components, not fresh independent validation.
Individual thresholds use 55 legitimate threshold emails; empirical FPR <=1% does not guarantee test/deployment FPR <=1%.
Individual latency excludes extraction, file reads and model loading; 50 warm single-row calls, four threads.
RCS was excluded because its live-lookup evidence was unavailable at historical email arrival.
STS is a fixed rule, not an ML score model. Component scores are inputs to the stack; individual flags are supplementary.
Precision is undefined when a model raises no phishing flags; it is displayed as N/A.
STS outperformed FINAL on this historical test. This does not establish a real-world benefit for the stack or justify selecting STS using this same test.

| Model | Accuracy | Precision | Recall | FPR | ROC-AUC | Warm model median |
|---|---:|---:|---:|---:|---:|---:|
| BAS | 49.62% | N/A (no flags) | 0.00% | 0.00% | 0.5809 | 2.00 ms |
| DMS | 48.23% | 14.29% | 0.55% | 3.36% | 0.6232 | 2.41 ms |
| LOS | 49.97% | 100.00% | 0.69% | 0.00% | 0.9500 | 30.32 ms |
| STS | 94.31% | 98.20% | 90.36% | 1.68% | 0.9667 | See full latency report |
| LINK | 86.88% | 91.24% | 81.82% | 7.97% | 0.9233 | See full latency report |
| FINAL | 89.52% | 90.90% | 88.02% | 8.95% | 0.9533 | See full latency report |
