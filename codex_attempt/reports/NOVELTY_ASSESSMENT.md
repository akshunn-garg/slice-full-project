# Preliminary novelty assessment: links first, email second

This is a preliminary technical/prior-art comparison, not a claim-by-claim patentability or freedom-to-operate opinion. No model, threshold, or training data was changed.

## Proposed idea and implemented behavior

The proposed workflow is: score each hyperlink in its email context, identify suspicious links, and aggregate the link evidence into an email decision. The SLiCE draft describes individually classified hyperlinks and source-link consistency.

The trained `codex_attempt` artifact is different: BAS, DMS, and LOS extract per-link features but generally average them across the email before the score models run. SRC contributes average/maximum/any summaries. The learned base models and stack therefore operate on email-level rows. They do not demonstrate supervised per-link risk estimates followed by a learned link-set classifier. The training labels are email labels, not verified labels for every embedded link.

Evidence: `phish_dataset_v2/feature_modules/bas_features.py` (aggregation near line 588), `dms_features.py` (near line 782), `los_features.py` (near line 664), `source_link_infra_features.py` (near line 761), and `codex_attempt/predict.py`.

## Related disclosures found

| Proposed broad element | Existing related disclosure | Implication |
|---|---|---|
| Score URL/email components and use their scores to decide the email threat | US10333974B2 describes scoring components including URLs and assigning the email the most severe component score; its description also discusses weighted averages. | The general links/components-first, email-second idea should not be presented as newly invented here. |
| Assess the danger of each link | US8769695B2 describes individual website-link threat probabilities using historical keyword combinations. | Per-link scoring alone is not a defensible novelty statement. |
| Combine sender/email evidence with URL evidence | US10601865B1 describes combined email and URL analysis within a credential-spearphishing detection workflow. | A source-plus-URL combination needs a specific distinguishing mechanism. This reference also uses webpage analysis; it does not establish anticipation of every webpage-free design. |
| Learn an email label from a collection of link instances with only email labels | Multiple-instance learning is an established framework (Dietterich et al., 1997; Maron and Lozano-Perez, 1997). | Using bag/instance learning is a possible appropriate method, not itself a newly invented learning principle. |

Sources:

- https://patents.google.com/patent/US10333974B2/en
- https://patents.google.com/patent/US8769695B2/en
- https://patents.google.com/patent/US10601865B1/en
- https://lis.csail.mit.edu/pubs/tlp/multiple-instance-aij.pdf
- https://papers.neurips.cc/paper/1346-a-framework-for-multiple-instance-learning.pdf

These references are relevant prior art to investigate, not a finding that an exact future claim is invalid or infringes a patent. Exact claim elements, dates, jurisdiction, and additional art matter.

## Most promising candidate contribution

Investigate a specific mechanism that scores each link using the relationships among authenticated sender, independently inferred claimed entity, visible link representation, actual destination, and evidence availability, then aggregates the complete link-level evidence while preserving which link supported the decision.

The contribution would need to lie in the concrete scoring/learning/aggregation rule and demonstrated benefit, rather than merely the ordering of the stages or the names of five scores. The following are research directions, not verified novelty findings:

1. Preserve co-occurring evidence on the same link. An identity mismatch on link A and obfuscation on link B should not automatically be treated as if both occurred on one malicious link.
2. Test robustness to additional benign links. Averaging a binary suspicious-link feature gives 1/21 with one suspicious and 20 benign links, but 1/101 after adding 80 benign links. This shows a possible dilution mechanism in feature means, not a demonstrated prediction failure of the existing multifeature model. Max pooling alone is already represented in related work and patent disclosures.
3. Represent missing authentication as missing evidence, with measured uncertainty, rather than equating it with failed authentication. Valid third-party destinations and delegated sending must be considered: unequal sender and destination domains are not proof of phishing.
4. Preserve a link-specific explanation and verify localization against annotated links. Do not assign a phishing-email label to every URL: phishing messages often contain legitimate footer and brand links. An email-level positive label also does not prove the email contains a malicious link; attachment-only and reply-based attacks require a separate scope definition.

## What would substantiate the contribution

Compare the unchanged email-summary approach against per-link scoring with mean/max/noisy-OR or other established aggregation baselines and the proposed mechanism. Measure email recall/FPR, link localization with independently reviewed link labels, legitimate delegated-link false alarms, missing-authentication cases, and benign-link addition/removal robustness. Keep campaigns and duplicate messages isolated, use arrival-time evidence, and reserve a new final external evaluation after development.

A possible research question is: Can preserving context and joint evidence for each hyperlink improve identification of the harmful link and resist benign-link dilution without increasing false alarms on legitimate external-service links?

No uniqueness, patentability, or performance improvement is established by this assessment alone.
