"""Render existing frozen reports; never fit models or change predictions."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
load=lambda name:json.loads((R/'reports'/name).read_text(encoding='utf-8'))
m=load('metrics.json'); a=load('leakage_audit.json'); l=load('latency.json'); manifest=load('model_manifest.json')
x=m['test']; lines=['# Measured results','', 'This is a stricter offline retrospective rerun on previously exposed project data, not an untouched external evaluation.','', '| Metric | Test result | 95% group-bootstrap interval |','|---|---:|---:|']
for k in ['accuracy','precision','recall','f1','fpr','roc_auc','pr_auc']:
 lo,hi=m['cluster_bootstrap_95pct'][k]
 lines.append(f'| {k} | {x[k]:.4f} | {lo:.4f} to {hi:.4f} |' if k in ('roc_auc','pr_auc') else f'| {k} | {100*x[k]:.2f}% | {100*lo:.2f}% to {100*hi:.2f}% |')
lines+=['',f"Test n={x['n']}; TN={x['tn']}, FP={x['fp']}, FN={x['fn']}, TP={x['tp']}. Threshold={x['threshold']:.6f}.",
 f"Resampled template/duplicate components: {m['test_groups']}. Confidence intervals resample these groups, not individual correlated emails.",'', '| Partition | Legitimate | Phishing |','|---|---:|---:|']
counts=a['counts']
for p in ['train','tune','calibration','threshold','test']:
 lines.append(f"| {p} | {counts.get('0',{}).get(p,0)} | {counts.get('1',{}).get(p,0)} |")
lines+=['',f"Excluded rows: {a['excluded_rows']}. Every audited cross-partition group overlap is zero.",'', '| Latency | Median | p95 |','|---|---:|---:|']
for k in ['single_email_warm_model','batch_amortized_per_email','offline_feature_extraction','raw_email_to_verdict_warm']:
 v=l[k]; lines.append(f"| {k} | {v['median_ms']:.2f} ms | {v['p95_ms']:.2f} ms |")
lines+=['',f"Model reload with OS cache warm: {l['reload_ms_os_cache_warm']:.2f} ms. Threads: {l['threads']}.",l['timing_notes'],'',
 f"Hypothetical precision at 1% prevalence: {100*m['precision_if_prevalence_0.01']:.2f}%; at 0.1%: {100*m['precision_if_prevalence_0.001']:.2f}%.",'',
 '## Limitations','']+['- '+v for v in m['limitations']]
lines+=['','Do not claim a leakage-free real-world accuracy from these data. A newly collected, prospectively locked, independently labelled evaluation is still required.','']
raw=load('raw_latency.json')
threshold=load('threshold_selection.json')
verification=load('verification.json')
lines+=['','## Original raw-email latency','', '| Stage | Median | p95 |','|---|---:|---:|']
for k in ['original_raw_feature_extraction','original_raw_model_inference','original_raw_read_to_verdict','fresh_process_read_to_verdict']:
 v=raw[k]; lines.append(f"| {k} | {v['median_ms']:.2f} ms | {v['p95_ms']:.2f} ms |")
lines += ['',raw['notes'],f"Original-vs-normalized score difference: {raw['normalized_vs_original_max_score_difference']}; network attempts: {raw['network_attempts']}.",'',
 '## Threshold and comparison cautions','',
 f"The threshold partition contained {threshold['tn']+threshold['fp']} legitimate emails and {threshold['tp']+threshold['fn']} phishing emails. It had {threshold['fp']} false positives. The empirical <=1% selection target was not a statistical guarantee: the later test FPR was {100*x['fpr']:.2f}%.",'',
 'This run changes the feature set, search grid, calibration protocol and training population relative to the legacy model. Its performance difference cannot be attributed entirely to data leakage. The main test window was already exposed during earlier project development; source bias and historical cohort selection remain.', '',
 f"Verification status: {verification['status']}. Final model and threshold were not changed after test evaluation.",'']
(R/'reports/RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
