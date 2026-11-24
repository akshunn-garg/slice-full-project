"""Export and evaluate frozen score models; never refit or select using test labels.

python individual_scores.py --export-evaluate
python individual_scores.py --family BAS --email path/to/email.eml
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from predict import ROOT, FAMILIES, RULE_WEIGHTS, extract, numeric, predict, rule_risk
from train import choose_threshold, metric


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict_individual(frame, family, model_dir=None):
    directory = Path(model_dir or ROOT / 'models' / 'individual')
    family = family.upper()
    if family == 'STS':
        config = json.loads((directory / 'sts_rules.json').read_text())
        score = np.asarray(rule_risk(frame), dtype=float)
        threshold = config['threshold']
    else:
        art = joblib.load(directory / f'{family.lower()}_model.joblib')
        score = art['model'].predict_proba(numeric(frame, art['feature_columns']))[:, 1]
        threshold = art['threshold']
    return pd.DataFrame({'score': score, 'flag': (score >= threshold).astype(int)}, index=frame.index)


def export_evaluate():
    destination = ROOT / 'reports' / 'individual_scores.json'
    if destination.exists():
        raise RuntimeError('Individual results already exist; refusing to overwrite or retune.')
    bundle_path = ROOT / 'models' / 'model.joblib'
    original_hash = digest(bundle_path)
    manifest = json.loads((ROOT / 'reports' / 'model_manifest.json').read_text())
    assert original_hash == manifest['model_sha256']
    bundle = joblib.load(bundle_path)
    data = pd.read_pickle(ROOT / 'private' / 'prepared.pkl')
    threshold_data = data[data.partition == 'threshold'].reset_index(drop=True)
    folder = ROOT / 'models' / 'individual'
    folder.mkdir(exist_ok=True)
    # All operating points are committed using THRESHOLD only before test reporting.
    threshold_scores = predict(threshold_data, bundle)
    scores = {family: threshold_scores[family + '_score'].to_numpy() for family in FAMILIES}
    scores['STS'] = threshold_scores.sts_risk.to_numpy()
    scores['LINK'] = threshold_scores.link_score.to_numpy()
    thresholds = {name: choose_threshold(threshold_data.label.to_numpy(), p) for name, p in scores.items()}
    algorithms = {}
    for family in FAMILIES:
        model = bundle['bases'][family]
        estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
        algorithms[family] = type(estimator).__name__
        artifact = {
            'family': family, 'model': model, 'feature_columns': bundle['features'][family],
            'threshold': thresholds[family], 'source_bundle_sha256': original_hash,
            'threshold_selection': 'Max recall at empirical FPR <=1%, existing THRESHOLD partition only.',
            'score_interpretation': 'Uncalibrated classifier score, not a deployment phishing probability.',
            'protocol': bundle['protocol'],
        }
        joblib.dump(artifact, folder / f'{family.lower()}_model.joblib', compress=3)
    (folder / 'sts_rules.json').write_text(json.dumps({
        'weights': RULE_WEIGHTS, 'threshold': thresholds['STS'],
        'implementation': 'predict.rule_risk', 'source_bundle_sha256': original_hash,
        'note': 'Fixed rule score, not a trained ML model; domain age is excluded.',
    }, indent=2), encoding='utf-8')
    (ROOT / 'reports' / 'individual_thresholds.json').write_text(json.dumps({
        'thresholds': thresholds, 'negative_count': int((threshold_data.label == 0).sum()),
        'positive_count': int((threshold_data.label == 1).sum()),
        'selection_partition': 'threshold', 'source_bundle_sha256': original_hash,
    }, indent=2), encoding='utf-8')

    # Reuse predictions from the already completed frozen test evaluation.
    saved = pd.read_csv(ROOT / 'private' / 'test_predictions.csv')
    test = data[data.partition == 'test'].reset_index(drop=True)
    assert test.email_id.tolist() == saved.email_id.tolist()
    y = test.label.to_numpy()
    np.testing.assert_array_equal(y, saved.label.to_numpy())
    rows = []
    for family in (*FAMILIES, 'STS', 'LINK'):
        column = family + '_score' if family in FAMILIES else ('sts_risk' if family == 'STS' else 'link_score')
        p = saved[column].to_numpy()
        result = metric(y, p, thresholds[family])
        result['precision_defined'] = bool(result['tp'] + result['fp'])
        if not result['precision_defined']:
            result['precision'] = None
        if family in FAMILIES:
            art = joblib.load(folder / f'{family.lower()}_model.joblib')
            def run(frame):
                return art['model'].predict_proba(numeric(frame, art['feature_columns']))[:, 1]
            np.testing.assert_allclose(run(test), p, atol=1e-12, rtol=1e-12)
            times = []
            for _ in range(5):
                run(test.iloc[:1])
            for i in range(50):
                frame = test.iloc[[i % len(test)]]
                t = time.perf_counter()
                run(frame)
                times.append((time.perf_counter() - t) * 1000)
            result['warm_model_median_ms'] = float(np.median(times))
            result['warm_model_p95_ms'] = float(np.percentile(times, 95))
            result['feature_count'] = len(art['feature_columns'])
        rows.append({'family': family, 'algorithm': algorithms.get(family, 'fixed rules' if family == 'STS' else 'LightGBM + Platt'), **result})
    final = json.loads((ROOT / 'reports' / 'metrics.json').read_text())['test']
    rows.append({'family': 'FINAL', 'algorithm': 'LINK + fixed STS weight 3', **final})
    assert digest(bundle_path) == original_hash
    report = {
        'source_bundle_sha256': original_hash, 'models_refitted': False,
        'combined_threshold_changed': False, 'test_rows': len(test),
        'notes': [
            'Descriptive evaluation of the existing frozen components, not fresh independent validation.',
            'Individual thresholds use 55 legitimate threshold emails; empirical FPR <=1% does not guarantee test/deployment FPR <=1%.',
            'Individual latency excludes extraction, file reads and model loading; 50 warm single-row calls, four threads.',
            'RCS was excluded because its live-lookup evidence was unavailable at historical email arrival.',
            'STS is a fixed rule, not an ML score model. Component scores are inputs to the stack; individual flags are supplementary.',
        ],
        'results': rows,
    }
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(ROOT / 'reports' / 'individual_scores.csv', index=False)
    lines = ['# Individual frozen score models', '', *report['notes'], '',
             '| Model | Accuracy | Precision | Recall | FPR | ROC-AUC | Warm model median |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        precision_text = f"{100*row['precision']:.2f}%" if row['precision'] is not None else 'N/A (no flags)'
        latency = f"{row['warm_model_median_ms']:.2f} ms" if 'warm_model_median_ms' in row else 'See full latency report'
        lines.append(f"| {row['family']} | {100*row['accuracy']:.2f}% | {precision_text} | {100*row['recall']:.2f}% | {100*row['fpr']:.2f}% | {row['roc_auc']:.4f} | {latency} |")
    (ROOT / 'reports' / 'INDIVIDUAL_SCORES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-evaluate', action='store_true')
    parser.add_argument('--family', choices=[*FAMILIES, 'STS'])
    parser.add_argument('--email')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    with threadpool_limits(limits=4):
        if args.export_evaluate:
            export_evaluate()
        elif args.family and args.email:
            frame, n = extract(Path(args.email).read_bytes())
            result = predict_individual(frame, args.family).iloc[0].to_dict()
            print(json.dumps({'family': args.family, 'in_scope': n > 0, **result}, indent=2))
        else:
            parser.error('Use --export-evaluate or --family FAMILY --email FILE')


if __name__ == '__main__':
    main()
