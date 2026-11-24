"""Blinded prospective evaluation. Prepare mail, review labels, then score once.

prepare INPUT_DIR --packet NAME --sampling 'how the incoming sample was selected'
evaluate --packet NAME
All review data and predictions remain under private/prospective/NAME.
"""
import argparse
import csv
import hashlib
import json
import math
import socket
import time
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

from predict import ROOT, extract, predict


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def model_lock():
    path = ROOT / 'models/model.joblib'
    manifest = json.loads((ROOT / 'reports/model_manifest.json').read_text())
    digest = sha(path.read_bytes())
    if digest != manifest['model_sha256']:
        raise ValueError('Frozen model hash differs from the original manifest.')
    return {'model_sha256': digest, 'threshold': manifest['threshold'],
            'minimum_received_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def packet_path(name):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Packet name must contain only letters, numbers, _ or -.')
    return ROOT / 'private/prospective' / name


def prepare(directory, name, sampling):
    destination = packet_path(name)
    if destination.exists():
        raise ValueError('Packet already exists; refusing to replace a locked sample.')
    files = sorted(Path(directory).resolve().glob('*.eml'))
    if not files:
        raise ValueError('No .eml files found. No sample created.')
    lock = model_lock()
    cutoff = datetime.fromisoformat(lock['minimum_received_utc'])
    known_raw, known_ids = set(), set()
    with (ROOT / 'private/offline_rows.jsonl').open(encoding='utf-8') as source:
        for line in source:
            rec = json.loads(line)
            known_raw.add(rec['raw_sha256'])
            if rec.get('message_id'):
                known_ids.add(rec['message_id'])
    accepted, excluded, seen_raw, seen_ids = [], [], set(), set()
    for path in files:
        raw = path.read_bytes()
        message = BytesParser(policy=policy.default).parsebytes(raw)
        digest = sha(raw)
        mid = str(message.get('Message-ID', '')).strip().lower()
        reason = None
        try:
            # Sender Date is deliberately not used as evidence of delivery time.
            received = parsedate_to_datetime(str(message.get('Received', '')).rsplit(';', 1)[-1].strip())
            if received.tzinfo is None:
                raise ValueError('missing timezone')
            received = received.astimezone(timezone.utc)
        except (ValueError, TypeError, IndexError):
            received = None
        if digest in known_raw or (mid and mid in known_ids):
            reason = 'known_development_message'
        elif digest in seen_raw or (mid and mid in seen_ids):
            reason = 'duplicate_in_batch'
        elif received is None:
            reason = 'no_trusted_delivery_timestamp_candidate'
        elif received <= cutoff:
            reason = 'delivery_predates_frozen_model'
        elif received.timestamp() > time.time() + 300:
            reason = 'future_delivery_timestamp'
        if reason:
            excluded.append({'file': str(path), 'reason': reason})
            continue
        seen_raw.add(digest)
        if mid:
            seen_ids.add(mid)
        accepted.append((digest[:20], digest, received.isoformat(), raw, message))
    if not accepted:
        raise ValueError(f'No fresh eligible emails. Exclusion counts: {pd.Series([x["reason"] for x in excluded]).value_counts().to_dict()}')
    (destination / 'raw').mkdir(parents=True)
    (destination / 'review').mkdir()
    rows, locked = [], []
    for identifier, digest, received, raw, message in accepted:
        (destination / 'raw' / (identifier + '.eml')).write_bytes(raw)
        parts = []
        for part in message.walk():
            if part.get_content_maintype() == 'text' and part.get_content_disposition() != 'attachment':
                body = part.get_payload(decode=True)
                if body:
                    try:
                        text = body.decode(part.get_content_charset() or 'utf-8', errors='replace')
                    except LookupError:
                        text = body.decode('utf-8', errors='replace')
                    if part.get_content_subtype() == 'html':
                        soup = BeautifulSoup(text, 'html.parser')
                        for tag in soup(['script', 'style']):
                            tag.decompose()
                        # Preserve targets as text, never execute or fetch them.
                        for tag in soup.find_all('a', href=True):
                            tag.append(' [target: ' + str(tag['href']) + ']')
                        text = soup.get_text(' ', strip=True)
                    parts.append(text)
        headers = '\n'.join(k + ': ' + str(message.get(k, '')) for k in ['From', 'Reply-To', 'Subject', 'Date'])
        (destination / 'review' / (identifier + '.txt')).write_text(headers + '\n\n' + '\n\n'.join(parts), encoding='utf-8')
        rows.append({'email_id': identifier, 'label': '', 'reviewer': '', 'evidence': ''})
        locked.append({'email_id': identifier, 'sha256': digest, 'received_utc': received})
    pd.DataFrame(rows).to_csv(destination / 'labels.csv', index=False)
    dump(destination / 'sample_lock.json', {**lock, 'sampling': sampling, 'rows': locked,
         'created_utc': datetime.now(timezone.utc).isoformat(), 'exclusions': excluded,
         'instructions': 'Independently review each email before scoring: label 0=non-phishing, 1=phishing, uncertain=unresolved. Supply reviewer and evidence. Do not infer labels from Inbox/Spam or authentication alone. Receiver timestamp authenticity must be established by collection provenance.'})
    print(f'Prepared {len(rows)} blinded review emails; no model predictions computed. Review: {destination / "labels.csv"}')


def proportion(successes, total):
    if not total:
        return {'value': None, 'wilson_95pct': None, 'denominator': 0}
    p = successes / total
    z = 1.959963984540054
    center = (p + z*z / (2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return {'value': p, 'wilson_95pct': [max(0, center-half), min(1, center+half)], 'denominator': total}


def evaluate(name):
    directory = packet_path(name)
    if (directory / 'metrics.json').exists():
        raise ValueError('Packet already evaluated; do not reuse its labels to retune.')
    lock = json.loads((directory / 'sample_lock.json').read_text())
    current = model_lock()
    if lock['model_sha256'] != current['model_sha256'] or lock['threshold'] != current['threshold']:
        raise ValueError('Model or threshold changed after sample lock.')
    labels = pd.read_csv(directory / 'labels.csv', dtype=str).fillna('')
    if labels.email_id.duplicated().any() or set(labels.email_id) != {r['email_id'] for r in lock['rows']}:
        raise ValueError('Review IDs must match the entire locked sample exactly.')
    if not labels.label.isin(['0', '1', 'uncertain']).all() or not labels.reviewer.str.strip().ne('').all() or not labels.evidence.str.strip().ne('').all():
        raise ValueError('Every sampled email needs label 0, 1 or uncertain, a reviewer and evidence; review before scoring.')
    # Freeze ground truth before loading the model or computing any predictions.
    dump(directory / 'ground_truth_lock.json', {'labels_sha256': sha((directory / 'labels.csv').read_bytes()), 'model_sha256': current['model_sha256']})
    labels = labels.set_index('email_id')
    bundle = joblib.load(ROOT / 'models/model.joblib')
    rows = []
    def deny(*args, **kwargs):
        raise RuntimeError('Network access is disabled during evaluation.')
    socket.create_connection = deny
    socket.getaddrinfo = deny
    requests.sessions.Session.request = deny
    with threadpool_limits(limits=4):
        for record in lock['rows']:
            identifier = record['email_id']
            raw = (directory / 'raw' / (identifier + '.eml')).read_bytes()
            if sha(raw) != record['sha256']:
                raise ValueError('Raw email changed after sample lock.')
            start = time.perf_counter()
            frame, links = extract(raw)
            output = predict(frame, bundle).iloc[0] if links else None
            rows.append({'email_id': identifier, 'label': labels.loc[identifier, 'label'], 'in_scope': links > 0,
                         'score': float(output.final_score) if links else None,
                         'flag': int(output.flag) if links else None, 'latency_ms': (time.perf_counter()-start)*1000})
    frame = pd.DataFrame(rows)
    scored = frame[frame.in_scope & frame.label.isin(['0', '1'])].copy()
    y = scored.label.astype(int).to_numpy()
    flag = scored.flag.astype(int).to_numpy()
    tp, tn = int(((y==1)&(flag==1)).sum()), int(((y==0)&(flag==0)).sum())
    fp, fn = int(((y==0)&(flag==1)).sum()), int(((y==1)&(flag==0)).sum())
    result = {'status': 'observed prospective sample only; no universal real-world accuracy claim',
              'sampling': lock['sampling'], 'total_sampled': len(frame), 'out_of_scope': int((~frame.in_scope).sum()),
              'uncertain': int((frame.label=='uncertain').sum()), 'scored_labelled': len(scored),
              'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': proportion(tp+tn,len(scored)),
              'precision': proportion(tp,tp+fp), 'recall': proportion(tp,tp+fn), 'fpr': proportion(fp,fp+tn),
              'roc_auc': float(roc_auc_score(y,scored.score)) if len(set(y))==2 else None,
              'pr_auc': float(average_precision_score(y,scored.score)) if len(set(y))==2 else None,
              'model_sha256': current['model_sha256'], 'threshold': current['threshold'],
              'latency_median_ms': float(scored.latency_ms.median()) if len(scored) else None,
              'latency_p95_ms': float(scored.latency_ms.quantile(.95)) if len(scored) else None,
              'limitations': ['Generalization is limited to the sampled source and period.', 'Wilson intervals assume independent emails; campaigns can make them optimistic.', 'One-class or small-positive samples cannot establish reliable phishing detection performance.', 'Uncertain and linkless emails are excluded from classifier metrics and reported separately.']}
    if current['model_sha256'] != model_lock()['model_sha256']:
        raise ValueError('Model changed during evaluation.')
    frame.to_csv(directory / 'predictions.csv', index=False)
    dump(directory / 'metrics.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'evaluate'])
    parser.add_argument('--input-dir')
    parser.add_argument('--packet', required=True)
    parser.add_argument('--sampling')
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not args.input_dir or not args.sampling:
            parser.error('prepare requires --input-dir and --sampling')
        prepare(args.input_dir, args.packet, args.sampling)
    else:
        evaluate(args.packet)
