"""Frozen new-source false-positive evaluation, never a phishing accuracy test."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
from pathlib import Path
import json
import hashlib
import time
import tarfile
import socket
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import joblib
import pandas as pd
import numpy as np
import requests
from threadpoolctl import threadpool_limits

from predict import ROOT, extract, normalise, predict
from train import fingerprints
from real_world_eval import proportion

DATA = ROOT / 'private/external/spamassassin'
REPORT = ROOT / 'reports/external_spamassassin'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def no_network(*args, **kwargs):
    raise RuntimeError('Network disabled during extraction and evaluation')


def worker(item):
    key, archive, raw = item
    socket.create_connection = no_network
    socket.getaddrinfo = no_network
    requests.sessions.Session.request = no_network
    norm, _ = normalise(raw, ['phishing@pot'])
    fp = fingerprints(norm)
    t = time.perf_counter()
    frame, links = extract(norm)
    return {'key': key, 'archive': archive, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
            **fp, 'links': links, 'features': frame.iloc[0].to_dict(),
            'extraction_ms_parallel': (time.perf_counter()-t)*1000}


def main():
    REPORT.mkdir(exist_ok=True)
    if (REPORT / 'metrics.json').exists():
        raise RuntimeError('External test already evaluated; refusing to replace results.')
    manifest = json.loads((ROOT / 'reports/model_manifest.json').read_text())
    source_hash = digest(ROOT / 'models/model.joblib')
    assert source_hash == manifest['model_sha256']
    protocol = {'type': 'new-source retrospective legitimate-only test',
        'model_sha256': source_hash, 'threshold': manifest['threshold'],
        'source': 'Apache SpamAssassin 20030228 easy_ham, easy_ham_2, hard_ham',
        'label': 'Publisher non-spam (ham), interpreted as non-phishing',
        'selection': 'Every email member of the three downloaded ham archives, no score-based selection',
        'overlap': 'Exclude matching normalized SHA256, Message-ID or normalized body fingerprint against all 10,622 development extraction records; deduplicate the new batch on the same keys',
        'scope': 'Only emails with links extracted by the unchanged feature pipeline; linkless coverage reported separately',
        'limitations': ['Legitimate-only data cannot estimate phishing recall, precision, ROC-AUC or overall two-class accuracy.',
           'Historical 2002-2003 mail predates modern authentication and is not a representative current mailbox.',
           'Publisher labels are inherited, not newly independently hand-verified.',
           'Canonical-body exclusion can conservatively remove related messages, not just byte-identical duplicates.'],
        'archives': json.loads((DATA / 'download.json').read_text())}
    protocol_path = REPORT / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    known = {k:set() for k in ['raw_sha256','message_id','body_fingerprint']}
    with (ROOT / 'private/offline_rows.jsonl').open(encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            for k in known:
                if rec.get(k): known[k].add(rec[k])
    items = []
    for spec in protocol['archives']:
        archive = DATA / spec['name']
        assert digest(archive) == spec['sha256']
        with tarfile.open(archive, 'r:bz2') as tar:
            for member in sorted(tar.getmembers(), key=lambda x:x.name):
                if not member.isfile() or Path(member.name).name == 'cmds':
                    continue
                # Read members in memory; never execute or extract archive paths.
                raw = tar.extractfile(member).read()
                items.append((spec['name']+'::'+member.name, spec['name'], raw))
    print(f'Loaded {len(items)} publisher-labelled ham emails', flush=True)
    cache_path = DATA / 'features.jsonl'
    records = {}
    if cache_path.exists():
        for line in cache_path.open(encoding='utf-8'):
            rec = json.loads(line); records[rec['key']] = rec
    pending = [item for item in items if item[0] not in records]
    with cache_path.open('a',encoding='utf-8') as f, ProcessPoolExecutor(max_workers=4) as pool:
        for rec in pool.map(worker,pending,chunksize=8):
            records[rec['key']] = rec
            f.write(json.dumps(rec,default=str)+'\n');f.flush()
            if len(records)%250 == 0: print(f'Processed {len(records)}/{len(items)}',flush=True)
    seen = {k:set() for k in known}
    kept, exclusions = [], []
    for key, archive, _ in items:
        rec = records[key]; reason = None
        for k in known:
            if rec.get(k) and rec[k] in known[k]: reason = 'development_overlap_'+k; break
        if reason is None:
            for k in seen:
                if rec.get(k) and rec[k] in seen[k]: reason = 'within_batch_repeat_'+k; break
        if reason:
            exclusions.append({'key':key,'reason':reason});continue
        for k in seen:
            if rec.get(k): seen[k].add(rec[k])
        kept.append(rec)
    pd.DataFrame(exclusions,columns=['key','reason']).to_csv(DATA/'exclusions.csv',index=False)
    scoped = [rec for rec in kept if rec['links']>0]
    if not scoped: raise ValueError('No compatible link-bearing emails; cannot compute in-scope FPR.')
    frame = pd.DataFrame([r['features'] for r in scoped])
    bundle = joblib.load(ROOT/'models/model.joblib')
    with threadpool_limits(limits=4):
        t=time.perf_counter(); out=predict(frame,bundle); batch_ms=(time.perf_counter()-t)*1000
    predictions = pd.concat([pd.DataFrame([{'key':r['key'],'archive':r['archive'],'label':0} for r in scoped]),out],axis=1)
    predictions.to_csv(DATA/'predictions.csv',index=False)
    fp = int(out.flag.sum()); total=len(out)
    metrics={'test_type':protocol['type'],'source_emails':len(items),'after_overlap_exclusion':len(kept),
        'exclusions':dict(Counter(r['reason'] for r in exclusions)),
        'in_scope':total,'linkless_out_of_scope':len(kept)-total,'in_scope_coverage':total/len(kept),
        'false_positives':fp,'true_negatives':total-fp,'false_positive_rate':proportion(fp,total),
        'correctly_allowed_legitimate_fraction':proportion(total-fp,total),
        'phishing_recall':None,'phishing_precision':None,'overall_two_class_accuracy':None,
        'model_sha256':source_hash,'threshold':bundle['threshold'],'batch_inference_ms':batch_ms,
        'by_archive':{},'limitations':protocol['limitations']}
    for name, group in predictions.groupby('archive'):
        count=len(group);flags=int(group.flag.sum())
        metrics['by_archive'][name]={'in_scope':count,'false_positives':flags,'fpr':proportion(flags,count)}
    assert digest(ROOT/'models/model.joblib')==source_hash
    (REPORT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    rate=metrics['false_positive_rate'];lo,hi=rate['wilson_95pct']
    lines=['# New-source SpamAssassin ham test','',
       f"Downloaded {len(items)} labelled legitimate emails; retained {len(kept)} after overlap checks.",
       f"Link-bearing, in-scope: {total}; linkless, out of scope: {len(kept)-total}.",
       f"False alarms: {fp}/{total} ({100*rate['value']:.2f}%; Wilson 95% interval {100*lo:.2f}% to {100*hi:.2f}%).",'',
       'This measures external false-positive behavior only. There are no labelled phishing emails in this test, so phishing recall, precision and overall two-class accuracy are unavailable.','',
       'Model and threshold were unchanged. No email links were fetched.','',
       'Source: https://spamassassin.apache.org/old/publiccorpus/readme.html','',*protocol['limitations']]
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__': main()
