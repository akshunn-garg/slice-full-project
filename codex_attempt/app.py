"""Local UI for the frozen codex_attempt model. Run: python app.py --port 5001."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import time
import webbrowser
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from threadpoolctl import threadpool_limits

from predict import ROOT, FAMILIES, extract, predict, src

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
MODEL_PATH = ROOT / 'models/model.joblib'
MODEL_LOCK = threading.Lock()
ANALYSIS_LOCK = threading.Lock()
STATE = {}


def load_model():
    with MODEL_LOCK:
        if 'bundle' not in STATE:
            manifest = json.loads((ROOT / 'reports/model_manifest.json').read_text(encoding='utf-8'))
            fingerprint = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
            if fingerprint != manifest['model_sha256']:
                raise RuntimeError('The model differs from the frozen model manifest.')
            bundle = joblib.load(MODEL_PATH)
            if float(bundle['threshold']) != float(manifest['threshold']):
                raise RuntimeError('The model threshold differs from its manifest.')
            algorithms = {}
            for family, model in bundle['bases'].items():
                estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
                algorithms[family] = type(estimator).__name__
            STATE.update(bundle=bundle, sha256=fingerprint, algorithms=algorithms)
    return STATE['bundle']


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def analyse(raw):
    bundle = load_model()
    started = time.perf_counter()
    frame, count = extract(raw)  # Exactly the same extraction path as predict.py.
    extracted = time.perf_counter()
    message = BytesParser(policy=policy.default).parsebytes(raw)
    headers = {k: str(message.get(k, ''))[:500] for k in ['From', 'To', 'Reply-To', 'Return-Path', 'Subject', 'Date']}
    row = frame.iloc[0].to_dict()
    authentication = []
    for kind in ['spf', 'dkim', 'dmarc']:
        result = str(row[f'STS_meta__{kind}_result']).lower()
        authentication.append({'name': kind.upper(), 'result': result,
            'state': 'pass' if result == 'pass' else ('unknown' if result in ['none', 'unknown', '', 'nan', 'neutral', 'temperror'] else 'other')})
    urls = src.extract_links(src.get_html(message))
    links = []
    for number, url in enumerate(urls, 1):
        try:
            domain = urlsplit(url).hostname or ''
        except ValueError:
            domain = ''
        links.append({'number': number, 'url': url, 'domain': domain})
    result = {'in_scope': count > 0, 'link_count': count, 'headers': headers,
              'authentication': authentication, 'links': links,
              'threshold': float(bundle['threshold']), 'final_score': None, 'link_score': None,
              'sts_risk': None, 'sts_trust': None, 'flag': None, 'components': [],
              'verdict': 'OUTSIDE MODEL SCOPE', 'features': {},
              'model_id': STATE['sha256'][:12], 'mode': 'Offline · codex_attempt'}
    model_started = time.perf_counter()
    if count:
        with threadpool_limits(limits=4):
            output = predict(frame, bundle).iloc[0]
        result.update(final_score=float(output.final_score), link_score=float(output.link_score),
                      sts_risk=float(output.sts_risk), sts_trust=1-float(output.sts_risk),
                      flag=bool(output.flag), verdict='MODEL FLAGGED' if output.flag else 'BELOW MODEL THRESHOLD')
        result['components'] = [{'name': family, 'score': float(output[family+'_score']),
              'algorithm': STATE['algorithms'][family], 'feature_count': len(bundle['features'][family])} for family in FAMILIES]
    model_ended = time.perf_counter()
    for family in ['BAS', 'DMS', 'LOS', 'SRC', 'STS']:
        result['features'][family] = [{'name': k.split('__',1)[1], 'value': value} for k,value in row.items() if k.startswith(family+'__')]
    result['timing_ms'] = {'features': (extracted-started)*1000,
        'models': (model_ended-model_started)*1000, 'total': (time.perf_counter()-started)*1000}
    return serializable(result)


@app.after_request
def response_headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'"
    return response


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/health')
def health():
    return jsonify(app='codex_attempt', ready='bundle' in STATE,
                   model_id=STATE.get('sha256', '')[:12], offline=True)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='The maximum upload size is 10 MB.'), 413


@app.post('/analyze')
def analyze_route():
    uploaded = request.files.get('eml')
    if uploaded is None or not (uploaded.filename or '').lower().endswith('.eml'):
        return jsonify(error='Choose a raw .eml email file.'), 400
    raw = uploaded.read()
    if not raw.strip():
        return jsonify(error='The email file is empty.'), 400
    if not ANALYSIS_LOCK.acquire(blocking=False):
        return jsonify(error='Another email is being analysed. Wait for it to finish and try again.'), 409
    try:
        result = analyse(raw)
        result['filename'] = uploaded.filename
        return jsonify(result)
    except Exception as error:
        app.logger.error('Analysis failed (%s)', type(error).__name__)
        return jsonify(error=f'Unable to analyse this email ({type(error).__name__}).'), 500
    finally:
        ANALYSIS_LOCK.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    print('Loading the frozen codex_attempt offline model...', flush=True)
    load_model()
    url = f'http://127.0.0.1:{args.port}'
    print(f'Ready: {url} | risk threshold {STATE["bundle"]["threshold"]:.6f}', flush=True)
    if not args.no_browser:
        threading.Timer(1, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True, use_reloader=False)
