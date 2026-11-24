"""Optional current-time RCS diagnostics. Never feeds the frozen offline model.

The parent app runs this worker with a wall-clock timeout. Raw mail arrives on
stdin, stays in memory, and is not written to disk. Only public HTTP(S) endpoints
on ports 80/443 are contacted, with redirect validation and DNS-pinned transport.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import ipaddress
import json
import socket
import ssl
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import urllib3

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'phish_dataset_v2'
MAX_LINKS = 5
MAX_REQUESTS = 30
MAX_REDIRECTS = 5
MAX_JSON_BYTES = 128 * 1024


class ScanStopped(ValueError):
    pass


@dataclass
class Response:
    url: str
    status_code: int
    headers: dict
    body: bytes = b''
    history: list = field(default_factory=list)

    def json(self):
        return json.loads(self.body)


class PublicClient:
    def __init__(self):
        self.requests = 0
        self.failures = []
        self.addresses = {}
        self.deadline = time.monotonic() + 35

    def public_ip(self, hostname):
        if hostname not in self.addresses:
            records = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            addresses = [record[4][0] for record in records]
            if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
                raise ScanStopped('Non-public network destination blocked.')
            self.addresses[hostname] = addresses[0]
        return self.addresses[hostname]

    def validate(self, url):
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ScanStopped('Unsupported URL.')
        if parsed.username is not None or parsed.password is not None:
            raise ScanStopped('URLs containing credentials are skipped.')
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        if port not in (80, 443):
            raise ScanStopped('Only ports 80 and 443 are scanned.')
        lowered = unquote(url).lower()
        if any(word in lowered for word in ('unsubscribe', 'optout', 'opt-out', 'logout', 'signout')):
            raise ScanStopped('Unsubscribe/sign-out URL skipped.')
        return parsed, port, self.public_ip(parsed.hostname)

    def fetch_once(self, url, read_json=False):
        if self.requests >= MAX_REQUESTS or time.monotonic() >= self.deadline:
            raise ScanStopped('Live lookup budget exhausted.')
        parsed, port, address = self.validate(url)
        self.requests += 1
        host = parsed.hostname.encode('idna').decode('ascii')
        host_header = f'[{host}]' if ':' in host else host
        if port != (443 if parsed.scheme == 'https' else 80):
            host_header += ':' + str(port)
        headers = {'Host': host_header, 'User-Agent': 'SLiCE-optional-live-RCS/1.0', 'Connection': 'close'}
        if parsed.scheme == 'https':
            pool = urllib3.HTTPSConnectionPool(address, port=port, server_hostname=host,
                assert_hostname=host, ssl_context=ssl.create_default_context())
        else:
            pool = urllib3.HTTPConnectionPool(address, port=port)
        response = None
        try:
            target = urlunsplit(('', '', parsed.path or '/', parsed.query, ''))
            response = pool.urlopen('GET', target, headers=headers, redirect=False,
                retries=False, preload_content=False, timeout=urllib3.Timeout(connect=2, read=3))
            body = b''
            if read_json and response.status == 200:
                body = response.read(MAX_JSON_BYTES + 1)
                if len(body) > MAX_JSON_BYTES:
                    raise ScanStopped('Lookup JSON exceeds the size limit.')
            return Response(url, response.status, dict(response.headers), body)
        finally:
            if response is not None:
                response.close()
            pool.close()

    def get(self, url, **_):
        """A small requests.get-compatible adapter for the legacy extractor."""
        try:
            read_json = urlsplit(url).hostname in ('rdap.org', 'ipinfo.io')
            history = []
            current = url
            for hop in range(MAX_REDIRECTS + 1):
                response = self.fetch_once(current, read_json=read_json)
                location = next((value for key, value in response.headers.items() if key.lower() == 'location'), None)
                if response.status_code in (301, 302, 303, 307, 308) and location:
                    if hop == MAX_REDIRECTS:
                        raise ScanStopped('Redirect limit reached.')
                    current = urljoin(current, location)
                    response.headers['Location'] = current
                    history.append(response)
                    continue
                if response.status_code >= 400:
                    self.failures.append(f'Lookup returned HTTP {response.status_code}.')
                response.history = history
                return response
            raise ScanStopped('Redirect limit reached.')
        except Exception as error:
            self.failures.append(str(error) if isinstance(error, ScanStopped) else f'Lookup failed ({type(error).__name__}).')
            raise


def load_legacy_module():
    path = LEGACY / 'feature_modules/rcs_features.py'
    spec = importlib.util.spec_from_file_location('_slice_optional_rcs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assess(raw):
    from email import policy
    from email.parser import BytesParser

    rcs = load_legacy_module()
    message = BytesParser(policy=policy.default).parsebytes(raw)
    links = rcs.extract_links(rcs.get_html(message))
    base = {'score': None, 'flag': None, 'threshold': None, 'features': {},
        'link_results': [], 'affects_final_score': False,
        'source': 'Legacy phish_dataset_v2 RCS model; current-time measurements',
        'limitations': ['This legacy model is not covered by the codex_attempt leakage audit or accuracy figures.',
            'Current redirects and domain/IP data are not historical arrival-time evidence.',
            'The original abuse-intelligence feature is a placeholder, not a live reputation check.']}
    if not links:
        return dict(base, status='not_applicable', reason='RCS needs HTML links.')
    if len(links) > MAX_LINKS:
        return dict(base, status='not_run', reason=f'Live RCS supports at most {MAX_LINKS} HTML links per email; no links were visited.')
    client = PublicClient()
    try:
        for link in links:
            client.validate(link['url'])
    except Exception as error:
        reason = str(error) if isinstance(error, ScanStopped) else f'URL validation failed ({type(error).__name__}).'
        return dict(base, status='blocked', reason=reason, request_count=0)
    rcs.requests = client  # Scoped to this isolated module; never patches the app.

    def resolve(domain):
        if not domain:
            client.failures.append('Final domain unavailable.')
            return None
        try:
            return client.public_ip(domain)
        except Exception as error:
            client.failures.append(str(error) if isinstance(error, ScanStopped) else 'Final IP unavailable.')
            return None

    original_age = rcs.get_domain_creation_date
    original_metadata = rcs.get_ip_metadata

    def creation_date(domain):
        value = original_age(domain)
        if value is None:
            client.failures.append('Domain registration date unavailable.')
        return value

    def ip_metadata(address):
        value = original_metadata(address)
        if any(item is None for item in value):
            client.failures.append('ASN/country metadata unavailable.')
        return value

    rcs.resolve_domain_to_ip = resolve
    rcs.get_domain_creation_date = creation_date
    rcs.get_ip_metadata = ip_metadata
    # Use the same normalizer as the frozen project, without importing its models.
    spec = importlib.util.spec_from_file_location('_slice_optional_normalizer', ROOT / 'vendor/build_v4/normalize.py')
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    normalized = raw if str(message.get('Authentication-Results', '')).startswith('slice-normalised;') else normalizer.normalise(raw, ['phishing@pot'])[0]
    rcs.open = lambda *_args, **_kwargs: io.BytesIO(normalized)
    features, metadata = rcs.extract_rcs_features('memory.eml')
    base.update(features=features, link_results=metadata['link_results'], request_count=client.requests,
        issues=list(dict.fromkeys(client.failures)), expected_domain=metadata['expected_domain'])
    if client.failures:
        return dict(base, status='partial', reason='Some live measurements are unavailable or blocked. No RCS model score is issued from incomplete measurements.')

    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    artifact_path = LEGACY / 'models/final3/slice_rcs_model.joblib'
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        artifact = joblib.load(artifact_path)
        columns = artifact['feature_columns']
        if any(not column.startswith('RCS__') for column in columns):
            raise ValueError('Legacy artifact contains non-RCS inputs.')
        frame = pd.DataFrame([{'RCS__'+key: value for key, value in features.items()}])
        matrix = frame[columns].apply(pd.to_numeric, errors='coerce').astype('float64').replace([np.inf, -np.inf], np.nan)
        for column, sentinels in artifact['preprocessing']['sentinels'].items():
            if column in matrix:
                matrix.loc[matrix[column].isin(sentinels), column] = np.nan
        with threadpool_limits(limits=4):
            score = float(artifact['model'].predict_proba(matrix)[0, 1])
    threshold = float(artifact['threshold'])
    return dict(base, status='complete', score=score, threshold=threshold, flag=score >= threshold,
        algorithm=type(artifact['model']).__name__, model_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        reason='Optional legacy RCS result only; the offline/STS verdict is unchanged.')


if __name__ == '__main__':
    try:
        result = assess(sys.stdin.buffer.read())
    except Exception as error:
        result = {'status': 'unavailable', 'score': None, 'affects_final_score': False,
                  'reason': f'Live RCS could not finish ({type(error).__name__}).'}
    print(json.dumps(result, allow_nan=False))
