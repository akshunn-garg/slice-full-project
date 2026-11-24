"""Live RCS transport and model integration, with synthetic network responses."""
import json
import socket
import unittest
from unittest.mock import Mock, patch

import live_rcs as rcs


def address_records(hostname, *_args, **_kwargs):
    address = '127.0.0.1' if hostname in ('127.0.0.1', 'localhost') else '8.8.8.8'
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 0))]


def email_with_links(count=1):
    return ('From: Sender <sender@example.com>\r\nContent-Type: text/html\r\n\r\n' +
            ''.join(f'<a href="https://example.com/{i}">https://example.com/{i}</a>' for i in range(count))).encode()


class RCSTests(unittest.TestCase):
    def test_private_destination_blocked_before_http(self):
        with patch.object(rcs.socket, 'getaddrinfo', side_effect=address_records), \
             patch.object(rcs.urllib3, 'HTTPSConnectionPool') as pool:
            with self.assertRaises(rcs.ScanStopped):
                rcs.PublicClient().get('https://127.0.0.1/')
            pool.assert_not_called()

    def test_redirect_to_private_destination_is_blocked(self):
        first = Mock(status=302, headers={'Location': 'https://127.0.0.1/internal'})
        pool = Mock()
        pool.urlopen.return_value = first
        with patch.object(rcs.socket, 'getaddrinfo', side_effect=address_records), \
             patch.object(rcs.urllib3, 'HTTPSConnectionPool', return_value=pool) as factory:
            with self.assertRaises(rcs.ScanStopped):
                rcs.PublicClient().get('https://example.com/')
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.call_args.args[0], '8.8.8.8')
            self.assertEqual(factory.call_args.kwargs['server_hostname'], 'example.com')
            self.assertEqual(factory.call_args.kwargs['assert_hostname'], 'example.com')

    def test_limits_and_unsubscribe_skip_without_http(self):
        with patch.object(rcs.socket, 'getaddrinfo', side_effect=AssertionError('No DNS expected')):
            self.assertEqual(rcs.assess(email_with_links(6))['status'], 'not_run')
        with patch.object(rcs.urllib3, 'HTTPSConnectionPool') as pool:
            raw = email_with_links().replace(b'example.com/0', b'example.com/unsubscribe')
            self.assertEqual(rcs.assess(raw)['status'], 'blocked')
            pool.assert_not_called()

    def test_complete_legacy_score_with_mocked_live_measurements(self):
        def pool_factory(*_args, **_kwargs):
            pool = Mock()
            def get(_method, _target, **kwargs):
                host = kwargs['headers']['Host']
                data = ({'events': [{'eventAction': 'registration', 'eventDate': '2020-01-01T00:00:00Z'}]}
                        if host == 'rdap.org' else {'org': 'AS15169 Example', 'country': 'US'})
                response = Mock(status=200, headers={})
                response.read.return_value = json.dumps(data).encode()
                return response
            pool.urlopen.side_effect = get
            return pool
        with patch.object(rcs.socket, 'getaddrinfo', side_effect=address_records), \
             patch.object(rcs.urllib3, 'HTTPSConnectionPool', side_effect=pool_factory):
            result = rcs.assess(email_with_links())
        self.assertEqual(result['status'], 'complete', result)
        self.assertGreaterEqual(result['score'], 0)
        self.assertLessEqual(result['score'], 1)
        self.assertEqual(result['flag'], result['score'] >= result['threshold'])
        self.assertEqual(result['request_count'], 3)
        self.assertEqual(len(result['features']), 9)
        self.assertFalse(result['affects_final_score'])

    def test_incomplete_lookup_does_not_produce_model_score(self):
        def unavailable_date(_client, url, read_json=False):
            if 'rdap.org' in url:
                return rcs.Response(url, 503, {})
            return rcs.Response(url, 200, {}, json.dumps({'org': 'AS15169 Example', 'country': 'US'}).encode())
        with patch.object(rcs.socket, 'getaddrinfo', side_effect=address_records), \
             patch.object(rcs.PublicClient, 'fetch_once', unavailable_date):
            result = rcs.assess(email_with_links())
        self.assertEqual(result['status'], 'partial')
        self.assertIsNone(result['score'])
        self.assertTrue(result['issues'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
