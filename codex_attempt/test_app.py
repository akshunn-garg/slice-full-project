"""App integration checks; these do not evaluate or tune the model."""
import hashlib
import io
import unittest
from unittest.mock import patch

from threadpoolctl import threadpool_limits

import app as web


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = web.load_model()
        cls.fixtures = web.ROOT.parent / 'phish_dataset_v2/tests/fixtures/inbox'
        cls.digest = hashlib.sha256(web.MODEL_PATH.read_bytes()).hexdigest()

    def setUp(self):
        self.client = web.app.test_client()

    def upload(self, raw, name='test.eml'):
        return self.client.post('/analyze', data={'eml': (io.BytesIO(raw), name)})

    def test_home_and_health(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'codex_attempt / frozen model', page.data)
        self.assertEqual(page.headers['Cache-Control'], 'no-store')
        health = self.client.get('/health').get_json()
        self.assertEqual(health['app'], 'codex_attempt')
        self.assertTrue(health['ready'])
        self.assertEqual(health['model_id'], self.digest[:12])

    def test_score_parity_and_offline_extraction(self):
        raw = (self.fixtures / '02_newsletter.eml').read_bytes()
        with patch('socket.create_connection', side_effect=AssertionError('Network forbidden')) as connection, \
             patch('socket.getaddrinfo', side_effect=AssertionError('DNS forbidden')) as dns, \
             patch('requests.sessions.Session.request', side_effect=AssertionError('HTTP forbidden')) as http:
            response = self.upload(raw)
            self.assertEqual(response.status_code, 200, response.data)
            result = response.get_json()
            frame, count = web.extract(raw)
            with threadpool_limits(limits=4):
                expected = web.predict(frame, self.bundle).iloc[0]
            connection.assert_not_called()
            dns.assert_not_called()
            http.assert_not_called()
        self.assertEqual(result['link_count'], count)
        self.assertEqual(result['threshold'], self.bundle['threshold'])
        self.assertEqual(result['flag'], bool(expected.flag))
        for score in ['final_score', 'link_score', 'sts_risk']:
            self.assertAlmostEqual(result[score], float(expected[score]), places=12)
        for component in result['components']:
            self.assertAlmostEqual(component['score'], float(expected[component['name']+'_score']), places=12)
        self.assertEqual(hashlib.sha256(web.MODEL_PATH.read_bytes()).hexdigest(), self.digest)

    def test_linkless_email_has_no_verdict(self):
        raw = (self.fixtures / '03_plaintext_no_links.eml').read_bytes()
        with patch.object(web, 'predict', side_effect=AssertionError('Out of scope')):
            response = self.upload(raw)
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertFalse(result['in_scope'])
        self.assertEqual(result['verdict'], 'OUTSIDE MODEL SCOPE')
        self.assertIsNone(result['final_score'])
        self.assertIsNone(result['flag'])

    def test_missing_authentication_is_unknown(self):
        raw = b'From: Sender <sender@example.com>\r\nContent-Type: text/html\r\n\r\n<a href="https://example.com/">Example</a>'
        response = self.upload(raw)
        self.assertEqual(response.status_code, 200)
        for evidence in response.get_json()['authentication']:
            self.assertEqual(evidence['state'], 'unknown')

    def test_invalid_empty_oversized_and_busy(self):
        self.assertEqual(self.client.post('/analyze').status_code, 400)
        self.assertEqual(self.upload(b'text', 'file.html').status_code, 400)
        self.assertEqual(self.upload(b'').status_code, 400)
        with patch.dict(web.app.config, {'MAX_CONTENT_LENGTH': 100}):
            self.assertEqual(self.upload(b'x'*200).status_code, 413)
        web.ANALYSIS_LOCK.acquire()
        try:
            self.assertEqual(self.upload(b'From: sender@example.com\r\n\r\nHello').status_code, 409)
        finally:
            web.ANALYSIS_LOCK.release()


if __name__ == '__main__':
    unittest.main(verbosity=2)
