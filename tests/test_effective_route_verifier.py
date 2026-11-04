from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "analysis"))

from effective_route_verifier import audit_endpoint, compute_hash


RUNTIME_EVIDENCE = {
    "show": {"ActiveState": "active", "SubState": "running"},
    "effective_exec_start": "{ argv[]=/opt/b10159/bin/llama-server -m /models/fable-tc-l1.0-Q4_K_M.gguf ; }",
    "effective_model_path": "/models/fable-tc-l1.0-Q4_K_M.gguf",
    "main_pid": 123,
    "process_cmdline": "/opt/b10159/bin/llama-server -m /models/fable-tc-l1.0-Q4_K_M.gguf",
    "collection_errors": [],
}


class TestEffectiveRouteVerifier(unittest.TestCase):
    def test_compute_hash_determinism(self):
        data1 = {"a": 1, "b": [2, 3]}
        data2 = {"b": [2, 3], "a": 1}
        self.assertEqual(compute_hash(data1), compute_hash(data2))

    @patch("effective_route_verifier.http_get_json")
    @patch("effective_route_verifier.http_post_json")
    def test_audit_verified_route(self, mock_post, mock_get):
        mock_get.side_effect = lambda url: {
            "http://127.0.0.1:8080/props": {
                "model_path": "/models/fable-tc-l1.0-Q4_K_M.gguf",
                "build_info": "b10159-test",
                "total_slots": 1,
                "default_generation_settings": {},
            },
            "http://127.0.0.1:8080/health": {"status": "ok", "system_fingerprint": "b10159"},
            "http://127.0.0.1:8080/slots": [{"id": 0, "state": "idle", "n_ctx": 8192}],
        }.get(url)

        mock_post.return_value = {
            "choices": [{"message": {"content": "route-receipt-ok"}}],
            "usage": {"total_tokens": 10},
            "timings": {"predicted_ms": 50.0},
        }

        receipt = audit_endpoint(
            "http://127.0.0.1:8080",
            expected_model_substring="fable",
            runtime_evidence=RUNTIME_EVIDENCE,
        )
        self.assertEqual(receipt.verdict, "VERIFIED")
        self.assertEqual(len(receipt.divergences), 0)
        self.assertEqual(receipt.levels["exercised"]["status"], "EXERCISED")

    @patch("effective_route_verifier.http_get_json")
    @patch("effective_route_verifier.http_post_json")
    def test_audit_model_mismatch_divergence(self, mock_post, mock_get):
        mock_get.side_effect = lambda url: {
            "http://127.0.0.1:8080/props": {"model_path": "/models/other-model.gguf"},
            "http://127.0.0.1:8080/health": {"status": "ok"},
            "http://127.0.0.1:8080/slots": [],
        }.get(url)

        mock_post.return_value = {"choices": [{"message": {"content": "ok"}}]}

        mismatch_runtime = dict(RUNTIME_EVIDENCE, effective_model_path="/models/other-model.gguf")
        receipt = audit_endpoint(
            "http://127.0.0.1:8080",
            expected_model_substring="fable",
            runtime_evidence=mismatch_runtime,
        )
        self.assertEqual(receipt.verdict, "DIVERGENT")
        self.assertTrue(any("MODEL_MISMATCH" in d for d in receipt.divergences))

    @patch("effective_route_verifier.http_get_json")
    @patch("effective_route_verifier.http_post_json")
    def test_missing_runtime_evidence_fails_closed(self, mock_post, mock_get):
        mock_get.side_effect = lambda url: {
            "http://127.0.0.1:8080/props": {"model_path": "/models/fable.gguf", "total_slots": 1},
            "http://127.0.0.1:8080/health": {"status": "ok"},
            "http://127.0.0.1:8080/slots": [{"id": 0, "n_ctx": 8192, "is_processing": False}],
        }.get(url)
        mock_post.return_value = {"choices": [{"message": {"content": "route-receipt-ok"}}]}
        receipt = audit_endpoint("http://127.0.0.1:8080", "fable", runtime_evidence=None)
        self.assertEqual(receipt.verdict, "DIVERGENT")
        self.assertTrue(any("RUNTIME_EVIDENCE_MISSING" in item for item in receipt.divergences))


if __name__ == "__main__":
    unittest.main()
