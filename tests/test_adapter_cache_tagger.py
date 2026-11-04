from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "analysis"))

from adapter_cache_tagger import AdapterCacheTagger, run_multi_tenant_simulation


class TestAdapterCacheTagger(unittest.TestCase):
    def test_key_isolation_across_adapters(self):
        tokens = (1, 2, 3, 4, 5)
        key_math = AdapterCacheTagger.compute_block_key("qwen", "lokr_math", tokens)
        key_code = AdapterCacheTagger.compute_block_key("qwen", "lokr_code", tokens)
        key_base = AdapterCacheTagger.compute_block_key("qwen", None, tokens)

        self.assertNotEqual(key_math, key_code)
        self.assertNotEqual(key_math, key_base)
        self.assertNotEqual(key_code, key_base)

    def test_cache_hit_same_adapter(self):
        tagger = AdapterCacheTagger(block_size=4, max_blocks=10)
        tokens = (10, 20, 30, 40)
        hit1, block1 = tagger.lookup_block("m1", "ad1", tokens)
        hit2, block2 = tagger.lookup_block("m1", "ad1", tokens)

        self.assertFalse(hit1)
        self.assertTrue(hit2)
        self.assertEqual(block1, block2)

    def test_simulation_zero_collisions(self):
        res = run_multi_tenant_simulation(num_requests=50, seed=123)
        self.assertEqual(res["stats"]["cross_adapter_collisions_detected"], 0)
        self.assertGreater(res["prefix_hit_rate_pct"], 70.0)


if __name__ == "__main__":
    unittest.main()
