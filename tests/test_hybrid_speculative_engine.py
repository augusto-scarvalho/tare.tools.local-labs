"""Tests for Hybrid Speculative Decoding Engine (SPEC-01)."""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analysis.hybrid_speculative_engine import HybridSpeculativeEngine, NGramTrie


class TestHybridSpeculativeEngine(unittest.TestCase):
    def test_ngram_trie_exact_retrieval(self):
        trie = NGramTrie(min_n=2, max_n=4)
        # Sequence with repeated code block pattern
        seq = [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
        trie.insert_sequence(seq)

        # Context suffix matching [10, 20, 30]
        draft = trie.draft([10, 20, 30], max_draft_len=2)
        self.assertEqual(draft, [40, 50])

    def test_speculative_speedup_on_structured_sequence(self):
        # Simulated target sequence of 100 tokens with repeating 5-token motifs
        target_ground_truth = ([1, 2, 3, 4, 5] * 20) + [999]  # 999 = EOS

        def mock_verifier(context: list[int], draft: list[int]) -> tuple[int, int]:
            curr_len = len(context)
            num_accepted = 0
            for i, tok in enumerate(draft):
                if curr_len + i < len(target_ground_truth) and tok == target_ground_truth[curr_len + i]:
                    num_accepted += 1
                else:
                    break
            next_pos = curr_len + num_accepted
            next_tok = target_ground_truth[next_pos] if next_pos < len(target_ground_truth) else 999
            return num_accepted, next_tok

        def mock_mtp(context: list[int], max_draft: int) -> list[int]:
            # MTP knows the next 2 tokens with 80% accuracy
            curr_len = len(context)
            if curr_len < len(target_ground_truth):
                return target_ground_truth[curr_len : curr_len + max_draft]
            return []

        engine = HybridSpeculativeEngine(
            target_verifier=mock_verifier,
            mtp_proposer=mock_mtp,
            max_draft_len=4,
        )

        prompt = [1, 2, 3, 4, 5]
        result = engine.generate(prompt_tokens=prompt, max_new_tokens=100, eos_token_id=999)

        self.assertGreater(result["mean_tokens_per_step"], 2.0)
        self.assertGreater(result["effective_speedup_factor"], 2.0)
        self.assertGreater(result["ngram_drafts"], 5)
        self.assertEqual(result["output_tokens"][-1], 999)


if __name__ == "__main__":
    unittest.main()
