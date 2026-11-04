#!/usr/bin/env python3
"""Hybrid Speculative Decoding Engine (N-Gram Trie + Neural MTP Proposer).

Combines lightweight RAM-based n-gram prefix matching with multi-token prediction (MTP)
to maximize token throughput in structured and repetitive sequences.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple


class NGramTrie:
    """Lightweight in-memory n-gram map for zero-GPU-overhead speculative drafting."""

    def __init__(self, min_n: int = 2, max_n: int = 5):
        self.min_n = min_n
        self.max_n = max_n
        # Map: tuple(prefix_tokens) -> list of continuation tokens
        self.ngram_table: Dict[Tuple[int, ...], List[int]] = {}

    def insert_sequence(self, tokens: List[int]) -> None:
        """Indexes all n-grams and their subsequent continuations from the token sequence."""
        seq_len = len(tokens)
        for n in range(self.min_n, self.max_n + 1):
            for i in range(seq_len - n):
                prefix = tuple(tokens[i : i + n])
                continuation = tokens[i + n : i + n + 8]
                if continuation:
                    self.ngram_table[prefix] = continuation

    def draft(self, context_suffix: List[int], max_draft_len: int = 4) -> List[int]:
        """Proposes a continuation draft given the current context suffix."""
        if not context_suffix or not self.ngram_table:
            return []

        # Find longest matching suffix in n-gram table
        for n in range(min(len(context_suffix), self.max_n), self.min_n - 1, -1):
            prefix = tuple(context_suffix[-n:])
            if prefix in self.ngram_table:
                return list(self.ngram_table[prefix][:max_draft_len])

        return []


class HybridSpeculativeEngine:
    """Orchestrates hybrid N-Gram + MTP speculative decoding against a target model."""

    def __init__(
        self,
        target_verifier: Callable[[List[int], List[int]], Tuple[int, int]],
        mtp_proposer: Optional[Callable[[List[int], int], List[int]]] = None,
        max_draft_len: int = 4,
    ):
        self.target_verifier = target_verifier
        self.mtp_proposer = mtp_proposer
        self.max_draft_len = max_draft_len
        self.ngram_trie = NGramTrie(min_n=2, max_n=5)

    def generate(
        self, prompt_tokens: List[int], max_new_tokens: int = 128, eos_token_id: int = 2
    ) -> dict:
        tokens = list(prompt_tokens)
        self.ngram_trie.insert_sequence(tokens)

        target_steps = 0
        total_accepted_tokens = 0
        ngram_draft_count = 0
        mtp_draft_count = 0

        while len(tokens) - len(prompt_tokens) < max_new_tokens:
            # 1. Propose draft: Try N-Gram Trie first
            draft = self.ngram_trie.draft(tokens, max_draft_len=self.max_draft_len)
            source = "ngram" if draft else "none"

            if not draft and self.mtp_proposer is not None:
                # Fallback to Neural MTP Proposer
                draft = self.mtp_proposer(tokens, self.max_draft_len)
                source = "mtp" if draft else "none"

            if source == "ngram":
                ngram_draft_count += 1
            elif source == "mtp":
                mtp_draft_count += 1

            # 2. Target Model Verification Step
            target_steps += 1
            num_accepted, correction_token = self.target_verifier(tokens, draft)

            # 3. Accept valid tokens
            if num_accepted > 0 and draft:
                for i in range(num_accepted):
                    tokens.append(draft[i])
                    if draft[i] == eos_token_id:
                        break

            # 4. Append correction token if not stopped
            if not (tokens and tokens[-1] == eos_token_id):
                tokens.append(correction_token)

            total_accepted_tokens += num_accepted + 1
            self.ngram_trie.insert_sequence(tokens[-self.max_draft_len - 10 :])

            if tokens[-1] == eos_token_id:
                break

        gen_count = len(tokens) - len(prompt_tokens)
        speedup = gen_count / target_steps if target_steps > 0 else 1.0

        return {
            "output_tokens": tokens,
            "generated_count": gen_count,
            "target_verification_steps": target_steps,
            "mean_tokens_per_step": round(speedup, 2),
            "ngram_drafts": ngram_draft_count,
            "mtp_drafts": mtp_draft_count,
            "effective_speedup_factor": round(speedup, 2),
        }
