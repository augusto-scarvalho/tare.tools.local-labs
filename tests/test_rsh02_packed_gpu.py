from __future__ import annotations

import numpy as np

from tools.research.rsh02_packed_gpu_worker import (
    build_lookup,
    canonical_lsb_codes,
    decode_huffman_cpu,
    huffman_lengths,
    pack_huffman,
    pack_int4,
)


def test_physical_huffman_roundtrip_multiple_blocks() -> None:
    symbols = np.asarray(([0] * 50 + [1] * 30 + [-1] * 20 + [2] * 10 + [-2] * 5) * 4, dtype=np.int8)
    lengths = huffman_lengths(symbols)
    codes = canonical_lsb_codes(lengths)
    lookup_symbols, lookup_lengths = build_lookup(codes)
    packed, offsets = pack_huffman(symbols, codes, block_symbols=128)
    decoded = decode_huffman_cpu(packed, offsets, len(symbols), lookup_symbols, lookup_lengths, block_symbols=128)
    assert np.array_equal(decoded, symbols)
    assert len(packed) > 3
    assert len(offsets) == 4


def test_signed_int4_is_physically_two_per_byte() -> None:
    symbols = np.asarray([-7, -1, 0, 1, 7, 3], dtype=np.int8)
    packed = pack_int4(symbols)
    assert len(packed) == 3
    decoded = []
    for byte in packed:
        for nibble in (byte & 0xF, byte >> 4):
            decoded.append(nibble - 16 if nibble >= 8 else nibble)
    assert decoded == symbols.tolist()
