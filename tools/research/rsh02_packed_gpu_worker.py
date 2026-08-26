#!/usr/bin/env python3
"""Physical block-Huffman and INT4 Triton decode benchmark for RSH-02."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import pathlib
import time
from collections import Counter
from typing import Iterable

BLOCK_SYMBOLS = 128
LOOKUP_BITS = 12
TENSOR_KEYS = (
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.language_model.layers.1.mlp.gate_proj.weight",
    "model.language_model.layers.1.mlp.up_proj.weight",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def huffman_lengths(symbols: Iterable[int]) -> dict[int, int]:
    counts = Counter(int(value) for value in symbols)
    heap = []
    serial = 0
    for symbol, count in sorted(counts.items()):
        heapq.heappush(heap, (count, serial, [symbol]))
        serial += 1
    lengths = {symbol: 0 for symbol in counts}
    if len(heap) == 1:
        return {next(iter(counts)): 1}
    while len(heap) > 1:
        left_count, _, left = heapq.heappop(heap)
        right_count, _, right = heapq.heappop(heap)
        for symbol in left + right:
            lengths[symbol] += 1
        heapq.heappush(heap, (left_count + right_count, serial, left + right))
        serial += 1
    return lengths


def reverse_bits(value: int, width: int) -> int:
    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def canonical_lsb_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    code = 0
    previous = 0
    result = {}
    for symbol, width in sorted(lengths.items(), key=lambda item: (item[1], item[0])):
        code <<= width - previous
        result[symbol] = (reverse_bits(code, width), width)
        code += 1
        previous = width
    return result


def build_lookup(codes: dict[int, tuple[int, int]], lookup_bits: int = LOOKUP_BITS):
    import numpy as np

    symbols = np.zeros(1 << lookup_bits, dtype=np.int8)
    lengths = np.zeros(1 << lookup_bits, dtype=np.uint8)
    for symbol, (code, width) in codes.items():
        if width > lookup_bits:
            raise ValueError(f"code length {width} exceeds lookup width {lookup_bits}")
        for suffix in range(1 << (lookup_bits - width)):
            index = code | (suffix << width)
            symbols[index] = symbol
            lengths[index] = width
    if (lengths == 0).any():
        raise ValueError("incomplete prefix lookup table")
    return symbols, lengths


def pack_huffman(symbols, codes: dict[int, tuple[int, int]], block_symbols: int = BLOCK_SYMBOLS):
    import numpy as np

    values = np.asarray(symbols, dtype=np.int8)
    output = bytearray()
    offsets = []
    for block_start in range(0, len(values), block_symbols):
        offsets.append(len(output))
        accumulator = 0
        bits = 0
        for raw in values[block_start:block_start + block_symbols]:
            code, width = codes[int(raw)]
            accumulator |= code << bits
            bits += width
            while bits >= 8:
                output.append(accumulator & 0xFF)
                accumulator >>= 8
                bits -= 8
        if bits:
            output.append(accumulator & 0xFF)
    output.extend(b"\0\0\0")
    return bytes(output), np.asarray(offsets, dtype=np.uint32)


def decode_huffman_cpu(bitstream: bytes, offsets, count: int, lookup_symbols, lookup_lengths, block_symbols: int = BLOCK_SYMBOLS):
    import numpy as np

    output = np.empty(count, dtype=np.int8)
    mask = len(lookup_symbols) - 1
    for block, byte_offset in enumerate(offsets):
        bit_position = 0
        start = block * block_symbols
        end = min(count, start + block_symbols)
        for index in range(start, end):
            absolute = int(byte_offset) + (bit_position >> 3)
            word = bitstream[absolute] | (bitstream[absolute + 1] << 8) | (bitstream[absolute + 2] << 16)
            lookup = (word >> (bit_position & 7)) & mask
            output[index] = lookup_symbols[lookup]
            bit_position += int(lookup_lengths[lookup])
    return output


def pack_int4(symbols) -> bytes:
    import numpy as np

    values = np.asarray(symbols, dtype=np.int8)
    if len(values) % 2:
        values = np.concatenate([values, np.zeros(1, dtype=np.int8)])
    low = values[0::2].astype(np.int16) & 0xF
    high = values[1::2].astype(np.int16) & 0xF
    return (low | (high << 4)).astype(np.uint8).tobytes()


def quantize_weights(model_file: pathlib.Path):
    import numpy as np
    import torch
    from safetensors import safe_open

    quantized = []
    identities = []
    with safe_open(str(model_file), framework="pt", device="cpu") as handle:
        for key in TENSOR_KEYS:
            tensor = handle.get_tensor(key).contiguous()
            raw = tensor.view(torch.int16).numpy().tobytes() if tensor.dtype == torch.bfloat16 else tensor.numpy().tobytes()
            values = tensor.float().reshape(-1, 64).numpy()
            scales = np.maximum(np.max(np.abs(values), axis=1, keepdims=True) / 7.0, 1e-12)
            q = np.clip(np.rint(values / scales), -7, 7).astype(np.int8).reshape(-1)
            quantized.append(q)
            identities.append({"key": key, "shape": list(tensor.shape), "elements": tensor.numel(), "tensor_sha256": sha256_bytes(raw), "symbol_sha256": sha256_bytes(q.tobytes())})
    return np.concatenate(quantized), identities


def benchmark_gpu(symbols, bitstream: bytes, offsets, lookup_symbols, lookup_lengths, int4_bytes: bytes, batches: int, iterations: int):
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    globals()["triton"] = triton
    globals()["tl"] = tl

    @triton.jit
    def huffman_kernel(stream, block_offsets, table_symbols, table_lengths, output, n_elements: tl.constexpr, block_symbols: tl.constexpr, lookup_mask: tl.constexpr):
        program = tl.program_id(0)
        output_base = program * block_symbols
        byte_base = tl.load(block_offsets + program).to(tl.int64)
        bit_position = 0
        for index in tl.static_range(0, block_symbols):
            absolute = byte_base + (bit_position >> 3)
            byte0 = tl.load(stream + absolute).to(tl.int32)
            byte1 = tl.load(stream + absolute + 1).to(tl.int32)
            byte2 = tl.load(stream + absolute + 2).to(tl.int32)
            word = byte0 | (byte1 << 8) | (byte2 << 16)
            lookup = (word >> (bit_position & 7)) & lookup_mask
            symbol = tl.load(table_symbols + lookup)
            width = tl.load(table_lengths + lookup).to(tl.int32)
            tl.store(output + output_base + index, symbol, mask=output_base + index < n_elements)
            bit_position += width

    @triton.jit
    def int4_kernel(stream, output, n_elements: tl.constexpr, block: tl.constexpr):
        offsets_out = tl.program_id(0) * block + tl.arange(0, block)
        packed = tl.load(stream + offsets_out // 2, mask=offsets_out < n_elements, other=0).to(tl.int32)
        nibble = tl.where((offsets_out & 1) == 0, packed & 0xF, (packed >> 4) & 0xF)
        signed = tl.where(nibble >= 8, nibble - 16, nibble)
        tl.store(output + offsets_out, signed, mask=offsets_out < n_elements)

    device = "cuda"
    stream_gpu = torch.from_numpy(np.frombuffer(bitstream, dtype=np.uint8).copy()).to(device)
    offsets_gpu = torch.from_numpy(offsets.astype(np.int64)).to(device)
    table_symbols_gpu = torch.from_numpy(lookup_symbols).to(device)
    table_lengths_gpu = torch.from_numpy(lookup_lengths).to(device)
    int4_gpu = torch.from_numpy(np.frombuffer(int4_bytes, dtype=np.uint8).copy()).to(device)
    huffman_out = torch.empty(len(symbols), dtype=torch.int8, device=device)
    int4_out = torch.empty_like(huffman_out)
    huffman_grid = (len(offsets),)
    int4_block = 256
    int4_grid = (triton.cdiv(len(symbols), int4_block),)

    def launch_huffman():
        huffman_kernel[huffman_grid](stream_gpu, offsets_gpu, table_symbols_gpu, table_lengths_gpu, huffman_out, n_elements=len(symbols), block_symbols=BLOCK_SYMBOLS, lookup_mask=(1 << LOOKUP_BITS) - 1, num_warps=1)

    def launch_int4():
        int4_kernel[int4_grid](int4_gpu, int4_out, n_elements=len(symbols), block=int4_block, num_warps=4)

    for _ in range(25):
        launch_huffman(); launch_int4()
    torch.cuda.synchronize()
    expected = torch.from_numpy(symbols).to(device)
    launch_huffman(); launch_int4(); torch.cuda.synchronize()
    huffman_exact = bool(torch.equal(huffman_out, expected))
    int4_exact = bool(torch.equal(int4_out, expected))

    def timed(launch):
        values = []
        for _ in range(batches):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                launch()
            end.record(); torch.cuda.synchronize()
            values.append(start.elapsed_time(end) / iterations)
        return values

    huffman_ms = timed(launch_huffman)
    int4_ms = timed(launch_int4)
    return {
        "huffman_exact": huffman_exact,
        "int4_exact": int4_exact,
        "huffman_batch_ms": huffman_ms,
        "int4_batch_ms": int4_ms,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=pathlib.Path, required=True)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    import numpy as np

    args.outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    symbols, tensor_identities = quantize_weights(args.model_file)
    lengths = huffman_lengths(symbols)
    codes = canonical_lsb_codes(lengths)
    lookup_symbols, lookup_lengths = build_lookup(codes)
    bitstream, offsets = pack_huffman(symbols, codes)
    cpu_decoded = decode_huffman_cpu(bitstream, offsets, len(symbols), lookup_symbols, lookup_lengths)
    if not np.array_equal(cpu_decoded, symbols):
        raise RuntimeError("CPU reference decoder mismatch")
    int4_bytes = pack_int4(symbols)

    symbols_path = args.outdir / "symbols.npy"
    packed_path = args.outdir / "huffman_packed.bin"
    offsets_path = args.outdir / "huffman_offsets.npy"
    int4_path = args.outdir / "int4_packed.bin"
    np.save(symbols_path, symbols, allow_pickle=False)
    packed_path.write_bytes(bitstream)
    np.save(offsets_path, offsets, allow_pickle=False)
    int4_path.write_bytes(int4_bytes)
    gpu = benchmark_gpu(symbols, bitstream, offsets, lookup_symbols, lookup_lengths, int4_bytes, args.batches, args.iterations)
    if not gpu["huffman_exact"] or not gpu["int4_exact"]:
        raise RuntimeError(f"GPU decode mismatch: {gpu}")

    huffman_median = float(np.median(gpu["huffman_batch_ms"]))
    int4_median = float(np.median(gpu["int4_batch_ms"]))
    codebook_bytes = lookup_symbols.nbytes + lookup_lengths.nbytes
    physical_huffman_bytes = len(bitstream) + offsets.nbytes + codebook_bytes
    metrics = {
        "actual_model_weight_elements": int(len(symbols)),
        "physical_packed_bitstream": True,
        "exact_roundtrip_rate": 1.0,
        "physical_bits_per_element": physical_huffman_bytes * 8.0 / len(symbols),
        "raw_huffman_bits_per_element": sum(Counter(symbols)[symbol] * width for symbol, width in lengths.items()) / len(symbols),
        "huffman_latency_ms": huffman_median,
        "int4_latency_ms": int4_median,
        "decoder_input_throughput_gbs": (physical_huffman_bytes / (huffman_median / 1000.0)) / 1e9,
        "int4_input_throughput_gbs": (len(int4_bytes) / (int4_median / 1000.0)) / 1e9,
        "latency_penalty_vs_int4": huffman_median / int4_median,
        "physical_huffman_bytes": physical_huffman_bytes,
        "physical_int4_bytes": len(int4_bytes),
        "offset_bytes": offsets.nbytes,
        "codebook_bytes": codebook_bytes,
    }
    result = {
        "schema": "rsh02-packed-gpu-worker-v1",
        "model_file": str(args.model_file),
        "model_file_sha256": sha256_file(args.model_file),
        "tensor_identities": tensor_identities,
        "symbol_histogram": {str(key): int(value) for key, value in sorted(Counter(symbols).items())},
        "huffman_code_lengths": {str(key): value for key, value in sorted(lengths.items())},
        "metrics": metrics,
        "timings": gpu,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (symbols_path, packed_path, offsets_path, int4_path)
        },
        "elapsed_seconds": time.time() - started,
    }
    (args.outdir / "worker.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "timings": gpu}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
