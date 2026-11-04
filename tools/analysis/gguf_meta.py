"""Read GGUF metadata without loading the model, and without llama.cpp's tools.

Written because `llama-gguf` is not among the targets this project builds, and because a
header read should not depend on a 17 KB launcher that pulls in the whole runtime. It
reads a few KB from the front of the file: no GPU, no model load, safe to run while a
benchmark is in flight.

    python gguf_meta.py model.gguf
    python gguf_meta.py a.gguf b.gguf --keys block_count,expert_count
"""
from __future__ import annotations

import argparse
import struct
import sys

# GGUF value type ids -> (struct code, size). Strings and arrays are handled separately.
_SCALAR = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
           6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}
_STRING, _ARRAY = 8, 9


def _read(f, code, size):
    return struct.unpack("<" + code, f.read(size))[0]


def _read_str(f) -> str:
    n = _read(f, "Q", 8)
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, vtype):
    if vtype in _SCALAR:
        code, size = _SCALAR[vtype]
        return _read(f, code, size)
    if vtype == _STRING:
        return _read_str(f)
    if vtype == _ARRAY:
        etype = _read(f, "I", 4)
        n = _read(f, "Q", 8)
        # Arrays here are tokenizer vocabularies of ~150k strings. Reading them costs
        # seconds and megabytes for information nobody asked for, so skip past instead.
        if etype == _STRING:
            for _ in range(n):
                f.seek(_read(f, "Q", 8), 1)
            return f"<{n} strings>"
        if etype in _SCALAR:
            _c, size = _SCALAR[etype]
            f.seek(size * n, 1)
            return f"<{n} values>"
        raise ValueError(f"array of unsupported type {etype}")
    raise ValueError(f"unsupported gguf type {vtype}")


def read_meta(path: str) -> dict:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = _read(f, "I", 4)
        n_tensors = _read(f, "Q", 8)
        n_kv = _read(f, "Q", 8)
        out = {"_version": version, "_n_tensors": n_tensors}
        for _ in range(n_kv):
            key = _read_str(f)
            out[key] = _read_value(f, _read(f, "I", 4))
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--keys", default=("architecture,block_count,expert_count,"
                                       "expert_used_count,embedding_length,"
                                       "feed_forward_length,expert_feed_forward_length,"
                                       "head_count,head_count_kv,file_type"))
    args = ap.parse_args()
    wanted = [k.strip() for k in args.keys.split(",") if k.strip()]

    rows = []
    for p in args.paths:
        try:
            m = read_meta(p)
        except Exception as exc:                       # noqa: BLE001
            print(f"{p}\tERROR\t{exc}")
            continue
        row = {"file": p.rsplit("/", 1)[-1][:44], "tensors": m.get("_n_tensors")}
        for w in wanted:
            # Keys are architecture-prefixed (`qwen3moe.block_count`), so match on suffix.
            hit = next((v for k, v in m.items() if k.endswith(w)), None)
            row[w] = hit
        rows.append(row)

    cols = ["file", "tensors"] + wanted
    print("\t".join(cols))
    for r in rows:
        print("\t".join("" if r.get(c) is None else str(r[c]) for c in cols))
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        import io
        import tempfile

        # Build a minimal GGUF by hand: the parser must survive strings, scalars and the
        # tokenizer arrays it deliberately skips rather than materialises.
        buf = io.BytesIO()
        buf.write(b"GGUF")
        buf.write(struct.pack("<I", 3))
        buf.write(struct.pack("<Q", 7))          # tensor count
        buf.write(struct.pack("<Q", 3))          # kv count

        def wstr(s):
            b = s.encode()
            buf.write(struct.pack("<Q", len(b))); buf.write(b)

        wstr("general.architecture"); buf.write(struct.pack("<I", _STRING)); wstr("qwen3moe")
        wstr("qwen3moe.block_count"); buf.write(struct.pack("<I", 4)); buf.write(struct.pack("<I", 48))
        wstr("tokenizer.ggml.tokens"); buf.write(struct.pack("<I", _ARRAY))
        buf.write(struct.pack("<I", _STRING)); buf.write(struct.pack("<Q", 2))
        for t in ("a", "bb"):
            b = t.encode(); buf.write(struct.pack("<Q", len(b))); buf.write(b)

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as fh:
            fh.write(buf.getvalue())
            name = fh.name
        m = read_meta(name)
        assert m["general.architecture"] == "qwen3moe", m
        assert m["qwen3moe.block_count"] == 48, m
        assert m["tokenizer.ggml.tokens"] == "<2 strings>", m
        assert m["_n_tensors"] == 7
        try:
            read_meta(__file__)
            raise AssertionError("a non-GGUF file must be rejected, not parsed as garbage")
        except ValueError:
            pass
        print("gguf_meta self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
