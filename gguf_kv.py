"""Read a few GGUF metadata keys without loading the model.

The layer count sets the TOP of the offload axis, and loading 21 GB to learn one integer
is the kind of cost this project exists to avoid. Stdlib only -- no numpy, no gguf-py.
"""
import struct, sys

T_UINT8,T_INT8,T_UINT16,T_INT16,T_UINT32,T_INT32,T_FLOAT32,T_BOOL,T_STRING,T_ARRAY,T_UINT64,T_INT64,T_FLOAT64 = range(13)
FIX = {T_UINT8:('<B',1),T_INT8:('<b',1),T_UINT16:('<H',2),T_INT16:('<h',2),T_UINT32:('<I',4),
       T_INT32:('<i',4),T_FLOAT32:('<f',4),T_BOOL:('<?',1),T_UINT64:('<Q',8),T_INT64:('<q',8),
       T_FLOAT64:('<d',8)}

def read(f, want):
    magic = f.read(4)
    assert magic == b'GGUF', f"not a GGUF file: {magic!r}"
    _ver, _ntensor, nkv = struct.unpack('<IQQ', f.read(20))
    out = {}
    def val(t):
        if t in FIX:
            fmt, n = FIX[t]
            return struct.unpack(fmt, f.read(n))[0]
        if t == T_STRING:
            (n,) = struct.unpack('<Q', f.read(8))
            return f.read(n).decode('utf-8', 'replace')
        if t == T_ARRAY:
            et, n = struct.unpack('<IQ', f.read(12))
            return [val(et) for _ in range(n)]
        raise ValueError(f"unknown gguf type {t}")
    for _ in range(nkv):
        (klen,) = struct.unpack('<Q', f.read(8))
        key = f.read(klen).decode('utf-8', 'replace')
        (t,) = struct.unpack('<I', f.read(4))
        v = val(t)
        if any(w in key for w in want):
            out[key] = v if not isinstance(v, list) else f"[{len(v)} items]"
    return out

if __name__ == "__main__":
    want = ("block_count", "expert", "embedding_length", "feed_forward",
            "attention.head_count", "context_length", "architecture", "name")
    with open(sys.argv[1], 'rb') as f:
        for k, v in read(f, want).items():
            print(f"{k:<48} {v}")
