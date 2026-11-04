import sys, re, gguf


def quant_name(value):
    try:
        return gguf.GGMLQuantizationType(int(value)).name
    except (TypeError, ValueError):
        return str(value)
for f in sys.argv[1:]:
    print("=====", f, "=====")
    try:
        r = gguf.GGUFReader(f)
    except Exception as e:
        print("  ERR:", e); continue
    arch = None
    for k, fld in r.fields.items():
        if k == "general.architecture":
            arch = bytes(fld.parts[fld.data[0]]).decode(errors="ignore")
    tensors = {t.name: t for t in r.tensors}
    names = list(tensors)
    nextn = [n for n in names if any(x in n.lower() for x in ("nextn", "mtp", "eh_proj", "enorm", "hnorm"))]
    blks = set(int(m.group(1)) for n in names for m in [re.search(r"blk\.(\d+)\.", n)] if m)
    print("  arch=", arch, " n_tensors=", len(names),
          " max_blk=", (max(blks) if blks else None),
          " n_blocks=", (len(blks) if blks else None))
    print("  nextn/mtp tensors:", len(nextn))
    for n in nextn[:10]:
        tensor = tensors[n]
        print("    ", n, "type=", quant_name(tensor.tensor_type),
              "shape=", tuple(int(v) for v in tensor.shape))
    if "output.weight" in tensors:
        tensor = tensors["output.weight"]
        print("  output.weight type=", quant_name(tensor.tensor_type),
              "shape=", tuple(int(v) for v in tensor.shape))
