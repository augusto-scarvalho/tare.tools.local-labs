import sys, re, gguf
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
    names = [t.name for t in r.tensors]
    nextn = [n for n in names if any(x in n.lower() for x in ("nextn", "mtp", "eh_proj", "enorm", "hnorm"))]
    blks = set(int(m.group(1)) for n in names for m in [re.search(r"blk\.(\d+)\.", n)] if m)
    print("  arch=", arch, " n_tensors=", len(names),
          " max_blk=", (max(blks) if blks else None),
          " n_blocks=", (len(blks) if blks else None))
    print("  nextn/mtp tensors:", len(nextn))
    for n in nextn[:10]:
        print("    ", n)
