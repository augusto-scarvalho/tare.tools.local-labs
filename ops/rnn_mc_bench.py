#!/usr/bin/env python
"""
RNN-04 MQAR benchmark (Multi-Query Associative Recall), packet sections 5/9/10.

Design goals:
- Process-STABLE reproducibility. Per-example RNG seeds derive from hashlib.blake2b of a canonical
  spec+index string, NOT Python builtin hash() (the RNN-08b defect). Two fresh Python processes with the
  same MQARSpec produce byte-identical examples and identical canonical IDs.
- Every sample has a stable canonical ID (blake2b of its token stream + labels).
- Independently variable axes (section 9): seq_len, num_pairs, num_queries, distractor_density
  (write->query distance is a recorded consequence of layout).
- In-context binding only: keys are re-randomized per example and the same key id maps to different
  values across examples, so no global key->value shortcut exists (section 10, "no lexical shortcut").
- Answer values never appear in the query region of the input (no leak); targets are placed at the query
  key positions (Zoology / Arora 2024a convention).

Token id layout (fixed):
  PAD=0  BOS=1  QSEP=2  FILL=3
  KEY ids in [KEY_LO, KEY_LO+num_keys)      VAL ids in [VAL_LO, VAL_LO+num_vals)
Runnable:
  python rnn_mc_bench.py --selftest --out <json>          # section 10 self-qualification (+cross-process)
  python rnn_mc_bench.py --emit-hashes --spec '<json>' --n N   # prints JSON list of example IDs (subproc)
"""
import argparse, hashlib, json, subprocess, sys
from dataclasses import dataclass, asdict
import numpy as np

PAD, BOS, QSEP, FILL = 0, 1, 2, 3
KEY_LO = 4


@dataclass(frozen=True)
class MQARSpec:
    seq_len: int
    num_pairs: int
    num_queries: int
    distractor_density: float = 0.0   # fraction of gap positions that are hard (key-like) distractors
    num_keys: int = 64
    num_vals: int = 64
    name: str = "mqar"                # part of the seed identity

    @property
    def val_lo(self):
        return KEY_LO + self.num_keys

    @property
    def vocab_size(self):
        return KEY_LO + self.num_keys + self.num_vals

    def canonical(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _seed_for(spec: MQARSpec, idx: int) -> int:
    h = hashlib.blake2b((spec.canonical() + f"|idx={idx}").encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")   # stable across processes/machines (unlike builtin hash())


def _example_id(input_ids, labels) -> str:
    payload = json.dumps([input_ids, labels], separators=(",", ":")).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def make_example(spec: MQARSpec, idx: int) -> dict:
    """Build one MQAR example. Returns exact-length (seq_len) input_ids + labels (no padding)."""
    if spec.num_queries > spec.num_pairs:
        raise ValueError("num_queries > num_pairs")
    if spec.num_pairs > spec.num_keys:
        raise ValueError("num_pairs > num_keys")
    rng = np.random.default_rng(_seed_for(spec, idx))

    key_perm = rng.permutation(spec.num_keys)
    written_keys = (key_perm[:spec.num_pairs] + KEY_LO).tolist()
    values = (rng.integers(0, spec.num_vals, size=spec.num_pairs) + spec.val_lo).tolist()
    kv = dict(zip(written_keys, values))

    # write region: BOS then k,v pairs
    write = [BOS]
    write_pos = {}
    for k, v in zip(written_keys, values):
        write_pos[k] = len(write)
        write += [k, v]

    # query region: QSEP then queried keys (each on its own position; target at that position)
    q_pair_idx = rng.permutation(spec.num_pairs)[:spec.num_queries]      # random query order (no copy shortcut)
    queried_keys = [written_keys[i] for i in q_pair_idx]
    query = [QSEP] + queried_keys

    gap = spec.seq_len - len(write) - len(query)
    if gap < 0:
        raise ValueError(f"seq_len {spec.seq_len} too short for spec (needs >= {len(write)+len(query)})")

    # gap fillers: neutral FILL + hard key-like distractors (keys NOT written, never followed by a value)
    n_hard = int(round(spec.distractor_density * gap))
    free_keys = (key_perm[spec.num_pairs:] + KEY_LO).tolist()
    hard = [free_keys[i % len(free_keys)] for i in range(n_hard)] if free_keys else [FILL] * n_hard
    gap_tokens = hard + [FILL] * (gap - n_hard)
    rng.shuffle(gap_tokens)

    input_ids = write + gap_tokens + query
    assert len(input_ids) == spec.seq_len

    labels = [-100] * spec.seq_len
    q_start = len(write) + len(gap_tokens) + 1   # skip QSEP
    answer_positions, pairs = [], []
    for j, k in enumerate(queried_keys):
        p = q_start + j
        labels[p] = kv[k]
        answer_positions.append(p)
        pairs.append(dict(key=int(k), value=int(kv[k]), write_pos=int(write_pos[k]),
                          query_pos=int(p), distance=int(p - write_pos[k])))

    return dict(input_ids=input_ids, labels=labels, attention_mask=[1] * spec.seq_len,
                answer_positions=answer_positions, pairs=pairs,
                example_id=_example_id(input_ids, labels), idx=idx, val_lo=spec.val_lo,
                num_vals=spec.num_vals)


def build_dataset(spec: MQARSpec, n: int, start: int = 0):
    return [make_example(spec, i) for i in range(start, start + n)]


# ---------- scorer (section 10) ----------
def score_predictions(pred_value_ids, examples):
    """pred_value_ids[ex_i][pos] -> predicted token id at each answer position. Accuracy over answers."""
    correct = total = 0
    for e, preds in zip(examples, pred_value_ids):
        for pos, pv in zip(e["answer_positions"], preds):
            total += 1
            correct += int(pv == e["labels"][pos])
    return correct / total if total else 0.0


def oracle_predictions(examples):
    return [[e["labels"][p] for p in e["answer_positions"]] for e in examples]


def random_predictions(examples, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for e in examples:
        out.append([int(rng.integers(0, e["num_vals"]) + e["val_lo"]) for _ in e["answer_positions"]])
    return out


# ---------- section 10 self-qualification ----------
def _emit_hashes(spec: MQARSpec, n: int):
    print(json.dumps([make_example(spec, i)["example_id"] for i in range(n)]))


def _cross_process_check(spec: MQARSpec, n: int):
    """Spawn TWO fresh subprocesses; compare example IDs. This is the real section-5 process-stability proof."""
    cmd = [sys.executable, __file__, "--emit-hashes", "--spec", spec.canonical(), "--n", str(n)]
    a = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    b = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    in_proc = json.dumps([make_example(spec, i)["example_id"] for i in range(n)])
    return dict(proc_a_eq_proc_b=(a == b), inproc_eq_proc=(json.loads(in_proc) == json.loads(a)),
                n=n, sample_ids=json.loads(a)[:3])


def run_selftest(out_path):
    spec = MQARSpec(seq_len=256, num_pairs=16, num_queries=8, distractor_density=0.3, name="selftest")
    ex = build_dataset(spec, 64)
    checks = {}

    # 1 known-good oracle PASS
    checks["oracle_pass"] = dict(acc=score_predictions(oracle_predictions(ex), ex), expect=1.0)
    # 2 known-bad random FAIL (well below oracle; ~1/num_vals)
    racc = score_predictions(random_predictions(ex, seed=7), ex)
    checks["random_fail"] = dict(acc=round(racc, 4), expect_below=0.15, chance=round(1 / spec.num_vals, 4))
    # 3 cross-process determinism (fresh subprocesses)
    checks["cross_process"] = _cross_process_check(spec, 32)
    # 4 requested seq_len == delivered
    checks["length_exact"] = dict(all_match=all(len(e["input_ids"]) == spec.seq_len for e in ex),
                                  seq_len=spec.seq_len)
    # 5 kv pair survives construction (written key at write_pos, value right after; query key present)
    surv = True
    for e in ex:
        for pr in e["pairs"]:
            wp = pr["write_pos"]
            surv &= (e["input_ids"][wp] == pr["key"] and e["input_ids"][wp + 1] == pr["value"]
                     and e["input_ids"][pr["query_pos"]] == pr["key"])
    checks["kv_survives"] = dict(ok=bool(surv))
    # 6 scorer correctness: perturbed prediction must score < 1 and exact must score 1
    bad = [[(-1) for _ in e["answer_positions"]] for e in ex]
    checks["scorer_correct"] = dict(exact=score_predictions(oracle_predictions(ex), ex),
                                    perturbed=score_predictions(bad, ex))
    # 7 no positional/candidate-order leak: value NOT in the input query/gap region; target varies by which
    #   key is queried (shuffled order) -> the answer cannot be read off a fixed position.
    leak = False
    for e in ex:
        qregion = set(e["input_ids"][min(e["answer_positions"]):])  # from first answer pos to end
        for p in e["answer_positions"]:
            leak |= (e["labels"][p] in qregion)   # true value token appearing in the query region = leak
    checks["no_answer_leak"] = dict(leak_detected=bool(leak))
    # 8 no lexical shortcut: same key id maps to >1 distinct value across the dataset
    kmap = {}
    for e in ex:
        for pr in e["pairs"]:
            kmap.setdefault(pr["key"], set()).add(pr["value"])
    multi = sum(1 for v in kmap.values() if len(v) > 1)
    checks["no_lexical_shortcut"] = dict(keys_seen=len(kmap), keys_with_multiple_values=multi,
                                         ok=(multi >= max(1, len(kmap) // 2)))

    passed = (
        abs(checks["oracle_pass"]["acc"] - 1.0) < 1e-9 and
        checks["random_fail"]["acc"] < checks["random_fail"]["expect_below"] and
        checks["cross_process"]["proc_a_eq_proc_b"] and checks["cross_process"]["inproc_eq_proc"] and
        checks["length_exact"]["all_match"] and checks["kv_survives"]["ok"] and
        abs(checks["scorer_correct"]["exact"] - 1.0) < 1e-9 and checks["scorer_correct"]["perturbed"] < 1.0 and
        (not checks["no_answer_leak"]["leak_detected"]) and checks["no_lexical_shortcut"]["ok"]
    )
    result = dict(
        packet="RNN-04", component="MQAR benchmark self-qualification (section 10)",
        SYNTHETIC_DATASET_REPRODUCIBILITY="QUALIFIED" if checks["cross_process"]["proc_a_eq_proc_b"]
        and checks["cross_process"]["inproc_eq_proc"] else "FAILED",
        BENCHMARK_SELFTEST="PASS" if passed else "FAIL",
        selftest_spec=asdict(spec), numpy=np.__version__, python=sys.version.split()[0],
        seed_method="hashlib.blake2b(canonical_spec+idx) -> int (process-stable; NOT builtin hash())",
        checks=checks,
    )
    if out_path:
        json.dump(result, open(out_path, "w"), indent=2)
    print(json.dumps({k: result[k] for k in ["SYNTHETIC_DATASET_REPRODUCIBILITY", "BENCHMARK_SELFTEST"]},
                     indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-hashes", action="store_true")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    if a.emit_hashes:
        _emit_hashes(MQARSpec(**json.loads(a.spec)), a.n)
    else:
        run_selftest(a.out)
