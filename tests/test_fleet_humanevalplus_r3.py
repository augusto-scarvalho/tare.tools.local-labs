import copy

from tools.research import run_fleet_humanevalplus as r1
from tools.research import run_fleet_humanevalplus_r3 as r3


def test_objective_target_set_is_exactly_thirteen_source_truncations():
    records = r3.read_jsonl(r3.SOURCE / "raw/samples.jsonl")
    selected = r3.select_targets(records)
    assert sum(map(len, selected.values())) == 13
    assert selected == {model: tuple(sorted(ids)) for model, ids in r3.EXPECTED_TARGETS.items()}


def test_correction_payload_changes_only_max_tokens():
    problem = r1.load_panel()[0]
    baseline = r1.payload("hauhaucs", problem)
    corrected = r3.correction_payload("hauhaucs", problem)
    assert corrected["max_tokens"] == 1536
    assert baseline | {"max_tokens": 1536} == corrected


def test_merge_preserves_every_non_target_record():
    source = r3.read_jsonl(r3.SOURCE / "raw/samples.jsonl")
    targets = [row for row in source if row.get("truncated")]
    corrections = []
    for row in targets:
        replacement = copy.deepcopy(row)
        replacement["completion"] += " corrected"
        replacement["truncated"] = False
        corrections.append(replacement)
    merged = r3.merge_records(source, corrections)
    target_keys = {(row["model"], row["task_id"]) for row in targets}
    for old, new in zip(source, merged):
        key = (old["model"], old["task_id"])
        if key not in target_keys:
            assert old == new
        else:
            assert new["completion"].endswith(" corrected")
