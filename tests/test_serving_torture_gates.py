from __future__ import annotations

from unittest.mock import patch

from tools.probes.slx01b_serving_torture import check_slots_idle


def test_missing_state_does_not_mask_processing_slot():
    slots = [{"id": 0, "is_processing": True}]
    with patch("tools.probes.slx01b_serving_torture.http_get_json", return_value=slots):
        idle, observed = check_slots_idle("http://127.0.0.1:8080")
    assert observed == slots
    assert idle is False


def test_every_slot_must_explicitly_report_not_processing():
    slots = [
        {"id": 0, "is_processing": False},
        {"id": 1, "is_processing": False},
    ]
    with patch("tools.probes.slx01b_serving_torture.http_get_json", return_value=slots):
        idle, observed = check_slots_idle("http://127.0.0.1:8080")
    assert observed == slots
    assert idle is True


def test_empty_slot_payload_fails_closed():
    with patch("tools.probes.slx01b_serving_torture.http_get_json", return_value=[]):
        idle, observed = check_slots_idle("http://127.0.0.1:8080")
    assert observed == []
    assert idle is False
