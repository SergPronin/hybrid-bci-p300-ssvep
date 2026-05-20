"""Тесты ssvep_analysis.burst_debug."""

from __future__ import annotations

import json
from pathlib import Path

from ssvep_analysis.burst_debug import (
    expected_msi_lamp_from_diag,
    summarize_burst_trace,
)


def test_expected_msi_from_diag() -> None:
    assert expected_msi_lamp_from_diag({"lamps_on_at_end_0idx": [2]}) == 3
    assert expected_msi_lamp_from_diag({"lamps_on_at_end_0idx": [0, 1]}) is None


def test_summarize_burst_trace(tmp_path: Path) -> None:
    p = tmp_path / "events.ndjson"
    rows = [
        {"event": "burst_msi", "data": {"winner": 3, "expected_from_markers": 3, "n_lamps_on_at_end": 1}},
        {"event": "burst_msi", "data": {"winner": 1, "expected_from_markers": 3, "n_lamps_on_at_end": 1}},
        {"event": "burst_msi", "data": {"winner": 2, "n_lamps_on_at_end": 2}},
    ]
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    s = summarize_burst_trace(p, target_lamp=3)
    assert s["n_burst_msi"] == 3
    assert abs(s["match_marker_fraction"] - 1 / 3) < 1e-6
    assert s["multi_lamp_at_end_count"] == 1
