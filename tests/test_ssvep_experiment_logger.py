"""Тесты SSVEP experiment logger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ssvep_analysis.experiment_logger import SSVEPExperimentLogger, coef_values


def test_logger_writes_events_and_eeg(tmp_path: Path) -> None:
    logger = SSVEPExperimentLogger.open_new(
        output_root=tmp_path,
        start_payload={"freqs_hz": [10.0, 12.0]},
    )
    logger.write("test_event", {"x": 1})
    times = np.linspace(0.0, 0.1, 25)
    eeg = np.random.randn(25, 2)
    logger.append_eeg_chunk(times, eeg)
    log_dir = logger.finalize(stop_payload={"reason": "test"}, channel_labels=["O1", "O2"])
    assert (log_dir / "events.ndjson").is_file()
    assert (log_dir / "manifest.json").is_file()
    assert (log_dir / "eeg.npz").is_file()
    lines = (log_dir / "events.ndjson").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3
    rec = json.loads(lines[0])
    assert rec["event"] == "experiment_start"
    data = np.load(log_dir / "eeg.npz", allow_pickle=True)
    assert data["eeg"].shape == (25, 2)


def test_coef_values_none_msi() -> None:
    class _M:
        Coef = None

    assert coef_values(_M(), 3) == [None, None, None]
