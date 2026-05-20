from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment_protocol.unified_logger import UnifiedExperimentLogger


def test_unified_logger_writes_session_files(tmp_path: Path) -> None:
    logger = UnifiedExperimentLogger.open_new(
        output_root=tmp_path,
        subject_id="subj",
        protocol_plan={"p300": {"trials": 2}, "ssvep": {"blocks": 2}},
        start_payload={"hello": "world"},
    )
    logger.write("custom_event", {"x": 1})
    logger.append_eeg_chunk([1.0, 1.1], np.zeros((2, 3)), stream="EEG", lsl_local_clock=123.0)
    logger.append_p300_trial({"trial": 1, "winner_key": "стимул_0"})
    logger.append_ssvep_block({"block": 1, "winner": 2})
    out_dir = logger.finalize(stop_payload={"reason": "done"})

    assert out_dir.exists()
    assert (out_dir / "events.ndjson").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "eeg.npz").exists()
    assert (out_dir / "p300_trials.ndjson").exists()
    assert (out_dir / "ssvep_blocks.ndjson").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "hybrid_protocol/v1"

