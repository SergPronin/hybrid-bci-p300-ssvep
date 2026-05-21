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
    logger.mark_experiment_start(kind="p300", phase="main", cue_target_tile_id=3)
    logger.record_marker(lsl_time=1.05, sample="-1|trial_start|target=3")
    logger.record_marker(lsl_time=1.10, sample="103|on")
    logger.append_eeg_chunk([1.2, 1.3], np.ones((2, 3)), stream="EEG", lsl_local_clock=124.0)
    logger.append_p300_trial({"trial": 1, "winner_key": "стимул_0"})
    logger.append_ssvep_block({"block": 1, "winner": 2})
    logger.append_experiment(
        {
            "kind": "p300",
            "phase": "main",
            "target": {"tile_id_to_watch": 3},
            "results": {"auc": {"winner_tile_id": 3}, "template_corr": {"winner_tile_id": 3}},
        }
    )
    out_dir = logger.finalize(stop_payload={"reason": "done"})

    assert out_dir.exists()
    assert (out_dir / "events.ndjson").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "eeg.npz").exists()
    assert (out_dir / "p300_trials.ndjson").exists()
    assert (out_dir / "ssvep_blocks.ndjson").exists()
    assert (out_dir / "experiments.ndjson").exists()

    exp_dir = out_dir / "experiments" / "exp_00001"
    assert (exp_dir / "experiment.json").exists()
    assert (exp_dir / "markers.ndjson").exists()
    assert (exp_dir / "events.ndjson").exists()
    assert (exp_dir / "eeg.npz").exists()
    bundle = json.loads((exp_dir / "experiment.json").read_text(encoding="utf-8"))
    assert bundle["experiment_number"] == 1
    assert bundle["target"]["tile_id_to_watch"] == 3
    assert len(bundle["markers"]) == 2

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("n_experiments") == 1
    assert manifest["schema"] == "hybrid_protocol/v2"

