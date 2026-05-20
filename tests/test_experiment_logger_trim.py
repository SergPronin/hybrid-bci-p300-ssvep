"""trim_eeg_samples в experiment_logger."""

from __future__ import annotations

import numpy as np

from ssvep_analysis.experiment_logger import SSVEPExperimentLogger


def test_trim_eeg_samples(tmp_path) -> None:
    log_dir = tmp_path / "run_test"
    log_dir.mkdir()
    fh = open(log_dir / "events.ndjson", "w", encoding="utf-8")
    logger = SSVEPExperimentLogger(log_dir, fh, "run_test")
    logger.append_eeg_chunk([0.0, 0.004], np.ones((2, 3)), write_chunk_event=False)
    logger.append_eeg_chunk([0.008, 0.012], np.ones((2, 3)) * 2, write_chunk_event=False)
    assert len(logger._eeg_times) == 4
    logger.trim_eeg_samples(3)
    assert len(logger._eeg_times) == 3
    eeg = np.vstack(logger._eeg_data)
    assert eeg.shape[0] == 3
    fh.close()
