"""SSVEP continuous → burst handoff without reopening serial (v2 main queue)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from experiment_protocol.protocol_runner import ProtocolConfig, ProtocolRunner, ProtocolState


def _fake_stream(name: str, stype: str) -> MagicMock:
    info = MagicMock()
    info.name = MagicMock(return_value=name)
    info.type = MagicMock(return_value=stype)
    return info


@patch("experiment_protocol.protocol_runner.stream_inlet_with_buffer")
@patch("experiment_protocol.protocol_runner.wait_for_stimulus_marker_stream")
@patch("experiment_protocol.protocol_runner.discover_eeg_streams")
@patch("ssvep_analysis.migalka_serial_controller.serial.Serial")
def test_ssvep_cont_to_burst_without_serial_reopen(
    mock_serial: MagicMock,
    mock_eeg: MagicMock,
    mock_mk: MagicMock,
    mock_inlet: MagicMock,
    tmp_path: Path,
) -> None:
    mock_eeg.return_value = [_fake_stream("EEG", "EEG")]
    bci_mk = _fake_stream("BCI_StimMarkers", "Markers")
    mock_mk.return_value = (bci_mk, [bci_mk])

    ser = MagicMock()
    ser.is_open = True
    mock_serial.return_value = ser

    eeg_inlet = MagicMock()
    eeg_inlet.pull_chunk = MagicMock(
        return_value=(np.zeros((10, 4)), [float(i) for i in range(10)])
    )
    mk_inlet = MagicMock()
    mk_inlet.pull_chunk = MagicMock(return_value=([], []))
    mock_inlet.side_effect = [eeg_inlet, mk_inlet]

    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    cfg = ProtocolConfig(
        output_root=tmp_path,
        subject_id="test",
        session_dir=session_dir,
        com_port="COM_TEST",
        eeg_stream_name="EEG",
        p300_calib_trials=0,
        p300_main_trials=0,
        ssvep_blocks_per_mode=1,
        pause_between_experiments_s=0.0,
        ssvep_block_sec=0.01,
        shuffle_seed=0,
    )
    runner = ProtocolRunner(cfg)
    runner.start()

    for _ in range(4000):
        runner.tick()
        if runner.state in (ProtocolState.Finalize, ProtocolState.Stopped):
            break
    for _ in range(200):
        runner.tick()
        if runner.state == ProtocolState.Stopped:
            break

    assert runner.state == ProtocolState.Stopped
    assert mock_serial.call_count == 1, "мигалка должна открыться один раз за сессию"
