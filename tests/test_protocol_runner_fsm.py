"""Smoke: ProtocolRunner v2 reaches SSVEP (mocked LSL + serial)."""

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


def _inlet_with_markers(markers: list) -> MagicMock:
    """Each pull_chunk returns next marker batch then empty."""
    inlet = MagicMock()
    idx = {"i": 0}

    def pull_chunk(timeout=0.0, max_samples=100):
        i = idx["i"]
        if i >= len(markers):
            return [], []
        m = markers[i]
        idx["i"] += 1
        return [[m]], [float(i)]

    inlet.pull_chunk = pull_chunk
    return inlet


@patch("experiment_protocol.protocol_runner.stream_inlet_with_buffer")
@patch("experiment_protocol.protocol_runner.wait_for_stimulus_marker_stream")
@patch("experiment_protocol.protocol_runner.discover_eeg_streams")
@patch("ssvep_analysis.migalka_serial_controller.serial.Serial")
def test_fsm_reaches_ssvep_and_opens_migalka(
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

    markers = [
        "-1|trial_start|target=4",
        "-2|trial_end",
    ]

    eeg_inlet = MagicMock()
    eeg_inlet.pull_chunk = MagicMock(
        return_value=(np.zeros((10, 4)), [float(i) for i in range(10)])
    )
    mk_inlet = _inlet_with_markers(markers)
    mock_inlet.side_effect = [eeg_inlet, mk_inlet]

    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    cfg = ProtocolConfig(
        output_root=tmp_path,
        subject_id="test",
        session_dir=session_dir,
        com_port="COM_TEST",
        eeg_stream_name="EEG",
        p300_calib_trials=1,
        calib_target_tile_id=4,
        template_warmup_target_epochs=1,
        p300_main_trials=0,
        ssvep_blocks_per_mode=1,
        pause_between_experiments_s=0.0,
        ssvep_block_sec=0.05,
        shuffle_seed=1,
    )
    runner = ProtocolRunner(cfg)
    runner.start()

    for _ in range(800):
        runner.tick()
        if runner.state in (
            ProtocolState.SSVEP_CONT,
            ProtocolState.SSVEP_BURST,
            ProtocolState.Finalize,
            ProtocolState.Stopped,
        ):
            break

    assert mock_serial.called, "serial.Serial должен вызываться для мигалки"
    assert runner.state in (
        ProtocolState.SSVEP_CONT,
        ProtocolState.SSVEP_BURST,
        ProtocolState.Finalize,
        ProtocolState.Stopped,
    ), f"застряли в {runner.state}"
