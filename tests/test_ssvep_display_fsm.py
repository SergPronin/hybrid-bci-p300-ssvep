"""FSM: SSVEP continuous -> burst -> finalize, флаги оверлея и migalka."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from experiment_protocol.protocol_runner import ProtocolConfig, ProtocolRunner, ProtocolState
from ssvep_analysis.online_engine import SSVEPDecision


def _fake_stream(name: str, stype: str) -> MagicMock:
    info = MagicMock()
    info.name = MagicMock(return_value=name)
    info.type = MagicMock(return_value=stype)
    return info


def _inlet_with_markers(markers: list) -> MagicMock:
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


@patch("ssvep_analysis.online_engine.SSVEPOnlineEngine.classify")
@patch("experiment_protocol.protocol_runner.stream_inlet_with_buffer")
@patch("experiment_protocol.protocol_runner.wait_for_stimulus_marker_stream")
@patch("experiment_protocol.protocol_runner.discover_eeg_streams")
@patch("ssvep_analysis.migalka_serial_controller.serial.Serial")
@patch("ssvep_analysis.migalka_serial_controller.time.sleep")
def test_ssvep_cont_burst_finalize_and_display_flags(
    _sleep: MagicMock,
    mock_serial: MagicMock,
    mock_classify: MagicMock,
    mock_eeg: MagicMock,
    mock_mk: MagicMock,
    mock_inlet: MagicMock,
    tmp_path: Path,
) -> None:
    mock_classify.return_value = SSVEPDecision(
        winner_1based=1,
        winner_0idx=0,
        coef=[0.5],
        mode="mock",
        classify_allowed=True,
        debug={},
    )
    mock_eeg.return_value = [_fake_stream("EEG", "EEG")]
    bci_mk = _fake_stream("BCI_StimMarkers", "Markers")
    mock_mk.return_value = (bci_mk, [bci_mk])

    ser = MagicMock()
    ser.is_open = True
    ser.readline = MagicMock(return_value=b"")
    mock_serial.return_value = ser

    eeg_inlet = MagicMock()
    eeg_inlet.pull_chunk = MagicMock(return_value=(np.zeros((10, 4)), list(range(10))))
    mk_inlet = _inlet_with_markers(
        ["-1|trial_start|target=0", "-2|trial_end", "-1|trial_start|target=0", "-2|trial_end"]
    )
    mock_inlet.side_effect = [eeg_inlet, mk_inlet]

    cleared: list[str] = []

    cfg = ProtocolConfig(
        output_root=tmp_path,
        subject_id="test",
        com_port="COM_TEST",
        eeg_stream_name="EEG",
        p300_calib_trials=0,
        p300_main_trials=0,
        ssvep_blocks_per_mode=1,
        shuffle_seed=0,
        pause_between_experiments_s=0.0,
        ssvep_block_sec=0.02,
        ssvep_freqs_hz=(10.0, 0.0, 0.0, 0.0),
    )
    runner = ProtocolRunner(cfg)
    runner.set_ssvep_display_clear_callback(lambda: cleared.append("ok"))

    runner.start()
    for _ in range(800):
        runner.tick()
        if runner.state == ProtocolState.Stopped:
            break

    assert runner.state == ProtocolState.Stopped
    assert not runner.ssvep_blackout_visible
    assert not runner.ssvep_cue_visible
    assert "ok" in cleared
