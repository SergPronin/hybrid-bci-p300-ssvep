"""После последнего continuous: halt + handoff M1, COM открыт; burst — только apply."""

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


@patch("ssvep_analysis.online_engine.SSVEPOnlineEngine.classify")
@patch("experiment_protocol.protocol_runner.stream_inlet_with_buffer")
@patch("experiment_protocol.protocol_runner.wait_for_stimulus_marker_stream")
@patch("experiment_protocol.protocol_runner.discover_eeg_streams")
def test_cont_to_burst_handoff_keeps_com_open(
    mock_eeg: MagicMock,
    mock_mk: MagicMock,
    mock_inlet: MagicMock,
    mock_classify: MagicMock,
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

    eeg_inlet = MagicMock()
    eeg_inlet.pull_chunk = MagicMock(return_value=(np.zeros((10, 4)), list(range(10))))
    mk_inlet = MagicMock()
    mk_inlet.pull_chunk = MagicMock(
        return_value=(
            [["-1|trial_start|target=0"], ["-2|trial_end"], ["-1|trial_start|target=0"], ["-2|trial_end"]],
            [0.0, 1.0, 2.0, 3.0],
        )
    )
    mock_inlet.side_effect = [eeg_inlet, mk_inlet]

    cfg = ProtocolConfig(
        output_root=tmp_path,
        subject_id="test",
        com_port="COM_TEST",
        eeg_stream_name="EEG",
        p300_trials_per_mode=1,
        ssvep_blocks_per_mode=1,
        pause_between_experiments_s=0.0,
        ssvep_block_sec=0.01,
    )
    runner = ProtocolRunner(cfg)

    with (
        patch.object(runner._migalka, "halt_lamps") as halt,
        patch.object(runner._migalka, "prepare_burst_handoff") as handoff,
        patch.object(runner._migalka, "stop_and_close") as stop_close,
        patch.object(runner._migalka, "standby_burst_between_phases") as standby,
        patch.object(runner._migalka, "open_and_start") as open_start,
        patch.object(runner._migalka, "reconfigure") as reconfigure,
        patch.object(runner._migalka, "is_open", return_value=True),
    ):
        runner.start()
        for _ in range(600):
            runner.tick()
            if runner.state == ProtocolState.Stopped:
                break

    assert runner.state == ProtocolState.Stopped
    halt.assert_called_once()
    handoff.assert_called_once()
    standby.assert_not_called()
    reconfigure.assert_not_called()
    assert open_start.call_count >= 2
    assert stop_close.call_count >= 1
