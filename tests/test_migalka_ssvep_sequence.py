"""Проверка последовательности команд мигалки для continuous / burst / handoff."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ssvep_analysis.migalka_serial_controller import MigalkaConfig, MigalkaSerialController


def _capture_controller() -> tuple[MigalkaSerialController, list[str], MagicMock]:
    lines: list[str] = []
    ser = MagicMock()
    ser.is_open = True

    def _write(b: bytes) -> int:
        lines.append(b.decode("utf-8").strip())
        return len(b)

    ser.write = _write
    ser.flush = MagicMock()
    ser.readline = MagicMock(return_value=b"")
    ser.close = MagicMock()

    ctrl = MigalkaSerialController(mirror_lsl=False)
    ctrl._ser = ser
    ctrl._running = True
    return ctrl, lines, ser


def test_continuous_apply_sends_m0_then_freqs_not_only_zeros() -> None:
    ctrl, lines, _ = _capture_controller()
    cfg = MigalkaConfig(port="COM1", mode=0, freqs=("10.0", "0", "0", "0", "0", "0"))
    ctrl._apply_config(cfg)
    assert lines[0] == "M 0"
    assert "10.0" in lines[-1]
    assert lines.count("M 0") == 1


def test_burst_apply_zeros_then_m1_then_active_freqs() -> None:
    ctrl, lines, _ = _capture_controller()
    cfg = MigalkaConfig(port="COM1", mode=1, freqs=("12.0", "0", "0", "0", "0", "0"))
    ctrl._apply_config(cfg)
    assert lines[0] == "0 0 0 0 0 0"
    assert lines[1] == "M 1"
    assert lines[-1] == "12.0 0 0 0 0 0"


def test_standby_burst_keeps_m1_zeros() -> None:
    ctrl, lines, _ = _capture_controller()
    ctrl.standby_burst_between_phases()
    assert "M 1" in lines
    assert lines[-1] == "0 0 0 0 0 0"


@patch("ssvep_analysis.migalka_serial_controller.serial.Serial")
def test_reconfigure_when_closed_no_deadlock(mock_serial: MagicMock) -> None:
    ser = MagicMock()
    ser.is_open = True
    ser.readline = MagicMock(return_value=b"")
    mock_serial.return_value = ser
    ctrl = MigalkaSerialController(mirror_lsl=False)
    cfg = MigalkaConfig(port="COM1", mode=1, freqs=("8.0", "0", "0", "0", "0", "0"))
    ctrl.reconfigure(cfg)
    assert mock_serial.called
    assert ctrl.is_open()
