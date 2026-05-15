"""Тесты ssvep_analysis.migalka_lsl."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ssvep_analysis.migalka_lsl import (
    SOURCE_ID,
    STREAM_NAME,
    MigalkaLslSender,
)


def test_stream_constants() -> None:
    assert STREAM_NAME == "MigalkaStimMarkers"
    assert SOURCE_ID == "migalka-due-001"


def test_send_lamp_event_no_outlet() -> None:
    s = MigalkaLslSender.__new__(MigalkaLslSender)
    s._outlet = None
    s.send_lamp_event(0, "on")  # не падает


@patch("ssvep_analysis.migalka_lsl.StreamOutlet")
@patch("ssvep_analysis.migalka_lsl.StreamInfo")
def test_send_lamp_event_push(mock_info: MagicMock, mock_outlet_cls: MagicMock) -> None:
    outlet = MagicMock()
    mock_outlet_cls.return_value = outlet
    s = MigalkaLslSender()
    s.send_lamp_event(2, "on")
    outlet.push_sample.assert_called_once_with(["102|on"])
    s.send_lamp_event(2, "off")
    outlet.push_sample.assert_called_with(["102|off"])


@patch("ssvep_analysis.migalka_lsl.StreamOutlet")
@patch("ssvep_analysis.migalka_lsl.StreamInfo")
def test_send_ignores_invalid(mock_info: MagicMock, mock_outlet_cls: MagicMock) -> None:
    outlet = MagicMock()
    mock_outlet_cls.return_value = outlet
    s = MigalkaLslSender()
    s.send_lamp_event(0, "blink")
    outlet.push_sample.assert_not_called()
