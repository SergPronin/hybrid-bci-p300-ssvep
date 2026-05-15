"""Тесты прослойки пакетного SSVEP."""

from __future__ import annotations

import numpy as np

from ssvep_analysis.burst_gate import (
    BurstGate,
    BurstGateConfig,
    parse_led_serial_line,
    parse_lsl_marker,
)


def test_parse_led_line() -> None:
    assert parse_led_serial_line("LED 2 12345 ON") == (2, True)
    assert parse_led_serial_line("LED 0 99 off") == (0, False)
    assert parse_led_serial_line("Mode=1") is None


def test_parse_lsl_marker() -> None:
    assert parse_lsl_marker("102|on") == (2, True)
    assert parse_lsl_marker(["103|off"]) == (3, False)


def test_burst_gate_allows_during_on() -> None:
    gate = BurstGate(BurstGateConfig(window_sec=2.0, min_on_fraction=0.7, min_on_sec=1.4))
    gate.set_active_lamps(2)
    gate.ingest_marker(0.0, "100|on")
    gate.ingest_marker(0.0, "101|on")
    t = np.linspace(0.0, 2.0, 500)
    ok, _ = gate.classify_allowed(t, now=2.0)
    assert ok


def test_burst_gate_blocks_after_off() -> None:
    gate = BurstGate(BurstGateConfig(window_sec=2.0, min_on_fraction=0.7, min_on_sec=1.4))
    gate.set_active_lamps(1)
    gate.ingest_marker(0.0, "100|on")
    gate.ingest_marker(0.5, "100|off")
    t = np.linspace(0.0, 2.0, 500)
    ok, reason = gate.classify_allowed(t, now=2.0)
    assert not ok
    assert "стимул" in reason.lower() or "%" in reason
