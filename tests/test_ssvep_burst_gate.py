"""Расширенные тесты ssvep_analysis.burst_gate."""

from __future__ import annotations

import numpy as np
import pytest

from ssvep_analysis.burst_gate import (
    BurstGate,
    BurstGateConfig,
    append_chunk_timestamps,
    parse_led_serial_line,
    parse_lsl_marker,
)


class TestParse:
    def test_led_whitespace(self) -> None:
        assert parse_led_serial_line("  LED 3 1 ON  ") == (3, True)

    def test_lsl_bytes(self) -> None:
        assert parse_lsl_marker(b"101|on") == (1, True)

    def test_lsl_rejects_low_id(self) -> None:
        assert parse_lsl_marker("5|on") is None

    def test_lsl_rejects_bad_lamp(self) -> None:
        assert parse_lsl_marker("199|on") is None


class TestAppendTimestamps:
    def test_per_sample_ts(self) -> None:
        prev = np.array([1.0, 2.0])
        out = append_chunk_timestamps(prev, [3.0, 4.0], 2, 250.0)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])

    def test_single_ts_expands(self) -> None:
        out = append_chunk_timestamps(np.zeros(0), [10.0], 4, 100.0)
        assert out.size == 4
        assert out[-1] == pytest.approx(10.0)
        assert out[1] - out[0] == pytest.approx(0.01)

    def test_empty_chunk(self) -> None:
        prev = np.array([1.0])
        out = append_chunk_timestamps(prev, [], 0, 250.0)
        assert out.size == 1

    def test_no_ts_continues(self) -> None:
        prev = np.array([0.0, 0.004])
        out = append_chunk_timestamps(prev, [], 2, 250.0)
        assert out.size == 4
        assert out[-1] > out[-2]


class TestBurstGate:
    def test_led_line_ingest(self) -> None:
        gate = BurstGate()
        gate.set_active_lamps(1)
        gate.ingest_led_line("LED 0 0 ON", 0.0)
        gate.ingest_led_line("LED 0 100 OFF", 0.5)
        t = np.linspace(0.0, 0.4, 100)
        assert gate.classify_allowed(t, now=0.4)[0]
        t2 = np.linspace(1.0, 2.0, 100)
        assert not gate.classify_allowed(t2, now=2.0)[0]

    def test_only_active_lamps_count(self) -> None:
        gate = BurstGate(BurstGateConfig(window_sec=1.0, min_on_fraction=0.5, min_on_sec=0.4))
        gate.set_active_lamps(1)  # только лампа 0
        gate.ingest_marker(0.0, "101|on")  # лампа 1 — не в active
        t = np.linspace(0.0, 1.0, 200)
        ok, _ = gate.classify_allowed(t, now=1.0)
        assert not ok

    def test_short_flash_below_threshold(self) -> None:
        gate = BurstGate(BurstGateConfig(window_sec=2.0, min_on_fraction=0.7, min_on_sec=1.4))
        gate.set_active_lamps(1)
        gate.ingest_marker(1.5, "100|on")
        gate.ingest_marker(1.9, "100|off")
        t = np.linspace(0.0, 2.0, 500)
        ok, reason = gate.classify_allowed(t, now=2.0)
        assert not ok
        assert "стимул" in reason or "ON" in reason

    def test_packet_like_on_window(self) -> None:
        """Вспышка ~2 с в конце окна — как пакетный режим."""
        gate = BurstGate(BurstGateConfig(window_sec=2.0, min_on_fraction=0.7, min_on_sec=1.4))
        gate.set_active_lamps(4)
        gate.ingest_marker(0.0, "100|on")
        gate.ingest_marker(0.0, "101|on")
        gate.ingest_marker(0.0, "102|on")
        gate.ingest_marker(0.0, "103|on")
        t = np.linspace(0.0, 2.0, 500)
        ok, reason = gate.classify_allowed(t, now=2.0)
        assert ok, reason

    def test_duplicate_on_ignored(self) -> None:
        gate = BurstGate()
        gate.set_active_lamps(1)
        gate.ingest_marker(0.0, "100|on")
        gate.ingest_marker(0.1, "100|on")
        gate.ingest_marker(0.5, "100|off")
        assert len(gate._intervals) == 1

    def test_empty_buf_times(self) -> None:
        gate = BurstGate()
        gate.set_active_lamps(1)
        ok, reason = gate.classify_allowed(np.zeros(0))
        assert not ok
        assert "меток" in reason

    def test_no_active_lamps(self) -> None:
        gate = BurstGate()
        gate.ingest_marker(0.0, "100|on")
        t = np.linspace(0.0, 2.0, 100)
        ok, reason = gate.classify_allowed(t, now=2.0)
        assert not ok
        assert "ламп" in reason
