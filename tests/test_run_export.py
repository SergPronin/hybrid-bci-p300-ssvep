"""Tests for manual run export helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from p300_analysis.run_export import export_run_continuous_csv, export_run_data_all_stims, stim_indices_in_run


def test_stim_indices_in_run_sorted_unique() -> None:
    run = {
        "epoch_segments": [
            {"stim_key": "стимул_8"},
            {"stim_key": "стимул_0"},
            {"stim_key": "стимул_8"},
            {"stim_key": "стимул_3"},
        ]
    }
    assert stim_indices_in_run(run) == [0, 3, 8]


def test_stim_indices_in_run_empty() -> None:
    assert stim_indices_in_run({}) == []
    assert stim_indices_in_run({"epoch_segments": []}) == []
    assert stim_indices_in_run({"epoch_segments": [{"stim_key": "bad"}]}) == []


def test_export_run_data_all_stims_one_csv() -> None:
    run = {
        "summary": {"analysis_params": {"sampling_rate_hz": 500.0}},
        "eeg_samples": [[1.0, 2.0], [3.0, 4.0]],
        "epoch_segments": [
            {
                "stim_key": "стимул_0",
                "marker_ts": 10.0,
                "eeg_ts": [10.0, 10.002],
                "eeg_samples": [[1.1, 2.1], [1.2, 2.2]],
            },
            {
                "stim_key": "стимул_1",
                "marker_ts": 20.0,
                "eeg_ts": [20.0],
                "eeg_samples": [[5.0, 6.0]],
            },
        ],
    }
    with TemporaryDirectory() as td:
        out = Path(td) / "all.csv"
        paths = export_run_data_all_stims(run_data=run, output_path=out, file_format="csv")
        assert len(paths) == 1
        assert paths[0] == out
        with open(paths[0], encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0][:5] == ["segment_id", "stim_key", "marker_ts", "sample_idx", "ts"]
        assert len(rows) == 1 + 2 + 1  # header + 2 samples seg0 + 1 sample seg1
        assert float(rows[1][0]) == 0.0 and rows[1][1] == "стимул_0"
        assert float(rows[3][0]) == 1.0 and rows[3][1] == "стимул_1"


def test_export_run_data_all_stims_sorted_by_marker_ts() -> None:
    """Позже в списке, но раньше по времени маркера — выходит первым в файле."""
    run = {
        "eeg_samples": [[1.0, 2.0]],
        "epoch_segments": [
            {
                "stim_key": "стимул_9",
                "marker_ts": 200.0,
                "eeg_ts": [200.0],
                "eeg_samples": [[9.0, 9.1]],
            },
            {
                "stim_key": "стимул_1",
                "marker_ts": 100.0,
                "eeg_ts": [100.0, 100.002],
                "eeg_samples": [[1.0, 2.0], [1.1, 2.1]],
            },
        ],
    }
    with TemporaryDirectory() as td:
        paths = export_run_data_all_stims(
            run_data=run, output_path=Path(td) / "o.csv", file_format="csv"
        )
        with open(paths[0], encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert float(rows[1][0]) == 0.0 and rows[1][1] == "стимул_1"
        assert float(rows[3][0]) == 1.0 and rows[3][1] == "стимул_9"


def test_export_run_data_all_stims_empty_raises() -> None:
    with TemporaryDirectory() as td:
        with pytest.raises(RuntimeError, match="Нет сегментов"):
            export_run_data_all_stims(
                run_data={"epoch_segments": []},
                output_path=Path(td) / "x.csv",
                file_format="csv",
            )


def test_export_run_continuous_csv_marker_and_in_epoch() -> None:
    run = {
        "summary": {"analysis_params": {"sampling_rate_hz": 500.0}},
        "eeg_ts": [100.0 + 0.002 * i for i in range(10)],
        "eeg_samples": [[float(i), float(i) * 2.0] for i in range(10)],
        "markers": [{"ts": 100.004, "value": "8|on"}],
        "epoch_segments": [],
        # Окно эпохи 10 мс → при 500 Гц это 5 сэмплов (включая сэмпл вспышки).
        "epoch_time_ms": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    }
    with TemporaryDirectory() as td:
        base = Path(td) / "run1.csv"
        p = export_run_continuous_csv(run_data=run, output_path=base)
        assert p.name == "run1_continuous.csv"
        with open(p, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == [
            "sample_idx",
            "t_rel_s",
            "ts",
            "ch_1",
            "ch_2",
            "marker",
            "in_epoch",
        ]
        # строка 3 (sample_idx=2) — ближайшая к маркеру 100.004
        assert float(rows[3][0]) == 2.0
        assert float(rows[3][-2]) == 8.0  # marker = 8
        assert float(rows[3][-1]) == 1.0  # in_epoch = 1
        # пауза до вспышки
        assert float(rows[1][-2]) == 0.0
        assert float(rows[1][-1]) == 0.0
        # t_rel_s от начала
        assert abs(float(rows[1][1]) - 0.0) < 1e-9
        assert abs(float(rows[3][1]) - 0.004) < 1e-9


def test_export_run_continuous_csv_marker_via_lsl_clock_mapping() -> None:
    """ЭЭГ-штамп идёт с шагом 1 с (эмулируем NeuroSpectrum), а маркер — в осях
    lsl_local_clock() со смещением marker_eeg_offset. Проверяем, что строка
    маркера определяется по частоте дискретизации и lsl_clock_at_buffer_end,
    а не по ``argmin`` сырого ts.
    """
    n = 1000
    srate = 500.0
    # Грубый ts: фактически все отсчёты попадают в 2 секунды.
    eeg_ts = [100.0 if i < 500 else 101.0 for i in range(n)]
    eeg_samples = [[0.0] for _ in range(n)]
    # Последний отсчёт соответствует lsl_clock_at_buffer_end = 12345.0
    last_lc = 12345.0
    # Маркер в сырой оси — 999.5; offset = last_lc - last_marker_lc не важен,
    # важно, чтобы (marker_ts + offset) = последняя секунда хвоста.
    # Пусть marker попал за 0.4 с до конца буфера → sample_idx = n-1 - round(0.4*500) = 999 - 200 = 799
    marker_ts_raw = 999.5
    offset = last_lc - (marker_ts_raw + 0.4)  # → marker_ts_raw+offset = last_lc - 0.4
    run = {
        "summary": {
            "analysis_params": {"sampling_rate_hz": srate},
            "marker_eeg_offset": offset,
            "lsl_clock_at_buffer_end": last_lc,
        },
        "eeg_ts": eeg_ts,
        "eeg_samples": eeg_samples,
        "markers": [{"ts": marker_ts_raw, "value": "3|on"}],
        "epoch_segments": [],
        "epoch_time_ms": [0.0, 10.0],
    }
    with TemporaryDirectory() as td:
        base = Path(td) / "runLC.csv"
        p = export_run_continuous_csv(run_data=run, output_path=base)
        with open(p, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
    expected_idx = n - 1 - int(round(0.4 * srate))
    body = rows[1:]
    markers_nonzero = [i for i, r in enumerate(body) if float(r[-2]) != 0.0]
    assert markers_nonzero == [expected_idx]
    assert float(body[expected_idx][-2]) == 3.0
    assert float(body[expected_idx][-1]) == 1.0
