"""Manual export of recorded P300 runs to txt/csv/xlsx."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _rows_to_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _rows_to_txt(path: Path, title: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


def _summary_rows(run_data: Dict[str, Any]) -> List[Tuple[str, Any]]:
    summary = run_data.get("summary") or {}
    params = summary.get("analysis_params") or {}
    return [
        ("run_seq", run_data.get("run_seq")),
        ("saved_at_ms", run_data.get("saved_at_ms")),
        ("n_markers", len(run_data.get("markers") or [])),
        ("n_eeg_samples", len(run_data.get("eeg_ts") or [])),
        ("n_winner_updates", len(run_data.get("winner_updates") or [])),
        ("n_epoch_classes", len(run_data.get("epochs_data") or {})),
        ("baseline_ms", params.get("baseline_ms")),
        ("window_x_ms", params.get("window_x_ms")),
        ("window_y_ms", params.get("window_y_ms")),
        ("epochs_after_trial_only", params.get("epochs_after_trial_only")),
        ("ui_winner_tile_id", summary.get("ui_winner_tile_id")),
        ("last_lsl_cue", summary.get("last_lsl_cue")),
        ("match_last_cue_vs_winner", summary.get("match_last_cue_vs_winner")),
    ]


def _marker_rows(run_data: Dict[str, Any]) -> List[Tuple[Any, Any]]:
    rows: List[Tuple[Any, Any]] = []
    for item in run_data.get("markers") or []:
        rows.append((item.get("ts"), item.get("value")))
    return rows


def _eeg_rows(run_data: Dict[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    eeg_ts = run_data.get("eeg_ts") or []
    eeg_samples = run_data.get("eeg_samples") or []
    for i, (ts, sample) in enumerate(zip(eeg_ts, eeg_samples)):
        row: List[Any] = [i, ts]
        row.extend(sample if isinstance(sample, list) else [sample])
        rows.append(row)
    return rows


def _winner_rows(run_data: Dict[str, Any]) -> List[Tuple[Any, Any, Any, Any]]:
    rows: List[Tuple[Any, Any, Any, Any]] = []
    for item in run_data.get("winner_updates") or []:
        rows.append(
            (
                item.get("event_seq"),
                item.get("winner_digit"),
                item.get("winner_key"),
                item.get("match_lsl_cue"),
            )
        )
    return rows


def _epoch_rows(run_data: Dict[str, Any]) -> List[Tuple[Any, Any, Any, Any]]:
    rows: List[Tuple[Any, Any, Any, Any]] = []
    time_ms = run_data.get("epoch_time_ms") or []
    epochs_data = run_data.get("epochs_data") or {}
    for stim_key, epochs in epochs_data.items():
        for epoch_idx, epoch_values in enumerate(epochs):
            for sample_idx, value in enumerate(epoch_values):
                t_ms = time_ms[sample_idx] if sample_idx < len(time_ms) else None
                rows.append((stim_key, epoch_idx, t_ms, value))
    return rows


def _epoch_raw_rows(run_data: Dict[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    segments = run_data.get("epoch_segments") or []
    for seg_idx, seg in enumerate(segments):
        stim_key = seg.get("stim_key")
        marker_ts = seg.get("marker_ts")
        eeg_ts = seg.get("eeg_ts") or []
        eeg_samples = seg.get("eeg_samples") or []
        for sample_idx, (ts, sample) in enumerate(zip(eeg_ts, eeg_samples)):
            row: List[Any] = [seg_idx, stim_key, marker_ts, sample_idx, ts]
            row.extend(sample if isinstance(sample, list) else [sample])
            rows.append(row)
    return rows


def export_run_data(
    *,
    run_data: Dict[str, Any],
    output_path: Path,
    file_format: str,
    include_summary: bool,
    include_markers: bool,
    include_eeg: bool,
    include_epochs: bool,
    include_winners: bool,
) -> List[Path]:
    """Export run data and return created file paths."""
    file_format = file_format.lower().strip()
    created: List[Path] = []
    stem = output_path.with_suffix("")

    summary_rows = _summary_rows(run_data)
    marker_rows = _marker_rows(run_data)
    eeg_rows = _eeg_rows(run_data)
    winner_rows = _winner_rows(run_data)
    epoch_rows = _epoch_rows(run_data)
    epoch_raw_rows = _epoch_raw_rows(run_data)

    if file_format in {"csv", "txt"}:
        ext = ".csv" if file_format == "csv" else ".txt"
        if include_summary:
            p = Path(f"{stem}_summary{ext}")
            if file_format == "csv":
                _rows_to_csv(p, ("field", "value"), summary_rows)
            else:
                _rows_to_txt(p, "Summary", ("field", "value"), summary_rows)
            created.append(p)
        if include_markers:
            p = Path(f"{stem}_markers{ext}")
            if file_format == "csv":
                _rows_to_csv(p, ("ts", "value"), marker_rows)
            else:
                _rows_to_txt(p, "Markers", ("ts", "value"), marker_rows)
            created.append(p)
        if include_eeg:
            max_ch = max((len(x) for x in (run_data.get("eeg_samples") or [])), default=0)
            eeg_header = ["sample_idx", "ts"] + [f"ch_{i+1}" for i in range(max_ch)]
            p = Path(f"{stem}_eeg{ext}")
            if file_format == "csv":
                _rows_to_csv(p, eeg_header, eeg_rows)
            else:
                _rows_to_txt(p, "EEG", eeg_header, eeg_rows)
            created.append(p)
        if include_winners:
            p = Path(f"{stem}_winners{ext}")
            if file_format == "csv":
                _rows_to_csv(
                    p, ("event_seq", "winner_digit", "winner_key", "match_lsl_cue"), winner_rows
                )
            else:
                _rows_to_txt(
                    p, "Winner updates", ("event_seq", "winner_digit", "winner_key", "match_lsl_cue"), winner_rows
                )
            created.append(p)
        if include_epochs:
            p = Path(f"{stem}_epochs{ext}")
            if file_format == "csv":
                _rows_to_csv(p, ("stim_key", "epoch_idx", "time_ms", "value"), epoch_rows)
            else:
                _rows_to_txt(p, "Epochs", ("stim_key", "epoch_idx", "time_ms", "value"), epoch_rows)
            created.append(p)
            max_ch = max(
                (len(x) for seg in (run_data.get("epoch_segments") or []) for x in (seg.get("eeg_samples") or [])),
                default=0,
            )
            raw_header = ["segment_idx", "stim_key", "marker_ts", "sample_idx", "ts"] + [
                f"ch_{i+1}" for i in range(max_ch)
            ]
            p_raw = Path(f"{stem}_epochs_raw_eeg{ext}")
            if file_format == "csv":
                _rows_to_csv(p_raw, raw_header, epoch_raw_rows)
            else:
                _rows_to_txt(p_raw, "Epochs raw EEG", raw_header, epoch_raw_rows)
            created.append(p_raw)
        return created

    if file_format == "ns_txt":
        # Single text file in a Neuron-Spectrum-like style for easy side-by-side comparison.
        if not include_eeg and not include_epochs:
            raise RuntimeError("Для NS TXT включите сырые ЭЭГ данные или эпохи.")
        eeg_ts: List[Any]
        eeg_samples: List[Any]
        if include_epochs and (run_data.get("epoch_segments") or []):
            eeg_ts = []
            eeg_samples = []
            for seg in run_data.get("epoch_segments") or []:
                eeg_ts.extend(seg.get("eeg_ts") or [])
                eeg_samples.extend(seg.get("eeg_samples") or [])
        else:
            eeg_ts = run_data.get("eeg_ts") or []
            eeg_samples = run_data.get("eeg_samples") or []
        if not eeg_samples:
            raise RuntimeError("Нет сырых ЭЭГ данных для NS TXT экспорта.")
        n_channels = max((len(x) for x in eeg_samples), default=0)
        if n_channels <= 0:
            raise RuntimeError("Не удалось определить число каналов для NS TXT.")
        summary = run_data.get("summary") or {}
        params = summary.get("analysis_params") or {}
        srate = params.get("sampling_rate_hz") or summary.get("eeg_stream_srate") or 0
        try:
            srate = int(round(float(srate))) if float(srate) > 0 else 0
        except Exception:
            srate = 0
        now = dt.datetime.now()
        if eeg_ts:
            duration_s = max(float(eeg_ts[-1]) - float(eeg_ts[0]), 0.0)
        elif srate > 0:
            duration_s = len(eeg_samples) / float(srate)
        else:
            duration_s = 0.0

        final_path = output_path if output_path.suffix else output_path.with_suffix(".txt")
        _ensure_parent(final_path)
        with open(final_path, "w", encoding="utf-8") as f:
            f.write("; Neuron-Spectrum.NET EEG TXT export file v.1\n")
            f.write("; EEG device name: LSL EEG stream\n")
            f.write("; EEG checkup: P300 Analyzer Export\n")
            f.write(f"; Checkup date: {now.strftime('%d.%m.%Y')}\n")
            f.write("; Patient name: \n")
            f.write("; Patient birthday: \n")
            f.write("; Montage: LSL channels\n")
            f.write("; Hi frequency filter: \n")
            f.write("; Low frequency filter: \n")
            f.write("; Notch filter: \n")
            f.write(f"; Sampling rate: {srate if srate > 0 else 'unknown'} Hz\n")
            f.write(f"; Derivations count: {n_channels}\n")
            for i in range(n_channels):
                f.write(f"; {i + 1}. EEG CH{i + 1} [{i}]\n")
            f.write(";\n")
            f.write(f"; Start recording date: {now.strftime('%d.%m.%Y')}\n")
            f.write(f"; Start recording time: {now.strftime('%H:%M:%S.%f')[:-3]}\n")
            f.write(f"; Record length: {duration_s:.3f} s\n")
            f.write("; Unit: microvolts\n")
            f.write("; Sygnal Type: EEG\n")
            f.write(";\n")
            for sample in eeg_samples:
                row_vals = []
                for val in sample:
                    try:
                        fv = float(val)
                    except Exception:
                        fv = 0.0
                    # Locale-like decimal comma to resemble NS export.
                    row_vals.append(f"{fv:.3f}".replace(".", ","))
                f.write(" ".join(row_vals) + " \n")
        created.append(final_path)
        return created

    if file_format == "xlsx":
        try:
            from openpyxl import Workbook
        except Exception as e:  # pragma: no cover - runtime dependency check
            raise RuntimeError(
                "Для экспорта в XLSX установите openpyxl (pip install openpyxl)."
            ) from e

        wb = Workbook()
        wb.remove(wb.active)

        def _add_sheet(name: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
            ws = wb.create_sheet(title=name[:31])
            ws.append(list(header))
            for row in rows:
                ws.append(list(row))

        if include_summary:
            _add_sheet("summary", ("field", "value"), summary_rows)
        if include_markers:
            _add_sheet("markers", ("ts", "value"), marker_rows)
        if include_eeg:
            max_ch = max((len(x) for x in (run_data.get("eeg_samples") or [])), default=0)
            eeg_header = ["sample_idx", "ts"] + [f"ch_{i+1}" for i in range(max_ch)]
            _add_sheet("eeg", eeg_header, eeg_rows)
        if include_winners:
            _add_sheet(
                "winners", ("event_seq", "winner_digit", "winner_key", "match_lsl_cue"), winner_rows
            )
        if include_epochs:
            _add_sheet("epochs", ("stim_key", "epoch_idx", "time_ms", "value"), epoch_rows)
            max_ch = max(
                (len(x) for seg in (run_data.get("epoch_segments") or []) for x in (seg.get("eeg_samples") or [])),
                default=0,
            )
            raw_header = ["segment_idx", "stim_key", "marker_ts", "sample_idx", "ts"] + [
                f"ch_{i+1}" for i in range(max_ch)
            ]
            _add_sheet("epochs_raw_eeg", raw_header, epoch_raw_rows)

        final_path = output_path if output_path.suffix.lower() == ".xlsx" else output_path.with_suffix(".xlsx")
        _ensure_parent(final_path)
        wb.save(final_path)
        created.append(final_path)
        return created

    raise ValueError(f"Unsupported file format: {file_format}")
