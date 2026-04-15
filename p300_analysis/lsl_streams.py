"""Поиск потоков LSL и создание StreamInlet с совместимым буфером."""

from __future__ import annotations

from typing import Any, List

from pylsl import StreamInfo, StreamInlet, resolve_byprop

from p300_analysis.constants import EEG_STREAM_TYPES, NEUROSPECTR_MARKER, SIMULATOR_NAME, SIMULATOR_SOURCE_ID


def _is_allowed_stream(info: StreamInfo) -> bool:
    try:
        name = (info.name() or "").strip().lower()
        sid = (info.source_id() or "").strip().lower()
    except Exception:
        return False
    if name == SIMULATOR_NAME.lower() or SIMULATOR_SOURCE_ID in sid:
        return True
    if NEUROSPECTR_MARKER in name or NEUROSPECTR_MARKER in sid:
        return True
    return False


def find_allowed_eeg_streams(timeout: float = 3.0) -> List[StreamInfo]:
    all_streams: List[StreamInfo] = []
    for stream_type in EEG_STREAM_TYPES:
        try:
            streams = resolve_byprop("type", stream_type, timeout=timeout)
            all_streams.extend(streams)
        except Exception:
            pass
    return [s for s in all_streams if _is_allowed_stream(s)]


def resolve_marker_streams(timeout: float = 0.5) -> List[StreamInfo]:
    try:
        return list(resolve_byprop("type", "Markers", timeout=timeout))
    except Exception:
        return []


def unwrap_combo_userdata(data: Any) -> Any:
    """QComboBox.itemData иногда отдаёт QVariant; pylsl ждёт «сырой» StreamInfo."""
    if data is None:
        return None
    try:
        from PyQt5.QtCore import QVariant

        if isinstance(data, QVariant):
            return data.value()
    except Exception:
        pass
    return data


def stream_inlet_with_buffer(info: StreamInfo, buffer_seconds: int) -> StreamInlet:
    """Создаёт inlet; разные сборки pylsl знают max_buffered или max_buflen."""
    try:
        return StreamInlet(info, max_buffered=buffer_seconds)
    except TypeError:
        pass
    try:
        return StreamInlet(info, max_buflen=buffer_seconds)
    except TypeError:
        pass
    return StreamInlet(info)
