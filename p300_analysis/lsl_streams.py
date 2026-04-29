"""Поиск потоков LSL и создание StreamInlet с совместимым буфером."""

from __future__ import annotations

from typing import Any, List, Set, Tuple

from pylsl import StreamInfo, StreamInlet, resolve_byprop, resolve_streams

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


def _marker_like_stream(info: StreamInfo) -> bool:
    """Эвристика: поток похож на маркеры стимуляции (не только type=Markers)."""
    try:
        stype = (info.type() or "").strip().lower()
        name = (info.name() or "").strip().lower()
    except Exception:
        return False
    if stype == "markers":
        return True
    if "marker" in stype or "stim" in name or "bci" in name:
        return True
    return False


def _append_unique_streams(
    target: List[StreamInfo],
    batch: List[StreamInfo],
    seen: Set[Tuple[str, str]],
) -> None:
    for s in batch:
        try:
            key = (s.name() or "", s.session_id() or "")
        except Exception:
            key = (str(s), "")
        if key not in seen:
            seen.add(key)
            target.append(s)


def resolve_marker_streams(
    timeout: float = 5.0,
    *,
    attempts: int = 2,
) -> List[StreamInfo]:
    """Поиск потоков маркеров (LSL discovery).

    Сначала несколько попыток resolve_byprop(type=Markers). Если пусто —
    resolve_streams (все потоки) и фильтр по типу/имени: на части LAN второй
    путь иногда находит поток, который не отвечает на узкий resolve.
    """
    merged: List[StreamInfo] = []
    seen: Set[Tuple[str, str]] = set()
    for _ in range(max(1, int(attempts))):
        try:
            batch = list(resolve_byprop("type", "Markers", timeout=float(timeout)))
        except Exception:
            batch = []
        _append_unique_streams(merged, batch, seen)
        if merged:
            return merged

    try:
        broad = list(resolve_streams(wait_time=float(timeout)))
    except Exception:
        broad = []
    markerish = [s for s in broad if _marker_like_stream(s)]
    _append_unique_streams(merged, markerish, seen)
    return merged


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
