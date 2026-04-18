"""Привязка маркера LSL к индексам эпохи в буфере ЭЭГ (без Qt, для тестов)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

# Доля уникальных меток времени на хвосте буфера; ниже — не используем fallback
# (иначе при грубом шаге 1 с несколько вспышек получают один start_idx).
FALLBACK_MIN_UNIQUE_TS_FRACTION = 0.12

FALLBACK_LOOKBACK_SAMPLES = 2000


def eeg_timestamps_sufficient_for_fallback(
    time_arr: np.ndarray, *, buf_len: int, lookback: int = FALLBACK_LOOKBACK_SAMPLES
) -> bool:
    """True, если по хвосту time_arr видно «достаточно уникальных» меток для безопасного fallback."""
    if time_arr.ndim != 1 or time_arr.size != buf_len or buf_len == 0:
        return False
    n = int(min(lookback, buf_len))
    sl = time_arr[-n:]
    frac = float(np.unique(sl).size) / float(sl.size)
    return frac >= FALLBACK_MIN_UNIQUE_TS_FRACTION


def resolve_epoch_indices_for_marker(
    *,
    marker_ts: float,
    buf_len: int,
    srate: float,
    epoch_len: int,
    lsl_ref: float,
    time_arr: np.ndarray,
    marker_eeg_offset: Optional[float],
    compute_start_index: Callable[[np.ndarray, float], Optional[int]],
) -> Tuple[Optional[int], Optional[int], bool]:
    """Возвращает (start_idx, end_idx, wait_more).

    wait_more=True — нужно дождаться ещё данных в буфере (эпоха ещё не помещается).
    (None, None, False) — маркер нельзя надёжно извлечь (отбросить).
    """
    mt = float(marker_ts)
    ref = float(lsl_ref)
    # Не отсекаем mt > ref: при ЭЭГ и маркерах с разных ПК liblsl обычно выравнивает время,
    # но кратковременный «зазор» + local_clock() даёт ложные «маркер из будущего» и вечный wait.
    # Как на main: считаем индекс как есть, затем wait по end_idx > buf (см. ниже).

    seconds_back = ref - mt
    start_idx = int(round(buf_len - 1 - seconds_back * srate))
    end_idx = int(start_idx + epoch_len)

    if 0 <= start_idx and end_idx <= buf_len:
        return start_idx, end_idx, False

    direct_needs_wait = end_idx > buf_len
    start_past_buffer = start_idx < 0

    ta = np.asarray(time_arr, dtype=np.float64).reshape(-1)
    # Маркер новее ref: как на main считаем только прямой индекс; fallback по eeg_times
    # даст ложный хвост буфера, если mt вне шкалы time_arr (типично при рассинхроне ПК).
    use_fallback = (mt <= ref) and eeg_timestamps_sufficient_for_fallback(ta, buf_len=buf_len)

    if use_fallback:
        candidates: list[float] = []
        if marker_eeg_offset is not None:
            candidates.append(mt + float(marker_eeg_offset))
        candidates.append(mt)
        for t_eff in candidates:
            fb_start = compute_start_index(ta, t_eff)
            if fb_start is None:
                continue
            fb_end = int(fb_start) + int(epoch_len)
            if 0 <= fb_start and fb_end <= buf_len:
                return int(fb_start), fb_end, False
            if fb_end > buf_len:
                return None, None, True

    if direct_needs_wait:
        return None, None, True

    # Маркер «в прошлом» относительно буфера или отказались от ненадёжного fallback.
    if start_past_buffer:
        return None, None, False

    return None, None, False
