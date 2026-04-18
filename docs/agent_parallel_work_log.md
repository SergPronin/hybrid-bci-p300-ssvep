# Журнал параллельной работы агентов

Файл для координации нескольких агентов/сессий: фиксируйте **кто**, **когда** и **что** менял, чтобы не дублировать и не ломать чужие правки.

## Правила

1. Перед крупным изменением добавьте секцию **In progress** с датой (UTC или локально — единообразно в проекте), идентификатором агента и кратким планом.
2. После завершения — перенесите в **Done** или обновите статус; укажите коммит/ветку если есть.
3. Конфликты с другой веткой — блок **Conflicts / Notes**.
4. Не удаляйте чужие записи; дополняйте новыми секциями сверху (после этого блока правил).

---

## Done — 2026-04-18 — epoch indexing + тесты

**Агент:** Cursor (composer)  
**Ветка:** `dev/add-save` (локально в клоне пользователя).

**Проблема:** при одном блоке стимулов несколько вспышек могли получать **одинаковое** окно эпохи из-за **fallback** по грубым `eeg_times` (мало уникальных меток).

**Решение:**

- Новый модуль `p300_analysis/epoch_indexing.py`: чистая функция `resolve_epoch_indices_for_marker`, проверка `marker_ts > lsl_ref + tol` → ждать данные; **fallback** только если `eeg_timestamps_sufficient_for_fallback` (доля уникальных меток на хвосте буфера ≥ порога).
- `qt_window.py`: делегирование в новый модуль (поведение UI = вызов той же логики).
- Тесты: `tests/test_epoch_indexing.py`.

**Команда проверки:** `python -m pytest tests/test_epoch_indexing.py tests/test_winner_metrics.py -v`

**Результат тестов:** см. последний запуск в CI/терминале; после правок — обновите блок ниже.

### Last test run (fill by agent)

- **Дата:** 2026-04-18  
- **Команда:** `python -m pytest tests/test_epoch_indexing.py tests/test_winner_metrics.py -v`  
- **Результат:** `9 passed` (7× `test_epoch_indexing`, 2× `test_winner_metrics`).  
- **Примечание:** `tests/test_log_replay.py` требует окружение с `pylsl` — в общем прогоне без pylsl не собирается.

---

## Done — 2026-04-18 — подробный NDJSON-лог каждого обследования

**Агент:** Cursor  
**Файлы:** `p300_analysis/exam_session_detail_logger.py`, интеграция в `p300_analysis/qt_window.py`, тесты `tests/test_exam_session_detail_logger.py`.  
**Папка логов:** `data/examination_logs/exam_run{seq}_{дата}_{id}.ndjson` (один файл на нажатие «Начать анализ» до остановки/сброса/отключения/закрытия окна).  
**События в логе:** `exam_start`, `lsl_marker_sample`, `trial_cue_lsl`, `stimulus_stream_started`, `eeg_chunk`, `time_alignment_calibrated`, `marker_epoch_*`, `epoch_extracted`, `winner_update`, `qt_tick`, обрезки буфера, `exam_end` + путь в `summary.detail_exam_log_path`.  
**Тесты:** `python -m pytest tests/test_exam_session_detail_logger.py tests/test_epoch_indexing.py tests/test_winner_metrics.py -v` → 13 passed.

---

## In progress

_(пусто — зарезервируйте перед началом работы)_
