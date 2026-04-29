# AI PROJECT LOG

---

## 2026-04-25 — Аудит состояния проекта и диагностика P300

### Контекст

Провёдено несколько сессий с живыми испытуемыми (Андрей, Сергей, Али). Программа-анализатор
стабильно работала на фотодатчике-симуляторе, но на людях выдаёт **случайные или неверные** плитки.
Проанализировано 8 continuous CSV-файлов, код пайплайна прочитан полностью.

---

### Текущее состояние проекта

#### Архитектура (что работает)

| Компонент | Файл | Статус |
|---|---|---|
| Визуальный стимулятор (PsychoPy) | `gui/gui.py`, `core/stimulus_controller.py`, `core/grid.py`, `core/tile.py` | ✅ работает |
| LSL-стриминг маркеров | `core/lsl.py` | ✅ работает |
| Запуск стимулятора отдельно | `run_stimulus.py` | ✅ работает |
| P300 анализатор (Qt GUI) | `p300_analysis/qt_window.py` | ⚠️ работает, но алгоритм плохой |
| Экспорт в continuous CSV | `p300_analysis/run_export.py` | ✅ работает |
| Базовое вычисление ERP / winner | `p300_analysis/erp_compute.py` | ⚠️ алгоритм примитивный |
| Baseline correction + AUC | `p300_analysis/signal_processing.py` | ⚠️ без фильтра, без артефактов |
| Парсинг маркеров | `p300_analysis/marker_parsing.py` | ✅ ок |
| Epoch geometry (dt, template) | `p300_analysis/epoch_geometry.py` | ✅ ок |
| Session recorder | `p300_analysis/session_recorder.py` | ✅ работает |
| Загрузка continuous CSV оффлайн | `p300_analysis/qt_window.py:_load_continuous_csv_for_analysis` | ✅ работает |
| Тесты | `tests/` | ✅ есть, частичные |

#### Результаты сессий с испытуемыми

| Файл | Ожид. плитка | Winner | Верно? | Причина ошибки |
|---|---|---|---|---|
| андрей1_continuous.csv | 5 | 5 | ✓ | — |
| андрей2_continuous.csv | ≠3 | 3 | ✗ | слабый P300 сигнал |
| андрей3_continuous.csv | ≠4 | 4 | ✗ | слабый P300 сигнал |
| серг1_continuous.csv | **2** | 3 | ✗ | ch_3 шумит (abs_mean 34 vs 13) |
| серг2_continuous.csv | **2** | 5 | ✗ | все каналы мёртвые (std 190–270) |
| серг3_continuous.csv | ? | 5 | ? | ch_3 шумит (abs_mean 72 vs 13) |
| серг4_continuous.csv | ? | 1 | ? | ch_3 подшумливает, мало контраста |
| серг5_continuous.csv | ? | 4 | ? | чистые каналы, неизвестна цель |

**Итого известных**: 5 сессий → 1 правильно (20%) = на уровне случайного угадывания из 8 (12.5%).

---

### Критические проблемы (по приоритету)

#### 🔴 P1 — Шумный электрод ch_3 (железо + отсутствие авто-исключения)

**Симптом**: в каждой серг-сессии `ch_3` по амплитуде в 2–5× превышает остальные каналы.
Из-за этого усреднение `mean(channels)` перед анализом фактически сводится к сигналу ch_3.

**Корень**: плохой контакт электрода. Программа должна обнаруживать это автоматически и исключать
канал из анализа. Сейчас авто-детекции нет вообще.

**Последствие**: шум одного электрода определяет «победителя».

#### 🔴 P2 — Нет полосового фильтра перед анализом

**Симптом**: `signal_processing.py` — baseline correction и AUC, но никакой фильтрации нет.

**Корень**: в буфере `eeg_buffer` сырые отсчёты. P300 живёт в диапазоне 0.5–20 Гц. Без фильтрации:
- DC-дрейф (пот, движение) съедает baseline
- 50 Гц + гармоники добавляют произвольный вклад в AUC
- Мышечные артефакты выше 30 Гц

Фильтр Баттерворта 0.5–20 Гц полностью убирает эти проблемы.

#### 🔴 P3 — Нет отбраковки артефактов

**Симптом**: на `серг3_continuous.csv` 30–50% эпох содержат пик-амплитуды >150 мкВ.
Сейчас все они идут в усреднение ERP как есть и портят его.

**Должно быть**: до усреднения каждая эпоха проверяется по пороговому значению (например, ±100–150 мкВ).
Эпохи, превышающие порог хотя бы на одном канале, выбрасываются.

#### 🔴 P4 — Каналы усредняются ДО анализа (1D вместо per-channel)

**Симптом**: `qt_window.py:1229-1232` — `buf_arr = np.mean(buf_2d[:, _valid], axis=1)`.
Дальше в `epochs_data` лежат 1-D массивы. Один шумный канал убивает сигнал с остальных.

**Должно быть**: `epochs_data[stim_key]` хранит `(epoch_len, n_channels)`. Метрика (AUC или peak)
считается поканально, затем усредняется по ROI.

#### 🟡 P5 — Метрика AUC `sum(|corrected|)` неселективна

**Симптом**: `erp_compute.py:75` — `np.sum(np.abs(corr_win), axis=1)`.

Проблемы:
- Награждает любую амплитуду, не только положительный P300-пик
- Рандомные шумы вносят такой же вклад, как настоящий ответ
- Не использует знак (P300 — положительный пик)

**Лучше**: `mean(corrected[250:450ms])` со знаком (signed mean), или AUC только по положительным
значениям в 200–500 мс.

#### 🟡 P6 — Baseline нестабилен при артефактах в pre-stimulus периоде

**Симптом**: `signal_processing.py:27` — `mean(raw[:baseline_idx])`. Если в первые 100 мс
происходит моргание, baseline уезжает на 20–50 мкВ и портит всю эпоху.

**Лучше**: `median(raw[:baseline_idx])` или линейный detrend по всей эпохе.

#### 🟡 P7 — Нет индикатора уверенности winner

**Симптом**: `argmax(metric)` без проверки margin = (top1 - top2) / top1.
Анализатор всегда выдаёт «победителя», даже если разница между плитками 0.1%.

**Нужно**: показывать margin в UI. При margin < 10% — предупреждение «ненадёжный результат».

#### 🟡 P8 — target_tile не сохраняется в continuous CSV

**Симптом**: в CSV нет колонки с целевой плиткой — проверить точность офлайн невозможно.
`trial_start|target=N` присутствует в NDJSON, но в CSV не прокидывается.

**Нужно**: добавить колонку `target_tile_id` в `export_run_continuous_csv`.

#### 🟢 P9 — Мало повторов в сессиях

- Серг1–3: 10–12 вспышек/класс. Минимум для P300: 15–20+.
- После отбраковки P3 останется ещё меньше.
- Нужно в UI рекомендовать 20+ блоков при коротком ISI.

#### 🟢 P10 — Нет регресс-теста по набору сессий

**Нужно**: скрипт `scripts/regression_test.py`, который:
- Читает папку с continuous CSV (содержащих target_tile_id из P8)
- Прогоняет пайплайн
- Выдаёт accuracy по сессиям и сводный отчёт

---

### Гипотезы о причинах в порядке вероятности

1. **Плохой контакт ch_3** → шум одного электрода => 60–70% ошибок у Сергея
2. **Отсутствие фильтра** → DC-дрейф / 50 Гц влияют на метрику => 20–30% ошибок
3. **Усреднение каналов до анализа** → потеря пространственной специфичности => 15–20% ошибок
4. **Артефакты не отбракованы** => случайные выбросы определяют winner
5. **Слабый P300 у испытуемых** → недостаточное внимание / ISI не оптимален => дополнительный шум

---

### Что нужно сделать в ближайшее время (план)

#### Sprint 1 — Критические алгоритмические правки (код)

| # | Задача | Файл | Трудоёмкость |
|---|---|---|---|
| 1 | Bandpass фильтр 0.5–20 Hz в `signal_processing.py` | `signal_processing.py` | ~30 мин |
| 2 | Per-channel epoch: хранить `(epoch_len, n_ch)` в `epochs_data` | `qt_window.py` | ~2 ч |
| 3 | Отбраковка эпох по порогу (configurable в UI) | `qt_window.py`, `erp_compute.py` | ~1 ч |
| 4 | Авто-детект шумных каналов + disabled-checkbox в UI | `qt_window.py` | ~1 ч |
| 5 | Метрика: signed mean 250–450 мс вместо AUC |x| | `erp_compute.py` | ~30 мин |

#### Sprint 2 — UI и диагностика

| # | Задача | Файл | Трудоёмкость |
|---|---|---|---|
| 6 | Margin indicator в результате winner | `qt_window.py` | ~30 мин |
| 7 | target_tile_id в continuous CSV | `run_export.py` | ~30 мин |
| 8 | Регресс-тест скрипт | `scripts/regression_test.py` | ~1 ч |

#### Sprint 3 — Протокол (не код)

| # | Задача |
|---|---|
| A | Перед каждой сессией: проверить импеданс всех электродов, особенно ch_3 |
| B | Инструкция испытуемому: считать вспышки целевой плитки, фиксировать взгляд |
| C | Минимум 20 блоков (=160 вспышек суммарно) |
| D | ISI не менее 200–300 мс между вспышками для нормального P300 |

---

### Архитектурное замечание

`qt_window.py` (2880 строк) содержит UI, LSL-обработку, извлечение эпох, анализ, экспорт и логику.
Необходим рефакторинг: выделить `EpochExtractor`, `ERP_Pipeline` и `SessionExporter` в отдельные
модули. Это упростит тестирование и уменьшит риск регрессий.

---

### Версия кода

- Последний коммит: `befecdb` (beta)
- Python-зависимости: PsychoPy, PyQt5, pyqtgraph, pylsl, numpy, scipy (опц.)
- Тесты: `tests/` — базовые, покрытие неполное

---

---

## 2026-04-25 — Sprint 1: Фильтр + артефакты + bad channel + margin

### Что сделано

#### `p300_analysis/signal_processing.py`
- **NEW** `bandpass_filter(X, fs, lo=0.5, hi=20.0, order=4)` — полосовой фильтр Баттерворта (filtfilt, нулевой сдвиг фазы). Работает с 1D и 2D массивами (n_samples, n_channels). Безопасен при коротких сигналах.
- **NEW** `detect_bad_channels(X, std_thresh=4.0, abs_thresh=3.0)` — возвращает индексы каналов, у которых std > 4·median(stds) или abs_mean > 3·median(abs_means).
- **CHANGED** `baseline_correction`: `np.mean` → `np.median`. Устойчивость к артефактам в pre-stimulus периоде.

#### `p300_analysis/erp_compute.py`
- **NEW** `artifact_reject_epochs(epochs, threshold_uv)` — отбрасывает эпохи с пиком |x| > threshold_uv.
- **CHANGED** `build_averaged_erp(epochs_data, epoch_len, artifact_threshold_uv=None)` — добавлен параметр порога; возвращает `(stim_keys, raw_averaged, rejected_counts)`.
- **CHANGED** `compute_winner_metrics` — добавлено поле `margin = (top1-top2)/|top1|` в debug_payload.
- **CHANGED** `winner_display_lines(winner_key, mode_short, lsl_cue_target_id, margin=None)` — добавлен параметр `margin`; выводит уверенность в процентах (высокая ≥30%, средняя ≥12%, низкая ⚠).

#### `p300_analysis/qt_window.py`
- **CHANGED** `_finalize_pending_epochs_for_stop`: `buf_2d_raw = stack(eeg_buffer)`, `buf_2d = bandpass_filter(buf_2d_raw, srate)`. Фильтр применяется ко всему буферу. Экспорт (`raw_segment`) берётся из `buf_2d_raw` (несырые данные сохраняются).
- **CHANGED** `_update_loop` (онлайн epoch extraction): то же — `buf_2d_raw` + `buf_2d` с фильтром.
- **CHANGED** `_load_continuous_csv_for_analysis` (офлайн CSV): `sig = bandpass_filter(sig_raw, fs_csv)` перед epoching.
- **CHANGED** `_redraw_from_epochs`: передаёт `artifact_threshold_uv` в `build_averaged_erp`; вызывает `detect_bad_channels` из последних 2000 отсчётов `eeg_buffer`; обновляет `_bad_ch_label`; передаёт `margin` в `winner_display_lines`.
- **NEW UI** `spin_artifact_thresh` (QSpinBox, 0–5000 мкВ, default 150) — порог отбраковки артефактов.
- **NEW UI** `_bad_ch_label` (QLabel, оранжевый) — показывает шумные каналы прямо в сайдбаре.

### Затронутые файлы
- `p300_analysis/signal_processing.py`
- `p300_analysis/erp_compute.py`
- `p300_analysis/qt_window.py`

### GitNexus detect_changes
- 27 символов touched, 4 файла, risk = **LOW**, affected_processes = 0.

### Что осталось (Sprint 2)
- P4: per-channel epoch storage (большой рефакторинг epochs_data → 2D)
- P7: target_tile_id в continuous CSV
- P8: регресс-тест скрипт

*Лог обновлён: 2026-04-25.*

---

## 2026-04-25 — Sprint 2: UI-цвета, панель каналов, target_tile_id, регресс-тест

### Что сделано

#### `p300_analysis/qt_window.py` (3 коммита)

**Единая тёмная тема (c778fca)**
- Заменены разрозненные `setStyleSheet(...)` на единый глобальный QSS-блок.
- Покрыты: `QWidget`, `QLabel`, `QCheckBox`, `QSpinBox`, `QComboBox`, `QScrollBar` (тонкий 8px), `QTextEdit`, `QGroupBox`, `QToolTip`.
- Устранена нечитаемость: чёрный текст на тёмном фоне, системный белый скроллбар.
- Добавлен `QGridLayout` в импорты.

**Панель "Состояние каналов" (c778fca)**
- Новая кнопка `📊 Состояние каналов` в sidebar → открывает `QDialog`.
- Таблица: канал / abs_mean / std / статус (✓ OK / ⚠ ШУМ).
- Кнопка "Отключить плохие" → снимает галочки у шумных каналов в ROI.
- Автообновление при каждом `_redraw_from_epochs`, если окно открыто.

**Исправления панели каналов (f04f9d3)**
- Источник данных: `eeg_buffer` (только при `_recording_epochs=True`) → `_eeg_channel_monitor_bufs` (живой всегда при подключении).
- Бандпасс-фильтрация перед `detect_bad_channels` → убирает DC-смещение, устранено ложное "всегда ШУМ".
- `_on_disable_bad_channels`: `blockSignals` при массовом снятии галочек + единый `_on_params_changed()` в конце.
- Плейсхолдер "Нет данных" когда LSL не подключён.

#### `p300_analysis/run_export.py` + `tests/test_run_export.py` (b744fc2)

- Импортирован `parse_trial_target_tile_id` из `marker_parsing`.
- В `export_run_continuous_csv` добавлена колонка **`target_tile_id`**:
  - `-1` до первого `trial_start` / после `trial_end`
  - `N` — значение из `trial_start|target=N`
- Обновлены все тесты (позиции `-2`/`-1` → `-3`/`-2`/`-1`).
- Добавлен новый тест `test_export_run_continuous_target_tile_id`.
- Результат: **10 passed, 1 skipped**.

#### `scripts/regression_test.py` (c0f7908)

Новый standalone-скрипт без Qt-зависимостей:
```
python scripts/regression_test.py *.csv [--baseline-ms 100] [--x-ms 200] [--y-ms 400] [--artifact-uv 150] [--channels 1,2,4] [-v]
```
- Тот же pipeline: bandpass → epoch extraction → artifact rejection → AUC winner.
- Читает `target_tile_id` колонку для автопроверки точности.
- Выводит таблицу: файл / цель / результат / ✓✗ / margin / кол-во эпох / шумные каналы.
- Итоговая строка: `Точность: N/M (X%)`.

#### `chore` (e168277)

- Удалены из трекинга `.cursor/` (11 файлов) и `.codegraph/config.json` — были force-added ранее, но находятся в `.gitignore`.
- Расширён `.gitignore`: данные сессий (`*.ndjson`, `*_continuous.csv`, `*.pkl`, `*.xlsx`, `saved_data/`, `data/`), AI-инструменты (`.cursor/`, `.codegraph/`, `.gitnexus`).

### Затронутые файлы
- `p300_analysis/qt_window.py`
- `p300_analysis/run_export.py`
- `tests/test_run_export.py`
- `scripts/regression_test.py`
- `.gitignore`

### GitNexus detect_changes
- `_setup_ui`, `export_run_continuous_csv`, `_refresh_ch_health` — risk **LOW**, upstream callers 0.
- Все изменения изолированы в UI-слое и экспорте.

### Нарушения правил (самокритика)
- Лог не обновлялся после каждого коммита — нарушение правила п.7. Исправлено сейчас.

### Что осталось (Sprint 3 / долгосрочно)
- **P4**: рефакторинг `epochs_data` → per-channel хранение `(epoch_len, n_ch)` — самый важный шаг для точности
- Сбор новых сессий с `target_tile_id` → прогон `regression_test.py` для оценки точности после Sprint 1

*Лог обновлён: 2026-04-25.*

---

## 2026-04-25 — Sprint 3: Per-channel ERP + нормализация каналов

### Проблема (критическая)
До этого коммита каналы усреднялись в 1D-сигнал **до** нарезки эпох. Шумный канал (`ch_3`, амплитуда в 3–5× больше) доминировал и маскировал P300 на тихих каналах.

### Что сделано (коммит `dae28a5`)

**`signal_processing.py`** — NEW `normalize_channels(X)`: делит каждый канал на его std.

**`erp_compute.py`**
- `artifact_reject_epochs` — работает с 2D `(epoch_len, n_ch)`.
- `build_averaged_erp` — 2D path: normalize → stack → mean по эпохам → mean по каналам → `(n_stim, epoch_len)`. 1D path: прежнее поведение.

**`qt_window.py`** — три места извлечения эпох: убрано `buf_arr = np.mean(channels)`, добавлено `buf_2d[start:end, valid_ch]` → `(epoch_len, n_ch)`.

**`tests/test_winner_metrics.py`** — 3 новых теста: per-channel нормализация, noisy-channel dominance, artifact_reject 2D.

### Результат: 27 passed, 6 skipped. GitNexus risk LOW.

*Лог обновлён: 2026-04-25.*
