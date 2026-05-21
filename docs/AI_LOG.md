# AI PROJECT LOG

---

## 2026-05-21 — Полный аудит ССВП: continuous не мигал, burst вспышка, чёрный экран

**Найдено:** (1) `_apply_config` для постоянного слал `0 0 0 0 0 0` перед рабочими частотами — лампы не мигали; (2) `reconfigure()` → `open_and_start()` под одним `Lock` = **deadlock**, пакетный блок зависал; (3) рекурсия `clear_ssvep_display` ↔ `_restore_operator_window`; (4) `_finalize` не в том же tick.

**Исправление:** migalka на `RLock`, постоянный = `M0` + freqs; пакетный = `M1` + off + freqs; `standby_burst_between_phases` без закрытия COM; callback `set_ssvep_display_clear_callback`; тесты `test_migalka_ssvep_sequence.py`, `test_ssvep_display_fsm.py`.

**GitNexus risk:** LOW.

---

## 2026-05-20 — Пакетный ССВП: режим M1 + снятие чёрного экрана в конце

**Проблемы:** пакетный режим мигал как постоянный (COM уже открыт — `open_and_start` не слал `M 1`); чёрный экран оставался после конца протокола.

**Исправление:** при открытом порте — `_send_mode` + freqs; перед каждым блоком `stop_and_close`; сброс blackout/cue в `_finalize` и при переходе в Finalize; GUI скрывает оверлей в `finalize`/`stopped`.

**GitNexus risk:** LOW.

---

## 2026-05-20 — Откат оверлея к 435b1e7 (после поломки c0262b8)

**Причина:** коммит c0262b8 сменил `Qt.Tool`→`Qt.Window` и `_ensure_fullscreen` — оверлей перестал показываться; убрали сброс cue в конце паузы.

**Откат:** оверлей как в 435b1e7 (`Qt.Tool`, простой `showFullScreen`); восстановлен `_set_ssvep_cue_overlay(None)` в `_tick_pause`; PsychoPy закрывается при `ssvep_cue_visible` или `ssvep_blackout_visible`.

---

## 2026-05-20 — Регрессия: оверлей ССВП не виден, мигалка не стартует

**Причины:** (1) PsychoPy fullscreen перекрывал Qt-оверлей во время паузы перед ССВП; (2) оверлей с подсказкой снимался в конце паузы до старта мигалки — при ошибке COM экран пустой; (3) `Qt.Tool` мешал показу fullscreen.

**Исправление:** стимулятор закрывается при `pause_before_ssvep`; подсказка не скрывается до чёрного blackout; при ошибке мигалки подсказка возвращается; оверлей на primary screen, флаги `Qt.Window`.

---

## 2026-05-20 — Чёрный экран на ноутбуке во время мигания мигалки

**Запрос:** во время ССВП-блока экран ноутбука не должен отвлекать — только физическая мигалка.

**Сделано:** `ssvep_blackout_visible` в `ProtocolRunner`; `SsvepCueOverlay.show_blackout()` (#000, без текста) с момента успешного `open_and_start` до конца блока. Подсказка «Эксперимент №…» только на паузе перед блоком.

**GitNexus risk:** LOW.

---

## 2026-05-20 — SSVEP зависание после оверлея (n_lamps NameError)

**Проблема:** после fullscreen-оверлея «ССВЕП эксперимент» мигалка не стартовала.

**Причина:** в `_start_ssvep_block()` использовалась переменная `n_lamps` без определения → `NameError` при окончании паузы.

**Исправление:** `n_lamps = max(1, len(active_freqs))`; try/except вокруг `_start_ssvep_block`; оверлей не перехватывает фокус на каждом tick; `processEvents()` после tick в GUI.

**GitNexus risk:** LOW (локальный баг в FSM SSVEP).

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

---

## 2026-05-14 — MSIController: smoke-скрипт pythonnet + требование .NET 8

### Контекст

Добавлен standalone `scripts/test_msi_import.py`: CoreCLR через pythonnet, `clr.AddReference` для ALGLIB
(из `msi-res/` или извлечение из `alglib.net.3.19.0.nupkg`), опционально `msi-res/deps/*.dll`
(например `CommunityToolkit.Mvvm.dll`). Preflight: если в `MSIController.dll` виден target `.NETCoreApp` v8,
а в `DOTNET_ROOT` нет `Microsoft.NETCore.App 8.x`, скрипт завершается с понятным сообщением (избегает
краша pythonnet на старом runtime). GitNexus `detect_changes` (unstaged): risk **low**; upstream impact
на `requirements.txt`: **LOW** (0 callers).

Дополнение: reflection членов `MSI` — через `msi.GetType().GetMembers` (экземпляр), не через pythonnet-обёртку класса; эвристика TFM по DLL — `max` по всем вхождениям `Version=vN`; выбор ALGLIB из nupkg учитывает `max(msi_major, max(установленные Microsoft.NETCore.App))`. Пакет **alglib.net 3.19.0** в `msi-res` не содержит `lib/net8.0/alglib.net.dll`, поэтому фактически берётся **net7.0** — ожидаемо, пока не обновлён nupkg или не положен свой `alglib.net.dll`. Добавлен standalone `scripts/test_msi_exec.py`: синтетический sin (10 Hz), шаблоны 10/12/15/20 Hz (sin+cos в `double[,]`), `ModelSignal` как `List<>` из generic свойства, вызов `MSIExec(double[,])`, проверка winner=1 (1-based для первой частоты).

*Лог обновлён: 2026-05-14.*

---

## 2026-05-14 — SSVEP MSI standalone demo (`ssvep_demo/`)

### Контекст

Каталог **`ssvep_demo/`**: стимул PyQt6 (4 частоты), синтетический LSL `fake_eeg_lsl.py` (UDP 17391
смена 10/12/15/20 Hz), `msi_realtime.py` (`RollingEEGBuffer`, `MSIRealtimeClassifier` через
`test_msi_exec` без дублирования CoreCLR), `realtime_gui.py` (LSL + pyqtgraph + MSI), лончер
`run_demo.py`. P300/Qt pipeline не затронуты. В `requirements.txt` добавлен **PyQt6** для демо.

*Лог обновлён: 2026-05-14 (ssvep_demo).*

---

## 2026-05-14 — `run_demo.py`: без `realpath(sys.executable)` на macOS venv

### Контекст

Удалён `_resolved_python_exe()` (`os.path.normpath(os.path.realpath(sys.executable))`): на macOS symlink
`.venv/bin/python` раскрывался в Framework Python, дочерние `subprocess` теряли пакеты venv (pylsl, PyQt6,
pyqtgraph). Лончер и `Popen(..., executable=...)` используют только `sys.executable`. GitNexus impact по
новому файлу недоступен (символ не в индексе); область изменений — только лончер, риск **LOW**.
Проверка: `.venv/bin/python -c "import pylsl; PyQt6; pyqtgraph"` — OK.

*Лог обновлён: 2026-05-14.*

---

## 2026-05-14 — `realtime_gui.py`: импорт `LostError` для текущего pylsl

### Контекст

В установленном **pylsl** класс `LostError` объявлен в `pylsl.util`, но не реэкспортирован из
`pylsl.__init__`, поэтому `from pylsl import LostError, StreamInlet, …` давал `ImportError`, который
перехватывался общим `except` и ошибочно выглядел как «Нужен pylsl». Разделены импорты: основные символы
из `pylsl`, `LostError` — с fallback на `pylsl.util`. Smoke: `run_demo.py` — LSL, MSI, предсказание
10 Hz в логе. Риск **LOW** (`detect_changes` unstaged).

*Лог обновлён: 2026-05-14.*

---

## 2026-05-14 — SSVEP MSI standalone demo (`ssvep_demo/`)

### Контекст

Каталог **`ssvep_demo/`**: стимул PyQt6 (4 частоты), синтетический LSL `fake_eeg_lsl.py` (UDP 17391
смена 10/12/15/20 Hz), `msi_realtime.py` (`RollingEEGBuffer`, `MSIRealtimeClassifier` через
`test_msi_exec` без дублирования CoreCLR), `realtime_gui.py` (LSL + pyqtgraph + MSI), лончер
`run_demo.py`. P300/Qt pipeline не затронуты. В `requirements.txt` добавлен **PyQt6** для демо.

*Лог обновлён: 2026-05-14.*

---

## 2026-05-14 — Standalone `ssvep_analyzer.py` (LSL → MSI, PyQt6)

### Контекст

Добавлен корневой **`ssvep_analyzer.py`**: только LSL EEG → MSI (`test_msi_exec`: `load_msi_runtime`,
`generate_model_signals`, `build_model_signal_list`, `numpy_to_double_matrix2d`) → pyqtgraph + большой
QLabel победителя; rolling buffer 2 s, классификация ~200 ms; частоты — до **6 ламп** через кнопку «+»
и **`QComboBox`** на строку: ровно **500** значений **1000/i Гц**, i = 1 … 500 (как в WinForms `DataGridViewComboBoxColumn`).
По умолчанию список ламп пустой.
UI каналов: чекбоксы по всем каналам потока (подписи из LSL desc/XML), отмеченные каналы — на графике
(стек с шагом `CHANNEL_PLOT_SEP`) и в матрицу **MSIExec**; кнопки **«Снять все»** / **«Выбрать все»** для массового выбора
(если ни один не отмечен — MSI не вызывается).
Переиспользован **`p300_analysis.lsl_streams.stream_inlet_with_buffer`** для inlet. Без `ssvep_demo`,
fake EEG, UDP, stimulus window. P300 / `test_msi_import` / `test_msi_exec` не менялись.

Smoke: `QT_QPA_PLATFORM=offscreen` — Apply frequencies + MSIExec на синтетическом буфере; `python3 scripts/test_msi_exec.py` — ok.
GitNexus `detect_changes` (unstaged) для неотслеживаемого файла мог не увидеть diff — риск по затронутым символам **LOW**.

*Лог обновлён: 2026-05-14.*

---

## 2026-05-14 — `migalka.py`: порт WinForms Migalka на Python

### Контекст

По запросу перенесена логика **`Migalka/MyForm.cs`**: GUI на **tkinter**, COM через **pyserial** (9600 8N1),
шесть частот в одной строке, переключатель **Постоянный / Пакетный** (`M 0` / `M 1`) с повторной отправкой
всех частот после смены режима, **`Settings.xml`** в текущей рабочей директории (как относительный путь в C#),
список частот **`0`** и **1000/i** для **i = 1 … 500**. Зависимость **`pyserial`** добавлена в `requirements.txt`.

Новый скрипт не меняет существующие модули анализа. Риск для основного пайплайна: **LOW**.

*Лог обновлён: 2026-05-14.*

---

## 2026-05-15 — Прослойка пакетного SSVEP (BurstGate + LSL Migalka)

### Контекст

Реализована прослойка для **пакетного** режима без изменений MSI DLL:
- **`ssvep_analysis/burst_gate.py`** — интервалы ON/OFF по LSL, гейтинг окна 2 с (`min_on_fraction`, `min_on_sec`).
- **`ssvep_analysis/migalka_lsl.py`** — поток **`MigalkaStimMarkers`** (`100+i|on|off`).
- **`migalka.py`** — парсинг `LED i t ON/OFF` → LSL; COM 115200.
- **`ssvep_analyzer.py`** — комбо «Постоянный / Пакетный»; в пакетном MSI только при достаточной вспышке; буфер LSL-времени EEG.
- **`tests/test_burst_gate.py`** — 4 теста.

Постоянный режим: поведение как раньше. Пакетный: запустить migalka → ssvep_analyzer (режим пакетный) → Connect EEG → Apply frequencies.

Риск для MSI/P300: **LOW** (новые модули + ветка в ssvep_analyzer).

*Лог обновлён: 2026-05-15.*

---

## 2026-05-20 — Второй winner: template correlation (P300)

### Задача
Добавить второй метод выбора «победителя» для P300 — **template correlation** — чтобы повысить устойчивость к шуму по сравнению с одной лишь AUC-метрикой, и интегрировать выбор режима в GUI.

### Изменения

| Файл | Что изменено |
|---|---|
| `p300_analysis/winner_selection.py` | Добавлен режим `template_corr` (лейблы/перечень режимов выбора победителя). |
| `p300_analysis/erp_compute.py` | `compute_winner_metrics` получил новый `winner_mode=template_corr` и опциональный `template_window`; реализована метрика корреляции с шаблоном. |
| `p300_analysis/qt_window.py` | В GUI добавлен новый пункт режима; шаблон строится из cue target (`lsl_cue_target_id`), с graceful fallback на AUC если шаблон недоступен. |
| `tests/test_winner_metrics.py` | Обновлены/добавлены unit-тесты для `winner_metrics` с новым режимом. |

### GitNexus
- Символов затронуто: 34
- Risk level: MEDIUM
- Affected processes: 4

### Влияние на алгоритм
Теперь winner может определяться не только по AUC, но и по корреляции ERP с шаблоном цели (строится по данным target cue). Если шаблон (или данные для него) отсутствует, логика прозрачно возвращается к AUC без падения пайплайна.

### Что осталось
- [ ] Добавить/проверить e2e smoke для GUI режима `template_corr` в окружении с установленными `pyqtgraph`/`pylsl`.
- [ ] Подумать о сохранении/логировании используемого шаблона и итоговой корреляции для диагностики качества.

*Обновлено: 2026-05-20 10:34*

## 2026-05-15 — Тесты SSVEP (burst_gate, analyzer, MSI templates)

Добавлены: `test_ssvep_burst_gate.py`, `test_ssvep_migalka_lsl.py`, `test_ssvep_msi_templates.py`, `test_ssvep_analyzer.py`, `pytest.ini`.
Запуск: `pytest tests/test_ssvep_*.py tests/test_burst_gate.py` — **37 passed, 1 skipped** (MSIExec integration без CoreCLR в CI-окружении).
Полный `pytest tests/`: 1 старый fail в `test_epoch_geometry` (float), не SSVEP.

*Лог обновлён: 2026-05-15.*

---

## 2026-05-15 — SSVEP Analyzer: layout + сессия

**`ssvep_analyzer.py`**: `QSplitter` — слева прокрутка (LSL, частоты, каналы, сессия, лог), справа графики; плитка **CURRENT TARGET** сверху правой колонки. Риск: **LOW** (UI).

*Лог обновлён: 2026-05-15.*

---

## 2026-05-15 — SSVEP Analyzer: параметры MSI в UI

**`ssvep_analyzer.py`**: группа «Параметры MSI» — дискретизация (Гц), число отсчётов, окно (с), точек, в буфере, время MSIExec; связь с `_nominal_fs`, `_n_template`, `_msi_window_sec`, burst-gate и Apply. Connect подставляет fs из LSL. Риск: **LOW**.

*Лог обновлён: 2026-05-15.*

---

## 2026-05-15 — SSVEP: полный лог эксперимента

**`ssvep_analysis/experiment_logger.py`**, **`ssvep_analyzer.py`**: кнопки «Начать анализ» / «Остановить» / «Сбросить» / «Сохранить лог»; каталог `ssvep_experiment_logs/run_*` с `events.ndjson`, `manifest.json`, `eeg.npz`, `plots/*.png`. События: EEG chunks, LSL markers, burst_gate, msi_classify, winner_changed. Риск: **LOW**.

*Лог обновлён: 2026-05-15.*

---

## 2026-05-20 — Автопротокол 60 прогонов (P300×2 + SSVEP×2)

### Задача
Внедрён единый автопротокол на 60 прогонов для гибридного эксперимента (P300×2 + SSVEP×2) с унифицированным логированием, единым форматом артефактов сессии и GUI-раннером.

### Изменения

| Файл | Что изменено |
|---|---|
| `experiment_protocol/unified_logger.py` | Единый логгер протокола: запись событий и артефактов сессии. |
| `experiment_protocol/protocol_runner.py` | Раннер протокола 60 прогонов (warmup/trial boundaries, orchestration P300+SSVEP). |
| `scripts/protocol_runner_gui.py` | GUI для запуска/контроля автопротокола. |
| `p300_analysis/online_engine.py` | Онлайн-движок P300 для интеграции в протокол. |
| `ssvep_analysis/online_engine.py` | Онлайн-движок SSVEP для интеграции в протокол. |
| `ssvep_analysis/migalka_serial_controller.py` | Управление «мигалкой» по COM (команды как в `migalka.py`: `M 0/1` + 6 частот). |
| `tests/test_protocol_logger.py` | E2E-проверка (unit) корректности протокольного логгера. |

### GitNexus
- Символов затронуто: 34
- Risk level: **MEDIUM**
- Affected processes: 4

### Влияние на алгоритм
- Протокол использует `cue target` из LSL как источник цели для warmup и как границы trial (trial boundaries).
- Все артефакты пишутся в единую папку `session_*`: `events.ndjson`, `manifest.json`, `eeg.npz`, `p300_trials.ndjson`, `ssvep_blocks.ndjson`.
- SSVEP-стимулятор («мигалка») управляется по COM командами в стиле `migalka.py`: режим `M 0/1` и набор из 6 частот.

### Что осталось
- [ ] Полный `pytest` в окружении может потребовать зависимости `pylsl`/`pyqtgraph` (в текущем CI/окружении доступны не всегда).

*Обновлено: 2026-05-20 11:02*

---

## 2026-05-20 — ProtocolRunner: граница P300 trial по `trial_start`, не по смене target

### Задача
При случайном таргете два подряд trial с **одной и той же** плиткой не давали второй записи trial в протоколе: граница считалась только при **смене** `cue_target_tile_id`.

### Изменение
**`experiment_protocol/protocol_runner.py`**: флаг `_p300_trial_armed` выставляется на каждый LSL-маркер `-1|trial_start|target=N` (через `parse_trial_target_tile_id`), после записи решения по trial сбрасывается. Убрана логика `_last_seen_cue` / смены cue как единственного условия старта trial. Событие в лог: `protocol_trial_start_arm`.

### Длительность / 60 единиц
Убран отдельный этап `WarmupTemplate` из дефолтного порядка протокола: после **15 P300(AUC)** идём **сразу** в **15 P300(template_corr)**, затем **15 SSVEP continuous** и **15 SSVEP burst**. Итого ровно **60** основных единиц без доп. “разогрева”.

### Паузы и нумерация экспериментов
В `experiment_protocol/protocol_runner.py` добавлена неблокирующая пауза `pause_between_experiments_s` (по умолчанию 2.0 c), которая вставляется:
- перед стартом P300(template_corr) после завершения P300(AUC)
- перед стартом первого SSVEP continuous блока после P300(template_corr)
- перед каждым SSVEP-блоком (continuous и burst), а также между режимами continuous → burst

В `status_text` добавлена нумерация **Эксперимент X/60** (P300 обновляется на `trial_start|target=N`, SSVEP — в ходе блока и на паузах).

### Verbose console + migalka SSVEP
- `experiment_protocol/protocol_log.py` — подробный вывод в консоль (`[protocol]`, `[migalka]`).
- `protocol_runner.py`: лог смены состояний, LSL trial_start/end, ошибки COM при старте мигалки; блок SSVEP не стартует по таймеру, пока COM не открыт.
- `protocol_runner_gui.py`: печать смены status и аргументов стимулятора.
- `tests/test_protocol_runner_fsm.py` — smoke FSM до SSVEP (mock serial/LSL).

### Перенос шаблона между режимами P300
В `p300_analysis/online_engine.py` добавлен внешний шаблон (`external_template_window`), который **не сбрасывается** при `reset()` и может использоваться в `template_corr`.

В `experiment_protocol/protocol_runner.py` во время P300(AUC) “в фоне” пытаемся собрать этот шаблон из первых `template_warmup_target_epochs` эпох для текущего `trial_start|target=N`. Когда шаблон готов — логируем событие `p300_external_template_ready`. Затем в блоке P300(template_corr) этот внешний шаблон используется как приоритетный.

### Управляемая цель в авто-стимуляторе для набора шаблона
В `gui/gui.py` / `run_app.py` добавлены параметры авто-режима:
- `--auto-calib-trials N`: первые N trial идут с фиксированной целью
- `--auto-calib-target-tile-id K`: какая цель (0..8) для этих N trial

Это помогает для каждого испытуемого быстрее/надёжнее собрать персональный шаблон в начале (в AUC), без удлинения протокола сверх 60.

Обновление: вместо “подряд фиксированной цели” добавлен **план** первых `--auto-plan-trials` trial, где выбранная плитка встречается нужное число раз **не подряд**. Число повторов можно задавать явно `--auto-plan-target-repeats` или считать автоматически через `--auto-plan-target-epochs` и `sequences` (в одном trial target даёт примерно `sequences` эпох).

### GitNexus
- `detect_changes` (unstaged в сессии): **MEDIUM** (как и другие правки протокола/GUI в ветке).

### Тесты
- `python -m py_compile experiment_protocol/protocol_runner.py`
- `pytest tests/test_protocol_logger.py -q`

### GUI: COM combo + мигалка + P300 каналы
- `scripts/protocol_runner_gui.py`: COM — `QComboBox` + «Обновить» (`serial.tools.list_ports`); группа «Мигалка» — 6 частот ламп (Hz); группа «P300» — профиль/каналы (1-based).
- `ProtocolConfig`: `roi_channels_0idx`; при Start логируются каналы и `migalka_freqs_Hz`.
- Риск GitNexus: LOW (GUI + config fields, без смены FSM).

### Протокол: автоматический переход P300 → ССВП + мигалка
- Маркеры только `BCI_StimMarkers` (не MigalkaStimMarkers); ожидание стимулятора до 20 с.
- Частота MSI = nominal_srate потока ЭЭГ (NeuroSpectrum 5000 Гц).
- Повтор открытия COM мигалки при ошибке; стимулятор: `auto_max_trials = p300×2+2`.
- GUI: пауза 3 с после старта PsychoPy перед preflight.

### GUI: выбор потока ЭЭГ (LSL)
- `discover_eeg_streams`, `select_eeg_stream`, `stream_display_label` в `lsl_streams.py`
- GUI: список + «Обновить» + «Проверить»; `ProtocolConfig.eeg_stream_name/session_id`

### GUI: частоты выпадающие, 4 лампы, SSVEP каналы
- Частоты: `QComboBox` + `lamp_frequency_choices()` (как SSVEP Analyzer); по умолчанию 4 лампы (`1000/i`), +/− до 6.
- `ProtocolConfig.ssvep_roi_channels_0idx`, `SSVEPParams.roi_channels_0idx` в MSI.
- Дефолт `ssvep_freqs_hz` = 4 частоты из `DEFAULT_LAMP_FREQS`.

### SSVEP Analyzer: подвисание / искажение графика
- Причины: Gantt пересборка ~25/s, автолог EEG без лимита, `vstack` буфера, ось графика не по LSL-времени.
- Фикс: throttle plot 120 ms / gantt 450 ms; EEG debug cap 90 s; `trim_eeg_samples`; ось X = LSL; лимит строк лога.
- GitNexus `_on_pull`: **LOW**.

### Пакетный SSVEP: автолог + фикс частых MSI
- **Причина 26/50**: MSI каждые 200 ms по скользящему 2 с буферу при открытом гейте; «хвосты» маркеров других ламп.
- `ssvep_burst_debug/run_*`: автолог (events, eeg.npz, `burst_summary.json`) при Connect в пакетном режиме.
- Пакетный MSI: не чаще одного `MSIExec` на длину окна; `burst_gate.window_diagnostics`, prune интервалов.
- События: `burst_msi` с `expected_from_markers`, `match_markers`, `on_fraction_by_lamp_0idx`.
- GitNexus: **LOW**.

### SSVEP burst: AttributeError float has no attribute size
- **Причина**: `can_classify()` передавал `float(self._buf_t[-1])` в `BurstGate.classify_allowed`, ожидается `np.ndarray`.
- **Фикс**: `np.asarray(self._buf_t)`; в `reset()` — `set_active_lamps(len(freqs_hz))`.
- Тест: `tests/test_ssvep_online_engine_burst.py`.

### SSVEP протокол: RuntimeError «runtime has already been loaded»
- **Причина**: `SSVEPOnlineEngine.reset()` обнулял `_msi`; повторный `load_msi_runtime()` → `set_runtime()` в pythonnet (один раз на процесс).
- **Фикс**: `_configure_coreclr` — пропуск, если `pythonnet._RUNTIME` или `_LOADED`; кэш `_MSI_RUNTIME_CACHE` в `load_msi_runtime()`; `reset()` не сбрасывает MSI, только буфер + `_apply_templates()`.
- Тест: `tests/test_msi_runtime_idempotent.py`.
- GitNexus `detect_changes`: **HIGH** (затронуты GUI/SSVEP потоки с `load_msi_runtime`), логика — идемпотентная, регрессия ожидается только при смене DOTNET_ROOT в одном процессе.

*Обновлено: 2026-05-20*

---

## 2026-05-21 — Аудит `experiment_protocol` (синхронизация с другого устройства)

### Состояние git
- Ветка `main`, синхронизирована с `origin/main` (HEAD `3a0a997`).
- Локальных незакоммиченных правок в `experiment_protocol/` нет.
- Последний коммит по протоколу: handoff continuous→burst, COM не закрывается между режимами; GUI ждёт `BCI_StimMarkers` при повторном запуске.

### Архитектура пакета (кратко)
| Файл | Роль |
|------|------|
| `protocol_runner.py` | FSM без UI: LSL EEG+маркеры, P300/SSVEP движки, COM мигалка, паузы, 60 единиц |
| `unified_logger.py` | Одна сессия: `events.ndjson`, `p300_trials.ndjson`, `ssvep_blocks.ndjson`, `eeg.npz`, `manifest.json` |
| `protocol_log.py` | Verbose stdout `[protocol]` |
| `ssvep_cue_overlay.py` | PyQt fullscreen: подсказка лампы / чёрный экран |
| `scripts/protocol_runner_gui.py` | Оператор: настройки, QTimer→`tick()`, оверлей, автозапуск PsychoPy |

### FSM (60 = 15×4 по умолчанию)
`idle` → `preflight` → `p300_auc` (15) → пауза → `p300_template` (15) → пауза+оверлей → `ssvep_continuous` (15) → handoff COM → `ssvep_burst` (15) → `finalize` → `stopped`.

### GitNexus
- `npx gitnexus analyze` — **FAILED** (`scopeResolution`: object not extensible; scope failed для `protocol_runner.py`, `protocol_runner_gui.py`).
- MCP `context(ProtocolRunner)` — symbol not found (индекс устарел/битый).
- **Риск правок протокола по графу сейчас не оценить** — после починки analyze перезапустить `impact`/`detect_changes`.

### Тесты (локально)
- `pytest tests/test_protocol_*` — ERROR: `ModuleNotFoundError: pylsl` в default conda env; нужен venv проекта с `pylsl`.

*Обновлено: 2026-05-21*

### Протокол v2 (калибровка + 45 shuffle + dual P300)
- `experiment_protocol/experiment_queue.py` — очередь 15+15+15, seed, баланс ламп.
- `protocol_runner.py` — FSM `p300_calib` → `main`; один trial P300 → AUC + `template_corr`; `experiments.ndjson`.
- `unified_logger.py` — `append_experiment`, путь в manifest.
- `protocol_runner_gui.py` — группы «Протокол» и «Стимулятор», все параметры (калибровка, main, пауза, seed, block_sec).
- `docs/PROTOCOL_V2_PLAN.md` — план и статус.
- Тесты: `test_experiment_queue.py`, обновлены FSM/handoff/logger.
- GitNexus: индекс по-прежнему битый (`analyze` failed).

### Протокол v2 — фаза 2 (stim_control + оверлей + CSV)
- `stim_control.py` + `--stim-control-dir`: P300 main trial только по команде протокола (между SSVEP).
- `protocol_instruction_overlay.py` + синхронизация в GUI (калибровка / P300 / SSVEP / пауза / blackout).
- `experiment_queue`: `target_tile_id` для P300 в очереди.
- `experiments.csv` при `finalize()`; сессия создаётся до стимулятора (`allocate_session_dir`).
- Тесты: `test_stim_control.py`.

### GUI: калибровка не перекрывает плитки PsychoPy
- `ProtocolInstructionOverlay` скрывается при `calib`/`p300`, пока запущен `run_app` (подсказки — на экране стимулятора).
