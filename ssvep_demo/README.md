# SSVEP + MSI standalone demo

Автономный прототип: **визуальный SSVEP-стимул** → **LSL EEG** (синтетика или реальный усилитель) → **скользящее окно** → **MSIController.MSIExec** → **предсказание частоты** в **PyQt6 + pyqtgraph**.

Не связан с P300-анализатором (`p300_analysis/`, `qt_window.py` не используются).

## Зависимости

Из корня репозитория (уже в основном `requirements.txt`):

- `numpy`, `pylsl`, `pyqtgraph`, `pythonnet`, `PyQt6`
- **.NET 8** runtime + файлы в `msi-res/` (см. `scripts/test_msi_import.py`)

```bash
pip install -r requirements.txt
```

## Запуск одной командой

Запускайте **тем же интерпретатором, где установлены зависимости** (активированный venv или явный путь к `.venv/bin/python`). Лончер порождает subprocess только с ``sys.executable`` текущего процесса (не ``python``/``python3`` из PATH).

```bash
cd /path/to/hybrid-bci-p300-ssvep
.venv/bin/python ssvep_demo/run_demo.py
```

Откроются три процесса: синтетический LSL, окно мигания, GUI анализатора. В GUI нажмите **«Подключиться»** к потоку `SSVEP-Demo-EEG`.

## Запуск по частям

```bash
export PYTHONPATH="$PWD:$PWD/scripts"

python -m ssvep_demo.fake_eeg_lsl
python -m ssvep_demo.stimulus_window
python -m ssvep_demo.realtime_gui
```

## Смена частоты в синтетике

Процесс `fake_eeg_lsl` слушает **UDP 127.0.0.1:17391** (байты `1`…`4`):

```bash
printf '1' | nc -u -w0 127.0.0.1 17391   # 10 Hz
printf '2' | nc -u -w0 127.0.0.1 17391   # 12 Hz
```

В логе `fake_eeg_lsl` появится строка `switched to … Hz`.

## Реальный EEG

Запустите только `stimulus_window` + `realtime_gui`, источник LSL с типом `EEG`, **2 канала**, по возможности **250 Hz** (классификатор и шаблоны MSI заточены под 250 Hz / окно 2 с).

## Ограничения прототипа

- Нет калибровки, меню, записи, повторного просмотра.
- MSI и шаблоны зашиты под **10 / 12 / 15 / 20 Hz** и **2 канала**.
- Качество классификации на синтетике зависит от фазы/шума; для «идеального» теста используйте `scripts/test_msi_exec.py`.

## Файлы

| Файл | Назначение |
|------|------------|
| `stimulus_window.py` | 4 квадрата SSVEP, F11 fullscreen, Esc закрыть |
| `fake_eeg_lsl.py` | LSL `EEG` 250 Hz, 2 ch, UDP смена частоты |
| `msi_realtime.py` | `RollingEEGBuffer`, `MSIRealtimeClassifier` (импорт `test_msi_exec`) |
| `realtime_gui.py` | LSL + буфер + MSI + график |
| `run_demo.py` | Лончер трёх процессов |
