# Hybrid BCI P300-SSVEP

Система визуальной стимуляции и мониторинга для гибридного BCI-интерфейса (P300) с поддержкой протокола LSL (Lab Streaming Layer).

## Требования

- **Python 3.10** (PsychoPy не поддерживает Python 3.11+)
- macOS / Linux / Windows

## Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd hybrid-bci-p300-ssvep
```

### 2. Виртуальное окружение (Python 3.10)

**macOS/Linux:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python3.10 -m venv .venv
.venv\Scripts\activate
```

### 3. Зависимости

```bash
pip install -r requirements.txt
```

## Запуск

### Приложение стимуляции (основное)

Запускает окно с сеткой 3×3 плиток и кнопками управления:

```bash
python -m app.main
```

**Управление:**
- **START** — начать стимуляцию (мигание плиток)
- **STOP** — остановить стимуляцию
- **Слайдер "Интервал"** — время между вспышками (ISI, 0.02–5.0 сек)
- **Слайдер "Длительность горения"** — длительность вспышки (0.05–2.0 сек)
- **Escape** — выход

### BCI Monitor Dashboard (мониторинг P300)

Дашборд для просмотра ЭЭГ и накопленных эпох P300 в реальном времени:

```bash
python scripts/bci_monitor_dashboard.py
```

**Перед запуском дашборда:**
1. Запустите приложение стимуляции (`python -m app.main`) и нажмите START
2. Запустите источник ЭЭГ (например, Нейроспектр) с LSL-трансляцией

**Зоны интерфейса:**
- **Верх** — бегущая волна ЭЭГ (10 сек), маркеры вспышек, переключатель фильтра 0.5–10 Гц
- **Слева внизу** — сетка 3×3 графиков усреднённых эпох P300 по плиткам
- **Справа внизу** — метрики (эпохи, качество сигнала, результат классификатора), кнопки Start Processing / Reset / Stop & Save

Подробнее: [scripts/README_DASHBOARD.md](scripts/README_DASHBOARD.md)

### Вспомогательные скрипты

| Скрипт | Назначение |
|--------|------------|
| `scripts/lsl_listen.py` | Проверка приёма маркеров LSL в консоли |
| `scripts/hardware_validation.py` | Визуализация каналов ЭЭГ и матрица ковариаций для проверки аппаратуры |

## Протокол LSL

Приложение стимуляции создаёт LSL-стрим маркеров:

- **Имя:** `BCI_StimMarkers`
- **Тип:** `Markers`
- **Формат:** `<tile_id>|<event>` (например `0|on`, `0|off`, `5|on`)

Дашборд ожидает два потока:
- маркеры из `BCI_StimMarkers`
- ЭЭГ с типом `EEG` (или как указано в `eeg_epoch_processor`)

## Структура проекта

```
hybrid-bci-p300-ssvep/
├── app/
│   └── main.py                    # Точка входа приложения стимуляции
├── config.py                      # Константы (окно, сетка, слайдеры)
├── core/
│   ├── tile.py                    # Модель плитки
│   ├── grid.py                    # Модель сетки плиток
│   ├── lsl.py                     # Отправка маркеров в LSL
│   └── stimulus_controller.py     # Контроллер стимуляции
├── gui/
│   └── gui.py                     # Интерфейс PsychoPy
├── scripts/
│   ├── bci_monitor_dashboard.py   # Дашборд мониторинга P300 (PyQt5 + pyqtgraph)
│   ├── eeg_epoch_processor.py     # Буфер ЭЭГ, извлечение эпох, фильтрация
│   ├── hardware_validation.py     # Валидация каналов и ковариация
│   ├── lsl_listen.py              # Прослушивание маркеров LSL
│   └── README_DASHBOARD.md        # Документация дашборда
├── test/
│   └── test_lsl_streams.py        # Тесты обнаружения LSL-стримов
├── requirements.txt
└── README.md
```

## Зависимости

- **psychopy** — визуализация и управление стимулами
- **pylsl** — Lab Streaming Layer
- **numpy**, **scipy** — обработка сигналов
- **PyQt5**, **pyqtgraph** — дашборд мониторинга

## Устранение проблем

### PermissionError при импорте psychopy

Удалите кэш темы PsychoPy:
```bash
rm -rf ~/.psychopy3
```

### LSL-стрим не найден

- Убедитесь, что приложение стимуляции запущено и нажата кнопка START
- Для дашборда — что ЭЭГ-источник транслирует поток в LSL
- Проверка: `python scripts/lsl_listen.py`

### Python 3.11+

Используйте Python 3.10 (ограничение PsychoPy).

## Лицензия

[Укажите лицензию]

## Контакты

[Укажите контакты]
