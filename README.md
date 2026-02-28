# Hybrid BCI P300-SSVEP Stimulus System

Система визуальной стимуляции для гибридного BCI-интерфейса с поддержкой протокола LSL (Lab Streaming Layer) для передачи маркеров событий.

## Требования

- Python 3.10 (PsychoPy не поддерживает Python 3.11+)
- macOS / Linux / Windows

## Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd hybrid-bci-p300-ssvep
```

### 2. Создание виртуального окружения с Python 3.10

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

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск приложения

### Базовый запуск

```bash
source .venv/bin/activate  # macOS/Linux
# или
.venv\Scripts\activate     # Windows

python -m app.main
```

### Управление

- **START** — начать стимуляцию (мигание плиток)
- **STOP** — остановить стимуляцию
- **Слайдер "Интервал"** — настройка времени между вспышками (ISI, 0.02–5.0 сек)
- **Слайдер "Длительность горения"** — настройка длительности вспышки (0.05–2.0 сек)
- **Escape** — выход из приложения

## Протокол LSL

Приложение автоматически создаёт LSL-стрим для отправки маркеров событий:

- **Имя стрима:** `BCI_StimMarkers`
- **Тип:** `Markers`
- **Формат маркера:** `<tile_id>|<event>`, где:
  - `tile_id` — ID плитки (0–8 для сетки 3×3)
  - `event` — тип события: `on` (загорание) или `off` (затухание)

### Примеры маркеров

```
0|on   — плитка 0 загорелась
0|off  — плитка 0 погасла
5|on   — плитка 5 загорелась
5|off  — плитка 5 погасла
```

### Проверка LSL-стрима

**В отдельном терминале** (с активированным venv):

```bash
python -c "
from pylsl import StreamInlet, resolve_byprop
print('Waiting for BCI_StimMarkers stream...')
streams = resolve_byprop('name', 'BCI_StimMarkers', timeout=5)
if not streams:
    print('No stream found!')
else:
    inlet = StreamInlet(streams[0])
    print('Connected! Receiving markers:')
    while True:
        sample, timestamp = inlet.pull_sample()
        print(f'{timestamp:.3f}: {sample[0]}')
"
```

**Важно:** сначала запустите приложение и нажмите START, затем запускайте скрипт проверки.

## Структура проекта

```
hybrid-bci-p300-ssvep/
├── app/
│   └── main.py              # Точка входа
├── core/
│   ├── grid.py              # Логика сетки плиток
│   ├── tile.py              # Модель плитки
│   └── stimulus_controller.py  # Контроллер стимуляции + LSL
├── gui/
│   └── gui.py               # Интерфейс PsychoPy
├── requirements.txt         # Зависимости
└── README.md               # Этот файл
```

## Зависимости

- **psychopy** — визуализация и управление стимулами
- **pylsl** — отправка маркеров через Lab Streaming Layer

## Разработка

### Переключение между версиями

Проект использует Git. Коммиты:
- `dafc3c8` — текущая версия с LSL
- `645dca9` — версия без LSL (слайдеры)

Для временного отката к версии без LSL:
```bash
git checkout 645dca9
```

Возврат к версии с LSL:
```bash
git checkout develop
```

### Работа с .gitignore

Проект игнорирует:
- `__pycache__/` и `*.pyc`
- `.venv/`, `venv/`
- `.idea/`, `.vscode/`
- `.psychopy3/`
- `.DS_Store`, `Thumbs.db`

## Устранение проблем

### Ошибка при импорте psychopy

**Проблема:** `PermissionError: [Errno 1] Operation not permitted: '/Users/.../.psychopy3/themes/ClassicDark.json'`

**Решение:** Запустите с правами доступа к `~/.psychopy3/` или удалите эту папку:
```bash
rm -rf ~/.psychopy3
```

### Python 3.13 не поддерживается

PsychoPy требует Python 3.8–3.10. Используйте Python 3.10.

### LSL-стрим не найден

Убедитесь, что:
1. Приложение запущено
2. Нажата кнопка START
3. В коде `StimulusController._init_lsl()` нет ошибок (проверьте консоль)

## Лицензия

[Укажите лицензию]

## Контакты

[Укажите контакты]
