# Hybrid BCI — стимуляция P300 + анализатор

Минимальный проект: визуальная сетка плиток с маркерами в LSL и отдельное окно анализа P300.

## Требования

- Python 3.10 (для PsychoPy см. актуальную документацию по версиям)
- macOS / Linux / Windows

## Установка

```bash
cd hybrid-bci-p300-ssvep
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Запуск

**Стимуляция (PsychoPy, 3×3 плитки, LSL Markers):** из корня репозитория:

```bash
python run_app.py
```

**Анализатор P300 (подключение к ЭЭГ и потоку Markers):**

```bash
python scripts/p300_analyzer.py
```

Лог анализатора: `scripts/p300_analyzer.log`.

## Структура

| Путь | Назначение |
|------|------------|
| `run_app.py` | Точка входа GUI; при необходимости перезапускает себя из `.venv` |
| `app/main.py` | Создаёт `StimulusApp` |
| `gui/gui.py` | Окно PsychoPy, слайдеры, сетка |
| `config.py` | Размеры окна, сетка, позиции элементов |
| `core/` | Сетка, плитки, контроллер последовательности, `LslMarkerSender` |
| `scripts/p300_analyzer.py` | Онлайн-усреднение ERP, выбор «плитки» по метрике |

Лицензия: MIT (см. `LICENSE`).
