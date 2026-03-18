# Hybrid BCI P300-SSVEP

Кросс‑платформенное приложение визуальной стимуляции для гибридного BCI (P300) с публикацией маркеров в LSL и отдельным GUI для валидации ЭЭГ.

## Требования

- Python 3.10 (PsychoPy не поддерживает 3.11+)
- macOS / Linux / Windows

## Установка

```bash
git clone <repository-url>
cd hybrid-bci-p300-ssvep

python3.10 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Запуск

- **Валидация аппаратуры ЭЭГ**  
  ```bash
  python scripts/hardware_validation.py
  ```  
  Окно Qt отображает каналы ЭЭГ, позволяет менять масштаб X/Y и сохранять данные в `saved_data/`.

## Быстрый старт из архива

1. Установите зависимости:
   ```bash
   python bootstrap_env.py
   ```
2. Запустите GUI:
   ```bash
   python run_app.py
   ```
3. Запустите валидацию ЭЭГ:
   ```bash
   python run_hardware_validation.py
   ```

## LSL‑потоки

- Маркеры стимуляции: поток `BCI_StimMarkers`, тип `Markers`, формат `tile_id|event` (например `3|on`, `3|off`).
- ЭЭГ: поток типа `EEG` или `Signal` от оборудования / симулятора.

## Структура

- `app/` — точка входа `main.py`.
- `core/` — сетка плиток, контроллер стимуляции, отправка LSL‑маркеров.
- `gui/` — GUI на PsychoPy.
- `scripts/` — утилиты, в т.ч. `hardware_validation.py`.

Лицензия: MIT (см. `LICENSE`).
