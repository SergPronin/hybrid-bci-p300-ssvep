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

## Сборка в исполняемый файл (exe/app)

Для `scripts/hardware_validation.py` есть готовый spec-файл PyInstaller: `hardware_validation.spec`.

- **Windows (exe)**

  ```bash
  py -3.10 -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  pip install -r requirements-dev.txt

  py -3.10 -m PyInstaller --clean --noconfirm hardware_validation.spec
  # или:
  py scripts\build_hardware_validation_exe.py
  ```

  Результат: `dist/hardware_validation/hardware_validation.exe`

- **macOS/Linux (app/onefolder)**

  ```bash
  python3.10 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install -r requirements-dev.txt

  python -m PyInstaller --clean --noconfirm hardware_validation.spec
  ```

  Результат: `dist/hardware_validation/hardware_validation` (на macOS можно упаковать дальше в `.app`, если нужно).

## LSL‑потоки

- Маркеры стимуляции: поток `BCI_StimMarkers`, тип `Markers`, формат `tile_id|event` (например `3|on`, `3|off`).
- ЭЭГ: поток типа `EEG` или `Signal` от оборудования / симулятора.

## Структура

- `app/` — точка входа `main.py`.
- `core/` — сетка плиток, контроллер стимуляции, отправка LSL‑маркеров.
- `gui/` — GUI на PsychoPy.
- `scripts/` — утилиты, в т.ч. `hardware_validation.py` и `lsl_record_minimal.py`.

Лицензия: MIT (см. `LICENSE`).
