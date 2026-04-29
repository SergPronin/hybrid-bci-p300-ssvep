# Как добавить новый метод выбора победителя без изменения текущей логики

Этот гайд описывает безопасный способ добавить новый `winner_mode` в проект, **не меняя поведение существующих режимов** (`auc`, `signed_mean`, `msi`).

---

## Цель

Добавить новый алгоритм как отдельный режим, при этом:

- не трогать формулы существующих методов;
- не ломать онлайн/оффлайн работу;
- получить видимый результат в UI;
- оставить возможность легко откатить изменение.

---

## Текущие точки расширения

В проекте выбор победителя проходит через:

1. `p300_analysis/winner_selection.py`  
   Хранит константы `WINNER_MODE_*` и короткие подписи.

2. `p300_analysis/erp_compute.py`  
   Функция `compute_winner_metrics(...)` выбирает `final_metric_values` по режиму и делает `argmax`.

3. `p300_analysis/qt_window.py`  
   Combobox с вариантами режима в UI.

---

## Важный контракт нового метода

Ваш алгоритм должен вернуть:

- `np.ndarray` формы `(n_stim,)`;
- где `n_stim = len(stim_keys)`;
- и правило: **больше значение = более вероятный победитель**  
  (потому что дальше используется `np.argmax(...)`).

Если у вас метрика вида “меньше лучше”, инвертируйте ее перед возвратом (например `-metric`).

---

## Рекомендуемая структура (без изменения текущих формул)

### 1) Создайте отдельный модуль алгоритма

Пример: `p300_analysis/my_new_algorithm.py`

```python
from __future__ import annotations
import numpy as np

def compute_my_method_scores(*, corr_win: np.ndarray, abs_auc_values: np.ndarray) -> np.ndarray:
    if corr_win.size == 0:
        return np.zeros(abs_auc_values.shape[0], dtype=np.float64)

    # TODO: вставьте формулу
    score = abs_auc_values.astype(np.float64, copy=True)

    # Защита от NaN/Inf
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
```

### 2) Добавьте новый режим в `winner_selection.py`

- константа: `WINNER_MODE_MY = "my_method"`
- подпись в `MODE_SHORT_LABELS`

Пример:

```python
WINNER_MODE_MY = "my_method"

MODE_SHORT_LABELS = {
    WINNER_MODE_AUC: "auc",
    WINNER_MODE_SIGNED_MEAN: "signed_mean",
    WINNER_MODE_MSI: "msi",
    WINNER_MODE_MY: "my_method",
}
```

### 3) Подключите только новую ветку в `compute_winner_metrics(...)`

В `p300_analysis/erp_compute.py`:

- импортируйте `WINNER_MODE_MY`;
- импортируйте функцию `compute_my_method_scores`;
- добавьте `elif winner_mode == WINNER_MODE_MY`.

Важно: **ветки `auc/signed_mean/msi` не менять**.

Пример:

```python
if winner_mode == WINNER_MODE_SIGNED_MEAN:
    final_metric_values = signed_mean_values
    mode_used = WINNER_MODE_SIGNED_MEAN
elif winner_mode == WINNER_MODE_MSI:
    final_metric_values = compute_msi_like_scores(corr_win=corr_win, abs_auc_values=abs_auc_values)
    mode_used = WINNER_MODE_MSI
elif winner_mode == WINNER_MODE_MY:
    final_metric_values = compute_my_method_scores(corr_win=corr_win, abs_auc_values=abs_auc_values)
    mode_used = WINNER_MODE_MY
else:
    final_metric_values = abs_auc_values
    mode_used = WINNER_MODE_AUC
```

### 4) Добавьте режим в UI (`qt_window.py`)

В блоке `self.combo_winner_mode.addItem(...)`:

```python
self.combo_winner_mode.addItem("My method (описание)", WINNER_MODE_MY)
```

После этого режим будет доступен и в онлайн, и в оффлайн (через общий `_redraw_from_epochs`).

---

## Почему это не ломает текущую логику

- существующие формулы не меняются;
- добавляется только новый независимый `elif`;
- если пользователь не выбирает новый режим, поведение идентично предыдущему.

---

## Что должно попасть в debug/export

Автоматически уже будет:

- `winner_rule`
- `final_metric_values`
- `margin`
- `chosen_winner_key`

Это помогает сравнивать новый метод с текущими.

---

## Проверочный чеклист (обязательно)

1. Синтаксис:
   - `python -m py_compile p300_analysis/your_file.py p300_analysis/erp_compute.py p300_analysis/qt_window.py`

2. Тесты:
   - `PYTHONPATH=. pytest -q tests/test_winner_metrics.py`
   - по возможности `PYTHONPATH=. pytest -q` (весь набор)

3. Оффлайн проверка:
   - загрузить `continuous CSV`,
   - переключить режимы в UI,
   - убедиться, что меняются `режим` и `⑤ Score по классам`.

4. Онлайн проверка:
   - запустить LSL,
   - переключать режимы во время накопления,
   - убедиться, что winner пересчитывается.

5. Регрессия по данным:
   - прогнать `scripts/regression_test.py` (или ваш сравнительный скрипт) по нескольким сессиям и сравнить `auc/msi/my_method`.

---

## Типичные ошибки

- Возвращается скаляр вместо вектора `(n_stim,)`.
- Вектор содержит `NaN/Inf`.
- Метрика инвертирована (“меньше лучше”), но используется `argmax`.
- Новый режим добавлен в вычисление, но не добавлен в UI.
- Режим добавлен в UI, но не добавлен в `winner_selection`.

---

## Рекомендация по процессу

- Делайте изменение в отдельной ветке.
- Сначала добавьте “пустой” метод (например, повторяющий `auc`) и убедитесь, что wiring работает.
- Потом меняйте только формулу в новом файле.

Так вы минимизируете риск и точно не заденете актуальную логику.
