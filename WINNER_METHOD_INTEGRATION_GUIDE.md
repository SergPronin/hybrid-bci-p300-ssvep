# Интеграция нового метода определения победителя (P300)

Этот документ описывает, как добавить новый метод выбора победителя в текущую архитектуру проекта **без поломки существующей логики**.

Документ рассчитан на разработчика, который хочет:
- быстро добавить новый режим расчета;
- минимально трогать существующий код;
- понимать, какие входы/выходы ожидаются;
- избежать типичных ошибок.

---

## 1. Где в проекте принимается решение о победителе

Цепочка вызовов:

1. `p300_analysis/qt_window.py`  
   UI передает выбранный `winner_mode` в расчеты.
2. `p300_analysis/erp_compute.py`  
   Функция `compute_winner_metrics(...)` считает метрики и выбирает победителя через `np.argmax(...)`.
3. `p300_analysis/winner_selection.py`  
   Хранит константы режимов и короткие подписи.

Ключевая идея:  
**Победитель = индекс максимума вектора `final_metric_values`.**

---

## 2. Минимальный путь добавления нового метода

Чтобы добавить новый метод, нужно изменить 3 места:

1. `winner_selection.py` — добавить новый `WINNER_MODE_*`.
2. `qt_window.py` — добавить этот режим в выпадающий список UI.
3. `erp_compute.py` — добавить ветку расчета `final_metric_values` для нового режима.

Это текущий простой и надежный путь под существующую архитектуру.

---

## 3. Контракт вашего метода (самое важное)

Ваш метод в итоге должен дать:

- `final_metric_values: np.ndarray`
- форма: `(n_stim,)`
- где `n_stim == len(stim_keys)`
- правило: **чем больше значение, тем лучше кандидат**

Почему: дальше вызывается `np.argmax(final_metric_values)`.

Если у вас метрика вида "меньше лучше", инвертируйте ее, например:

```python
final_metric_values = -distance_values
```

---

## 4. Пошаговая инструкция с шаблонами

### Шаг 1. Добавьте константу режима в `winner_selection.py`

Добавьте новый идентификатор:

```python
WINNER_MODE_MY_METHOD = "my_method"
```

И добавьте короткую подпись:

```python
MODE_SHORT_LABELS = {
    WINNER_MODE_AUC: "auc",
    WINNER_MODE_SIGNED_MEAN: "signed_mean",
    WINNER_MODE_MY_METHOD: "my_method",
}
```

Рекомендации:
- используйте `snake_case`;
- не используйте пробелы;
- делайте имя стабильным (чтобы не ломать логи/экспорт).

---

### Шаг 2. Добавьте режим в UI (`qt_window.py`)

Найдите место с `self.combo_winner_mode.addItem(...)` и добавьте пункт:

```python
self.combo_winner_mode.addItem(
    "Мой метод (краткое описание)", WINNER_MODE_MY_METHOD
)
```

Также добавьте импорт `WINNER_MODE_MY_METHOD` рядом с другими `WINNER_MODE_*`.

---

### Шаг 3. Подключите расчет в `erp_compute.py`

В `compute_winner_metrics(...)` уже есть расчет базовых метрик:

- `abs_auc_values`
- `signed_mean_values`
- `corr_win`
- и др.

Добавьте ветку для нового режима:

```python
if winner_mode == WINNER_MODE_SIGNED_MEAN:
    final_metric_values = signed_mean_values
elif winner_mode == WINNER_MODE_MY_METHOD:
    final_metric_values = ...  # ВАША ФОРМУЛА
else:
    final_metric_values = abs_auc_values
```

Важно:
- `final_metric_values` должен быть `np.ndarray`;
- длина должна совпадать с количеством стимулов;
- избегайте `NaN/Inf`.

---

## 5. Рекомендуемый формат "коннектора" (чтобы человек менял 1 место)

Чтобы другому разработчику было проще, вынесите формулу в отдельную функцию в `erp_compute.py`.

Пример:

```python
def compute_custom_winner_metric(
    *,
    corr_win: np.ndarray,
    abs_auc_values: np.ndarray,
    signed_mean_values: np.ndarray,
    dt_m: float,
) -> np.ndarray:
    """
    Возвращает вектор score формы (n_stim,).
    Больше score => выше шанс победителя.
    """
    # TODO: заменить на пользовательскую формулу
    return abs_auc_values
```

И используйте ее:

```python
elif winner_mode == WINNER_MODE_MY_METHOD:
    final_metric_values = compute_custom_winner_metric(
        corr_win=corr_win,
        abs_auc_values=abs_auc_values,
        signed_mean_values=signed_mean_values,
        dt_m=dt_m,
    )
```

Плюсы:
- человек меняет только функцию-коннектор;
- меньше риска повредить основную логику.

---

## 6. Какие данные доступны внутри расчета

В контексте `compute_winner_metrics(...)` обычно есть:

- `stim_keys: List[str]` — метки стимулов;
- `corrected: np.ndarray` — baseline-corrected ERP;
- `time_ms: np.ndarray` — временная ось;
- `window_x_ms`, `window_y_ms` — окно анализа;
- `corr_win: np.ndarray` — срез `corrected` по окну;
- `abs_auc_values: np.ndarray` — AUC по модулю;
- `signed_mean_values: np.ndarray` — signed mean;
- `positive_peak_values: np.ndarray` — positive peak;
- `dt_m: float` — шаг по времени.

Этого достаточно для большинства кастомных формул.

---

## 7. Требования к качеству результата

Перед возвратом итоговой метрики проверяйте:

1. `final_metric_values.ndim == 1`
2. `final_metric_values.shape[0] == len(stim_keys)`
3. `np.all(np.isfinite(final_metric_values))`

При желании можно добавить защиту:

```python
final_metric_values = np.nan_to_num(final_metric_values, nan=0.0, posinf=0.0, neginf=0.0)
```

---

## 8. Как считается уверенность (margin) и почему это важно

В текущем коде уверенность считается как:

```text
margin = (top1 - top2) / |top1|
```

Где:
- `top1` — лучший score;
- `top2` — второй score.

Следствие:
- если ваши scores очень "сжатые", margin может быть низким даже при корректном выборе;
- если top1 около нуля, margin будет нестабильнее.

Если ваш метод меняет масштаб scores, это нормально, но стоит проверить поведение margin на реальных данных.

---

## 9. Типичные ошибки и как избежать

1. **Не добавили режим в UI**  
   Метод есть в коде, но его нельзя выбрать.

2. **Неправильная форма массива**  
   Например, `(n_stim, 1)` вместо `(n_stim,)`.

3. **Инвертирован смысл метрики**  
   Вы используете "меньше лучше", но код берет `argmax`.

4. **Появляются `NaN`**  
   Из-за деления на ноль или нестабильной нормализации.

5. **Изменили лишнюю логику**  
   Не трогайте фильтрацию/эпохи/базовую коррекцию, если задача только в выборе победителя.

---

## 10. Быстрый smoke-check после интеграции

1. Запустите `scripts/p300_analyzer.py`.
2. Убедитесь, что новый режим виден в UI-комбо выбора победителя.
3. Выберите новый режим и дайте системе накопить минимум эпох.
4. Проверьте, что:
   - нет ошибок в логах;
   - победитель отображается;
   - debug-метрики выглядят разумно;
   - margin не всегда нулевой/аномальный.

---

## 11. Готовый "копипаст"-шаблон для разработчика

Ниже шаблон, который можно дать человеку: он заполняет только формулу в одном месте.

```python
# 1) winner_selection.py
WINNER_MODE_MY_METHOD = "my_method"

MODE_SHORT_LABELS = {
    WINNER_MODE_AUC: "auc",
    WINNER_MODE_SIGNED_MEAN: "signed_mean",
    WINNER_MODE_MY_METHOD: "my_method",
}
```

```python
# 2) qt_window.py (в setup UI)
self.combo_winner_mode.addItem("Мой метод", WINNER_MODE_MY_METHOD)
```

```python
# 3) erp_compute.py
def compute_custom_winner_metric(
    *,
    corr_win: np.ndarray,
    abs_auc_values: np.ndarray,
    signed_mean_values: np.ndarray,
    dt_m: float,
) -> np.ndarray:
    # TODO: подставить формулу
    return abs_auc_values


# Внутри compute_winner_metrics(...)
if winner_mode == WINNER_MODE_SIGNED_MEAN:
    final_metric_values = signed_mean_values
elif winner_mode == WINNER_MODE_MY_METHOD:
    final_metric_values = compute_custom_winner_metric(
        corr_win=corr_win,
        abs_auc_values=abs_auc_values,
        signed_mean_values=signed_mean_values,
        dt_m=dt_m,
    )
else:
    final_metric_values = abs_auc_values
```

---

## 12. Рекомендация для дальнейшего улучшения

Если планируется много новых методов от разных людей, стоит перейти на плагинную регистрацию стратегий (отдельная папка стратегий + авто-реестр).  
Но для текущей архитектуры и минимальных правок инструкция выше — наиболее практичная.

