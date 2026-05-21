"""Построение и управление P300-шаблонами (эталонами) для классификации."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from p300_analysis.calibration import load_examples_from_paths
from p300_analysis.erp_compute import build_averaged_erp
from p300_analysis.signal_processing import baseline_correction


def build_p300_template(
    calibration_files: List[Path],
    *,
    baseline_ms: int = 100,
    artifact_uv: float = 60.0,
    use_car: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Построить P300-эталон из нескольких калибровочных сессий.

    Накапливает эпохи со всех сессий по каждому стимулу и вычисляет
    их среднее значение. Это среднее значение служит эталоном для CCA.

    Args:
        calibration_files: список путей к CSV файлам с калибровочными данными
        baseline_ms: длина baseline окна в миллисекундах
        artifact_uv: порог отклонения артефактов в микровольтах
        use_car: применить общую ссылку на усредненный канал

    Returns:
        (template_array, time_ms, all_stim_templates)
        - template_array: среднее P300 (усреднено по всем стимулам) форма (epoch_len,)
        - time_ms: временная шкала в миллисекундах форма (epoch_len,)
        - all_stim_templates: словарь {stim_key: эталон} для каждого стимула отдельно
    """
    if not calibration_files:
        raise ValueError("calibration_files не может быть пустой")

    examples = load_examples_from_paths(
        calibration_files,
        baseline_ms=baseline_ms,
        artifact_uv=artifact_uv,
        use_car=use_car,
    )
    if not examples:
        raise RuntimeError("Не удалось загрузить примеры калибровки")

    if len(examples) < 5:
        raise ValueError(f"Нужно минимум 5 сессий для построения эталона, получено {len(examples)}")

    # Накопить эпохи со всех примеров
    all_epochs: Dict[str, List[np.ndarray]] = {}

    for ex in examples:
        for stim_key, epochs_tuple in ex.epochs_data.items():
            if stim_key not in all_epochs:
                all_epochs[stim_key] = []
            all_epochs[stim_key].extend(list(epochs_tuple))

    if not all_epochs:
        raise RuntimeError("Нет накопленных эпох")

    # Усредните эпохи для каждого стимула
    epoch_len = examples[0].time_ms.shape[0]
    stim_keys = sorted(all_epochs.keys())
    stim_templates: Dict[str, np.ndarray] = {}
    averaged_erps = []

    for stim_key in stim_keys:
        epochs = all_epochs[stim_key]
        # Обработка эпох: если они 2D (epoch_len, n_ch), усредняем по каналам; если 1D, просто усредняем
        if epochs and epochs[0].ndim == 2:
            # Каждая эпоха - (epoch_len, n_ch)
            stacked = np.stack(epochs, axis=0)  # (n_epochs, epoch_len, n_ch)
            averaged = np.mean(stacked, axis=(0, 2))  # Средняя по эпохам и каналам -> (epoch_len,)
        else:
            # 1D эпохи
            stacked = np.stack(epochs, axis=0)  # (n_epochs, epoch_len)
            averaged = np.mean(stacked, axis=0)  # (epoch_len,)

        stim_templates[stim_key] = averaged
        averaged_erps.append(averaged)

    # Вычислить общий эталон P300 (среднее по всем стимулам)
    overall_template = np.mean(np.stack(averaged_erps, axis=0), axis=0)

    # Применить базовую коррекцию
    time_ms = examples[0].time_ms
    corrected = baseline_correction(
        overall_template.reshape(1, -1),
        time_ms,
        baseline_ms=baseline_ms,
    )
    corrected_overall = corrected[0, :]

    # Коррекция для каждого стимула
    corrected_stim_templates = {}
    for stim_key, template in stim_templates.items():
        corrected_single = baseline_correction(
            template.reshape(1, -1),
            time_ms,
            baseline_ms=baseline_ms,
        )
        corrected_stim_templates[stim_key] = corrected_single[0, :]

    return corrected_overall, time_ms, corrected_stim_templates


def save_p300_template(
    template: np.ndarray,
    time_ms: np.ndarray,
    stim_templates: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Сохранить P300-эталон в файл.

    Args:
        template: главный эталон, форма (epoch_len,)
        time_ms: временная шкала, форма (epoch_len,)
        stim_templates: словарь {stim_key: template} для каждого стимула
        output_path: путь для сохранения (по умолчанию .npz формат)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохранить в NPZ (удобно для numpy массивов)
    save_dict = {
        "template": template,
        "time_ms": time_ms,
    }
    for stim_key, stim_template in stim_templates.items():
        # Заменить спецсимволы на безопасные имена
        safe_key = f"stim_{stim_key.replace(' ', '_').replace('_', 'X')}"
        save_dict[safe_key] = stim_template

    np.savez_compressed(str(output_path), **save_dict)

    # Также сохранить метаинформацию в JSON для лучшей читаемости
    meta_path = output_path.with_suffix('.json')
    meta = {
        "template_shape": template.shape,
        "time_ms_shape": time_ms.shape,
        "stim_keys": list(stim_templates.keys()),
        "source": "build_p300_template",
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_p300_template(
    input_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Загрузить P300-эталон из файла.

    Args:
        input_path: путь к сохраненному файлу (.npz)

    Returns:
        (template, time_ms, stim_templates)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Template file not found: {input_path}")

    data = np.load(str(input_path))
    template = np.array(data['template'], dtype=np.float64)
    time_ms = np.array(data['time_ms'], dtype=np.float64)

    stim_templates: Dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith('stim_'):
            # Восстановить оригинальное имя стимула (обратная замена)
            stim_key = key[5:].replace('X', '_')
            stim_templates[stim_key] = np.array(data[key], dtype=np.float64)

    return template, time_ms, stim_templates


def get_best_matching_template(
    stim_templates: Dict[str, np.ndarray],
    window_start_ms: float,
    window_end_ms: float,
    time_ms: np.ndarray,
) -> np.ndarray:
    """Выбрать лучший эталон в указанном временном окне.

    Если есть стимул-специфичный шаблон в этом окне, используем его.
    Иначе, усредняем все доступные шаблоны.

    Args:
        stim_templates: словарь {stim_key: template}
        window_start_ms: начало окна в миллисекундах
        window_end_ms: конец окна в миллисекундах
        time_ms: временная шкала

    Returns:
        лучший эталон для этого окна, форма (epoch_len,)
    """
    if not stim_templates:
        # Возвращать пустой массив, если нет шаблонов
        return np.array([], dtype=np.float64)

    # Если есть несколько шаблонов, усредняем их
    templates_list = list(stim_templates.values())
    if len(templates_list) == 1:
        return templates_list[0]
    else:
        # Усредняем все доступные шаблоны
        return np.mean(np.stack(templates_list, axis=0), axis=0)
