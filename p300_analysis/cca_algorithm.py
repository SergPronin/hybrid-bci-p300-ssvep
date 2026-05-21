"""Canonical Correlation Analysis (CCA) для выбора плитки на основе P300."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import eigh


def compute_cca_scores(
    *,
    corr_win: np.ndarray,
    abs_auc_values: np.ndarray,
    p300_template: Optional[np.ndarray] = None,
    n_components: int = 1,
) -> np.ndarray:
    """Canonical Correlation Analysis (CCA) для оценки соответствия P300-шаблону.

    Использует CCA для вычисления корреляции между каждым стимулом (строка corr_win)
    и P300-шаблоном (либо загруженным, либо синтетическим).

    Args:
        corr_win: базально-скорректированный сигнал, форма (n_stim, win_len)
        abs_auc_values: энергия окна для каждого стимула (не используется для вычисления,
                       но возвращается при пустом corr_win)
        p300_template: загруженный P300-эталон, форма (epoch_len,), или None для синтетического
        n_components: количество CCA компонент для использования (обычно 1)

    Returns:
        вектор формы (n_stim,), где большее значение означает лучший кандидат (P300-подобный ответ)
    """
    n_stim = abs_auc_values.shape[0]
    if n_stim == 0:
        return np.zeros(0, dtype=np.float64)
    if corr_win.size == 0 or corr_win.shape[1] < 2:
        return abs_auc_values.astype(np.float64, copy=True)

    win_len = corr_win.shape[1]

    # Выбрать шаблон: загруженный или синтетический
    if p300_template is not None and len(p300_template) > 0:
        # Использовать загруженный эталон
        template = p300_template.astype(np.float64)
        if len(template) != win_len:
            # Интерполировать на нужную длину если не совпадает
            template = np.interp(
                np.linspace(0, 1, win_len),
                np.linspace(0, 1, len(template)),
                template
            )
        template -= np.mean(template)
    else:
        # Синтетический P300-подобный шаблон: гауссиан с центром в середине окна
        t = np.linspace(-1.0, 1.0, win_len, dtype=np.float64)
        template = np.exp(-0.5 * (t / 0.35) ** 2)
        template -= np.mean(template)

    template = template.reshape(-1, 1)  # (win_len, 1)

    # Стандартизация каждого стимула по каналам
    X = corr_win.astype(np.float64)  # (n_stim, win_len)
    Y = template.astype(np.float64)  # (win_len, 1)

    # Каждая эпоха (строка X) будет коррелирована с Y
    scores = np.zeros(n_stim, dtype=np.float64)

    for i in range(n_stim):
        x_row = X[i : i + 1, :].T  # (win_len, 1) - один стимул
        try:
            corr = _cca_correlation(x_row, Y, n_components=n_components)
            scores[i] = float(corr)
        except (np.linalg.LinAlgError, ValueError):
            # Если CCA не может быть вычислена, используем простую корреляцию Пирсона
            x_norm = x_row.flatten() - np.mean(x_row)
            y_norm = Y.flatten() - np.mean(Y)
            denom = np.linalg.norm(x_norm) * np.linalg.norm(y_norm)
            if denom > 1e-12:
                corr = float(np.abs(np.dot(x_norm, y_norm) / denom))
            else:
                corr = 0.0
            scores[i] = corr

    # Защита от NaN/Inf
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


def _cca_correlation(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 1,
) -> float:
    """Вычисляет первую каноническую корреляцию между X и Y.

    Args:
        X: матрица (n_samples, n_features_x)
        Y: матрица (n_samples, n_features_y)
        n_components: количество компонент (обычно 1)

    Returns:
        первая каноническая корреляция (скаляр в диапазоне [0, 1])
    """
    # Стандартизация
    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = Y.mean(axis=0, keepdims=True)
    X_centered = X - x_mean
    Y_centered = Y - y_mean

    x_std = X_centered.std(axis=0, ddof=1)
    y_std = Y_centered.std(axis=0, ddof=1)

    # Избегаем деления на ноль
    x_std[x_std < 1e-12] = 1.0
    y_std[y_std < 1e-12] = 1.0

    X_std = X_centered / x_std
    Y_std = Y_centered / y_std

    n = X_std.shape[0]
    p = X_std.shape[1]
    q = Y_std.shape[1]

    n_components = min(n_components, p, q, n - 1)
    if n_components < 1:
        return 0.0

    # Ковариационные матрицы
    Sxx = (X_std.T @ X_std) / (n - 1)
    Syy = (Y_std.T @ Y_std) / (n - 1)
    Sxy = (X_std.T @ Y_std) / (n - 1)

    # Регуляризация для стабильности
    reg = 1e-8
    Sxx_reg = Sxx + reg * np.eye(p)
    Syy_reg = Syy + reg * np.eye(q)

    # Обобщенное собственное разложение
    Syy_inv = np.linalg.pinv(Syy_reg, rcond=1e-10)
    Cxx = Sxy @ Syy_inv @ Sxy.T
    Sxx_inv = np.linalg.pinv(Sxx_reg, rcond=1e-10)

    eigvals, eigvecs = eigh(Cxx, Sxx_reg)

    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]

    correlation = np.sqrt(np.clip(eigvals[0], 0.0, 1.0))
    return float(correlation)
