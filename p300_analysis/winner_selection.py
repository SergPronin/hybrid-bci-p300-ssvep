"""Константы и подписи выбора класса-победителя по ERP."""

from __future__ import annotations

WINNER_MODE_AUC = "auc"

MODE_SHORT_LABELS = {
    WINNER_MODE_AUC: "auc",
}


def mode_to_short_label(mode_used: str) -> str:
    return MODE_SHORT_LABELS.get(mode_used, mode_used)
