"""Константы и подписи выбора класса-победителя по ERP."""

from __future__ import annotations

WINNER_MODE_AUC = "auc"
WINNER_MODE_SIGNED_MEAN = "signed_mean"
WINNER_MODE_MSI = "msi"

MODE_SHORT_LABELS = {
    WINNER_MODE_AUC: "auc",
    WINNER_MODE_SIGNED_MEAN: "signed_mean",
    WINNER_MODE_MSI: "msi",
}


def mode_to_short_label(mode_used: str) -> str:
    return MODE_SHORT_LABELS.get(mode_used, mode_used)
