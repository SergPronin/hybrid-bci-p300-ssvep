"""Модель плитки стимульной сетки."""

from dataclasses import dataclass


@dataclass
class Tile:
    """
    Одна ячейка (плитка) в сетке стимулов.

    Attributes:
        id: Уникальный идентификатор плитки (0 .. size²-1).
        row: Номер строки в сетке (0 — верх).
        col: Номер столбца в сетке (0 — левый).
        active: Признак активной подсветки (мигания).
    """

    id: int
    row: int
    col: int
    active: bool = False
