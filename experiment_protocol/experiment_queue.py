"""План основного блока: три цельных сегмента, случайный только порядок сегментов."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Идентификаторы макро-блоков (порядок из 6 перестановок задаётся shuffle_seed)
BLOCK_P300 = "p300"
BLOCK_SSVEP_CONT = "ssvep_continuous"
BLOCK_SSVEP_BURST = "ssvep_burst"
BLOCK_ORDER_LABELS_RU: Dict[str, str] = {
    BLOCK_P300: "P300 (калибровка + 15 main)",
    BLOCK_SSVEP_CONT: "ССВП непрерывный (15)",
    BLOCK_SSVEP_BURST: "ССВП пакетный (15)",
}


@dataclass(frozen=True)
class QueueItem:
    kind: str  # "p300" | "ssvep"
    ssvep_mode: Optional[str] = None  # "continuous" | "burst"
    target_lamp_0idx: Optional[int] = None  # для SSVEP
    target_tile_id: Optional[int] = None  # для P300 (0..8), назначается при сборке очереди
    p300_phase: Optional[str] = None  # "calib" | "main" — только для kind=p300
    p300_block_index: Optional[int] = None  # 1..N внутри блока P300 (калибровка + main подряд)
    p300_block_total: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "ssvep_mode": self.ssvep_mode,
            "target_lamp_0idx": self.target_lamp_0idx,
            "target_tile_id": self.target_tile_id,
            "p300_phase": self.p300_phase,
            "p300_block_index": self.p300_block_index,
            "p300_block_total": self.p300_block_total,
        }


def _balanced_lamp_indices(n_items: int, n_lamps: int) -> List[int]:
    """Распределить n_items по лампам 0..n_lamps-1 максимально равномерно."""
    n_lamps = max(1, int(n_lamps))
    n_items = max(0, int(n_items))
    per = n_items // n_lamps
    rem = n_items % n_lamps
    out: List[int] = []
    for lamp in range(n_lamps):
        out.extend([lamp] * (per + (1 if lamp < rem else 0)))
    return out


def _assign_p300_tile_targets(items: List[QueueItem], *, n_tiles: int, rng: random.Random) -> List[QueueItem]:
    n_tiles = max(1, int(n_tiles))
    p300_idx = [i for i, x in enumerate(items) if x.kind == "p300"]
    tiles = _balanced_lamp_indices(len(p300_idx), n_tiles)
    rng.shuffle(tiles)
    out = list(items)
    for i, tile in zip(p300_idx, tiles):
        old = out[i]
        out[i] = QueueItem(
            kind="p300",
            target_tile_id=int(tile),
            p300_phase=old.p300_phase,
            p300_block_index=old.p300_block_index,
            p300_block_total=old.p300_block_total,
        )
    return out


def format_block_order_ru(block_order: Sequence[str]) -> str:
    return " → ".join(BLOCK_ORDER_LABELS_RU.get(str(b), str(b)) for b in block_order)


def _build_p300_block_items(
    *,
    p300_calib_trials: int,
    calib_target_tile_id: int,
    p300_main_trials: int,
    n_p300_tiles: int,
    rng: random.Random,
) -> List[QueueItem]:
    """Калибровка + main подряд в одном блоке P300."""
    items: List[QueueItem] = []
    for _ in range(int(p300_calib_trials)):
        items.append(
            QueueItem(
                kind="p300",
                p300_phase="calib",
                target_tile_id=int(calib_target_tile_id),
            )
        )
    main_items: List[QueueItem] = [QueueItem(kind="p300", p300_phase="main") for _ in range(int(p300_main_trials))]
    main_items = _assign_p300_tile_targets(main_items, n_tiles=int(n_p300_tiles), rng=rng)
    items.extend(main_items)
    total = len(items)
    out: List[QueueItem] = []
    for i, it in enumerate(items):
        out.append(
            QueueItem(
                kind=it.kind,
                ssvep_mode=it.ssvep_mode,
                target_lamp_0idx=it.target_lamp_0idx,
                target_tile_id=it.target_tile_id,
                p300_phase=it.p300_phase,
                p300_block_index=int(i) + 1,
                p300_block_total=int(total),
            )
        )
    return out


def build_main_queue(
    *,
    p300_calib_trials: int,
    calib_target_tile_id: int,
    p300_trials: int,
    ssvep_continuous: int,
    ssvep_burst: int,
    n_active_lamps: int,
    n_p300_tiles: int = 9,
    shuffle_seed: Optional[int] = None,
) -> Tuple[List[QueueItem], int, List[str]]:
    """
    Собрать очередь main: три цельных блока подряд, перемешан только их порядок.

    Блок P300 = калибровка (N) + 15 main подряд, без разрыва SSVEP между ними.

    Returns (items, seed_used, block_order).
    """
    seed_used = int(shuffle_seed) if shuffle_seed is not None else random.randrange(1 << 31)
    rng = random.Random(seed_used)

    p300_items = _build_p300_block_items(
        p300_calib_trials=int(p300_calib_trials),
        calib_target_tile_id=int(calib_target_tile_id),
        p300_main_trials=int(p300_trials),
        n_p300_tiles=int(n_p300_tiles),
        rng=rng,
    )

    cont_lamps = _balanced_lamp_indices(int(ssvep_continuous), int(n_active_lamps))
    cont_items = [
        QueueItem(kind="ssvep", ssvep_mode="continuous", target_lamp_0idx=int(lamp))
        for lamp in cont_lamps
    ]
    burst_lamps = _balanced_lamp_indices(int(ssvep_burst), int(n_active_lamps))
    burst_items = [
        QueueItem(kind="ssvep", ssvep_mode="burst", target_lamp_0idx=int(lamp)) for lamp in burst_lamps
    ]

    segments: Dict[str, List[QueueItem]] = {
        BLOCK_P300: p300_items,
        BLOCK_SSVEP_CONT: cont_items,
        BLOCK_SSVEP_BURST: burst_items,
    }
    block_order = [BLOCK_P300, BLOCK_SSVEP_CONT, BLOCK_SSVEP_BURST]
    rng.shuffle(block_order)

    items: List[QueueItem] = []
    for block in block_order:
        items.extend(segments[block])
    return items, seed_used, list(block_order)


def queue_summary(items: Sequence[QueueItem]) -> Dict[str, int]:
    n_p300 = sum(1 for x in items if x.kind == "p300")
    n_calib = sum(1 for x in items if x.kind == "p300" and x.p300_phase == "calib")
    n_p300_main = sum(1 for x in items if x.kind == "p300" and x.p300_phase == "main")
    n_cont = sum(1 for x in items if x.kind == "ssvep" and x.ssvep_mode == "continuous")
    n_burst = sum(1 for x in items if x.kind == "ssvep" and x.ssvep_mode == "burst")
    return {
        "p300": n_p300,
        "p300_calib": n_calib,
        "p300_main": n_p300_main,
        "ssvep_continuous": n_cont,
        "ssvep_burst": n_burst,
        "total": len(items),
    }
