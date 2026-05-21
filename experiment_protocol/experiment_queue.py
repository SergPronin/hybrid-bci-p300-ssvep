"""План основного блока: 15 P300 + 15 SSVEP continuous + 15 SSVEP burst (перемешано)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class QueueItem:
    kind: str  # "p300" | "ssvep"
    ssvep_mode: Optional[str] = None  # "continuous" | "burst"
    target_lamp_0idx: Optional[int] = None  # для SSVEP
    target_tile_id: Optional[int] = None  # для P300 (0..8), назначается при сборке очереди
    phase: str = "main"  # "calib" | "main"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "ssvep_mode": self.ssvep_mode,
            "target_lamp_0idx": self.target_lamp_0idx,
            "target_tile_id": self.target_tile_id,
            "phase": self.phase,
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
        )
    return out


def build_main_queue(
    *,
    p300_trials: int,
    ssvep_continuous: int,
    ssvep_burst: int,
    n_active_lamps: int,
    n_p300_tiles: int = 9,
    shuffle_seed: Optional[int] = None,
) -> Tuple[List[QueueItem], int]:
    """
    Собрать и перемешать очередь основного блока.

    Returns (items, seed_used).
    """
    items: List[QueueItem] = []
    for _ in range(int(p300_trials)):
        items.append(QueueItem(kind="p300"))
    cont_lamps = _balanced_lamp_indices(int(ssvep_continuous), int(n_active_lamps))
    for lamp in cont_lamps:
        items.append(QueueItem(kind="ssvep", ssvep_mode="continuous", target_lamp_0idx=int(lamp)))
    burst_lamps = _balanced_lamp_indices(int(ssvep_burst), int(n_active_lamps))
    for lamp in burst_lamps:
        items.append(QueueItem(kind="ssvep", ssvep_mode="burst", target_lamp_0idx=int(lamp)))

    seed_used = int(shuffle_seed) if shuffle_seed is not None else random.randrange(1 << 31)
    rng = random.Random(seed_used)
    rng.shuffle(items)
    items = _assign_p300_tile_targets(items, n_tiles=int(n_p300_tiles), rng=rng)
    return items, seed_used


def queue_summary(items: Sequence[QueueItem]) -> Dict[str, int]:
    n_p300 = sum(1 for x in items if x.kind == "p300")
    n_cont = sum(1 for x in items if x.kind == "ssvep" and x.ssvep_mode == "continuous")
    n_burst = sum(1 for x in items if x.kind == "ssvep" and x.ssvep_mode == "burst")
    return {"p300": n_p300, "ssvep_continuous": n_cont, "ssvep_burst": n_burst, "total": len(items)}
