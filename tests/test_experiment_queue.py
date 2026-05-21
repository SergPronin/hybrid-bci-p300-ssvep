from __future__ import annotations

from experiment_protocol.experiment_queue import (
    BLOCK_SSVEP_BURST,
    BLOCK_SSVEP_CONT,
    build_main_queue,
    format_block_order_ru,
    queue_summary,
)


def _segment_kinds(items) -> list[str]:
    kinds: list[str] = []
    for x in items:
        k = "p300" if x.kind == "p300" else str(x.ssvep_mode)
        if not kinds or kinds[-1] != k:
            kinds.append(k)
    return kinds


def test_build_main_queue_counts_and_shuffle_reproducible() -> None:
    items, seed, order = build_main_queue(
        p300_calib_trials=5,
        calib_target_tile_id=4,
        p300_trials=15,
        ssvep_continuous=15,
        ssvep_burst=15,
        n_active_lamps=4,
        shuffle_seed=42,
    )
    assert len(items) == 50
    assert len(order) == 3
    s = queue_summary(items)
    assert s["p300"] == 20
    assert s["p300_calib"] == 5
    assert s["p300_main"] == 15
    assert s["ssvep_continuous"] == 15
    assert s["ssvep_burst"] == 15

    items2, seed2, order2 = build_main_queue(
        p300_calib_trials=5,
        calib_target_tile_id=4,
        p300_trials=15,
        ssvep_continuous=15,
        ssvep_burst=15,
        n_active_lamps=4,
        shuffle_seed=42,
    )
    assert seed == seed2 == 42
    assert order == order2
    assert [x.to_dict() for x in items] == [x.to_dict() for x in items2]
    calib = [x for x in items if x.p300_phase == "calib"]
    assert len(calib) == 5
    assert all(int(x.target_tile_id) == 4 for x in calib)
    for x in items:
        if x.kind == "p300" and x.p300_phase == "main":
            assert x.target_tile_id is not None
            assert 0 <= int(x.target_tile_id) < 9


def test_p300_calib_then_main_contiguous() -> None:
    items, _seed, order = build_main_queue(
        p300_calib_trials=3,
        calib_target_tile_id=2,
        p300_trials=4,
        ssvep_continuous=1,
        ssvep_burst=1,
        n_active_lamps=4,
        shuffle_seed=0,
    )
    p300_seg = [x for x in items if x.kind == "p300"]
    assert len(p300_seg) == 7
    assert [x.p300_phase for x in p300_seg[:3]] == ["calib", "calib", "calib"]
    assert [x.p300_phase for x in p300_seg[3:]] == ["main", "main", "main", "main"]
    assert p300_seg[0].p300_block_index == 1
    assert p300_seg[-1].p300_block_index == 7
    assert p300_seg[-1].p300_block_total == 7
    # макро-блок P300 — один непрерывный сегмент в очереди
    kinds = _segment_kinds(items)
    assert kinds.count("p300") == 1


def test_three_contiguous_blocks_only() -> None:
    items, _seed, order = build_main_queue(
        p300_calib_trials=0,
        calib_target_tile_id=4,
        p300_trials=3,
        ssvep_continuous=2,
        ssvep_burst=2,
        n_active_lamps=4,
        shuffle_seed=1,
    )
    kinds = _segment_kinds(items)
    assert len(kinds) == 3
    assert set(kinds) == {"p300", "continuous", "burst"}
    mode_map = {
        "p300": "p300",
        "ssvep_continuous": "continuous",
        "ssvep_burst": "burst",
    }
    assert kinds == [mode_map[b] for b in order]


def test_block_order_has_six_permutations_over_many_seeds() -> None:
    seen = set()
    for seed in range(120):
        _items, _s, order = build_main_queue(
            p300_calib_trials=1,
            calib_target_tile_id=4,
            p300_trials=1,
            ssvep_continuous=1,
            ssvep_burst=1,
            n_active_lamps=4,
            shuffle_seed=seed,
        )
        seen.add(tuple(order))
    assert len(seen) == 6


def test_format_block_order_ru() -> None:
    text = format_block_order_ru([BLOCK_SSVEP_CONT, BLOCK_SSVEP_BURST, "p300"])
    assert "ССВП непрерывный" in text
    assert "→" in text
