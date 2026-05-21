from __future__ import annotations

from experiment_protocol.experiment_queue import build_main_queue, queue_summary


def test_build_main_queue_counts_and_shuffle_reproducible() -> None:
    items, seed = build_main_queue(
        p300_trials=15,
        ssvep_continuous=15,
        ssvep_burst=15,
        n_active_lamps=4,
        shuffle_seed=42,
    )
    assert len(items) == 45
    s = queue_summary(items)
    assert s["p300"] == 15
    assert s["ssvep_continuous"] == 15
    assert s["ssvep_burst"] == 15

    items2, seed2 = build_main_queue(
        p300_trials=15,
        ssvep_continuous=15,
        ssvep_burst=15,
        n_active_lamps=4,
        shuffle_seed=42,
    )
    assert seed == seed2 == 42
    assert [x.to_dict() for x in items] == [x.to_dict() for x in items2]
    for x in items:
        if x.kind == "p300":
            assert x.target_tile_id is not None
            assert 0 <= int(x.target_tile_id) < 9
