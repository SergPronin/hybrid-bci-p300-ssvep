from __future__ import annotations

from pathlib import Path

from experiment_protocol import stim_control as sc


def test_stim_control_roundtrip(tmp_path: Path) -> None:
    sc.write_paused(tmp_path, reason="test", message="wait")
    cmd = sc.read_control(tmp_path)
    assert cmd is not None
    assert cmd["state"] == "paused"

    sc.write_trial_request(
        tmp_path,
        target_tile_id=5,
        experiment_index=3,
        experiment_total=45,
    )
    cmd2 = sc.read_control(tmp_path)
    assert cmd2 is not None
    assert cmd2["state"] == "trial"
    assert int(cmd2["target_tile_id"]) == 5
