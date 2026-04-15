"""Loader tests for the run_id / run_index assignment logic."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from mariha.curriculum.loader import load_curriculum


def _write_run_tsv(
    func_dir: Path,
    subject: str,
    session: str,
    run_number: int,
    rows: list[dict],
) -> None:
    """Write a fake events TSV for one BIDS run."""
    func_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{subject}_{session}_task-mario_run-{run_number:02d}"
        "_desc-scenes_events.tsv"
    )
    df = pd.DataFrame(rows)
    df.to_csv(func_dir / name, sep="\t", index=False)


def _scene_row(
    clip_code: str,
    scene_id: str,
    stim_file: str,
    *,
    level: str = "w1l1",
    phase: str = "discovery",
    frame_start: int = 0,
    frame_stop: int = 100,
) -> dict:
    return {
        "trial_type": "scene",
        "clip_code": clip_code,
        "scene_id": scene_id,
        "level": level,
        "phase": phase,
        "outcome": "completed",
        "frame_start": frame_start,
        "frame_stop": frame_stop,
        "stim_file": stim_file,
    }


def test_loader_assigns_run_id_and_run_index():
    """3 runs × 2 clips each → run_index assigned in chronological order."""
    with tempfile.TemporaryDirectory() as tmp:
        scenes_root = Path(tmp)
        subject = "sub-01"
        ses = "ses-001"
        func_dir = scenes_root / subject / ses / "func"

        # Three runs; clip_codes increase monotonically across runs so the
        # global sort interleaves nothing — runs stay contiguous.
        _write_run_tsv(func_dir, subject, ses, 1, [
            _scene_row("00000000000001", "w1l1s0", f"{subject}/{ses}/g/c1.bk2"),
            _scene_row("00000000000002", "w1l1s1", f"{subject}/{ses}/g/c2.bk2"),
        ])
        _write_run_tsv(func_dir, subject, ses, 2, [
            _scene_row("00000000000003", "w1l1s0", f"{subject}/{ses}/g/c3.bk2"),
            _scene_row("00000000000004", "w1l2s0", f"{subject}/{ses}/g/c4.bk2"),
        ])
        _write_run_tsv(func_dir, subject, ses, 3, [
            _scene_row("00000000000005", "w1l1s1", f"{subject}/{ses}/g/c5.bk2"),
            _scene_row("00000000000006", "w1l2s0", f"{subject}/{ses}/g/c6.bk2"),
        ])

        specs = load_curriculum(
            subject_id=subject,
            scenes_root=scenes_root,
            require_existing_states=False,
        )

    assert len(specs) == 6

    # Three distinct run_ids, assigned indices 0,1,2 in encounter order.
    run_ids_in_order = []
    seen = set()
    for s in specs:
        if s.run_id not in seen:
            run_ids_in_order.append(s.run_id)
            seen.add(s.run_id)
    assert run_ids_in_order == [
        "ses-001_run-01",
        "ses-001_run-02",
        "ses-001_run-03",
    ]

    # run_index matches encounter position for every spec.
    expected_index = {
        "ses-001_run-01": 0,
        "ses-001_run-02": 1,
        "ses-001_run-03": 2,
    }
    for spec in specs:
        assert spec.run_index == expected_index[spec.run_id]

    # Clips sharing a run_id are contiguous in the sorted output.
    last_run = None
    seen_runs: set[str] = set()
    for spec in specs:
        if spec.run_id != last_run:
            assert spec.run_id not in seen_runs, (
                f"run_id {spec.run_id} reappeared non-contiguously"
            )
            seen_runs.add(spec.run_id)
            last_run = spec.run_id


def test_loader_rejects_non_contiguous_runs():
    """If clip_codes are interleaved across runs, the loader must raise."""
    with tempfile.TemporaryDirectory() as tmp:
        scenes_root = Path(tmp)
        subject = "sub-99"
        ses = "ses-001"
        func_dir = scenes_root / subject / ses / "func"

        # Run 1 has clip_codes 1 and 3; run 2 has clip_code 2 — interleaved.
        _write_run_tsv(func_dir, subject, ses, 1, [
            _scene_row("00000000000001", "w1l1s0", f"{subject}/{ses}/g/a.bk2"),
            _scene_row("00000000000003", "w1l1s1", f"{subject}/{ses}/g/c.bk2"),
        ])
        _write_run_tsv(func_dir, subject, ses, 2, [
            _scene_row("00000000000002", "w1l1s0", f"{subject}/{ses}/g/b.bk2"),
        ])

        with pytest.raises(ValueError, match="Non-contiguous run_id"):
            load_curriculum(
                subject_id=subject,
                scenes_root=scenes_root,
                require_existing_states=False,
            )
