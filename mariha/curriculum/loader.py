"""Curriculum loader: parse human gameplay data into an ordered episode list.

Reads BIDS-structured events TSV files from ``mario.scenes`` for a given
subject and returns an ordered list of ``EpisodeSpec`` objects sorted by
``clip_code`` (ascending ordinal), which reflects the temporal order in
which the human encountered each scene.

Usage::

    from mariha.curriculum.loader import load_curriculum

    specs = load_curriculum(subject_id="sub-01")
    print(f"{len(specs)} episodes, {len({s.scene_id for s in specs})} unique scenes")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from mariha.curriculum.episode import EpisodeSpec

logger = logging.getLogger(__name__)

_RUN_RE = re.compile(r"_run-(\d+)_")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]  # MariHA/
_SCENES_ROOT = _REPO_ROOT / "data" / "mario.scenes"

# Columns kept as metadata (everything not promoted to a named field).
_METADATA_COLS = [
    "ScoreGained",
    "CoinsGained",
    "Lives_lost",
    "Hits_taken",
    "Enemies_killed",
    "Powerups_collected",
    "Bricks_smashed",
    "X_Traveled",
    "rep_index",
    "onset",
    "duration",
    "frame_start",
    "frame_stop",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_curriculum(
    subject_id: str,
    scenes_root: Path = _SCENES_ROOT,
    require_existing_states: bool = True,
) -> list[EpisodeSpec]:
    """Load the ordered episode list for a subject.

    Scans all ``sub-{id}/ses-*/func/*_desc-scenes_events.tsv`` files,
    filters for scene-level rows (``trial_type == 'scene'``), resolves
    the ``.state`` file path for each clip, and returns all episodes sorted
    by ``clip_code`` (ascending).

    Args:
        subject_id: Subject identifier, e.g. ``'sub-01'``.
        scenes_root: Root of the ``mario.scenes`` dataset.
        require_existing_states: If ``True`` (default), skip clips whose
            ``.state`` file does not exist on disk and log a warning.
            Set to ``False`` to include all clips regardless.

    Returns:
        List of ``EpisodeSpec`` objects in chronological (clip_code) order.

    Raises:
        FileNotFoundError: If no events TSV files are found for the subject.
    """
    subject_dir = scenes_root / subject_id
    if not subject_dir.exists():
        raise FileNotFoundError(
            f"Subject directory not found: {subject_dir}"
        )

    tsv_files = sorted(subject_dir.glob("ses-*/func/*_desc-scenes_events.tsv"))
    if not tsv_files:
        raise FileNotFoundError(
            f"No events TSV files found for {subject_id} in {subject_dir}"
        )

    rows: list[pd.Series] = []
    for tsv_path in tsv_files:
        session = tsv_path.parts[-3]  # ses-XXX
        m = _RUN_RE.search(tsv_path.name)
        if m is None:
            raise ValueError(
                f"Cannot parse run number from TSV filename: {tsv_path.name}"
            )
        run_number = int(m.group(1))

        # Read clip_code as string to preserve leading zeros (14-char ordinal key).
        df = pd.read_csv(tsv_path, sep="\t", dtype={"clip_code": str})

        # Keep only scene-level rows.
        scene_rows = df[df["trial_type"] == "scene"].copy()
        scene_rows["_session"] = session
        scene_rows["_run_number"] = run_number
        rows.append(scene_rows)

    all_scenes = pd.concat(rows, ignore_index=True)

    if all_scenes.empty:
        logger.warning("No scene rows found for %s.", subject_id)
        return []

    specs: list[EpisodeSpec] = []
    skipped = 0

    for _, row in all_scenes.iterrows():
        # Zero-pad to 14 characters to restore the original ordinal sort key
        # (pandas reads the numeric clip_code as float, stripping leading zeros).
        raw_code = row["clip_code"]
        if pd.isna(raw_code) or str(raw_code).lower() == "nan":
            continue
        clip_code = str(raw_code).zfill(14)
        session = str(row["_session"])
        run_number = int(row["_run_number"])
        run_id = f"{session}_run-{run_number:02d}"
        scene_id = str(row["scene_id"])
        level = str(row.get("level", ""))
        outcome = str(row.get("outcome", "failed"))
        phase = str(row.get("phase", ""))
        frame_start = int(row["frame_start"])
        frame_stop = int(row["frame_stop"])
        max_steps = max(1, frame_stop - frame_start)

        # Resolve the .state file path from the stim_file column.
        state_file = _resolve_state_file(
            scenes_root=scenes_root,
            subject_id=subject_id,
            session=session,
            stim_file=str(row.get("stim_file", "")),
            clip_code=clip_code,
            level=level,
            scene_id=scene_id,
        )

        if state_file is None:
            skipped += 1
            continue

        if require_existing_states and not state_file.exists():
            logger.warning(
                "State file not found, skipping clip %s: %s",
                clip_code,
                state_file,
            )
            skipped += 1
            continue

        metadata = {
            col: row.get(col)
            for col in _METADATA_COLS
            if col in row.index
        }

        specs.append(
            EpisodeSpec(
                state_file=state_file,
                max_steps=max_steps,
                scene_id=scene_id,
                clip_code=clip_code,
                subject=subject_id,
                session=session,
                run_number=run_number,
                run_id=run_id,
                run_index=-1,  # assigned below after the global sort
                outcome=outcome,
                phase=phase,
                level=level,
                metadata=metadata,
            )
        )

    # Sort by clip_code ascending (ordinal — lower = earlier in the session).
    specs.sort(key=lambda s: s.clip_code)

    # Assign chronological run_index by first-encounter order in the sorted list.
    # Clips sharing a run_id must be contiguous here — clip_code is ordinal
    # within a run, and runs are time-disjoint by construction (each events
    # TSV is one fMRI acquisition).
    run_index_by_id: dict[str, int] = {}
    last_run_id: str | None = None
    for spec in specs:
        if spec.run_id not in run_index_by_id:
            run_index_by_id[spec.run_id] = len(run_index_by_id)
        elif last_run_id != spec.run_id:
            raise ValueError(
                f"Non-contiguous run_id in chronological order: {spec.run_id} "
                f"reappears after {last_run_id}. "
                "Clip codes do not preserve per-run contiguity."
            )
        spec.run_index = run_index_by_id[spec.run_id]
        last_run_id = spec.run_id

    logger.info(
        "Loaded %d episodes for %s (%d skipped). "
        "Runs: %d. Unique scenes: %d. Total frames: %d.",
        len(specs),
        subject_id,
        skipped,
        len(run_index_by_id),
        len({s.scene_id for s in specs}),
        sum(s.max_steps for s in specs),
    )
    return specs


def print_curriculum_summary(specs: list[EpisodeSpec]) -> None:
    """Print a human-readable summary of a loaded curriculum.

    Args:
        specs: List of ``EpisodeSpec`` objects as returned by
            ``load_curriculum()``.
    """
    if not specs:
        print("Empty curriculum.")
        return

    unique_scenes = sorted({s.scene_id for s in specs})
    total_frames = sum(s.max_steps for s in specs)
    total_minutes = total_frames / (60 * 60)  # 60 fps, 60 seconds

    print(f"Subject:       {specs[0].subject}")
    print(f"Episodes:      {len(specs)}")
    print(f"Unique scenes: {len(unique_scenes)}")
    print(f"Total frames:  {total_frames:,}  (~{total_minutes:.1f} min at 60 fps)")
    print(f"Clip range:    {specs[0].clip_code}  →  {specs[-1].clip_code}")

    # Per-scene breakdown.
    from collections import Counter
    counts = Counter(s.scene_id for s in specs)
    print("\nTop 10 scenes by episode count:")
    for sid, n in counts.most_common(10):
        print(f"  {sid:<12} {n:4d} episodes")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_state_file(
    scenes_root: Path,
    subject_id: str,
    session: str,
    stim_file: str,
    clip_code: str,
    level: str,
    scene_id: str,
) -> Optional[Path]:
    """Resolve the absolute path to a clip's ``.state`` file.

    Derives the path from the ``stim_file`` column in the events TSV by
    replacing the ``.bk2`` extension with ``.state``.

    Args:
        scenes_root: Root of the ``mario.scenes`` dataset.
        subject_id: Subject identifier.
        session: Session identifier.
        stim_file: ``stim_file`` field from the events TSV row
            (relative path from ``scenes_root``).
        clip_code: Clip code (used as a fallback for path construction).
        level: Level string (e.g. ``'w1l1'``).
        scene_id: Scene identifier (e.g. ``'w1l1s0'``).

    Returns:
        Absolute ``Path`` to the ``.state`` file, or ``None`` if the
        ``stim_file`` field is unusable.
    """
    if not stim_file or stim_file == "nan":
        logger.debug(
            "No stim_file for clip %s — cannot resolve state path.", clip_code
        )
        return None

    # stim_file is a relative path like:
    #   sub-01/ses-001/gamelogs/sub-01_ses-001_task-mario_level-w1l1_scene-0_clip-XXX.bk2
    bk2_path = scenes_root / stim_file
    state_path = bk2_path.with_suffix(".state")

    if state_path.exists():
        return state_path

    # The clip number embedded in the filename may be off by ±1 from the
    # value recorded in stim_file (a known discrepancy in some event TSVs
    # where frame_start was incremented after the files were saved).
    # Search nearby clip numbers before giving up.
    gamelog_dir = bk2_path.parent
    parts = bk2_path.stem.rsplit("_clip-", 1)
    if len(parts) == 2:
        base, clip_num_str = parts
        try:
            clip_num = int(clip_num_str)
            n_digits = len(clip_num_str)
            for delta in (-1, 1, -2, 2):
                candidate = gamelog_dir / f"{base}_clip-{clip_num + delta:0{n_digits}d}.state"
                if candidate.exists():
                    logger.debug(
                        "State file clip number adjusted by %+d for clip %s: %s",
                        delta,
                        clip_code,
                        candidate.name,
                    )
                    return candidate
        except ValueError:
            pass

    return state_path  # non-existent; caller will log the warning
