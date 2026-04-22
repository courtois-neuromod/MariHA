"""Generate per-scene scenario.json files and a scenes metadata index from the mastersheet.

stable-retro's scenario.json format cannot express composite RAM expressions like
``player_x_posHi * 256 + player_x_posLo``, so all termination and reward logic lives in
Python (see ``scene.py``).  These scenario files serve one purpose only: telling stable-retro
which RAM variables to expose in the ``info`` dict on every ``step()``.

Usage (CLI)::

    python -m mariha.env.scenario_gen
    # or via the installed entry-point:
    mariha-generate-scenarios
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

import os

_REPO_ROOT = Path(__file__).resolve().parents[2]  # MariHA/
_DATA_ROOT = Path(os.environ["MARIHA_DATA_ROOT"]) if "MARIHA_DATA_ROOT" in os.environ else _REPO_ROOT / "data"
_MASTERSHEET = _DATA_ROOT / "mario.scenes" / "sourcedata" / "scenes_info" / "scenes_mastersheet.csv"
_SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"

# ---------------------------------------------------------------------------
# RAM variables to expose in every info dict
# ---------------------------------------------------------------------------

# Defined in data/mario/stimuli/SuperMarioBros-Nes/data.json.
# All termination / reward logic uses these values; keeping the list here
# as a single source of truth makes it easy to add variables later.
INFO_VARIABLES: list[str] = [
    "player_x_posHi",
    "player_x_posLo",
    "player_y_pos",
    "lives",
    "score",
    "coins",
    "stage",
    "scrolling",
    "powerstate",
]

# ---------------------------------------------------------------------------
# Scene ID helpers
# ---------------------------------------------------------------------------

# Binary game-design pattern columns in the mastersheet (28 columns).
_PATTERN_COLUMNS: list[str] = [
    "Layout", "Enemy", "2-Horde", "3-Horde", "4-Horde", "Roof", "Gap",
    "Multiple gaps", "Variable gaps", "Gap enemy", "Pillar gap", "Valley",
    "Pipe valley", "Empty valley", "Enemy valley", "Roof valley", "2-Path",
    "3-Path", "Risk/Reward", "Stair up", "Stair down", "Empty stair valley",
    "Enemy stair valley", "Gap stair valley", "Reward", "Moving platform",
    "Flagpole", "Beginning", "Bonus zone", "Waterworld", "Checkpoint",
]


def make_scene_id(world: int, level: int, scene: int) -> str:
    """Return the canonical scene identifier string, e.g. ``'w1l1s0'``."""
    return f"w{world}l{level}s{scene}"


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def _build_scenario_json() -> dict:
    """Return the scenario.json content shared by every scene.

    Reward and done conditions are intentionally left empty — they are
    computed in Python inside ``SceneEnv.step()``.  The ``info`` block
    instructs stable-retro to read and expose the listed RAM addresses.
    """
    return {
        "reward": {},
        "done": {"variables": {}},
        "info": {var: {} for var in INFO_VARIABLES},
    }


def generate_scenarios(
    mastersheet_path: Path = _MASTERSHEET,
    output_dir: Path = _SCENARIOS_DIR,
) -> dict[str, dict]:
    """Generate per-scene scenario.json files and a ``scenes_metadata.json`` index.

    Args:
        mastersheet_path: Path to ``scenes_mastersheet.csv``.
        output_dir: Directory where scenario files and the metadata index are written.

    Returns:
        The scenes metadata dictionary (``scene_id`` → metadata dict).

    Raises:
        FileNotFoundError: If ``mastersheet_path`` does not exist.
    """
    if not mastersheet_path.exists():
        raise FileNotFoundError(f"Mastersheet not found: {mastersheet_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(mastersheet_path)

    # Drop the trailing summary row (World is NaN).
    df = df.dropna(subset=["World"]).copy()
    df["World"] = df["World"].astype(int)
    df["Level"] = df["Level"].astype(int)
    df["Scene"] = df["Scene"].astype(int)

    scenario_content = _build_scenario_json()
    metadata: dict[str, dict] = {}

    for _, row in df.iterrows():
        world, level, scene = int(row["World"]), int(row["Level"]), int(row["Scene"])
        scene_id = make_scene_id(world, level, scene)

        # Write the scenario.json (identical content for every scene — the
        # exit point is stored in metadata, not in the JSON).
        scenario_path = output_dir / f"{scene_id}.json"
        with open(scenario_path, "w") as fh:
            json.dump(scenario_content, fh, indent=2)

        # Collect metadata for the index.
        entry = {
            "scene_id": scene_id,
            "world": world,
            "level": level,
            "scene": scene,
            "entry_point": int(row["Entry point"]),
            "exit_point": int(row["Exit point"]),
            "patterns": {
                col: bool(row[col])
                for col in _PATTERN_COLUMNS
                if col in row.index
            },
        }
        metadata[scene_id] = entry

    # Write the metadata index.
    index_path = output_dir / "scenes_metadata.json"
    with open(index_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info(
        "Generated %d scenario files and scenes_metadata.json in %s",
        len(metadata),
        output_dir,
    )
    return metadata


def load_metadata(output_dir: Path = _SCENARIOS_DIR) -> dict[str, dict]:
    """Load the pre-generated scenes metadata index.

    Args:
        output_dir: Directory containing ``scenes_metadata.json``.

    Returns:
        Dictionary mapping ``scene_id`` to its metadata dict.

    Raises:
        FileNotFoundError: If ``scenes_metadata.json`` does not exist.
            Run ``generate_scenarios()`` first.
    """
    index_path = output_dir / "scenes_metadata.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"scenes_metadata.json not found in {output_dir}. "
            "Run `mariha-generate-scenarios` first."
        )
    with open(index_path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: generate all scenario files."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metadata = generate_scenarios()
    print(f"Done. {len(metadata)} scenes written to {_SCENARIOS_DIR}/")


if __name__ == "__main__":
    main()
