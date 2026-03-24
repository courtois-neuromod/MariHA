"""CLI script: generate all per-scene scenario.json files and scenes_metadata.json.

Run from the MariHA/ root::

    python scripts/generate_scenarios.py
    # or after pip install -e .:
    mariha-generate-scenarios
"""

from mariha.env.scenario_gen import main

if __name__ == "__main__":
    main()
