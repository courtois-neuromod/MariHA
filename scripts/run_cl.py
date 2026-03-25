"""Continual learning training script (backwards-compatibility shim).

Delegates to ``run_benchmark.py`` after mapping ``--cl_method`` to
``--algorithm``.  Prefer using ``mariha-run`` directly for new workflows.

Usage (unchanged from before)::

    python scripts/run_cl.py --subject sub-01 --cl_method ewc --seed 0

Or via the entry point::

    mariha-run-cl --subject sub-01 --cl_method ewc
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    # Map legacy --cl_method <name> → --algorithm <name> so run_benchmark
    # can handle it uniformly.  Any value of --cl_method that is not None
    # becomes --algorithm; None falls back to "sac".
    argv = sys.argv[1:]
    if "--cl_method" in argv:
        idx = argv.index("--cl_method")
        method_value = argv[idx + 1] if idx + 1 < len(argv) else None
        if method_value and not method_value.startswith("--"):
            argv[idx] = "--algorithm"
        else:
            # --cl_method with no value or bare flag: remove it (defaults to sac)
            argv.pop(idx)
    sys.argv = [sys.argv[0]] + argv

    from scripts.run_benchmark import main as run_main
    run_main()


if __name__ == "__main__":
    main()
