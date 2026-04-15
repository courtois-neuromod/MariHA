"""Experiment logging: console, TSV file, TensorBoard, and WandB stub.

``EpochLogger`` accumulates scalar statistics over many environment steps and
writes them out once per logging epoch.  It supports three backends:

- ``"tsv"``         — tab-separated-values file in the run directory.
- ``"tensorboard"`` — TensorBoard scalar summaries.
- ``"wandb"``       — Weights & Biases (optional extra; must be installed).

Usage::

    from mariha.utils.logging import EpochLogger

    logger = EpochLogger(
        output_dir="experiments/run_01",
        logger_output=["tsv", "tensorboard"],
        config=vars(args),
    )
    logger.store({"train/return": 42.0, "train/ep_length": 100})
    logger.log_tabular("train/return", with_min_and_max=True)
    logger.dump_tabular()

Ported from COOM; Neptune / mrunner backends removed.  WandB is imported
lazily so the package works without it installed.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import os.path as osp
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from mariha.utils.running import get_readable_timestamp, get_random_string

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal colour helpers
# ---------------------------------------------------------------------------

_COLOR_CODES = dict(
    gray=30, red=31, green=32, yellow=33, blue=34,
    magenta=35, cyan=36, white=37, crimson=38,
)


def colorize(string: str, color: str, bold: bool = False) -> str:
    """Wrap ``string`` in ANSI colour escape codes."""
    attr = [str(_COLOR_CODES[color])]
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


# ---------------------------------------------------------------------------
# Base logger
# ---------------------------------------------------------------------------


class Logger:
    """Write experiment diagnostics to file, console, and optional backends.

    Args:
        output_dir: Directory for log files.  Created if it does not exist.
            Defaults to ``./experiments/<group_id>/<timestamp>_<random>``.
        logger_output: List of output backends to enable.  Any combination of
            ``"tsv"``, ``"tensorboard"``, ``"wandb"``.
        config: Hyperparameter dict saved as ``config.json`` in ``output_dir``.
        group_id: Used to build the default ``output_dir`` path.
        exp_name: Human-readable experiment label.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        logger_output: List[str] = ("tsv",),
        config: Optional[Dict[str, Any]] = None,
        group_id: str = "default",
        exp_name: Optional[str] = None,
    ) -> None:
        self.logger_output = list(logger_output)

        run_id = get_readable_timestamp() + "_" + get_random_string()
        self.output_dir = output_dir or f"./experiments/{group_id}/{run_id}"
        if osp.exists(self.output_dir):
            print(f"Warning: Log dir {self.output_dir} already exists — appending.")
        else:
            os.makedirs(self.output_dir)

        # TSV output file
        self.output_file = None
        if "tsv" in self.logger_output:
            tsv_path = osp.join(self.output_dir, "progress.tsv")
            self.output_file = open(tsv_path, "w")
            atexit.register(self.output_file.close)

        # TensorBoard
        if "tensorboard" in self.logger_output:
            self.tb_writer = tf.summary.create_file_writer(self.output_dir)
            self.tb_writer.set_as_default()

        # WandB (optional)
        if "wandb" in self.logger_output:
            try:
                import wandb  # noqa: F401
                self._wandb_enabled = True
            except ImportError:
                print("Warning: wandb not installed — skipping wandb logging.")
                self._wandb_enabled = False
        else:
            self._wandb_enabled = False

        self.first_row = True
        self.log_headers: List[str] = []
        self.log_current_row: Dict[str, Any] = {}
        self.exp_name = exp_name

        if config is not None:
            self.save_config(config)

        print(colorize(f"Logging to {self.output_dir}", "green", bold=True))

    def log(self, msg: str, color: str = "green") -> None:
        """Print a timestamped, colourised message to stdout."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(colorize(f"{ts} — {msg}", color, bold=True))

    def log_tabular(self, key: str, val: Any) -> None:
        """Record a scalar value for the current epoch.

        Call once per key per epoch, then call ``dump_tabular()`` to flush.
        """
        if key not in self.log_headers:
            self.log_headers.append(key)
        self.log_current_row[key] = val

    def save_config(self, config: Dict[str, Any]) -> None:
        """Serialise the experiment config to ``config.json``."""
        config_json = _convert_json(config)
        output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
        with open(osp.join(self.output_dir, "config.json"), "w") as f:
            f.write(output)

    def dump_tabular(self) -> None:
        """Write all accumulated scalars to each enabled backend and reset."""
        vals = []
        key_lens = [len(k) for k in self.log_headers]
        max_key_len = max(15, max(key_lens)) if key_lens else 15
        fmt = f"| %{max_key_len}s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)

        step = self.log_current_row.get("total_env_steps")

        for key in self.log_headers:
            val = self.log_current_row.get(key, 0.0)
            valstr = "%8.3g" % val if hasattr(val, "__float__") else str(val)
            print(fmt % (key, valstr))
            vals.append(val)

            if "tensorboard" in self.logger_output and step is not None:
                tf.summary.scalar(key, data=val, step=step)

            if self._wandb_enabled and step is not None:
                try:
                    import wandb
                    wandb.log({key: val}, step=int(step))
                except Exception:
                    pass

        if "tensorboard" in self.logger_output:
            tf.summary.flush()

        print("-" * n_slashes, flush=True)

        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()

        self.log_current_row.clear()
        self.first_row = False


# ---------------------------------------------------------------------------
# EpochLogger — accumulates statistics over multiple steps
# ---------------------------------------------------------------------------


class EpochLogger(Logger):
    """Logger that averages/min/maxes quantities accumulated within an epoch.

    Instead of calling ``log_tabular`` with a single value, call ``store``
    with a dict of scalars many times per epoch.  At the end of the epoch,
    call ``log_tabular(key, with_min_and_max=True)`` to record the mean (and
    optionally min/max), then ``dump_tabular()`` to flush.

    Example::

        for step in range(epoch_len):
            logger.store({"train/return": ep_return})
        logger.log_tabular("train/return", with_min_and_max=True)
        logger.dump_tabular()
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._epoch_dict: Dict[str, List[float]] = {}
        # Optional ``TrainingProgress`` tracker attached by the benchmark
        # context builder.  When set, every scalar value passed to ``store``
        # is forwarded to ``progress.update_metrics`` so agent-specific
        # metrics (buffer_fill_pct, epsilon, entropy, ...) land in the
        # terminal display without agents importing the progress module.
        self.progress: Any = None

    def store(self, data: Dict[str, Any], display: bool = True) -> None:
        """Accumulate scalar values.  ``data`` may contain scalars, arrays, or TF tensors.

        When ``display=False``, values are still written to TSV/TensorBoard/WandB
        but are not forwarded to the live terminal progress display.  Use for
        high-cardinality keys (e.g. per-task metrics across many tasks) that
        would flood the monitor.
        """
        forwarded: Dict[str, float] = {}
        for k, v in data.items():
            if k not in self._epoch_dict:
                self._epoch_dict[k] = []
            # Convert TF tensors to numpy for uniform handling.
            try:
                import tensorflow as tf
                if isinstance(v, tf.Tensor):
                    v = v.numpy()
            except ImportError:
                pass
            if isinstance(v, np.ndarray):
                self._epoch_dict[k].extend(v.flatten().tolist())
            elif isinstance(v, (list, tuple)):
                self._epoch_dict[k].extend([float(x) for x in v])
            else:
                fv = float(v)
                self._epoch_dict[k].append(fv)
                forwarded[k] = fv

        # Forward scalar values to the progress tracker (if any).  Only
        # forward scalars — arrays and sequences are meant for TSV/TB
        # aggregation, not single-value display widgets.
        if display and self.progress is not None and forwarded:
            try:
                self.progress.update_metrics(**forwarded)
            except Exception:
                # Progress display must never break the training loop.
                pass

    def log_tabular(
        self,
        key: str,
        val: Any = None,
        with_min_and_max: bool = False,
        average_only: bool = False,
    ) -> None:
        """Record statistics for a key accumulated via ``store()``.

        Args:
            key: Metric name.
            val: If provided, use this value directly (skip stored values).
            with_min_and_max: Also record ``key/min`` and ``key/max``.
            average_only: Record only the mean (suppress min/max even if stored).
        """
        if val is not None:
            super().log_tabular(key, val)
            return

        if key not in self._epoch_dict or len(self._epoch_dict[key]) == 0:
            super().log_tabular(key, 0.0)
            return

        values = np.array(self._epoch_dict[key], dtype=np.float64)
        mean = float(np.mean(values))
        super().log_tabular(key, mean)

        if with_min_and_max and not average_only:
            super().log_tabular(f"{key}/min", float(np.min(values)))
            super().log_tabular(f"{key}/max", float(np.max(values)))

        del self._epoch_dict[key]

    def dump_tabular(self) -> None:
        """Flush accumulated statistics and clear internal buffers."""
        super().dump_tabular()
        self._epoch_dict.clear()


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _convert_json(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serialisable form."""
    if isinstance(obj, dict):
        return {k: _convert_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if callable(obj):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)
