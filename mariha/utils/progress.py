"""Terminal progress tracker for training runs.

Curriculum training runs iterate over thousands of human clips, and the
default per-episode log line is too sparse to tell at a glance which clip
is playing, how far into the sequence we are, or how long the run will
take.  This module provides a small progress-display layer that is driven
entirely by the environment and the logger — **agent code is
progress-agnostic**.

Three implementations are provided:

- ``NullProgress`` — preserves the legacy ``Ep N | return=... | len=... |
  buf=...%`` per-episode print.  Used for ``--progress off``.
- ``LineProgress`` — prints one enriched line per episode with clip code,
  clip position, rolling stats, rate, and ETA.  No extra dependency.
- ``LiveProgress`` — persistent ``rich.Live`` dashboard pinned to the
  bottom of the terminal, with regular log output scrolling above it.

The factory ``build_progress(mode, fallback_log)`` chooses the right
implementation, falling back to ``LineProgress`` if ``rich`` is missing
or stdout is not a TTY.

Wiring (who calls what, so new agents don't have to think about any of it):

- ``build_benchmark_context`` / ``build_single_scene_context`` build the
  progress object, call ``init_meta`` to seed agent/subject/totals, and
  attach it to both the logger (``logger.progress``) and the env
  (``ContinualLearningEnv(progress=...)``).
- The **env** calls ``on_reset`` at the start of each episode and
  ``on_episode_end`` once the previous episode's final step has been
  consumed.  The env tracks ``episode_return`` / ``episode_len`` /
  ``global_step`` itself.
- The **logger** forwards every scalar passed to ``store({...})`` into
  ``progress.update_metrics`` so agent-specific counters like
  ``buffer_fill_pct`` land in the display without agent code touching
  the progress tracker.
- The scripts wrap ``agent.run()`` in ``with logger.progress:`` so
  ``LiveProgress`` opens/closes its ``rich.Live`` context cleanly.

Agents don't import this module.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from mariha.utils.logging import colorize


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProgressSnapshot:
    """Immutable snapshot of the training state displayed by renderers."""

    agent_name: str = ""
    subject: str = ""
    seed: int = 0
    clip_total: Optional[int] = None
    total_steps: Optional[int] = None

    # Current clip
    clip_code: str = ""
    clip_index: int = 0          # 0-based
    scene_id: str = ""
    session: str = ""
    level: str = ""
    phase: str = ""
    human_outcome: str = ""
    max_steps: int = 0

    # Latest episode
    episode_return: float = 0.0
    episode_len: int = 0
    cleared: bool = False
    outcome: str = ""
    global_step: int = 0

    # Rolling stats
    return_mean: float = 0.0
    success_cleared: int = 0
    success_total: int = 0

    # Throughput
    rate_ewma: float = 0.0       # clips per second
    eta_seconds: Optional[float] = None

    # Task progress (continual learning)
    task_idx: int = 0
    total_tasks: int = 0

    episodes_completed: int = 0

    # Arbitrary scalar metrics forwarded from ``logger.store(...)`` calls.
    # Renderers can pick out well-known keys (e.g. ``buffer_fill_pct``) and
    # ignore the rest.
    extra_metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class TrainingProgress(ABC):
    """Abstract progress tracker.

    Implementations receive episode-boundary events from the training loop
    and render them however they like.  The base class owns rolling
    statistics (clip-rate EWMA, success window, return window) so every
    implementation shares the same counters.
    """

    # Rolling window sizes.
    _RATE_WINDOW = 20
    _SUCCESS_WINDOW = 50
    _RETURN_WINDOW = 50
    _RATE_EWMA_ALPHA = 0.2

    def __init__(self) -> None:
        self._snap = ProgressSnapshot()
        self._started: bool = False
        self._start_time: float = 0.0
        self._last_episode_time: Optional[float] = None
        self._rate_ewma: float = 0.0
        self._rate_samples: int = 0
        self._success_buf: Deque[bool] = deque(maxlen=self._SUCCESS_WINDOW)
        self._return_buf: Deque[float] = deque(maxlen=self._RETURN_WINDOW)
        self._episodes_completed: int = 0
        self._scene_ids: List[str] = []

    # ------------------------------------------------------------------
    # Meta init (called before `start` so the factory can pre-populate
    # agent/subject/total info without kicking the wall clock).
    # ------------------------------------------------------------------

    def init_meta(
        self,
        *,
        agent_name: str,
        subject: str,
        seed: int,
        clip_total: Optional[int] = None,
        total_steps: Optional[int] = None,
        scene_ids: Optional[List[str]] = None,
    ) -> None:
        """Populate agent/subject/budget fields before the loop starts.

        ``scene_ids``, when supplied, lets ``on_reset`` resolve task indices
        automatically so the env (and any agent) never has to call
        ``on_task_switch`` explicitly.
        """
        self._snap.agent_name = agent_name
        self._snap.subject = subject
        self._snap.seed = seed
        self._snap.clip_total = clip_total
        self._snap.total_steps = total_steps
        if scene_ids is not None:
            self._scene_ids = list(scene_ids)
            self._snap.total_tasks = len(self._scene_ids)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin the display and start the wall clock."""
        self._started = True
        self._start_time = time.monotonic()
        self._last_episode_time = self._start_time
        self._on_start()

    def stop(self) -> None:
        """Tear down the display cleanly (called on exit or exception)."""
        if not self._started:
            return
        self._on_stop()
        self._started = False

    def __enter__(self) -> "TrainingProgress":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.stop()

    # ------------------------------------------------------------------
    # Events from the training loop
    # ------------------------------------------------------------------

    def on_reset(self, info: dict) -> None:
        """Called by the env at the start of each new episode."""
        s = self._snap
        prev_scene = s.scene_id
        s.clip_code = str(info.get("clip_code", ""))
        s.clip_index = int(info.get("clip_index", 0))
        if info.get("clip_total") is not None and s.clip_total is None:
            s.clip_total = int(info["clip_total"])
        s.scene_id = str(info.get("scene_id", ""))
        s.session = str(info.get("session", ""))
        s.level = str(info.get("level", ""))
        s.phase = str(info.get("phase", ""))
        s.human_outcome = str(info.get("human_outcome", ""))
        s.max_steps = int(info.get("max_steps", 0))
        if self._scene_ids and s.scene_id in self._scene_ids:
            s.task_idx = self._scene_ids.index(s.scene_id)
            s.total_tasks = len(self._scene_ids)
        # Scene boundary → emit a task-switch log line (once we have a
        # previous scene to compare against; the very first reset is silent).
        if prev_scene and prev_scene != s.scene_id:
            self.log(
                f"[task switch] → {s.scene_id} ({s.task_idx}/{s.total_tasks})",
                color="magenta",
            )
        self._on_reset()

    def on_episode_end(
        self,
        *,
        episode_return: float,
        episode_len: int,
        terminated: bool,
        truncated: bool,
        cleared: bool,
        outcome: Optional[str],
        global_step: int,
    ) -> None:
        """Called by the env once the episode's final transition has been seen."""
        now = time.monotonic()
        if self._last_episode_time is not None:
            dt = max(1e-6, now - self._last_episode_time)
            inst_rate = 1.0 / dt
            if self._rate_samples == 0:
                self._rate_ewma = inst_rate
            else:
                self._rate_ewma = (
                    self._RATE_EWMA_ALPHA * inst_rate
                    + (1.0 - self._RATE_EWMA_ALPHA) * self._rate_ewma
                )
            self._rate_samples += 1
        self._last_episode_time = now

        self._success_buf.append(bool(cleared))
        self._return_buf.append(float(episode_return))
        self._episodes_completed += 1

        s = self._snap
        s.episode_return = float(episode_return)
        s.episode_len = int(episode_len)
        s.cleared = bool(cleared)
        s.outcome = str(outcome or "")
        s.global_step = int(global_step)
        s.episodes_completed = self._episodes_completed
        s.return_mean = (
            sum(self._return_buf) / len(self._return_buf) if self._return_buf else 0.0
        )
        s.success_cleared = sum(1 for c in self._success_buf if c)
        s.success_total = len(self._success_buf)
        s.rate_ewma = self._rate_ewma
        s.eta_seconds = self._estimate_eta()

        self._on_episode_end()

    def update_metrics(self, **metrics: float) -> None:
        """Merge arbitrary scalar metrics into the snapshot.

        Called by ``EpochLogger.store`` so every agent can surface custom
        scalars (e.g. ``buffer_fill_pct``, ``epsilon``, ``entropy``) to the
        display without importing anything from this module.  Non-numeric
        values are silently skipped.
        """
        for key, value in metrics.items():
            try:
                self._snap.extra_metrics[key] = float(value)
            except (TypeError, ValueError):
                continue

    # ------------------------------------------------------------------
    # Generic log routing
    # ------------------------------------------------------------------

    @abstractmethod
    def log(self, msg: str, color: Optional[str] = None) -> None:
        """Print a log line (scrollback) that interleaves with the display."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_eta(self) -> Optional[float]:
        """Estimate remaining wall-clock seconds based on the EWMA rate."""
        s = self._snap
        if self._rate_ewma <= 0.0:
            return None
        # Prefer clip-based ETA when we know the total.
        if s.clip_total is not None and s.clip_total > 0:
            remaining = max(0, s.clip_total - s.clip_index - 1)
            return remaining / self._rate_ewma
        # Fallback: step-budget ETA (single-scene mode).
        if s.total_steps is not None and s.total_steps > 0 and s.global_step > 0:
            elapsed = time.monotonic() - self._start_time
            if elapsed <= 0:
                return None
            step_rate = s.global_step / elapsed
            if step_rate <= 0:
                return None
            remaining_steps = max(0, s.total_steps - s.global_step)
            return remaining_steps / step_rate
        return None

    # Hooks for subclasses.
    def _on_start(self) -> None:
        pass

    def _on_stop(self) -> None:
        pass

    def _on_reset(self) -> None:
        pass

    def _on_episode_end(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _fmt_rate_per_min(rate_per_s: float) -> str:
    if rate_per_s <= 0:
        return "--"
    return f"{rate_per_s * 60.0:.1f}/min"


def _fmt_success(cleared: int, total: int) -> str:
    if total == 0:
        return "n/a"
    if total < 10:
        return f"{cleared}/{total}"
    return f"{cleared / total * 100:.0f}% ({cleared}/{total})"


# ---------------------------------------------------------------------------
# NullProgress — preserves legacy terminal output
# ---------------------------------------------------------------------------


class NullProgress(TrainingProgress):
    """Preserves the legacy per-episode ``Ep N | return=... | len=... | buf=%`` print.

    Used for ``--progress off``.  Delegates ``log()`` to a provided fallback
    (typically ``logger.log``) so existing messages stay unchanged.
    """

    def __init__(
        self,
        fallback_log: Optional[Callable[..., None]] = None,
    ) -> None:
        super().__init__()
        self._fallback_log = fallback_log

    def _on_episode_end(self) -> None:
        s = self._snap
        buf_pct = s.extra_metrics.get("buffer_fill_pct", 0.0)
        msg = (
            f"Ep {s.episodes_completed:5d} | return={s.episode_return:7.2f} | "
            f"len={s.episode_len:4d} | buf={buf_pct:.1f}%"
        )
        self.log(msg, color="green")

    def log(self, msg: str, color: Optional[str] = None) -> None:
        if self._fallback_log is not None:
            try:
                self._fallback_log(msg, color=color or "green")
                return
            except TypeError:
                self._fallback_log(msg)
                return
        print(colorize(msg, color or "green", bold=True))


# ---------------------------------------------------------------------------
# LineProgress — one enriched line per episode
# ---------------------------------------------------------------------------


class LineProgress(TrainingProgress):
    """Prints one enriched summary line per episode.  No extra dependency."""

    def __init__(
        self,
        fallback_log: Optional[Callable[..., None]] = None,
    ) -> None:
        super().__init__()
        self._fallback_log = fallback_log

    def _on_start(self) -> None:
        s = self._snap
        total = s.clip_total if s.clip_total is not None else "?"
        self.log(
            f"[{s.subject} | {s.agent_name}] progress: line mode — {total} clips",
            color="cyan",
        )

    def _on_episode_end(self) -> None:
        s = self._snap
        total = s.clip_total if s.clip_total is not None else "?"
        pos = f"{s.clip_index + 1}/{total}"
        clip = s.clip_code or "?"
        scene = s.scene_id or "?"
        scene_ctx = []
        if s.level:
            scene_ctx.append(f"lvl={s.level}")
        if s.phase:
            scene_ctx.append(f"ph={s.phase}")
        if s.human_outcome:
            scene_ctx.append(f"human={s.human_outcome}")
        scene_str = f"{scene} ({' '.join(scene_ctx)})" if scene_ctx else scene

        buf_pct = s.extra_metrics.get("buffer_fill_pct", 0.0)
        msg = (
            f"[{s.subject} | {s.agent_name}] clip {pos} {scene_str} code={clip} | "
            f"ret={s.episode_return:.2f} μ={s.return_mean:.2f} len={s.episode_len} "
            f"cleared={s.cleared} | buf={buf_pct:.0f}% | "
            f"{_fmt_rate_per_min(s.rate_ewma)} ETA {_fmt_eta(s.eta_seconds)} | "
            f"succ {_fmt_success(s.success_cleared, s.success_total)}"
        )
        self.log(msg, color="green")

    def log(self, msg: str, color: Optional[str] = None) -> None:
        if self._fallback_log is not None:
            try:
                self._fallback_log(msg, color=color or "green")
                return
            except TypeError:
                self._fallback_log(msg)
                return
        print(colorize(msg, color or "green", bold=True))


# ---------------------------------------------------------------------------
# LiveProgress — persistent rich.Live dashboard
# ---------------------------------------------------------------------------


class LiveProgress(TrainingProgress):
    """Persistent rich.Live dashboard pinned to the bottom of the terminal."""

    def __init__(self) -> None:
        super().__init__()
        self._console = None
        self._live = None
        self._rich_panel = None
        self._rich_table = None
        self._rich_text = None

    def _on_start(self) -> None:
        # Lazy import so the factory can fall back if rich is missing.
        from rich.console import Console
        from rich.live import Live

        self._console = Console()
        self._live = Live(
            self._build_renderable(),
            console=self._console,
            refresh_per_second=4,
            redirect_stdout=True,
            redirect_stderr=True,
            transient=False,
        )
        self._live.__enter__()

    def _on_stop(self) -> None:
        if self._live is not None:
            try:
                self._live.update(self._build_renderable())
                self._live.__exit__(None, None, None)
            finally:
                self._live = None

    def _on_reset(self) -> None:
        self._refresh()

    def _on_episode_end(self) -> None:
        self._refresh()

    def log(self, msg: str, color: Optional[str] = None) -> None:
        if self._console is not None:
            style = color if color else None
            self._console.log(msg, style=style)
        else:
            print(colorize(msg, color or "green", bold=True))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_renderable())

    def _build_renderable(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.progress_bar import ProgressBar
        from rich.console import Group
        from rich.text import Text

        s = self._snap

        # Progress bar: clip-based if known, otherwise step-based.
        if s.clip_total is not None and s.clip_total > 0:
            completed = s.clip_index + (1 if s.episodes_completed > 0 else 0)
            total = s.clip_total
            bar_label = f"Clip {completed}/{total}"
        elif s.total_steps is not None and s.total_steps > 0:
            completed = s.global_step
            total = s.total_steps
            bar_label = f"Step {completed}/{total}"
        else:
            completed = s.episodes_completed
            total = max(1, s.episodes_completed)
            bar_label = f"Episode {completed}"

        pct = (completed / total * 100.0) if total > 0 else 0.0
        bar = ProgressBar(total=total, completed=completed, width=40)
        bar_line = Text.assemble(
            (f"{bar_label:<22}", "bold"),
            "  ",
        )
        pct_line = Text(
            f"  {pct:5.1f}%   ETA {_fmt_eta(s.eta_seconds)}   rate {_fmt_rate_per_min(s.rate_ewma)}"
        )

        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(justify="right", style="bold cyan")
        tbl.add_column(justify="left")

        header = f"{s.agent_name or '?'}   subject={s.subject or '?'}   seed={s.seed}"
        tbl.add_row("Run", header)
        tbl.add_row("Clip", s.clip_code or "?")

        scene_bits = [s.scene_id or "?"]
        if s.level:
            scene_bits.append(f"lvl={s.level}")
        if s.phase:
            scene_bits.append(f"ph={s.phase}")
        if s.session:
            scene_bits.append(f"ses={s.session}")
        tbl.add_row("Scene", "   ".join(scene_bits))

        if s.max_steps:
            tbl.add_row("Budget", f"episode max_steps={s.max_steps}")

        tbl.add_row(
            "Return",
            f"last={s.episode_return:7.2f}   μ={s.return_mean:7.2f}   "
            f"cleared={s.cleared}   human={s.human_outcome or '-'}",
        )
        tbl.add_row(
            "Success",
            f"{_fmt_success(s.success_cleared, s.success_total)}   "
            f"(window={self._SUCCESS_WINDOW})",
        )
        buf_pct = s.extra_metrics.get("buffer_fill_pct")
        if buf_pct is not None:
            tbl.add_row(
                "Buffer",
                f"{buf_pct:.1f}%   global_steps={s.global_step:,}",
            )
        else:
            tbl.add_row("Steps", f"global={s.global_step:,}")

        # Surface any other scalar metrics that the agent forwarded via
        # ``logger.store``.  We already render ``buffer_fill_pct`` above, so
        # skip it here.  Common keys include ``epsilon`` (DQN) and
        # ``alpha``/``entropy`` (SAC); unknown keys are shown verbatim.
        extras = {
            k: v for k, v in s.extra_metrics.items() if k != "buffer_fill_pct"
        }
        if extras:
            bits = "   ".join(f"{k}={v:.3g}" for k, v in sorted(extras.items()))
            tbl.add_row("Metrics", bits)

        if s.total_tasks:
            tbl.add_row("Task", f"{s.task_idx}/{s.total_tasks}")

        body = Group(bar_line, bar, pct_line, Text(""), tbl)
        panel = Panel(
            body,
            title="MariHA training",
            border_style="cyan",
            padding=(1, 2),
        )
        return panel


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_progress(
    mode: str,
    fallback_log: Optional[Callable[..., None]] = None,
) -> TrainingProgress:
    """Build a progress tracker for the requested mode.

    Args:
        mode: ``"live"``, ``"line"``, or ``"off"``.
        fallback_log: Callable used by ``NullProgress`` / ``LineProgress``
            for log routing (typically ``logger.log``).

    Returns:
        A ``TrainingProgress`` instance.  Silently degrades ``live`` → ``line``
        if stdout is not a TTY or ``rich`` is not importable.
    """
    mode = (mode or "live").lower()

    if mode == "off":
        return NullProgress(fallback_log=fallback_log)

    if mode == "line":
        return LineProgress(fallback_log=fallback_log)

    if mode == "live":
        if not sys.stdout.isatty():
            print(
                colorize(
                    "[progress] stdout is not a TTY — falling back to --progress line.",
                    "yellow",
                    bold=True,
                )
            )
            return LineProgress(fallback_log=fallback_log)
        try:
            import rich  # noqa: F401
        except ImportError:
            print(
                colorize(
                    "[progress] 'rich' is not installed — falling back to --progress line. "
                    "Install it with `pip install rich`.",
                    "yellow",
                    bold=True,
                )
            )
            return LineProgress(fallback_log=fallback_log)
        return LiveProgress()

    raise ValueError(f"Unknown --progress mode: {mode!r}")
