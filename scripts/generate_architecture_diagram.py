"""Generate a PlotNeuralNet-style PNG architecture diagram for a MariHA model.

Follows ``courtois-neuromod/mario.tutorials``'s ``generate_architecture_diagram.py``:
the network is described with PlotNeuralNet's ``tikzeng`` primitives, emitted as
a LaTeX/TikZ file, compiled with ``pdflatex``, and rasterised to PNG.

    tikzeng  ->  .tex  --(pdflatex)-->  .pdf  --(pdftoppm)-->  .png

The diagrams reflect the networks in ``mariha/rl/models.py``: the shared
``BaseCNN`` backbone (4 stride-2 conv blocks -> Flatten -> Dense 512), the
task one-hot concatenated to the 512-d features, then per-agent heads.

Usage::

    python scripts/generate_architecture_diagram.py --model sac
    python scripts/generate_architecture_diagram.py --model ppo --n-actions 9
    python scripts/generate_architecture_diagram.py --model all

Output: ``assets/model_architectures/<model>_architecture.png``.

This is developer tooling — regular benchmark users never need it. Install
its dependencies into an isolated environment with::

    uv pip install -e ".[diagrams]"

Requirements:
    * Python: the ``diagrams`` optional-dependency group (PyMuPDF).
    * System: ``pdflatex`` (TeX Live, e.g. ``apt install texlive-latex-extra``).
    * PlotNeuralNet: auto-cloned to ``<repo>/PlotNeuralNet`` if absent
      (override with ``--plotneuralnet`` or the ``PLOTNEURALNET_DIR`` env var).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "assets" / "model_architectures"
PLOTNN_URL = "https://github.com/HarisIqbal88/PlotNeuralNet.git"

# ``ddqn`` shares DQN's network; ``random`` has no network and is omitted.
SUPPORTED_MODELS = ("sac", "ppo", "dqn", "ddqn")


# ---------------------------------------------------------------------------
# PlotNeuralNet bootstrap
# ---------------------------------------------------------------------------


def ensure_plotneuralnet(explicit: Path | None) -> Path:
    """Return the PlotNeuralNet directory, cloning it if necessary.

    Search order: ``--plotneuralnet`` arg, ``PLOTNEURALNET_DIR`` env var,
    ``<repo>/PlotNeuralNet``. If none exist, the repo is cloned into
    ``<repo>/PlotNeuralNet``.
    """
    candidates = [
        explicit,
        Path(os.environ["PLOTNEURALNET_DIR"]) if "PLOTNEURALNET_DIR" in os.environ else None,
        REPO_ROOT / "PlotNeuralNet",
    ]
    for cand in candidates:
        if cand is not None and (cand / "pycore" / "tikzeng.py").is_file():
            return cand.resolve()

    target = REPO_ROOT / "PlotNeuralNet"
    print(f"[diagram] PlotNeuralNet not found — cloning into {target} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", PLOTNN_URL, str(target)], check=True
    )
    return target.resolve()


# ---------------------------------------------------------------------------
# Architecture definitions (PlotNeuralNet tikzeng calls)
# ---------------------------------------------------------------------------
#
# tikzeng is imported lazily inside build_arch() once sys.path is set, so the
# helpers below receive the primitives they need as explicit arguments.


def _backbone(tz) -> List[str]:
    """Shared BaseCNN backbone + task one-hot concat, as tikzeng layers.

    4 stride-2 conv blocks (84 -> 42 -> 21 -> 11 -> 6, 32 filters each),
    Flatten (1152), Dense projection (512), then Concatenate with the task
    one-hot. The branch point for every agent head is the ``concat`` node.
    """
    return [
        tz.to_Conv("input", 84, 4, offset="(0,0,0)", to="(0,0,0)",
                   height=30, depth=30, width=1, caption="Input 84x84x4"),

        tz.to_Conv("conv1", 42, 32, offset="(2.7,0,0)", to="(input-east)",
                   height=24, depth=24, width=3, caption="Conv 3x3 s2"),
        tz.to_connection("input", "conv1"),
        tz.to_Conv("conv2", 21, 32, offset="(2.7,0,0)", to="(conv1-east)",
                   height=18, depth=18, width=3, caption="Conv 3x3 s2"),
        tz.to_connection("conv1", "conv2"),
        tz.to_Conv("conv3", 11, 32, offset="(2.7,0,0)", to="(conv2-east)",
                   height=12, depth=12, width=3, caption="Conv 3x3 s2"),
        tz.to_connection("conv2", "conv3"),
        tz.to_Conv("conv4", 6, 32, offset="(2.7,0,0)", to="(conv3-east)",
                   height=8, depth=8, width=3, caption="Conv 3x3 s2"),
        tz.to_connection("conv3", "conv4"),

        tz.to_SoftMax("flatten", 1152, offset="(2.7,0,0)", to="(conv4-east)",
                      width=1.5, height=6, depth=6, opacity=0.6, caption="Flatten"),
        tz.to_connection("conv4", "flatten"),
        tz.to_SoftMax("dense", 512, offset="(2.7,0,0)", to="(flatten-east)",
                      width=2, height=10, depth=10, opacity=0.7, caption="Dense ReLU"),
        tz.to_connection("flatten", "dense"),

        # Task one-hot input, sitting below the dense projection.
        tz.to_SoftMax("task", "", offset="(2.7,-7,0)", to="(flatten-east)",
                      width=1.5, height=5, depth=5, opacity=0.7, caption="Task one-hot"),
        tz.to_SoftMax("concat", "", offset="(3,0,0)", to="(dense-east)",
                      width=2.2, height=11, depth=11, opacity=0.7,
                      caption="Concat 512+task"),
        tz.to_connection("dense", "concat"),
        tz.to_connection("task", "concat"),
    ]


def _heads(tz, model: str, n_actions: int) -> List[str]:
    """Per-agent output heads branching from the ``concat`` node."""
    if model == "ppo":
        return [
            tz.to_SoftMax("actor", n_actions, offset="(3.2,3,0)", to="(concat-east)",
                          width=1.5, height=5, depth=5, opacity=0.9,
                          caption="Actor (logits)"),
            tz.to_connection("concat", "actor"),
            tz.to_SoftMax("critic", 1, offset="(3.2,-3,0)", to="(concat-east)",
                          width=1.5, height=3, depth=3, opacity=0.9,
                          caption="Critic (value)"),
            tz.to_connection("concat", "critic"),
        ]
    if model == "sac":
        return [
            tz.to_SoftMax("actor", n_actions, offset="(3.2,3,0)", to="(concat-east)",
                          width=1.5, height=5, depth=5, opacity=0.9,
                          caption="Actor (logits)"),
            tz.to_connection("concat", "actor"),
            tz.to_SoftMax("critic", n_actions, offset="(3.2,-3,0)", to="(concat-east)",
                          width=1.5, height=5, depth=5, opacity=0.9,
                          caption="Critic (Q-values)"),
            tz.to_connection("concat", "critic"),
        ]
    # dqn / ddqn: single Q-head.
    return [
        tz.to_SoftMax("qhead", n_actions, offset="(3.2,0,0)", to="(concat-east)",
                      width=1.5, height=5, depth=5, opacity=0.9,
                      caption="Q-values"),
        tz.to_connection("concat", "qhead"),
    ]


def build_arch(tz, plotnn_dir: Path, model: str, n_actions: int) -> List[str]:
    """Assemble the full tikzeng architecture list for ``model``."""
    key = "dqn" if model == "ddqn" else model
    return [
        tz.to_head(str(plotnn_dir)),
        tz.to_cor(),
        tz.to_begin(),
        *_backbone(tz),
        *_heads(tz, key, n_actions),
        tz.to_end(),
    ]


# ---------------------------------------------------------------------------
# Compile + rasterise
# ---------------------------------------------------------------------------


def _require(tool: str, hint: str) -> None:
    """Abort with a clear message if a required system tool is missing."""
    if shutil.which(tool) is None:
        sys.exit(f"ERROR: '{tool}' not found on PATH. {hint}")


def _pdf_to_png(pdf: Path, png: Path, dpi: int = 300) -> None:
    """Rasterise the first page of ``pdf`` to ``png`` using PyMuPDF.

    PyMuPDF renders the PDF itself, so no system rasteriser (poppler /
    ImageMagick) is required — only the ``diagrams`` extra.
    """
    try:
        import pymupdf  # PyMuPDF >= 1.24 exposes the top-level name
    except ImportError:
        try:
            import fitz as pymupdf  # older PyMuPDF import name
        except ImportError:
            sys.exit(
                "ERROR: PyMuPDF is required to rasterise the PDF. Install the "
                "diagram tooling:  uv pip install -e \".[diagrams]\""
            )
    with pymupdf.open(pdf) as doc:
        doc[0].get_pixmap(dpi=dpi).save(png)


def render(tz, plotnn_dir: Path, model: str, n_actions: int, out_dir: Path) -> Path:
    """Build, compile, and rasterise ``model``'s diagram. Returns the PNG path."""
    arch = build_arch(tz, plotnn_dir, model, n_actions)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{model}_architecture.png"

    with tempfile.TemporaryDirectory(prefix="mariha_arch_") as tmp:
        tmp_dir = Path(tmp)
        tex_path = tmp_dir / f"{model}_architecture.tex"

        # tikzeng's to_generate() prints every emitted line — silence it.
        with contextlib.redirect_stdout(io.StringIO()):
            tz.to_generate(arch, str(tex_path))

        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(tmp_dir), str(tex_path)],
            cwd=tmp_dir, capture_output=True, text=True,
        )
        pdf_path = tmp_dir / f"{model}_architecture.pdf"
        if proc.returncode != 0 or not pdf_path.is_file():
            log = (tmp_dir / f"{model}_architecture.log")
            tail = log.read_text(errors="replace").splitlines()[-25:] if log.is_file() \
                else proc.stdout.splitlines()[-25:]
            sys.exit("ERROR: pdflatex failed:\n  " + "\n  ".join(tail))

        _pdf_to_png(pdf_path, png_path)

    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PlotNeuralNet-style PNG architecture diagram "
                    "for a MariHA model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, choices=(*SUPPORTED_MODELS, "all"),
        help="Model to diagram ('all' renders every supported model).",
    )
    parser.add_argument(
        "--n-actions", type=int, default=9,
        help="Size of the discrete action space (MariHA uses 9).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Directory for the generated PNG(s).",
    )
    parser.add_argument(
        "--plotneuralnet", type=Path, default=None,
        help="Path to a PlotNeuralNet checkout (auto-cloned if omitted).",
    )
    args = parser.parse_args()

    _require("pdflatex", "Install TeX Live, e.g. 'apt install texlive-latex-extra'.")

    plotnn_dir = ensure_plotneuralnet(args.plotneuralnet)
    sys.path.insert(0, str(plotnn_dir))
    from pycore import tikzeng as tz  # noqa: E402 — path set above

    models: Tuple[str, ...] = (
        SUPPORTED_MODELS if args.model == "all" else (args.model,)
    )
    for model in models:
        path = render(tz, plotnn_dir, model, args.n_actions, args.out_dir)
        print(f"[diagram] wrote {path}")


if __name__ == "__main__":
    main()
