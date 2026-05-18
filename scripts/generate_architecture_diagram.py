"""Generate a PlotNeuralNet-style PNG architecture *panel* for a MariHA model.

Follows ``courtois-neuromod/mario.tutorials``'s ``generate_architecture_diagram.py``:
the network is described with PlotNeuralNet's ``tikzeng`` primitives, emitted
as a LaTeX/TikZ file, compiled with ``pdflatex``, and rasterised to PNG.

    architecture spec  ->  tikzeng  ->  .tex  --(pdflatex)-->  .pdf  --(PyMuPDF)-->  .png

Unlike a single-network diagram, each agent is drawn as a *panel*: one stacked
PlotNeuralNet diagram per network the agent instantiates. Where one network is
a Polyak copy of another (a target network), a text note in the gap between
the stacks records that — the target update is a whole-network weight copy,
not a layer-to-layer connection, so it is deliberately not drawn as an arrow.
This matters because, e.g., SAC builds five independent networks (actor + twin
critics + twin target critics), not one.

The topology comes from ``scripts/architecture_specs.py`` — a hand-maintained
registry kept honest by ``scripts/tests/test_architecture_specs.py``.

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
from typing import Dict, List, Tuple

try:  # installed package / `python -m`
    from scripts.architecture_specs import AgentArch, Link, Network, get_spec
except ImportError:  # run directly: python scripts/generate_architecture_diagram.py
    from architecture_specs import AgentArch, Link, Network, get_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "assets" / "model_architectures"
PLOTNN_URL = "https://github.com/HarisIqbal88/PlotNeuralNet.git"

# ``ddqn`` shares DQN's network; ``random`` has no network and is omitted.
SUPPORTED_MODELS = ("sac", "ppo", "dqn", "ddqn")

# --- Layout constants (PlotNeuralNet units) --------------------------------
# Vertical spacing between stacked network diagrams. Tuned so stacks sit
# close without overlapping; override per-run with --row-gap.
ROW_GAP_DEFAULT = 12.0
# Backbone box geometry: (zlabel spatial size, box height==depth). The input's
# height is the tallest box and drives the minimum safe ROW_GAP.
_INPUT_H = 28
_CONV_GEOM = [(42, 24), (21, 20), (11, 14), (6, 9)]  # conv1..conv4


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
# TikZ helpers
# ---------------------------------------------------------------------------


def _node(net_id: str, suffix: str) -> str:
    """Return a globally-unique, TikZ-safe node name for ``net_id``'s box.

    Underscores and other punctuation are stripped because they are unsafe in
    TikZ node names; the network id only needs to stay *unique*, not readable.
    """
    safe = "".join(ch for ch in net_id if ch.isalnum())
    return f"{safe}{suffix}"


def _tex(text: str) -> str:
    """Escape LaTeX-special characters in display text."""
    repl = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}
    return "".join(repl.get(ch, ch) for ch in text)


def _cap(text: str) -> str:
    """Brace-wrap a caption so multi-word values survive pgfkeys parsing."""
    return "{" + _tex(text) + "}"


# ---------------------------------------------------------------------------
# Per-network and per-link rendering
# ---------------------------------------------------------------------------


def _network_layers(tz, net: Network, row: int, row_gap: float) -> List[str]:
    """Return the tikzeng layer list for one network stacked at ``row``.

    The shared ``BaseCNN`` backbone (input -> 4 conv blocks -> Flatten ->
    Dense 512 -> Concatenate task one-hot) is drawn for *every* network — the
    networks genuinely do not share weights — followed by the network's head(s).
    """
    nid = lambda s: _node(net.id, s)  # noqa: E731 — terse local alias
    origin = f"(0,{-row * row_gap:.2f},0)"  # absolute origin for this stack

    layers = [
        tz.to_Conv(nid("input"), 84, 4, offset="(0,0,0)", to=origin,
                   height=_INPUT_H, depth=_INPUT_H, width=1,
                   caption=_cap("Input 84x84x4")),
    ]
    prev = "input"
    for i, (s_filer, hw) in enumerate(_CONV_GEOM, start=1):
        layers.append(
            tz.to_Conv(nid(f"conv{i}"), s_filer, 32, offset="(2.7,0,0)",
                       to=f"({nid(prev)}-east)", height=hw, depth=hw, width=3,
                       caption=_cap("Conv 3x3 s2")))
        layers.append(tz.to_connection(nid(prev), nid(f"conv{i}")))
        prev = f"conv{i}"

    layers += [
        tz.to_SoftMax(nid("flatten"), 1152, offset="(2.7,0,0)",
                      to=f"({nid('conv4')}-east)", width=1.5, height=6, depth=6,
                      opacity=0.6, caption=_cap("Flatten")),
        tz.to_connection(nid("conv4"), nid("flatten")),
        tz.to_SoftMax(nid("dense"), 512, offset="(2.7,0,0)",
                      to=f"({nid('flatten')}-east)", width=2, height=10, depth=10,
                      opacity=0.7, caption=_cap("Dense ReLU")),
        tz.to_connection(nid("flatten"), nid("dense")),
        # Task one-hot sits just above the dense projection and feeds the
        # concat. Kept small and low so it reads as part of the network.
        tz.to_SoftMax(nid("task"), "", offset="(2.7,4,0)",
                      to=f"({nid('flatten')}-east)", width=1.5, height=3, depth=3,
                      opacity=0.7, caption=_cap("Task one-hot")),
        tz.to_SoftMax(nid("concat"), "", offset="(3,0,0)",
                      to=f"({nid('dense')}-east)", width=2.2, height=11, depth=11,
                      opacity=0.7, caption=_cap("Concat 512+task")),
        tz.to_connection(nid("dense"), nid("concat")),
        tz.to_connection(nid("task"), nid("concat")),
    ]

    # Output heads branch off the concat node: one head straight, two heads
    # fan out up/down (PPO's actor + critic).
    offsets = ["(3.4,0,0)"] if len(net.heads) == 1 else ["(3.6,3,0)", "(3.6,-3,0)"]
    for head, offset in zip(net.heads, offsets):
        layers.append(
            tz.to_SoftMax(nid(head.id), head.units, offset=offset,
                          to=f"({nid('concat')}-east)", width=1.5, height=5,
                          depth=5, opacity=0.9, caption=_cap(head.label)))
        layers.append(tz.to_connection(nid("concat"), nid(head.id)))

    return layers


def _annotation_text(link: Link, by_id: Dict[str, Network]) -> str:
    """Build the note describing a Polyak target-network relationship."""
    src = _tex(by_id[link.src].title)
    dst = _tex(by_id[link.dst].title)
    # `\\\\` in this source emits a literal `\\` — a TikZ line break.
    return (f"({dst} = periodic Polyak copy\\\\"
            f"of {src}, all weights)")


def _annotation_layer(link: Link, by_id: Dict[str, Network],
                      rows: Dict[str, int], row_gap: float) -> str:
    """Render a Link as an italic note centred in the gap between two stacks.

    A target update copies the *entire* network's weights, so it is shown as
    a note — not an arrow piercing a single layer box.
    """
    y = -(rows[link.src] + 0.5) * row_gap  # midpoint between the two stacks
    text = _annotation_text(link, by_id)
    return (
        "\n\\node[anchor=west, align=left, text=black!70, "
        "font=\\sffamily\\small\\itshape] "
        f"at (0,{y:.2f},0) {{{text}}};\n"
    )


def _network_title(net: Network) -> str:
    """Return a raw-TikZ label node placed to the left of a network stack."""
    box = _node(net.id, "input")
    return (
        "\n\\node[anchor=east, font=\\sffamily\\bfseries\\large] at "
        f"([xshift=-1.6cm]{box}-west) {_cap(net.title)};\n"
    )


def _panel_title(arch: AgentArch) -> str:
    """Return a raw-TikZ title node placed above the first network stack."""
    box = _node(arch.networks[0].id, "input")
    return (
        "\n\\node[anchor=south, font=\\sffamily\\bfseries\\LARGE] at "
        f"([yshift=1.5cm]{box}-north) {_cap(arch.title)};\n"
    )


def build_arch(tz, plotnn_dir: Path, arch: AgentArch, row_gap: float) -> List[str]:
    """Assemble the full tikzeng layer list for an agent's panel."""
    rows = {net.id: i for i, net in enumerate(arch.networks)}
    by_id = {net.id: net for net in arch.networks}

    layers: List[str] = [tz.to_head(str(plotnn_dir)), tz.to_cor(), tz.to_begin()]
    for row, net in enumerate(arch.networks):
        layers += _network_layers(tz, net, row, row_gap)
    # Titles and annotations reference box anchors / coordinates, so they
    # come after all boxes are placed.
    layers += [_network_title(net) for net in arch.networks]
    layers += [_annotation_layer(link, by_id, rows, row_gap)
               for link in arch.links]
    layers.append(_panel_title(arch))
    layers.append(tz.to_end())
    return layers


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


def render(tz, plotnn_dir: Path, model: str, n_actions: int,
           out_dir: Path, row_gap: float) -> Path:
    """Build, compile, and rasterise ``model``'s panel. Returns the PNG path."""
    arch = get_spec(model, n_actions)
    layers = build_arch(tz, plotnn_dir, arch, row_gap)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{model}_architecture.png"

    with tempfile.TemporaryDirectory(prefix="mariha_arch_") as tmp:
        tmp_dir = Path(tmp)
        tex_path = tmp_dir / f"{model}_architecture.tex"

        # tikzeng's to_generate() prints every emitted line — silence it.
        with contextlib.redirect_stdout(io.StringIO()):
            tz.to_generate(layers, str(tex_path))

        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(tmp_dir), str(tex_path)],
            cwd=tmp_dir, capture_output=True, text=True,
        )
        pdf_path = tmp_dir / f"{model}_architecture.pdf"
        if proc.returncode != 0 or not pdf_path.is_file():
            log = tmp_dir / f"{model}_architecture.log"
            tail = (log.read_text(errors="replace").splitlines()[-25:]
                    if log.is_file() else proc.stdout.splitlines()[-25:])
            sys.exit("ERROR: pdflatex failed:\n  " + "\n  ".join(tail))

        _pdf_to_png(pdf_path, png_path)

    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PlotNeuralNet-style PNG architecture panel "
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
        "--row-gap", type=float, default=ROW_GAP_DEFAULT,
        help="Vertical spacing between stacked networks. Lower = more compact.",
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
        path = render(tz, plotnn_dir, model, args.n_actions,
                       args.out_dir, args.row_gap)
        print(f"[diagram] wrote {path}")


if __name__ == "__main__":
    main()
