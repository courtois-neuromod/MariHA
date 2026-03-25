"""Sphinx configuration for MariHA documentation."""

import os
import sys

# Make the mariha package importable during doc build.
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

project = "MariHA"
copyright = "2026, NeuroMod"
author = "NeuroMod"
release = "0.1.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",          # pull docstrings into API pages
    "sphinx.ext.napoleon",         # Google/NumPy docstring styles
    "sphinx.ext.viewcode",         # [source] links next to every symbol
    "sphinx.ext.intersphinx",      # cross-reference Python / NumPy docs
    "sphinx.ext.autosummary",      # summary tables in API pages
    "sphinx_autodoc_typehints",    # render type hints in descriptions
    "sphinx_copybutton",           # copy-button on code blocks
    "myst_parser",                 # optional: write .md files in Sphinx
]

# ---------------------------------------------------------------------------
# Autodoc settings
# ---------------------------------------------------------------------------

# Heavy runtime dependencies are not installed in the RTD build environment.
# Mock them so autodoc can import mariha modules without errors.
autodoc_mock_imports = [
    "tensorflow",
    "tf_keras",
    "keras",
    "retro",
    "cv2",
    "gymnasium",
    "gym",
    "numpy",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstyle = True
napoleon_numpy_docstyle = True
napoleon_include_init_with_doc = True

autosummary_generate = True

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# ---------------------------------------------------------------------------
# HTML output — Furo theme
# ---------------------------------------------------------------------------

html_theme = "furo"
html_title = "MariHA"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/courtois-neuromod/mariha",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_static_path = ["_static"]

# ---------------------------------------------------------------------------
# Source file suffixes
# ---------------------------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
