"""MkDocs hook: copy tutorial notebooks from examples/ into docs/tutorials/.

The runnable notebooks in examples/ are the source of truth for the tutorials.
Before each build (local or CI), published notebooks matching ``NN_*.ipynb``
are copied into docs/tutorials/ where mkdocs-jupyter renders them. Notebooks
prefixed with an underscore (``_dev_*.ipynb``) are development-only and are
not copied. The copies are gitignored; only the examples/ originals are
tracked.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("mkdocs.hooks.copy_tutorials")


def on_pre_build(config) -> None:
    """Copy published example notebooks into the docs tutorials directory.

    Parameters
    ----------
    config : mkdocs.config.defaults.MkDocsConfig
        The mkdocs configuration for the current build.
    """
    docs_dir = Path(config["docs_dir"])
    examples_dir = docs_dir.parent / "examples"
    dest_dir = docs_dir / "tutorials"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for notebook in sorted(examples_dir.glob("[0-9][0-9]_*.ipynb")):
        target = dest_dir / notebook.name
        # mtime guard: avoid rewriting unchanged files, which would trigger
        # rebuild loops under `mkdocs serve`
        if not target.exists() or notebook.stat().st_mtime > target.stat().st_mtime:
            shutil.copy2(notebook, target)
            logger.info("Copied %s to %s", notebook.name, dest_dir)
