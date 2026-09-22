"""Repository path resolution shared by entry points and scripts.

Layout in this repo::

    FieldLens/                 <- REPO_ROOT (data/, runs/, checkpoints/, dashboard/)
      training/                <- CODE_ROOT (train.py, configs/, fieldlens/)
        fieldlens/paths.py
      data/
      dashboard/

Set ``FIELDLENS_ROOT`` to keep data/outputs somewhere other than the checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

# fieldlens/paths.py -> fieldlens/ -> training/
PACKAGE_ROOT = Path(__file__).resolve().parent
CODE_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = CODE_ROOT / "configs"

# Data and outputs live at the FieldLens repo root (parent of training/),
# unless FIELDLENS_ROOT overrides that.
REPO_ROOT = Path(os.environ.get("FIELDLENS_ROOT", CODE_ROOT.parent)).expanduser().resolve()


def repo_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a config path: absolute paths are kept, relative ones join REPO_ROOT."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def config_path(name: str) -> Path:
    """Path to a YAML file in ``configs/`` (under training/, not the data root)."""
    return CONFIG_DIR / name
