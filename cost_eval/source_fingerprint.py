"""Stable source-tree fingerprints used by frozen predictions and profiles."""

from __future__ import annotations

import hashlib
from pathlib import Path


def python_source_tree_sha256(root) -> str:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
