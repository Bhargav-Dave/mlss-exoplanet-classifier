"""Small shared utilities."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def repo_root_from_script(script_file: str | Path) -> Path:
    """Return repo root for scripts stored one level below the root."""
    return Path(script_file).resolve().parents[1]
