"""Helpers for reading the deep agent's virtual filesystem (`state["files"]`).

The state backend normalizes paths to absolute form, so a file the prompt calls
`final_report.md` comes back keyed as `/final_report.md`. These helpers look up
either spelling and unwrap whichever record shape the backend stored.
"""

from typing import Any


def _content(entry: Any) -> str:
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("content", "text", "data"):
            if key in entry:
                return str(entry[key])
    return str(entry)


def normalize(path: str) -> str:
    return "/" + path.lstrip("/")


def read_file(files: dict, path: str) -> str:
    """Read one file from agent state, tolerating leading-slash differences."""
    if not files:
        return ""
    for candidate in (path, normalize(path), path.lstrip("/")):
        if candidate in files:
            return _content(files[candidate])
    return ""


def list_files(files: dict, exclude: tuple[str, ...] = ()) -> list[str]:
    """Sorted paths in agent state, minus the ones named in `exclude`."""
    skip = {normalize(path) for path in exclude}
    return sorted(path for path in (files or {}) if normalize(path) not in skip)
