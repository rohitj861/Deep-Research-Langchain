"""Helpers for reading the deep agent's virtual filesystem (`state["files"]`).

The state backend normalizes paths to absolute form, so a file the prompt calls
`final_report.md` comes back keyed as `/final_report.md`. These helpers look up
either spelling and unwrap whichever record shape the backend stored.

Models also improvise: a report asked for at `final_report.md` may land at
`/reports/final_report.md`, `/Final_Report.md`, or `/final-report.md`. An exact
lookup calls that "missing" while the file sits right there, so `find_file`
widens the search before giving up.
"""

import posixpath
import re
from typing import Any

_NORMALIZE = re.compile(r"[^a-z0-9]+")


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


def _stem_key(path: str) -> str:
    """Comparable form of a filename: lowercase, punctuation stripped, no extension."""
    stem = posixpath.splitext(posixpath.basename(path))[0]
    return _NORMALIZE.sub("", stem.lower())


def read_file(files: dict, path: str) -> str:
    """Read one file by exact path, tolerating leading-slash differences."""
    if not files:
        return ""
    for candidate in (path, normalize(path), path.lstrip("/")):
        if candidate in files:
            return _content(files[candidate])
    return ""


def find_file(files: dict, name: str) -> tuple[str, str]:
    """Locate a file by name, tolerating the directory and spelling the agent used.

    Returns `(path, content)`, or `("", "")` if nothing matches. Tried in order:
    the exact path, the same basename in any directory, then the same basename
    ignoring case, punctuation, and extension.
    """
    if not files:
        return "", ""

    for candidate in (name, normalize(name), name.lstrip("/")):
        if candidate in files:
            return candidate, _content(files[candidate])

    target = posixpath.basename(name).lower()
    for path in sorted(files):
        if posixpath.basename(path).lower() == target:
            return path, _content(files[path])

    target_stem = _stem_key(name)
    for path in sorted(files):
        if _stem_key(path) == target_stem:
            return path, _content(files[path])

    return "", ""


def list_files(files: dict, exclude: tuple[str, ...] = ()) -> list[str]:
    """Sorted paths in agent state, minus the ones named in `exclude`."""
    skip = {normalize(path) for path in exclude}
    return sorted(path for path in (files or {}) if normalize(path) not in skip)
