"""Presentation helpers for the Streamlit app.

Kept out of `app.py` so tests can import them without executing the script —
importing `app` would run the password gate, the sidebar, and everything else.
"""

import streamlit as st

TOOL_LABELS = {
    "write_todos": "📝 Updating the plan",
    "task": "🚀 Delegating to subagent",
    "tavily_search": "🔎 Searching the web",
    "write_file": "💾 Writing",
    "edit_file": "✏️ Editing",
    "read_file": "📖 Reading",
    "ls": "📂 Listing files",
    "glob": "📂 Finding files",
    "grep": "🔍 Searching notes",
    "get_weather": "🌤️ Checking weather",
}

STATUS_ICONS = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}


def shorten(text: str, limit: int) -> str:
    """Trim to `limit` characters on a word boundary, marking that it was cut.

    A hard slice ends lines mid-word ("...what are the p"), which reads like the
    agent produced a truncated instruction rather than the log abbreviating it.
    """
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    head = collapsed[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.-")
    return f"{head or collapsed[:limit]}…"


def describe_tool_call(call: dict) -> str:
    """One activity-log line for a tool call."""
    name = call.get("name", "tool")
    args = call.get("args", {}) or {}
    label = TOOL_LABELS.get(name, f"🔧 {name}")

    if name == "task":
        return f"{label} `{args.get('subagent_type', '?')}` — {shorten(args.get('description', ''), 120)}"
    if name in {"write_file", "edit_file", "read_file"}:
        return f"{label} `{args.get('file_path', '?')}`"
    if name == "tavily_search":
        return f"{label}: _{shorten(args.get('query', ''), 120)}_"
    if name == "write_todos":
        return label
    if args:
        first = next(iter(args.values()))
        return f"{label} `{shorten(first, 100)}`"
    return label


def todo_markdown(todos: list[dict]) -> str:
    return "\n\n".join(
        f"{STATUS_ICONS.get(todo.get('status'), '⬜')} {todo.get('content', '')}" for todo in todos
    )


def render_todos(todos: list[dict], slot) -> None:
    """Draw the live plan. `slot` must be an st.empty() so each update replaces the last."""
    if not todos:
        return
    slot.markdown("**Research plan**\n\n" + todo_markdown(todos))


def render_history_plan(todos: list[dict]) -> None:
    """Replay a finished turn's plan, collapsed so it costs one line until opened."""
    if not todos:
        return
    with st.expander(f"Research plan ({len(todos)} steps)"):
        st.markdown(todo_markdown(todos))
