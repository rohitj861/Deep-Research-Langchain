"""Streamlit front end for the Deep Research deep agent."""

import os
import pathlib
import tempfile
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent import build_research_agent, ensure_report, run_config
from auth import require_password
from config import (
    DEFAULT_DEPTH,
    DEFAULT_PROVIDER,
    DEPTH_PRESETS,
    NOTES_DIR,
    PROVIDER_LABELS,
    Provider,
    QUESTION_FILE,
    REPORT_FILE,
    get_depth_preset,
    get_model_name,
    get_provider_spec,
    has_api_key,
    search_enabled,
)
from errors import explain
from ui import describe_tool_call, render_history_plan, render_todos
from exporter import export_to_pdf
from utils import list_files, read_file

st.set_page_config(page_title="Deep Research AI", page_icon="🧠", layout="wide")

def _init_state() -> None:
    st.session_state.setdefault("thread_id", uuid.uuid4().hex)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("files", {})
    st.session_state.setdefault("todos", [])
    # One checkpointer for the whole session. Rebuilding the agent (a provider or
    # depth change) must not silently wipe the thread it is still pointing at.
    st.session_state.setdefault("checkpointer", InMemorySaver())


def _reset_session() -> None:
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.history = []
    st.session_state.files = {}
    st.session_state.todos = []
    st.session_state.pop("agent", None)
    st.session_state.pop("agent_signature", None)
    st.session_state.checkpointer = InMemorySaver()


def _get_agent(provider: Provider, depth: str):
    """Cache the compiled agent per (provider, depth) so the thread's memory survives reruns."""
    signature = (provider.value, depth, get_model_name(provider))
    if st.session_state.get("agent_signature") != signature:
        st.session_state.agent = build_research_agent(
            provider, depth, checkpointer=st.session_state.checkpointer
        )
        st.session_state.agent_signature = signature
    return st.session_state.agent


def _stream_run(agent, question: str, depth: str, activity, plan_box) -> dict:
    """Stream one agent turn, rendering activity as it happens. Returns the final state."""
    config = run_config(depth, st.session_state.thread_id)
    lines: list[str] = []

    for chunk in agent.stream({"messages": [HumanMessage(content=question)]}, config, stream_mode="updates"):
        for update in chunk.values():
            if not isinstance(update, dict):
                continue

            if update.get("todos"):
                st.session_state.todos = update["todos"]
                render_todos(st.session_state.todos, plan_box)

            for message in update.get("messages", []) or []:
                if isinstance(message, AIMessage):
                    for call in message.tool_calls or []:
                        lines.append(describe_tool_call(call))
                elif isinstance(message, ToolMessage) and message.name == "task":
                    lines.append("↩️ Subagent finished and reported back")

            if lines:
                activity.markdown("\n\n".join(lines[-14:]))

    state = agent.get_state(config).values
    if not read_file(state.get("files", {}), REPORT_FILE):
        # The prompt makes the report mandatory, but a model can still shortcut a
        # question it judges trivial. One bounded follow-up turn closes that gap.
        lines.append("📄 Report missing — asking the agent to write it")
        activity.markdown("\n\n".join(lines[-14:]))
        state = ensure_report(agent, config, state)
    return state


@st.cache_data(show_spinner=False)
def _report_pdf(report: str) -> bytes:
    """Render the report to PDF once and cache it, keyed on the report text."""
    path = os.path.join(tempfile.gettempdir(), "research_report.pdf")
    export_to_pdf(report, path)
    return pathlib.Path(path).read_bytes()


def _final_text(state: dict) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            # `.text` flattens structured content blocks (Gemini returns those).
            return getattr(message, "text", None) or str(message.content)
    return ""


# Gate before any UI renders, so a locked app leaks nothing but the form.
if not require_password():
    st.stop()

_init_state()

with st.sidebar:
    st.header("Model")
    provider = st.radio(
        "Provider",
        list(Provider),
        index=list(Provider).index(DEFAULT_PROVIDER),
        format_func=lambda p: PROVIDER_LABELS.get(p, p.value),
    )
    spec = get_provider_spec(provider)
    st.caption(f"Model spec: `{spec.lc_provider}:{get_model_name(provider)}`")

    if has_api_key(provider):
        st.success(f"{spec.key_envs[0]} found", icon="✅")
    else:
        st.error(f"Set `{spec.key_envs[0]}` in `.env`", icon="🔑")
        st.caption(f"[Get a key]({spec.console_url})")

    st.header("Depth")
    depth = st.selectbox(
        "Research depth",
        list(DEPTH_PRESETS),
        index=list(DEPTH_PRESETS).index(DEFAULT_DEPTH if DEFAULT_DEPTH in DEPTH_PRESETS else "Standard"),
    )
    st.caption(get_depth_preset(depth).description)

    st.header("Web search")
    if search_enabled():
        st.success("Tavily search enabled", icon="🌐")
    else:
        st.warning("No `TAVILY_API_KEY` — the agent will answer from model knowledge only.", icon="⚠️")

    st.divider()
    if st.button("New session", use_container_width=True):
        _reset_session()
        st.rerun()

st.title("Deep Research AI")
st.caption("A LangChain **deep agent**: planning todos, a virtual filesystem, and research subagents.")

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        if entry.get("content"):
            st.markdown(entry["content"])
        render_history_plan(entry.get("todos", []))

question = st.chat_input("Ask a research question, or follow up on the last report...")

if question:
    if not has_api_key(provider):
        st.error(f"No API key for {spec.label}. Add `{spec.key_envs[0]}` to your `.env` and restart.")
        st.stop()

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    agent = _get_agent(provider, depth)

    with st.chat_message("assistant"):
        # st.empty() (not st.container()) — the agent calls write_todos several
        # times per run, and each render must REPLACE the plan, not stack another
        # copy underneath it.
        plan_box = st.empty()
        with st.status("Researching...", expanded=True) as status:
            activity = st.empty()
            try:
                state = _stream_run(agent, question, depth, activity, plan_box)
            except Exception as exc:  # surfaced in the UI rather than as a stack trace
                status.update(label="Run failed", state="error")
                headline, next_step = explain(exc)
                st.error(headline)
                st.info(next_step)
                st.stop()
            status.update(label="Research complete", state="complete")

        st.session_state.files = state.get("files", {}) or {}
        st.session_state.todos = state.get("todos", []) or []

        summary = _final_text(state)
        if summary:
            st.markdown(summary)
        # Keep the turn even without a summary, so its plan is still recoverable.
        if summary or st.session_state.todos:
            st.session_state.history.append(
                {"role": "assistant", "content": summary, "todos": st.session_state.todos}
            )

files = st.session_state.files
report = read_file(files, REPORT_FILE)

if files:
    st.divider()
    report_tab, notes_tab = st.tabs(["📄 Report", "🗂️ Workspace"])

    with report_tab:
        if report:
            st.markdown(report)
            left, right = st.columns(2)
            left.download_button(
                "Download Markdown",
                data=report.encode("utf-8"),
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
            right.download_button(
                "Download PDF",
                data=_report_pdf(report),
                file_name="research_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info(f"The agent has not written `{REPORT_FILE}` yet.")

    with notes_tab:
        st.caption("The agent's virtual filesystem — notes live here instead of in the context window.")
        note_paths = list_files(files, exclude=(REPORT_FILE,))
        for path in note_paths:
            icon = "📌" if path.lstrip("/") == QUESTION_FILE else "📝"
            with st.expander(f"{icon} {path}"):
                st.markdown(read_file(files, path))
        if not note_paths:
            st.info(f"No notes under `{NOTES_DIR}/` yet.")
