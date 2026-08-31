"""Run the full Deep Research agent from the terminal, no Streamlit needed.

    python examples/research_cli.py "Compare vector databases for RAG in 2026"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import build_research_agent, ensure_report, run_config  # noqa: E402
from config import DEFAULT_DEPTH, DEFAULT_PROVIDER, REPORT_FILE  # noqa: E402
from exporter import export_to_pdf  # noqa: E402
from tracing import flush as flush_traces  # noqa: E402
from utils import find_file  # noqa: E402


def main() -> int:
    topic = " ".join(sys.argv[1:]).strip()
    if not topic:
        print("Usage: python examples/research_cli.py \"<research topic>\"")
        return 1

    agent = build_research_agent(DEFAULT_PROVIDER, DEFAULT_DEPTH)
    config = run_config(DEFAULT_DEPTH, thread_id="cli")

    for chunk in agent.stream({"messages": [{"role": "user", "content": topic}]}, config, stream_mode="updates"):
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            for message in update.get("messages", []) or []:
                for call in getattr(message, "tool_calls", None) or []:
                    print(f"  [{node}] {call['name']}")

    state = ensure_report(agent, config, agent.get_state(config).values)
    flush_traces()
    report = find_file(state.get("files", {}), REPORT_FILE)[1]
    if not report:
        print("\nThe agent did not produce a report.")
        print(state["messages"][-1].text)
        return 1

    Path("research_report.md").write_text(report, encoding="utf-8")
    export_to_pdf(report, "research_report.pdf")
    print("\nWrote research_report.md and research_report.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
