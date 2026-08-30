"""End-to-end run of the compiled graph against a scripted model.

Exercises the parts that only show up when the graph actually executes: the
planning tool updating state, `task` dispatching into a subagent, and files a
subagent writes propagating back into the parent's state.
"""

import os
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent import build_research_agent, run_config
from config import QUESTION_FILE, REPORT_FILE
from utils import read_file

REPORT = "# Test Report\n\n## Executive Summary\nEverything checks out.\n"


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed message script; accepts tool binding without using it."""

    def bind_tools(self, tools, **kwargs):
        return self


def _script():
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {
                            "todos": [
                                {"content": "Investigate the topic", "status": "in_progress"},
                                {"content": "Write the report", "status": "pending"},
                            ]
                        },
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "write_file", "args": {"file_path": QUESTION_FILE, "content": "The brief"}, "id": "c2"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "Investigate the topic", "subagent_type": "research-agent"},
                        "id": "c3",
                    }
                ],
            ),
            # --- inside the research-agent's own context window ---
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "write_file", "args": {"file_path": "notes/topic.md", "content": "findings"}, "id": "s1"}
                ],
            ),
            AIMessage(content="Wrote notes/topic.md"),
            # --- back in the orchestrator ---
            AIMessage(
                content="",
                tool_calls=[{"name": "write_file", "args": {"file_path": REPORT_FILE, "content": REPORT}, "id": "c4"}],
            ),
            AIMessage(content="Done. The full report is in final_report.md."),
        ]
    )


class AgentFlowTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.agent = build_research_agent(model=ScriptedModel(messages=_script()), depth="Basic")
        self.config = run_config("Basic", thread_id="flow-test")
        self.updates = list(
            self.agent.stream({"messages": [{"role": "user", "content": "topic"}]}, self.config, stream_mode="updates")
        )
        self.state = self.agent.get_state(self.config).values

    def _streamed_tool_calls(self) -> list[str]:
        names = []
        for chunk in self.updates:
            for update in chunk.values():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", []) or []:
                    for call in getattr(message, "tool_calls", None) or []:
                        names.append(call["name"])
        return names

    def test_orchestrator_tool_calls_reach_the_update_stream(self):
        # This is what the Streamlit activity log renders from.
        self.assertEqual(self._streamed_tool_calls(), ["write_todos", "write_file", "task", "write_file"])

    def test_plan_lands_in_state(self):
        self.assertEqual(
            [todo["content"] for todo in self.state["todos"]],
            ["Investigate the topic", "Write the report"],
        )

    def test_subagent_files_propagate_to_parent_state(self):
        self.assertEqual(read_file(self.state["files"], "notes/topic.md"), "findings")

    def test_report_is_written_to_the_virtual_filesystem(self):
        self.assertEqual(read_file(self.state["files"], REPORT_FILE), REPORT)

    def test_final_message_is_a_summary_not_the_report(self):
        final = self.state["messages"][-1]
        self.assertIn("final_report.md", final.content)
        self.assertNotIn("Executive Summary", final.content)


if __name__ == "__main__":
    unittest.main()
