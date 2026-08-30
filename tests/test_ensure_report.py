"""The report is guaranteed, not merely requested.

`prompts.py` makes the research workflow mandatory, but a model can still answer a
question it judges trivial straight into chat. `ensure_report` is the backstop: one
bounded follow-up turn on the same thread, so the agent still has its notes.
"""

import os
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent import build_research_agent, ensure_report, run_config
from config import REPORT_FILE
from utils import read_file

REPORT = "# Report\n\n## Executive Summary\nWritten after the nudge.\n"


class _Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class EnsureReportTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, script, thread):
        agent = build_research_agent(model=_Scripted(messages=iter(script)), depth="Basic")
        config = run_config("Basic", thread_id=thread)
        agent.invoke({"messages": [{"role": "user", "content": "a question"}]}, config)
        return agent, config, agent.get_state(config).values

    def test_shortcut_answer_still_ends_with_a_report(self):
        # Turn 1 answers directly; the nudge turn then writes the file.
        agent, config, state = self._run(
            [
                AIMessage(content="LangGraph is a graph runtime. Done."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "write_file", "args": {"file_path": REPORT_FILE, "content": REPORT}, "id": "n1"}
                    ],
                ),
                AIMessage(content="Report written."),
            ],
            thread="shortcut",
        )
        self.assertEqual(read_file(state.get("files", {}), REPORT_FILE), "")  # nothing yet

        after = ensure_report(agent, config, state)
        self.assertEqual(read_file(after["files"], REPORT_FILE), REPORT)

    def test_existing_report_is_left_alone(self):
        agent, config, state = self._run(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "write_file", "args": {"file_path": REPORT_FILE, "content": REPORT}, "id": "w1"}
                    ],
                ),
                AIMessage(content="Done."),
            ],
            thread="already-there",
        )
        before = len(state["messages"])

        after = ensure_report(agent, config, state)
        # Same state object back, no extra model turn spent.
        self.assertEqual(len(after["messages"]), before)
        self.assertEqual(read_file(after["files"], REPORT_FILE), REPORT)

    def test_nudge_runs_on_the_same_thread_so_notes_survive(self):
        agent, config, state = self._run(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "write_file", "args": {"file_path": "notes/a.md", "content": "findings"}, "id": "s1"}
                    ],
                ),
                AIMessage(content="Answered directly, no report."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "write_file", "args": {"file_path": REPORT_FILE, "content": REPORT}, "id": "n1"}
                    ],
                ),
                AIMessage(content="Report written."),
            ],
            thread="keeps-notes",
        )
        after = ensure_report(agent, config, state)
        self.assertEqual(read_file(after["files"], "notes/a.md"), "findings")
        self.assertEqual(read_file(after["files"], REPORT_FILE), REPORT)


class MandatoryWorkflowPromptTests(unittest.TestCase):
    def test_prompt_forbids_answering_without_a_report(self):
        from prompts import orchestrator_prompt

        text = orchestrator_prompt(4)
        self.assertIn("Non-negotiable", text)
        self.assertIn(REPORT_FILE, text)
        for phrase in ("even when the question looks simple", "never no report"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
