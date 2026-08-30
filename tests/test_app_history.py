"""Each turn keeps the plan that produced it.

Before this, `todos` lived in one session-wide slot that the next run overwrote,
so an earlier turn's plan became unrecoverable the moment you asked a follow-up.
"""

import os
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
import agent as agent_mod
from config import REPORT_FILE
from tests._harness import app_test

class _Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _turn(tag, steps):
    """A realistic turn: plan, write the report, summarize.

    The report write matters — `ensure_report` nudges any turn that ends without
    one, which would consume the next turn's scripted messages.
    """
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": s, "status": "completed"} for s in steps]},
                    "id": tag,
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": REPORT_FILE, "content": f"# Report {tag}\n"},
                    "id": tag + "-w",
                }
            ],
        ),
        AIMessage(content=f"Summary {tag}."),
    ]


class HistoryPlanTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

        real = agent_mod.build_research_agent
        script = iter(_turn("A", ["A: origins", "A: adoption"]) + _turn("B", ["B: pricing"]))

        def fake(*args, **kwargs):
            return real(model=_Scripted(messages=script), depth="Basic", checkpointer=kwargs.get("checkpointer"))

        agent_patch = patch.object(agent_mod, "build_research_agent", fake)
        agent_patch.start()
        self.addCleanup(agent_patch.stop)

        ctx = app_test()  # unlocked
        self.at = ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.at.run()
        self.at.chat_input[0].set_value("first question").run()
        self.at.chat_input[0].set_value("second question").run()

    def _assistant_entries(self):
        return [e for e in self.at.session_state["history"] if e["role"] == "assistant"]

    def test_run_completes_without_error(self):
        self.assertEqual([str(e.value) for e in self.at.exception], [])

    def test_each_turn_stores_its_own_plan(self):
        plans = [[t["content"] for t in e.get("todos", [])] for e in self._assistant_entries()]
        self.assertEqual(plans, [["A: origins", "A: adoption"], ["B: pricing"]])

    def test_earlier_plan_survives_a_follow_up(self):
        first = self._assistant_entries()[0]
        self.assertIn("A: origins", [t["content"] for t in first["todos"]])

    def test_no_duplicate_rendering_across_turns(self):
        values = [m.value for m in self.at.markdown]
        for needle in ("first question", "second question", "Summary A.", "Summary B."):
            self.assertEqual(sum(1 for v in values if needle in v), 1, f"{needle!r} rendered more than once")

    def test_plan_tab_is_gone(self):
        # Two tabs now: Report and Workspace. The plan lives with its turn.
        labels = [t.label for t in self.at.tabs] if hasattr(self.at, "tabs") else []
        self.assertNotIn("✅ Plan", labels)


if __name__ == "__main__":
    unittest.main()
