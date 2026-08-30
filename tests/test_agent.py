import os
import unittest
from unittest.mock import patch

from agent import FILESYSTEM_TOOLS, build_research_agent, build_subagents, run_config
from config import DEPTH_PRESETS


def _tool_names(agent) -> set[str]:
    return set(agent.nodes["tools"].bound.tools_by_name)


class SubagentTests(unittest.TestCase):
    def test_subagents_declare_required_keys(self):
        for subagent in build_subagents(3, []):
            self.assertTrue({"name", "description", "system_prompt"} <= set(subagent))

    def test_research_agent_receives_the_research_tools(self):
        sentinel = object()
        research = next(sa for sa in build_subagents(3, [sentinel]) if sa["name"] == "research-agent")
        self.assertIn(sentinel, research["tools"])

    def test_critique_agent_has_no_extra_tools(self):
        critique = next(sa for sa in build_subagents(3, [object()]) if sa["name"] == "critique-agent")
        self.assertEqual(critique["tools"], [])


class AgentAssemblyTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key", "TAVILY_API_KEY": ""}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_agent_exposes_planning_filesystem_and_delegation_tools(self):
        names = _tool_names(build_research_agent("gemini", "Standard"))
        self.assertIn("write_todos", names)   # planning
        self.assertIn("task", names)          # subagent delegation
        self.assertIn("get_weather", names)   # custom tool
        self.assertTrue(set(FILESYSTEM_TOOLS) <= names)

    def test_shell_execution_is_not_exposed(self):
        names = _tool_names(build_research_agent("gemini", "Basic"))
        self.assertNotIn("execute", names)
        self.assertNotIn("delete", names)

    def test_state_tracks_files_and_todos(self):
        agent = build_research_agent("gemini", "Basic")
        keys = set(agent.get_output_jsonschema()["properties"])
        self.assertTrue({"files", "todos", "messages"} <= keys)

    def test_run_config_carries_depth_budget_and_thread(self):
        config = run_config("Advanced", thread_id="abc")
        self.assertEqual(config["recursion_limit"], DEPTH_PRESETS["Advanced"].recursion_limit)
        self.assertEqual(config["configurable"]["thread_id"], "abc")


if __name__ == "__main__":
    unittest.main()
