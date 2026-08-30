"""Rendering regressions in the Streamlit layer.

These import `ui`, never `app` — importing the script would execute the password
gate and the whole page as a side effect.

A real run calls `write_todos` several times (initial plan, then a status update
per completed step). Each of those pushes a fresh render of the plan, so the
element the plan is drawn into must REPLACE its contents, not append to them.
`st.container()` appends; `st.empty()` replaces.
"""

import unittest
from unittest.mock import MagicMock

import ui


class TodoRenderTargetTests(unittest.TestCase):
    def test_repeated_renders_write_to_one_replaceable_slot(self):
        slot = MagicMock()
        for status in ("pending", "in_progress", "completed"):
            ui.render_todos([{"content": "Investigate X", "status": status}], slot)

        # Three updates, three writes into the same slot. With st.empty() each one
        # replaces the last; with st.container() they would stack up on the page.
        self.assertEqual(slot.markdown.call_count, 3)

    def test_empty_todo_list_renders_nothing(self):
        slot = MagicMock()
        ui.render_todos([], slot)
        slot.markdown.assert_not_called()

    def test_render_marks_each_status_with_its_own_icon(self):
        slot = MagicMock()
        ui.render_todos(
            [
                {"content": "Done step", "status": "completed"},
                {"content": "Active step", "status": "in_progress"},
                {"content": "Later step", "status": "pending"},
            ],
            slot,
        )
        rendered = slot.markdown.call_args[0][0]
        self.assertIn("✅ Done step", rendered)
        self.assertIn("🔄 Active step", rendered)
        self.assertIn("⬜ Later step", rendered)

    def test_unknown_status_falls_back_to_the_pending_icon(self):
        slot = MagicMock()
        ui.render_todos([{"content": "Odd", "status": "bogus"}], slot)
        self.assertIn("⬜ Odd", slot.markdown.call_args[0][0])


class TodoMarkdownTests(unittest.TestCase):
    def test_one_line_per_todo(self):
        rendered = ui.todo_markdown([{"content": "A", "status": "pending"},
                                       {"content": "B", "status": "completed"}])
        self.assertIn("⬜ A", rendered)
        self.assertIn("✅ B", rendered)

    def test_empty_list_renders_empty_string(self):
        self.assertEqual(ui.todo_markdown([]), "")


class ToolCallDescriptionTests(unittest.TestCase):
    def test_delegation_names_the_subagent(self):
        line = ui.describe_tool_call(
            {"name": "task", "args": {"subagent_type": "research-agent", "description": "Look into X"}}
        )
        self.assertIn("research-agent", line)
        self.assertIn("Look into X", line)

    def test_file_writes_name_the_path(self):
        line = ui.describe_tool_call({"name": "write_file", "args": {"file_path": "notes/x.md"}})
        self.assertIn("notes/x.md", line)

    def test_search_shows_the_query(self):
        line = ui.describe_tool_call({"name": "tavily_search", "args": {"query": "what is RAG"}})
        self.assertIn("what is RAG", line)

    def test_long_arguments_are_truncated(self):
        line = ui.describe_tool_call({"name": "mystery_tool", "args": {"blob": "x" * 500}})
        self.assertLess(len(line), 200)

    def test_unknown_tool_still_renders_a_label(self):
        self.assertIn("mystery_tool", ui.describe_tool_call({"name": "mystery_tool", "args": {}}))


if __name__ == "__main__":
    unittest.main()
