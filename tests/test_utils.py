import unittest

from utils import list_files, read_file


class VirtualFilesystemTests(unittest.TestCase):
    """The state backend normalizes paths to absolute form, so lookups must tolerate both."""

    FILES = {"/final_report.md": "REPORT", "/notes/topic.md": "NOTE", "/question.md": "Q"}

    def test_reads_a_relative_path_against_absolute_keys(self):
        self.assertEqual(read_file(self.FILES, "final_report.md"), "REPORT")

    def test_reads_an_absolute_path(self):
        self.assertEqual(read_file(self.FILES, "/final_report.md"), "REPORT")

    def test_missing_file_returns_empty_string(self):
        self.assertEqual(read_file(self.FILES, "nope.md"), "")

    def test_empty_state_returns_empty_string(self):
        self.assertEqual(read_file({}, "final_report.md"), "")

    def test_unwraps_dict_shaped_file_records(self):
        self.assertEqual(read_file({"/a.md": {"content": "X"}}, "a.md"), "X")

    def test_list_files_excludes_by_relative_name(self):
        self.assertEqual(list_files(self.FILES, exclude=("final_report.md",)), ["/notes/topic.md", "/question.md"])


if __name__ == "__main__":
    unittest.main()
