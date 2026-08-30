import unittest

from utils import find_file, list_files, read_file


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


class FindFileTests(unittest.TestCase):
    """The report is located even when the agent improvised the path.

    An exact lookup reported "the agent has not written final_report.md yet" while
    the file sat in state under a slightly different name, and `ensure_report` then
    spent a whole extra turn rewriting it.
    """

    def _find(self, path):
        return find_file({path: "REPORT BODY"}, "final_report.md")

    def test_exact_path(self):
        self.assertEqual(self._find("/final_report.md")[1], "REPORT BODY")

    def test_relative_path(self):
        self.assertEqual(self._find("final_report.md")[1], "REPORT BODY")

    def test_written_into_a_subdirectory(self):
        path, body = self._find("/reports/final_report.md")
        self.assertEqual(body, "REPORT BODY")
        self.assertEqual(path, "/reports/final_report.md")

    def test_different_capitalisation(self):
        self.assertEqual(self._find("/Final_Report.md")[1], "REPORT BODY")

    def test_hyphen_instead_of_underscore(self):
        self.assertEqual(self._find("/final-report.md")[1], "REPORT BODY")

    def test_different_extension(self):
        self.assertEqual(self._find("/final_report.markdown")[1], "REPORT BODY")

    def test_unrelated_file_is_not_matched(self):
        self.assertEqual(self._find("/notes/topic.md"), ("", ""))

    def test_empty_state(self):
        self.assertEqual(find_file({}, "final_report.md"), ("", ""))

    def test_exact_match_wins_over_a_fuzzy_one(self):
        files = {"/final_report.md": "EXACT", "/reports/Final-Report.md": "FUZZY"}
        self.assertEqual(find_file(files, "final_report.md")[1], "EXACT")

    def test_returns_the_path_so_callers_can_exclude_it(self):
        path, _ = find_file({"/reports/final_report.md": "X"}, "final_report.md")
        self.assertEqual(path, "/reports/final_report.md")
