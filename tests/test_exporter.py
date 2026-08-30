import os
import tempfile
import unittest

from exporter import export_to_pdf, _inline


class ExporterTests(unittest.TestCase):
    def test_inline_escapes_xml_before_formatting(self):
        self.assertIn("&lt;script&gt;", _inline("<script>"))

    def test_inline_converts_markdown_emphasis_and_links(self):
        rendered = _inline("**bold** and [site](https://example.com)")
        self.assertIn("<b>bold</b>", rendered)
        self.assertIn('href="https://example.com"', rendered)

    def test_export_renders_headings_lists_and_tables(self):
        markdown = (
            "# Title <unescaped>\n\n"
            "## Key Insights\n- point one\n- point two\n\n"
            "## Comparison Table\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "report.pdf")
            export_to_pdf(markdown, path)
            self.assertGreater(os.path.getsize(path), 1000)

    def test_export_handles_empty_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "empty.pdf")
            export_to_pdf("", path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
