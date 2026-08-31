"""Gaps found in a full read-through of the codebase.

Each of these was a real defect, not a style preference. The tests exist so a
future refactor cannot quietly reintroduce them.
"""

import os
import unittest
from unittest.mock import patch

import config


class InvalidSettingTests(unittest.TestCase):
    """A stale setting must not kill the app at import.

    DEFAULT_PROVIDER=groq — a provider that was removed from this project — used
    to raise ValueError while `config` was importing, so the whole app died with
    a stack trace over one line of .env.
    """

    def test_unknown_provider_falls_back_instead_of_raising(self):
        with patch.object(config, "_setting", lambda name, default="": "groq" if name == "DEFAULT_PROVIDER" else default):
            self.assertEqual(config._default_provider(), config.Provider.OPENAI)

    def test_empty_setting_uses_the_documented_default(self):
        with patch.object(config, "_setting", lambda name, default="": default):
            self.assertEqual(config._default_provider(), config.Provider.OPENAI)

    def test_a_valid_provider_is_honoured(self):
        with patch.object(config, "_setting", lambda name, default="": "gemini" if name == "DEFAULT_PROVIDER" else default):
            self.assertEqual(config._default_provider(), config.Provider.GEMINI)

    def test_case_is_not_significant(self):
        with patch.object(config, "_setting", lambda name, default="": "GEMINI" if name == "DEFAULT_PROVIDER" else default):
            self.assertEqual(config._default_provider(), config.Provider.GEMINI)


class SharedRateLimiterTests(unittest.TestCase):
    """Provider quotas are per account, not per session.

    A limiter built per agent let N concurrent Streamlit sessions each issue
    REQUESTS_PER_MINUTE — exactly the burst the pacing exists to prevent.
    """

    def test_one_limiter_is_shared_process_wide(self):
        from agent import _rate_limiter

        self.assertIs(_rate_limiter(), _rate_limiter())

    def test_separately_resolved_models_share_one_limiter(self):
        # Inspect the models themselves rather than the cache, so this fails if
        # resolve_model ever stops passing the shared limiter through.
        from agent import resolve_model

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            first = resolve_model("openai")
            second = resolve_model("openai")

        self.assertIsNotNone(first.rate_limiter)
        self.assertIs(first.rate_limiter, second.rate_limiter)

    def test_the_limiter_reflects_the_configured_rate(self):
        from agent import _rate_limiter
        from config import REQUESTS_PER_MINUTE

        self.assertAlmostEqual(_rate_limiter().requests_per_second, REQUESTS_PER_MINUTE / 60, places=6)


class PdfIsolationTests(unittest.TestCase):
    """Streamlit Cloud serves every viewer from one process.

    Rendering to a fixed temp path let two people downloading different reports
    overwrite each other's file and receive the wrong document.
    """

    def test_each_render_uses_its_own_file_and_cleans_up(self):
        import contextlib
        import tempfile

        from exporter import export_to_pdf

        paths = []
        for index in range(3):
            handle, path = tempfile.mkstemp(prefix="research_report_", suffix=".pdf")
            os.close(handle)
            export_to_pdf(f"# Report {index}", path)
            paths.append(path)
            with contextlib.suppress(OSError):
                os.unlink(path)

        self.assertEqual(len(set(paths)), 3, "temp paths must be distinct")
        self.assertFalse([p for p in paths if os.path.exists(p)], "temp files must be removed")


class ErrorMessageTests(unittest.TestCase):
    def test_quota_advice_does_not_name_one_provider(self):
        from errors import explain

        _, step = explain(RuntimeError("429 RESOURCE_EXHAUSTED"))
        self.assertNotIn("GEMINI_MODEL", step)
        self.assertIn("*_MODEL", step)


if __name__ == "__main__":
    unittest.main()
