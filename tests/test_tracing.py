"""Langfuse tracing wiring.

Tracing is strictly optional: with no keys the callback list is empty and every
code path behaves exactly as before. A tracing failure must never break a run.
"""

import os
import unittest
from unittest.mock import patch

import config
import tracing
from agent import run_config

KEYS = {
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "LANGFUSE_HOST": "https://cloud.langfuse.com",
}


class TracingDisabledTests(unittest.TestCase):
    def setUp(self):
        config._streamlit_secrets.cache_clear()
        self.addCleanup(config._streamlit_secrets.cache_clear)
        patcher = patch.dict(os.environ, dict.fromkeys(KEYS, ""), clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_not_enabled_without_keys(self):
        self.assertFalse(config.tracing_enabled())

    def test_no_callbacks_without_keys(self):
        self.assertEqual(tracing.build_callbacks(), [])

    def test_run_config_omits_callbacks_when_off(self):
        self.assertNotIn("callbacks", run_config("Basic", "t"))

    def test_flush_is_safe_when_unconfigured(self):
        tracing.flush()  # must not raise


class TracingEnabledTests(unittest.TestCase):
    def setUp(self):
        config._streamlit_secrets.cache_clear()
        self.addCleanup(config._streamlit_secrets.cache_clear)
        patcher = patch.dict(os.environ, KEYS, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_enabled_when_both_keys_present(self):
        self.assertTrue(config.tracing_enabled())

    def test_one_key_alone_is_not_enough(self):
        with patch.dict(os.environ, {"LANGFUSE_SECRET_KEY": ""}, clear=False):
            self.assertFalse(config.tracing_enabled())

    def test_keys_are_published_to_the_sdk_env(self):
        with patch.object(config, "_streamlit_secrets", lambda: KEYS):
            with patch.dict(os.environ, dict.fromkeys(KEYS, ""), clear=False):
                self.assertTrue(config.ensure_tracing_env())
                self.assertEqual(os.environ["LANGFUSE_PUBLIC_KEY"], "pk-lf-test")

    def test_host_defaults_when_unset(self):
        with patch.dict(os.environ, {"LANGFUSE_HOST": ""}, clear=False):
            os.environ.pop("LANGFUSE_HOST", None)
            config.ensure_tracing_env()
            self.assertTrue(os.environ["LANGFUSE_HOST"].startswith("https://"))

    def test_callbacks_are_langchain_handlers(self):
        from langchain_core.callbacks import BaseCallbackHandler

        callbacks = tracing.build_callbacks()
        self.assertTrue(callbacks, "expected a Langfuse handler")
        self.assertTrue(all(isinstance(c, BaseCallbackHandler) for c in callbacks))

    def test_run_config_carries_callbacks(self):
        self.assertIn("callbacks", run_config("Basic", "t"))

    def test_a_broken_handler_does_not_break_the_run(self):
        with patch("langfuse.langchain.CallbackHandler", side_effect=RuntimeError("boom")):
            self.assertEqual(tracing.build_callbacks(), [])


if __name__ == "__main__":
    unittest.main()
