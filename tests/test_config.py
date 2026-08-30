import os
import unittest
from unittest.mock import patch

import config


class ConfigTests(unittest.TestCase):
    def test_model_spec_uses_langchain_provider_prefix(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.6-flash"}, clear=False):
            self.assertEqual(config.get_model_spec(config.Provider.GEMINI), "google_genai:gemini-3.6-flash")

    def test_model_env_var_overrides_default(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4.1-mini"}, clear=False):
            self.assertEqual(config.get_model_spec("openai"), "openai:gpt-4.1-mini")

    def test_openai_is_selectable(self):
        self.assertEqual(config.get_provider_spec("openai").key_envs, ("OPENAI_API_KEY",))

    def test_gemini_accepts_either_key_env(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": "legacy"}, clear=False):
            self.assertEqual(config.get_api_key("gemini"), "legacy")

    def test_ensure_provider_env_publishes_key_for_sdk(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": "legacy"}, clear=False):
            config.ensure_provider_env("gemini")
            self.assertEqual(os.environ["GOOGLE_API_KEY"], "legacy")

    def test_every_provider_has_a_spec(self):
        for provider in config.Provider:
            spec = config.get_provider_spec(provider)
            self.assertTrue(spec.lc_provider and spec.default_model and spec.key_envs)

    def test_depth_presets_increase_budget(self):
        budgets = [config.get_depth_preset(name).recursion_limit for name in ("Basic", "Standard", "Advanced")]
        self.assertEqual(budgets, sorted(budgets))

    def test_unknown_depth_falls_back_to_standard(self):
        self.assertEqual(config.get_depth_preset("Nonsense").label, "Standard")


if __name__ == "__main__":
    unittest.main()
