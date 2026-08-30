"""Config resolution across local (.env) and Streamlit Cloud (st.secrets).

On Cloud there is no `.env`; keys arrive via `st.secrets`, which Streamlit does not
export to the environment. Anything an SDK reads from `os.environ` therefore has to
be published explicitly.
"""

import os
import unittest
from unittest.mock import patch

import config


class SettingResolutionTests(unittest.TestCase):
    def setUp(self):
        config._streamlit_secrets.cache_clear()
        self.addCleanup(config._streamlit_secrets.cache_clear)

    def test_environment_wins_over_secrets_locally(self):
        with patch.object(config, "_streamlit_secrets", lambda: {"OPENAI_MODEL": "from-secrets"}):
            with patch.dict(os.environ, {"OPENAI_MODEL": "from-env"}, clear=False):
                self.assertEqual(config.get_model_name("openai"), "from-env")

    def test_secrets_used_when_environment_is_empty(self):
        with patch.object(config, "_streamlit_secrets", lambda: {"OPENAI_MODEL": "from-secrets"}):
            with patch.dict(os.environ, {"OPENAI_MODEL": ""}, clear=False):
                self.assertEqual(config.get_model_name("openai"), "from-secrets")

    def test_api_key_resolves_from_secrets(self):
        with patch.object(config, "_streamlit_secrets", lambda: {"OPENAI_API_KEY": "sk-from-secrets"}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
                self.assertEqual(config.get_api_key("openai"), "sk-from-secrets")
                self.assertTrue(config.has_api_key("openai"))

    def test_falls_back_to_default_when_nothing_is_set(self):
        with patch.object(config, "_streamlit_secrets", dict):
            with patch.dict(os.environ, {"OPENAI_MODEL": ""}, clear=False):
                self.assertEqual(config.get_model_name("openai"), "gpt-5.4-mini")

    def test_missing_secrets_file_is_not_an_error(self):
        # st.secrets raises when no secrets.toml exists; the CLI and tests rely on
        # that being swallowed.
        self.assertEqual(config._streamlit_secrets(), {})


class SdkEnvPublishingTests(unittest.TestCase):
    def setUp(self):
        config._streamlit_secrets.cache_clear()
        self.addCleanup(config._streamlit_secrets.cache_clear)

    def test_provider_key_is_published_for_the_sdk(self):
        with patch.object(config, "_streamlit_secrets", lambda: {"OPENAI_API_KEY": "sk-cloud"}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
                config.ensure_provider_env("openai")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-cloud")

    def test_tavily_key_is_published_for_the_sdk(self):
        with patch.object(config, "_streamlit_secrets", lambda: {"TAVILY_API_KEY": "tvly-cloud"}):
            with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
                config.ensure_search_env()
                self.assertEqual(os.environ["TAVILY_API_KEY"], "tvly-cloud")

    def test_search_tool_is_built_from_a_cloud_secret_alone(self):
        from tools import build_search_tools

        with patch.object(config, "_streamlit_secrets", lambda: {"TAVILY_API_KEY": "tvly-cloud"}):
            with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
                tools = build_search_tools()
        self.assertEqual([t.name for t in tools], ["tavily_search"])


if __name__ == "__main__":
    unittest.main()
