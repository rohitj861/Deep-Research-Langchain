import os
import unittest
from unittest.mock import patch

from tools import build_search_tools, get_weather


class SearchToolTests(unittest.TestCase):
    def test_no_search_tool_without_a_tavily_key(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
            self.assertEqual(build_search_tools(), [])

    def test_search_tool_is_built_when_key_present(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False):
            tools = build_search_tools()
        self.assertEqual([tool.name for tool in tools], ["tavily_search"])


class WeatherToolTests(unittest.TestCase):
    def test_unknown_city_reports_cleanly(self):
        with patch("tools.weather.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"results": []}
            mock_get.return_value.raise_for_status.return_value = None
            result = get_weather.invoke({"city": "Nowhereville"})
        self.assertIn("No location found", result)

    def test_network_failure_is_returned_not_raised(self):
        import requests

        with patch("tools.weather.requests.get", side_effect=requests.ConnectionError("down")):
            result = get_weather.invoke({"city": "Pune"})
        self.assertIn("Weather lookup failed", result)


if __name__ == "__main__":
    unittest.main()
