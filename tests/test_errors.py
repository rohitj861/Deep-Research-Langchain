import unittest

from errors import explain

QUOTA = (
    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
    "limit: 20, model: gemini-3.6-flash 'retryDelay': '54s'"
)


class ErrorExplanationTests(unittest.TestCase):
    def test_quota_error_reports_the_limit_and_retry_delay(self):
        headline, step = explain(RuntimeError(QUOTA))
        self.assertIn("quota exceeded", headline.lower())
        self.assertIn("20", headline)
        self.assertIn("54s", step)

    def test_missing_model_is_distinguished_from_quota(self):
        headline, _ = explain(RuntimeError("404 NOT_FOUND. This model is not supported"))
        self.assertIn("not available", headline)

    def test_bad_key_is_distinguished(self):
        headline, _ = explain(RuntimeError("PERMISSION_DENIED: invalid credentials"))
        self.assertIn("rejected your API key", headline)

    def test_unknown_error_falls_back_to_the_exception_text(self):
        headline, _ = explain(ValueError("something odd"))
        self.assertIn("ValueError", headline)
        self.assertIn("something odd", headline)


if __name__ == "__main__":
    unittest.main()
