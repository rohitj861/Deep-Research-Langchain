"""Shared AppTest setup.

Two things make driving `app.py` from tests fiddly:

- `config._streamlit_secrets` is `lru_cache`d, so a value read in one test would
  otherwise leak into the next.
- `_setting` reads the environment before secrets, and the developer's own `.env`
  sets `APP_PASSWORD` — so a test must pin *both* sources to be deterministic.

`app_test()` pins them together. An empty password means the gate is disabled and
the app renders unlocked, which is what most tests want.
"""

import os
from contextlib import contextmanager
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import config

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@contextmanager
def app_test(password: str = "", timeout: int = 120):
    """Yield an AppTest whose password gate is set to `password` ("" = unlocked)."""
    config._streamlit_secrets.cache_clear()
    with patch.dict(os.environ, {"APP_PASSWORD": password}, clear=False):
        at = AppTest.from_file(APP, default_timeout=timeout)
        at.secrets["APP_PASSWORD"] = password
        try:
            yield at
        finally:
            config._streamlit_secrets.cache_clear()


def unlock(at: AppTest, password: str) -> AppTest:
    """Submit the login form and return the unlocked app."""
    at.text_input[0].set_value(password)
    at.button[0].click().run()
    return at
