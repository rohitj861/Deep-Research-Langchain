"""Password gate for the deployed app.

Streamlit Community Cloud apps are public by URL, so a shared password keeps a
deployment from spending your API budget on whoever finds it. This is a front
door, not real user auth — everyone shares one password and there are no accounts.
For per-person access, use Streamlit's viewer allowlist (Settings -> Sharing) instead.
"""

import hmac

import streamlit as st

from config import app_password

_AUTHENTICATED = "_authenticated"
_INPUT = "_password_input"


def _verify() -> None:
    """Check the submitted password, then drop the plaintext from session state."""
    entered = st.session_state.get(_INPUT, "")
    # compare_digest keeps the check constant-time, so response timing does not
    # leak how much of the password was correct.
    st.session_state[_AUTHENTICATED] = hmac.compare_digest(entered, app_password())
    st.session_state[_INPUT] = ""


def require_password() -> bool:
    """Return True when the visitor may see the app; render the login form otherwise."""
    expected = app_password()

    if not expected:
        st.warning(
            "No `APP_PASSWORD` is set, so anyone with the URL can use this app "
            "and spend its API budget. Set one in your Streamlit secrets.",
            icon="⚠️",
        )
        return True

    if st.session_state.get(_AUTHENTICATED):
        return True

    st.title("Deep Research AI")
    st.caption("This app is password protected.")

    with st.form("login"):
        st.text_input("Password", type="password", key=_INPUT)
        st.form_submit_button("Enter", on_click=_verify)

    # Only after a real attempt — not on the first render.
    if st.session_state.get(_AUTHENTICATED) is False:
        st.error("Incorrect password.")

    return False
