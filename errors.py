"""Turn provider exceptions into something a user can act on.

Provider SDKs raise errors carrying a wall of JSON. The UI shows one line and a
next step instead.
"""

import re

_RETRY_RE = re.compile(r"retryDelay['\"]?:\s*['\"]?(\d+)s")
_LIMIT_RE = re.compile(r"limit:\s*(\d+)")


def explain(exc: BaseException) -> tuple[str, str]:
    """Return a (headline, next step) pair for an exception raised during a run."""
    text = str(exc)
    name = type(exc).__name__

    if "RESOURCE_EXHAUSTED" in text or "429" in text or "RateLimit" in name:
        limit = _LIMIT_RE.search(text)
        retry = _RETRY_RE.search(text)
        headline = "Provider quota exceeded."
        if limit:
            headline = f"Provider quota exceeded (limit: {limit.group(1)} requests)."
        step = (
            "A research run makes many model calls. Either switch the provider's `*_MODEL` "
            "setting to a model with a larger allowance, pick a different provider in the "
            "sidebar, lower `REQUESTS_PER_MINUTE`, or enable billing on your provider account."
        )
        if retry:
            step = f"Retry in about {retry.group(1)}s, or " + step[0].lower() + step[1:]
        return headline, step

    if "NOT_FOUND" in text or "ModelNotFound" in name:
        return (
            "That model is not available to your API key.",
            "Set the provider's `*_MODEL` variable in `.env` to a model your account can reach.",
        )

    if "API key" in text or "PERMISSION_DENIED" in text or "401" in text or "Unauthorized" in text:
        return (
            "The provider rejected your API key.",
            "Check the key in `.env` for the provider selected in the sidebar, then restart the app.",
        )

    if "recursion" in text.lower() or "GraphRecursionError" in name:
        return (
            "The agent hit its step budget before finishing.",
            "Lower the research depth, or raise the depth's `recursion_limit` in `config.py`.",
        )

    return f"{name}: {text[:300]}", "See the terminal running Streamlit for the full traceback."
