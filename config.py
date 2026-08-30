"""Runtime configuration: providers, models, and research depth presets.

Every provider is expressed as a LangChain model spec string
(``"<lc_provider>:<model>"``) so it can be handed straight to
``create_deep_agent(model=...)``, which resolves it via ``init_chat_model``.
"""

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def _streamlit_secrets() -> dict:
    """Secrets injected by Streamlit Cloud, or {} anywhere else.

    `st.secrets` raises when no secrets file exists, which is the normal case for
    the CLI and the test suite, so the lookup is guarded.
    """
    try:
        import streamlit as st

        return dict(st.secrets)
    except Exception:
        return {}


def _setting(name: str, default: str = "") -> str:
    """Read one setting: environment (incl. `.env`) first, then Streamlit secrets.

    Environment wins locally so a stale global `secrets.toml` cannot quietly
    override the `.env` the README tells you to use. On Streamlit Cloud there is
    no `.env`, so secrets are the only source and win by default.
    """
    value = os.getenv(name, "") or _streamlit_secrets().get(name, "")
    return str(value).strip() or default


class Provider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass(frozen=True)
class ProviderSpec:
    label: str
    lc_provider: str
    default_model: str
    model_env: str
    key_envs: tuple[str, ...]
    console_url: str
    # Env var the LangChain integration itself reads, when it differs from key_envs[0].
    sdk_key_env: str = ""


PROVIDER_SPECS: dict[Provider, ProviderSpec] = {
    Provider.GEMINI: ProviderSpec(
        label="Gemini (Google AI Studio)",
        lc_provider="google_genai",
        default_model="gemini-3.1-flash-lite",
        model_env="GEMINI_MODEL",
        key_envs=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        sdk_key_env="GOOGLE_API_KEY",
        console_url="https://aistudio.google.com/apikey",
    ),
    Provider.OPENAI: ProviderSpec(
        label="OpenAI",
        lc_provider="openai",
        default_model="gpt-5.4-mini",
        model_env="OPENAI_MODEL",
        key_envs=("OPENAI_API_KEY",),
        console_url="https://platform.openai.com/api-keys",
    ),
}

PROVIDER_LABELS = {provider: spec.label for provider, spec in PROVIDER_SPECS.items()}

DEFAULT_PROVIDER = Provider(_setting("DEFAULT_PROVIDER", Provider.OPENAI.value).lower())


@dataclass(frozen=True)
class DepthPreset:
    """How hard the deep agent should work on a topic.

    ``subagent_calls`` is a soft target injected into the system prompt;
    ``recursion_limit`` is the hard LangGraph step budget for the run.
    """

    label: str
    subagent_calls: int
    searches_per_task: int
    recursion_limit: int
    description: str


DEPTH_PRESETS: dict[str, DepthPreset] = {
    "Basic": DepthPreset("Basic", 2, 2, 60, "Two quick sub-investigations. Fastest, cheapest."),
    "Standard": DepthPreset("Standard", 4, 3, 120, "Four sub-investigations plus a critique pass."),
    "Advanced": DepthPreset("Advanced", 6, 5, 250, "Six sub-investigations, deeper searches, critique and revision."),
}

DEFAULT_DEPTH = _setting("RESEARCH_DEPTH", "Standard")

# Free tiers cap requests per minute, and a deep agent run is a burst of many calls.
# Pacing the model client is far cheaper than letting the run die halfway through.
REQUESTS_PER_MINUTE = float(_setting("REQUESTS_PER_MINUTE", "12"))
MAX_RETRIES = int(_setting("MAX_RETRIES", "3"))


def _coerce(provider: Provider | str) -> Provider:
    return provider if isinstance(provider, Provider) else Provider(str(provider).lower())


def get_provider_spec(provider: Provider | str) -> ProviderSpec:
    return PROVIDER_SPECS[_coerce(provider)]


def get_api_key(provider: Provider | str) -> str:
    """First non-empty key among the provider's accepted env vars."""
    spec = get_provider_spec(provider)
    for env_name in spec.key_envs:
        value = _setting(env_name)
        if value:
            return value
    return ""


def get_model_name(provider: Provider | str) -> str:
    spec = get_provider_spec(provider)
    return _setting(spec.model_env) or spec.default_model


def get_model_spec(provider: Provider | str) -> str:
    """LangChain model spec, e.g. ``"google_genai:gemini-3.6-flash"``."""
    spec = get_provider_spec(provider)
    return f"{spec.lc_provider}:{get_model_name(provider)}"


def ensure_provider_env(provider: Provider | str) -> str:
    """Publish the resolved key under the env var the SDK actually reads.

    Gemini accepts ``GEMINI_API_KEY`` in this project's ``.env`` for backwards
    compatibility, but ``langchain-google-genai`` only looks at ``GOOGLE_API_KEY``.
    """
    spec = get_provider_spec(provider)
    key = get_api_key(provider)
    target = spec.sdk_key_env or spec.key_envs[0]
    if key:
        os.environ[target] = key
    return key


def has_api_key(provider: Provider | str) -> bool:
    return bool(get_api_key(provider))


def app_password() -> str:
    """Shared password gating the deployed app. Empty means no gate."""
    return _setting("APP_PASSWORD")


def tavily_api_key() -> str:
    return _setting("TAVILY_API_KEY")


def ensure_search_env() -> str:
    """Publish the Tavily key to the env var `langchain-tavily` reads.

    On Streamlit Cloud the key arrives via `st.secrets`, which Streamlit does not
    export to the environment, so the SDK would never see it.
    """
    key = tavily_api_key()
    if key:
        os.environ["TAVILY_API_KEY"] = key
    return key


def search_enabled() -> bool:
    return bool(tavily_api_key())


def get_depth_preset(depth: str) -> DepthPreset:
    return DEPTH_PRESETS.get(depth, DEPTH_PRESETS["Standard"])


REPORT_FILE = "final_report.md"
QUESTION_FILE = "question.md"
NOTES_DIR = "notes"
