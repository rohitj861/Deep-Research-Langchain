"""The minimal deep agent - the shape everything else in this repo builds on.

    python examples/quickstart.py "What is the weather in Pune?"

Requires GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepagents import create_deep_agent  # noqa: E402

from config import ensure_provider_env, get_model_spec  # noqa: E402
from tools import get_weather  # noqa: E402

ensure_provider_env("gemini")

agent = create_deep_agent(
    model=get_model_spec("gemini"),  # -> "google_genai:gemini-3.6-flash"
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the weather in Pune?"
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    # .text flattens structured content blocks (Gemini returns those, not a plain string).
    print(result["messages"][-1].text)
